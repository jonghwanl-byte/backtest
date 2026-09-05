"""
krw_study.py — 코어 TAA 전략의 원화 기준 성과 측정.

목적:
    검증된 달러 기준 성과(CAGR 9.9% / MDD -14.5% / Sharpe 0.877)와
    원화 투자자가 실제로 겪는 성과의 괴리를 측정한다.

변형 A만 수행한다:
    신호 = 달러 가격 기준 (기존 로직 그대로, 손대지 않음)
    손익 = 원화 기준 (자산만 환산, 현금은 원화 유지)

    이건 새 전략이 아니라 기존 전략의 정직한 재측정이다.

실행:
    GitHub Actions 수동 디스패치 (krw_study.yml)
출력:
    CSV artifact + Step Summary + Telegram
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf

import fx as fxlib


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    """
    GitHub Actions는 정의되지 않은 변수에 빈 문자열을 주입하므로
    os.environ.get(key, default)의 default가 무력화된다.
    위성 봇에서 겪었던 그 버그의 회피책.
    """
    v = os.environ.get(key, "")
    return v if v.strip() else default


TICKERS = env("TICKERS", "QQQ,TLT,GLD").split(",")
FX_SOURCE = env("FX_SOURCE", "fred")            # fred | yfinance
START = env("START", "")                         # 빈 값이면 전체 이력
KRW_CASH_RATE = float(env("KRW_CASH_RATE", "0.0"))

TG_TOKEN = env("TELEGRAM_BOT_TOKEN")
TG_CHAT = env("TELEGRAM_CHAT_ID")

KST = timezone(timedelta(hours=9))

SUBPERIODS = {
    "2008 금융위기":    ("2007-10-01", "2009-03-31"),
    "2020 코로나":      ("2020-02-01", "2020-04-30"),
    "2022 동반하락":    ("2022-01-01", "2022-10-31"),
    "2009-10 원화강세": ("2009-03-01", "2010-12-31"),
    "2026 원화강세":    ("2026-06-01", "2026-12-31"),
}


# ---------------------------------------------------------------------------
# ★★★ 연결 지점 — 여기만 종환님 코드에 맞게 채우면 됩니다 ★★★
# ---------------------------------------------------------------------------

def build_weights(prices_usd: pd.DataFrame) -> pd.DataFrame:
    """
    달러 가격 -> 자산별 목표비중 DataFrame.

    반환 규격:
        index   = prices_usd.index 와 동일 (또는 부분집합)
        columns = prices_usd.columns 와 동일 (QQQ, TLT, GLD)
        값      = 0.0 ~ 1.0, 행 합계 <= 1.0 (나머지는 현금)
        ★ 체결 지연(lag)이 이미 반영된 상태여야 함.
          여기서 lag를 빼먹으면 룩어헤드 편향이 다시 들어온다.

    taa/ 패키지에 이미 있는 함수를 그대로 호출하세요. 예:

        from taa.signals import compute_weights
        return compute_weights(
            prices_usd,
            base_weights={"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20},
            ma_periods=(20, 120, 200),
            hysteresis=(0.015, -0.025),
            scalar=(1.00, 0.75, 0.50, 0.00),
            execution_lag=1,
        )

    함수명/시그니처가 다르면 그에 맞게 바꾸시면 됩니다.
    이 파일에서 신호 로직을 새로 짜지 마세요 — 기존 검증된 코드를 재사용해야
    달러 기준 결과와의 비교가 성립합니다.
    """
    raise NotImplementedError(
        "build_weights()를 taa/ 패키지의 기존 신호 함수에 연결하세요."
    )


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def load_prices(tickers: list[str]) -> pd.DataFrame:
    """워밍업을 위해 period='max'. 고정 기간 사용 금지."""
    df = yf.download(
        tickers, period="max", auto_adjust=True,
        progress=False, group_by="column",
    )
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[tickers].dropna(how="all")
    px = px.dropna()                     # 전 종목 공통 구간만 사용
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def load_fx() -> pd.Series:
    if FX_SOURCE == "yfinance":
        s = fxlib.load_fx_yfinance()
    else:
        s = fxlib.load_fx_fred()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def usd_returns_from(weights: pd.DataFrame, prices_usd: pd.DataFrame) -> pd.Series:
    """대조군: 동일 비중을 달러 기준으로 굴린 수익률."""
    r = prices_usd.pct_change().fillna(0.0)
    w = weights.reindex(r.index).ffill().fillna(0.0)[r.columns]
    return (w * r).sum(axis=1)


def fmt_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:,.2f}%"


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def write_step_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def send_telegram(html: str) -> None:
    if not (TG_TOKEN and TG_CHAT):
        print("[telegram] 자격증명 없음 — 전송 생략")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for chunk in [html[i:i + 3800] for i in range(0, len(html), 3800)]:
        for attempt in range(3):
            try:
                r = requests.post(url, timeout=20, data={
                    "chat_id": TG_CHAT, "text": chunk,
                    "parse_mode": "HTML", "disable_web_page_preview": True,
                })
                if r.ok:
                    break
                print(f"[telegram] {r.status_code} {r.text[:200]}")
            except Exception as e:
                print(f"[telegram] 시도 {attempt + 1} 실패: {e}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[1/5] 가격 로드: {TICKERS}")
    prices_usd = load_prices(TICKERS)
    print(f"      {prices_usd.index[0].date()} ~ {prices_usd.index[-1].date()} "
          f"({len(prices_usd)}일)")

    print(f"[2/5] 환율 로드: {FX_SOURCE}")
    fx_series = load_fx()
    print(f"      {fx_series.index[0].date()} ~ {fx_series.index[-1].date()}")

    # 환율 이력이 시작되기 전 구간은 버린다. bfill로 채운 값이
    # 성과 계산에 섞이면 결과가 오염된다.
    cutoff = max(prices_usd.index[0], fx_series.index[0])
    if START:
        cutoff = max(cutoff, pd.Timestamp(START))

    print("[3/5] 신호 생성 (달러 가격 기준)")
    weights = build_weights(prices_usd)

    print("[4/5] 원화 환산 및 성과 계산")
    prices_krw = fxlib.to_krw(prices_usd, fx_series)

    r_usd = usd_returns_from(weights, prices_usd).loc[cutoff:]
    r_krw = fxlib.portfolio_returns_krw(
        weights, prices_krw, krw_cash_rate_annual=KRW_CASH_RATE
    ).loc[cutoff:]

    comp = fxlib.compare(r_usd, r_krw)
    subs = fxlib.subperiod_table(r_usd, r_krw, SUBPERIODS)

    print("[5/5] 저장 및 리포트")
    comp.to_csv("krw_study_summary.csv", encoding="utf-8-sig")
    subs.to_csv("krw_study_subperiods.csv", encoding="utf-8-sig")
    pd.DataFrame({"usd": r_usd, "krw": r_krw}).to_csv(
        "krw_study_returns.csv", encoding="utf-8-sig"
    )

    print("\n" + comp.to_string())
    print("\n" + subs.to_string())

    # --- Step Summary -----------------------------------------------------
    md = [
        "## 원화 기준 백테스트 결과",
        f"기간: {r_usd.index[0].date()} ~ {r_usd.index[-1].date()}  "
        f"| 환율소스: {FX_SOURCE}",
        "",
        "### 전체 구간",
        "| 기준 | CAGR | MDD | Sharpe | Vol |",
        "|---|---|---|---|---|",
    ]
    for label, row in comp.iterrows():
        md.append(f"| {label} | {fmt_pct(row['CAGR'])} | {fmt_pct(row['MDD'])} "
                  f"| {row['Sharpe']:.3f} | {fmt_pct(row['Vol'])} |")
    md += ["", "### 구간별", "| 구간 | USD 수익 | KRW 수익 | USD MDD | KRW MDD |",
           "|---|---|---|---|---|"]
    for name, row in subs.iterrows():
        md.append(f"| {name} | {fmt_pct(row['USD 수익'])} | {fmt_pct(row['KRW 수익'])} "
                  f"| {fmt_pct(row['USD MDD'])} | {fmt_pct(row['KRW MDD'])} |")
    write_step_summary("\n".join(md))

    # --- Telegram ---------------------------------------------------------
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [f"<b>원화 기준 백테스트</b>  <i>{now}</i>", ""]
    for label, row in comp.iterrows():
        lines.append(f"<b>{label}</b>  CAGR {fmt_pct(row['CAGR'])} / "
                     f"MDD {fmt_pct(row['MDD'])} / SR {row['Sharpe']:.3f}")
    lines.append("")
    lines.append("<b>구간별 MDD (USD → KRW)</b>")
    for name, row in subs.iterrows():
        lines.append(f"· {name}: {fmt_pct(row['USD MDD'])} → {fmt_pct(row['KRW MDD'])}")
    send_telegram("\n".join(lines))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        send_telegram(f"<b>원화 백테스트 실패</b>\n<pre>{traceback.format_exc()[-1500:]}</pre>")
        sys.exit(1)
