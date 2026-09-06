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
    CSV artifact + Step Summary (콘솔 출력)
"""

from __future__ import annotations

import os
import sys
import traceback
import tempfile
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

import fx as fxlib


# yfinance는 시간대 정보를 SQLite에 캐싱한다. 멀티 티커 다운로드가
# 스레드로 동시에 이 파일을 물면 "database is locked"가 난다.
# 실행마다 격리된 임시 경로를 쓰게 해서 회피한다.
_YF_CACHE = os.path.join(tempfile.gettempdir(), "yf_cache")
os.makedirs(_YF_CACHE, exist_ok=True)
for _setter in ("set_tz_cache_location", "set_cache_location"):
    try:
        getattr(yf, _setter)(_YF_CACHE)
    except Exception:
        pass


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

def _download_one(ticker: str, tries: int = 4) -> pd.Series:
    """
    티커 하나를 단일 스레드로 받는다.

    threads=False가 핵심이다. 멀티 티커 병렬 다운로드가 tz 캐시 SQLite를
    동시에 물면서 'database is locked'를 유발했다.
    """
    last_err = None
    for i in range(tries):
        try:
            df = yf.download(
                ticker, period="max", auto_adjust=True,
                progress=False, threads=False,
            )
            if df is not None and not df.empty:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce").dropna()
                if len(s) > 100:
                    print(f"      {ticker}: {len(s)}일  "
                          f"({s.index[0].date()} ~ {s.index[-1].date()})")
                    return s
                last_err = f"데이터가 {len(s)}행뿐"
            else:
                last_err = "빈 응답"
        except Exception as e:
            last_err = repr(e)
        wait = 3 * (i + 1)
        print(f"      {ticker}: 시도 {i + 1}/{tries} 실패 ({last_err}) — {wait}초 대기")
        time.sleep(wait)

    raise RuntimeError(f"{ticker} 다운로드 실패: {last_err}")


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """
    워밍업을 위해 period='max'. 고정 기간 사용 금지.

    한 종목이라도 실패하면 즉시 중단한다. 조용히 NaN 컬럼으로 넘어가면
    dropna()가 전 구간을 날려버리고, 엉뚱한 곳에서 IndexError가 난다.
    """
    series = {t: _download_one(t.strip()) for t in tickers}
    px = pd.concat(series, axis=1)
    px.columns = [t.strip() for t in tickers]

    px.index = pd.to_datetime(px.index)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)

    before = len(px)
    px = px.dropna()                     # 전 종목 공통 구간만 사용
    print(f"      공통 구간: {len(px)}일 (전체 {before}일에서 정렬)")

    if px.empty:
        raise RuntimeError(
            "전 종목 공통 구간이 비었습니다. 티커별 시작일이 겹치지 않는지 확인하세요."
        )
    return px


def load_fx() -> pd.Series:
    if FX_SOURCE == "yfinance":
        s = fxlib.load_fx_yfinance()
    else:
        s = fxlib.load_fx_fred()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    if s.empty:
        raise RuntimeError(f"환율 데이터가 비었습니다 (소스: {FX_SOURCE})")
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

    print(f"\n완료: {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
