"""
krw_study.py — 코어 TAA 전략의 원화 기준 성과 측정.

접근
----
전략 로직을 새로 짜지 않는다. taa.engine.run_backtest()를 그대로 쓰고
가격만 원화로 바꿔 넣는다. 그래야 비용/세금/리밸런싱 규칙이 검증된
것과 동일하게 유지되고, 달러 결과와의 비교가 성립한다.

변형 A: 신호는 달러 가격, 체결/손익은 원화
    signal_prices=prices_usd 로 신호를 고정하고
    prices=prices_krw 로 체결한다.
    engine.run_backtest()에 signal_prices 인자가 추가되어 있어야 한다.

전제 조건
--------
taa/engine.py 의 run_backtest 에 다음 인자가 추가되어 있을 것:

    signal_prices: pd.DataFrame | None = None
    ...
    tw = target_weights(
        prices if signal_prices is None else signal_prices, ...
    )

게이트
-----
USD 기준 결과가 기존 검증값(CAGR 9.9% / MDD -14.5% / Sharpe 0.877)과
맞지 않으면 하네스가 잘못된 것이다. KRW 결과를 해석하기 전에
반드시 USD 쪽부터 확인할 것.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
from dataclasses import replace
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from taa.config import StrategyConfig, ExecConfig, TICKERS, CASH_PROXY
from taa.engine import run_backtest


# yfinance tz 캐시 SQLite 락 회피
_YF_CACHE = os.path.join(tempfile.gettempdir(), "yf_cache")
os.makedirs(_YF_CACHE, exist_ok=True)
for _setter in ("set_tz_cache_location", "set_cache_location"):
    try:
        getattr(yf, _setter)(_YF_CACHE)
    except Exception:
        pass

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    """Actions가 미정의 변수에 빈 문자열을 주입하는 문제 회피."""
    v = os.environ.get(key, "")
    return v if v.strip() else default


FX_SOURCE = env("FX_SOURCE", "fred")            # fred | yfinance
START = env("START", "")                        # 성과 집계 시작일. run_study.py와 맞출 것
KRW_CASH_RATE = float(env("KRW_CASH_RATE", "0.02"))   # 원화 현금 연이율
KRW_DEDUCTION = float(env("KRW_DEDUCTION", "2500000")) # 양도세 기본공제 (원)

SUBPERIODS = {
    "2008 금융위기":     ("2007-10-01", "2009-03-31"),
    "2020 코로나":       ("2020-02-01", "2020-04-30"),
    "2022 동반하락":     ("2022-01-01", "2022-10-31"),
    "2009-10 원화강세":  ("2009-03-01", "2010-12-31"),
    "2025-26 원화사이클": ("2025-01-01", "2026-12-31"),
}


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def _download(ticker: str, tries: int = 4) -> pd.Series:
    """단일 스레드 + 재시도. threads=True면 tz 캐시 락이 난다."""
    last = None
    for i in range(tries):
        try:
            df = yf.download(ticker, period="max", auto_adjust=True,
                             progress=False, threads=False)
            if df is not None and not df.empty:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce").dropna()
                if len(s) > 100:
                    print(f"      {ticker}: {len(s)}일 "
                          f"({s.index[0].date()} ~ {s.index[-1].date()})")
                    return s
                last = f"{len(s)}행뿐"
            else:
                last = "빈 응답"
        except Exception as e:
            last = repr(e)
        print(f"      {ticker}: 시도 {i+1}/{tries} 실패 ({last})")
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"{ticker} 다운로드 실패: {last}")


def _naive_index(obj):
    obj.index = pd.to_datetime(obj.index)
    if getattr(obj.index, "tz", None) is not None:
        obj.index = obj.index.tz_localize(None)
    return obj


def load_prices() -> pd.DataFrame:
    px = pd.concat({t: _download(t) for t in TICKERS}, axis=1)
    px.columns = list(TICKERS)
    px = _naive_index(px).dropna()
    if px.empty:
        raise RuntimeError("전 종목 공통 구간이 비었습니다.")
    print(f"      공통: {px.index[0].date()} ~ {px.index[-1].date()} ({len(px)}일)")
    return px


def load_usd_cash(index: pd.DatetimeIndex, ex: ExecConfig) -> pd.Series:
    """^IRX(13주 T-Bill 연율 %) -> 일간 수익률."""
    if not ex.use_real_cash_rate:
        return pd.Series(ex.flat_cash_rate / 252.0, index=index)
    irx = _naive_index(_download(CASH_PROXY))
    annual = (irx / 100.0 - ex.cash_spread).clip(lower=0.0)
    return (annual.reindex(index).ffill().bfill() / 252.0).rename("cash")


def load_fx() -> pd.Series:
    if FX_SOURCE == "yfinance":
        s = _download("KRW=X")
    else:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"
        df = pd.read_csv(url, parse_dates=[0], index_col=0)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s = _naive_index(s)
    s.name = "USDKRW"
    print(f"      환율: {s.index[0].date()} ~ {s.index[-1].date()} ({len(s)}행)")
    return s


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def stats(returns: pd.Series) -> dict:
    curve = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    vol = returns.std() * np.sqrt(252)
    return {
        "CAGR": curve.iloc[-1] ** (1.0 / years) - 1.0,
        "MDD": (curve / curve.cummax() - 1.0).min(),
        "Sharpe": (returns.mean() * 252) / vol if vol > 0 else np.nan,
        "Vol": vol,
    }


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:,.2f}%"


def subperiods(r_usd: pd.Series, r_krw: pd.Series) -> pd.DataFrame:
    rows = []
    for name, (s, e) in SUBPERIODS.items():
        u, k = r_usd.loc[s:e], r_krw.loc[s:e]
        if len(u) < 5:
            continue
        uc, kc = (1 + u).cumprod(), (1 + k).cumprod()
        rows.append({
            "구간": name,
            "USD 수익": uc.iloc[-1] - 1, "KRW 수익": kc.iloc[-1] - 1,
            "USD MDD": (uc / uc.cummax() - 1).min(),
            "KRW MDD": (kc / kc.cummax() - 1).min(),
        })
    return pd.DataFrame(rows).set_index("구간") if rows else pd.DataFrame()


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> int:
    strat = StrategyConfig()
    ex_usd = ExecConfig(apply_tax=False)

    print(f"[1/5] 가격 로드 {list(TICKERS)}")
    px_usd = load_prices()

    print("[2/5] 현금/환율 로드")
    cash_usd = load_usd_cash(px_usd.index, ex_usd)
    fx = load_fx()

    fx_aligned = fx.reindex(px_usd.index).ffill().bfill()
    stale = (px_usd.index > fx.index[-1]).sum()
    if stale:
        print(f"      경고: 환율이 {fx.index[-1].date()}에서 끊김. "
              f"최근 {stale}영업일은 ffill로 채워짐.")

    px_krw = px_usd.mul(fx_aligned, axis=0)

    warm = pd.Timestamp(START) if START else None

    print("[3/5] USD 기준 백테스트 (게이트)")
    res_usd = run_backtest(px_usd, cash_usd, strat, ex_usd, warmup_from=warm)

    print("[4/5] KRW 기준 백테스트 (신호는 USD 고정)")
    fx0 = float(fx_aligned.iloc[0])
    ex_krw = replace(
        ex_usd,
        initial_capital=ex_usd.initial_capital * fx0,   # 동일 규모로 환산
        annual_deduction_usd=KRW_DEDUCTION,             # 원화 표시이므로 250만원
        use_real_cash_rate=False,
        flat_cash_rate=KRW_CASH_RATE,                   # 현금은 원화, 환노출 없음
    )
    cash_krw = pd.Series(KRW_CASH_RATE / 252.0, index=px_usd.index)

    res_krw = run_backtest(
        px_krw, cash_krw, strat, ex_krw,
        warmup_from=warm, signal_prices=px_usd,
    )

    print("[5/5] 집계")
    r_usd, r_krw = res_usd.returns, res_krw.returns
    comp = pd.DataFrame(
        [dict(기준="USD", **stats(r_usd)), dict(기준="KRW", **stats(r_krw))]
    ).set_index("기준")
    subs = subperiods(r_usd, r_krw)

    # 신호 동일성 확인: 두 런의 목표비중 경로가 같아야 변형 A가 성립
    diff = (res_usd.weights.sum(axis=1) - res_krw.weights.sum(axis=1)).abs()
    print(f"      비중 경로 최대 괴리: {diff.max():.4f} "
          f"(리밸런싱 타이밍 차이로 0은 아님)")

    comp.to_csv("krw_summary.csv", encoding="utf-8-sig")
    subs.to_csv("krw_subperiods.csv", encoding="utf-8-sig")
    pd.DataFrame({"usd": r_usd, "krw": r_krw}).to_csv(
        "krw_returns.csv", encoding="utf-8-sig")
    pd.DataFrame({"equity_usd": res_usd.equity, "equity_krw": res_krw.equity,
                  "fx": fx_aligned.reindex(res_usd.equity.index)}
                 ).to_csv("krw_equity.csv", encoding="utf-8-sig")

    print("\n" + comp.to_string())
    print("\n" + subs.to_string())
    print("\nUSD 진단:", {k: round(v, 4) for k, v in res_usd.diagnostics.items()})
    print("KRW 진단:", {k: round(v, 4) for k, v in res_krw.diagnostics.items()})

    md = [
        "## 원화 기준 백테스트",
        f"기간 {r_usd.index[0].date()} ~ {r_usd.index[-1].date()} | 환율 {FX_SOURCE}",
        "",
        "### 게이트 — USD 결과가 9.9% / -14.5% / 0.877 근처인가?",
        "| 기준 | CAGR | MDD | Sharpe | Vol |", "|---|---|---|---|---|",
    ]
    for label, row in comp.iterrows():
        md.append(f"| {label} | {pct(row['CAGR'])} | {pct(row['MDD'])} "
                  f"| {row['Sharpe']:.3f} | {pct(row['Vol'])} |")
    if not subs.empty:
        md += ["", "### 구간별",
               "| 구간 | USD 수익 | KRW 수익 | USD MDD | KRW MDD |",
               "|---|---|---|---|---|"]
        for name, row in subs.iterrows():
            md.append(f"| {name} | {pct(row['USD 수익'])} | {pct(row['KRW 수익'])} "
                      f"| {pct(row['USD MDD'])} | {pct(row['KRW MDD'])} |")
    write_summary("\n".join(md))

    print(f"\n완료 {datetime.now(KST):%Y-%m-%d %H:%M KST}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TypeError as e:
        if "signal_prices" in str(e):
            print("\n>>> taa/engine.py 의 run_backtest 에 signal_prices 인자를 "
                  "추가하세요. 스크립트 상단 docstring 참고.\n")
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
