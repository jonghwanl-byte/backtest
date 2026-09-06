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

# 현금을 어느 통화로 보유하는가.
#   usd: 달러 단기채(^IRX 금리 + 환노출). 실제 운용 방식.
#        스칼라 0% 구간에 포트폴리오 전액이 달러 현금이 되므로
#        위기 성과가 이 설정에 크게 좌우된다.
#   krw: 원화 예치(고정금리, 환노출 없음). 대조군.
CASH_CCY = env("CASH_CCY", "usd").lower()

# 원화 현금 금리 소스.
#   fred: 한국 3개월 은행간금리 (FRED IR3TIB01KRM156N, 월간, 1991~)
#         2008년처럼 신호가 꺼져 전액 현금인 해에 실제로 5%대를 받았다.
#         고정 2%로 두면 원화 현금 쪽이 부당하게 불리해진다.
#   flat: KRW_CASH_RATE 고정. 민감도 확인용.
KRW_RATE_SOURCE = env("KRW_RATE_SOURCE", "fred").lower()

# 연금계좌는 해외주식 양도소득세가 없다.
# 검증된 기준값(CAGR 9.9% / MDD -14.5% / Sharpe 0.877)이 '세전, 연금계좌'
# 기준이므로 게이트를 맞추려면 false 여야 한다.
APPLY_TAX = env("APPLY_TAX", "false").lower() in ("1", "true", "yes")

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


def _fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=[0], index_col=0)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return _naive_index(s)


def load_krw_cash(index: pd.DatetimeIndex, ex: ExecConfig) -> pd.Series:
    """원화 단기채 ETF 파킹 수익률 (일간).

    ETF 보수/스프레드는 ex.cash_spread 로 차감한다.
    """
    if KRW_RATE_SOURCE == "flat":
        print(f"      원화금리: 고정 {KRW_CASH_RATE:.2%}")
        return pd.Series(KRW_CASH_RATE / 252.0, index=index, name="cash")

    s = _fred("IR3TIB01KRM156N")          # 한국 3개월 은행간금리 (월간, %)
    annual = (s / 100.0 - ex.cash_spread).clip(lower=0.0)
    daily = (annual.reindex(index.union(annual.index)).ffill()
             .reindex(index).ffill().bfill() / 252.0)
    print(f"      원화금리: FRED {s.index[0].date()}~{s.index[-1].date()}, "
          f"평균 {annual.mean():.2%}")
    return daily.rename("cash")


def load_fx() -> pd.Series:
    if FX_SOURCE == "yfinance":
        s = _download("KRW=X")
    else:
        s = _fred("DEXKOUS")
    s = _naive_index(s)
    s.name = "USDKRW"
    print(f"      환율: {s.index[0].date()} ~ {s.index[-1].date()} ({len(s)}행)")
    return s


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def stats(returns: pd.Series, cash: pd.Series | None = None) -> dict:
    """CAGR / MDD / 샤프.

    Sharpe    : 원시 (평균수익 / 변동성)
    Sharpe_ex : 무위험수익 차감 후. taa/metrics.py 정의와 대조하는 용도.
                게이트 확인은 이쪽 값으로 볼 것.
    """
    curve = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    vol = returns.std() * np.sqrt(252)
    out = {
        "CAGR": curve.iloc[-1] ** (1.0 / years) - 1.0,
        "MDD": (curve / curve.cummax() - 1.0).min(),
        "Sharpe": (returns.mean() * 252) / vol if vol > 0 else np.nan,
        "Sharpe_ex": np.nan,
        "Vol": vol,
    }
    if cash is not None and vol > 0:
        rf = cash.reindex(returns.index).fillna(0.0)
        ex = returns - rf
        out["Sharpe_ex"] = (ex.mean() * 252) / (ex.std() * np.sqrt(252))
    return out


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
    ex_usd = ExecConfig(apply_tax=APPLY_TAX)

    print(f"[설정] apply_tax={APPLY_TAX}  START={START or '(전체)'}  "
          f"fx={FX_SOURCE}  cash_ccy={CASH_CCY}  krw_rate={KRW_RATE_SOURCE}")
    if APPLY_TAX:
        print("       주의: 과세 적용 상태입니다. 검증 기준값(9.9%/-14.5%)은 "
              "세전·연금계좌 기준이므로 게이트가 맞지 않습니다.")

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
    )

    if CASH_CCY == "usd":
        # 현금을 달러 단기채로 보유 -> 원화 기준으로는 환노출이 있다.
        # 엔진이 매일 cash *= (1 + rf[i]) 하므로 환변동을 합성해 넘긴다.
        #   원화 기준 수익률 = (1 + 달러금리) x (1 + 환율변동) - 1
        # 스칼라 0% 구간(위기)에 포트폴리오가 100% 달러 현금이 되므로
        # 이 항이 위기 성과를 좌우한다.
        fx_ret = fx_aligned.pct_change().fillna(0.0)
        cash_krw = ((1.0 + cash_usd) * (1.0 + fx_ret) - 1.0).rename("cash")
        print("      현금: USD 단기채 (^IRX + 환노출)")
    else:
        cash_krw = load_krw_cash(px_usd.index, ex_usd)
        print("      현금: KRW 단기채 ETF (환노출 없음)")

    res_krw = run_backtest(
        px_krw, cash_krw, strat, ex_krw,
        warmup_from=warm, signal_prices=px_usd,
    )

    print("[5/5] 집계")
    r_usd, r_krw = res_usd.returns, res_krw.returns
    comp = pd.DataFrame([
        dict(기준="USD", **stats(r_usd, cash_usd)),
        dict(기준="KRW", **stats(r_krw, cash_krw)),
    ]).set_index("기준")
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
        f"기간 {r_usd.index[0].date()} ~ {r_usd.index[-1].date()} "
        f"| 환율 {FX_SOURCE} | 현금 {CASH_CCY.upper()} | 과세 {APPLY_TAX}",
        "",
        "### 게이트 — USD 결과가 9.9% / -14.5% / 0.877 근처인가?",
        "| 기준 | CAGR | MDD | Sharpe(원시) | Sharpe(초과) | Vol |",
        "|---|---|---|---|---|---|",
    ]
    for label, row in comp.iterrows():
        md.append(f"| {label} | {pct(row['CAGR'])} | {pct(row['MDD'])} "
                  f"| {row['Sharpe']:.3f} | {row['Sharpe_ex']:.3f} "
                  f"| {pct(row['Vol'])} |")
    if not subs.empty:
        md += ["", "### 구간별",
               "| 구간 | USD 수익 | KRW 수익 | USD MDD | KRW MDD |",
               "|---|---|---|---|---|"]
        for name, row in subs.iterrows():
            md.append(f"| {name} | {pct(row['USD 수익'])} | {pct(row['KRW 수익'])} "
                      f"| {pct(row['USD MDD'])} | {pct(row['KRW MDD'])} |")
    md += ["", "### 진단", "| 항목 | USD | KRW |", "|---|---|---|"]
    for k in ("n_trades", "trades_per_year", "annual_turnover",
              "time_in_market", "avg_equity_exposure"):
        u = res_usd.diagnostics.get(k, float("nan"))
        k2 = res_krw.diagnostics.get(k, float("nan"))
        md.append(f"| {k} | {u:,.3f} | {k2:,.3f} |")
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
