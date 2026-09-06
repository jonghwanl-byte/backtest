"""
slope_study.py — MA 기울기 조건 추가 실험.

가설
----
현재 신호는 가격이 MA 위/아래인지만 본다. 완만한 장기 하락(slow grind)
구간에서는 MA가 가격을 따라 내려오므로 상대 이격도가 계속 밴드 안에 머물고,
신호가 켜진 채 오래 물린다. 2026년 TLT가 이 상태다.

수정: ON 조건에 "MA 자체가 상승 중"을 추가한다.
      MA[t] > MA[t-w], lookback은 각 MA 기간과 동일 (새 파라미터 0개).
      20/120/200 세 MA 전부에 적용.

사전 합격 기준 (실행 전 확정)
--------------------------
주 기준 (전부 충족)
  - MDD        : -14.50% 대비 악화 없음
  - CAGR       : 9.97% 대비 -0.5%p 이내
  - Sharpe(초과): 0.890 대비 하락 없음
부 기준 (하나라도 위반 시 기각)
  - 2020-03~05 수익이 기존 대비 -3%p 초과 악화
  - time_in_market < 0.90

참고 지표 (판정에 사용하지 않음)
  - QQQ / TLT / GLD 자산별 단독 성과

주의: 자산별 성과는 진단용이다. TLT가 개선됐다는 이유로 전체 기각을
      뒤집으면 사후 합리화가 된다.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from taa.config import StrategyConfig, ExecConfig, TICKERS, CASH_PROXY
from taa.engine import run_backtest
from taa import signals as sig


_YF_CACHE = os.path.join(tempfile.gettempdir(), "yf_cache")
os.makedirs(_YF_CACHE, exist_ok=True)
for _s in ("set_tz_cache_location", "set_cache_location"):
    try:
        getattr(yf, _s)(_YF_CACHE)
    except Exception:
        pass


def env(key: str, default: str = "") -> str:
    v = os.environ.get(key, "")
    return v if v.strip() else default


START = env("START", "")
APPLY_TAX = env("APPLY_TAX", "false").lower() in ("1", "true", "yes")

# 기존 검증값 (게이트 및 합격 기준 기준선)
BASE = {"CAGR": 0.0997, "MDD": -0.1450, "Sharpe_ex": 0.890}


# ---------------------------------------------------------------------------
# 기울기 조건이 적용된 신호 함수
# ---------------------------------------------------------------------------

def hysteresis_state_slope(
    price: pd.Series, ma: pd.Series, up: float, dn: float, window: int
) -> pd.Series:
    """기존 히스테리시스 + MA 상승 조건.

    기존:  price > ma*up -> ON,  price < ma*dn -> OFF,  사이는 유지
    추가:  ma[t] <= ma[t-window] 이면 무조건 OFF

    기울기는 하드 게이트로 넣는다. 상태 유지 구간에서도 MA가 하락 중이면
    OFF로 강제해야 slow grind 를 잡을 수 있다. 진입 조건에만 넣으면
    이미 ON 인 상태가 그대로 유지되어 문제가 해결되지 않는다.
    """
    st = sig.hysteresis_state(price, ma, up, dn)
    rising = ma > ma.shift(window)
    return st.where(rising.fillna(False), 0.0)


def asset_score_slope(
    price: pd.Series, up: float, dn: float, ma_windows: Sequence[int]
) -> pd.Series:
    total = pd.Series(0.0, index=price.index)
    for w in ma_windows:
        ma = price.rolling(window=w, min_periods=w).mean()
        total = total + hysteresis_state_slope(price, ma, up, dn, w)
    return total


def target_weights_slope(
    prices: pd.DataFrame,
    base_weights: Dict[str, float],
    bands: Dict[str, tuple],
    ma_windows: Sequence[int],
    scalar_map: Dict[int, float],
) -> pd.DataFrame:
    out = {}
    for t, bw in base_weights.items():
        if bw <= 0:
            out[t] = pd.Series(0.0, index=prices.index)
            continue
        up, dn = bands[t]
        sc = asset_score_slope(prices[t], up, dn, ma_windows)
        out[t] = bw * sc.map(lambda x: scalar_map.get(int(x), 0.0)).astype(float)
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def _download(ticker: str, tries: int = 4) -> pd.Series:
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
                    print(f"      {ticker}: {len(s)}일")
                    return s
                last = f"{len(s)}행"
            else:
                last = "빈 응답"
        except Exception as e:
            last = repr(e)
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"{ticker} 실패: {last}")


def _naive(o):
    o.index = pd.to_datetime(o.index)
    if getattr(o.index, "tz", None) is not None:
        o.index = o.index.tz_localize(None)
    return o


def load_all() -> tuple[pd.DataFrame, pd.Series]:
    px = pd.concat({t: _download(t) for t in TICKERS}, axis=1)
    px.columns = list(TICKERS)
    px = _naive(px).dropna()
    irx = _naive(_download(CASH_PROXY))
    ex = ExecConfig()
    annual = (irx / 100.0 - ex.cash_spread).clip(lower=0.0)
    cash = (annual.reindex(px.index).ffill().bfill() / 252.0).rename("cash")
    print(f"      공통: {px.index[0].date()} ~ {px.index[-1].date()} ({len(px)}일)")
    return px, cash


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def stats(r: pd.Series, cash: pd.Series | None = None) -> dict:
    curve = (1 + r).cumprod()
    years = len(r) / 252.0
    vol = r.std() * np.sqrt(252)
    out = {
        "CAGR": curve.iloc[-1] ** (1 / years) - 1,
        "MDD": (curve / curve.cummax() - 1).min(),
        "Sharpe_ex": np.nan,
        "Vol": vol,
    }
    if cash is not None and vol > 0:
        ex = r - cash.reindex(r.index).fillna(0.0)
        out["Sharpe_ex"] = (ex.mean() * 252) / (ex.std() * np.sqrt(252))
    return out


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x*100:,.2f}%"


def period_return(r: pd.Series, s: str, e: str) -> float:
    seg = r.loc[s:e]
    return np.nan if len(seg) < 2 else (1 + seg).prod() - 1


def write_summary(text: str) -> None:
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> int:
    strat = StrategyConfig()
    ex = ExecConfig(apply_tax=APPLY_TAX)
    warm = pd.Timestamp(START) if START else None
    print(f"[설정] apply_tax={APPLY_TAX}  START={START or '(전체)'}")

    print("[1/4] 데이터")
    px, cash = load_all()

    print("[2/4] 베이스라인")
    base = run_backtest(px, cash, strat, ex, warmup_from=warm)

    print("[3/4] 기울기 조건 적용")
    orig = sig.target_weights
    try:
        # engine 은 from .signals import target_weights 로 이름을 바인딩하므로
        # 양쪽 네임스페이스를 모두 교체해야 한다.
        import taa.engine as eng
        sig.target_weights = target_weights_slope
        eng.target_weights = target_weights_slope
        slope = run_backtest(px, cash, strat, ex, warmup_from=warm)
    finally:
        sig.target_weights = orig
        eng.target_weights = orig

    print("[4/4] 집계")
    rb, rs = base.returns, slope.returns
    sb, ss = stats(rb, cash), stats(rs, cash)

    # --- 판정 -------------------------------------------------------------
    c20_b = period_return(rb, "2020-03-01", "2020-05-31")
    c20_s = period_return(rs, "2020-03-01", "2020-05-31")
    tim = slope.diagnostics["time_in_market"]

    checks = [
        ("MDD 악화 없음", ss["MDD"] >= BASE["MDD"] - 1e-9,
         f"{pct(ss['MDD'])} vs {pct(BASE['MDD'])}"),
        ("CAGR -0.5%p 이내", ss["CAGR"] >= BASE["CAGR"] - 0.005,
         f"{pct(ss['CAGR'])} vs {pct(BASE['CAGR'])}"),
        ("Sharpe(초과) 하락 없음", ss["Sharpe_ex"] >= BASE["Sharpe_ex"] - 1e-9,
         f"{ss['Sharpe_ex']:.3f} vs {BASE['Sharpe_ex']:.3f}"),
        ("2020-03~05 -3%p 이내", (c20_s - c20_b) >= -0.03,
         f"{pct(c20_s)} vs {pct(c20_b)} (차 {pct(c20_s - c20_b)})"),
        ("time_in_market >= 0.90", tim >= 0.90, f"{tim:.3f}"),
    ]
    verdict = "채택" if all(ok for _, ok, _ in checks) else "기각"

    # --- 자산별 단독 성과 (참고) -----------------------------------------
    solo = []
    for t in TICKERS:
        one = {k: (1.0 if k == t else 0.0) for k in TICKERS}
        s1 = StrategyConfig(base_weights=one, bands=strat.bands,
                            ma_windows=strat.ma_windows,
                            scalar_map=strat.scalar_map)
        rb1 = run_backtest(px, cash, s1, ex, warmup_from=warm).returns
        try:
            import taa.engine as eng
            sig.target_weights = target_weights_slope
            eng.target_weights = target_weights_slope
            rs1 = run_backtest(px, cash, s1, ex, warmup_from=warm).returns
        finally:
            sig.target_weights = orig
            eng.target_weights = orig
        a, b = stats(rb1, cash), stats(rs1, cash)
        solo.append({
            "자산": t,
            "CAGR(기존)": a["CAGR"], "CAGR(기울기)": b["CAGR"],
            "MDD(기존)": a["MDD"], "MDD(기울기)": b["MDD"],
            "SR(기존)": a["Sharpe_ex"], "SR(기울기)": b["Sharpe_ex"],
        })
    solo_df = pd.DataFrame(solo).set_index("자산")

    # --- 출력 -------------------------------------------------------------
    comp = pd.DataFrame([dict(설정="기존", **sb),
                         dict(설정="기울기", **ss)]).set_index("설정")
    comp.to_csv("slope_summary.csv", encoding="utf-8-sig")
    solo_df.to_csv("slope_solo.csv", encoding="utf-8-sig")
    pd.DataFrame({"base": rb, "slope": rs}).to_csv(
        "slope_returns.csv", encoding="utf-8-sig")

    print("\n" + comp.to_string())
    print("\n" + solo_df.to_string())
    print(f"\n판정: {verdict}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("\n기존 진단:", {k: round(v, 3) for k, v in base.diagnostics.items()})
    print("기울기 진단:", {k: round(v, 3) for k, v in slope.diagnostics.items()})

    md = [f"## MA 기울기 조건 실험 — **{verdict}**", "",
          "### 사전 기준 대조", "| 기준 | 결과 | 판정 |", "|---|---|---|"]
    for name, ok, detail in checks:
        md.append(f"| {name} | {detail} | {'통과' if ok else '**미달**'} |")
    md += ["", "### 전체 성과", "| 설정 | CAGR | MDD | Sharpe(초과) | Vol |",
           "|---|---|---|---|---|"]
    for label, row in comp.iterrows():
        md.append(f"| {label} | {pct(row['CAGR'])} | {pct(row['MDD'])} "
                  f"| {row['Sharpe_ex']:.3f} | {pct(row['Vol'])} |")
    md += ["", "### 자산별 단독 성과 (참고 — 판정에 사용하지 않음)",
           "| 자산 | CAGR 기존→기울기 | MDD 기존→기울기 | SR 기존→기울기 |",
           "|---|---|---|---|"]
    for t, row in solo_df.iterrows():
        md.append(f"| {t} | {pct(row['CAGR(기존)'])} → {pct(row['CAGR(기울기)'])} "
                  f"| {pct(row['MDD(기존)'])} → {pct(row['MDD(기울기)'])} "
                  f"| {row['SR(기존)']:.3f} → {row['SR(기울기)']:.3f} |")
    md += ["", "### 진단", "| 항목 | 기존 | 기울기 |", "|---|---|---|"]
    for k in ("trades_per_year", "annual_turnover", "time_in_market",
              "avg_equity_exposure"):
        md.append(f"| {k} | {base.diagnostics[k]:,.3f} "
                  f"| {slope.diagnostics[k]:,.3f} |")
    write_summary("\n".join(md))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
