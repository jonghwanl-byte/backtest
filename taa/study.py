"""검증 스위트. 각 함수가 '의견'이 아니라 '숫자'를 낸다."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SCALAR_RULES, WEIGHT_SCENARIOS, ExecConfig, GridSpec, StrategyConfig
from .engine import buy_and_hold, run_backtest, simple_ma_filter
from .fastscan import GridScanner, grid_distribution, neighborhood_robustness, surface
from .metrics import (
    annual_table,
    deflated_sharpe,
    haircut_sharpe,
    perf_stats,
    vol_target_scale,
)
from .periods import (
    IS_OOS,
    MARKET_TYPES,
    REGIMES,
    block_bootstrap,
    rolling_windows,
    slice_mask,
    walk_forward_splits,
)


# ============================================================================
# 검증 1. 룩어헤드 민감도
# ============================================================================
def test_execution_lag(
    prices, cash, strat: StrategyConfig, ex: ExecConfig, lags=(-1, 0, 1, 2)
) -> pd.DataFrame:
    """체결 시점을 바꿔가며 성과 붕괴 정도 측정.

    lag=-1(당일 신호를 당일 수익에 적용)에서만 성적이 좋다면 룩어헤드 착시다.
    """
    rows = []
    for lag in lags:
        sc = GridScanner(prices, cash, strat.ma_windows, exec_lag=lag,
                         cost_rate=(ex.cost_bps + ex.slippage_bps) / 10_000)
        r = sc.series_for(strat.bands, strat.base_weights, strat.scalar_map)
        st = perf_stats(r, cash)
        label = {-1: "당일신호=당일수익 (룩어헤드)", 0: "당일 종가 체결",
                 1: "익일 종가 체결 (현실)", 2: "2일 지연"}.get(lag, f"lag={lag}")
        rows.append({"exec_lag": lag, "설명": label,
                     **{k: st[k] for k in ("CAGR", "Vol", "Sharpe", "MDD", "MDD/Vol", "Calmar")}})
    df = pd.DataFrame(rows)
    base = df.loc[df.exec_lag == 1, "Sharpe"].values
    if len(base):
        df["Sharpe_vs_현실"] = df["Sharpe"] - base[0]
    return df


# ============================================================================
# 검증 2. 구간별 성과 (레짐 / IS-OOS / 시장국면 / 롤링)
# ============================================================================
def _stats_for_mask(r: pd.Series, cash: pd.Series, mask) -> Dict[str, float]:
    sub = r[mask]
    if len(sub) < 40:
        return {}
    st = perf_stats(sub, cash)
    return {k: st.get(k, np.nan) for k in ("CAGR", "Vol", "Sharpe", "MDD", "Calmar", "Years")}


def subperiod_table(
    r: pd.Series, cash: pd.Series, periods: Dict[str, Tuple[str, str]],
    bench: pd.Series | None = None
) -> pd.DataFrame:
    rows = []
    for name, (s, e) in periods.items():
        m = slice_mask(r.index, s, e)
        st = _stats_for_mask(r, cash, m)
        if not st:
            continue
        row = {"구간": name, "시작": str(pd.Timestamp(s).date()), **st}
        if bench is not None:
            b = _stats_for_mask(bench.reindex(r.index).fillna(0), cash, m)
            row["Bench_CAGR"] = b.get("CAGR", np.nan)
            row["Bench_MDD"] = b.get("MDD", np.nan)
            row["초과"] = row["CAGR"] - row["Bench_CAGR"]
        rows.append(row)
    return pd.DataFrame(rows)


def market_type_table(r: pd.Series, cash: pd.Series, bench: pd.Series | None = None):
    rows = []
    for name, spans in MARKET_TYPES.items():
        m = np.zeros(len(r), dtype=bool)
        for s, e in spans:
            m |= slice_mask(r.index, s, e)
        st = _stats_for_mask(r, cash, m)
        if not st:
            continue
        row = {"국면": name, **st}
        if bench is not None:
            b = _stats_for_mask(bench.reindex(r.index).fillna(0), cash, m)
            row["Bench_CAGR"] = b.get("CAGR", np.nan)
            row["초과"] = row["CAGR"] - row["Bench_CAGR"]
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_table(r: pd.Series, cash: pd.Series, years: int = 3, step_months: int = 6):
    rows = []
    for label, s, e in rolling_windows(r.index, years, step_months):
        m = np.asarray((r.index >= s) & (r.index < e))
        st = _stats_for_mask(r, cash, m)
        if st:
            rows.append({"window": label, **st})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["summary"] = {
            "worst_CAGR": df.CAGR.min(), "median_CAGR": df.CAGR.median(),
            "worst_Sharpe": df.Sharpe.min(), "median_Sharpe": df.Sharpe.median(),
            "pct_windows_positive": (df.CAGR > 0).mean(),
            "worst_MDD": df.MDD.min(),
        }
    return df


# ============================================================================
# 검증 3. 격자 전수 스캔 + 견고성
# ============================================================================
def full_grid(
    prices, cash, spec: GridSpec, ex: ExecConfig,
    ma_windows=(20, 120, 200),
    weight_scenarios: Dict[str, Dict[str, float]] | None = None,
    rules: Dict[str, Dict[int, float]] | None = None,
    mask=None, vary: Sequence[str] | None = ("QQQ",),
    fixed_bands: Dict | None = None, link_bands: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    weight_scenarios = weight_scenarios or WEIGHT_SCENARIOS
    rules = rules or {"100/50/25/0": SCALAR_RULES["100/50/25/0"]}
    fixed_bands = fixed_bands or {"QQQ": (1.03, 0.98), "TLT": (1.03, 0.98), "GLD": (1.025, 0.97)}

    sc = GridScanner(prices, cash, ma_windows, ex.exec_lag,
                     (ex.cost_bps + ex.slippage_bps) / 10_000)
    bp = spec.band_pairs()
    n_vary = len(vary) if (vary and not link_bands) else 1
    total = (len(bp) if link_bands else len(bp) ** n_vary) * len(weight_scenarios) * len(rules)
    if verbose:
        print(f"  격자 케이스 수: {total:,}")

    out = []
    for wname, w in weight_scenarios.items():
        for rname, rule in rules.items():
            df = sc.scan(bp, w, rule, rname, mask=mask, link_bands=link_bands,
                         fixed_bands=fixed_bands, vary=vary)
            df["scenario"] = wname
            out.append(df)
    res = pd.concat(out, ignore_index=True)
    res.attrs["n_trials"] = len(res)
    return res


def robustness_report(df: pd.DataFrame, ticker="QQQ", metric="Sharpe") -> Dict:
    dist = grid_distribution(df, metric)
    surf = surface(df, ticker, metric)
    nb = neighborhood_robustness(df, ticker, metric)
    best = df.loc[df[metric].idxmax()]
    bi, bj = best[f"{ticker}_up"], best[f"{ticker}_dn"]
    try:
        nb_score = float(nb.loc[bi, bj])
    except KeyError:
        nb_score = np.nan
    return {
        "distribution": dist,
        "surface": surf,
        "neighborhood": nb,
        "best_cell": (bi, bj),
        "best_metric": float(best[metric]),
        "neighborhood_score": nb_score,
        "isolation": float(best[metric]) - nb_score,
        "verdict": ("고원(견고)" if (best[metric] - nb_score) < 0.05
                    else "완만" if (best[metric] - nb_score) < 0.12 else "첨탑(과최적화 의심)"),
    }


def overfit_diagnostics(df: pd.DataFrame, n_obs: int, metric="Sharpe") -> Dict:
    s = df[metric].to_numpy()
    best = float(s.max())
    dsr = deflated_sharpe(best, float(np.var(s, ddof=1)), len(s), n_obs)
    dsr["haircut_Sharpe"] = haircut_sharpe(best, len(s), n_obs)
    dsr["median_Sharpe"] = float(np.median(s))
    dsr["best_minus_median"] = best - float(np.median(s))
    return dsr


# ============================================================================
# 검증 4. 워크포워드 (진짜 기대치)
# ============================================================================
def walk_forward(
    prices, cash, spec: GridSpec, ex: ExecConfig,
    base_weights: Dict[str, float], scalar_map: Dict[int, float],
    ma_windows=(20, 120, 200), is_years=4, oos_years=1,
    vary=("QQQ",), objective="Sharpe", verbose=True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """IS에서 최적화 -> OOS에 적용 -> OOS 수익률만 이어붙임.

    이 결과가 실전에서 기대할 수 있는 유일하게 정직한 숫자다.
    """
    sc = GridScanner(prices, cash, ma_windows, ex.exec_lag,
                     (ex.cost_bps + ex.slippage_bps) / 10_000)
    bp = spec.band_pairs()
    idx = prices.index
    fixed = {"QQQ": (1.03, 0.98), "TLT": (1.03, 0.98), "GLD": (1.025, 0.97)}

    oos_parts, log = [], []
    for is_s, is_e, oos_e in walk_forward_splits(idx, is_years, oos_years):
        m_is = np.asarray((idx >= is_s) & (idx < is_e))
        if m_is.sum() < 300:
            continue
        res = sc.scan(bp, base_weights, scalar_map, "wf", mask=m_is,
                      vary=vary, fixed_bands=fixed)
        best = res.loc[res[objective].idxmax()]
        bands = dict(fixed)
        for t in vary:
            bands[t] = (1 + best[f"{t}_up"], 1 + best[f"{t}_dn"])  # dn은 이미 음수 저장

        r_all = sc.series_for(bands, base_weights, scalar_map)
        m_oos = np.asarray((idx >= is_e) & (idx < oos_e))
        oos = r_all[m_oos]
        if len(oos) < 20:
            continue
        oos_parts.append(oos)
        st = perf_stats(oos, cash)
        log.append({
            "IS": f"{is_s.date()}~{is_e.date()}", "OOS": f"{is_e.date()}~{oos_e.date()}",
            "선택밴드": " ".join(f"{t}{best[f'{t}_up']:+.1%}/{best[f'{t}_dn']:+.1%}" for t in vary),
            "IS_Sharpe": float(best[objective]),
            "OOS_Sharpe": st.get("Sharpe", np.nan),
            "OOS_CAGR": st.get("CAGR", np.nan), "OOS_MDD": st.get("MDD", np.nan),
        })
        if verbose:
            print(f"    WF {log[-1]['OOS']}: IS SR={log[-1]['IS_Sharpe']:.2f} "
                  f"-> OOS SR={log[-1]['OOS_Sharpe']:.2f}")

    stitched = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    stitched.name = "walk_forward_OOS"
    df = pd.DataFrame(log)
    if not df.empty:
        df.attrs["degradation"] = float(df.IS_Sharpe.mean() - df.OOS_Sharpe.mean())
        df.attrs["hit_rate"] = float((df.OOS_Sharpe > 0).mean())
    return stitched, df


# ============================================================================
# 검증 5. 공정 비교 (변동성 정규화) + 세금 + 벤치마크
# ============================================================================
def vol_matched_comparison(
    series_map: Dict[str, pd.Series], cash: pd.Series, target_vol=0.10
) -> pd.DataFrame:
    rows = []
    for name, r in series_map.items():
        raw = perf_stats(r, cash)
        vt = perf_stats(vol_target_scale(r, cash, target_vol), cash)
        rows.append({
            "전략": name,
            "CAGR": raw.get("CAGR"), "Vol": raw.get("Vol"),
            "MDD": raw.get("MDD"), "Sharpe": raw.get("Sharpe"),
            f"CAGR@{target_vol:.0%}vol": vt.get("CAGR"),
            f"MDD@{target_vol:.0%}vol": vt.get("MDD"),
            "MDD/Vol": raw.get("MDD/Vol"),
        })
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


def tax_impact(prices, cash, strat: StrategyConfig, ex: ExecConfig,
               warmup_from=None) -> pd.DataFrame:
    rows = []
    for label, cfg in [
        ("비용/세금 없음", replace(ex, cost_bps=0, slippage_bps=0, apply_tax=False)),
        ("거래비용만", replace(ex, apply_tax=False)),
        ("거래비용+양도세", ex),
        ("+정수주(소액계좌)", replace(ex, fractional_shares=False, initial_capital=20_000)),
        ("매일 리밸런싱", replace(ex, rebalance_mode="daily")),
    ]:
        res = run_backtest(prices, cash, strat, cfg, warmup_from)
        st = perf_stats(res.returns, cash)
        rows.append({
            "가정": label, "CAGR": st.get("CAGR"), "MDD": st.get("MDD"),
            "Sharpe": st.get("Sharpe"),
            "연매매횟수": res.diagnostics["trades_per_year"],
            "누적수수료$": res.diagnostics["total_fees"],
            "누적세금$": res.diagnostics["total_tax"],
        })
    df = pd.DataFrame(rows)
    df["CAGR_손실"] = df["CAGR"].iloc[0] - df["CAGR"]
    return df


def benchmark_suite(prices, cash, ex: ExecConfig) -> Dict[str, pd.Series]:
    c = (ex.cost_bps + ex.slippage_bps) / 10_000
    out = {
        "QQQ 매수보유": buy_and_hold(prices, {"QQQ": 1.0}),
        "정적 80/10/10": buy_and_hold(prices, {"QQQ": 0.8, "TLT": 0.1, "GLD": 0.1}),
        "정적 60/20/20": buy_and_hold(prices, {"QQQ": 0.6, "TLT": 0.2, "GLD": 0.2}),
        "QQQ 200MA 단순필터": simple_ma_filter(prices, cash, "QQQ", 200, ex.exec_lag, c),
        "QQQ 120MA 단순필터": simple_ma_filter(prices, cash, "QQQ", 120, ex.exec_lag, c),
    }
    return out


def bootstrap_ci(r: pd.Series, n_sims=2000, block=21) -> pd.DataFrame:
    bs = block_bootstrap(r, n_sims, block)
    obs = perf_stats(r)
    return pd.DataFrame({
        "관측치": [obs.get("CAGR"), obs.get("MDD"), obs.get("Sharpe")],
        "p5": bs.quantile(0.05).values,
        "중앙값": bs.median().values,
        "p95": bs.quantile(0.95).values,
    }, index=bs.columns).assign(
        백분위=lambda d: [(bs[c] <= d.loc[c, "관측치"]).mean() for c in bs.columns])
