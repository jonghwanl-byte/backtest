"""성과 지표. 1D(단일 케이스) + 2D(격자 스캔) 벡터화 버전 모두 제공."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER = 0.5772156649015329


# ----------------------------------------------------------------------------
# 2D 벡터화 (열 = 케이스)
# ----------------------------------------------------------------------------
def perf_matrix(rets: np.ndarray, rf: np.ndarray) -> Dict[str, np.ndarray]:
    """rets: (T, K) 일간 수익률, rf: (T,) 일간 무위험수익률."""
    if rets.ndim == 1:
        rets = rets[:, None]
    T = rets.shape[0]
    eq = np.cumprod(1.0 + rets, axis=0)
    peak = np.maximum.accumulate(eq, axis=0)
    dd = eq / peak - 1.0

    years = T / TRADING_DAYS
    cagr = eq[-1] ** (1.0 / years) - 1.0
    vol = rets.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

    ex = rets - rf[:, None]
    sharpe = np.divide(
        ex.mean(axis=0) * TRADING_DAYS,
        ex.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS),
        out=np.zeros(rets.shape[1]),
        where=ex.std(axis=0, ddof=1) > 0,
    )
    downside = np.where(ex < 0, ex, 0.0)
    dstd = np.sqrt((downside ** 2).sum(axis=0) / max(T - 1, 1)) * np.sqrt(TRADING_DAYS)
    sortino = np.divide(
        ex.mean(axis=0) * TRADING_DAYS, dstd, out=np.zeros(rets.shape[1]), where=dstd > 0
    )
    mdd = dd.min(axis=0)
    ulcer = np.sqrt((dd ** 2).mean(axis=0))

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Calmar": np.divide(cagr, -mdd, out=np.zeros_like(cagr), where=mdd < 0),
        "MDD/Vol": np.divide(-mdd, vol, out=np.zeros_like(vol), where=vol > 0),
        "Ulcer": ulcer,
    }


# ----------------------------------------------------------------------------
# 1D 상세 (단일 케이스 전체 리포트)
# ----------------------------------------------------------------------------
def perf_stats(
    rets: pd.Series, rf: pd.Series | None = None, extra: Dict | None = None
) -> Dict[str, float]:
    r = rets.dropna()
    if len(r) < 20:
        return {}
    rf_a = (rf.reindex(r.index).fillna(0.0).to_numpy() if rf is not None else np.zeros(len(r)))
    m = perf_matrix(r.to_numpy()[:, None], rf_a)
    out = {k: float(v[0]) for k, v in m.items()}

    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1.0

    # 낙폭 회복 기간
    under = dd < -1e-9
    longest, cur = 0, 0
    for u in under.to_numpy():
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    out["MDD_Days"] = float(longest)

    ann = (1 + r).groupby(r.index.year).prod() - 1
    out["BestYear"] = float(ann.max())
    out["WorstYear"] = float(ann.min())
    out["PosYears"] = float((ann > 0).mean())

    mon = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out["MonthlyWin"] = float((mon > 0).mean())
    out["Skew"] = float(stats.skew(r))
    out["Kurtosis"] = float(stats.kurtosis(r, fisher=True))
    out["VaR95"] = float(np.percentile(r, 5))
    out["CVaR95"] = float(r[r <= np.percentile(r, 5)].mean())
    out["Years"] = len(r) / TRADING_DAYS
    if extra:
        out.update(extra)
    return out


def annual_table(rets: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    r = rets.dropna()
    rows = {"Return": (1 + r).groupby(r.index.year).prod() - 1}
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    rows["MDD"] = dd.groupby(dd.index.year).min()
    rows["Vol"] = r.groupby(r.index.year).std() * np.sqrt(TRADING_DAYS)
    df = pd.DataFrame(rows)
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        df["Bench"] = (1 + b).groupby(b.index.year).prod() - 1
        df["Excess"] = df["Return"] - df["Bench"]
    return df


# ----------------------------------------------------------------------------
# 과최적화 진단
# ----------------------------------------------------------------------------
def deflated_sharpe(
    sr_ann: float,
    sr_variance_across_trials: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> Dict[str, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    n_trials 번 탐색해서 뽑은 최고 샤프가, 진짜 실력인지 다중검정 노이즈인지 판정.
    반환 DSR = P(진짜 샤프 > 0). 0.95 미만이면 통계적으로 유의하다고 보기 어렵다.
    """
    sr = sr_ann / np.sqrt(TRADING_DAYS)          # 일간 단위로 변환
    sr_std = np.sqrt(max(sr_variance_across_trials, 1e-12)) / np.sqrt(TRADING_DAYS)
    K = max(int(n_trials), 2)

    # 순수 노이즈에서 K회 시행 시 기대되는 최대 샤프 (SR0)
    z1 = stats.norm.ppf(1 - 1.0 / K)
    z2 = stats.norm.ppf(1 - 1.0 / (K * np.e))
    sr0 = sr_std * ((1 - EULER) * z1 + EULER * z2)

    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-9))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom)

    return {
        "SR_ann": sr_ann,
        "SR0_ann": float(sr0 * np.sqrt(TRADING_DAYS)),
        "DSR": float(dsr),
        "n_trials": K,
        "verdict": "통과" if dsr > 0.95 else ("의심" if dsr > 0.75 else "탈락"),
    }


def haircut_sharpe(sr_ann: float, n_trials: int, n_obs: int) -> float:
    """다중검정 보정 후 기대 샤프 (Bonferroni 근사)."""
    se = np.sqrt((1 + 0.5 * (sr_ann / np.sqrt(TRADING_DAYS)) ** 2) / n_obs) * np.sqrt(TRADING_DAYS)
    z = stats.norm.ppf(1 - 0.05 / (2 * max(n_trials, 1)))
    return float(sr_ann - z * se)


def vol_target_scale(rets: pd.Series, rf: pd.Series, target_vol: float = 0.10) -> pd.Series:
    """변동성을 target_vol로 맞춘 수익률.

    '리스크 관리 개선'과 '그냥 덜 투자한 것'을 구분하기 위한 공정 비교 도구.
    초과수익 부분만 스케일하고 무위험수익은 그대로 둔다 (레버리지/디레버리지 가정).
    """
    ex = rets - rf.reindex(rets.index).fillna(0.0)
    realized = ex.std() * np.sqrt(TRADING_DAYS)
    k = target_vol / realized if realized > 0 else 1.0
    return (rf.reindex(rets.index).fillna(0.0) + k * ex).rename(f"{rets.name}_vt")
