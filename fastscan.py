"""대규모 격자 스캔 (수십만 케이스).

수학적 분해
-----------
각 자산의 신호는 자기 밴드에만 의존하므로, 포트폴리오 수익률은
자산별로 미리 계산한 벡터의 선형 결합으로 표현된다:

    ret_t = rf_t + Σ_a  bw_a · S_a[t] · (r_a,t − rf_t)  −  c · Σ_a bw_a · |ΔS_a[t]|

여기서 S_a 는 이미 체결지연이 반영된 스칼라 시리즈.
따라서 밴드 조합 수가 K^3 개여도 자산별 K개만 계산하면 된다.

주의: 이 스캐너는 '매일 목표비중으로 리밸런싱' + '세금 미반영' 근사다.
최종 후보는 반드시 engine.run_backtest()로 전수 검증할 것.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import perf_matrix
from .signals import ScalarCache


def effective_scalar(scalar: np.ndarray, exec_lag: int) -> np.ndarray:
    """체결 지연 반영.

    exec_lag = -1 : 당일 신호를 당일 수익에 적용  ->  룩어헤드 (비교용)
    exec_lag =  0 : 당일 종가 체결, 익일 수익부터 반영 (관행적 백테스트)
    exec_lag =  1 : 익일 종가 체결, 모레 수익부터 반영 (현실, 기본값)
    """
    shift = 1 + exec_lag
    out = np.empty_like(scalar)
    if shift <= 0:
        return scalar.copy()
    out[:shift] = 0.0
    out[shift:] = scalar[:-shift]
    return out


class GridScanner:
    def __init__(
        self,
        prices: pd.DataFrame,
        cash_daily: pd.Series,
        ma_windows: Sequence[int] = (20, 120, 200),
        exec_lag: int = 1,
        cost_rate: float = 0.0010,
    ):
        self.prices = prices
        self.index = prices.index
        self.rf = cash_daily.reindex(self.index).fillna(0.0).to_numpy(dtype=np.float64)
        self.rets = prices.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        self.tickers = list(prices.columns)
        self.cache = ScalarCache(prices, ma_windows)
        self.exec_lag = exec_lag
        self.cost_rate = cost_rate
        self._D: Dict[Tuple[str, float, float, str], np.ndarray] = {}
        self._T: Dict[Tuple[str, float, float, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    def _components(self, ticker: str, band, scalar_map, rule_name: str):
        key = (ticker, round(band[0], 6), round(band[1], 6), rule_name)
        if key not in self._D:
            s = self.cache.scalar(ticker, band[0], band[1], scalar_map)
            s = effective_scalar(s, self.exec_lag)
            j = self.tickers.index(ticker)
            self._D[key] = s * (self.rets[:, j] - self.rf)
            turn = np.abs(np.diff(s, prepend=0.0))
            self._T[key] = turn
        return self._D[key], self._T[key]

    # ------------------------------------------------------------------
    def scan(
        self,
        band_pairs: Sequence[Tuple[float, float]],
        base_weights: Dict[str, float],
        scalar_map: Dict[int, float],
        rule_name: str = "default",
        mask: np.ndarray | None = None,
        chunk: int = 400,
        link_bands: bool = False,
        fixed_bands: Dict[str, Tuple[float, float]] | None = None,
        vary: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """밴드 격자 전수 평가.

        link_bands=True  : 세 자산 동일 밴드 (케이스 수 = K)
        vary=['QQQ']     : QQQ만 격자, 나머지는 fixed_bands 사용
        기본             : 전 조합 (K^n)
        """
        active = [t for t in self.tickers if base_weights.get(t, 0) > 0]
        vary = list(vary) if vary else active
        fixed_bands = fixed_bands or {}

        combos: List[Dict[str, Tuple[float, float]]] = []
        if link_bands:
            for bp in band_pairs:
                combos.append({t: bp for t in active})
        else:
            def rec(i, cur):
                if i == len(vary):
                    full = dict(cur)
                    for t in active:
                        if t not in full:
                            full[t] = fixed_bands.get(t, band_pairs[0])
                    combos.append(full)
                    return
                for bp in band_pairs:
                    cur[vary[i]] = bp
                    rec(i + 1, cur)
            rec(0, {})

        rows, buf, meta = [], [], []
        for cb in combos:
            base = self.rf.copy()
            for t in active:
                D, Tn = self._components(t, cb[t], scalar_map, rule_name)
                bw = base_weights[t]
                base = base + bw * D - self.cost_rate * bw * Tn
            buf.append(base)
            meta.append(cb)
            if len(buf) >= chunk:
                rows.extend(self._flush(buf, meta, base_weights, scalar_map, rule_name, mask))
                buf, meta = [], []
        if buf:
            rows.extend(self._flush(buf, meta, base_weights, scalar_map, rule_name, mask))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _flush(self, buf, meta, base_weights, scalar_map, rule_name, mask):
        M = np.column_stack(buf)
        rf = self.rf
        if mask is not None:
            M, rf = M[mask], rf[mask]
        stats = perf_matrix(M, rf)

        # 회전율 (연간, 단면)
        out = []
        for k, cb in enumerate(meta):
            row = {f"{t}_band": f"{cb[t][0] - 1:+.1%}/{cb[t][1] - 1:+.1%}" for t in cb}
            for t in cb:
                row[f"{t}_up"] = round(cb[t][0] - 1, 4)
                row[f"{t}_dn"] = round(cb[t][1] - 1, 4)
            row["rule"] = rule_name
            row["weights"] = "/".join(f"{int(base_weights.get(t, 0) * 100)}" for t in self.tickers)
            for name, arr in stats.items():
                row[name] = float(arr[k])
            out.append(row)
        return out

    # ------------------------------------------------------------------
    def series_for(
        self,
        bands: Dict[str, Tuple[float, float]],
        base_weights: Dict[str, float],
        scalar_map: Dict[int, float],
        rule_name: str = "default",
    ) -> pd.Series:
        """단일 조합의 일간 수익률 시리즈 (근사 모델)."""
        base = self.rf.copy()
        for t, bw in base_weights.items():
            if bw <= 0:
                continue
            D, Tn = self._components(t, bands[t], scalar_map, rule_name)
            base = base + bw * D - self.cost_rate * bw * Tn
        return pd.Series(base, index=self.index, name="strategy")

    def turnover_for(
        self, bands, base_weights, scalar_map, rule_name: str = "default"
    ) -> float:
        """연평균 편도 회전율."""
        tot = np.zeros(len(self.index))
        for t, bw in base_weights.items():
            if bw <= 0:
                continue
            _, Tn = self._components(t, bands[t], scalar_map, rule_name)
            tot += bw * Tn
        return float(tot.sum() / (len(self.index) / 252))


# ----------------------------------------------------------------------------
# 격자 분포 요약 (1등만 보지 말라)
# ----------------------------------------------------------------------------
def grid_distribution(df: pd.DataFrame, metric: str = "Sharpe") -> pd.Series:
    s = df[metric]
    return pd.Series(
        {
            "n_cases": len(s),
            "max": s.max(),
            "p95": s.quantile(0.95),
            "p75": s.quantile(0.75),
            "median": s.median(),
            "p25": s.quantile(0.25),
            "min": s.min(),
            "std": s.std(),
            "max_minus_median": s.max() - s.median(),
            "top_decile_share": (s >= s.quantile(0.9)).mean(),
        }
    )


def surface(df: pd.DataFrame, ticker: str = "QQQ", metric: str = "Sharpe") -> pd.DataFrame:
    """(상단, 하단) 2D 표면. 첨탑 vs 고원 판정용."""
    return df.pivot_table(
        index=f"{ticker}_up", columns=f"{ticker}_dn", values=metric, aggfunc="mean"
    ).sort_index(ascending=False)


def neighborhood_robustness(
    df: pd.DataFrame, ticker: str = "QQQ", metric: str = "Sharpe"
) -> pd.DataFrame:
    """각 셀의 '이웃 평균' 점수. 최적점이 고립되어 있으면 과최적화 신호."""
    piv = surface(df, ticker, metric)
    arr = piv.to_numpy(dtype=float)
    pad = np.pad(arr, 1, mode="edge")
    nb = np.zeros_like(arr)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            nb += pad[1 + di : 1 + di + arr.shape[0], 1 + dj : 1 + dj + arr.shape[1]]
    nb /= 9.0
    out = pd.DataFrame(nb, index=piv.index, columns=piv.columns)
    return out
