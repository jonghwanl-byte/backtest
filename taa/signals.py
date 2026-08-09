"""히스테리시스 상태기계 -> 스코어 -> 목표비중.

핵심 원칙
---------
1. 상태는 경로 의존적이다. 반드시 전체 히스토리로 워밍업해야 한다.
   (짧은 창으로 계산하면 상태가 0에서 시작해 실제와 다른 신호가 나온다.)
2. 신호 계산에 미래 정보가 절대 들어가면 안 된다. rolling().mean()은 과거만 본다.
3. 체결 지연은 여기서 하지 않고 engine/fastscan에서 shift로 적용한다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def hysteresis_state(price: pd.Series, ma: pd.Series, up: float, dn: float) -> pd.Series:
    """상단 돌파 -> ON, 하단 이탈 -> OFF, 사이면 직전 상태 유지.

    원본 코드의 if/else 분기와 수학적으로 동일하되 벡터화:
        price > ma*up   -> 1
        price < ma*dn   -> 0
        그 외           -> 직전값 유지 (ffill)
    """
    upper, lower = ma * up, ma * dn
    st = pd.Series(np.nan, index=price.index)
    st[price > upper] = 1.0
    st[price < lower] = 0.0
    st = st.ffill().fillna(0.0)
    st[ma.isna()] = 0.0  # MA 미형성 구간은 OFF
    return st


def asset_score(
    price: pd.Series, up: float, dn: float, ma_windows: Sequence[int]
) -> pd.Series:
    """MA별 ON/OFF 합계 (0~len(ma_windows))."""
    total = pd.Series(0.0, index=price.index)
    for w in ma_windows:
        ma = price.rolling(window=w, min_periods=w).mean()
        total = total + hysteresis_state(price, ma, up, dn)
    return total


def asset_scalar(
    price: pd.Series,
    up: float,
    dn: float,
    ma_windows: Sequence[int],
    scalar_map: Dict[int, float],
) -> pd.Series:
    """스코어를 투입 스케일(0~1)로 변환."""
    sc = asset_score(price, up, dn, ma_windows)
    return sc.map(lambda x: scalar_map.get(int(x), 0.0)).astype(float)


class ScalarCache:
    """(티커, up, dn, ma_windows) -> 스칼라 시리즈 캐시.

    격자 탐색에서 동일 신호를 수만 번 재계산하는 것을 막는다.
    비중 시나리오/스케일링 룰이 바뀌어도 '스코어'는 재사용 가능하므로
    스코어 레벨에서 캐싱한다.
    """

    def __init__(self, prices: pd.DataFrame, ma_windows: Sequence[int]):
        self.prices = prices
        self.ma_windows = tuple(ma_windows)
        self._score: Dict[Tuple[str, float, float], np.ndarray] = {}

    def score(self, ticker: str, up: float, dn: float) -> np.ndarray:
        key = (ticker, round(up, 6), round(dn, 6))
        if key not in self._score:
            s = asset_score(self.prices[ticker], up, dn, self.ma_windows)
            self._score[key] = s.to_numpy(dtype=np.float64)
        return self._score[key]

    def scalar(
        self, ticker: str, up: float, dn: float, scalar_map: Dict[int, float]
    ) -> np.ndarray:
        sc = self.score(ticker, up, dn)
        lut = np.array([scalar_map.get(i, 0.0) for i in range(len(self.ma_windows) + 1)])
        return lut[sc.astype(np.int64)]

    def warm(self, tickers: Sequence[str], band_pairs: Sequence[Tuple[float, float]]):
        for t in tickers:
            for up, dn in band_pairs:
                self.score(t, up, dn)
        return self


def target_weights(
    prices: pd.DataFrame,
    base_weights: Dict[str, float],
    bands: Dict[str, Tuple[float, float]],
    ma_windows: Sequence[int],
    scalar_map: Dict[int, float],
) -> pd.DataFrame:
    """일별 목표비중 DataFrame (현금은 1 - 합계)."""
    out = {}
    for t, bw in base_weights.items():
        if bw <= 0:
            out[t] = pd.Series(0.0, index=prices.index)
            continue
        up, dn = bands[t]
        out[t] = bw * asset_scalar(prices[t], up, dn, ma_windows, scalar_map)
    return pd.DataFrame(out, index=prices.index)


def state_report(
    prices: pd.DataFrame,
    bands: Dict[str, Tuple[float, float]],
    ma_windows: Sequence[int],
) -> pd.DataFrame:
    """실전 봇 검증용: 마지막 날 각 MA 상태/이격도 표."""
    rows = []
    for t, (up, dn) in bands.items():
        if t not in prices.columns:
            continue
        for w in ma_windows:
            ma = prices[t].rolling(w, min_periods=w).mean()
            st = hysteresis_state(prices[t], ma, up, dn)
            rows.append(
                {
                    "ticker": t,
                    "ma": w,
                    "state": "ON" if st.iloc[-1] == 1 else "OFF",
                    "prev": "ON" if st.iloc[-2] == 1 else "OFF",
                    "price": prices[t].iloc[-1],
                    "ma_value": ma.iloc[-1],
                    "disparity": prices[t].iloc[-1] / ma.iloc[-1] - 1,
                }
            )
    return pd.DataFrame(rows)
