"""구간 분할. '언제 통했고 언제 안 통했나'를 분리해서 본다."""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

# 매크로 레짐 (GLD 상장 2004-11 이후 3자산 동시 가능)
REGIMES: Dict[str, Tuple[str, str]] = {
    "2005-2007 유동성 랠리": ("2005-01-01", "2007-10-08"),
    "2007-2009 글로벌 금융위기": ("2007-10-09", "2009-03-09"),
    "2009-2011 회복+QE": ("2009-03-10", "2011-04-29"),
    "2011-2012 유럽 재정위기": ("2011-04-30", "2012-06-01"),
    "2012-2015 저변동 상승": ("2012-06-02", "2015-07-20"),
    "2015-2016 차이나 쇼크": ("2015-07-21", "2016-02-11"),
    "2016-2018 트럼프 랠리": ("2016-02-12", "2018-01-26"),
    "2018 변동성/4분기 급락": ("2018-01-27", "2018-12-24"),
    "2019 완화 랠리": ("2018-12-25", "2020-02-19"),
    "2020 코로나 크래시": ("2020-02-20", "2020-03-23"),
    "2020-2021 유동성 버블": ("2020-03-24", "2021-12-31"),
    "2022 금리인상 약세장": ("2022-01-01", "2022-10-12"),
    "2022-2025 AI 랠리": ("2022-10-13", "2030-12-31"),
}

# 표본 내 / 표본 외 (README 최적화 구간 = 최근 10년)
IS_OOS: Dict[str, Tuple[str, str]] = {
    "OOS_초기 (2005-2014)": ("2005-01-01", "2014-12-31"),
    "IS_최적화구간 (2015-2024)": ("2015-01-01", "2024-12-31"),
    "OOS_최신 (2025-)": ("2025-01-01", "2030-12-31"),
}

# 특성별 시장 국면
MARKET_TYPES: Dict[str, List[Tuple[str, str]]] = {
    "급락장 (V자 회복)": [("2020-02-20", "2020-08-31"), ("2018-09-20", "2019-04-30")],
    "완만한 약세장 (휩쏘 다발)": [("2007-10-09", "2009-03-09"), ("2022-01-01", "2022-12-31")],
    "횡보장": [("2011-04-30", "2012-06-01"), ("2015-05-01", "2016-06-30")],
    "강한 추세 상승장": [("2013-01-01", "2014-12-31"), ("2023-01-01", "2024-12-31")],
}


def slice_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return np.asarray((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))


def rolling_windows(
    index: pd.DatetimeIndex, years: int = 3, step_months: int = 6
) -> Iterator[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """겹치는 롤링 구간. '어느 3년을 잘라도 되는가' 검증."""
    start = index[0]
    end = index[-1]
    cur = start
    while cur + pd.DateOffset(years=years) <= end:
        stop = cur + pd.DateOffset(years=years)
        yield f"{cur.date()}~{stop.date()}", cur, stop
        cur = cur + pd.DateOffset(months=step_months)


def walk_forward_splits(
    index: pd.DatetimeIndex, is_years: int = 4, oos_years: int = 1
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(IS시작, IS끝=OOS시작, OOS끝) 목록."""
    out = []
    cur = index[0]
    end = index[-1]
    while True:
        is_end = cur + pd.DateOffset(years=is_years)
        oos_end = is_end + pd.DateOffset(years=oos_years)
        if is_end >= end:
            break
        out.append((cur, is_end, min(oos_end, end)))
        cur = cur + pd.DateOffset(years=oos_years)
    return out


def block_bootstrap(
    rets: pd.Series, n_sims: int = 1000, block: int = 21, seed: int = 0
) -> pd.DataFrame:
    """블록 부트스트랩. MDD/샤프의 신뢰구간 추정.

    '내가 본 MDD -11%'가 운이었는지, 재추출해도 유지되는지 확인.
    """
    rng = np.random.default_rng(seed)
    r = rets.dropna().to_numpy()
    n = len(r)
    nb = int(np.ceil(n / block))
    rows = []
    for _ in range(n_sims):
        starts = rng.integers(0, max(n - block, 1), size=nb)
        path = np.concatenate([r[s : s + block] for s in starts])[:n]
        eq = np.cumprod(1 + path)
        dd = eq / np.maximum.accumulate(eq) - 1
        rows.append(
            {
                "CAGR": eq[-1] ** (252 / n) - 1,
                "MDD": dd.min(),
                "Sharpe": path.mean() / path.std(ddof=1) * np.sqrt(252) if path.std() > 0 else 0,
            }
        )
    return pd.DataFrame(rows)
