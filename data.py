"""가격/현금금리 데이터 로딩. 재시도 + 검증 + 로컬 캐시."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import CASH_PROXY, INCEPTION

CACHE_DIR = Path(os.environ.get("TAA_CACHE", "./_cache"))


# ----------------------------------------------------------------------------
# 다운로드
# ----------------------------------------------------------------------------
def _download(tickers: Sequence[str], start: str, end: str | None, retries: int = 4):
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                list(tickers),
                start=start,
                end=end,
                auto_adjust=True,       # 명시적으로 배당/분할 조정. Adj Close 폴백 금지.
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df
            last_err = ValueError("빈 데이터프레임")
        except Exception as e:  # noqa: BLE001
            last_err = e
        wait = 2 ** attempt
        print(f"  [retry {attempt + 1}/{retries}] 다운로드 실패({last_err}). {wait}s 후 재시도")
        time.sleep(wait)
    raise RuntimeError(f"데이터 다운로드 최종 실패: {last_err}")


def _extract_close(df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """auto_adjust=True 이므로 'Close'가 곧 조정종가."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        field = "Close" if "Close" in lvl0 else "Adj Close"
        out = df[field].copy()
    else:
        out = df[["Close"]].copy()
        out.columns = list(tickers)[:1]
    return out


def load_prices(
    tickers: Sequence[str],
    start: str = "1999-01-01",
    end: str | None = None,
    use_cache: bool = True,
    max_staleness_days: int = 7,
) -> pd.DataFrame:
    """조정종가 DataFrame 반환. 컬럼 = 티커, 인덱스 = 거래일."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"px_{'_'.join(sorted(tickers))}_{start}_{end or 'live'}.csv"
    cache = CACHE_DIR / key

    if use_cache and cache.exists():
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        if age_h < 12:
            px = pd.read_csv(cache, index_col=0, parse_dates=True)
            print(f"  캐시 사용: {cache.name} ({age_h:.1f}h old)")
            return _validate(px, tickers, max_staleness_days, live=end is None)

    raw = _download(tickers, start, end)
    px = _extract_close(raw, tickers)
    px = px.reindex(columns=list(tickers))
    px = px.ffill()
    px = px.dropna(how="all")
    px.to_csv(cache)
    return _validate(px, tickers, max_staleness_days, live=end is None)


def _validate(px: pd.DataFrame, tickers, max_staleness_days: int, live: bool) -> pd.DataFrame:
    missing = set(tickers) - set(px.columns)
    if missing:
        raise ValueError(f"다운로드 누락 티커: {missing}")

    for t in tickers:
        s = px[t].dropna()
        if len(s) < 250:
            raise ValueError(f"{t}: 데이터 {len(s)}행 -- 너무 짧음")
        # 비정상 점프 탐지 (미조정 데이터 섞임 감지)
        r = s.pct_change().dropna()
        bad = r[(r.abs() > 0.35)]
        if len(bad) > 0:
            print(f"  [warn] {t}: 일간 ±35% 초과 {len(bad)}건 -> {list(bad.index.date)[:3]}")

    if live:
        stale = (pd.Timestamp.utcnow().tz_localize(None) - px.index[-1]).days
        if stale > max_staleness_days:
            raise ValueError(f"데이터가 낡음: 최종 {px.index[-1].date()} ({stale}일 전)")

    print(f"  가격 데이터: {px.index[0].date()} ~ {px.index[-1].date()} ({len(px)}행)")
    return px


# ----------------------------------------------------------------------------
# 현금 수익률
# ----------------------------------------------------------------------------
def load_cash_rate(
    index: pd.DatetimeIndex,
    use_real: bool = True,
    flat: float = 0.02,
    spread: float = 0.0025,
    use_cache: bool = True,
) -> pd.Series:
    """일간 현금 수익률(단리 일할) Series 반환.

    use_real=True면 ^IRX(13주 T-Bill 할인율) 실제 시계열 사용.
    README의 'flat 2%' 가정은 2015~2021 저금리기를 과대평가하므로 기본 False 권장 안 함.
    """
    if not use_real:
        return pd.Series(flat / 252.0, index=index, name="cash")

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / "irx.csv"
        if use_cache and cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
            irx = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            raw = _download([CASH_PROXY], start="1990-01-01", end=None)
            irx = _extract_close(raw, [CASH_PROXY]).iloc[:, 0]
            irx.to_frame("irx").to_csv(cache)
        ann = (irx / 100.0).reindex(index).ffill().bfill() - spread
        ann = ann.clip(lower=0.0)
        return (ann / 252.0).rename("cash")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] ^IRX 로딩 실패({e}) -> flat {flat:.2%} 사용")
        return pd.Series(flat / 252.0, index=index, name="cash")


# ----------------------------------------------------------------------------
# 오프라인 테스트용 합성 데이터
# ----------------------------------------------------------------------------
def synthetic_prices(
    tickers: Sequence[str],
    n_days: int = 5400,
    seed: int = 7,
    start: str = "2004-01-02",
) -> pd.DataFrame:
    """네트워크 없이 엔진 검증용. 실제 자산과 유사한 드리프트/변동성/상관/레짐."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    mu = {"QQQ": 0.11 / 252, "TLT": 0.03 / 252, "GLD": 0.06 / 252}
    sd = {"QQQ": 0.21 / np.sqrt(252), "TLT": 0.14 / np.sqrt(252), "GLD": 0.16 / np.sqrt(252)}
    corr = np.array([[1.0, -0.30, 0.05], [-0.30, 1.0, 0.15], [0.05, 0.15, 1.0]])
    L = np.linalg.cholesky(corr)

    # 변동성 레짐 (약세장 구간 삽입)
    vol_mult = np.ones(n_days)
    for s, e, m in [(1000, 1400, 2.6), (2600, 2680, 3.2), (3800, 4050, 1.9)]:
        vol_mult[s:e] = m
    drift_mult = np.ones(n_days)
    for s, e, m in [(1000, 1400, -3.0), (2600, 2680, -9.0), (3800, 4050, -2.5)]:
        drift_mult[s:e] = m

    z = rng.standard_normal((n_days, 3)) @ L.T
    cols = {}
    for j, t in enumerate(tickers[:3]):
        r = mu[t] * drift_mult + sd[t] * vol_mult * z[:, j]
        cols[t] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)
