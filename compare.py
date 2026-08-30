#!/usr/bin/env python3
"""
주식 자산 비교 — QQQ vs QLD (TLT/GLD 공통)

확정 설정을 고정한 채 주식 자산만 바꿔 나란히 비교한다.
밴드가 QQQ 변동성에 맞춰 최적화된 값이므로, SPY 전용 밴드도 함께 탐색한다.

  기본 비중   주식 60 / TLT 20 / GLD 20
  밴드        +1.5% / -2.5%  (세 자산 공통)
  이동평균    20 / 120 / 200
  스케일링    100 / 75 / 50 / 0
  체결        익일 종가 (exec_lag=1)

사용:
  python compare.py              # QQQ vs SPY
  python compare.py SPY VTI IWM  # 임의 주식 자산들
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 확정 설정 =======================================================
EQUITIES = sys.argv[1:] if len(sys.argv) > 1 else ["QQQ", "QLD"]
SATELLITE = ["TLT", "GLD"]
BASE_W = {"EQ": 0.60, "TLT": 0.20, "GLD": 0.20}

MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

EXEC_LAG = 1
COST = 0.0010
CASH_RATE = 0.02
START = "2004-11-18"      # GLD 상장일
TD = 252

# 주식 전용 밴드 탐색 격자
UP_GRID = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]
DN_GRID = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]

PERIODS = {
    "2007-2009 금융위기": ("2007-10-09", "2009-06-01"),
    "2011 유럽위기": ("2011-04-30", "2012-06-01"),
    "2015-2016 횡보장": ("2015-07-17", "2017-02-17"),
    "2018 4분기 급락": ("2018-09-20", "2019-04-30"),
    "2020 코로나": ("2020-02-19", "2020-08-31"),
    "2021-2023 금리인상": ("2021-11-19", "2023-07-18"),
    "2023-2025 AI 랠리": ("2023-01-01", "2025-12-31"),
}
IS_OOS = {
    "표본외 초기 (2005-2014)": ("2005-01-01", "2014-12-31"),
    "최적화 구간 (2015-2024)": ("2015-01-01", "2024-12-31"),
    "표본외 최신 (2025-)": ("2025-01-01", "2030-12-31"),
}
# =======================================================================


def scalar(px: pd.Series, up: float, dn: float) -> pd.Series:
    """히스테리시스 상태 합계 -> 투입 스케일. 전체 히스토리로 워밍업."""
    score = pd.Series(0.0, index=px.index)
    for n in MA_PERIODS:
        ma = px.rolling(n, min_periods=n).mean()
        st = pd.Series(np.nan, index=px.index)
        st[px > ma * up] = 1.0
        st[px < ma * dn] = 0.0
        st = st.ffill().fillna(0.0)
        st[ma.isna()] = 0.0
        score += st
    return score.map(lambda s: SCALAR_MAP.get(int(s), 0.0)).astype(float)


def strategy(px: pd.DataFrame, eq: str, up=BAND_UP, dn=BAND_DN) -> pd.Series:
    rets = px.pct_change().fillna(0.0)
    rf = CASH_RATE / TD
    r = pd.Series(rf, index=px.index)
    for t, key in [(eq, "EQ"), ("TLT", "TLT"), ("GLD", "GLD")]:
        w = (BASE_W[key] * scalar(px[t], up, dn)).shift(1 + EXEC_LAG).fillna(0.0)
        r += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)
    return r


def static_mix(px: pd.DataFrame, eq: str) -> pd.Series:
    """정적 60/20/20, 매월 리밸런싱."""
    rets = px[[eq, "TLT", "GLD"]].pct_change().fillna(0.0)
    w = pd.Series([BASE_W["EQ"], BASE_W["TLT"], BASE_W["GLD"]],
                  index=[eq, "TLT", "GLD"])
    out = pd.Series(0.0, index=px.index)
    for _, idx in rets.groupby(rets.index.to_period("M")).groups.items():
        seg = rets.loc[idx]
        e = (w * (1 + seg).cumprod()).sum(axis=1)
        out.loc[idx] = e.pct_change().fillna(e.iloc[0] - 1)
    return out


def ma_filter(px: pd.DataFrame, eq: str, n=200) -> pd.Series:
    """200일선 단순 필터 (위면 100%, 아래면 현금)."""
    p = px[eq]
    ma = p.rolling(n, min_periods=n).mean()
    sig = (p > ma).astype(float).shift(1 + EXEC_LAG).fillna(0.0)
    rf = CASH_RATE / TD
    return rf + sig * (p.pct_change().fillna(0.0) - rf) \
        - COST * sig.diff().abs().fillna(0.0)


def stats(r: pd.Series) -> dict:
    e = (1 + r).cumprod()
    dd = e / e.cummax() - 1
    yrs = len(r) / TD
    ex = r - CASH_RATE / TD
    ann = (1 + r).groupby(r.index.year).prod() - 1
    down = np.where(ex < 0, ex, 0.0)
    dstd = np.sqrt((down ** 2).sum() / (len(r) - 1)) * np.sqrt(TD)
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(TD)
    return {"CAGR": cagr, "Vol": vol, "MDD": dd.min(),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(TD),
            "Sortino": ex.mean() * TD / dstd if dstd > 0 else 0.0,
            "Calmar": cagr / -dd.min(), "MDD/Vol": -dd.min() / vol,
            "최악연도": ann.min(), "손실연수": int((ann < 0).sum())}


def seg(r: pd.Series, s0: str, e0: str):
    m = np.asarray((r.index >= pd.Timestamp(s0)) & (r.index <= pd.Timestamp(e0)))
    if m.sum() < 30:
        return None, None
    s = r[m]
    e = (1 + s).cumprod()
    return float((1 + s).prod() - 1), float((e / e.cummax() - 1).min())


def main():
    tickers = sorted(set(EQUITIES) | set(SATELLITE))
    px = yf.download(tickers, start="1990-01-01", auto_adjust=True,
                     progress=False, threads=False)["Close"]
    px = px[tickers].ffill().dropna()
    px = px[px.index >= pd.Timestamp("1999-01-01")]
    m = np.asarray(px.index >= pd.Timestamp(START))

    S, B, F = {}, {}, {}
    for eq in EQUITIES:
        S[eq] = strategy(px, eq)[m]
        B[eq] = static_mix(px, eq)[m]
        F[eq] = ma_filter(px, eq)[m]
    idx = S[EQUITIES[0]].index

    print(f"평가구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)/TD:.1f}년)")
    print(f"설정: 주식60/TLT20/GLD20  밴드+1.5%/-2.5%  MA20/120/200  "
          f"스케일100/75/50/0\n")

    # ---------- 1. 동적 전략 비교 ----------
    print("=" * 78)
    print("■ 1. 동적 전략 (주식 자산별)\n")
    print(f"  {'주식':<7}{'CAGR':>8}{'Vol':>8}{'MDD':>9}{'Sharpe':>9}"
          f"{'Sortino':>9}{'Calmar':>8}{'MDD/Vol':>9}{'최악연도':>9}")
    print("  " + "─" * 74)
    for eq in EQUITIES:
        s = stats(S[eq])
        print(f"  {eq:<7}{s['CAGR']:>8.2%}{s['Vol']:>8.2%}{s['MDD']:>9.2%}"
              f"{s['Sharpe']:>9.3f}{s['Sortino']:>9.3f}{s['Calmar']:>8.2f}"
              f"{s['MDD/Vol']:>9.2f}{s['최악연도']:>9.1%}")

    # ---------- 2. 벤치마크 대비 ----------
    print("\n" + "=" * 78)
    print("■ 2. 벤치마크 대비 (같은 주식 자산 기준)\n")
    for eq in EQUITIES:
        bh = px[eq].pct_change().fillna(0.0)[m]
        print(f"  [{eq}]")
        print(f"    {'전략':<18}{'CAGR':>9}{'MDD':>10}{'Sharpe':>9}{'MDD/Vol':>9}")
        print("    " + "─" * 55)
        for label, r in [("동적 TAA", S[eq]), ("정적 60/20/20", B[eq]),
                         (f"{eq} 200MA 필터", F[eq]), (f"{eq} 매수보유", bh)]:
            s = stats(r)
            print(f"    {label:<18}{s['CAGR']:>9.2%}{s['MDD']:>10.2%}"
                  f"{s['Sharpe']:>9.3f}{s['MDD/Vol']:>9.2f}")
        print()

    # ---------- 3. 구간별 ----------
    print("=" * 78)
    print("■ 3. 구간별 수익률 (괄호는 구간 MDD)\n")
    print(f"  {'구간':<22}" + "".join(f"{eq:>20}" for eq in EQUITIES))
    print("  " + "─" * (22 + 20 * len(EQUITIES)))
    for name, (s0, e0) in PERIODS.items():
        cells = []
        for eq in EQUITIES:
            v, d = seg(S[eq], s0, e0)
            cells.append(f"{v:>10.1%} ({d:>6.1%})" if v is not None else f"{'-':>20}")
        print(f"  {name:<22}" + "".join(f"{c:>20}" for c in cells))

    # ---------- 4. 표본 내외 ----------
    print("\n" + "=" * 78)
    print("■ 4. 표본 내 / 표본 외\n")
    print(f"  {'구간':<24}" + "".join(f"{eq + ' CAGR':>12}{eq + ' SR':>10}"
                                       for eq in EQUITIES))
    print("  " + "─" * (24 + 22 * len(EQUITIES)))
    for name, (s0, e0) in IS_OOS.items():
        cells = []
        for eq in EQUITIES:
            mm = np.asarray((idx >= pd.Timestamp(s0)) & (idx <= pd.Timestamp(e0)))
            if mm.sum() < 60:
                cells.append(f"{'-':>12}{'-':>10}")
                continue
            s = stats(S[eq][mm])
            cells.append(f"{s['CAGR']:>12.2%}{s['Sharpe']:>10.3f}")
        print(f"  {name:<24}" + "".join(cells))

    # ---------- 5. 주식 전용 밴드 탐색 ----------
    print("\n" + "=" * 78)
    print("■ 5. 주식 전용 밴드 탐색 (TLT/GLD는 ±1.5/2.5 고정)\n")
    print("  밴드가 QQQ 변동성에 맞춰진 값이므로, 다른 자산엔 최적이 아닐 수 있다.\n")
    print(f"  {'주식':<7}{'현재밴드SR':>12}{'최적밴드':>16}{'최적SR':>10}"
          f"{'중앙값SR':>10}{'최대-중앙':>11}{'판정':>10}")
    print("  " + "─" * 76)

    rets = px.pct_change().fillna(0.0)
    rf = CASH_RATE / TD
    sat = pd.Series(rf, index=px.index)
    for t, key in [("TLT", "TLT"), ("GLD", "GLD")]:
        w = (BASE_W[key] * scalar(px[t], BAND_UP, BAND_DN)).shift(1 + EXEC_LAG).fillna(0.0)
        sat += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)

    for eq in EQUITIES:
        rows = []
        for u in UP_GRID:
            for d in DN_GRID:
                w = (BASE_W["EQ"] * scalar(px[eq], 1 + u, 1 - d)).shift(1 + EXEC_LAG).fillna(0.0)
                r = (sat + w * (rets[eq] - rf) - COST * w.diff().abs().fillna(0.0))[m]
                rows.append({"up": u, "dn": d, "SR": stats(r)["Sharpe"]})
        g = pd.DataFrame(rows)
        cur = g[(g.up == 0.015) & (g.dn == 0.025)].SR.iloc[0]
        best = g.loc[g.SR.idxmax()]
        med = g.SR.median()
        gap = best.SR - med
        verdict = "고원" if gap < 0.08 else ("완만" if gap < 0.15 else "첨탑")
        print(f"  {eq:<7}{cur:>12.3f}"
              f"{f'+{best.up:.1%}/-{best.dn:.1%}':>16}{best.SR:>10.3f}"
              f"{med:>10.3f}{gap:>11.3f}{verdict:>10}")

    # ---------- 6. 연도별 ----------
    print("\n" + "=" * 78)
    print("■ 6. 연도별 수익률\n")
    print(f"  {'연도':<7}" + "".join(f"{eq:>10}" for eq in EQUITIES)
          + "".join(f"{eq + 'B&H':>11}" for eq in EQUITIES))
    print("  " + "─" * (7 + 10 * len(EQUITIES) + 11 * len(EQUITIES)))
    for y in sorted(set(idx.year)):
        cells = []
        for eq in EQUITIES:
            r = S[eq]
            cells.append(f"{(1 + r[r.index.year == y]).prod() - 1:>10.1%}")
        for eq in EQUITIES:
            b = px[eq].pct_change().fillna(0.0)[m]
            cells.append(f"{(1 + b[b.index.year == y]).prod() - 1:>11.1%}")
        print(f"  {y:<7}" + "".join(cells))

    # ---------- 7. 상관 ----------
    print("\n" + "=" * 78)
    print("■ 7. 자산 상관 (일간수익률, 평가구간)\n")
    for eq in EQUITIES:
        sub = rets[[eq, "TLT", "GLD"]][m]
        c = sub.corr()
        pairs = [(a, b, c.loc[a, b]) for a, b in combinations(sub.columns, 2)]
        avg = np.mean([p[2] for p in pairs])
        detail = "  ".join(f"{a}-{b} {v:+.2f}" for a, b, v in pairs)
        print(f"  [{eq}]  평균 {avg:+.3f}   {detail}")

    print("\n" + "=" * 78)
    print("■ 판단 가이드")
    print("  · MDD/Vol 이 낮을수록 같은 변동성에서 낙폭을 잘 막은 것")
    print("  · 표본외(2005-2014) 샤프가 표본내의 절반 이상이면 견고")
    print("  · 5번에서 최적밴드가 현재와 크게 다르면, 그 자산엔 재최적화가 필요하다는 뜻")
    print("    (다만 재최적화는 과최적화 위험을 키운다 — 고원 판정인지 확인할 것)")


if __name__ == "__main__":
    main()
