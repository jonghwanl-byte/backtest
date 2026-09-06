"""전수 시뮬레이션 엔진.

fastscan 근사와 달리 다음을 실제로 모사한다:
  - 주식 수 보유 (정수 주 옵션 -> 소액계좌 현실성)
  - 이동평균법 취득단가 추적 -> 실현손익 계산
  - 국내 거주자 해외주식 양도소득세 (연 250만원 공제 후 22%, 익년 5월 납부)
  - 신호 변경 시에만 매매 + 드리프트 허용치 (매일 리밸런싱 비용 과대계상 방지)
  - 체결 지연 (룩어헤드 차단)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExecConfig, StrategyConfig
from .signals import target_weights


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    cash: pd.Series
    tax_paid: pd.Series
    realized_gains: pd.Series
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def pretax_returns(self) -> pd.Series:
        """세금 납부일의 현금 유출을 되돌린 세전 수익률(비교용)."""
        adj = self.returns.copy()
        for d, amt in self.tax_paid[self.tax_paid > 0].items():
            if d in adj.index:
                v = self.equity.loc[:d].iloc[-2] if len(self.equity.loc[:d]) > 1 else self.equity.iloc[0]
                adj.loc[d] = adj.loc[d] + amt / max(v, 1e-9)
        return adj


def _tax_payment_dates(index: pd.DatetimeIndex, month: int) -> Dict[int, pd.Timestamp]:
    """각 연도별 납부일(해당 연도 `month`월 마지막 거래일)."""
    out = {}
    for y in sorted(set(index.year)):
        cand = index[(index.year == y) & (index.month == month)]
        if len(cand):
            out[y] = cand[-1]
    return out


def run_backtest(
    prices: pd.DataFrame,
    cash_daily: pd.Series,
    strat: StrategyConfig,
    ex: ExecConfig,
    warmup_from: pd.Timestamp | None = None,
    signal_prices: pd.DataFrame | None = None,   # ← 추가
) -> BacktestResult:
    """전체 히스토리로 신호를 워밍업한 뒤, warmup_from 이후만 성과 집계.

    signal_prices: 신호 계산용 가격을 체결 가격과 분리할 때 사용.
        None이면 prices를 그대로 쓴다(기존 동작).
        원화 백테스트에서 '신호는 달러, 체결은 원화'를 구현하는 데 쓴다.
    """
    tw = target_weights(
        prices if signal_prices is None else signal_prices,   # ← 변경
        strat.base_weights, strat.bands, strat.ma_windows, strat.scalar_map
    )
    # 체결 지연 반영: 신호 T -> 체결 T+exec_lag -> 수익 T+exec_lag+1 부터
    tw_eff = tw.shift(1 + ex.exec_lag).fillna(0.0)

    tickers = [t for t in prices.columns if strat.base_weights.get(t, 0) > 0]
    idx = prices.index
    px = prices[tickers].to_numpy(dtype=float)
    tgt = tw_eff[tickers].to_numpy(dtype=float)
    rf = cash_daily.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    cost = (ex.cost_bps + ex.slippage_bps) / 10_000.0
    n, m = px.shape

    shares = np.zeros(m)
    avg_cost = np.zeros(m)
    cash = float(ex.initial_capital)

    eq = np.zeros(n)
    cash_hist = np.zeros(n)
    w_hist = np.zeros((n, m))
    tax_hist = np.zeros(n)
    trades: List[dict] = []
    realized_by_year: Dict[int, float] = {}
    pay_dates = _tax_payment_dates(idx, ex.tax_payment_month)
    paid_for: set[int] = set()

    start_i = 0
    if warmup_from is not None:
        start_i = int(np.searchsorted(idx.to_numpy(), np.datetime64(warmup_from)))
        start_i = min(max(start_i, 0), n - 1)

    for i in range(n):
        # 1) 현금 이자
        cash *= 1.0 + rf[i]

        # 2) 세금 납부 (익년 5월)
        y = idx[i].year
        for py in list(realized_by_year.keys()):
            if py >= y or py in paid_for:
                continue
            d = pay_dates.get(py + 1)
            if d is not None and idx[i] == d and ex.apply_tax:
                gain = realized_by_year[py]
                tax = max(0.0, gain - ex.annual_deduction_usd) * ex.tax_rate
                cash -= tax
                tax_hist[i] += tax
                paid_for.add(py)

        value = cash + float(shares @ px[i])
        if value <= 0:
            eq[i:] = 0.0
            break

        # 3) 리밸런싱 판정 (백테스트 시작 이후에만 실제 매매)
        if i >= start_i:
            cur_w = shares * px[i] / value
            want = tgt[i]
            if ex.rebalance_mode == "daily":
                need = np.abs(cur_w - want).max() > 1e-6
            else:
                signal_changed = i > 0 and np.abs(tgt[i] - tgt[i - 1]).max() > 1e-9
                drifted = np.abs(cur_w - want).max() > ex.drift_tolerance
                need = signal_changed or drifted

            if need:
                for j in range(m):
                    pj = px[i, j]
                    target_val = want[j] * value
                    cur_val = shares[j] * pj
                    d_shares = (target_val - cur_val) / pj
                    if not ex.fractional_shares:
                        d_shares = np.trunc(d_shares)
                    if abs(d_shares * pj) < max(1.0, value * 1e-5):
                        continue
                    notional = d_shares * pj
                    fee = abs(notional) * cost
                    if d_shares < 0:
                        sell = min(-d_shares, shares[j])
                        if sell <= 0:
                            continue
                        gain = (pj - avg_cost[j]) * sell
                        realized_by_year[y] = realized_by_year.get(y, 0.0) + gain
                        shares[j] -= sell
                        cash += sell * pj - fee
                        if shares[j] <= 1e-12:
                            shares[j], avg_cost[j] = 0.0, 0.0
                        trades.append(dict(date=idx[i], ticker=tickers[j], side="SELL",
                                           shares=sell, price=pj, fee=fee, gain=gain))
                    else:
                        if d_shares * pj + fee > cash:
                            avail = max(cash, 0.0) / (1 + cost)
                            d_shares = avail / pj
                            if not ex.fractional_shares:
                                d_shares = np.trunc(d_shares)
                            if d_shares <= 0:
                                continue
                            fee = d_shares * pj * cost
                        tot = shares[j] + d_shares
                        avg_cost[j] = (avg_cost[j] * shares[j] + pj * d_shares) / tot
                        shares[j] = tot
                        cash -= d_shares * pj + fee
                        trades.append(dict(date=idx[i], ticker=tickers[j], side="BUY",
                                           shares=d_shares, price=pj, fee=fee, gain=0.0))
                value = cash + float(shares @ px[i])

        eq[i] = value
        cash_hist[i] = cash
        w_hist[i] = shares * px[i] / value

    equity = pd.Series(eq, index=idx, name="equity").iloc[start_i:]
    equity = equity / equity.iloc[0]
    rets = equity.pct_change().fillna(0.0).rename("strategy")

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf = tdf.set_index("date")
        tdf = tdf.loc[tdf.index >= idx[start_i]]

    years = len(equity) / 252
    notional = float(tdf["shares"].mul(tdf["price"]).sum()) if not tdf.empty else 0.0
    diag = {
        "n_trades": 0 if tdf.empty else len(tdf),
        "trades_per_year": (0 if tdf.empty else len(tdf)) / max(years, 1e-9),
        "annual_turnover": notional / max(equity.mean() * ex.initial_capital, 1e-9) / max(years, 1e-9),
        "total_fees": 0.0 if tdf.empty else float(tdf["fee"].sum()),
        "total_tax": float(tax_hist.sum()),
        "time_in_market": float((w_hist[start_i:].sum(axis=1) > 0.01).mean()),
        "avg_equity_exposure": float(w_hist[start_i:].sum(axis=1).mean()),
    }

    return BacktestResult(
        equity=equity,
        returns=rets,
        weights=pd.DataFrame(w_hist, index=idx, columns=tickers).iloc[start_i:],
        trades=tdf,
        cash=pd.Series(cash_hist, index=idx).iloc[start_i:],
        tax_paid=pd.Series(tax_hist, index=idx).iloc[start_i:],
        realized_gains=pd.Series(realized_by_year).sort_index(),
        diagnostics=diag,
    )


# ----------------------------------------------------------------------------
# 벤치마크
# ----------------------------------------------------------------------------
def buy_and_hold(prices: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    w = w / w.sum()
    r = prices.pct_change().fillna(0.0)
    eq = (1 + (r * w).sum(axis=1)).cumprod()
    return eq.pct_change().fillna(0.0).rename("B&H")


def simple_ma_filter(
    prices: pd.DataFrame,
    cash_daily: pd.Series,
    ticker: str = "QQQ",
    window: int = 200,
    exec_lag: int = 1,
    cost: float = 0.0010,
) -> pd.Series:
    """가장 무식한 벤치마크: 200일선 위면 100%, 아래면 현금.

    복잡한 15개 파라미터 시스템이 이걸 얼마나 이기는지가 진짜 부가가치다.
    """
    p = prices[ticker]
    ma = p.rolling(window, min_periods=window).mean()
    sig = (p > ma).astype(float).shift(1 + exec_lag).fillna(0.0)
    rf = cash_daily.reindex(p.index).fillna(0.0)
    r = p.pct_change().fillna(0.0)
    turn = sig.diff().abs().fillna(0.0)
    return (rf + sig * (r - rf) - cost * turn).rename(f"{ticker}_MA{window}")
