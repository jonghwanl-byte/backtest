import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest_v1():
    print("[Version 1] 절대 모멘텀 필터 백테스트 시작...")
    
    # 1. 파라미터 (단기채 SHY를 벤치마크 마크용으로 추가)
    tickers = ['QQQ', 'TLT', 'GLD', 'SHY']
    base_weights = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25, 'SHY': 0.00} 
    mas = [20, 120, 200]
    scalar_map = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.00}
    
    # 2. 데이터 다운로드
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2004-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
    data = data.dropna()
    returns = data.pct_change().dropna()

    # [핵심 로직 1] SHY(단기채)의 6개월(120일) 수익률 모멘텀 사전 계산
    shy_price = data['SHY']
    shy_mom = shy_price / shy_price.shift(120) - 1

    portfolio_return = pd.Series(0.0, index=returns.index)
    asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

    # 3. Hysteresis TAA 로직 적용
    for ticker in tickers:
        price = data[ticker]
        total_signals = pd.Series(0, index=price.index)
        
        for ma_period in mas:
            ma = price.rolling(window=ma_period).mean()
            state = np.zeros(len(price))
            p_vals = price.values
            m_vals = ma.values
            curr_state = 0
            
            for i in range(len(p_vals)):
                if np.isnan(m_vals[i]): continue
                if p_vals[i] > m_vals[i] * 1.03: curr_state = 1
                elif p_vals[i] < m_vals[i] * 0.97: curr_state = 0
                state[i] = curr_state
            total_signals += state
            
        # [핵심 로직 2] TLT에만 절대 모멘텀 필터 적용
        if ticker == 'TLT':
            tlt_mom = price / price.shift(120) - 1
            # TLT 모멘텀이 SHY(현금이자)보다 낮으면 점수를 0으로 강제 초기화
            filter_condition = tlt_mom < shy_mom
            total_signals = np.where(filter_condition, 0, total_signals)
            total_signals = pd.Series(total_signals, index=price.index)
            
        invested_fraction = total_signals.map(scalar_map)
        invested_fraction = invested_fraction.shift(1).fillna(0)
        
        actual_weight = invested_fraction * base_weights[ticker]
        asset_weights[ticker] = actual_weight
        portfolio_return += actual_weight * returns[ticker]

    # 4. Cash 수익 및 성과 계산
    total_invested = asset_weights.sum(axis=1)
    cash_weight = 1.0 - total_invested
    portfolio_return += cash_weight * (0.02 / 252)

    cum_returns = (1 + portfolio_return).cumprod()
    years = len(cum_returns) / 252.0
    cagr = cum_returns.iloc[-1] ** (1 / years) - 1
    ann_vol = portfolio_return.std() * np.sqrt(252)
    sharpe = (portfolio_return.mean() * 252 - 0.02) / ann_vol
    mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()

    print("\n" + "="*50)
    print("   [V1] TLT 절대 모멘텀 필터 적용 백테스트")
    print("="*50)
    print(f"▶ 연평균 수익 (CAGR) : {cagr*100:.2f}%")
    print(f"▶ 최대 낙폭 (MDD)    : {mdd*100:.2f}%")
    print(f"▶ 샤프 지수 (Sharpe) : {sharpe:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_backtest_v1()
