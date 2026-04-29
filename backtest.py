import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 다운로드 및 전처리
tickers = ['QQQ', 'TLT', 'GLD', 'XLE']
data = pd.DataFrame()

for ticker in tickers:
    # 2004년 말 GLD 상장 이후부터의 데이터를 맞추기 위해 2005년 시작
    df = yf.download(ticker, start='2005-01-01', progress=False)
    # yfinance 버전 호환성 처리
    if isinstance(df.columns, pd.MultiIndex):
        data[ticker] = df['Close'].iloc[:, 0]
    else:
        data[ticker] = df['Close']

data = data.dropna()
returns = data.pct_change().dropna()

# 2. 파라미터 및 로직 설정 (요청하신 비중 적용)
base_weights = {'QQQ': 0.40, 'TLT': 0.30, 'GLD': 0.20, 'XLE': 0.10}
mas = [20, 120, 200]

portfolio_return = pd.Series(0.0, index=returns.index)
asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

# Hysteresis 전략 시뮬레이션
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
            if np.isnan(m_vals[i]):
                continue
            # +3% / -3% Hysteresis Band
            if p_vals[i] > m_vals[i] * 1.03:
                curr_state = 1
            elif p_vals[i] < m_vals[i] * 0.97:
                curr_state = 0
            state[i] = curr_state
            
        total_signals += state
        
    # 신호 개수(0~3)에 따른 투자 비중 (0%, 33.3%, 66.6%, 100%)
    invested_fraction = total_signals / 3.0
    # Look-ahead bias(미래참조오류) 방지를 위해 하루 Shift (오늘 신호로 내일 수익률 적용)
    invested_fraction = invested_fraction.shift(1).fillna(0)
    
    # 최종 할당 비중 계산
    actual_weight = invested_fraction * base_weights[ticker]
    asset_weights[ticker] = actual_weight
    
    # 포트폴리오 수익률 합산
    portfolio_return += actual_weight * returns[ticker]

# 파킹(Cash) 비중에 대한 이자 수익 (보수적으로 연 2% 가정)
total_invested = asset_weights.sum(axis=1)
cash_weight = 1.0 - total_invested
cash_return = 0.02 / 252  
portfolio_return += cash_weight * cash_return

# 3. 성과 지표 계산
cum_returns = (1 + portfolio_return).cumprod()
years = len(cum_returns) / 252.0

cagr = cum_returns.iloc[-1] ** (1 / years) - 1
ann_ret = portfolio_return.mean() * 252
ann_vol = portfolio_return.std() * np.sqrt(252)
sharpe = (ann_ret - 0.02) / ann_vol  # 무위험수익률 2% 차감

roll_max = cum_returns.cummax()
drawdown = (cum_returns - roll_max) / roll_max
mdd = drawdown.min()

# 4. 결과 출력
print("=== Independent-Hysteresis-TAA Backtest Result ===")
print(f"포트폴리오 비중 : QQQ 40%, TLT 30%, GLD 20%, XLE 10%")
print(f"시뮬레이션 기간 : {cum_returns.index[0].date()} ~ {cum_returns.index[-1].date()}")
print("-" * 50)
print(f"CAGR         : {cagr*100:.2f}%")
print(f"MDD          : {mdd*100:.2f}%")
print(f"연 변동성    : {ann_vol*100:.2f}%")
print(f"Sharpe Ratio : {sharpe:.2f}")

# 그래프 출력 원할 경우
# plt.plot(cum_returns)
# plt.show()
