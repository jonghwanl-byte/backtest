import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. 파라미터 설정 (궁극의 하이브리드 모델 기준)
tickers = ['QQQ', 'TLT', 'GLD']
base_weights = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}
ma_windows = [20, 120, 200]
scalar_map = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# 하이브리드 비대칭 밴드 설정
bands = {
    'QQQ': {'upper': 0.025, 'lower': 0.025},
    'TLT': {'upper': 0.030, 'lower': 0.025},
    'GLD': {'upper': 0.025, 'lower': 0.025}
}

# 2. 데이터 다운로드 (Adj Close 사용, MultiIndex 방어)
print("데이터를 다운로드하는 중입니다...")
data = yf.download(tickers, start="2004-01-01", end="2026-12-31", progress=False)

if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.levels[0]:
        df = data['Adj Close'].copy()
    else:
        df = data['Close'].copy()
else:
    df = data.copy()

df = df[tickers].dropna()

# 3. 1일 지연 확인형(1-Day Confirmation) 이격도 로직 계산
ma_states = pd.DataFrame(index=df.index)
daily_scores = pd.DataFrame(0, index=df.index, columns=tickers)

for ticker in tickers:
    score_sum = pd.Series(0, index=df.index)
    
    upper_band_pct = bands[ticker]['upper']
    lower_band_pct = bands[ticker]['lower']
    
    for w in ma_windows:
        ma = df[ticker].rolling(window=w).mean()
        upper_line = ma * (1 + upper_band_pct)
        lower_line = ma * (1 - lower_band_pct)
        
        # [핵심 로직 변경] 당일의 이격도 돌파/이탈 상태(Condition) 확인
        # 1: 상단 돌파, -1: 하단 이탈, 0: 밴드 내 위치
        condition = pd.Series(0, index=df.index)
        condition[df[ticker] > upper_line] = 1
        condition[df[ticker] < lower_line] = -1
        
        # 어제와 오늘의 Condition 확인 (Shift 사용)
        yesterday_condition = condition.shift(1)
        
        # 상태(State) 기록 로직
        state = pd.Series(np.nan, index=df.index)
        
        # 매수(1.0): 어제도 뚫었고, 오늘도 연속으로 뚫었을 때만
        buy_signal = (yesterday_condition == 1) & (condition == 1)
        state[buy_signal] = 1.0
        
        # 매도(0.0): 어제도 이탈했고, 오늘도 연속으로 이탈했을 때만
        sell_signal = (yesterday_condition == -1) & (condition == -1)
        state[sell_signal] = 0.0
        
        # 조건이 충족되지 않은 날은 이전 신호(state)를 그대로 유지 (ffill)
        state = state.ffill().fillna(0.0)
        
        score_sum += state
        
    daily_scores[ticker] = score_sum

# 4. 포트폴리오 비중 및 수익률 계산
scalars = daily_scores.replace(scalar_map)

target_weights = scalars.copy()
for ticker in tickers:
    target_weights[ticker] = scalars[ticker] * base_weights[ticker]

# 실제 매매는 신호 발생 다음 날 종가로 이루어지므로 shift(1) 적용
actual_weights = target_weights.shift(1).fillna(0)
actual_weights['Cash'] = 1.0 - actual_weights[tickers].sum(axis=1)

daily_returns = df.pct_change().fillna(0)
daily_returns['Cash'] = 0.0

port_returns = (actual_weights * daily_returns).sum(axis=1)
cumulative_returns = (1 + port_returns).cumprod()

# 5. 성과 지표 계산
total_years = len(df) / 252
cagr = cumulative_returns.iloc[-1] ** (1 / total_years) - 1

rolling_max = cumulative_returns.cummax()
drawdown = cumulative_returns / rolling_max - 1
mdd = drawdown.min()

volatility = port_returns.std() * np.sqrt(252)
sharpe_ratio = (cagr - 0.02) / volatility if volatility != 0 else 0

# 6. 턴오버(거래 횟수) 계산
weight_diff = actual_weights[tickers].diff().fillna(0)
turnover_events = (weight_diff != 0).sum()
total_turnover = turnover_events.sum()
annual_turnover = total_turnover / total_years

# 7. 결과 출력
print("="*50)
print(" 1-Day Confirmation Hysteresis-TAA Backtest Result ")
print("="*50)
print(f"목표 비중    : QQQ {base_weights['QQQ']:.0%}, TLT {base_weights['TLT']:.0%}, GLD {base_weights['GLD']:.0%}")
print(f"하이브리드 밴드: QQQ ±2.5%, TLT +3.0%/-2.5%, GLD ±2.5%")
print(f"테스트 기간  : {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n")

print(f"▶ 연평균 수익 (CAGR) : {cagr:.2%}")
print(f"▶ 최대 낙폭 (MDD)    : {mdd:.2%}")
print(f"▶ 연평균 변동성      : {volatility:.2%}")
print(f"▶ 샤프 지수 (Sharpe) : {sharpe_ratio:.2f}\n")

print(f"▶ 총 리밸런싱 횟수   : {total_turnover}회 (연평균 {annual_turnover:.1f}회)")
print(f"   [상세] QQQ: {turnover_events['QQQ']}회 | TLT: {turnover_events['TLT']}회 | GLD: {turnover_events['GLD']}회")
print("="*50)

# 8. 그래프 저장
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='1-Day Confirmation TAA')
plt.title('1-Day Confirmation Hysteresis TAA Cumulative Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('backtest_result.png')
print("\n[알림] 그래프가 'backtest_result.png'로 저장되었습니다.")
