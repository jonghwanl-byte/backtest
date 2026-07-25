import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# [1. 전략 파라미터 설정]
# ==========================================
TICKERS_BASE = ['QQQ', 'TLT', 'GLD']
TICKER_ALT = 'DIA'
ALL_TICKERS = TICKERS_BASE + [TICKER_ALT]

# 기본 자산 목표 비중
BASE_WEIGHTS = {
    'QQQ': 0.50,
    'TLT': 0.25,
    'GLD': 0.25
}

# 이동평균선 기간
MA_WINDOWS = [20, 120, 200]

# 신호 강도 맵핑
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# 종목별 이격도 밴드 (매수 돌파 / 매도 이탈)
# DIA는 요청하신 대로 매수 +3% / 매도 -3% 룰을 적용합니다.
BANDS = {
    'QQQ': (1.025, 0.975),  # 기존 하이브리드 최적값
    'TLT': (1.030, 0.975),  # 기존 하이브리드 최적값
    'GLD': (1.025, 0.975),  # 기존 하이브리드 최적값
    'DIA': (1.030, 0.970)   # 신규 적용: 매수 +3% / 매도 -3%
}

# ==========================================
# [2. 데이터 다운로드]
# ==========================================
print("... 시장 데이터 다운로드 중 (QQQ, TLT, GLD, DIA) ...")
data = yf.download(ALL_TICKERS, start="2004-01-01", progress=False)

# MultiIndex 호환성 처리
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.get_level_values(0):
        prices_df = data['Adj Close'].ffill().dropna()
    else:
        prices_df = data['Close'].ffill().dropna()
else:
    prices_df = data.ffill().dropna()

# ==========================================
# [3. 이동평균 및 Hysteresis 밴드 계산]
# ==========================================
ma_lines = {}
upper_bands = {}
lower_bands = {}

for ticker in ALL_TICKERS:
    up_mult, dn_mult = BANDS[ticker]
    for window in MA_WINDOWS:
        ma_key = f"{ticker}_{window}"
        ma = prices_df[ticker].rolling(window=window).mean()
        ma_lines[ma_key] = ma
        upper_bands[ma_key] = ma * up_mult
        lower_bands[ma_key] = ma * dn_mult

# ==========================================
# [4. 시뮬레이션 및 신호 스칼라 추출]
# ==========================================
print("... Hysteresis 신호 처리 및 비중 계산 중 ...")

# 각 종목의 투입 강도(0, 0.5, 0.75, 1.0)를 저장할 DataFrame
scalars = pd.DataFrame(0.0, index=prices_df.index, columns=ALL_TICKERS)
current_states = {f"{ticker}_{window}": 0.0 for ticker in ALL_TICKERS for window in MA_WINDOWS}

start_idx = max(MA_WINDOWS)

# 시간순 루프 (과거 데이터를 훑으며 신호 판별)
for i in range(start_idx, len(prices_df)):
    today_scores = {}
    
    for ticker in ALL_TICKERS:
        score = 0
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            prev_state = current_states[ma_key]
            
            price = prices_df[ticker].iloc[i]
            upper = upper_bands[ma_key].iloc[i]
            lower = lower_bands[ma_key].iloc[i]
            
            if pd.isna(upper):
                new_state = 0.0
            elif prev_state == 1.0:
                new_state = 1.0 if price >= lower else 0.0
            else:
                new_state = 1.0 if price > upper else 0.0
                
            current_states[ma_key] = new_state
            score += int(new_state)
            
        today_scores[ticker] = score
        
    # 점수(0~3)를 투자 비중(Scalar)으로 변환하여 저장
    for ticker in ALL_TICKERS:
        scalars.loc[prices_df.index[i], ticker] = SCALAR_MAP[today_scores[ticker]]

# ==========================================
# [5. 포트폴리오 비중 (Weight) 할당 로직]
# ==========================================
weights = pd.DataFrame(0.0, index=prices_df.index, columns=ALL_TICKERS + ['CASH'])

# 1. Base 자산 비중 계산
weights['QQQ'] = BASE_WEIGHTS['QQQ'] * scalars['QQQ']
weights['TLT'] = BASE_WEIGHTS['TLT'] * scalars['TLT']
weights['GLD'] = BASE_WEIGHTS['GLD'] * scalars['GLD']

# 2. Base 자산에서 발생한 현금(Cash) 합산
base_cash = (BASE_WEIGHTS['QQQ'] * (1 - scalars['QQQ']) + 
             BASE_WEIGHTS['TLT'] * (1 - scalars['TLT']) + 
             BASE_WEIGHTS['GLD'] * (1 - scalars['GLD']))

# 3. DIA 투입 및 최종 현금 비중 확정 (Rule 1, 2, 3 적용)
weights['DIA'] = base_cash * scalars['DIA']
weights['CASH'] = base_cash * (1 - scalars['DIA'])

# ==========================================
# [6. 수익률 및 성과 지표(백테스트) 계산]
# ==========================================
# 일일 자산 수익률
daily_returns = prices_df.pct_change()
daily_returns['CASH'] = 0.0  # 현금 수익률 보수적으로 0% 가정

# Look-ahead bias 방지: 어제 종가 기준 확정된 비중으로 오늘의 수익을 얻음 (shift(1))
port_returns = (weights.shift(1) * daily_returns).sum(axis=1)

# 누적 수익률 및 낙폭(MDD) 계산
cum_returns = (1 + port_returns).cumprod()
roll_max = cum_returns.cummax()
drawdown = (cum_returns / roll_max) - 1.0

# 성과 지표 산출
days = len(port_returns)
years = days / 252

cagr = (cum_returns.iloc[-1] ** (1 / years)) - 1
mdd = drawdown.min()
vol = port_returns.std() * np.sqrt(252)
sharpe = cagr / vol if vol != 0 else 0

# 매매 횟수 (Turnover)
# 목표 비중(weights)이 어제와 0.1% 이상 달라졌을 때 리밸런싱 이벤트로 간주
weight_diff = weights.diff().abs().sum(axis=1)
rebalance_count = (weight_diff > 0.001).sum()

print("\n" + "="*50)
print(" 🚀 DIA-Alternative Hysteresis-TAA Backtest Result")
print("="*50)
print("목표 비중   : QQQ 50%, TLT 25%, GLD 25%")
print("대체 자산   : 현금 발생 시 DIA 우선 투입 (남은 비중만 현금화)")
print("신호 강도   : 1개=50%, 2개=75%, 3개=100%")
print(f"테스트 기간 : {prices_df.index[start_idx].strftime('%Y-%m-%d')} ~ {prices_df.index[-1].strftime('%Y-%m-%d')}\n")

print(f"▶ 연평균 수익 (CAGR) : {cagr*100:.2f}%")
print(f"▶ 최대 낙폭 (MDD)    : {mdd*100:.2f}%")
print(f"▶ 연평균 변동성      : {vol*100:.2f}%")
print(f"▶ 샤프 지수 (Sharpe) : {sharpe:.2f}\n")
print(f"▶ 총 리밸런싱 횟수   : {rebalance_count}회 (연평균 {rebalance_count/years:.1f}회)")
print("="*50)

# ==========================================
# [7. 차트 생성 및 저장]
# ==========================================
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(cum_returns.index, cum_returns * 100, label='Portfolio Cumulative Return', color='blue')
plt.yscale('log')
plt.title('DIA-Alternative Hysteresis-TAA Performance (Log Scale)')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(drawdown.index, drawdown * 100, label=f'Drawdown (MDD {mdd*100:.2f}%)', color='red', alpha=0.8)
plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.2)
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_dia_alternative.png')
print("\n[알림] 차트가 'backtest_dia_alternative.png' 파일로 저장되었습니다.")
