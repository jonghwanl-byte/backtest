import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ==========================================
# [1. 전략 파라미터 설정]
# ==========================================
TICKERS_BASE = ['QQQ', 'TLT', 'GLD']
TICKER_ALT = 'DIA'
ALL_TICKERS = TICKERS_BASE + [TICKER_ALT]

BASE_WEIGHTS = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}
MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

BANDS = {
    'QQQ': (1.025, 0.975),  # 매수 +2.5% / 매도 -2.5%
    'TLT': (1.030, 0.975),  # 매수 +3.0% / 매도 -2.5%
    'GLD': (1.025, 0.975),  # 매수 +2.5% / 매도 -2.5%
    'DIA': (1.030, 0.970)   # 매수 +3.0% / 매도 -3.0%
}

# ==========================================
# [2. 데이터 다운로드 및 전처리]
# ==========================================
print("... 최신 시장 데이터 다운로드 중 (QQQ, TLT, GLD, DIA) ...")
try:
    data = yf.download(ALL_TICKERS, start="2004-01-01", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prices_df = data['Adj Close'].ffill().dropna()
        else:
            prices_df = data['Close'].ffill().dropna()
    else:
        prices_df = data.ffill().dropna()
        
except Exception as e:
    print(f"데이터 다운로드 실패: {e}")
    sys.exit(1)

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
# [4. Hysteresis 로직 (신호 스칼라 추출)]
# ==========================================
print("... 신호 스칼라 판별 중 ...")
scalars = pd.DataFrame(0.0, index=prices_df.index, columns=ALL_TICKERS)
current_states = {f"{ticker}_{window}": 0.0 for ticker in ALL_TICKERS for window in MA_WINDOWS}

start_idx = max(MA_WINDOWS)

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
        
    for ticker in ALL_TICKERS:
        scalars.loc[prices_df.index[i], ticker] = SCALAR_MAP[today_scores[ticker]]

# ==========================================
# [5. 포트폴리오 모델 모델링 (Model A vs Model B)]
# ==========================================
# Model A: 오리지널 하이브리드 (나머지는 현금)
weights_A = pd.DataFrame(0.0, index=prices_df.index, columns=TICKERS_BASE + ['CASH'])
weights_A['QQQ'] = BASE_WEIGHTS['QQQ'] * scalars['QQQ']
weights_A['TLT'] = BASE_WEIGHTS['TLT'] * scalars['TLT']
weights_A['GLD'] = BASE_WEIGHTS['GLD'] * scalars['GLD']
weights_A['CASH'] = 1.0 - (weights_A['QQQ'] + weights_A['TLT'] + weights_A['GLD'])

# Model B: DIA 대체 투입 하이브리드
weights_B = pd.DataFrame(0.0, index=prices_df.index, columns=ALL_TICKERS + ['CASH'])
weights_B['QQQ'] = BASE_WEIGHTS['QQQ'] * scalars['QQQ']
weights_B['TLT'] = BASE_WEIGHTS['TLT'] * scalars['TLT']
weights_B['GLD'] = BASE_WEIGHTS['GLD'] * scalars['GLD']

base_cash = 1.0 - (weights_B['QQQ'] + weights_B['TLT'] + weights_B['GLD'])
weights_B['DIA'] = base_cash * scalars['DIA']
weights_B['CASH'] = base_cash * (1.0 - scalars['DIA'])

# ==========================================
# [6. 수익률 계산 및 성과 지표 산출]
# ==========================================
daily_returns = prices_df.pct_change().fillna(0)
daily_returns['CASH'] = 0.0  # 보수적으로 현금 이자 0% 가정

# 수익률 시계열 (Look-ahead 방지를 위해 Shift(1) 적용)
ret_A = (weights_A.shift(1) * daily_returns[weights_A.columns]).sum(axis=1).iloc[start_idx:]
ret_B = (weights_B.shift(1) * daily_returns[weights_B.columns]).sum(axis=1).iloc[start_idx:]

cum_A = (1 + ret_A).cumprod()
cum_B = (1 + ret_B).cumprod()

dd_A = (cum_A / cum_A.cummax()) - 1.0
dd_B = (cum_B / cum_B.cummax()) - 1.0

days = len(ret_A)
years = days / 252

# 모델 A 지표
cagr_A = (cum_A.iloc[-1] ** (1 / years)) - 1
mdd_A = dd_A.min()
vol_A = ret_A.std() * np.sqrt(252)
sharpe_A = cagr_A / vol_A if vol_A != 0 else 0
turnover_A = (weights_A.diff().abs().sum(axis=1) > 0.001).sum()

# 모델 B 지표
cagr_B = (cum_B.iloc[-1] ** (1 / years)) - 1
mdd_B = dd_B.min()
vol_B = ret_B.std() * np.sqrt(252)
sharpe_B = cagr_B / vol_B if vol_B != 0 else 0
turnover_B = (weights_B.diff().abs().sum(axis=1) > 0.001).sum()

# ==========================================
# [7. 결과 출력 및 차트 생성]
# ==========================================
print("\n" + "="*50)
print(" 🚀 A/B Model Comparison Backtest")
print("="*50)
print(f"테스트 기간: {ret_A.index[0].strftime('%Y-%m-%d')} ~ {ret_A.index[-1].strftime('%Y-%m-%d')}\n")

print("[Model A] 순정 하이브리드 (QQQ 50 / TLT 25 / GLD 25 -> 잔여 현금)")
print(f"▶ CAGR   : {cagr_A*100:.2f}%")
print(f"▶ MDD    : {mdd_A*100:.2f}%")
print(f"▶ Sharpe : {sharpe_A:.2f}")
print(f"▶ Turnover: {turnover_A}회\n")

print("[Model B] DIA 대체 투입 (잔여 현금 -> DIA 우선 투입)")
print(f"▶ CAGR   : {cagr_B*100:.2f}%")
print(f"▶ MDD    : {mdd_B*100:.2f}%")
print(f"▶ Sharpe : {sharpe_B:.2f}")
print(f"▶ Turnover: {turnover_B}회")
print("="*50)

# 차트 그리기
plt.figure(figsize=(14, 10))

# 1. 누적 수익률 비교
plt.subplot(2, 1, 1)
plt.plot(cum_A.index, cum_A * 100, label=f'Model A (Base) CAGR {cagr_A*100:.2f}%', color='blue')
plt.plot(cum_B.index, cum_B * 100, label=f'Model B (DIA Alt) CAGR {cagr_B*100:.2f}%', color='green')
plt.yscale('log')
plt.title('Cumulative Returns Comparison (Log Scale)')
plt.ylabel('Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. MDD 비교
plt.subplot(2, 1, 2)
plt.plot(dd_A.index, dd_A * 100, label=f'Model A MDD {mdd_A*100:.2f}%', color='blue', alpha=0.6)
plt.plot(dd_B.index, dd_B * 100, label=f'Model B MDD {mdd_B*100:.2f}%', color='green', alpha=0.6)
plt.fill_between(dd_A.index, dd_A * 100, 0, color='blue', alpha=0.1)
plt.fill_between(dd_B.index, dd_B * 100, 0, color='green', alpha=0.1)
plt.title('Drawdown Comparison')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compare_dia_vs_base.png')
print("\n[알림] 차트가 'compare_dia_vs_base.png' 파일로 저장되었습니다.")
