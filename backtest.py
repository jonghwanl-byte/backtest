import yfinance as yf
import pandas as pd
import numpy as np
import sys

# ==========================================
# [1. 파라미터 및 단일 밴드 조합 설정]
# ==========================================
TICKERS = ['QQQ', 'TLT', 'GLD']
WEIGHTS = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}

# 현금 연이율 2.0% 설정 및 일일 수익률 환산 (복리 기준)
CASH_ANNUAL_RETURN = 0.020
DAILY_CASH_RETURN = (1 + CASH_ANNUAL_RETURN) ** (1 / 252.0) - 1

# 회원님께서 지정하신 자산별 밴드 룰
BANDS = {
    'QQQ': (1.020, 0.975),  # 매수 +2.0% / 매도 -2.5%
    'TLT': (1.030, 0.975),  # 매수 +3.0% / 매도 -2.5%
    'GLD': (1.025, 0.975)   # 매수 +2.5% / 매도 -2.5%
}

MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# ==========================================
# [2. 데이터 다운로드 (최대 기간)]
# ==========================================
print(">>> 야후 파이낸스 데이터 다운로드 중 (period='max')...")
data_full = yf.download(TICKERS, period="max", progress=False)

if data_full.empty:
    print("데이터 다운로드 실패")
    sys.exit(1)

# 다중 인덱스 처리 및 결측치 제거
if isinstance(data_full.columns, pd.MultiIndex):
    prices = data_full['Adj Close'].ffill().dropna() if 'Adj Close' in data_full.columns.get_level_values(0) else data_full['Close'].ffill().dropna()
else:
    prices = data_full['Adj Close'].ffill().dropna() if 'Adj Close' in data_full.columns else data_full['Close'].ffill().dropna()

# 자산별 내일의 일일 수익률 계산
daily_returns = prices.pct_change().shift(-1)

# ==========================================
# [3. 백테스트 시뮬레이션 연산]
# ==========================================
print(">>> 시그널 생성 및 포트폴리오 수익률 계산 중...")
weights_df = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)

for ticker in TICKERS:
    up_mult, dn_mult = BANDS[ticker]
    score_series = pd.Series(0, index=prices.index)
    
    for w in MA_WINDOWS:
        ma = prices[ticker].rolling(window=w).mean()
        upper = ma * up_mult
        lower = ma * dn_mult
        
        cond_up = prices[ticker] > upper
        cond_dn = prices[ticker] < lower
        
        state = pd.Series(np.nan, index=prices.index)
        state[cond_up] = 1.0
        state[cond_dn] = 0.0
        state = state.ffill().fillna(0.0)
        
        score_series += state
        
    scalar_series = score_series.map(SCALAR_MAP).fillna(0.0)
    
    # 일별 투자 비중(Weight) 할당
    weights_df[ticker] = scalar_series * WEIGHTS[ticker]

# 1. 포트폴리오에 투자된 총 비중 계산
total_invested_weight = weights_df.sum(axis=1)

# 2. 남은 현금 비중 계산
cash_weight = 1.0 - total_invested_weight

# 3. 자산 기여 수익률 + 현금 이자 수익 합산
port_returns = (weights_df * daily_returns).sum(axis=1) + (cash_weight * DAILY_CASH_RETURN)

# 워밍업 기간 및 마지막 결측치 행 제거
start_idx = MA_WINDOWS[-1]
port_returns = port_returns.iloc[start_idx:-1] 

# ==========================================
# [4. 지표 추출 및 출력]
# ==========================================
equity_curve = (1 + port_returns).cumprod()
years = len(equity_curve) / 252.0

cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
vol = port_returns.std() * np.sqrt(252)
sharpe = (cagr - 0.02) / vol if vol != 0 else 0  # 무위험 수익률 2% 가정
roll_max = equity_curve.cummax()
mdd = (equity_curve / roll_max - 1).min()

print("\n" + "="*50)
print("📊 [단일 포트폴리오 백테스트 결과]")
print("="*50)
print(f" - 자산 비중: QQQ 50% / TLT 25% / GLD 25%")
print(f" - QQQ 밴드 : +2.0% / -2.5%")
print(f" - TLT 밴드 : +3.0% / -2.5%")
print(f" - GLD 밴드 : +2.5% / -2.5%")
print(f" - 현금 이자: 연 2.0% 복리 반영")
print("-" * 50)
print(f" - CAGR (연평균 수익률): {cagr:.2%}")
print(f" - MDD  (최대 낙폭)  : {mdd:.2%}")
print(f" - Sharpe Ratio      : {sharpe:.3f}")
print("="*50)
