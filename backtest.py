import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 다운로드 및 전처리 (yfinance 호환성 반영)
ticker = "NVDA"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
data = data.squeeze() if isinstance(data, pd.DataFrame) and data.shape[1] == 1 else data

# 2. 보조 지표 계산
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA60'] = data['Close'].rolling(window=60).mean()
data['Vol20'] = data['Volume'].rolling(window=20).mean()
data['High52w'] = data['Close'].rolling(window=252).max()

# RSI 계산
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
data['RSI'] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)

# 3. 백테스트 변수 설정
initial_capital = 100000.0
cash = initial_capital
shares = 0
avg_buy_price = 0.0
portfolio_value = []

# 상태 추적 변수
buy_stage = 0  # 0: 무포지션, 1: 1차 매수완료, 2: 2차 완료, 3: 3차 완료
sell_stage = 0 # 익절 단계 추적
cooldown_days = 0 # 재매수 금지 기간

# 4. 백테스트 메인 루프 (한눈에 보는 흐름)
for i in range(len(data)):
    current_price = data['Close'].iloc[i]
    current_vol = data['Volume'].iloc[i]
    ma20 = data['MA20'].iloc[i]
    ma60 = data['MA60'].iloc[i]
    vol20 = data['Vol20'].iloc[i]
    high52 = data['High52w'].iloc[i]
    rsi = data['RSI'].iloc[i]
    
    # 냉각기 감소
    if cooldown_days > 0:
        cooldown_days -= 1
        
    profit_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0

    # ----------------------------------------------------
    # [매도 및 손절 시점] - 리스크 관리가 최우선이므로 먼저 체크
    # ----------------------------------------------------
    if shares > 0:
        # 손절 및 추세 종료 (3차 매도 조건 & 기계적 손절)
        if current_price < ma60 or profit_pct <= -15:
            cash += shares * current_price
            shares = 0
            buy_stage = 0
            sell_stage = 0
            avg_buy_price = 0.0
            cooldown_days = 20 # 약 4주간 재매수 금지 (핵심 조건 적용)
            
        # 1차 익절 (+15~20% 구간, RSI 70 이상)
        elif profit_pct >= 15 and sell_stage == 0 and rsi >= 70:
            sell_shares = shares * 0.30
            cash += sell_shares * current_price
            shares -= sell_shares
            sell_stage = 1
            
        # 2차 익절 (+30~40% 구간, 20일선 이탈 시)
        elif profit_pct >= 30 and sell_stage == 1 and current_price < ma20:
            sell_shares = shares * 0.57 # 남은 물량 중 약 40% (전체 기준)
            cash += sell_shares * current_price
            shares -= sell_shares
            sell_stage = 2

    # ----------------------------------------------------
    # [매수 및 불타기 시점] - 냉각기가 끝난 후 진입
    # ----------------------------------------------------
    if cooldown_days == 0:
        # 1차 매수 (신고가 근접, 거래량 폭증, 20일선 위)
        if buy_stage == 0 and current_price >= high52 * 0.95 and current_vol >= vol20 * 1.5 and current_price > ma20:
            invest_amount = cash * 0.40
            new_shares = invest_amount / current_price
            shares += new_shares
            cash -= invest_amount
            avg_buy_price = current_price
            buy_stage = 1
            
        # 2차 매수 (+5~10% 상승 확인, 20일선 유지)
        elif buy_stage == 1 and 5 <= profit_pct <= 15 and current_price > ma20:
            invest_amount = cash * 0.50 # 남은 현금의 50% (전체 자금의 약 30%)
            new_shares = invest_amount / current_price
            avg_buy_price = ((shares * avg_buy_price) + (new_shares * current_price)) / (shares + new_shares)
            shares += new_shares
            cash -= invest_amount
            buy_stage = 2
            
        # 3차 매수 (눌림목: 20일선 부근 지지)
        elif buy_stage == 2 and (current_price <= ma20 * 1.03 and current_price >= ma20 * 0.98) and current_vol < vol20:
            invest_amount = cash # 남은 현금 전액 투입
            new_shares = invest_amount / current_price
            avg_buy_price = ((shares * avg_buy_price) + (new_shares * current_price)) / (shares + new_shares)
            shares += new_shares
            cash -= invest_amount
            buy_stage = 3

    # 매일 포트폴리오 가치 기록
    current_value = cash + (shares * current_price)
    portfolio_value.append(current_value)

data['Portfolio_Value'] = portfolio_value

# 5. 결과 계산 및 출력
total_return = ((data['Portfolio_Value'].iloc[-1] / initial_capital) - 1) * 100
buy_and_hold_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100

print(f"[{ticker} 분할 매매 최종 수익률 요약]")
print(f"Buy & Hold (단순 보유)       : {buy_and_hold_return:>8.2f} %")
print(f"단계별 분할 매매 전략        : {total_return:>8.2f} %")
