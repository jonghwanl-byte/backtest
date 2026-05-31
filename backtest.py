import yfinance as yf
import pandas as pd
import numpy as np

# 테스트할 종목 리스트 (엔비디아, 테슬라)
tickers = ["NVDA", "TSLA"]

print("🚀 'Let profits run' 전략 백테스트를 시작합니다...\n" + "="*50)

for ticker in tickers:
    # 1. 데이터 다운로드 및 전처리 (최신 yfinance 호환)
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data = data.squeeze() if isinstance(data, pd.DataFrame) and data.shape[1] == 1 else data

    # 2. 보조 지표 계산
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    data['Vol20'] = data['Volume'].rolling(window=20).mean()
    data['High52w'] = data['Close'].rolling(window=252).max()
    
    data.dropna(inplace=True)

    # 3. 백테스트 변수 설정
    initial_capital = 100000.0
    cash = initial_capital
    shares = 0
    avg_buy_price = 0.0
    portfolio_value = []

    # 상태 추적 변수
    buy_stage = 0  # 0: 무포지션, 1: 1차(40%), 2: 2차(30%), 3: 3차(30%) 매수완료
    cooldown_days = 0 # 손절 후 재매수 금지 기간 (약 4주)

    # 4. 백테스트 메인 루프
    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        current_vol = data['Volume'].iloc[i]
        ma20 = data['MA20'].iloc[i]
        ma60 = data['MA60'].iloc[i]
        vol20 = data['Vol20'].iloc[i]
        high52 = data['High52w'].iloc[i]
        
        if cooldown_days > 0:
            cooldown_days -= 1
            
        profit_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0

        # ----------------------------------------------------
        # [매도 시점] - 익절 삭제! 오직 '추세 붕괴' 시에만 전량 매도
        # ----------------------------------------------------
        if shares > 0:
            # 60일선 하향 이탈 또는 고점 대비 -15% 손절 조건만 유지
            if current_price < ma60 or profit_pct <= -15:
                cash += shares * current_price
                shares = 0
                buy_stage = 0
                avg_buy_price = 0.0
                cooldown_days = 20 # 매도 후 4주간 냉각기
                
        # ----------------------------------------------------
        # [매수 시점] - 피라미딩(불타기) 구조 유지
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
                
            # 2차 매수 (불타기: +5~10% 상승 확인, 20일선 유지)
            elif buy_stage == 1 and 5 <= profit_pct <= 15 and current_price > ma20:
                invest_amount = cash * 0.50 # 남은 현금의 50%
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
        portfolio_value.append(cash + (shares * current_price))

    data['Portfolio_Value'] = portfolio_value
    total_return = ((data['Portfolio_Value'].iloc[-1] / initial_capital) - 1) * 100
    buy_and_hold_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100

    # 결과 출력
    print(f"[{ticker} 최종 누적 수익률 요약]")
    print(f"Buy & Hold (단순 보유) : {buy_and_hold_return:>8.2f} %")
    print(f"Let profits run 전략   : {total_return:>8.2f} %")
    print("-" * 50)
