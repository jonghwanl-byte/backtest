import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. 백테스트 기본 설정
# -----------------------------------
TICKERS = ['MSFT', 'NVDA', 'TSM'] 
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
TRAILING_STOPS = [10.0, 15.0, 20.0] # 고점 대비 하락 청산 임계값 (%)

# -----------------------------------
# 2. 종목별 트레일링 스탑 백테스트
# -----------------------------------
for TICKER in TICKERS:
    print(f"\n{'='*50}")
    print(f"[{TICKER}] 데이터를 다운로드하고 백테스트를 진행합니다...")
    print(f"{'='*50}")
    
    # 데이터 다운로드 및 전처리
    raw_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
        
    data = pd.DataFrame()
    data['Close'] = raw_data['Close'].squeeze()
    daily_return = data['Close'].pct_change()
    
    # 결과 저장용 데이터프레임 (단순 보유 수익률 기록)
    results = pd.DataFrame(index=data.index)
    results['Buy & Hold'] = data['Close'] / data['Close'].iloc[0]
    
    # -----------------------------------
    # 3. 트레일링 스탑 로직 구현
    # -----------------------------------
    for ts in TRAILING_STOPS:
        positions = pd.Series(0.0, index=data.index)
        hwm = data['Close'].iloc[0] # High Water Mark (최고점 기록 변수)
        current_pos = 1.0 # 최초 100% 매수 상태로 시작
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            
            if current_pos == 1.0:
                # [주식 보유 중] 고점 갱신 확인
                if current_price > hwm:
                    hwm = current_price
                # [주식 보유 중] 리스크 관리: 고점 대비 ts% 하락 시 전량 매도
                elif current_price <= hwm * (1 - (ts / 100)):
                    current_pos = 0.0 
            else:
                # [현금 보유 중] 상승장 재진입: 주가가 이전 최고점을 다시 돌파할 때 매수
                if current_price > hwm:
                    current_pos = 1.0
                    hwm = current_price # 새로운 고점 갱신 시작
                    
            positions.iloc[i] = current_pos
            
        # 전략 수익률 계산 (미래 참조 방지를 위해 익일 반영)
        strategy_return = positions.shift(1) * daily_return
        results[f'Trailing Stop -{ts}%'] = (1 + strategy_return.fillna(0)).cumprod()
        
    # -----------------------------------
    # 4. 결과 시각화
    # -----------------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Buy & Hold'], label='Buy & Hold (단순 보유)', color='black', linewidth=2, linestyle='--')
    
    colors = ['blue', 'orange', 'red']
    for i, ts in enumerate(TRAILING_STOPS):
        plt.plot(results.index, results[f'Trailing Stop -{ts}%'], label=f'Trailing Stop -{ts}%', color=colors[i], alpha=0.8, linewidth=2)
        
    plt.title(f"{TICKER} 트레일링 스탑 백테스트 결과 ({START_DATE} ~ {END_DATE})", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (누적 수익률)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

    # -----------------------------------
    # 5. 최종 수익률 결과 출력
    # -----------------------------------
    print(f"\n[ {TICKER} 최종 누적 수익률 요약 ]")
    for col in results.columns:
        final_return = (results[col].iloc[-1] - 1) * 100
        print(f"{col:<25}: {final_return:>8.2f} %")
