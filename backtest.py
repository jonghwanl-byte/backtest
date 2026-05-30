import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. 백테스트 기본 설정
# -----------------------------------
TICKERS = ['MSFT', 'NVDA', 'TSM'] # 대상 종목: 마이크로소프트, 엔비디아, TSMC
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# -----------------------------------
# 2. 종목별 모멘텀 백테스트 반복 실행
# -----------------------------------
for TICKER in TICKERS:
    print(f"\n{'='*50}")
    print(f"[{TICKER}] 데이터를 다운로드하고 백테스트를 진행합니다...")
    print(f"{'='*50}")
    
    # 데이터 다운로드
    raw_data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # yfinance 최신 버전의 다중 인덱스 구조 평탄화 및 1차원 강제 변환
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    data = pd.DataFrame()
    data['Close'] = raw_data['Close'].squeeze() 

    # 이동평균선 (5일, 20일, 60일) 계산
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()

    # 결과를 저장할 데이터프레임 생성 및 단순 보유(Buy & Hold) 수익률 기록
    results = pd.DataFrame(index=data.index)
    results['Buy & Hold'] = data['Close'] / data['Close'].iloc[0]
    daily_return = data['Close'].pct_change()

    # -----------------------------------
    # 3. 모멘텀 전략 매매 로직
    # -----------------------------------
    # 매수 조건: 종가가 이동평균선 '위'에 있을 때 상승 추세로 판단 (True=1, False=0)
    sig5 = (data['Close'] > data['MA5']).astype(int)
    sig20 = (data['Close'] > data['MA20']).astype(int)
    sig60 = (data['Close'] > data['MA60']).astype(int)
    
    # 당일 상승 추세를 만족하는 이평선의 총 개수 합산 (0 ~ 3개)
    total_signals = sig5 + sig20 + sig60
    
    # 신호 개수에 따른 투자 비중(Weight) 할당
    weights = pd.Series(0.0, index=data.index)
    weights[total_signals == 3] = 1.0   # 3개 이평선 모두 돌파 (강한 상승장): 100% 비중
    weights[total_signals == 2] = 0.75  # 2개 이평선 돌파: 75% 비중
    weights[total_signals == 1] = 0.50  # 1개 이평선 돌파: 50% 비중
    weights[total_signals == 0] = 0.0   # 모든 이평선 이탈 (하락장): 0% 비중 (전액 현금)
    
    # 전략 수익률 계산 (미래 참조 방지를 위해 익일 반영)
    strategy_return = weights.shift(1) * daily_return
    
    # 전략 누적 수익률 계산 및 저장
    cumulative_return = (1 + strategy_return.fillna(0)).cumprod()
    results['Momentum Strategy'] = cumulative_return

    # -----------------------------------
    # 4. 결과 시각화
    # -----------------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Buy & Hold'], label='Buy & Hold (단순 보유)', color='black', linewidth=2, linestyle='--')
    plt.plot(results.index, results['Momentum Strategy'], label='Momentum Strategy (동적 비중)', color='red', alpha=0.8, linewidth=2)

    plt.title(f"{TICKER} 다중 이동평균선 모멘텀 백테스트 결과 ({START_DATE} ~ {END_DATE})", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (누적 수익률)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

    # -----------------------------------
    # 5. 최종 수익률 요약 출력
    # -----------------------------------
    final_bh = (results['Buy & Hold'].iloc[-1] - 1) * 100
    final_mom = (results['Momentum Strategy'].iloc[-1] - 1) * 100
    
    print(f"\n[ {TICKER} 최종 누적 수익률 요약 ]")
    print(f"{'Buy & Hold (단순 보유)':<25}: {final_bh:>8.2f} %")
    print(f"{'Momentum Strategy':<25}: {final_mom:>8.2f} %")
