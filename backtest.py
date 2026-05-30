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

# -----------------------------------
# 2. 종목별 거래량 모멘텀 백테스트 실행
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
    data['Volume'] = raw_data['Volume'].squeeze()

    # 가격 및 거래량 이동평균선 계산
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    data['Vol_MA20'] = data['Volume'].rolling(window=20).mean()

    results = pd.DataFrame(index=data.index)
    results['Buy & Hold'] = data['Close'] / data['Close'].iloc[0]
    daily_return = data['Close'].pct_change()

    # -----------------------------------
    # 3. 수정된 거래량 동반 매수 신호 판별 함수
    # -----------------------------------
    def get_flexible_volume_breakout(price, ma, volume, vol_ma):
        # 1) 상향 돌파: 어제는 이평선 이하, 오늘은 이평선 초과
        cross_up = (price > ma) & (price.shift(1) <= ma.shift(1))
        
        # 2) 거래량 조건 완화: 당일 거래량이 1.2배(20%) 이상 터졌는지 확인
        vol_condition = volume > (vol_ma * 1.2)
        
        # 3) 유예 기간 부여: 최근 3일 이내에 거래량 조건이 한 번이라도 만족되었는지 확인 (Window 적용)
        vol_surge_window = vol_condition.rolling(window=3).max() == 1
        
        # 4) 진짜 돌파 조건 확립 (NaN 값 처리 포함)
        valid_breakout = cross_up & vol_surge_window.fillna(False)
        
        # 5) 하향 이탈: 주가가 다시 이평선 아래로 떨어짐
        break_down = price < ma
        
        # 6) 포지션 상태 저장
        sig = pd.Series(np.nan, index=price.index)
        sig.loc[valid_breakout] = 1.0
        sig.loc[break_down] = 0.0
        return sig.ffill().fillna(0.0)

    # 각 이평선에 대해 완화된 거래량 동반 신호 계산
    sig5 = get_flexible_volume_breakout(data['Close'], data['MA5'], data['Volume'], data['Vol_MA20'])
    sig20 = get_flexible_volume_breakout(data['Close'], data['MA20'], data['Volume'], data['Vol_MA20'])
    sig60 = get_flexible_volume_breakout(data['Close'], data['MA60'], data['Volume'], data['Vol_MA20'])
    
    total_signals = sig5 + sig20 + sig60
    
    # 신호 개수에 따른 비중 할당
    weights = pd.Series(0.0, index=data.index)
    weights[total_signals == 3] = 1.0   
    weights[total_signals == 2] = 0.75  
    weights[total_signals == 1] = 0.50  
    weights[total_signals == 0] = 0.0   
    
    # 전략 누적 수익률 계산
    strategy_return = weights.shift(1) * daily_return
    results['Flex Volume Momentum'] = (1 + strategy_return.fillna(0)).cumprod()

    # -----------------------------------
    # 4. 최종 결과 출력 및 시각화
    # -----------------------------------
    final_bh = (results['Buy & Hold'].iloc[-1] - 1) * 100
    final_mom = (results['Flex Volume Momentum'].iloc[-1] - 1) * 100
    
    print(f"\n[ {TICKER} 최종 누적 수익률 요약 ]")
    print(f"{'Buy & Hold (단순 보유)':<25}: {final_bh:>8.2f} %")
    print(f"{'Flex Volume Momentum':<25}: {final_mom:>8.2f} %")

    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Buy & Hold'], label='Buy & Hold', color='black', linewidth=2, linestyle='--')
    plt.plot(results.index, results['Flex Volume Momentum'], label='Flexible Volume Momentum', color='blue', alpha=0.8, linewidth=2)
    plt.title(f"{TICKER} 거래량 필터 완화 백테스트 결과 ({START_DATE} ~ {END_DATE})", fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()
