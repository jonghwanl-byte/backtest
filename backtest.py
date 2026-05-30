import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. 백테스트 기본 설정
# -----------------------------------
TICKER = 'MSFT'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
THRESHOLDS = [3.0, 2.5, 2.0, 1.5, 1.0] # 이격도 센서 작동 기준치 (%)

# -----------------------------------
# 2. 데이터 다운로드 및 전처리 (에러 해결 핵심 구간)
# -----------------------------------
print(f"[{TICKER}] 데이터를 다운로드하는 중...")
raw_data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# 🛠️ 에러 해결: yfinance 최신 버전의 다중 인덱스 구조 평탄화
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

# 'Close' 데이터를 명확하게 1차원 Series로 강제 변환하여 연산(Align) 오류 방지
data = pd.DataFrame()
data['Close'] = raw_data['Close'].squeeze() 

# -----------------------------------
# 3. 이동평균선 계산
# -----------------------------------
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA60'] = data['Close'].rolling(window=60).mean()

# 결과를 저장할 데이터프레임 생성 및 단순 보유(Buy & Hold) 수익률 기록
results = pd.DataFrame(index=data.index)
results['Buy & Hold'] = data['Close'] / data['Close'].iloc[0]
daily_return = data['Close'].pct_change()

# -----------------------------------
# 4. 이격도 임계값별 백테스트 실행
# -----------------------------------
for t in THRESHOLDS:
    # 3개 이평선 각각에 대한 이격도 매수 신호 발생 여부 확인 (True=1, False=0)
    sig5 = (data['Close'] <= data['MA5'] * (1 - (t / 100))).astype(int)
    sig20 = (data['Close'] <= data['MA20'] * (1 - (t / 100))).astype(int)
    sig60 = (data['Close'] <= data['MA60'] * (1 - (t / 100))).astype(int)
    
    # 당일 켜진 센서(신호)의 총 개수 합산 (0 ~ 3개)
    total_signals = sig5 + sig20 + sig60
    
    # 신호 개수에 따른 투자 비중(Weight) 할당
    weights = pd.Series(0.0, index=data.index)
    weights[total_signals == 3] = 1.0   # 3개 만족: 100% 비중
    weights[total_signals == 2] = 0.75  # 2개 만족: 75% 비중
    weights[total_signals == 1] = 0.50  # 1개 만족: 50% 비중
    weights[total_signals == 0] = 0.0   # 0개 만족: 0% 비중 (전액 현금)
    
    # 전략 수익률 계산 
    strategy_return = weights.shift(1) * daily_return
    
    # 전략 누적 수익률 계산 및 저장
    cumulative_return = (1 + strategy_return.fillna(0)).cumprod()
    results[f'Threshold {t}%'] = cumulative_return

# -----------------------------------
# 5. 결과 시각화
# -----------------------------------
plt.figure(figsize=(14, 7))
plt.plot(results.index, results['Buy & Hold'], label='Buy & Hold (단순 보유)', color='black', linewidth=2, linestyle='--')

colors = ['red', 'orange', 'green', 'blue', 'purple']
for i, t in enumerate(THRESHOLDS):
    plt.plot(results.index, results[f'Threshold {t}%'], label=f'Threshold {t}% (Dynamic Weight)', color=colors[i], alpha=0.8)

plt.title(f"{TICKER} 다중 이격도 센서 비중 조절 백테스트 결과 ({START_DATE} ~ {END_DATE})", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return (누적 수익률)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------------
# 6. 최종 수익률 요약 출력
# -----------------------------------
print("\n[ 최종 누적 수익률 요약 ]")
for col in results.columns:
    final_return = (results[col].iloc[-1] - 1) * 100
    print(f"{col:<25}: {final_return:>8.2f} %")
