import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import sys

# ==========================================
# [1. 파라미터 및 729가지 밴드 조합 설정]
# ==========================================
TICKERS = ['QQQ', 'TLT', 'GLD']
WEIGHTS = {'QQQ': 0.60, 'TLT': 0.20, 'GLD': 0.20}

MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# 사용자가 요청한 9가지 밴드 조합 (상단 배수, 하단 배수)
BANDS_LIST = [
    (1.030, 0.970), (1.030, 0.975), (1.030, 0.980), # 매수 +3% / 매도 -3%, -2.5%, -2%
    (1.025, 0.970), (1.025, 0.975), (1.025, 0.980), # 매수 +2.5% / 매도 -3%, -2.5%, -2%
    (1.020, 0.970), (1.020, 0.975), (1.020, 0.980)  # 매수 +2% / 매도 -3%, -2.5%, -2%
]

# ==========================================
# [2. 데이터 다운로드 (최대 기간)]
# ==========================================
print(">>> 야후 파이낸스 데이터 다운로드 중 (period='max')...")
data_full = yf.download(TICKERS, period="max", progress=False)

if data_full.empty:
    print("데이터 다운로드 실패")
    sys.exit(1)

# 다중 인덱스 처리 및 결측치 제거 (가장 늦게 상장한 GLD 기준일로 통합됨)
if isinstance(data_full.columns, pd.MultiIndex):
    prices = data_full['Adj Close'].ffill().dropna() if 'Adj Close' in data_full.columns.get_level_values(0) else data_full['Close'].ffill().dropna()
else:
    prices = data_full['Adj Close'].ffill().dropna() if 'Adj Close' in data_full.columns else data_full['Close'].ffill().dropna()

# 자산별 내일의 일일 수익률 미리 계산 (shift(-1))
daily_returns = prices.pct_change().shift(-1)

# ==========================================
# [3. 고속 연산: 자산별 9가지 시나리오 수익률 선계산]
# ==========================================
print(">>> 각 자산별 9가지 밴드 시나리오 독립 연산 중 (속도 최적화)...")
asset_band_returns = {ticker: {} for ticker in TICKERS}

for ticker in TICKERS:
    for band in BANDS_LIST:
        up_mult, dn_mult = band
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
        
        # [해당 밴드 적용 시 자산의 포트폴리오 기여 수익률] = (기본 비중 * 스칼라) * 자산 일일 수익률
        asset_band_returns[ticker][band] = scalar_series * WEIGHTS[ticker] * daily_returns[ticker]

# ==========================================
# [4. 729가지 조합 합산 및 지표 추출]
# ==========================================
print(">>> 729가지 포트폴리오 전체 조합 백테스트 진행 중...")
results = []
start_idx = MA_WINDOWS[-1] # 200일선 계산을 위한 초기 워밍업 기간 제외

# itertools.product를 사용하여 9 x 9 x 9 = 729가지 조합 순회
for qqq_band, tlt_band, gld_band in itertools.product(BANDS_LIST, BANDS_LIST, BANDS_LIST):
    
    # 3개 자산의 사전 계산된 기여 수익률을 단순 합산하여 포트폴리오 총 수익률 산출
    port_ret = (asset_band_returns['QQQ'][qqq_band] + 
                asset_band_returns['TLT'][tlt_band] + 
                asset_band_returns['GLD'][gld_band])
    
    # 워밍업 기간 및 마지막 결측치 행 제거
    port_ret = port_ret.iloc[start_idx:-1] 
    
    # 지표 계산
    equity_curve = (1 + port_ret).cumprod()
    years = len(equity_curve) / 252.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    
    vol = port_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol != 0 else 0  # 무위험 수익률 2% 가정
    
    roll_max = equity_curve.cummax()
    mdd = (equity_curve / roll_max - 1).min()
    
    # 결과 저장 (출력 포맷 정리)
    results.append({
        'QQQ_Band': f"+{qqq_band[0]-1:.1%}/-{1-qqq_band[1]:.1%}",
        'TLT_Band': f"+{tlt_band[0]-1:.1%}/-{1-tlt_band[1]:.1%}",
        'GLD_Band': f"+{gld_band[0]-1:.1%}/-{1-gld_band[1]:.1%}",
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe
    })

# ==========================================
# [5. 결과 정렬 및 상위 10개 출력]
# ==========================================
df_results = pd.DataFrame(results)

# 샤프지수 기준으로 내림차순 정렬하여 최적의 상위 10개 추출
df_sorted = df_results.sort_values(by='Sharpe', ascending=False).head(10)

print("\n" + "="*80)
print("🏆 [Top 10 포트폴리오 최적 밴드 조합] (샤프지수 기준)")
print("="*80)

# 터미널 가독성을 높이기 위한 포맷터 적용
print(df_sorted.to_string(
    index=False, 
    formatters={
        'CAGR': '{:.2%}'.format, 
        'MDD': '{:.2%}'.format, 
        'Sharpe': '{:.3f}'.format
    }
))
print("="*80)
