import yfinance as yf
import numpy as np
import pandas as pd

# ==========================================
# [1. 전략 파라미터 설정]
# ==========================================
# 현재 포트폴리오
TICKERS_CURRENT = ['QQQ', 'TLT', 'GLD']
WEIGHTS_CURRENT = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}
BANDS_CURRENT = {'QQQ': (1.025, 0.975), 'TLT': (1.030, 0.975), 'GLD': (1.025, 0.975)}

# 수정 포트폴리오 (주식 75%, 안전자산 25%)
TICKERS_MODIFIED = ['QQQ', 'SPY', 'DIA', 'GLD', 'TLT']
WEIGHTS_MODIFIED = {'QQQ': 0.25, 'SPY': 0.25, 'DIA': 0.25, 'GLD': 0.125, 'TLT': 0.125}
BANDS_MODIFIED = {
    'QQQ': (1.025, 0.975),
    'SPY': (1.030, 0.970),  # 3% 룰 적용
    'DIA': (1.030, 0.970),  # 3% 룰 적용
    'GLD': (1.025, 0.975),
    'TLT': (1.030, 0.975)
}

MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# ==========================================
# [2. 공통 백테스트 함수]
# ==========================================
def run_backtest(data_prices, tickers, weights, bands):
    prices = data_prices[tickers]
    daily_returns = prices.pct_change().shift(-1)
    
    weights_df = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    
    for ticker in tickers:
        up_mult, dn_mult = bands[ticker]
        base_w = weights[ticker]
        
        score_series = pd.Series(0, index=prices.index)
        
        for w in MA_WINDOWS:
            ma = prices[ticker].rolling(window=w).mean()
            upper = ma * up_mult
            lower = ma * dn_mult
            
            # 주가가 상단 밴드 위면 1.0, 하단 밴드 아래면 0.0, 그 사이면 이전 상태 유지
            cond_up = prices[ticker] > upper
            cond_dn = prices[ticker] < lower
            
            state = pd.Series(np.nan, index=prices.index)
            state[cond_up] = 1.0
            state[cond_dn] = 0.0
            state = state.ffill().fillna(0.0)
            
            score_series += state
            
        scalar_series = score_series.map(SCALAR_MAP).fillna(0.0)
        weights_df[ticker] = scalar_series * base_w

    # 포트폴리오 일일 수익률 계산 및 누적 수익률(Equity Curve) 산출
    port_returns = (weights_df * daily_returns).sum(axis=1)
    start_idx = MA_WINDOWS[-1]  # 200일 MA 계산을 위한 초기 기간 제외
    port_returns = port_returns.iloc[start_idx:-1] 
    
    equity_curve = (1 + port_returns).cumprod()
    return equity_curve

# ==========================================
# [3. 지표 산출 함수]
# ==========================================
def get_metrics(equity_curve):
    years = len(equity_curve) / 252.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    
    daily_ret = equity_curve.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol != 0 else 0  # 무위험 수익률 2% 가정
    
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1
    mdd = drawdown.min()
    
    return cagr, mdd, sharpe

# ==========================================
# [4. 메인 실행]
# ==========================================
if __name__ == "__main__":
    print(">>> 야후 파이낸스에서 시장 데이터 다운로드 중...")
    all_tickers = list(set(TICKERS_CURRENT + TICKERS_MODIFIED))
    
    # 넉넉한 백테스트 기간 확보
    data = yf.download(all_tickers, period="max", progress=False)['Close']
    data = data.ffill().dropna()

    print(">>> 백테스트 계산 중...\n")
    
    # 현재 포트폴리오 테스트
    eq_current = run_backtest(data, TICKERS_CURRENT, WEIGHTS_CURRENT, BANDS_CURRENT)
    c_cagr, c_mdd, c_sharpe = get_metrics(eq_current)
    
    # 수정 포트폴리오 테스트
    eq_modified = run_backtest(data, TICKERS_MODIFIED, WEIGHTS_MODIFIED, BANDS_MODIFIED)
    m_cagr, m_mdd, m_sharpe = get_metrics(eq_modified)

    # 결과 출력
    print("="*50)
    print("📊 [현재 포트폴리오] (QQQ 50 / TLT 25 / GLD 25)")
    print(f" - CAGR  : {c_cagr:.2%}")
    print(f" - MDD   : {c_mdd:.2%}")
    print(f" - Sharpe: {c_sharpe:.2f}")
    print("-" * 50)
    print("📊 [수정 포트폴리오] (QQQ 25 / SPY 25 / DIA 25 / GLD 12.5 / TLT 12.5)")
    print(f" - CAGR  : {m_cagr:.2%}")
    print(f" - MDD   : {m_mdd:.2%}")
    print(f" - Sharpe: {m_sharpe:.2f}")
    print("="*50)
