import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest():
    print("데이터를 다운로드하는 중입니다...")
    
    # 1. 대상 자산 및 파라미터 설정 (최적화된 오리지널 비중)
    tickers = ['QQQ', 'TLT', 'GLD']
    base_weights = {'QQQ': 0.5, 'TLT': 0.3, 'GLD': 0.2}
    mas = [20, 120, 200]
    
    # 2. 데이터 다운로드 (yfinance 최신 버전 호환성 강화)
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2004-01-01', progress=False)
        
        # 반환된 데이터가 MultiIndex인 경우 최상단 컬럼명만 추출
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 구버전('Adj Close' 존재)과 신버전('Close'가 곧 수정주가) 모두 대응
        if 'Adj Close' in df.columns:
            data[ticker] = df['Adj Close']
        else:
            data[ticker] = df['Close']
            
    data = data.dropna()
    returns = data.pct_change().dropna()

    print("백테스트를 계산하는 중입니다...")
    
    # 3. 포트폴리오 수익률 및 비중 추적 변수 초기화
    portfolio_return = pd.Series(0.0, index=returns.index)
    asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

    # 4. Hysteresis TAA 로직 적용
    for ticker in tickers:
        price = data[ticker]
        total_signals = pd.Series(0, index=price.index)
        
        for ma_period in mas:
            ma = price.rolling(window=ma_period).mean()
            
            # Hysteresis 상태 배열 초기화
            state = np.zeros(len(price))
            p_vals = price.values
            m_vals = ma.values
            curr_state = 0
            
            # +/- 3% 밴드 적용 (검증된 안정적인 3% 유지)
            for i in range(len(p_vals)):
                if np.isnan(m_vals[i]):
                    continue
                if p_vals[i] > m_vals[i] * 1.03:
                    curr_state = 1
                elif p_vals[i] < m_vals[i] * 0.97:
                    curr_state = 0
                state[i] = curr_state
                
            total_signals += state
            
        # 신호(0~3)에 따른 투자 비중 계산 (0%, 33.3%, 66.6%, 100%)
        invested_fraction = total_signals / 3.0
        # 미래 참조 오류 방지
        invested_fraction = invested_fraction.shift(1).fillna(0)
        
        actual_weight = invested_fraction * base_weights[ticker]
        asset_weights[ticker] = actual_weight
        portfolio_return += actual_weight * returns[ticker]

    # 5. 파킹(Cash) 비중에 대한 이자 수익 계산 (보수적으로 연 2% 가정)
    total_invested = asset_weights.sum(axis=1)
    cash_weight = 1.0 - total_invested
    cash_return = 0.02 / 252  
    portfolio_return += cash_weight * cash_return

    # 6. 성과 지표 (Metrics) 계산
    cum_returns = (1 + portfolio_return).cumprod()
    years = len(cum_returns) / 252.0

    cagr = cum_returns.iloc[-1] ** (1 / years) - 1
    ann_ret = portfolio_return.mean() * 252
    ann_vol = portfolio_return.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol

    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    mdd = drawdown.min()

    # 6.5 거래 횟수(Turnover) 계산
    weight_changes = asset_weights.diff().fillna(0)
    trades_per_asset = (weight_changes != 0).sum()
    total_trades = trades_per_asset.sum()
    trades_per_year = total_trades / years

    # 7. 결과 터미널 출력
    print("\n" + "="*50)
    print("   Independent-Hysteresis-TAA Backtest Result")
    print("="*50)
    print(f"목표 비중    : QQQ {base_weights['QQQ']*100:.0f}%, TLT {base_weights['TLT']*100:.0f}%, GLD {base_weights['GLD']*100:.0f}%)
    print(f"테스트 기간  : {cum_returns.index[0].date()} ~ {cum_returns.index[-1].date()}")
    print("-" * 50)
    print(f"▶ 연평균 수익 (CAGR) : {cagr*100:.2f}%")
    print(f"▶ 최대 낙폭 (MDD)    : {mdd*100:.2f}%")
    print(f"▶ 연평균 변동성      : {ann_vol*100:.2f}%")
    print(f"▶ 샤프 지수 (Sharpe) : {sharpe:.2f}")
    print("-" * 50)
    print(f"▶ 총 리밸런싱 횟수   : {total_trades:.0f}회 (연평균 {trades_per_year:.1f}회)")
    print(f"   [상세] QQQ: {trades_per_asset['QQQ']}회 | TLT: {trades_per_asset['TLT']}회 | GLD: {trades_per_asset['GLD']}회")
    print("="*50)

    # 8. 그래프 이미지 저장
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(cum_returns.index, cum_returns, label='Portfolio', color='blue')
    plt.title('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
    plt.title('Drawdown (%)')
    plt.ylabel('MDD (%)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_result.png', dpi=150)
    print("\n[알림] 그래프가 'backtest_result.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_backtest()
