import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest():
    print("데이터를 다운로드하는 중입니다...")
    
    # 1. 대상 자산 및 파라미터 설정
    tickers = ['QQQ', 'TLT', 'GLD', 'XLE']
    base_weights = {'QQQ': 0.40, 'TLT': 0.30, 'GLD': 0.20, 'XLE': 0.10}
    mas = [20, 120, 200]
    
    # 2. 데이터 다운로드 (yfinance 다중 다운로드 방식 적용)
    # 2004년부터 다운로드하되, GLD 상장일(2004-11) 기준으로 모든 데이터가 있는 시점부터 정렬됨
    df_raw = yf.download(tickers, start='2004-01-01', progress=False)
    
    # [핵심 수정 사항] 'Close' 대신 'Adj Close'(수정 종가: 배당 및 분할 반영) 사용 ⭐
    data = df_raw['Adj Close'].dropna() 
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
            
            # +/- 3% 밴드 적용
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
        
        # Look-ahead bias(미래참조오류) 방지: 어제의 신호로 오늘 투자
        invested_fraction = invested_fraction.shift(1).fillna(0)
        
        # 실제 포트폴리오 내 비중 할당
        actual_weight = invested_fraction * base_weights[ticker]
        asset_weights[ticker] = actual_weight
        
        # 포트폴리오 일일 수익률 합산
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
    sharpe = (ann_ret - 0.02) / ann_vol  # 무위험수익률 2% 차감

    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    mdd = drawdown.min()

    # 7. 결과 터미널 출력
    print("\n" + "="*50)
    print("   Independent-Hysteresis-TAA Backtest Result")
    print("="*50)
    print(f"전략 모델    : 배당 재투자 반영 (Adj Close 적용)")
    print(f"목표 비중    : QQQ {base_weights['QQQ']*100:.0f}%, TLT {base_weights['TLT']*100:.0f}%, GLD {base_weights['GLD']*100:.0f}%, XLE {base_weights['XLE']*100:.0f}%")
    print(f"테스트 기간  : {cum_returns.index[0].date()} ~ {cum_returns.index[-1].date()}")
    print("-" * 50)
    print(f"▶ 누적 수익률 : {(cum_returns.iloc[-1] - 1)*100:.2f}%")
    print(f"▶ 연평균 수익 (CAGR) : {cagr*100:.2f}%")
    print(f"▶ 최대 낙폭 (MDD)    : {mdd*100:.2f}%")
    print(f"▶ 연평균 변동성      : {ann_vol*100:.2f}%")
    print(f"▶ 샤프 지수 (Sharpe) : {sharpe:.2f}")
    print("="*50)

    # 8. 그래프 이미지 저장 (GitHub Actions Artifact 확인용)
    plt.figure(figsize=(12, 6))
    
    # 누적 수익률 차트
    plt.subplot(2, 1, 1)
    plt.plot(cum_returns.index, cum_returns, label='Portfolio (Total Return)', color='blue')
    plt.title('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 낙폭(MDD) 차트
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
