import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_4asset_hybrid_backtest():
    print("4개 자산 시장 데이터를 안전하게 다운로드하는 중입니다...")
    
    # 1. 대상 자산 및 목표 비중 설정
    tickers = ['QQQ', 'TLT', 'IEF', 'GLD']
    base_weights = {
        'QQQ': 0.50, 
        'TLT': 0.125, 
        'IEF': 0.125, 
        'GLD': 0.25
    }
    
    mas = [20, 120, 200]
    scalar_map = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.00}
    cash_rate = 0.02 / 252  # 현금 파킹 시 연 2% 이자 가정
    
    # [최종 진화] 자산별 영점 조준이 끝난 궁극의 밴드 세팅
    # IEF는 중기 '채권'이므로 TLT와 동일한 룰을 적용합니다.
    bands = {
        'QQQ': (1.025, 0.975),  # 매수 +2.5% / 매도 -2.5%
        'TLT': (1.030, 0.975),  # 매수 +3.0% / 매도 -2.5%
        'IEF': (1.030, 0.975),  # 매수 +3.0% / 매도 -2.5%
        'GLD': (1.025, 0.975)   # 매수 +2.5% / 매도 -2.5%
    }
    
    # 2. 데이터 안전하게 개별 다운로드
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2004-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
    # IEF 상장일(2002년) 이후 데이터가 모두 존재하는 시점부터 맞춤
    data = data.ffill().dropna()
    returns = data.pct_change().dropna()

    print("\n4개 자산 포트폴리오 백테스트 시뮬레이션 중...\n")
    
    portfolio_return = pd.Series(0.0, index=returns.index)
    asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

    # 3. Hysteresis TAA 로직 적용
    for ticker in tickers:
        price = data[ticker]
        total_signals = pd.Series(0, index=price.index)
        up_band, dn_band = bands[ticker]
        
        for ma_period in mas:
            ma = price.rolling(window=ma_period).mean()
            state = np.zeros(len(price))
            p_vals = price.values
            m_vals = ma.values
            curr_state = 0
            
            for i in range(len(p_vals)):
                if np.isnan(m_vals[i]): continue
                if p_vals[i] > m_vals[i] * up_band: curr_state = 1
                elif p_vals[i] < m_vals[i] * dn_band: curr_state = 0
                state[i] = curr_state
                
            total_signals += state
            
        invested_fraction = total_signals.map(scalar_map).shift(1).fillna(0)
        actual_weight = invested_fraction * base_weights[ticker]
        asset_weights[ticker] = actual_weight
        portfolio_return += actual_weight * returns[ticker]

    # 4. 현금(Cash) 비중에 대한 이자 수익
    total_invested = asset_weights.sum(axis=1)
    cash_weight = 1.0 - total_invested
    portfolio_return += cash_weight * cash_rate

    # 5. 성과 지표 계산
    cum_returns = (1 + portfolio_return).cumprod()
    years = len(cum_returns) / 252.0

    cagr = cum_returns.iloc[-1] ** (1 / years) - 1
    ann_vol = portfolio_return.std() * np.sqrt(252)
    sharpe = (portfolio_return.mean() * 252 - 0.02) / ann_vol

    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    mdd = drawdown.min()

    # 6. 매매 횟수 계산
    weight_changes = asset_weights.diff().fillna(0)
    trades_per_asset = (weight_changes != 0).sum()
    total_trades = trades_per_asset.sum()
    trades_per_year = total_trades / years

    # 7. 결과 출력
    print("="*65)
    print(" 🚀 Ultimate Hybrid 4-Asset (QQQ/TLT/IEF/GLD) 🚀")
    print("="*65)
    print(f"목표 비중 : QQQ 50%, TLT 12.5%, IEF 12.5%, GLD 25%")
    print(f"테스트기간: {cum_returns.index[0].date()} ~ {cum_returns.index[-1].date()}")
    print("-" * 65)
    print(f"▶ 연평균 수익 (CAGR) : {cagr*100:.2f}%")
    print(f"▶ 최대 낙폭 (MDD)    : {mdd*100:.2f}%")
    print(f"▶ 샤프 지수 (Sharpe) : {sharpe:.2f}")
    print("-" * 65)
    print(f"▶ 총 리밸런싱 횟수   : {total_trades:.0f}회 (연평균 {trades_per_year:.1f}회)")
    print(f"   [상세] QQQ: {trades_per_asset['QQQ']} | TLT: {trades_per_asset['TLT']} | IEF: {trades_per_asset['IEF']} | GLD: {trades_per_asset['GLD']}")
    print("="*65)

    # 8. 차트 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(cum_returns.index, cum_returns, label=f"Portfolio (CAGR {cagr*100:.2f}%)", color='purple', linewidth=1.5)
    plt.title('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, label=f"MDD {mdd*100:.2f}%", color='red', linewidth=1)
    plt.title('Drawdown (%)')
    plt.ylabel('MDD (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('portfolio_4asset.png', dpi=150)
    print("\n[알림] 백테스트 결과 그래프가 'portfolio_4asset.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_4asset_hybrid_backtest()
