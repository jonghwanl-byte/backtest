import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_weight_comparison():
    print("시장 데이터를 다운로드하는 중입니다...")
    
    tickers = ['QQQ', 'TLT', 'GLD']
    mas = [20, 120, 200]
    scalar_map = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.00}
    cash_rate = 0.02 / 252  
    
    # [최종 진화] 자산별 영점 조준이 끝난 궁극의 밴드 세팅
    bands = {
        'QQQ': (1.025, 0.975),
        'TLT': (1.030, 0.975),
        'GLD': (1.025, 0.975)
    }
    
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2004-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
    data = data.ffill().dropna()
    returns = data.pct_change().dropna()

    # 두 가지 비중 시나리오 정의
    scenarios = {
        "1) 60 / 15 / 25 (금 비중 높임)": {'QQQ': 0.60, 'TLT': 0.15, 'GLD': 0.25},
        "2) 60 / 25 / 15 (채권 비중 높임)": {'QQQ': 0.60, 'TLT': 0.25, 'GLD': 0.15}
    }

    print("\n비중 비교 백테스트 시뮬레이션 중...\n")
    print("="*65)
    print(" ⚖️ QQQ 60% 상태에서 TLT vs GLD 방어력 비교 ⚖️")
    print("="*65)

    plt.figure(figsize=(14, 10))
    colors = ['blue', 'green']

    for idx, (scenario_name, base_weights) in enumerate(scenarios.items()):
        portfolio_return = pd.Series(0.0, index=returns.index)
        asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

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

        total_invested = asset_weights.sum(axis=1)
        cash_weight = 1.0 - total_invested
        portfolio_return += cash_weight * cash_rate

        cum_returns = (1 + portfolio_return).cumprod()
        years = len(cum_returns) / 252.0

        cagr = cum_returns.iloc[-1] ** (1 / years) - 1
        ann_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = (portfolio_return.mean() * 252 - 0.02) / ann_vol

        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        mdd = drawdown.min()

        weight_changes = asset_weights.diff().fillna(0)
        trades_per_asset = (weight_changes != 0).sum()
        total_trades = trades_per_asset.sum()

        print(f"[{scenario_name}]")
        print(f"  ▶ CAGR: {cagr*100:.2f}% | MDD: {mdd*100:.2f}% | Sharpe: {sharpe:.2f}")
        print(f"  ▶ 총 리밸런싱: {total_trades:.0f}회 (QQQ {trades_per_asset['QQQ']}, TLT {trades_per_asset['TLT']}, GLD {trades_per_asset['GLD']})")
        print("-" * 65)

        plt.subplot(2, 1, 1)
        plt.plot(cum_returns.index, cum_returns, label=f"{scenario_name} (CAGR {cagr*100:.2f}%)", color=colors[idx], linewidth=1.5)
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdown.index, drawdown * 100, label=f"{scenario_name} (MDD {mdd*100:.2f}%)", color=colors[idx], linewidth=1.5, alpha=0.8)

    plt.subplot(2, 1, 1)
    plt.title('Cumulative Return Comparison (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Drawdown Comparison (%)')
    plt.ylabel('MDD (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('weight_comparison_60.png', dpi=150)
    print("\n[알림] 비교 그래프가 'weight_comparison_60.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_weight_comparison()
