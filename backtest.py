import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_comparison_backtest():
    print("시장 데이터를 안전하게 다운로드하는 중입니다...")
    
    tickers = ['QQQ', 'TLT', 'GLD']
    base_weights = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}
    mas = [20, 120, 200]
    scalar_map = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.00}
    cash_rate = 0.02 / 252  # 현금 이자 연 2%
    
    # 데이터 안전하게 개별 다운로드 (MultiIndex 오류 방지)
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2004-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
    data = data.ffill().dropna()
    returns = data.pct_change().dropna()

    # 두 가지 시나리오 정의
    scenarios = {
        "1) 기존 순정 모델": {
            'QQQ': (1.030, 0.970),
            'TLT': (1.030, 0.975),
            'GLD': (1.030, 0.970)
        },
        "2) 궁극의 하이브리드 (QQQ/GLD ±2.5%, TLT +3%/-2.5%)": {
            'QQQ': (1.025, 0.975),
            'TLT': (1.030, 0.975),
            'GLD': (1.025, 0.975)
        }
    }

    print("\n포트폴리오 백테스트 시뮬레이션 중...\n")
    print("="*60)
    print(" 🥊 순정 모델 vs 궁극의 하이브리드 모델 비교 🥊")
    print("="*60)

    # 그래프 준비
    plt.figure(figsize=(14, 10))
    colors = ['gray', 'purple']

    for idx, (scenario_name, bands) in enumerate(scenarios.items()):
        portfolio_return = pd.Series(0.0, index=returns.index)
        asset_weights = pd.DataFrame(index=returns.index, columns=tickers)

        # 전략 로직 적용
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

        # 현금 이자 반영
        total_invested = asset_weights.sum(axis=1)
        cash_weight = 1.0 - total_invested
        portfolio_return += cash_weight * cash_rate

        # 성과 지표 계산
        cum_returns = (1 + portfolio_return).cumprod()
        years = len(cum_returns) / 252.0

        cagr = cum_returns.iloc[-1] ** (1 / years) - 1
        ann_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = (portfolio_return.mean() * 252 - 0.02) / ann_vol

        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        mdd = drawdown.min()

        # 매매 횟수 계산
        weight_changes = asset_weights.diff().fillna(0)
        trades_per_asset = (weight_changes != 0).sum()
        total_trades = trades_per_asset.sum()

        # 결과 터미널 출력
        print(f"[{scenario_name}]")
        print(f"  ▶ CAGR: {cagr*100:.2f}% | MDD: {mdd*100:.2f}% | Sharpe: {sharpe:.2f}")
        print(f"  ▶ 총 리밸런싱: {total_trades:.0f}회 (QQQ {trades_per_asset['QQQ']}, TLT {trades_per_asset['TLT']}, GLD {trades_per_asset['GLD']})")
        print("-" * 60)

        # 차트 그리기
        plt.subplot(2, 1, 1)
        linewidth = 2.0 if idx == 1 else 1.5
        plt.plot(cum_returns.index, cum_returns, label=f"{scenario_name} (CAGR {cagr*100:.2f}%)", color=colors[idx], linewidth=linewidth)
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdown.index, drawdown * 100, label=f"{scenario_name} (MDD {mdd*100:.2f}%)", color=colors[idx], linewidth=linewidth, alpha=0.8)

    # 차트 꾸미기
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
    plt.savefig('ultimate_comparison.png', dpi=150)
    print("\n[알림] 비교 그래프가 'ultimate_comparison.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_comparison_backtest()
