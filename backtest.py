import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_tlt_comparison():
    print("TLT 데이터를 다운로드하는 중입니다...")
    
    # 1. 데이터 다운로드
    ticker = 'TLT'
    df = yf.download(ticker, start='2004-01-01', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    
    # To-be 신호 강도
    scalar_map = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.00}
    cash_rate = 0.02 / 252 # 현금 이자 연 2%

    # 2. 4가지 테스트 시나리오 정의
    scenarios = [
        {
            "name": "기존 (3% 룰)",
            "up_band": 1.03, "dn_band": 0.97, "mas": [20, 120, 200]
        },
        {
            "name": "대안 A (비대칭: 매수 +3% / 매도 -1%)",
            "up_band": 1.03, "dn_band": 0.99, "mas": [20, 120, 200]
        },
        {
            "name": "대안 B (1.5% 맞춤형 밴드)",
            "up_band": 1.015, "dn_band": 0.985, "mas": [20, 120, 200]
        },
        {
            "name": "대안 C (MA 기간 60/120/200, 3% 룰)",
            "up_band": 1.03, "dn_band": 0.97, "mas": [60, 120, 200]
        }
    ]

    print("각 시나리오별 백테스트를 계산하는 중입니다...\n")
    print("="*60)
    print("   TLT 단독 (100%) 하락장 탈출 전략 비교 백테스트")
    print("="*60)

    # 그래프 그리기 준비
    plt.figure(figsize=(14, 10))
    colors = ['gray', 'blue', 'green', 'red']

    for idx, sc in enumerate(scenarios):
        mas = sc["mas"]
        up_band = sc["up_band"]
        dn_band = sc["dn_band"]
        
        total_signals = pd.Series(0, index=prices.index)
        
        for ma_period in mas:
            ma = prices.rolling(window=ma_period).mean()
            state = np.zeros(len(prices))
            p_vals = prices.values
            m_vals = ma.values
            curr_state = 0
            
            for i in range(len(p_vals)):
                if np.isnan(m_vals[i]):
                    continue
                # 진입/이탈 조건
                if p_vals[i] > m_vals[i] * up_band:
                    curr_state = 1
                elif p_vals[i] < m_vals[i] * dn_band:
                    curr_state = 0
                state[i] = curr_state
                
            total_signals += state
            
        # 비중 계산
        invested_fraction = total_signals.map(scalar_map).shift(1).fillna(0)
        cash_fraction = 1.0 - invested_fraction
        
        # 포트폴리오 수익률 = (TLT 수익) + (현금 이자 수익)
        port_returns = (invested_fraction * returns) + (cash_fraction * cash_rate)
        
        # 성과 지표 계산
        cum_returns = (1 + port_returns).cumprod()
        years = len(cum_returns) / 252.0
        cagr = cum_returns.iloc[-1] ** (1 / years) - 1
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = (port_returns.mean() * 252 - 0.02) / ann_vol
        
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        mdd = drawdown.min()
        
        # 매매 횟수 계산
        weight_changes = invested_fraction.diff().fillna(0)
        total_trades = (weight_changes != 0).sum()
        trades_per_year = total_trades / years

        # 결과 출력
        print(f"[{idx+1}] {sc['name']}")
        print(f"    ▶ CAGR: {cagr*100:.2f}% | MDD: {mdd*100:.2f}% | Sharpe: {sharpe:.2f}")
        print(f"    ▶ 총 매매 횟수: {total_trades}회 (연평균 {trades_per_year:.1f}회)")
        print("-" * 60)

        # 그래프 추가
        plt.subplot(2, 1, 1)
        plt.plot(cum_returns.index, cum_returns, label=f"{sc['name']} (CAGR {cagr*100:.2f}%)", color=colors[idx], linewidth=1.5 if idx > 0 else 1)
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdown.index, drawdown * 100, label=f"{sc['name']} (MDD {mdd*100:.2f}%)", color=colors[idx], linewidth=1.5 if idx > 0 else 1)

    # 차트 꾸미기
    plt.subplot(2, 1, 1)
    plt.title('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Drawdown (%)')
    plt.ylabel('MDD (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('tlt_comparison.png', dpi=150)
    print("\n[알림] 비교 그래프가 'tlt_comparison.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_tlt_comparison()
