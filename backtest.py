import yfinance as yf
import numpy as np
import pandas as pd

# Define parameters
TICKERS_CURRENT = ['QQQ', 'TLT', 'GLD']
WEIGHTS_CURRENT = {'QQQ': 0.50, 'TLT': 0.25, 'GLD': 0.25}
BANDS_CURRENT = {'QQQ': (1.025, 0.975), 'TLT': (1.030, 0.975), 'GLD': (1.025, 0.975)}

TICKERS_JD = ['QQQ', 'SPY', 'DIA', 'GLD', 'TLT', 'FBTC', 'RUM', 'OILK']
WEIGHTS_JD = {'QQQ': 0.24, 'SPY': 0.24, 'DIA': 0.24, 'GLD': 0.06, 'TLT': 0.06, 'FBTC': 0.06, 'RUM': 0.06, 'OILK': 0.02}
CASH_JD = 0.02
BANDS_JD = {
    'QQQ': (1.025, 0.975),
    'TLT': (1.030, 0.975),
    'GLD': (1.025, 0.975),
    'SPY': (1.03, 0.97),
    'DIA': (1.03, 0.97),
    'FBTC': (1.03, 0.97),
    'RUM': (1.03, 0.97),
    'OILK': (1.03, 0.97)
}

MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}

# Download Data
all_tickers = list(set(TICKERS_CURRENT + TICKERS_JD))
# FBTC inception is Jan 2024. RUM is Sep 2022. 
# We will use BTC-USD as a proxy for FBTC prior to 2024 to get a meaningful backtest, 
# otherwise the backtest is only from Oct 2024 (200 days after Jan 2024).
proxy_tickers = all_tickers.copy()
if 'FBTC' in proxy_tickers:
    proxy_tickers.remove('FBTC')
    proxy_tickers.append('BTC-USD')

data = yf.download(proxy_tickers, period="5y", progress=False)['Close']
data.rename(columns={'BTC-USD': 'FBTC'}, inplace=True, errors='ignore')
data = data.ffill().dropna(how='all')

def run_backtest(tickers, base_weights, bands, cash_weight=0.0):
    prices = data[tickers].dropna()
    if len(prices) == 0:
        return None
    
    # Calculate MAs and bands
    states = {}
    for ticker in tickers:
        up_mult, dn_mult = bands[ticker]
        states[ticker] = {}
        for w in MA_WINDOWS:
            ma = prices[ticker].rolling(window=w).mean()
            upper = ma * up_mult
            lower = ma * dn_mult
            
            # Vectorized state calculation (simplified for backtest: close vs bands)
            # 1 if price > upper, 0 if price < lower, else previous state
            cond_up = prices[ticker] > upper
            cond_dn = prices[ticker] < lower
            
            state_series = pd.Series(np.nan, index=prices.index)
            state_series[cond_up] = 1.0
            state_series[cond_dn] = 0.0
            state_series = state_series.ffill().fillna(0.0) # default 0 before crossing
            
            states[ticker][w] = state_series
            
    # Calculate score
    portfolio_daily_return = pd.Series(0.0, index=prices.index)
    daily_returns = prices.pct_change().shift(-1) # return from today close to tomorrow close
    
    # Start backtest after 200 days
    start_idx = MA_WINDOWS[-1]
    
    equity_curve = [1.0]
    
    for i in range(start_idx, len(prices)-1):
        total_weight = 0.0
        port_ret = 0.0
        
        for ticker in tickers:
            score = 0
            for w in MA_WINDOWS:
                score += states[ticker][w].iloc[i]
            
            scalar = SCALAR_MAP.get(score, 0.0)
            target_w = base_weights[ticker] * scalar
            
            ret = daily_returns[ticker].iloc[i]
            port_ret += target_w * ret
            total_weight += target_w
            
        # Add cash return (assume 0% for simplicity)
        cash_alloc = (1.0 - total_weight - cash_weight) + cash_weight
        # no return on cash
        
        equity_curve.append(equity_curve[-1] * (1 + port_ret))
        
    dates = prices.index[start_idx:len(prices)]
    eq_df = pd.DataFrame({'Equity': equity_curve}, index=dates)
    return eq_df

eq_current = run_backtest(TICKERS_CURRENT, WEIGHTS_CURRENT, BANDS_CURRENT, cash_weight=0.0)
eq_jd = run_backtest(TICKERS_JD, WEIGHTS_JD, BANDS_JD, cash_weight=CASH_JD)

def get_metrics(eq_series):
    eq = eq_series['Equity']
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    
    daily_ret = eq.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol != 0 else 0
    
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1
    mdd = drawdown.min()
    
    return cagr, mdd, sharpe

if eq_current is not None and eq_jd is not None:
    cagr_c, mdd_c, sh_c = get_metrics(eq_current)
    cagr_j, mdd_j, sh_j = get_metrics(eq_jd)
    print(f"Current: CAGR={cagr_c:.2%}, MDD={mdd_c:.2%}, Sharpe={sh_c:.2f}")
    print(f"JD Vance: CAGR={cagr_j:.2%}, MDD={mdd_j:.2%}, Sharpe={sh_j:.2f}")
else:
    print("Not enough data to run backtest.")
