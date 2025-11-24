import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import os

# LOAD DATA
returns = pd.read_parquet("data/returns.parquet")
garch_volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
garch_var95 = pd.DataFrame(index=returns.index, columns=returns.columns)

# ARCH(1,1) MODELS
for ticker in returns.columns:
    y = returns[ticker]
    y_scaled = y * 100 
    
    # Train model
    model = arch_model(y_scaled, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    
    # Extract results
    forecast_vol = res.conditional_volatility / 100
    mu = res.params['mu'] / 100
    var_95 = mu - (forecast_vol * 1.645)
    
    # Store
    garch_volatility[ticker] = forecast_vol
    garch_var95[ticker] = var_95

# SAVE RESULTS
garch_volatility.to_parquet("data/garch_volatility.parquet")
garch_var95.to_parquet("data/garch_var95.parquet")

# NVDA Plot
if 'NVDA' in returns.columns:
    plt.figure(figsize=(12, 6))
    
    ticker = 'NVDA'
    sigma = garch_volatility[ticker]
    
    # Plot
    plt.plot(returns.index, returns[ticker], color='gray', alpha=0.4, label='Real Returns', linewidth=0.6)
    plt.plot(sigma.index, 2 * sigma, color='red', linewidth=1.5, label='GARCH Volatility (+/- 2σ)')
    plt.plot(sigma.index, -2 * sigma, color='red', linewidth=1.5)
    
    plt.title(f"GARCH(1,1) Volatility Forecast: {ticker}", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.ylabel("Log Return / Volatility")
    plt.tight_layout()
    
    plt.savefig("charts/garch_NVDA.png", dpi=300)
    plt.close()

# COMBO chart
n_tickers = len(returns.columns)
fig, axes = plt.subplots(nrows=n_tickers, ncols=1, figsize=(12, 4 * n_tickers), sharex=True)

for i, ticker in enumerate(returns.columns):
    ax = axes[i]
    sigma = garch_volatility[ticker]
    
    # x axis
    ax.plot(returns.index, returns[ticker], color='gray', alpha=0.4, label='Real Returns', linewidth=0.5)
    ax.plot(sigma.index, 2 * sigma, color='#D32F2F', linewidth=1.2, label='GARCH Volatility (+/- 2σ)')
    ax.plot(sigma.index, -2 * sigma, color='#D32F2F', linewidth=1.2)
    
    ax.set_title(f"{ticker}", fontsize=12, loc='left', fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Legend only for first chart
    if i == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("charts/garch_combined_summary.png", dpi=300)
plt.close()
print(f"CAHRTS GENERATED")