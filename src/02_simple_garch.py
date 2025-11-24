import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import os

# LOAD DATA
returns = pd.read_parquet("data/returns.parquet")
garch_volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
garch_var95 = pd.DataFrame(index=returns.index, columns=returns.columns)

# Create subfolder for charts
os.makedirs("charts/simple_garch", exist_ok=True)

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

# NVDA Plot (Full History)
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
    
    # Save to subfolder
    plt.savefig("charts/simple_garch/garch_NVDA.png", dpi=300)
    plt.close()

    # --- NVDA Plot (2020 Zoom) ---
    plt.figure(figsize=(12, 6))
    
    # Slice data for 2020
    zoom_start = "2020-01-01"
    zoom_end = "2020-12-31"
    
    zoom_ret = returns.loc[zoom_start:zoom_end, ticker]
    zoom_vol = sigma.loc[zoom_start:zoom_end]
    
    # Plot Zoom
    plt.plot(zoom_ret.index, zoom_ret, color='gray', alpha=0.4, label='Real Returns', linewidth=1.0)
    plt.plot(zoom_vol.index, 2 * zoom_vol, color='red', linewidth=2.0, label='GARCH Volatility (+/- 2σ)')
    plt.plot(zoom_vol.index, -2 * zoom_vol, color='red', linewidth=2.0)
    
    plt.title(f"GARCH(1,1) Reaction to COVID-19: {ticker} (2020 Zoom)", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylabel("Log Return / Volatility")
    plt.tight_layout()
    
    # Save to subfolder
    plt.savefig("charts/simple_garch/garch_NVDA_2020_zoom.png", dpi=300)
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
# Save to subfolder
plt.savefig("charts/simple_garch/garch_combined_summary.png", dpi=300)
plt.close()
print(f"CAHRTS GENERATED")