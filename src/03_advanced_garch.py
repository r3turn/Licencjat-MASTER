import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import os

# LOAD DATA
returns = pd.read_parquet("data/returns.parquet")
advanced_volatility = pd.DataFrame(index=returns.index)
os.makedirs("charts/advanced_garch", exist_ok=True)

# GARCH, EGARCH, GJR
models_config = [
    {'name': 'GARCH',  'vol': 'Garch',  'p': 1, 'o': 0, 'q': 1, 'color': 'blue',   'style': '--'}, # Standard
    {'name': 'EGARCH', 'vol': 'EGarch', 'p': 1, 'o': 0, 'q': 1, 'color': 'green',  'style': '-'},  # Log
    {'name': 'GJR',    'vol': 'Garch',  'p': 1, 'o': 1, 'q': 1, 'color': 'purple', 'style': ':'}   # Asimetric
]

for ticker in returns.columns:
    y = returns[ticker]
    y_scaled = y * 100 
    
    for config in models_config:
        model_name = config['name']
        col_name = f"{ticker}_{model_name}"
        
        try:
            model = arch_model(y_scaled, vol=config['vol'], p=config['p'], o=config['o'], q=config['q'], dist='Normal')
            res = model.fit(disp='off')
            forecast_vol = res.conditional_volatility / 100
            advanced_volatility[col_name] = forecast_vol
            
        except Exception as e:
            print(f"Failed to fit {model_name}: {e}")

# SAVE RESULTS
advanced_volatility.to_parquet("data/advanced_garch_volatility.parquet")

# PLOTTING
if 'NVDA' in returns.columns:
    ticker = "NVDA"
    
    # --- NVDA FULL ---
    plt.figure(figsize=(14, 7))

    plt.plot(returns.index, returns[ticker], color='gray', alpha=0.3, label='Real Returns', linewidth=0.5)
    
    # Modele
    for config in models_config:
        name = config['name']
        col = f"{ticker}_{name}"
        if col in advanced_volatility.columns:
            sigma = advanced_volatility[col]
            plt.plot(sigma.index, 2 * sigma, 
                     color=config['color'], 
                     linestyle=config['style'], 
                     linewidth=1.0,
                     label=f"{name}")

    plt.title(f"Model Battle: GARCH vs EGARCH vs GJR ({ticker} 2005-2024)", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylabel("Volatility (Sigma)")
    plt.tight_layout()
    plt.savefig("charts/advanced_garch/model_battle_NVDA_full.png", dpi=300)
    plt.close()

    # --- NVDA ZOOM (COVID-19) ---
    start_zoom, end_zoom = "2020-01-01", "2020-12-31"
    
    zoom_ret = returns.loc[start_zoom:end_zoom, ticker]
    zoom_vol = advanced_volatility.loc[start_zoom:end_zoom]
    
    plt.figure(figsize=(12, 6))
    plt.plot(zoom_ret.index, zoom_ret, color='gray', alpha=0.3, label='Real Returns', linewidth=1.0)
    
    for config in models_config:
        col = f"{ticker}_{config['name']}"
        if col in zoom_vol.columns:
            sigma = zoom_vol[col]
            plt.plot(sigma.index, 2 * sigma, 
                     color=config['color'], 
                     linestyle=config['style'], 
                     linewidth=2.0,
                     label=f"{config['name']}")

    plt.title(f"Asymmetry Check: {ticker} (COVID-19 Zoom)", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("charts/advanced_garch/model_battle_NVDA_covid.png", dpi=300)
    plt.close()

# --- COMBO CHART ---
n_tickers = len(returns.columns)
fig, axes = plt.subplots(nrows=n_tickers, ncols=1, figsize=(14, 4 * n_tickers), sharex=True)
if n_tickers == 1: axes = [axes]

for i, ticker in enumerate(returns.columns):
    ax = axes[i]
    ax.plot(returns.index, returns[ticker], color='gray', alpha=0.15, linewidth=0.5)
    for config in models_config:
        col = f"{ticker}_{config['name']}"
        if col in advanced_volatility.columns:
            sigma = advanced_volatility[col]
            ax.plot(sigma.index, 2 * sigma, color=config['color'], linestyle=config['style'], linewidth=1.0, label=f"{config['name']}")
    ax.set_title(f"{ticker}", fontsize=12, loc='left', fontweight='bold')
    ax.grid(True, alpha=0.2)
    if i == 0: ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("charts/advanced_garch/ALL_MODELS_SUMMARY.png", dpi=300)
plt.close()

print(f"Charts saved in 'charts/advanced_garch/'")