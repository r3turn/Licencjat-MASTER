import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- DATA DOWNLOAD SECTION ---

# ticker setup
TICKER_LIST = ['AAPL', 'NVDA', 'JPM', 'XOM', 'KGH.WA']
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

# data download
raw_data = yf.download(tickers=TICKER_LIST, start=START_DATE, end=END_DATE, auto_adjust=False)

df_prices = raw_data['Adj Close'].ffill().dropna()
df_volume = raw_data['Volume'].ffill().loc[df_prices.index] # index match

print(df_prices.head())
print(df_volume.head())

# log returns
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# saving dataframes
os.makedirs("data", exist_ok=True)
df_prices.to_parquet("data/prices.parquet")
df_volume.to_parquet("data/volume.parquet")
df_returns.to_parquet("data/returns.parquet")
print("Dataframes successfully saved to 'data/' folder.")

# --- VISUALIZATION SECTION ---

os.makedirs("charts", exist_ok=True)

# Visualizing Volatility Clustering
# periods of calm vs. periods of high volatility (clusters)
plt.figure(figsize=(15, 10))
for i, ticker in enumerate(TICKER_LIST, 1):
    plt.subplot(len(TICKER_LIST), 1, i)
    plt.plot(df_returns.index, df_returns[ticker], label=ticker, linewidth=0.8, alpha=0.8)
    plt.title(f"Log Returns: {ticker}", fontsize=10, loc='left')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("charts/returns_plot.png", dpi=300)
plt.close()

# Distributions vs Normal Distribution
# This chart proves that returns are NOT normally distributed
plt.figure(figsize=(15, 8))
for i, ticker in enumerate(TICKER_LIST, 1):
    plt.subplot(2, 3, i) # Grid layout (2 rows, 3 columns)
    
    # Plot histogram of actual data
    sns.histplot(df_returns[ticker], bins=100, kde=True, stat="density", color='skyblue', alpha=0.6)
    
    # Calculate and plot ideal Normal Distribution curve for comparison
    mu, std = df_returns[ticker].mean(), df_returns[ticker].std()
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    plt.plot(x, p, 'r--', linewidth=2, label='Normal Dist')
    
    plt.title(f"Distribution: {ticker}")
    plt.xlim(-0.1, 0.1) # Limit x-axis to focus on the center
    if i == 1: plt.legend()

plt.tight_layout()
plt.savefig("charts/distribution_plot.png", dpi=300)
plt.close()

print("Charts successfully saved to 'charts/' folder")