import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
import os
from params import TICKERS, ARCH_NLAGS, set_seed

# Reproducibility
set_seed()

# Foldery
os.makedirs("data/processed", exist_ok=True)
os.makedirs("charts/02_preprocessing", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Wczytaj ceny
prices = pd.read_parquet("data/raw/prices.parquet")
print(f"Wczytano ceny: {prices.shape}")

# Log returns
returns = np.log(prices / prices.shift(1)).dropna()
returns.to_parquet("data/processed/returns.parquet")
print(f"Zapisano: data/processed/returns.parquet")

# Squared returns (proxy zmienności)
squared_returns = returns ** 2
squared_returns.to_parquet("data/processed/squared_returns.parquet")
print(f"Zapisano: data/processed/squared_returns.parquet")

# === STATYSTYKI OPISOWE ===
stats_df = pd.DataFrame(index=TICKERS)
stats_df["Mean"] = returns.mean()
stats_df["Std"] = returns.std()
stats_df["Skewness"] = returns.skew()
stats_df["Kurtosis"] = returns.kurtosis()
stats_df["Min"] = returns.min()
stats_df["Max"] = returns.max()

# Jarque-Bera test (normalność)
for ticker in TICKERS:
    jb_stat, jb_pval = stats.jarque_bera(returns[ticker])
    stats_df.loc[ticker, "JB_stat"] = jb_stat
    stats_df.loc[ticker, "JB_pval"] = jb_pval

# ARCH test (heteroskedastyczność)
for ticker in TICKERS:
    arch_stat, arch_pval, _, _ = het_arch(returns[ticker], nlags=ARCH_NLAGS)
    stats_df.loc[ticker, "ARCH_stat"] = arch_stat
    stats_df.loc[ticker, "ARCH_pval"] = arch_pval

stats_df.to_csv("results/02_preprocessing_stats.csv")
print(f"\nStatystyki zapisane: results/02_preprocessing_stats.csv")
print(stats_df.round(4).to_string())

# === WYKRESY ===

# 1. Histogram zwrotów (wszystkie tickery)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, ticker in enumerate(TICKERS):
    ax = axes[i]
    data = returns[ticker]

    ax.hist(data, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='none')

    # Theoretical normal
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', lw=2, label='Rozkład normalny')

    ax.set_title(f"{ticker}")
    ax.set_xlabel("Logarytmiczna stopa zwrotu")
    ax.set_ylabel("Gęstość")
    ax.legend()

axes[-1].axis('off')  # Ukryj 6. subplot (mamy 5 tickerów)
plt.suptitle("Rozkład logarytmicznych stóp zwrotu vs rozkład normalny", fontsize=14)
plt.tight_layout()
plt.savefig("charts/02_preprocessing/returns_histogram.png", dpi=150)
plt.close()

# 2. ACF zwrotów i squared returns (wszystkie tickery)
fig, axes = plt.subplots(len(TICKERS), 2, figsize=(12, 3 * len(TICKERS)))

for i, ticker in enumerate(TICKERS):
    plot_acf(returns[ticker], lags=40, ax=axes[i, 0], title=f"ACF stóp zwrotu ({ticker})")
    plot_acf(squared_returns[ticker], lags=40, ax=axes[i, 1], title=f"ACF kwadratów stóp zwrotu ({ticker})")

plt.suptitle("Autokorelacja: stopy zwrotu vs kwadraty stóp zwrotu", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("charts/02_preprocessing/acf_comparison.png", dpi=150)
plt.close()

# 3. QQ-plot
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, ticker in enumerate(TICKERS):
    stats.probplot(returns[ticker], dist="norm", plot=axes[i])
    axes[i].set_title(f"Wykres Q-Q: {ticker}")
    axes[i].set_xlabel("Kwantyle teoretyczne")
    axes[i].set_ylabel("Kwantyle empiryczne")

axes[-1].axis('off')
plt.suptitle("Wykres kwantyl-kwantyl vs rozkład normalny", fontsize=14)
plt.tight_layout()
plt.savefig("charts/02_preprocessing/qq_plots.png", dpi=150)
plt.close()

# 4. Time series zwrotów
fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 2.5 * len(TICKERS)), sharex=True)

for i, ticker in enumerate(TICKERS):
    axes[i].plot(returns.index, returns[ticker], linewidth=0.5, color='steelblue')
    axes[i].set_ylabel(ticker)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Data")
plt.suptitle("Logarytmiczne stopy zwrotu (szereg czasowy)", fontsize=14)
plt.tight_layout()
plt.savefig("charts/02_preprocessing/returns_timeseries.png", dpi=150)
plt.close()

print(f"\nWykresy zapisane w charts/02_preprocessing/")
