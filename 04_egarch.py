# 04_egarch.py - Model EGARCH(1,1)
#
# Exponential GARCH - modeluje asymetrię (leverage effect).
# Logarytmiczna specyfikacja gwarantuje dodatnią wariancję.
# Negatywne szoki mają większy wpływ na zmienność niż pozytywne.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from params import TICKERS, INITIAL_TRAIN_SIZE, REFIT_EVERY
from utils.walk_forward import walk_forward_garch
from utils.metrics import calculate_all_metrics

# Foldery
os.makedirs("results/04_egarch", exist_ok=True)
os.makedirs("charts/04_egarch", exist_ok=True)

# Wczytaj dane
returns = pd.read_parquet("data/processed/returns.parquet")
print(f"Wczytano dane: {returns.shape}")
print(f"Okres: {returns.index[0].date()} - {returns.index[-1].date()}")

# Konfiguracja modelu EGARCH(1,1)
MODEL_CONFIG = {
    'vol': 'EGARCH',
    'p': 1,
    'q': 1,
}
MODEL_NAME = "EGARCH(1,1)"

# Wyniki dla wszystkich tickerów
all_results = []

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} - {ticker}")
    print('='*50)

    # Walk-forward validation
    wf_result = walk_forward_garch(
        returns_series=returns[ticker],
        model_config=MODEL_CONFIG,
        initial_train_size=INITIAL_TRAIN_SIZE,
        refit_every=REFIT_EVERY,
    )

    forecasts = wf_result['forecasts']
    realized = wf_result['realized']
    dates = wf_result['dates']

    # Oblicz metryki
    metrics = calculate_all_metrics(realized, forecasts)

    print(f"\nWyniki {ticker}:")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.4f}")
    print(f"  Model fitowany {wf_result['fit_count']} razy")

    # Zapisz wyniki
    result_entry = {
        'ticker': ticker,
        'model': MODEL_NAME,
        **metrics,
        'fit_count': wf_result['fit_count'],
    }
    all_results.append(result_entry)

    # Zapisz prognozy
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecasts,
        'realized': realized,
        'error': forecasts - realized,
    })
    forecast_df.to_parquet(f"results/04_egarch/forecasts_{ticker}.parquet")

    # --- WYKRESY ---

    # 1. Forecast vs Realized
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(dates, realized, label='Realized (r²)', alpha=0.7, linewidth=0.5)
    axes[0].plot(dates, forecasts, label=f'{MODEL_NAME} forecast', alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel('Wariancja (σ²)')
    axes[0].set_title(f'{ticker} - {MODEL_NAME}: Prognoza vs Realizacja')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, forecasts - realized, color='red', alpha=0.5, linewidth=0.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Błąd (forecast - realized)')
    axes[1].set_xlabel('Data')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"charts/04_egarch/forecast_vs_realized_{ticker}.png", dpi=150)
    plt.close()

    # 2. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(realized, forecasts, alpha=0.3, s=5)
    max_val = max(realized.max(), forecasts.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Idealne dopasowanie')

    ax.set_xlabel('Realized (r²)')
    ax.set_ylabel('Forecast (σ²)')
    ax.set_title(f'{ticker} - {MODEL_NAME}: Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"charts/04_egarch/scatter_{ticker}.png", dpi=150)
    plt.close()

# Zapisz podsumowanie
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/04_egarch/metrics_summary.csv", index=False)
print(f"\n{'='*50}")
print(f"PODSUMOWANIE {MODEL_NAME}")
print('='*50)
print(results_df.to_string(index=False))

# Wykres porównawczy
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics_to_plot = ['rmse', 'mae', 'qlike']
titles = ['RMSE', 'MAE', 'QLIKE']

for ax, metric, title in zip(axes, metrics_to_plot, titles):
    ax.bar(results_df['ticker'], results_df[metric], color='darkorange', edgecolor='black')
    ax.set_ylabel(title)
    ax.set_title(f'{MODEL_NAME} - {title}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{MODEL_NAME} - Porównanie metryk między tickerami', fontsize=14)
plt.tight_layout()
plt.savefig("charts/04_egarch/metrics_comparison.png", dpi=150)
plt.close()

print(f"\nWyniki zapisane w results/04_egarch/")
print(f"Wykresy zapisane w charts/04_egarch/")
