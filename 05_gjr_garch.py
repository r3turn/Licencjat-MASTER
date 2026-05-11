# 05_gjr_garch.py - Model GJR-GARCH(1,1,1)
#
# GJR-GARCH (Glosten-Jagannathan-Runkle) - alternatywny model asymetrii.
# Używa wskaźnika I(ε<0) do modelowania leverage effect.
# Parametr 'o' kontroluje asymetrię.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from params import TICKERS, INITIAL_TRAIN_SIZE, REFIT_EVERY
from utils.walk_forward import walk_forward_garch
from utils.metrics import calculate_all_metrics

# Foldery
os.makedirs("results/05_gjr_garch", exist_ok=True)
os.makedirs("charts/05_gjr_garch", exist_ok=True)

# Wczytaj dane
returns = pd.read_parquet("data/processed/returns.parquet")
print(f"Wczytano dane: {returns.shape}")
print(f"Okres: {returns.index[0].date()} - {returns.index[-1].date()}")

# Konfiguracja modelu GJR-GARCH(1,1,1)
# W bibliotece arch: o=1 oznacza asymetrię (leverage term)
MODEL_CONFIG = {
    'vol': 'GARCH',
    'p': 1,
    'o': 1,  # GJR term (asymetria)
    'q': 1,
}
MODEL_NAME = "GJR-GARCH(1,1,1)"

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
    forecast_df.to_parquet(f"results/05_gjr_garch/forecasts_{ticker}.parquet")

    # --- WYKRESY ---

    # Stopy zwrotu + pasma ±2σ
    ret_aligned = returns[ticker].reindex(pd.to_datetime(dates)).values
    sigma = np.sqrt(forecasts)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, ret_aligned, color='steelblue', linewidth=0.5, alpha=0.8, label='Stopy zwrotu')
    ax.plot(dates,  2 * sigma, color='red', linewidth=0.9, label=f'±2σ ({MODEL_NAME})')
    ax.plot(dates, -2 * sigma, color='red', linewidth=0.9)
    ax.fill_between(dates, -2 * sigma, 2 * sigma, alpha=0.15, color='red')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Stopa zwrotu / ±2σ')
    ax.set_xlabel('Data')
    ax.set_title(f'{ticker} — {MODEL_NAME}: stopy zwrotu i pasma zmienności ±2σ')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"charts/05_gjr_garch/forecast_vs_realized_{ticker}.png", dpi=150)
    plt.close()

# Zapisz podsumowanie
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/05_gjr_garch/metrics_summary.csv", index=False)
print(f"\n{'='*50}")
print(f"PODSUMOWANIE {MODEL_NAME}")
print('='*50)
print(results_df.to_string(index=False))

print(f"\nWyniki zapisane w results/05_gjr_garch/")
print(f"Wykresy zapisane w charts/05_gjr_garch/")
