# 04_egarch.py - Model EGARCH(1,1)
#
# Exponential GARCH — modeluje asymetrię (leverage effect).
# Logarytmiczna specyfikacja gwarantuje dodatnią wariancję.
# Negatywne szoki mają większy wpływ na zmienność niż pozytywne.
# Metodologia: podział 80/10/10, prognozy 1-step-ahead na zbiorze testowym.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from params import TICKERS, TRAIN_RATIO, VAL_RATIO
from utils.static_split import static_split_garch
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
panels = {}

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} - {ticker}")
    print('='*50)

    result = static_split_garch(
        returns_series=returns[ticker],
        model_config=MODEL_CONFIG,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    forecasts = result['forecasts']
    realized  = result['realized']
    dates     = result['dates']

    metrics = calculate_all_metrics(realized, forecasts)

    print(f"\nWyniki {ticker}:")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.4f}")

    all_results.append({
        'ticker': ticker,
        'model': MODEL_NAME,
        **metrics,
    })

    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecasts,
        'realized': realized,
        'error': forecasts - realized,
    })
    forecast_df.to_parquet(f"results/04_egarch/forecasts_{ticker}.parquet")

    panels[ticker] = {'dates': dates, 'realized': realized, 'forecasts': forecasts}

# --- COMBO: 5 paneli (1 per spółka), σ_t prognoza vs |r_t| realizacja ---
fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 2.6 * len(TICKERS)), sharex=False)
all_sigma_real = np.concatenate([np.sqrt(np.clip(panels[t]['realized'], 0, None)) for t in TICKERS])
y_max = np.percentile(all_sigma_real, 99) * 1.05

for ax, ticker in zip(axes, TICKERS):
    dates    = panels[ticker]['dates']
    real_sig = np.sqrt(np.clip(panels[ticker]['realized'], 0, None))
    pred_sig = np.sqrt(np.clip(panels[ticker]['forecasts'], 0, None))
    ax.plot(dates, real_sig, color='#000000', linewidth=0.6, alpha=0.85, label=r'Realizacja $|r_t|$')
    ax.plot(dates, pred_sig, color='#e74c3c', linewidth=1.1, label=fr'Prognoza $\hat{{\sigma}}_t$ ({MODEL_NAME})')
    ax.set_ylim(0, y_max)
    ax.set_ylabel(r'$\sigma_t$', fontsize=10)
    ax.set_title(ticker, loc='left', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.25)
axes[-1].set_xlabel('Data')
fig.suptitle(f'{MODEL_NAME} — prognoza odchylenia standardowego vs realizacja $|r_t|$ (5 spółek)',
             fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig("charts/04_egarch/forecast_vs_realized_all.png", dpi=150, bbox_inches='tight')
plt.close()

results_df = pd.DataFrame(all_results)
results_df.to_csv("results/04_egarch/metrics_summary.csv", index=False)

print(f"\n{'='*50}")
print(f"PODSUMOWANIE {MODEL_NAME}")
print('='*50)
print(results_df.to_string(index=False))

print(f"\nWyniki zapisane w results/04_egarch/")
print(f"Wykres zbiorczy: charts/04_egarch/forecast_vs_realized_all.png")
