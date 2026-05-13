# 07_gru.py - Model GRU dla prognozowania zmienności
#
# Sieć GRU (Gated Recurrent Unit) — uproszczona alternatywa dla LSTM.
# Ma 2 bramki (reset, update) zamiast 3 — mniej parametrów, szybsze trenowanie.
# Metodologia: podział 80/10/10, sliding window L=30 na zbiorze testowym.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from params import TICKERS, TRAIN_RATIO, VAL_RATIO, WINDOW_SIZE, set_seed
from utils.gru_utils import static_split_gru
from utils.metrics import calculate_all_metrics

# ============================================================
# KONFIGURACJA GRU
# ============================================================

# Architektura sieci
HIDDEN_SIZE   = 32      # Rozmiar warstwy ukrytej
NUM_LAYERS    = 1       # Liczba warstw GRU
DROPOUT       = 0.1     # Dropout dla regularyzacji

# Trenowanie
EPOCHS        = 100     # Max epok (early stopping zazwyczaj zatrzyma wcześniej)
BATCH_SIZE    = 32      # Rozmiar batcha
LEARNING_RATE = 0.001   # Learning rate dla Adam
PATIENCE      = 15      # Early stopping patience

MODEL_NAME = "GRU"

# ============================================================
# SETUP
# ============================================================

set_seed()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Używam: {DEVICE.upper()}")

os.makedirs("results/07_gru", exist_ok=True)
os.makedirs("charts/07_gru", exist_ok=True)

returns = pd.read_parquet("data/processed/returns.parquet")
print(f"Wczytano dane: {returns.shape}")
print(f"Okres: {returns.index[0].date()} - {returns.index[-1].date()}")

print(f"\nKonfiguracja GRU:")
print(f"  Hidden size: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
print(f"  Window size (L): {WINDOW_SIZE}, Epochs: {EPOCHS}, Patience: {PATIENCE}")

# ============================================================
# EWALUACJA 80/10/10
# ============================================================

all_results = []
panels = {}

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} - {ticker}")
    print('='*50)

    result = static_split_gru(
        returns_series=returns[ticker],
        window_size=WINDOW_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        device=DEVICE,
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
    forecast_df.to_parquet(f"results/07_gru/forecasts_{ticker}.parquet")

    panels[ticker] = {'dates': dates, 'realized': realized, 'forecasts': forecasts}

# ============================================================
# WYKRES ZBIORCZY (5 paneli, σ_t prognoza vs |r_t| realizacja)
# ============================================================

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
plt.savefig("charts/07_gru/forecast_vs_realized_all.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# PODSUMOWANIE
# ============================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv("results/07_gru/metrics_summary.csv", index=False)

print(f"\n{'='*50}")
print(f"PODSUMOWANIE {MODEL_NAME}")
print('='*50)
print(results_df.to_string(index=False))

print(f"\nWyniki zapisane w results/07_gru/")
print(f"Wykres zbiorczy: charts/07_gru/forecast_vs_realized_all.png")
