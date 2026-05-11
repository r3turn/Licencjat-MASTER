# 07_gru.py - Model GRU dla prognozowania zmienności
#
# Sieć GRU (Gated Recurrent Unit) - uproszczona alternatywa dla LSTM.
# Zalety: szybsze trenowanie, mniej parametrów, często podobne wyniki.
# GRU ma 2 bramki (reset, update) vs 3 w LSTM (forget, input, output).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from params import (
    TICKERS, INITIAL_TRAIN_SIZE, WINDOW_SIZE, VAL_RATIO, set_seed
)
from utils.gru_utils import walk_forward_gru
from utils.metrics import calculate_all_metrics

# ============================================================
# KONFIGURACJA GRU
# ============================================================

# Architektura sieci
HIDDEN_SIZE = 32      # Rozmiar warstwy ukrytej (32-64 typowo wystarcza)
NUM_LAYERS = 1        # Liczba warstw GRU (1-2)
DROPOUT = 0.1         # Dropout dla regularyzacji

# Trenowanie
EPOCHS = 100          # Max epok (early stopping zazwyczaj zatrzyma wcześniej)
BATCH_SIZE = 32       # Rozmiar batcha
LEARNING_RATE = 0.001 # Learning rate dla Adam
PATIENCE = 15         # Early stopping patience

# Walk-forward
# UWAGA: GRU trenuje się szybciej niż LSTM, ale nadal wolniej niż GARCH
# refit co 250 dni jest rozsądnym kompromisem czas vs świeżość modelu
REFIT_EVERY_GRU = 250

MODEL_NAME = "GRU"

# ============================================================
# SETUP
# ============================================================

# Reproducibility - KLUCZOWE dla sieci neuronowych
set_seed()

# Sprawdź dostępność GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Używam: {DEVICE.upper()}")

# Foldery
os.makedirs("results/07_gru", exist_ok=True)
os.makedirs("charts/07_gru", exist_ok=True)

# Wczytaj dane
returns = pd.read_parquet("data/processed/returns.parquet")
print(f"Wczytano dane: {returns.shape}")
print(f"Okres: {returns.index[0].date()} - {returns.index[-1].date()}")

print(f"\nKonfiguracja GRU:")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Layers: {NUM_LAYERS}")
print(f"  Window size: {WINDOW_SIZE}")
print(f"  Refit every: {REFIT_EVERY_GRU} dni")

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

all_results = []

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} - {ticker}")
    print('='*50)

    # Walk-forward validation
    wf_result = walk_forward_gru(
        returns_series=returns[ticker],
        window_size=WINDOW_SIZE,
        initial_train_size=INITIAL_TRAIN_SIZE,
        refit_every=REFIT_EVERY_GRU,
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

    forecasts = wf_result['forecasts']
    realized = wf_result['realized']
    dates = wf_result['dates']

    # Oblicz metryki
    metrics = calculate_all_metrics(realized, forecasts)

    print(f"\nWyniki {ticker}:")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.4f}")
    print(f"  Model trenowany {wf_result['fit_count']} razy")

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
    forecast_df.to_parquet(f"results/07_gru/forecasts_{ticker}.parquet")

    # ============================================================
    # WYKRESY
    # ============================================================

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
    plt.savefig(f"charts/07_gru/forecast_vs_realized_{ticker}.png", dpi=150)
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
print(f"Wykresy zapisane w charts/07_gru/")
