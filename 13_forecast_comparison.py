# 13_forecast_comparison.py — Wykresy porównawcze prognoz wszystkich modeli
#
# Generuje TRZY wykresy:
#  1. forecast_comparison_all.png — 5 paneli (po jednym na ticker), każdy panel
#     pokazuje realizację |r_t| + prognozy wszystkich 5 modeli.
#  2. forecast_comparison_NVDA_stacked.png — 5 paneli (po jednym na model)
#     dla NVDA: czytelne porównanie zachowania każdego modelu na tej samej spółce.
#  3. forecast_vs_realized_scatter.png — 5 paneli (po jednym na model), scatter
#     σ_t prognoza vs |r_t| realizacja, kropki kolorowane wg tickera, linia 45°.
#
# WAŻNE: oś w jednostkach odchylenia standardowego (σ_t = √σ²_t),
# realizacja jako |r_t| = √r²_t. Pierwiastkowanie utrzymuje czytelność skali
# (r² ma pojedyncze wybicia w dni covidowe).
#
# Wejście: forecasts_{ticker}.parquet z 03_garch/, 04_egarch/, 05_gjr_garch/,
#          06_lstm/, 07_gru/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from params import TICKERS

OUT_DIR = "charts/13_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("GARCH(1,1)",       "results/03_garch/forecasts_{ticker}.parquet",     "#1f77b4"),
    ("EGARCH(1,1)",      "results/04_egarch/forecasts_{ticker}.parquet",    "#ff7f0e"),
    ("GJR-GARCH(1,1,1)", "results/05_gjr_garch/forecasts_{ticker}.parquet", "#2ca02c"),
    ("LSTM",             "results/06_lstm/forecasts_{ticker}.parquet",      "#9467bd"),
    ("GRU",              "results/07_gru/forecasts_{ticker}.parquet",       "#17becf"),
]

fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 2.8 * len(TICKERS)), sharex=False)

# Wspólna skala Y: percentyl 99 z realizacji ze wszystkich spółek
all_real_sigma = []
for ticker in TICKERS:
    df0 = pd.read_parquet(MODELS[0][1].format(ticker=ticker))
    all_real_sigma.append(np.sqrt(np.clip(df0["realized"].values, 0, None)))
y_max = np.percentile(np.concatenate(all_real_sigma), 99) * 1.05

for ax, ticker in zip(axes, TICKERS):
    print(f"  Panel {ticker}...")

    # Realizacja (z dowolnego pliku — wszystkie modele mają tę samą serię)
    df_real = pd.read_parquet(MODELS[0][1].format(ticker=ticker))
    df_real["date"] = pd.to_datetime(df_real["date"])
    df_real = df_real.sort_values("date")
    real_sig = np.sqrt(np.clip(df_real["realized"].values, 0, None))
    ax.plot(df_real["date"].values, real_sig, color="#000000", linewidth=0.55,
            alpha=0.85, label=r"Realizacja $|r_t|$")

    # Prognozy 5 modeli
    for model_name, path_tpl, color in MODELS:
        df = pd.read_parquet(path_tpl.format(ticker=ticker))
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        pred_sig = np.sqrt(np.clip(df["forecast"].values, 0, None))
        ax.plot(df["date"].values, pred_sig, color=color, linewidth=1.0,
                alpha=0.85, label=model_name)

    ax.set_ylim(0, y_max)
    ax.set_ylabel(r"$\sigma_t$", fontsize=10)
    ax.set_title(ticker, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)

# Wspólna legenda nad pierwszym panelem
axes[0].legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
axes[-1].set_xlabel("Data", fontsize=10)

fig.suptitle(
    r"Porównanie prognoz $\hat{\sigma}_t$ wszystkich modeli vs realizacja $|r_t|$ (5 spółek)",
    fontsize=12, fontweight="bold", y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.985])
out_path = f"{OUT_DIR}/forecast_comparison_all.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {out_path}")

# ============================================================
# WYKRES 2: NVDA STACKED — 5 paneli, każdy panel = jeden model
# ============================================================

print("\nWykres 2 — NVDA stacked (5 modeli, 1 spółka)...")

STACK_TICKER = "NVDA"
fig, axes = plt.subplots(len(MODELS), 1, figsize=(13, 1.95 * len(MODELS)), sharex=True)

# Wspólna skala Y dla wszystkich paneli (NVDA)
df0 = pd.read_parquet(MODELS[0][1].format(ticker=STACK_TICKER))
real_sig_nvda = np.sqrt(np.clip(df0["realized"].values, 0, None))
y_max_nvda = np.percentile(real_sig_nvda, 99) * 1.05

for ax, (model_name, path_tpl, color) in zip(axes, MODELS):
    df = pd.read_parquet(path_tpl.format(ticker=STACK_TICKER))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    real_sig = np.sqrt(np.clip(df["realized"].values, 0, None))
    pred_sig = np.sqrt(np.clip(df["forecast"].values, 0, None))

    ax.plot(df["date"].values, real_sig, color="#000000", linewidth=0.55,
            alpha=0.85, label=r"Realizacja $|r_t|$")
    ax.plot(df["date"].values, pred_sig, color=color, linewidth=1.3,
            label=fr"Prognoza $\hat{{\sigma}}_t$ ({model_name})")
    ax.set_ylim(0, y_max_nvda)
    ax.set_ylabel(r"$\sigma_t$", fontsize=10)
    ax.set_title(model_name, loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Data", fontsize=10)
fig.suptitle(
    f"{STACK_TICKER} — porównanie prognoz $\\hat{{\\sigma}}_t$ pięciu modeli na tej samej spółce",
    fontsize=12, fontweight="bold", y=0.997
)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out_path = f"{OUT_DIR}/forecast_comparison_NVDA_stacked.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {out_path}")

# ============================================================
# WYKRES 3: SCATTER σ_t prognoza vs |r_t| realizacja (5 modeli × 5 spółek)
# ============================================================

print("\nWykres 3 — scatter prognoza vs realizacja...")

TICKER_COLORS = {
    "AAPL": "#1f77b4",
    "NVDA": "#d62728",
    "JPM":  "#2ca02c",
    "XOM":  "#9467bd",
    "SCCO": "#ff7f0e",
}

# Ustal wspólną maksymalną wartość osi (99 percentyl realizacji ze wszystkich tickerów × modeli)
all_real_max = []
for model_name, path_tpl, _ in MODELS:
    for ticker in TICKERS:
        df = pd.read_parquet(path_tpl.format(ticker=ticker))
        all_real_max.append(np.sqrt(np.clip(df["realized"].values, 0, None)))
axis_max = np.percentile(np.concatenate(all_real_max), 99) * 1.05

# Układ: 2 wiersze × 3 kolumny (5 modeli + 1 pusty panel na legendę)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes_flat = axes.flatten()

for ax, (model_name, path_tpl, _) in zip(axes_flat[:len(MODELS)], MODELS):
    for ticker in TICKERS:
        df = pd.read_parquet(path_tpl.format(ticker=ticker))
        real_sig = np.sqrt(np.clip(df["realized"].values, 0, None))
        pred_sig = np.sqrt(np.clip(df["forecast"].values, 0, None))
        ax.scatter(real_sig, pred_sig,
                   s=12, alpha=0.5,
                   color=TICKER_COLORS.get(ticker, "#999999"),
                   edgecolor="none")

    ax.plot([0, axis_max], [0, axis_max], color="#000000",
            linewidth=1.1, linestyle="--", alpha=0.75)

    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(model_name, fontsize=11.5, fontweight="bold")
    ax.set_xlabel(r"Realizacja $|r_t|$", fontsize=10)
    ax.set_ylabel(r"Prognoza $\hat{\sigma}_t$", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=9)

# Legenda w ostatnim panelu (zamiast pustki)
from matplotlib.lines import Line2D
legend_ax = axes_flat[-1]
legend_ax.axis('off')
legend_elems = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=TICKER_COLORS[t], markersize=11,
           label=t) for t in TICKERS
]
legend_elems.append(
    Line2D([0], [0], color='#000000', linestyle='--', linewidth=1.4,
           label=r"Prognoza idealna ($\hat{\sigma}_t = |r_t|$)")
)
legend_ax.legend(handles=legend_elems, loc='center', fontsize=11,
                 frameon=True, title='Spółka', title_fontsize=11.5,
                 labelspacing=1.1)

fig.suptitle("Prognoza vs realizacja — kalibracja modeli na zbiorze testowym",
             fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = f"{OUT_DIR}/forecast_vs_realized_scatter.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {out_path}")

print("\nGotowe.")
