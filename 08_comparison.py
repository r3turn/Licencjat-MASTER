# 08_comparison.py - Porównanie wszystkich modeli
#
# Zbiera metryki ze wszystkich modeli i tworzy:
# - Tabelę porównawczą w formacie Markdown
# - Wykresy porównawcze
# - Ranking modeli

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from params import TICKERS

# ============================================================
# KONFIGURACJA
# ============================================================

# Modele do porównania (folder -> nazwa wyświetlana)
MODELS = {
    '03_garch': 'GARCH(1,1)',
    '04_egarch': 'EGARCH(1,1)',
    '05_gjr_garch': 'GJR-GARCH(1,1,1)',
    '06_lstm': 'LSTM',
    '07_gru': 'GRU',
}

# Metryki do porównania
METRICS = ['rmse', 'mae', 'qlike']
METRICS_LABELS = {
    'rmse': 'RMSE',
    'mae': 'MAE',
    'qlike': 'QLIKE',
}

# Kierunek (True = mniejsze lepsze)
METRICS_LOWER_BETTER = {
    'rmse': True,
    'mae': True,
    'qlike': True,
}

# Foldery
os.makedirs("results/08_comparison", exist_ok=True)
os.makedirs("charts/08_comparison", exist_ok=True)

# ============================================================
# WCZYTAJ DANE
# ============================================================

print("Wczytuję wyniki modeli...")

all_data = []

for folder, model_name in MODELS.items():
    filepath = f"results/{folder}/metrics_summary.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['model'] = model_name  # nadpisz nazwę dla spójności
        all_data.append(df)
        print(f"  {model_name}: {len(df)} tickerów")
    else:
        print(f"  {model_name}: BRAK DANYCH ({filepath})")

if not all_data:
    print("\nBrak danych do porównania!")
    exit(1)

# Połącz wszystkie dane
combined = pd.concat(all_data, ignore_index=True)
print(f"\nŁącznie: {len(combined)} wierszy ({len(combined['model'].unique())} modeli)")

# ============================================================
# TABELE PORÓWNAWCZE
# ============================================================

def create_comparison_table(metric):
    """Tworzy tabelę porównawczą dla jednej metryki."""
    pivot = combined.pivot(index='ticker', columns='model', values=metric)
    # Uporządkuj kolumny według MODELS
    cols = [m for m in MODELS.values() if m in pivot.columns]
    pivot = pivot[cols]
    return pivot


def highlight_best(pivot, lower_better=True):
    """Dodaje oznaczenie najlepszego wyniku w każdym wierszu."""
    result = pivot.copy().astype(str)
    for idx in pivot.index:
        row = pivot.loc[idx]
        if lower_better:
            best_val = row.min()
        else:
            best_val = row.max()
        best_col = row[row == best_val].index[0]
        result.loc[idx, best_col] = f"**{pivot.loc[idx, best_col]:.6f}**"
    return result


# ============================================================
# GENERUJ MARKDOWN
# ============================================================

print("\nGeneruję raport Markdown...")

md_lines = []
md_lines.append("# Porównanie modeli prognozowania zmienności\n")
md_lines.append(f"*Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

# Podsumowanie
md_lines.append("## Podsumowanie\n")
md_lines.append(f"- **Modele:** {', '.join(MODELS.values())}")
md_lines.append(f"- **Tickery:** {', '.join(TICKERS)}")
md_lines.append(f"- **Metryki:** {', '.join(METRICS_LABELS.values())}\n")

# Tabele dla każdej metryki
for metric in METRICS:
    label = METRICS_LABELS[metric]
    lower_better = METRICS_LOWER_BETTER[metric]

    md_lines.append(f"## {label}\n")
    md_lines.append(f"*{'Mniejsze = lepsze' if lower_better else 'Większe = lepsze'}*\n")

    pivot = create_comparison_table(metric)

    # Formatowanie liczb
    if metric == 'qlike':
        formatted = pivot.map(lambda x: f"{x:.4f}")
    else:
        formatted = pivot.map(lambda x: f"{x:.6f}")

    # Dodaj średnią
    mean_row = pivot.mean()
    if metric == 'qlike':
        mean_formatted = mean_row.map(lambda x: f"{x:.4f}")
    else:
        mean_formatted = mean_row.map(lambda x: f"{x:.6f}")

    # Markdown table
    header = "| Ticker | " + " | ".join(pivot.columns) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(pivot.columns)) + "|"

    md_lines.append(header)
    md_lines.append(separator)

    for ticker in pivot.index:
        row_vals = []
        best_val = pivot.loc[ticker].min() if lower_better else pivot.loc[ticker].max()
        for col in pivot.columns:
            val = pivot.loc[ticker, col]
            fmt = f"{val:.4f}" if metric == 'qlike' else f"{val:.6f}"
            if val == best_val:
                fmt = f"**{fmt}**"  # pogrubienie najlepszego
            row_vals.append(fmt)
        md_lines.append(f"| {ticker} | " + " | ".join(row_vals) + " |")

    # Średnia
    mean_vals = []
    best_mean = mean_row.min() if lower_better else mean_row.max()
    for col in pivot.columns:
        val = mean_row[col]
        fmt = f"{val:.4f}" if metric == 'qlike' else f"{val:.6f}"
        if val == best_mean:
            fmt = f"**{fmt}**"
        mean_vals.append(fmt)
    md_lines.append(f"| **Średnia** | " + " | ".join(mean_vals) + " |")
    md_lines.append("")

# Ranking ogólny
md_lines.append("## Ranking ogólny\n")
md_lines.append("Liczba \"zwycięstw\" (najlepsza metryka dla tickera):\n")

wins = {model: 0 for model in MODELS.values()}

for metric in METRICS:
    pivot = create_comparison_table(metric)
    lower_better = METRICS_LOWER_BETTER[metric]

    for ticker in pivot.index:
        row = pivot.loc[ticker]
        if lower_better:
            winner = row.idxmin()
        else:
            winner = row.idxmax()
        wins[winner] += 1

# Sortuj po liczbie wygranych
wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)

md_lines.append("| Model | Wygrane |")
md_lines.append("|-------|---------|")
for model, count in wins_sorted:
    total = len(TICKERS) * len(METRICS)
    pct = count / total * 100
    md_lines.append(f"| {model} | {count}/{total} ({pct:.1f}%) |")
md_lines.append("")

# Zapisz Markdown
md_content = "\n".join(md_lines)
md_path = "results/08_comparison/comparison_report.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f"Zapisano: {md_path}")

# Zapisz też CSV ze wszystkimi danymi
combined.to_csv("results/08_comparison/all_metrics.csv", index=False)
print("Zapisano: results/08_comparison/all_metrics.csv")

# ============================================================
# WYKRESY
# ============================================================

print("\nGeneruję wykres metryk...")

# Kolejność modeli pogrupowana wg rodziny (GARCH-y najpierw, sieci dalej)
model_order = [
    'GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1,1)',
    'LSTM', 'GRU',
]
model_order = [m for m in model_order if m in MODELS.values()]

# Kolory wg rodziny modeli (GARCH = niebieskie tony, NN = pomarańczowo-czerwone)
family_colors = {
    'GARCH(1,1)':       '#2c5f8d',
    'EGARCH(1,1)':      '#4a86b8',
    'GJR-GARCH(1,1,1)': '#73a9d4',
    'LSTM':             '#c0392b',
    'GRU':              '#e67e22',
}
bar_colors = [family_colors[m] for m in model_order]

# Markery per ticker (kropki rzeczywistych obserwacji)
ticker_markers = ['o', 's', '^', 'D', 'v']
ticker_color = '#1a1a1a'

# Skalowanie metryk żeby etykiety były czytelne
metric_scale = {
    'rmse':  (1e3, r'RMSE  ($\times 10^{-3}$)'),
    'mae':   (1e4, r'MAE  ($\times 10^{-4}$)'),
    'qlike': (1.0, 'QLIKE'),
}
metric_fmt = {'rmse': '{:.4f}', 'mae': '{:.3f}', 'qlike': '{:.4f}'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))

for ax_i, metric in enumerate(METRICS):
    ax = axes[ax_i]
    pivot = create_comparison_table(metric)

    scale, ylabel = metric_scale[metric]
    means = np.array([pivot[m].mean() for m in model_order]) * scale
    per_ticker_vals = {m: pivot[m].values * scale for m in model_order}

    x = np.arange(len(model_order))
    bars = ax.bar(x, means, color=bar_colors,
                  edgecolor='#222222', linewidth=0.7, width=0.62,
                  alpha=0.92, zorder=2)

    # Pogrubiona ramka dla najlepszego (najmniejsza wartość)
    best_idx = int(np.argmin(means))
    bars[best_idx].set_linewidth(2.6)
    bars[best_idx].set_edgecolor('#000000')

    rng_y = means.max() - means.min()
    label_offset = rng_y * 0.10 if metric != 'qlike' else abs(rng_y) * 0.10

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + label_offset,
                metric_fmt[metric].format(val),
                ha='center', va='bottom', fontsize=9.5, color='#111111',
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=25, ha='right', fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(METRICS_LABELS[metric], fontsize=12.5, fontweight='bold', pad=8)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

    # Linia odniesienia: najlepszy model (przerywana)
    ax.axhline(y=means.min(), color='#444444', linestyle=':', linewidth=0.9, alpha=0.7, zorder=1)

    # Y-lim: ZOOM na różnice między modelami (margines proporcjonalny do rozrzutu średnich)
    y_low  = means.min() - rng_y * 0.35
    y_high = means.max() + rng_y * 0.55
    ax.set_ylim(y_low, y_high)

# Legenda (wspólna, pod wykresami)
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='#4a86b8', edgecolor='#222222', label='Modele rodziny GARCH'),
    Patch(facecolor='#c0392b', edgecolor='#222222', label='Sieci rekurencyjne (LSTM, GRU)'),
    Patch(facecolor='none',    edgecolor='#000000', linewidth=2.6, label='Najlepszy w danej metryce'),
]
fig.legend(handles=legend_elems, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9.5, frameon=False)

fig.suptitle('Porównanie modeli prognozowania zmienności — średnie metryki na zbiorze testowym (n = 5 spółek)',
             fontsize=12.5, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig("charts/08_comparison/metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("Zapisano wykres: charts/08_comparison/metrics_comparison.png")

# ============================================================
# PODSUMOWANIE
# ============================================================

print(f"\n{'='*50}")
print("PODSUMOWANIE PORÓWNANIA")
print('='*50)

print("\nŚrednie metryki:")
summary = combined.groupby('model')[METRICS].mean()
# Sortuj według QLIKE (główna metryka w literaturze volatility)
summary = summary.sort_values('qlike')
print(summary.to_string())

print(f"\n{'='*50}")
print(f"Najlepszy model według QLIKE: {summary['qlike'].idxmin()}")
print(f"Najlepszy model według RMSE:  {summary['rmse'].idxmin()}")
print(f"Najlepszy model według MAE:   {summary['mae'].idxmin()}")
print('='*50)

print(f"\nRaport: results/08_comparison/comparison_report.md")
