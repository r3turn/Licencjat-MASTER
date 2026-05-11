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

print("\nGeneruję wykresy...")

# Kolory dla modeli
colors = {
    'GARCH(1,1)': 'steelblue',
    'EGARCH(1,1)': 'darkorange',
    'GJR-GARCH(1,1,1)': 'forestgreen',
    'LSTM': 'purple',
    'GRU': 'teal',
}

# 1. Kombinowany wykres NN: LSTM + GRU w stylu ±2σ (NVDA)
import matplotlib.dates as mdates

CHART_TICKER = "NVDA"
covid = pd.Timestamp('2020-03-16')
returns_all = pd.read_parquet("data/processed/returns.parquet")

lstm_df = pd.read_parquet(f"results/06_lstm/forecasts_{CHART_TICKER}.parquet")
gru_df  = pd.read_parquet(f"results/07_gru/forecasts_{CHART_TICKER}.parquet")

for df in (lstm_df, gru_df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

idx_nn = lstm_df.index.intersection(gru_df.index)
lstm_df = lstm_df.loc[idx_nn]; gru_df = gru_df.loc[idx_nn]
dates_nn = lstm_df.index

ret_nn      = returns_all[CHART_TICKER].reindex(dates_nn).values
lstm_sigma  = np.sqrt(lstm_df['forecast'].values)
gru_sigma   = np.sqrt(gru_df['forecast'].values)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.subplots_adjust(hspace=0.08)

# Stopy zwrotu
ax1.plot(dates_nn, ret_nn, color='steelblue', lw=0.5, alpha=0.8, label='Stopy zwrotu', zorder=2)

# Pasma ±2σ — LSTM
ax1.plot(dates_nn,  2 * lstm_sigma, color='crimson',    lw=1.0, label='±2σ LSTM', zorder=3)
ax1.plot(dates_nn, -2 * lstm_sigma, color='crimson',    lw=1.0, zorder=3)

# Pasma ±2σ — GRU
ax1.plot(dates_nn,  2 * gru_sigma, color='darkorange', lw=1.0, linestyle='--', label='±2σ GRU', zorder=3)
ax1.plot(dates_nn, -2 * gru_sigma, color='darkorange', lw=1.0, linestyle='--', zorder=3)

ax1.axhline(0, color='black', lw=0.5)
ax1.axvline(x=covid, color='gray', linestyle=':', lw=1.2, alpha=0.8)
ax1.text(covid, float(np.nanmax(np.abs(ret_nn))) * 0.85,
         'COVID\n2020', fontsize=8, color='gray', ha='center')

ax1.set_ylabel('Stopa zwrotu / ±2σ', fontsize=11)
ax1.set_title(f'{CHART_TICKER} — LSTM i GRU: stopy zwrotu i pasma zmienności ±2σ', fontsize=13)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.25)

# Dolny panel: różnica σ_LSTM − σ_GRU
diff_sigma = (lstm_sigma - gru_sigma) * 1e4
diff_colors = ['crimson' if d >= 0 else 'darkorange' for d in diff_sigma]
ax2.bar(dates_nn, diff_sigma, width=1, color=diff_colors, alpha=0.7)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('σ_LSTM − σ_GRU\n(×10⁴)', fontsize=9)
ax2.set_xlabel('Data', fontsize=11)
ax2.grid(True, alpha=0.25)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig("charts/08_comparison/nn_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Wykres radarowy (średnie metryki znormalizowane)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Normalizacja metryk (0-1, gdzie 1 = najgorszy)
normalized = {}
for metric in METRICS:
    pivot = create_comparison_table(metric)
    means = pivot.mean()
    # Normalizuj do 0-1
    min_val, max_val = means.min(), means.max()
    if max_val > min_val:
        normalized[metric] = (means - min_val) / (max_val - min_val)
    else:
        normalized[metric] = means * 0

# Przygotuj dane do wykresu
angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
angles += angles[:1]  # zamknij wykres

for model in MODELS.values():
    if model not in normalized[METRICS[0]].index:
        continue
    values = [normalized[m][model] for m in METRICS]
    values += values[:1]  # zamknij
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors.get(model, 'gray'))
    ax.fill(angles, values, alpha=0.1, color=colors.get(model, 'gray'))

ax.set_xticks(angles[:-1])
ax.set_xticklabels([METRICS_LABELS[m] for m in METRICS])
ax.set_title('Porównanie modeli (znormalizowane)\nMniejszy obszar = lepszy model')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig("charts/08_comparison/radar_comparison.png", dpi=150)
plt.close()

# 3. Ranking: pogrupowany wykres słupkowy (średnia ranga wg metryki)
model_list = list(MODELS.values())
n_models   = len(model_list)

# Oblicz średnią rangę każdego modelu dla każdej metryki (uśrednione po tickerach)
metric_avg_ranks = {}   # metric -> {model: avg_rank}
overall_avg      = {model: 0.0 for model in model_list}

for metric in METRICS:
    pivot = create_comparison_table(metric)
    metric_avg_ranks[metric] = {}
    for model in model_list:
        ranks = [pivot.loc[ticker].rank()[model] for ticker in TICKERS]
        metric_avg_ranks[metric][model] = np.mean(ranks)
        overall_avg[model] += np.mean(ranks)

for model in model_list:
    overall_avg[model] /= len(METRICS)

# Sortuj modele od najlepszego (najniższy overall avg rank)
model_labels_r = sorted(model_list, key=lambda m: overall_avg[m])

metric_colors = {
    'rmse':  '#2166ac',
    'mae':   '#d6604d',
    'qlike': '#1a9641',
}

bar_h = 0.22
y     = np.arange(len(model_labels_r))

fig, ax = plt.subplots(figsize=(9, 4.5))

for k, metric in enumerate(METRICS):
    offset = (k - 1) * bar_h
    vals   = [metric_avg_ranks[metric][m] for m in model_labels_r]
    ax.barh(y + offset, vals, bar_h * 0.92,
            label=METRICS_LABELS[metric],
            color=metric_colors[metric],
            alpha=0.85, edgecolor='white', linewidth=0.4)

# Pionowa linia "idealna ranga 1"
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, lw=1)

ax.set_yticks(y)
ax.set_yticklabels(model_labels_r, fontsize=11)
ax.set_xlabel('Średnia pozycja w rankingu (1 = najlepszy)', fontsize=10)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlim(0.5, 5.5)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
ax.set_title('Średnia pozycja w rankingu według metryki\n(uśredniono po 5 spółkach; modele posortowane od najlepszego)',
             fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("charts/08_comparison/ranking_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("Zapisano wykresy w charts/08_comparison/")

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
