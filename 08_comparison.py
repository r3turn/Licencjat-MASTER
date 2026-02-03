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
}

# 1. Wykres słupkowy dla każdej metryki (grouped bar)
for metric in METRICS:
    label = METRICS_LABELS[metric]
    pivot = create_comparison_table(metric)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(TICKERS))
    width = 0.2
    multiplier = 0

    for model in pivot.columns:
        offset = width * multiplier
        bars = ax.bar(x + offset, pivot[model], width, label=model,
                      color=colors.get(model, 'gray'), edgecolor='black', linewidth=0.5)
        multiplier += 1

    ax.set_xlabel('Ticker')
    ax.set_ylabel(label)
    ax.set_title(f'Porównanie modeli - {label}')
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(TICKERS)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"charts/08_comparison/{metric}_comparison.png", dpi=150)
    plt.close()

# 2. Wykres radarowy (średnie metryki znormalizowane)
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

# 3. Heatmapa rankingów
fig, ax = plt.subplots(figsize=(10, 6))

# Oblicz rankingi dla każdego tickera i metryki
rankings = []
for ticker in TICKERS:
    for metric in METRICS:
        pivot = create_comparison_table(metric)
        row = pivot.loc[ticker]
        lower_better = METRICS_LOWER_BETTER[metric]

        if lower_better:
            rank = row.rank()
        else:
            rank = row.rank(ascending=False)

        for model, r in rank.items():
            rankings.append({
                'ticker': ticker,
                'metric': METRICS_LABELS[metric],
                'model': model,
                'rank': r
            })

rank_df = pd.DataFrame(rankings)
rank_pivot = rank_df.groupby('model')['rank'].mean().sort_values()

# Bar chart średnich rankingów
ax.barh(rank_pivot.index, rank_pivot.values, color=[colors.get(m, 'gray') for m in rank_pivot.index],
        edgecolor='black')
ax.set_xlabel('Średni ranking (1 = najlepszy)')
ax.set_title('Średni ranking modeli')
ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Najlepszy możliwy')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("charts/08_comparison/ranking_comparison.png", dpi=150)
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
