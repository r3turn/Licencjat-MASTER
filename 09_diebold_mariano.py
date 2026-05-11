# 09_diebold_mariano.py - Test Diebolda-Mariano
#
# Formalny test statystyczny porównujący dokładność prognoz modeli.
# H0: Oba modele mają równą dokładność
# H1: Modele różnią się dokładnością
#
# Test DM jest standardem w literaturze forecasting (Diebold & Mariano, 1995).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

from params import TICKERS
from utils.metrics import diebold_mariano_test

# ============================================================
# KONFIGURACJA
# ============================================================

# Modele do porównania (folder -> nazwa)
MODELS = {
    '03_garch': 'GARCH(1,1)',
    '04_egarch': 'EGARCH(1,1)',
    '05_gjr_garch': 'GJR-GARCH(1,1,1)',
    '06_lstm': 'LSTM',
    '07_gru': 'GRU',
}

# Poziom istotności
ALPHA = 0.05

# Folder wyników
os.makedirs("results/09_diebold_mariano", exist_ok=True)
os.makedirs("charts/09_diebold_mariano", exist_ok=True)

# ============================================================
# WCZYTAJ PROGNOZY
# ============================================================

print("Wczytuję prognozy modeli...")

# Struktura: {ticker: {model: DataFrame}}
all_forecasts = {}

for ticker in TICKERS:
    all_forecasts[ticker] = {}

    for folder, model_name in MODELS.items():
        filepath = f"results/{folder}/forecasts_{ticker}.parquet"
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            all_forecasts[ticker][model_name] = df
            print(f"  {ticker} - {model_name}: {len(df)} obserwacji")
        else:
            print(f"  {ticker} - {model_name}: BRAK DANYCH")

# ============================================================
# TESTY DIEBOLD-MARIANO
# ============================================================

print("\n" + "="*60)
print("TESTY DIEBOLD-MARIANO")
print("="*60)

# Wyniki dla każdego tickera
all_results = []

for ticker in TICKERS:
    print(f"\n--- {ticker} ---")

    models_data = all_forecasts[ticker]
    model_names = list(models_data.keys())

    # Znajdź wspólny zakres dat (wszystkie modele muszą mieć te same daty)
    # Użyj pierwszego modelu jako referencji
    ref_model = model_names[0]
    ref_df = models_data[ref_model]

    # Dla każdej pary modeli
    for model1, model2 in combinations(model_names, 2):
        df1 = models_data[model1]
        df2 = models_data[model2]

        # Wyrównaj długość (weź krótszą)
        min_len = min(len(df1), len(df2))

        # Błędy prognozy (forecast - realized)
        errors1 = df1['forecast'].values[-min_len:] - df1['realized'].values[-min_len:]
        errors2 = df2['forecast'].values[-min_len:] - df2['realized'].values[-min_len:]

        # Test DM
        dm_stat, p_value = diebold_mariano_test(errors1, errors2, h=1)

        # Interpretacja
        if p_value < ALPHA:
            if dm_stat > 0:
                winner = model2
                interpretation = f"{model2} lepszy"
            else:
                winner = model1
                interpretation = f"{model1} lepszy"
            significant = "TAK"
        else:
            winner = "brak"
            interpretation = "brak różnicy"
            significant = "NIE"

        # RMSE dla kontekstu
        rmse1 = np.sqrt(np.mean(errors1**2))
        rmse2 = np.sqrt(np.mean(errors2**2))

        result = {
            'ticker': ticker,
            'model_1': model1,
            'model_2': model2,
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'significant': significant,
            'winner': winner,
            'rmse_1': rmse1,
            'rmse_2': rmse2,
        }
        all_results.append(result)

        # Print
        sig_marker = "*" if p_value < ALPHA else ""
        print(f"  {model1} vs {model2}: DM={dm_stat:+.3f}, p={p_value:.4f}{sig_marker} -> {interpretation}")

# ============================================================
# PODSUMOWANIE
# ============================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv("results/09_diebold_mariano/dm_results.csv", index=False)

print("\n" + "="*60)
print("PODSUMOWANIE")
print("="*60)

# Ile razy każdy model wygrał istotnie
wins = {}
for model in MODELS.values():
    wins[model] = len(results_df[(results_df['winner'] == model)])

print("\nLiczba istotnych zwycięstw (p < 0.05):")
for model, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
    total_comparisons = len(TICKERS) * (len(MODELS) - 1)  # każdy model vs reszta
    print(f"  {model}: {count}")

# Porównanie sieci neuronowych vs GARCH
print("\n--- Sieci neuronowe vs GARCH ---")

nn_models = ['LSTM', 'GRU']
garch_models = ['GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1,1)']

nn_vs_garch = results_df[
    ((results_df['model_1'].isin(nn_models)) & (results_df['model_2'].isin(garch_models))) |
    ((results_df['model_2'].isin(nn_models)) & (results_df['model_1'].isin(garch_models)))
]

nn_wins = 0
garch_wins = 0
no_diff = 0

for _, row in nn_vs_garch.iterrows():
    if row['significant'] == 'TAK':
        if row['winner'] in nn_models:
            nn_wins += 1
        else:
            garch_wins += 1
    else:
        no_diff += 1

print(f"  Sieci neuronowe wygrywają: {nn_wins}")
print(f"  GARCH wygrywają: {garch_wins}")
print(f"  Brak istotnej różnicy: {no_diff}")

# LSTM vs GRU
print("\n--- LSTM vs GRU ---")
lstm_vs_gru = results_df[
    ((results_df['model_1'] == 'LSTM') & (results_df['model_2'] == 'GRU')) |
    ((results_df['model_1'] == 'GRU') & (results_df['model_2'] == 'LSTM'))
]

for _, row in lstm_vs_gru.iterrows():
    ticker = row['ticker']
    p = row['p_value']
    winner = row['winner'] if row['significant'] == 'TAK' else 'brak różnicy'
    print(f"  {ticker}: p={p:.4f} -> {winner}")

# ============================================================
# RAPORT MARKDOWN
# ============================================================

md_lines = []
md_lines.append("# Test Diebolda-Mariano - Wyniki\n")
md_lines.append("Test statystyczny porównujący dokładność prognoz modeli.\n")
md_lines.append(f"- **Poziom istotności:** α = {ALPHA}")
md_lines.append(f"- **H0:** Oba modele mają równą dokładność")
md_lines.append(f"- **H1:** Modele różnią się dokładnością\n")

md_lines.append("## Podsumowanie zwycięstw\n")
md_lines.append("| Model | Istotne zwycięstwa |")
md_lines.append("|-------|-------------------|")
for model, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
    md_lines.append(f"| {model} | {count} |")

md_lines.append("\n## Sieci neuronowe vs GARCH\n")
md_lines.append(f"- Sieci neuronowe wygrywają: **{nn_wins}**")
md_lines.append(f"- GARCH wygrywają: **{garch_wins}**")
md_lines.append(f"- Brak istotnej różnicy: **{no_diff}**")

md_lines.append("\n## LSTM vs GRU (bezpośrednie porównanie)\n")
md_lines.append("| Ticker | p-value | Wynik |")
md_lines.append("|--------|---------|-------|")
for _, row in lstm_vs_gru.iterrows():
    ticker = row['ticker']
    p = row['p_value']
    sig = "**" if row['significant'] == 'TAK' else ""
    winner = row['winner'] if row['significant'] == 'TAK' else 'brak różnicy'
    md_lines.append(f"| {ticker} | {p:.4f} | {sig}{winner}{sig} |")

md_lines.append("\n## Szczegółowe wyniki\n")
md_lines.append("| Ticker | Model 1 | Model 2 | DM stat | p-value | Istotne? | Lepszy |")
md_lines.append("|--------|---------|---------|---------|---------|----------|--------|")

for _, row in results_df.iterrows():
    md_lines.append(
        f"| {row['ticker']} | {row['model_1']} | {row['model_2']} | "
        f"{row['dm_statistic']:+.3f} | {row['p_value']:.4f} | "
        f"{row['significant']} | {row['winner']} |"
    )

md_content = "\n".join(md_lines)

with open("results/09_diebold_mariano/dm_report.md", 'w', encoding='utf-8') as f:
    f.write(md_content)

# ============================================================
# WYKRES: Heatmapa wyników DM (spółki × pary modeli)
# ============================================================

# Wiersze: 5 spółek, kolumny: 6 par NN vs GARCH + 1 para LSTM vs GRU
pairs = [
    ('LSTM', 'GARCH(1,1)'),
    ('LSTM', 'EGARCH(1,1)'),
    ('LSTM', 'GJR-GARCH(1,1,1)'),
    ('GRU',  'GARCH(1,1)'),
    ('GRU',  'EGARCH(1,1)'),
    ('GRU',  'GJR-GARCH(1,1,1)'),
    ('LSTM', 'GRU'),
]
col_labels = ['LSTM\nvs\nGARCH', 'LSTM\nvs\nEGARCH', 'LSTM\nvs\nGJR',
              'GRU\nvs\nGARCH',  'GRU\nvs\nEGARCH',  'GRU\nvs\nGJR',
              'LSTM\nvs\nGRU']

# color_matrix: +1 = model_1 wygrywa istotnie, -1 = model_2 wygrywa istotnie, 0 = brak różnicy
color_matrix = np.zeros((len(TICKERS), len(pairs)))
text_matrix  = np.empty((len(TICKERS), len(pairs)), dtype=object)

for j, (m1, m2) in enumerate(pairs):
    for i, ticker in enumerate(TICKERS):
        row = results_df[
            (results_df['ticker'] == ticker) &
            (((results_df['model_1'] == m1) & (results_df['model_2'] == m2)) |
             ((results_df['model_1'] == m2) & (results_df['model_2'] == m1)))
        ]
        if row.empty:
            text_matrix[i, j] = '—'
            continue
        row = row.iloc[0]
        p = row['p_value']
        text_matrix[i, j] = f"p={p:.3f}"
        if row['significant'] == 'TAK':
            color_matrix[i, j] = 1 if row['winner'] == m1 else -1

# Niestandardowa mapa kolorów: czerwony (-1), żółty (0), zielony (+1)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('dm', ['#d73027', '#ffffbf', '#1a9850'], N=256)

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(color_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(TICKERS)))
ax.set_yticklabels(TICKERS, fontsize=11)

# Linia oddzielająca LSTM vs GARCH i GRU vs GARCH
ax.axvline(x=2.5, color='black', linewidth=2)
# Linia oddzielająca LSTM vs GRU od reszty
ax.axvline(x=5.5, color='black', linewidth=2)

for i in range(len(TICKERS)):
    for j in range(len(pairs)):
        txt = text_matrix[i, j]
        ax.text(j, i, txt, ha='center', va='center', fontsize=8.5, color='black')

# Legenda ręczna
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1a9850', label='Model z lewej wygrywa (p<0.05)'),
    Patch(facecolor='#ffffbf', label='Brak istotnej różnicy'),
    Patch(facecolor='#d73027', label='Model z prawej wygrywa (p<0.05)'),
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.38, 1.02), fontsize=9)

ax.set_title('Test Diebolda-Mariano: wyniki dla każdej spółki (α = 0.05)', fontsize=12)
plt.tight_layout()
plt.savefig("charts/09_diebold_mariano/dm_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nWyniki zapisane:")
print(f"  - results/09_diebold_mariano/dm_results.csv")
print(f"  - results/09_diebold_mariano/dm_report.md")
print(f"  - charts/09_diebold_mariano/dm_heatmap.png")
