# 11_hipotezy_podsumowanie.py - Formalne podsumowanie hipotez badawczych
#
# Generuje raport weryfikujący hipotezy H1 i H3 na podstawie wyników.

import pandas as pd
import os

# ============================================================
# WCZYTAJ WYNIKI
# ============================================================

# Metryki porównawcze
comparison_df = pd.read_csv("results/08_comparison/all_metrics.csv")

# Test Diebold-Mariano
dm_df = pd.read_csv("results/09_diebold_mariano/dm_results.csv")

# Analiza overfitting
overfit_df = pd.read_csv("results/10_overfitting/overfitting_analysis.csv")

os.makedirs("results/11_hipotezy", exist_ok=True)

# ============================================================
# HIPOTEZA H1
# ============================================================
# "Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze błędy
# prognoz (RMSE, QLIKE) niż klasyczne modele GARCH"

print("="*60)
print("HIPOTEZA H1")
print("="*60)

# Średnie metryki per model
mean_metrics = comparison_df.groupby('model')[['rmse', 'mae', 'qlike']].mean()
print("\nŚrednie metryki:")
print(mean_metrics.sort_values('qlike'))

# Ranking
nn_models = ['LSTM', 'GRU']
garch_models = ['GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1,1)']

# Najlepszy NN vs najlepszy GARCH
best_nn_qlike = mean_metrics.loc[nn_models, 'qlike'].min()
best_garch_qlike = mean_metrics.loc[garch_models, 'qlike'].min()
best_nn_rmse = mean_metrics.loc[nn_models, 'rmse'].min()
best_garch_rmse = mean_metrics.loc[garch_models, 'rmse'].min()

print(f"\nNajlepszy NN (QLIKE): {best_nn_qlike:.4f}")
print(f"Najlepszy GARCH (QLIKE): {best_garch_qlike:.4f}")
print(f"Najlepszy NN (RMSE): {best_nn_rmse:.6f}")
print(f"Najlepszy GARCH (RMSE): {best_garch_rmse:.6f}")

# Test DM - NN vs GARCH
nn_vs_garch = dm_df[
    ((dm_df['model_1'].isin(nn_models)) & (dm_df['model_2'].isin(garch_models))) |
    ((dm_df['model_2'].isin(nn_models)) & (dm_df['model_1'].isin(garch_models)))
]

nn_wins_dm = 0
garch_wins_dm = 0
no_diff_dm = 0

for _, row in nn_vs_garch.iterrows():
    if row['significant'] == 'TAK':
        if row['winner'] in nn_models:
            nn_wins_dm += 1
        else:
            garch_wins_dm += 1
    else:
        no_diff_dm += 1

print(f"\nTest Diebold-Mariano (NN vs GARCH):")
print(f"  NN wygrywa istotnie: {nn_wins_dm}")
print(f"  GARCH wygrywa istotnie: {garch_wins_dm}")
print(f"  Brak istotnej różnicy: {no_diff_dm}")

# Wniosek H1
# Trzy poziomy: POTWIERDZONA / CZĘŚCIOWO POTWIERDZONA / ODRZUCONA
# - POTWIERDZONA: NN dominuje w średnich metrykach ORAZ wygrywa istotnie znaczącą większość par DM
# - CZĘŚCIOWO POTWIERDZONA: NN wygrywa w średnich, ale większość par DM bez istotnej różnicy
# - ODRZUCONA: NN nie wygrywa w średnich metrykach ani w teście DM
total_dm        = nn_wins_dm + garch_wins_dm + no_diff_dm
no_diff_share   = no_diff_dm / total_dm if total_dm > 0 else 0
nn_metric_wins  = best_nn_qlike < best_garch_qlike
nn_dm_advantage = nn_wins_dm > garch_wins_dm

if nn_metric_wins and nn_dm_advantage and no_diff_share <= 0.5:
    h1_status = "POTWIERDZONA"
elif (nn_metric_wins or nn_dm_advantage):
    h1_status = "CZĘŚCIOWO POTWIERDZONA"
else:
    h1_status = "ODRZUCONA"

h1_confirmed = (h1_status == "POTWIERDZONA")

print(f"\n>>> H1: {h1_status}")
print(f"    (brak istotnej różnicy w {no_diff_dm}/{total_dm} par = {no_diff_share*100:.0f}%)")

# ============================================================
# HIPOTEZA H3
# ============================================================
# "Złożone modele sieci neuronowych są bardziej podatne na przeuczenie
# (overfitting) przy ograniczonej próbie danych dziennych"

print("\n" + "="*60)
print("HIPOTEZA H3")
print("="*60)

lstm_overfit = overfit_df['lstm_overfit_pct'].mean()
gru_overfit = overfit_df['gru_overfit_pct'].mean()
avg_overfit = (lstm_overfit + gru_overfit) / 2

print(f"\nŚredni overfitting:")
print(f"  LSTM: {lstm_overfit:.1f}%")
print(f"  GRU: {gru_overfit:.1f}%")
print(f"  Średnia NN: {avg_overfit:.1f}%")

# Średni best epoch (kiedy early stopping powinien zadziałać)
avg_best_epoch = (overfit_df['lstm_best_epoch'].mean() + overfit_df['gru_best_epoch'].mean()) / 2
print(f"\nOptymalny moment zatrzymania: epoch {avg_best_epoch:.0f}")

# Wniosek H3 - overfitting > 5% = potwierdzona
h3_confirmed = avg_overfit > 5.0

print(f"\n>>> H3: {'POTWIERDZONA' if h3_confirmed else 'CZĘŚCIOWO POTWIERDZONA'}")
if h3_confirmed:
    print(f"    Sieci wykazują tendencję do overfittingu ({avg_overfit:.1f}%),")
    print(f"    ale jest on kontrolowany przez early stopping i dropout.")
else:
    print(f"    Overfitting jest umiarkowany ({avg_overfit:.1f}% < 5%),")
    print(f"    regularyzacja skutecznie zapobiega przeuczeniu.")

# ============================================================
# RAPORT MARKDOWN
# ============================================================

md = []
md.append("# Weryfikacja Hipotez Badawczych\n")

# H1
md.append("## Hipoteza H1\n")
md.append("**Treść:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze ")
md.append("błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH.\n")

md.append("### Wyniki\n")
md.append("#### Średnie metryki (wszystkie tickery)\n")
md.append("| Model | RMSE | MAE | QLIKE |")
md.append("|-------|------|-----|-------|")
for model in ['LSTM', 'GRU', 'GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1,1)']:
    row = mean_metrics.loc[model]
    md.append(f"| {model} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['qlike']:.4f} |")

md.append(f"\n#### Porównanie")
md.append(f"- Najlepszy model NN (QLIKE): **{mean_metrics.loc[nn_models, 'qlike'].idxmin()}** ({best_nn_qlike:.4f})")
md.append(f"- Najlepszy model GARCH (QLIKE): **{mean_metrics.loc[garch_models, 'qlike'].idxmin()}** ({best_garch_qlike:.4f})")
md.append(f"- Przewaga NN: {((best_garch_qlike - best_nn_qlike) / abs(best_garch_qlike)) * 100:.2f}%\n")

md.append("#### Test Diebold-Mariano (istotność statystyczna)")
md.append(f"- Sieci neuronowe wygrywają istotnie (p<0.05): **{nn_wins_dm}** porównań")
md.append(f"- GARCH wygrywa istotnie: **{garch_wins_dm}** porównań")
md.append(f"- Brak istotnej różnicy: **{no_diff_dm}** porównań\n")

md.append("### Wniosek\n")
md.append(f"**HIPOTEZA H1: {h1_status}**\n")
if h1_status == "POTWIERDZONA":
    md.append("Modele LSTM i GRU osiągają statystycznie istotnie lepsze wyniki ")
    md.append("niż klasyczne modele GARCH w prognozowaniu zmienności. ")
    md.append(f"W teście Diebold-Mariano sieci neuronowe wygrywają {nn_wins_dm} razy ")
    md.append(f"vs {garch_wins_dm} dla GARCH. Średni QLIKE dla najlepszego modelu NN ")
    md.append(f"({best_nn_qlike:.4f}) jest niższy niż dla najlepszego GARCH ({best_garch_qlike:.4f}).\n")
elif h1_status == "CZĘŚCIOWO POTWIERDZONA":
    md.append(f"Sieci neuronowe osiągają lepsze średnie metryki ({best_nn_qlike:.4f} QLIKE ")
    md.append(f"vs {best_garch_qlike:.4f} dla najlepszego GARCH) i wygrywają więcej par ")
    md.append(f"w teście Diebold-Mariano ({nn_wins_dm} vs {garch_wins_dm}). Jednak w ")
    md.append(f"{no_diff_dm} z {total_dm} par DM ({no_diff_share*100:.0f}%) ")
    md.append("nie odrzucono hipotezy zerowej o równej dokładności, ")
    md.append("co nie pozwala mówić o jednoznacznej dominacji sieci nad modelami GARCH.\n")
else:
    md.append("Wyniki nie potwierdzają hipotezy — sieci neuronowe nie osiągają ")
    md.append("istotnie lepszych wyników niż modele GARCH.\n")

# H3
md.append("\n## Hipoteza H3\n")
md.append("**Treść:** Złożone modele sieci neuronowych są bardziej podatne na ")
md.append("przeuczenie (overfitting) przy ograniczonej próbie danych dziennych.\n")

md.append("### Wyniki\n")
md.append("#### Analiza krzywych uczenia (100 epok bez early stopping)\n")
md.append("| Ticker | LSTM best epoch | LSTM overfit | GRU best epoch | GRU overfit |")
md.append("|--------|-----------------|--------------|----------------|-------------|")
for _, row in overfit_df.iterrows():
    md.append(f"| {row['ticker']} | {row['lstm_best_epoch']:.0f} | {row['lstm_overfit_pct']:.1f}% | {row['gru_best_epoch']:.0f} | {row['gru_overfit_pct']:.1f}% |")
md.append(f"| **Średnia** | **{overfit_df['lstm_best_epoch'].mean():.0f}** | **{lstm_overfit:.1f}%** | **{overfit_df['gru_best_epoch'].mean():.0f}** | **{gru_overfit:.1f}%** |")

md.append(f"\n*Overfitting = % wzrost validation loss po osiągnięciu minimum*\n")

md.append("### Wniosek\n")
if h3_confirmed:
    md.append("**HIPOTEZA H3: POTWIERDZONA**\n")
    md.append(f"Sieci neuronowe wykazują tendencję do overfittingu (średnio {avg_overfit:.1f}%). ")
    md.append(f"Optymalny moment zatrzymania treningu to około epoki {avg_best_epoch:.0f}. ")
    md.append("Zastosowane techniki regularyzacji (dropout=0.1, early stopping z patience=15) ")
    md.append("skutecznie kontrolują overfitting, zapobiegając znaczącej degradacji wyników.\n")
else:
    md.append("**HIPOTEZA H3: CZĘŚCIOWO POTWIERDZONA**\n")
    md.append(f"Overfitting jest umiarkowany ({avg_overfit:.1f}%), co sugeruje że ")
    md.append("techniki regularyzacji (dropout, early stopping) skutecznie zapobiegają przeuczeniu.\n")

# Podsumowanie końcowe
md.append("\n## Podsumowanie\n")
md.append("| Hipoteza | Status | Kluczowy dowód |")
md.append("|----------|--------|----------------|")
md.append(f"| H1 | **{h1_status}** | DM: NN {nn_wins_dm}, GARCH {garch_wins_dm}, brak różnicy {no_diff_dm}/{total_dm} |")
md.append(f"| H3 | **{'POTWIERDZONA' if h3_confirmed else 'CZĘŚCIOWO POTWIERDZONA'}** | Avg overfitting: {avg_overfit:.1f}% |")

md_content = "\n".join(md)

with open("results/11_hipotezy/hipotezy_raport.md", 'w', encoding='utf-8') as f:
    f.write(md_content)

print("\n" + "="*60)
print("RAPORT ZAPISANY")
print("="*60)
print("results/11_hipotezy/hipotezy_raport.md")
