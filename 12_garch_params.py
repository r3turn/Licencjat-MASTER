# 12_garch_params.py - Estymacja parametrów modeli GARCH na zbiorze treningowym
#
# Dopasowuje każdy wariant GARCH na 80% zbioru treningowego (identycznie jak
# w skryptach 03/04/05) i zapisuje estymowane parametry do tabeli.
# Parametry te są dokładnie tymi, które wygenerowały prognozy testowe
# porównywane następnie z modelami LSTM i GRU.
#
# Spójność z resztą pipeline'u:
#   - GARCH (03/04/05): estymacja na 80% train -> prognozy na 10% test
#   - LSTM/GRU (06/07): trening na 80% train  -> prognozy na 10% test
# Tabela parametrów GARCH odzwierciedla zatem ten sam zbiór danych, na którym
# uczone były sieci neuronowe.

import pandas as pd
import numpy as np
import os
from arch import arch_model

from params import TICKERS, TRAIN_RATIO

os.makedirs("results/12_garch_params", exist_ok=True)

# Konfiguracje modeli (identyczne jak w 03/04/05)
MODELS_CONFIG = {
    'GARCH(1,1)':       {'vol': 'GARCH',  'p': 1, 'q': 1, 'o': 0},
    'EGARCH(1,1)':      {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1},
    'GJR-GARCH(1,1,1)': {'vol': 'GARCH',  'p': 1, 'q': 1, 'o': 1},
}

returns = pd.read_parquet("data/processed/returns.parquet")
print(f"Dane: {returns.shape}, okres: {returns.index[0].date()} — {returns.index[-1].date()}")

n_total    = len(returns)
train_size = int(n_total * TRAIN_RATIO)
print(f"Zbiór treningowy: pierwsze {train_size} obserwacji ({TRAIN_RATIO*100:.0f}%)")
print(f"Okres treningowy: {returns.index[0].date()} — {returns.index[train_size-1].date()}\n")

rows = []

for ticker in TICKERS:
    # Estymacja wyłącznie na zbiorze treningowym (80%) — spójność z 03/04/05
    y = returns[ticker].iloc[:train_size].values * 100  # skalowanie wymagane przez arch_model

    for model_name, config in MODELS_CONFIG.items():
        print(f"  Estimating {model_name} — {ticker}...", end=" ")
        try:
            model = arch_model(y, mean='Constant', dist='t', **config)
            res   = model.fit(disp='off', show_warning=False)
            p     = res.params

            omega = float(p['omega'])
            alpha = float(p.get('alpha[1]', np.nan))
            beta  = float(p['beta[1]'])
            gamma = float(p.get('gamma[1]', np.nan))
            nu    = float(p.get('nu', np.nan))

            # Miara persystencji zmienności
            if model_name == 'GARCH(1,1)':
                persistence = alpha + beta
            elif model_name == 'GJR-GARCH(1,1,1)':
                persistence = alpha + 0.5 * gamma + beta
            else:
                # EGARCH: beta pełni rolę persystencji
                persistence = beta

            rows.append({
                'Model':       model_name,
                'Ticker':      ticker,
                'omega':       omega,
                'alpha':       alpha,
                'beta':        beta,
                'gamma':       gamma,
                'nu':          nu,
                'persistence': persistence,
            })
            print(f"alpha={alpha:.4f}, beta={beta:.4f}, persist={persistence:.4f}")

        except Exception as e:
            print(f"BŁĄD: {e}")

df = pd.DataFrame(rows)
df.to_csv("results/12_garch_params/garch_params.csv", index=False)

# ============================================================
# TABELA ZBIORCZA (pivot: tickery jako wiersze, parametry jako kolumny)
# ============================================================

print("\n" + "="*70)
print("PARAMETRY MODELI GARCH — ZBIÓR TRENINGOWY (80%)")
print("="*70)

for model_name in MODELS_CONFIG:
    sub = df[df['Model'] == model_name][
        ['Ticker', 'omega', 'alpha', 'beta', 'gamma', 'nu', 'persistence']
    ].set_index('Ticker')
    print(f"\n{model_name}:")
    print(sub.round(5).to_string())

# ============================================================
# SPRAWDZENIE WARUNKÓW STABILNOŚCI
# ============================================================

print("\n" + "="*70)
print("WARUNKI STABILNOŚCI (persistence < 1)")
print("="*70)

unstable = df[df['persistence'] >= 1.0]
if unstable.empty:
    print("Wszystkie modele stabilne (persistence < 1).")
else:
    print("UWAGA — niestabilne modele:")
    print(unstable[['Model', 'Ticker', 'persistence']].to_string(index=False))

print(f"\nŚrednia persystencja:")
print(df.groupby('Model')['persistence'].mean().round(4).to_string())

print(f"\nWyniki zapisane: results/12_garch_params/garch_params.csv")
