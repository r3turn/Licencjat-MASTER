# Notatki Metodologiczne - Archiwum do Pracy Licencjackiej

## 1. Uzasadnienie INITIAL_TRAIN_SIZE = 1000 dni

### Dlaczego 1000?

**Argument 1: Stabilność estymacji GARCH**

GARCH(1,1) ma 4 parametry: μ, ω, α, β

Reguła kciuka w ekonometrii: minimum 250-500 obserwacji dla stabilnej estymacji.
1000 daje margines bezpieczeństwa. Badania pokazują, że przy <500 obserwacji
estymatory GARCH są niestabilne (duża wariancja parametrów).

**Argument 2: Cykle rynkowe**

1000 dni ≈ 4 lata (252 dni handlowych/rok)

4 lata to wystarczająco długo, żeby złapać:
- Okresy niskiej zmienności (bull market)
- Okresy wysokiej zmienności (korekty, kryzysy)
- Różne reżimy rynkowe

Gdyby wziąć 250 dni (1 rok), można trafić tylko na spokojny okres
i model nie nauczyłby się reagować na skoki zmienności.

**Argument 3: Literatura**

| Paper | Initial window |
|-------|----------------|
| Hansen & Lunde (2005) | 1000 dni |
| Patton (2011) | 500-1000 dni |
| Liu et al. (2019) - LSTM vs GARCH | 1000 dni |

To jest standard w literaturze o forecasting zmienności.

**Argument 4: Trade-off**

- Za mało (np. 250): Niestabilne parametry, model nie widzi różnych reżimów
- Za dużo (np. 3000): Mniej obserwacji out-of-sample do testowania,
  stare dane mogą być nieistotne (rynki się zmieniły)

1000 to kompromis - wystarczająco dużo do estymacji, wystarczająco dużo zostaje na test.

**Sensitivity analysis**

Dla robustness można sprawdzić wyniki dla różnych okien:
INITIAL_TRAIN_SIZES = [500, 750, 1000, 1250, 1500]

Jeśli ranking modeli jest stabilny - wynik jest robust.

---

## 2. Walk-Forward Validation - co to i dlaczego

### Problem z klasycznym train/test split

W klasycznym ML:
```
[=== TRAIN 70% ===][== VAL 15% ==][== TEST 15% ==]
```

To NIE działa dla prognozowania finansowego:

1. **Rynki się zmieniają** - model wytrenowany na 2005-2018 może być bezużyteczny
   w 2020 (COVID) czy 2022 (inflacja). Parametry GARCH "starzeją się".

2. **W rzeczywistości re-trenujesz** - żaden trader nie używa modelu z 2015 roku
   do prognoz w 2024. Co tydzień/miesiąc aktualizuje się model o nowe dane.

### Jak działa Walk-Forward

```
Dzień 1000: Train na [1-1000], prognozuj dzień 1001
Dzień 1001: Train na [1-1001], prognozuj dzień 1002
...
Dzień 4500: Train na [1-4500], prognozuj dzień 4501
```

Lub z oknem (expanding window z re-fit co N dni):
```
Dzień 1000: Train na [1-1000],    prognozuj 1001
Dzień 1050: Train na [1-1050],    prognozuj 1051   (re-fit co 50 dni)
Dzień 1100: Train na [1-1100],    prognozuj 1101
```

**Symuluje rzeczywisty proces inwestycyjny** - prognozujesz tylko na podstawie
danych które byłyby dostępne w danym momencie (brak look-ahead bias).

### Parametry w projekcie

```python
INITIAL_TRAIN_SIZE = 1000  # Pierwsze okno: 1000 dni (~4 lata)
FORECAST_HORIZON = 1       # Prognoza 1 dzień do przodu
REFIT_EVERY = 50           # Re-trenuj co 50 dni (oszczędność czasu)
```

### Różnica in-sample vs out-of-sample

**BEZ walk-forward (źle):**
```
GARCH fit na całych danych 2005-2024
         ↓
"Prognoza" = conditional_volatility (= dopasowanie, nie prognoza!)
         ↓
RMSE = bardzo niski (bo model "widział" te dane)
```

**Z walk-forward (dobrze):**
```
Dzień 2020-03-15:
  GARCH fit na 2005-01-01 do 2020-03-14
         ↓
  Prognoza na 2020-03-16 (model NIE widział tego dnia)
         ↓
  Porównaj z realized volatility 2020-03-16
```

---

## 3. Proxy zmienności - kwadraty zwrotów

### Problem

Zmienność (σ) jest **nieobserwowalna** - nie możemy jej zmierzyć bezpośrednio.

### Rozwiązanie

Używamy **proxy** (przybliżenia):
- r_t^2 (kwadrat zwrotu) - najprostsze
- |r_t| (wartość bezwzględna) - bardziej odporne na outliers
- Realized Volatility (suma kwadratów zwrotów intraday) - dokładniejsze, ale wymaga danych HF

### Uzasadnienie r_t^2

Dla procesu: r_t = σ_t * ε_t, gdzie ε_t ~ N(0,1)

E[r_t^2] = E[σ_t^2 * ε_t^2] = σ_t^2 * E[ε_t^2] = σ_t^2

Czyli r_t^2 jest **nieobciążonym estymatorem** wariancji σ_t^2.

Wada: wysoka wariancja estymatora (szum).

### Literatura

- Andersen & Bollerslev (1998) - "Answering the Skeptics"
- Patton (2011) - "Volatility Forecast Comparison Using Imperfect Proxies"

---

## 4. Metryki oceny prognoz zmienności

### RMSE (Root Mean Squared Error)

```
RMSE = sqrt(mean((σ_forecast - σ_proxy)^2))
```

Standardowa miara, ale wrażliwa na outliers.

### MAE (Mean Absolute Error)

```
MAE = mean(|σ_forecast - σ_proxy|)
```

Bardziej odporna na outliers niż RMSE.

### QLIKE (Quasi-Likelihood Loss)

```
QLIKE = mean(log(σ^2_forecast) + σ^2_proxy / σ^2_forecast)
```

**Standard w literaturze o zmienności.** Karze niedoszacowanie zmienności
bardziej niż przeszacowanie (asymetryczna funkcja straty).

Uzasadnienie: niedoszacowanie ryzyka jest gorsze niż przeszacowanie
(lepiej mieć za dużo kapitału na pokrycie strat niż za mało).

### Test Diebolda-Mariano

Test statystyczny sprawdzający czy różnica w błędach prognoz
dwóch modeli jest istotna statystycznie.

H0: E[L(e_A)] = E[L(e_B)]  (modele równie dobre)
H1: E[L(e_A)] ≠ E[L(e_B)]  (jeden model lepszy)

Gdzie L() to funkcja straty (np. squared error).

---

## 5. Rozkład t-Studenta vs Normalny w GARCH

### Problem z rozkładem normalnym

Dane finansowe mają "grube ogony" (fat tails / excess kurtosis).
- Rozkład normalny: kurtoza = 3
- Twoje dane: kurtoza 5-18

### Rozwiązanie

Użyj rozkładu t-Studenta w GARCH:
```python
arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
```

Rozkład t ma dodatkowy parametr (stopnie swobody ν) który kontroluje
"grubość" ogonów. Dla ν → ∞ zbliża się do normalnego.

### Alternatywy

- Skewed t (skośny t-Studenta) - dla asymetrii
- GED (Generalized Error Distribution)

---

## 6. Efekt asymetrii (leverage effect)

### Co to jest?

Spadki cen powodują większy wzrost zmienności niż wzrosty cen tej samej wielkości.

Wyjaśnienie: spadek ceny → wzrost dźwigni finansowej (D/E ratio) →
większe ryzyko → większa zmienność.

### Modele uwzględniające asymetrię

**EGARCH (Nelson, 1991):**
```
log(σ_t^2) = ω + α * g(z_{t-1}) + β * log(σ_{t-1}^2)
gdzie g(z) = θ*z + γ*(|z| - E|z|)
```

**GJR-GARCH (Glosten, Jagannathan, Runkle, 1993):**
```
σ_t^2 = ω + (α + γ*I_{t-1}) * ε_{t-1}^2 + β * σ_{t-1}^2
gdzie I_{t-1} = 1 jeśli ε_{t-1} < 0
```

### Hipoteza H2 w pracy

"Uwzględnienie efektu asymetrii (EGARCH/GJR-GARCH) pozwala na skuteczniejszą
estymację ryzyka (VaR) niż podstawowy GARCH(1,1) oraz modele sieciowe."

---

## 7. Data Leakage i Look-Ahead Bias

### Look-Ahead Bias

Używanie informacji z przyszłości do budowy modelu/prognoz.

**Przykład błędu:**
```python
# ŹLE - standaryzacja na całych danych
scaler.fit(all_data)
train_scaled = scaler.transform(train_data)
```

**Poprawnie:**
```python
# DOBRZE - standaryzacja tylko na train
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)
```

### Data Leakage

Wyciek informacji ze zbioru testowego do treningowego.

W kontekście walk-forward: przy każdym re-fit musisz upewnić się,
że model widzi TYLKO dane do momentu t, nie później.

---

## 8. Referencje do cytowania

### GARCH
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity"

### EGARCH / GJR-GARCH
- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns"
- Glosten, Jagannathan, Runkle (1993). "On the Relation between Expected Value and Volatility"

### LSTM/GRU
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"

### Volatility Forecasting
- Hansen & Lunde (2005). "A Forecast Comparison of Volatility Models"
- Patton (2011). "Volatility Forecast Comparison Using Imperfect Proxies"

### LSTM vs GARCH
- Liu et al. (2019). "Novel volatility forecasting using deep learning"
- Kim & Won (2018). "Forecasting stock market volatility with LSTM"

---

*Ostatnia aktualizacja: 2026-02-02*
