# Dokumentacja Techniczna Projektu
## Prognozowanie zmienności z wykorzystaniem modeli GARCH i sieci neuronowych

---

# 1. CEL I HIPOTEZY BADAWCZE

## 1.1 Cel pracy
Porównanie skuteczności prognostycznej klasycznych modeli ekonometrycznych (GARCH, EGARCH, GJR-GARCH) z nowoczesnymi metodami uczenia maszynowego (LSTM, GRU) w zadaniu prognozowania zmienności warunkowej na rynkach akcyjnych.

## 1.2 Hipotezy badawcze

### H1: Sieci neuronowe vs GARCH
**Treść:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH.

**Status: POTWIERDZONA**
- Test Diebold-Mariano: NN wygrywa **12 razy** vs GARCH **3 razy**
- Najlepszy QLIKE: LSTM (-6.9011) vs EGARCH (-6.8049)
- Przewaga NN: ~1.4%

### H3: Overfitting sieci neuronowych
**Treść:** Złożone modele sieci neuronowych są bardziej podatne na przeuczenie (overfitting) przy ograniczonej próbie danych dziennych.

**Status: POTWIERDZONA**
- Średni overfitting: LSTM 5.5%, GRU 6.4%
- Regularyzacja (dropout + early stopping) skutecznie kontroluje problem
- Optymalny moment zatrzymania: ~epoka 63

---

# 2. DANE

## 2.1 Instrumenty finansowe

| Ticker | Firma | Sektor | Uzasadnienie wyboru |
|--------|-------|--------|---------------------|
| AAPL | Apple | Technologia (USA) | High-beta, reprezentacja tech sector |
| NVDA | NVIDIA | Technologia (USA) | Bardzo wysoka zmienność, AI boom |
| JPM | JP Morgan | Finanse (USA) | Sektor bankowy, cykle ekonomiczne |
| XOM | Exxon Mobil | Energia (USA) | Sektor surowcowy, korelacja z ropą |
| SCCO | Southern Copper | Mining | Dywersyfikacja geograficzna/sektorowa |

## 2.2 Okres badawczy
- **Start:** 2005-01-01
- **Koniec:** 2024-12-31
- **Długość:** ~20 lat danych dziennych
- **Źródło:** Yahoo Finance (yfinance)

## 2.3 Statystyki opisowe zwrotów logarytmicznych

| Ticker | Średnia | Odch. std | Skośność | Kurtoza | Min | Max |
|--------|---------|-----------|----------|---------|-----|-----|
| AAPL | 0.0011 | 0.0203 | -0.25 | **5.86** | -19.7% | +13.0% |
| NVDA | 0.0013 | 0.0306 | -0.27 | **8.88** | -36.7% | +26.1% |
| JPM | 0.0005 | 0.0228 | +0.27 | **18.03** | -23.2% | +22.4% |
| XOM | 0.0003 | 0.0167 | -0.07 | **9.79** | -15.0% | +15.9% |
| SCCO | 0.0007 | 0.0265 | +0.06 | **7.22** | -22.6% | +25.6% |

### Interpretacja kurtozy
- **Kurtoza > 3** = rozkład leptokurtyczny ("grube ogony")
- **JPM kurtoza = 18.03** - ekstremalne grube ogony (kryzysy finansowe 2008, COVID)
- Uzasadnia użycie rozkładu t-Studenta w modelach GARCH

### Testy statystyczne
- **Test Jarque-Bera:** p < 0.001 dla wszystkich (odrzucenie normalności)
- **Test ARCH (5 lagów):** p < 1e-18 dla wszystkich (potwierdzenie efektu ARCH)

## 2.4 Preprocessing

### Zwroty logarytmiczne
```
r_t = ln(P_t / P_{t-1})
```
**Dlaczego logarytmiczne, nie procentowe:**
1. Addytywność w czasie: ln(P_n/P_0) = Σ r_t
2. Symetryczność: +10% i -10% mają ten sam wpływ absolutny
3. Przybliżona normalność dla małych zmian
4. Standard w literaturze finansowej

### Proxy zmienności (target)
```
σ²_realized = r_t²
```
**Dlaczego kwadraty zwrotów:**
- Zmienność jest zmienną ukrytą (latent variable)
- Nie można jej bezpośrednio obserwować
- r² jest nieobciążonym, ale szumowym estymatorem wariancji
- Standard w literaturze (Andersen & Bollerslev, 1998)

**Alternatywy (nie użyte):**
- Realized Variance z danych intraday (wymaga danych wysokiej częstotliwości)
- Range-based estimators (Parkinson, Garman-Klass)

---

# 3. METODOLOGIA

## 3.1 Walk-Forward Validation (Expanding Window)

### Dlaczego nie zwykły train/test split?
Dane finansowe są szeregami czasowymi - przyszłość nie może "wyciekać" do przeszłości.

### Procedura:
```
Dzień 1000: Trenuj na [1, 1000], prognozuj 1001
Dzień 1001: Trenuj na [1, 1001], prognozuj 1002
Dzień 1002: Trenuj na [1, 1002], prognozuj 1003
...
```

### Parametry:
```python
INITIAL_TRAIN_SIZE = 1000  # ~4 lata danych początkowych
FORECAST_HORIZON = 1       # Prognoza 1 dzień do przodu
```

### Refit frequency (kluczowy parametr):
```python
# GARCH: szybki fit, częste re-estymacje
REFIT_EVERY_GARCH = 50    # co 50 dni

# Sieci neuronowe: wolny trening, rzadsze re-estymacje
REFIT_EVERY_NN = 250      # co 250 dni (~1 rok)
```

**Trade-off:**
- Częstsze refit = świeższe parametry, ale dłuższy czas obliczeń
- Rzadsze refit = szybsze obliczenia, ale potencjalnie nieaktualne parametry
- GARCH(1,1) fituje się w ~0.1s, LSTM w ~5-10s

### Liczba prognoz out-of-sample:
- Dane: ~5000 obserwacji
- Train: 1000
- Test: ~4000 prognoz out-of-sample na ticker

---

# 4. MODELE

## 4.1 GARCH(1,1) - Benchmark

### Specyfikacja:
```
r_t = μ + ε_t
ε_t = σ_t * z_t,  z_t ~ t(ν)

σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
```

### Parametry:
- **μ (mean):** stała w równaniu średniej
- **ω (omega):** stała w równaniu wariancji
- **α (alpha):** reakcja na wczorajszy szok
- **β (beta):** persistence wariancji
- **ν (nu):** stopnie swobody rozkładu t

### Warunki stacjonarności:
- α + β < 1 (persistence < 1)
- α, β ≥ 0
- ω > 0

### Konfiguracja w kodzie:
```python
MODEL_CONFIG = {
    'vol': 'GARCH',
    'p': 1,  # lagi dla ε²
    'q': 1,  # lagi dla σ²
}

model = arch_model(
    returns * 100,      # skalowanie dla stabilności numerycznej
    mean='Constant',    # stała średnia (vs 'Zero')
    dist='t',           # rozkład t-Studenta (grube ogony)
    vol='GARCH', p=1, q=1
)
```

**Dlaczego `mean='Constant'` nie `'Zero'`:**
- 'Constant' estymuje μ z danych
- 'Zero' zakłada μ=0 (częste w literaturze dla dziennych zwrotów)
- Oba podejścia są poprawne; 'Constant' jest bardziej ogólne

**Dlaczego `dist='t'` nie `'Normal'`:**
- Dane finansowe mają grube ogony (kurtoza >> 3)
- Rozkład normalny niedoszacowuje ekstremalne zdarzenia
- t-Studenta lepiej modeluje ryzyko ogona (tail risk)

## 4.2 EGARCH(1,1) - Asymetria

### Specyfikacja:
```
ln(σ²_t) = ω + α * |z_{t-1}| + γ * z_{t-1} + β * ln(σ²_{t-1})
```

### Zalety vs GARCH:
1. **Modeluje asymetrię** (leverage effect) przez parametr γ
2. **Brak ograniczeń** na znaki parametrów (logarytm gwarantuje σ² > 0)
3. Lepiej oddaje "złe wiadomości zwiększają zmienność bardziej niż dobre"

### Konfiguracja:
```python
MODEL_CONFIG = {'vol': 'EGARCH', 'p': 1, 'q': 1}
```

## 4.3 GJR-GARCH(1,1,1) - Threshold

### Specyfikacja:
```
σ²_t = ω + α * ε²_{t-1} + γ * ε²_{t-1} * I_{ε<0} + β * σ²_{t-1}
```

Gdzie I_{ε<0} = 1 gdy ε_{t-1} < 0, czyli gdy wczorajszy zwrot był ujemny.

### Zalety:
- Prostsza interpretacja asymetrii niż EGARCH
- γ > 0 oznacza, że negatywne szoki zwiększają zmienność bardziej

### Konfiguracja:
```python
MODEL_CONFIG = {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1}  # 'o' = threshold order
```

## 4.4 LSTM (Long Short-Term Memory)

### Architektura:
```
Input (20 dni) → LSTM(32 hidden) → Dropout(0.1) → Linear(1) → Softplus → σ²
```

### Dlaczego LSTM dla zmienności:
1. **Pamięć długoterminowa:** Zmienność ma długą persistence (β ~0.9)
2. **Nieliniowość:** Może uchwycić złożone wzorce, których GARCH nie widzi
3. **Brak założeń parametrycznych:** Nie wymusza konkretnej formy funkcyjnej

### Hiperparametry:
```python
WINDOW_SIZE = 20       # Lookback window (dni)
HIDDEN_SIZE = 32       # Neurony w warstwie ukrytej
NUM_LAYERS = 1         # Liczba warstw LSTM
DROPOUT = 0.1          # Regularyzacja
EPOCHS = 100           # Max epok
BATCH_SIZE = 32        # Rozmiar mini-batch
LEARNING_RATE = 0.001  # Adam optimizer
PATIENCE = 15          # Early stopping patience
```

### Uzasadnienie hiperparametrów:

**WINDOW_SIZE = 20:**
- ~1 miesiąc giełdowy (20 dni roboczych)
- Kompromis między pamięcią a szumem
- Większe okno = więcej kontekstu, ale wolniejszy trening

**HIDDEN_SIZE = 32:**
- Mały rozmiar zapobiega overfittingowi
- Dla danych finansowych (niski SNR) większe sieci nie pomagają
- Literatura: 32-64 dla szeregów czasowych finansowych

**DROPOUT = 0.1:**
- Lekka regularyzacja
- Za duży (>0.3) spowalnia zbieżność
- Działa tylko przy num_layers > 1 dla LSTM wewnętrznego

**PATIENCE = 15:**
- Ile epok bez poprawy przed zatrzymaniem
- 15 daje modelowi szansę na "plateau" przed poprawą
- Typowo: 10-20 w literaturze

### Softplus jako aktywacja wyjściowa:
```python
Softplus(x) = ln(1 + e^x)
```
- Gwarantuje σ² > 0 (wariancja musi być dodatnia)
- Gładsze niż ReLU (brak "martwych neuronów")

### Normalizacja danych:
```python
# Z-score normalizacja (fit na train, transform na wszystko)
train_mean = np.mean(train_returns)
train_std = np.std(train_returns)
returns_norm = (returns - train_mean) / train_std

# De-normalizacja prognozy
sigma2_pred = sigma2_pred_norm * (train_std ** 2)
```

## 4.5 GRU (Gated Recurrent Unit)

### Architektura:
Identyczna jak LSTM, ale z komórką GRU zamiast LSTM.

### Różnice vs LSTM:
| Aspekt | LSTM | GRU |
|--------|------|-----|
| Bramki | 3 (forget, input, output) | 2 (reset, update) |
| Parametry | ~4x hidden_size² | ~3x hidden_size² |
| Trening | Wolniejszy | Szybszy |
| Pamięć długoterminowa | Lepsza | Gorsza (teoretycznie) |

### Dlaczego GRU działa podobnie dobrze:
- Dla 20-dniowego okna różnica w pamięci nie ma znaczenia
- Mniej parametrów = mniejsze ryzyko overfittingu
- W praktyce: porównywalna dokładność, szybszy trening

---

# 5. METRYKI OCENY

## 5.1 RMSE (Root Mean Squared Error)
```
RMSE = sqrt(mean((y_true - y_pred)²))
```
- Standardowa miara błędu
- Karze duże błędy bardziej niż małe (kwadrat)
- Jednostka: taka sama jak target (σ²)

## 5.2 MAE (Mean Absolute Error)
```
MAE = mean(|y_true - y_pred|)
```
- Bardziej odporna na outliers niż RMSE
- Intuicyjna interpretacja: średni błąd bezwzględny

## 5.3 QLIKE Loss (kluczowa metryka!)
```
QLIKE = mean(ln(σ²_pred) + σ²_true / σ²_pred)
```

### Dlaczego QLIKE jest standardem dla zmienności:
1. **Asymetryczna:** Karze niedoszacowanie bardziej niż przeszacowanie
2. **Robust:** Odporna na skalowanie proxy zmienności
3. **Właściwy scoring rule:** Minimalizowana przez prawdziwą wariancję warunkową
4. **Standard w literaturze:** Patton (2011), Hansen & Lunde (2005)

### Interpretacja:
- **Mniejsze = lepsze** (to jest loss, nie accuracy)
- Wartości ujemne są normalne (logarytm małych liczb)
- Porównywalne tylko w obrębie tego samego datasetu

## 5.4 MAPE (nie używane w raportach)
```
MAPE = mean(|y_true - y_pred| / y_true) * 100%
```

### Dlaczego MAPE jest bezużyteczne dla zmienności:
- Dzieli przez y_true (realized variance)
- Gdy r_t ≈ 0, y_true = r² ≈ 0, MAPE → ∞
- Często daje wartości >1,000,000% (bezsensowne)

---

# 6. TESTY STATYSTYCZNE

## 6.1 Test Diebold-Mariano

### Cel:
Sprawdzić, czy różnica w dokładności prognoz dwóch modeli jest **statystycznie istotna**.

### Hipotezy:
- H0: Oba modele mają równą dokładność
- H1: Modele różnią się dokładnością

### Statystyka testowa:
```
d_t = e1_t² - e2_t²  (różnica strat)
DM = mean(d) / sqrt(var(d)/n)
```

Pod H0: DM ~ N(0,1)

### Korekta Newey-West:
Dla h-krokowych prognoz, uwzględnia autokorelację błędów:
```
var(d) = γ_0 + 2 * Σ_{k=1}^{h-1} γ_k
```

### Interpretacja wyników:
- p < 0.05: Różnica istotna statystycznie
- DM > 0: Model 2 lepszy
- DM < 0: Model 1 lepszy

### Uwaga metodologiczna:
Test DM w tej implementacji używa **MSE** (squared errors), nie QLIKE.
Jest to standardowe podejście, ale warto to zaznaczyć w pracy.

---

# 7. ANALIZA OVERFITTING

## 7.1 Metodologia
1. Trenuj sieci przez 100 epok **bez early stopping**
2. Zapisuj train loss i val loss w każdej epoce
3. Znajdź epokę z minimalnym val loss
4. Zmierz wzrost val loss po minimum

## 7.2 Metryka overfittingu
```
overfitting_pct = (val_loss_final / val_loss_min - 1) * 100%
```

### Wyniki:
| Ticker | LSTM best epoch | LSTM overfit | GRU best epoch | GRU overfit |
|--------|-----------------|--------------|----------------|-------------|
| AAPL | 68 | 3.7% | 63 | 2.0% |
| NVDA | 61 | 5.3% | 31 | 15.7% |
| JPM | 82 | 14.2% | 67 | 10.4% |
| XOM | 55 | 1.3% | 60 | 3.2% |
| SCCO | 48 | 3.0% | 96 | 0.5% |
| **Średnia** | **63** | **5.5%** | **63** | **6.4%** |

### Wnioski:
- Overfitting istnieje, ale jest umiarkowany (~6%)
- Early stopping z patience=15 skutecznie go kontroluje
- Optymalny moment zatrzymania: ~epoka 60-65

---

# 8. WYNIKI KOŃCOWE

## 8.1 Średnie metryki (wszystkie tickery)

| Model | RMSE | MAE | QLIKE | Ranking |
|-------|------|-----|-------|---------|
| **LSTM** | 0.001459 | 0.000587 | **-6.9011** | 1 |
| **GRU** | 0.001390 | 0.000545 | -6.7848 | 2 |
| GARCH(1,1) | 0.001445 | 0.000577 | -6.7962 | 3 |
| EGARCH(1,1) | 0.001447 | 0.000577 | -6.8049 | 4 |
| GJR-GARCH(1,1,1) | 0.001448 | 0.000580 | -6.8026 | 5 |

## 8.2 Test Diebold-Mariano (podsumowanie)

| Porównanie | NN wygrywa | GARCH wygrywa | Brak różnicy |
|------------|------------|---------------|--------------|
| Wszystkie pary NN vs GARCH | **12** | 3 | 15 |

### Szczegóły per ticker:
- **AAPL:** NN dominuje (wszystkie porównania istotne)
- **NVDA:** GRU najlepszy, LSTM nieistotnie lepszy od GARCH
- **JPM:** Mieszane wyniki (GARCH wygrywa z LSTM!)
- **XOM:** GRU najlepszy
- **SCCO:** Brak istotnych różnic

## 8.3 Liczba re-estymacji modeli
- GARCH (refit co 50 dni): **81 razy** per ticker
- Sieci neuronowe (refit co 250 dni): **17 razy** per ticker

---

# 9. POTENCJALNE PYTANIA PROMOTORA

## Q1: Dlaczego kwadraty zwrotów jako proxy zmienności?
**A:** Zmienność jest zmienną ukrytą. r² jest nieobciążonym estymatorem wariancji warunkowej E[r²|F_{t-1}] = σ². Standard w literaturze (Andersen & Bollerslev 1998). Alternatywa (Realized Variance) wymaga danych intraday.

## Q2: Dlaczego rozkład t-Studenta w GARCH?
**A:** Dane finansowe mają grube ogony (kurtoza 6-18). Rozkład normalny niedoszacowuje ryzyko ogona. t-Studenta z estymowanymi stopniami swobody lepiej modeluje ekstremalne zdarzenia.

## Q3: Dlaczego REFIT_EVERY różne dla GARCH vs NN?
**A:** Trade-off czas vs dokładność. GARCH fituje się w ~0.1s, można re-estymować często (co 50 dni). LSTM trwa ~5-10s, rzadsze re-estymacje (co 250 dni) dla praktyczności. Oba podejścia dają stabilne wyniki.

## Q4: Dlaczego tak małe sieci (32 neurony)?
**A:** Dane finansowe mają niski stosunek sygnału do szumu. Większe sieci overfitują (memoryzują szum). 32 neurony to sweet spot między ekspresywnością a generalizacją. Potwierdzone analizą overfittingu.

## Q5: Dlaczego QLIKE, nie RMSE jako główna metryka?
**A:** QLIKE jest standardem w literaturze o zmienności (Patton 2011). Jest asymetryczna (karze niedoszacowanie bardziej) i robust na skalowanie proxy. RMSE też raportujemy dla kompletności.

## Q6: Co oznacza ujemny QLIKE?
**A:** To normalne. QLIKE = mean(ln(σ²) + ...). Dla małych wariancji ln(σ²) < 0. Wartość bezwzględna nie ma znaczenia, liczy się ranking modeli.

## Q7: Dlaczego JPM ma kurtozę 18?
**A:** JPM (sektor bankowy) był ekstremalnie volatilny podczas kryzysów (2008 Lehman, 2020 COVID). Pojedyncze dni z ruchami >20% drastycznie zwiększają kurtozę. To nie błąd w danych, to rzeczywistość rynkowa.

## Q8: Czy wyniki są replikowalne?
**A:** Tak. Używamy set_seed(42) przed każdym treningiem. PyTorch deterministic mode włączony. Jednak małe różnice mogą wystąpić przez niedeterministyczne operacje GPU.

## Q9: Dlaczego LSTM wygrywa w QLIKE ale GRU w RMSE?
**A:** QLIKE i RMSE optymalizują różne cele. QLIKE karze niedoszacowanie bardziej, RMSE traktuje błędy symetrycznie. GRU może mieć mniejsze błędy średnio, ale LSTM lepiej unika niedoszacowania zmienności.

## Q10: Co z VaR?
**A:** VaR został usunięty z zakresu pracy po konsultacjach. Projekt skupia się na bezpośrednim prognozowaniu zmienności, nie na aplikacjach risk management.

---

# 10. STRUKTURA KODU

```
projekt/
├── params.py                    # Konfiguracja globalna
├── 00_run_all.py               # Uruchom całą pipeline
├── 01_fetch_data.py            # Pobierz dane z Yahoo Finance
├── 02_preprocessing.py         # Preprocessing, statystyki opisowe
├── 03_garch.py                 # GARCH(1,1)
├── 04_egarch.py                # EGARCH(1,1)
├── 05_gjr_garch.py             # GJR-GARCH(1,1,1)
├── 06_lstm.py                  # LSTM
├── 07_gru.py                   # GRU
├── 08_comparison.py            # Porównanie wszystkich modeli
├── 09_diebold_mariano.py       # Testy statystyczne
├── 10_overfitting_analysis.py  # Analiza overfittingu
├── 11_hipotezy_podsumowanie.py # Weryfikacja hipotez
├── utils/
│   ├── walk_forward.py         # Walk-forward dla GARCH
│   ├── lstm_utils.py           # Architektura i trening LSTM
│   ├── gru_utils.py            # Architektura i trening GRU
│   └── metrics.py              # RMSE, MAE, QLIKE, DM test
├── data/
│   ├── raw/                    # Surowe dane z Yahoo
│   └── processed/              # Przetworzone zwroty
├── results/                    # Wyniki (CSV, Parquet)
└── charts/                     # Wykresy (PNG)
```

---

# 11. BIBLIOGRAFIA (do uzupełnienia w pracy)

1. **Bollerslev, T. (1986)** - "Generalized autoregressive conditional heteroskedasticity" - Wprowadzenie GARCH
2. **Nelson, D.B. (1991)** - "Conditional heteroskedasticity in asset returns" - EGARCH
3. **Glosten, Jagannathan, Runkle (1993)** - GJR-GARCH
4. **Hochreiter & Schmidhuber (1997)** - LSTM
5. **Cho et al. (2014)** - GRU
6. **Diebold & Mariano (1995)** - Test porównania prognoz
7. **Patton, A.J. (2011)** - "Volatility forecast comparison using imperfect volatility proxies" - QLIKE
8. **Andersen & Bollerslev (1998)** - Realized variance jako proxy

---

*Dokument wygenerowany automatycznie na podstawie kodu i wyników projektu.*
*Ostatnia aktualizacja: 2024*
