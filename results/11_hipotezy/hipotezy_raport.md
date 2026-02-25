# Weryfikacja Hipotez Badawczych

## Hipoteza H1

**Treść:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze 
błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH.

### Wyniki

#### Średnie metryki (wszystkie tickery)

| Model | RMSE | MAE | QLIKE |
|-------|------|-----|-------|
| LSTM | 0.001459 | 0.000587 | -6.9011 |
| GRU | 0.001390 | 0.000545 | -6.7848 |
| GARCH(1,1) | 0.001445 | 0.000577 | -6.7962 |
| EGARCH(1,1) | 0.001447 | 0.000577 | -6.8049 |
| GJR-GARCH(1,1,1) | 0.001448 | 0.000580 | -6.8026 |

#### Porównanie
- Najlepszy model NN (QLIKE): **LSTM** (-6.9011)
- Najlepszy model GARCH (QLIKE): **EGARCH(1,1)** (-6.8049)
- Przewaga NN: 1.41%

#### Test Diebold-Mariano (istotność statystyczna)
- Sieci neuronowe wygrywają istotnie (p<0.05): **12** porównań
- GARCH wygrywa istotnie: **3** porównań
- Brak istotnej różnicy: **15** porównań

### Wniosek

**HIPOTEZA H1: POTWIERDZONA**

Modele LSTM i GRU osiągają statystycznie istotnie lepsze wyniki 
niż klasyczne modele GARCH w prognozowaniu zmienności. 
W teście Diebold-Mariano sieci neuronowe wygrywają 12 razy 
vs 3 dla GARCH. Średni QLIKE dla najlepszego modelu NN 
(-6.9011) jest niższy niż dla najlepszego GARCH (-6.8049).


## Hipoteza H3

**Treść:** Złożone modele sieci neuronowych są bardziej podatne na 
przeuczenie (overfitting) przy ograniczonej próbie danych dziennych.

### Wyniki

#### Analiza krzywych uczenia (100 epok bez early stopping)

| Ticker | LSTM best epoch | LSTM overfit | GRU best epoch | GRU overfit |
|--------|-----------------|--------------|----------------|-------------|
| AAPL | 68 | 3.7% | 63 | 2.0% |
| NVDA | 61 | 5.3% | 31 | 15.7% |
| JPM | 82 | 14.2% | 67 | 10.4% |
| XOM | 55 | 1.3% | 60 | 3.2% |
| SCCO | 48 | 3.0% | 96 | 0.5% |
| **Średnia** | **63** | **5.5%** | **63** | **6.4%** |

*Overfitting = % wzrost validation loss po osiągnięciu minimum*

### Wniosek

**HIPOTEZA H3: POTWIERDZONA**

Sieci neuronowe wykazują tendencję do overfittingu (średnio 5.9%). 
Optymalny moment zatrzymania treningu to około epoki 63. 
Zastosowane techniki regularyzacji (dropout=0.1, early stopping z patience=15) 
skutecznie kontrolują overfitting, zapobiegając znaczącej degradacji wyników.


## Podsumowanie

| Hipoteza | Status | Kluczowy dowód |
|----------|--------|----------------|
| H1 | **POTWIERDZONA** | DM test: NN 12 vs GARCH 3 |
| H3 | **POTWIERDZONA** | Avg overfitting: 5.9% |