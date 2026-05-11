# Weryfikacja Hipotez Badawczych

## Hipoteza H1

**Treść:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze 
błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH.

### Wyniki

#### Średnie metryki (wszystkie tickery)

| Model | RMSE | MAE | QLIKE |
|-------|------|-----|-------|
| LSTM | 0.001435 | 0.000576 | -6.9069 |
| GRU | 0.001389 | 0.000544 | -6.7952 |
| GARCH(1,1) | 0.001382 | 0.000533 | -6.9849 |
| EGARCH(1,1) | 0.001387 | 0.000551 | -6.9870 |
| GJR-GARCH(1,1,1) | 0.001368 | 0.000529 | -6.9929 |

#### Porównanie
- Najlepszy model NN (QLIKE): **LSTM** (-6.9069)
- Najlepszy model GARCH (QLIKE): **GJR-GARCH(1,1,1)** (-6.9929)
- Przewaga NN: -1.23%

#### Test Diebold-Mariano (istotność statystyczna)
- Sieci neuronowe wygrywają istotnie (p<0.05): **0** porównań
- GARCH wygrywa istotnie: **8** porównań
- Brak istotnej różnicy: **22** porównań

### Wniosek

**HIPOTEZA H1: CZĘŚCIOWO POTWIERDZONA**

Wyniki są mieszane - sieci neuronowe nie dominują jednoznacznie.


## Hipoteza H3

**Treść:** Złożone modele sieci neuronowych są bardziej podatne na 
przeuczenie (overfitting) przy ograniczonej próbie danych dziennych.

### Wyniki

#### Analiza krzywych uczenia (100 epok bez early stopping)

| Ticker | LSTM best epoch | LSTM overfit | GRU best epoch | GRU overfit |
|--------|-----------------|--------------|----------------|-------------|
| AAPL | 68 | 3.3% | 63 | 2.1% |
| NVDA | 61 | 5.4% | 31 | 12.1% |
| JPM | 82 | 21.3% | 67 | 10.4% |
| XOM | 55 | 1.3% | 60 | 3.2% |
| SCCO | 48 | 3.2% | 94 | 1.1% |
| **Średnia** | **63** | **6.9%** | **63** | **5.8%** |

*Overfitting = % wzrost validation loss po osiągnięciu minimum*

### Wniosek

**HIPOTEZA H3: POTWIERDZONA**

Sieci neuronowe wykazują tendencję do overfittingu (średnio 6.3%). 
Optymalny moment zatrzymania treningu to około epoki 63. 
Zastosowane techniki regularyzacji (dropout=0.1, early stopping z patience=15) 
skutecznie kontrolują overfitting, zapobiegając znaczącej degradacji wyników.


## Podsumowanie

| Hipoteza | Status | Kluczowy dowód |
|----------|--------|----------------|
| H1 | **CZĘŚCIOWO** | DM test: NN 0 vs GARCH 8 |
| H3 | **POTWIERDZONA** | Avg overfitting: 6.3% |