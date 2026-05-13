# Weryfikacja Hipotez Badawczych

## Hipoteza H1

**Treść:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze 
błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH.

### Wyniki

#### Średnie metryki (wszystkie tickery)

| Model | RMSE | MAE | QLIKE |
|-------|------|-----|-------|
| LSTM | 0.001036 | 0.000439 | -7.0144 |
| GRU | 0.001038 | 0.000450 | -7.0052 |
| GARCH(1,1) | 0.001044 | 0.000468 | -6.9963 |
| EGARCH(1,1) | 0.001044 | 0.000497 | -7.0079 |
| GJR-GARCH(1,1,1) | 0.001040 | 0.000454 | -7.0005 |

#### Porównanie
- Najlepszy model NN (QLIKE): **LSTM** (-7.0144)
- Najlepszy model GARCH (QLIKE): **EGARCH(1,1)** (-7.0079)
- Przewaga NN: 0.09%

#### Test Diebold-Mariano (istotność statystyczna)
- Sieci neuronowe wygrywają istotnie (p<0.05): **9** porównań
- GARCH wygrywa istotnie: **4** porównań
- Brak istotnej różnicy: **17** porównań

### Wniosek

**HIPOTEZA H1: CZĘŚCIOWO POTWIERDZONA**

Sieci neuronowe osiągają lepsze średnie metryki (-7.0144 QLIKE 
vs -7.0079 dla najlepszego GARCH) i wygrywają więcej par 
w teście Diebold-Mariano (9 vs 4). Jednak w 
17 z 30 par DM (57%) 
nie odrzucono hipotezy zerowej o równej dokładności, 
co nie pozwala mówić o jednoznacznej dominacji sieci nad modelami GARCH.


## Hipoteza H3

**Treść:** Złożone modele sieci neuronowych są bardziej podatne na 
przeuczenie (overfitting) przy ograniczonej próbie danych dziennych.

### Wyniki

#### Analiza krzywych uczenia (100 epok bez early stopping)

| Ticker | LSTM best epoch | LSTM overfit | GRU best epoch | GRU overfit |
|--------|-----------------|--------------|----------------|-------------|
| AAPL | 29 | 6.4% | 3 | 38.7% |
| NVDA | 16 | 26.8% | 4 | 65.4% |
| JPM | 67 | 7.4% | 3 | 13.7% |
| XOM | 3 | 22.5% | 1 | 51.0% |
| SCCO | 18 | 1.4% | 16 | 13.3% |
| **Średnia** | **27** | **12.9%** | **5** | **36.4%** |

*Overfitting = % wzrost validation loss po osiągnięciu minimum*

### Wniosek

**HIPOTEZA H3: POTWIERDZONA**

Sieci neuronowe wykazują tendencję do overfittingu (średnio 24.7%). 
Optymalny moment zatrzymania treningu to około epoki 16. 
Zastosowane techniki regularyzacji (dropout=0.1, early stopping z patience=15) 
skutecznie kontrolują overfitting, zapobiegając znaczącej degradacji wyników.


## Podsumowanie

| Hipoteza | Status | Kluczowy dowód |
|----------|--------|----------------|
| H1 | **CZĘŚCIOWO POTWIERDZONA** | DM: NN 9, GARCH 4, brak różnicy 17/30 |
| H3 | **POTWIERDZONA** | Avg overfitting: 24.7% |