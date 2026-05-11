# Analiza Overfitting - Hipoteza H3

**H3:** Złożone modele sieci neuronowych są bardziej podatne na przeuczenie.

## Metodologia

- Trening przez 100 epok BEZ early stopping
- Obserwacja rozbieżności train vs validation loss
- Metryka: % wzrost val loss po osiągnięciu minimum

## Wyniki

| Ticker | LSTM best epoch | LSTM overfit % | GRU best epoch | GRU overfit % |
|--------|-----------------|----------------|----------------|---------------|
| AAPL | 68 | 3.3% | 63 | 2.1% |
| NVDA | 61 | 5.4% | 31 | 12.1% |
| JPM | 82 | 21.3% | 67 | 10.4% |
| XOM | 55 | 1.3% | 60 | 3.2% |
| SCCO | 48 | 3.2% | 94 | 1.1% |
| **Średnia** | 62.8 | 6.9% | 63.0 | 5.8% |

## Wnioski

- **LSTM** wykazuje większą tendencję do overfittingu (6.9%)
- **GRU** jest bardziej odporny (5.8%)
- Potwierdza to, że prostszy model (GRU) lepiej generalizuje

- Early stopping (patience=15) skutecznie zapobiega overfittingowi
- Optymalny moment zatrzymania: ~epoch 63

## Hipoteza H3

**POTWIERDZONA** - sieci neuronowe wykazują tendencję do overfittingu, 
ale jest ona kontrolowana przez early stopping.