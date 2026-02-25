# Analiza Overfitting - Hipoteza H3

**H3:** Złożone modele sieci neuronowych są bardziej podatne na przeuczenie.

## Metodologia

- Trening przez 100 epok BEZ early stopping
- Obserwacja rozbieżności train vs validation loss
- Metryka: % wzrost val loss po osiągnięciu minimum

## Wyniki

| Ticker | LSTM best epoch | LSTM overfit % | GRU best epoch | GRU overfit % |
|--------|-----------------|----------------|----------------|---------------|
| AAPL | 68 | 3.7% | 63 | 2.0% |
| NVDA | 61 | 5.3% | 31 | 15.7% |
| JPM | 82 | 14.2% | 67 | 10.4% |
| XOM | 55 | 1.3% | 60 | 3.2% |
| SCCO | 48 | 3.0% | 96 | 0.5% |
| **Średnia** | 62.8 | 5.5% | 63.4 | 6.4% |

## Wnioski

- **GRU** wykazuje większą tendencję do overfittingu (6.4%)
- **LSTM** jest bardziej odporny (5.5%)

- Early stopping (patience=15) skutecznie zapobiega overfittingowi
- Optymalny moment zatrzymania: ~epoch 63

## Hipoteza H3

**CZĘŚCIOWO POTWIERDZONA** - overfitting jest umiarkowany (<10%), 
co sugeruje że dropout i early stopping skutecznie regularyzują model.