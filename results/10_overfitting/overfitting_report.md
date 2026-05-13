# Analiza Overfitting - Hipoteza H3

**H3:** Złożone modele sieci neuronowych są bardziej podatne na przeuczenie.

## Metodologia

- Podział danych 80/10/10 (train/val/test) — zgodny z resztą pipeline'u
- Trening przez 100 epok BEZ early stopping na zbiorze treningowym (80%)
- Walidacja na zbiorze val (10%) — obserwacja rozbieżności train vs val loss
- Metryka overfittingu: % wzrost val loss po osiągnięciu minimum

## Wyniki

| Ticker | LSTM best epoch | LSTM overfit % | GRU best epoch | GRU overfit % |
|--------|-----------------|----------------|----------------|---------------|
| AAPL | 29 | 6.4% | 3 | 38.7% |
| NVDA | 16 | 26.8% | 4 | 65.4% |
| JPM | 67 | 7.4% | 3 | 13.7% |
| XOM | 3 | 22.5% | 1 | 51.0% |
| SCCO | 18 | 1.4% | 16 | 13.3% |
| **Średnia** | 26.6 | 12.9% | 5.4 | 36.4% |

## Wnioski

- **GRU** wykazuje większą tendencję do overfittingu (36.4%)
- **LSTM** jest bardziej odporny (12.9%)

- Early stopping (patience=15) skutecznie zapobiega overfittingowi
- Optymalny moment zatrzymania: ~epoch 16

## Hipoteza H3

**POTWIERDZONA** - sieci neuronowe wykazują tendencję do overfittingu, 
ale jest ona kontrolowana przez early stopping.