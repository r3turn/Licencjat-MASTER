# Test Diebolda-Mariano - Wyniki

Test statystyczny porównujący dokładność prognoz modeli.

- **Poziom istotności:** α = 0.05
- **H0:** Oba modele mają równą dokładność
- **H1:** Modele różnią się dokładnością

## Podsumowanie zwycięstw

| Model | Istotne zwycięstwa |
|-------|-------------------|
| GARCH(1,1) | 3 |
| GJR-GARCH(1,1,1) | 3 |
| EGARCH(1,1) | 2 |
| GRU | 2 |
| LSTM | 0 |

## Sieci neuronowe vs GARCH

- Sieci neuronowe wygrywają: **0**
- GARCH wygrywają: **8**
- Brak istotnej różnicy: **22**

## LSTM vs GRU (bezpośrednie porównanie)

| Ticker | p-value | Wynik |
|--------|---------|-------|
| AAPL | 0.4218 | brak różnicy |
| NVDA | 0.0010 | **GRU** |
| JPM | 0.0002 | **GRU** |
| XOM | 0.2404 | brak różnicy |
| SCCO | 0.7984 | brak różnicy |

## Szczegółowe wyniki

| Ticker | Model 1 | Model 2 | DM stat | p-value | Istotne? | Lepszy |
|--------|---------|---------|---------|---------|----------|--------|
| AAPL | GARCH(1,1) | EGARCH(1,1) | -0.342 | 0.7326 | NIE | brak |
| AAPL | GARCH(1,1) | GJR-GARCH(1,1,1) | -0.037 | 0.9708 | NIE | brak |
| AAPL | GARCH(1,1) | LSTM | -0.498 | 0.6187 | NIE | brak |
| AAPL | GARCH(1,1) | GRU | +0.679 | 0.4970 | NIE | brak |
| AAPL | EGARCH(1,1) | GJR-GARCH(1,1,1) | +0.062 | 0.9507 | NIE | brak |
| AAPL | EGARCH(1,1) | LSTM | -0.493 | 0.6222 | NIE | brak |
| AAPL | EGARCH(1,1) | GRU | +0.854 | 0.3932 | NIE | brak |
| AAPL | GJR-GARCH(1,1,1) | LSTM | -0.388 | 0.6983 | NIE | brak |
| AAPL | GJR-GARCH(1,1,1) | GRU | +0.446 | 0.6556 | NIE | brak |
| AAPL | LSTM | GRU | +0.803 | 0.4218 | NIE | brak |
| NVDA | GARCH(1,1) | EGARCH(1,1) | -0.502 | 0.6157 | NIE | brak |
| NVDA | GARCH(1,1) | GJR-GARCH(1,1,1) | +1.079 | 0.2804 | NIE | brak |
| NVDA | GARCH(1,1) | LSTM | -3.074 | 0.0021 | TAK | GARCH(1,1) |
| NVDA | GARCH(1,1) | GRU | -0.041 | 0.9669 | NIE | brak |
| NVDA | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.569 | 0.1166 | NIE | brak |
| NVDA | EGARCH(1,1) | LSTM | -2.238 | 0.0252 | TAK | EGARCH(1,1) |
| NVDA | EGARCH(1,1) | GRU | +0.192 | 0.8475 | NIE | brak |
| NVDA | GJR-GARCH(1,1,1) | LSTM | -3.193 | 0.0014 | TAK | GJR-GARCH(1,1,1) |
| NVDA | GJR-GARCH(1,1,1) | GRU | -0.493 | 0.6222 | NIE | brak |
| NVDA | LSTM | GRU | +3.279 | 0.0010 | TAK | GRU |
| JPM | GARCH(1,1) | EGARCH(1,1) | -1.260 | 0.2077 | NIE | brak |
| JPM | GARCH(1,1) | GJR-GARCH(1,1,1) | +1.662 | 0.0965 | NIE | brak |
| JPM | GARCH(1,1) | LSTM | -3.647 | 0.0003 | TAK | GARCH(1,1) |
| JPM | GARCH(1,1) | GRU | +0.079 | 0.9369 | NIE | brak |
| JPM | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.814 | 0.0696 | NIE | brak |
| JPM | EGARCH(1,1) | LSTM | -3.059 | 0.0022 | TAK | EGARCH(1,1) |
| JPM | EGARCH(1,1) | GRU | +0.462 | 0.6441 | NIE | brak |
| JPM | GJR-GARCH(1,1,1) | LSTM | -5.715 | 0.0000 | TAK | GJR-GARCH(1,1,1) |
| JPM | GJR-GARCH(1,1,1) | GRU | -0.990 | 0.3224 | NIE | brak |
| JPM | LSTM | GRU | +3.740 | 0.0002 | TAK | GRU |
| XOM | GARCH(1,1) | EGARCH(1,1) | -0.092 | 0.9268 | NIE | brak |
| XOM | GARCH(1,1) | GJR-GARCH(1,1,1) | +1.202 | 0.2293 | NIE | brak |
| XOM | GARCH(1,1) | LSTM | +0.591 | 0.5543 | NIE | brak |
| XOM | GARCH(1,1) | GRU | -1.150 | 0.2503 | NIE | brak |
| XOM | EGARCH(1,1) | GJR-GARCH(1,1,1) | +0.904 | 0.3659 | NIE | brak |
| XOM | EGARCH(1,1) | LSTM | +0.573 | 0.5669 | NIE | brak |
| XOM | EGARCH(1,1) | GRU | -1.297 | 0.1948 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | LSTM | +0.330 | 0.7413 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | GRU | -1.537 | 0.1242 | NIE | brak |
| XOM | LSTM | GRU | -1.174 | 0.2404 | NIE | brak |
| SCCO | GARCH(1,1) | EGARCH(1,1) | -0.465 | 0.6419 | NIE | brak |
| SCCO | GARCH(1,1) | GJR-GARCH(1,1,1) | +0.884 | 0.3766 | NIE | brak |
| SCCO | GARCH(1,1) | LSTM | -2.361 | 0.0182 | TAK | GARCH(1,1) |
| SCCO | GARCH(1,1) | GRU | -1.698 | 0.0895 | NIE | brak |
| SCCO | EGARCH(1,1) | GJR-GARCH(1,1,1) | +0.610 | 0.5419 | NIE | brak |
| SCCO | EGARCH(1,1) | LSTM | -1.912 | 0.0559 | NIE | brak |
| SCCO | EGARCH(1,1) | GRU | -1.456 | 0.1454 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | LSTM | -2.676 | 0.0075 | TAK | GJR-GARCH(1,1,1) |
| SCCO | GJR-GARCH(1,1,1) | GRU | -1.865 | 0.0622 | NIE | brak |
| SCCO | LSTM | GRU | -0.255 | 0.7984 | NIE | brak |