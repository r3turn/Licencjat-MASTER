# Test Diebolda-Mariano - Wyniki

Test statystyczny porównujący dokładność prognoz modeli.

- **Poziom istotności:** α = 0.05
- **H0:** Oba modele mają równą dokładność
- **H1:** Modele różnią się dokładnością

## Podsumowanie zwycięstw

| Model | Istotne zwycięstwa |
|-------|-------------------|
| LSTM | 6 |
| GRU | 5 |
| GARCH(1,1) | 3 |
| GJR-GARCH(1,1,1) | 3 |
| EGARCH(1,1) | 1 |

## Sieci neuronowe vs GARCH

- Sieci neuronowe wygrywają: **9**
- GARCH wygrywają: **4**
- Brak istotnej różnicy: **17**

## LSTM vs GRU (bezpośrednie porównanie)

| Ticker | p-value | Wynik |
|--------|---------|-------|
| AAPL | 0.5580 | brak różnicy |
| NVDA | 0.1196 | brak różnicy |
| JPM | 0.5309 | brak różnicy |
| XOM | 0.0290 | **LSTM** |
| SCCO | 0.0243 | **LSTM** |

## Szczegółowe wyniki

| Ticker | Model 1 | Model 2 | DM stat | p-value | Istotne? | Lepszy |
|--------|---------|---------|---------|---------|----------|--------|
| AAPL | GARCH(1,1) | EGARCH(1,1) | -2.889 | 0.0039 | TAK | GARCH(1,1) |
| AAPL | GARCH(1,1) | GJR-GARCH(1,1,1) | -1.520 | 0.1286 | NIE | brak |
| AAPL | GARCH(1,1) | LSTM | -2.702 | 0.0069 | TAK | GARCH(1,1) |
| AAPL | GARCH(1,1) | GRU | -2.600 | 0.0093 | TAK | GARCH(1,1) |
| AAPL | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.212 | 0.2254 | NIE | brak |
| AAPL | EGARCH(1,1) | LSTM | -1.557 | 0.1195 | NIE | brak |
| AAPL | EGARCH(1,1) | GRU | -1.361 | 0.1735 | NIE | brak |
| AAPL | GJR-GARCH(1,1,1) | LSTM | -2.551 | 0.0108 | TAK | GJR-GARCH(1,1,1) |
| AAPL | GJR-GARCH(1,1,1) | GRU | -2.648 | 0.0081 | TAK | GJR-GARCH(1,1,1) |
| AAPL | LSTM | GRU | +0.586 | 0.5580 | NIE | brak |
| NVDA | GARCH(1,1) | EGARCH(1,1) | -0.081 | 0.9351 | NIE | brak |
| NVDA | GARCH(1,1) | GJR-GARCH(1,1,1) | +3.083 | 0.0020 | TAK | GJR-GARCH(1,1,1) |
| NVDA | GARCH(1,1) | LSTM | +2.274 | 0.0230 | TAK | LSTM |
| NVDA | GARCH(1,1) | GRU | +3.259 | 0.0011 | TAK | GRU |
| NVDA | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.496 | 0.1347 | NIE | brak |
| NVDA | EGARCH(1,1) | LSTM | +1.557 | 0.1194 | NIE | brak |
| NVDA | EGARCH(1,1) | GRU | +2.193 | 0.0283 | TAK | GRU |
| NVDA | GJR-GARCH(1,1,1) | LSTM | +0.613 | 0.5401 | NIE | brak |
| NVDA | GJR-GARCH(1,1,1) | GRU | +1.731 | 0.0835 | NIE | brak |
| NVDA | LSTM | GRU | +1.556 | 0.1196 | NIE | brak |
| JPM | GARCH(1,1) | EGARCH(1,1) | +2.024 | 0.0429 | TAK | EGARCH(1,1) |
| JPM | GARCH(1,1) | GJR-GARCH(1,1,1) | +0.743 | 0.4576 | NIE | brak |
| JPM | GARCH(1,1) | LSTM | +2.188 | 0.0287 | TAK | LSTM |
| JPM | GARCH(1,1) | GRU | +2.359 | 0.0183 | TAK | GRU |
| JPM | EGARCH(1,1) | GJR-GARCH(1,1,1) | -0.816 | 0.4144 | NIE | brak |
| JPM | EGARCH(1,1) | LSTM | +1.451 | 0.1469 | NIE | brak |
| JPM | EGARCH(1,1) | GRU | +2.094 | 0.0362 | TAK | GRU |
| JPM | GJR-GARCH(1,1,1) | LSTM | +2.300 | 0.0215 | TAK | LSTM |
| JPM | GJR-GARCH(1,1,1) | GRU | +2.547 | 0.0109 | TAK | GRU |
| JPM | LSTM | GRU | +0.627 | 0.5309 | NIE | brak |
| XOM | GARCH(1,1) | EGARCH(1,1) | -1.369 | 0.1709 | NIE | brak |
| XOM | GARCH(1,1) | GJR-GARCH(1,1,1) | +0.612 | 0.5404 | NIE | brak |
| XOM | GARCH(1,1) | LSTM | +0.866 | 0.3864 | NIE | brak |
| XOM | GARCH(1,1) | GRU | -1.030 | 0.3031 | NIE | brak |
| XOM | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.398 | 0.1620 | NIE | brak |
| XOM | EGARCH(1,1) | LSTM | +1.246 | 0.2127 | NIE | brak |
| XOM | EGARCH(1,1) | GRU | -0.542 | 0.5877 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | LSTM | +0.821 | 0.4115 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | GRU | -1.291 | 0.1968 | NIE | brak |
| XOM | LSTM | GRU | -2.184 | 0.0290 | TAK | LSTM |
| SCCO | GARCH(1,1) | EGARCH(1,1) | +0.640 | 0.5224 | NIE | brak |
| SCCO | GARCH(1,1) | GJR-GARCH(1,1,1) | -1.067 | 0.2860 | NIE | brak |
| SCCO | GARCH(1,1) | LSTM | +1.480 | 0.1387 | NIE | brak |
| SCCO | GARCH(1,1) | GRU | -0.911 | 0.3624 | NIE | brak |
| SCCO | EGARCH(1,1) | GJR-GARCH(1,1,1) | -1.098 | 0.2722 | NIE | brak |
| SCCO | EGARCH(1,1) | LSTM | +0.830 | 0.4064 | NIE | brak |
| SCCO | EGARCH(1,1) | GRU | -1.083 | 0.2788 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | LSTM | +2.654 | 0.0079 | TAK | LSTM |
| SCCO | GJR-GARCH(1,1,1) | GRU | -0.399 | 0.6900 | NIE | brak |
| SCCO | LSTM | GRU | -2.252 | 0.0243 | TAK | LSTM |