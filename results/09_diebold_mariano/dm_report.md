# Test Diebolda-Mariano - Wyniki

Test statystyczny porównujący dokładność prognoz modeli.

- **Poziom istotności:** α = 0.05
- **H0:** Oba modele mają równą dokładność
- **H1:** Modele różnią się dokładnością

## Podsumowanie zwycięstw

| Model | Istotne zwycięstwa |
|-------|-------------------|
| GRU | 10 |
| EGARCH(1,1) | 7 |
| LSTM | 6 |
| GARCH(1,1) | 4 |
| GJR-GARCH(1,1,1) | 2 |

## Sieci neuronowe vs GARCH

- Sieci neuronowe wygrywają: **14**
- GARCH wygrywają: **3**
- Brak istotnej różnicy: **13**

## LSTM vs GRU (bezpośrednie porównanie)

| Ticker | p-value | Wynik |
|--------|---------|-------|
| AAPL | 0.6007 | brak różnicy |
| NVDA | 0.0439 | **GRU** |
| JPM | 0.0002 | **GRU** |
| XOM | 0.6964 | brak różnicy |
| SCCO | 0.3354 | brak różnicy |

## Szczegółowe wyniki

| Ticker | Model 1 | Model 2 | DM stat | p-value | Istotne? | Lepszy |
|--------|---------|---------|---------|---------|----------|--------|
| AAPL | GARCH(1,1) | EGARCH(1,1) | +9.592 | 0.0000 | TAK | EGARCH(1,1) |
| AAPL | GARCH(1,1) | GJR-GARCH(1,1,1) | -1.006 | 0.3146 | NIE | brak |
| AAPL | GARCH(1,1) | LSTM | +3.252 | 0.0011 | TAK | LSTM |
| AAPL | GARCH(1,1) | GRU | +2.971 | 0.0030 | TAK | GRU |
| AAPL | EGARCH(1,1) | GJR-GARCH(1,1,1) | -3.828 | 0.0001 | TAK | EGARCH(1,1) |
| AAPL | EGARCH(1,1) | LSTM | +2.815 | 0.0049 | TAK | LSTM |
| AAPL | EGARCH(1,1) | GRU | +2.611 | 0.0090 | TAK | GRU |
| AAPL | GJR-GARCH(1,1,1) | LSTM | +3.429 | 0.0006 | TAK | LSTM |
| AAPL | GJR-GARCH(1,1,1) | GRU | +3.130 | 0.0017 | TAK | GRU |
| AAPL | LSTM | GRU | +0.523 | 0.6007 | NIE | brak |
| NVDA | GARCH(1,1) | EGARCH(1,1) | -1.506 | 0.1321 | NIE | brak |
| NVDA | GARCH(1,1) | GJR-GARCH(1,1,1) | -3.720 | 0.0002 | TAK | GARCH(1,1) |
| NVDA | GARCH(1,1) | LSTM | +3.061 | 0.0022 | TAK | LSTM |
| NVDA | GARCH(1,1) | GRU | +3.229 | 0.0012 | TAK | GRU |
| NVDA | EGARCH(1,1) | GJR-GARCH(1,1,1) | -1.609 | 0.1076 | NIE | brak |
| NVDA | EGARCH(1,1) | LSTM | +3.224 | 0.0013 | TAK | LSTM |
| NVDA | EGARCH(1,1) | GRU | +3.385 | 0.0007 | TAK | GRU |
| NVDA | GJR-GARCH(1,1,1) | LSTM | +3.705 | 0.0002 | TAK | LSTM |
| NVDA | GJR-GARCH(1,1,1) | GRU | +3.765 | 0.0002 | TAK | GRU |
| NVDA | LSTM | GRU | +2.015 | 0.0439 | TAK | GRU |
| JPM | GARCH(1,1) | EGARCH(1,1) | -3.126 | 0.0018 | TAK | GARCH(1,1) |
| JPM | GARCH(1,1) | GJR-GARCH(1,1,1) | +0.045 | 0.9644 | NIE | brak |
| JPM | GARCH(1,1) | LSTM | -2.481 | 0.0131 | TAK | GARCH(1,1) |
| JPM | GARCH(1,1) | GRU | +1.093 | 0.2744 | NIE | brak |
| JPM | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.456 | 0.1455 | NIE | brak |
| JPM | EGARCH(1,1) | LSTM | -2.146 | 0.0319 | TAK | EGARCH(1,1) |
| JPM | EGARCH(1,1) | GRU | +1.805 | 0.0711 | NIE | brak |
| JPM | GJR-GARCH(1,1,1) | LSTM | -2.481 | 0.0131 | TAK | GJR-GARCH(1,1,1) |
| JPM | GJR-GARCH(1,1,1) | GRU | +1.093 | 0.2742 | NIE | brak |
| JPM | LSTM | GRU | +3.717 | 0.0002 | TAK | GRU |
| XOM | GARCH(1,1) | EGARCH(1,1) | +3.317 | 0.0009 | TAK | EGARCH(1,1) |
| XOM | GARCH(1,1) | GJR-GARCH(1,1,1) | +2.027 | 0.0427 | TAK | GJR-GARCH(1,1,1) |
| XOM | GARCH(1,1) | LSTM | +1.863 | 0.0625 | NIE | brak |
| XOM | GARCH(1,1) | GRU | +2.224 | 0.0262 | TAK | GRU |
| XOM | EGARCH(1,1) | GJR-GARCH(1,1,1) | -3.916 | 0.0001 | TAK | EGARCH(1,1) |
| XOM | EGARCH(1,1) | LSTM | +1.588 | 0.1122 | NIE | brak |
| XOM | EGARCH(1,1) | GRU | +1.900 | 0.0574 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | LSTM | +1.717 | 0.0860 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | GRU | +2.071 | 0.0383 | TAK | GRU |
| XOM | LSTM | GRU | -0.390 | 0.6964 | NIE | brak |
| SCCO | GARCH(1,1) | EGARCH(1,1) | +4.435 | 0.0000 | TAK | EGARCH(1,1) |
| SCCO | GARCH(1,1) | GJR-GARCH(1,1,1) | -3.144 | 0.0017 | TAK | GARCH(1,1) |
| SCCO | GARCH(1,1) | LSTM | +1.623 | 0.1045 | NIE | brak |
| SCCO | GARCH(1,1) | GRU | +1.072 | 0.2838 | NIE | brak |
| SCCO | EGARCH(1,1) | GJR-GARCH(1,1,1) | -5.138 | 0.0000 | TAK | EGARCH(1,1) |
| SCCO | EGARCH(1,1) | LSTM | +1.297 | 0.1945 | NIE | brak |
| SCCO | EGARCH(1,1) | GRU | +0.774 | 0.4390 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | LSTM | +1.723 | 0.0848 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | GRU | +1.156 | 0.2475 | NIE | brak |
| SCCO | LSTM | GRU | -0.963 | 0.3354 | NIE | brak |