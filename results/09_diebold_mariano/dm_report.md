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
| GARCH(1,1) | 4 |
| LSTM | 4 |
| GJR-GARCH(1,1,1) | 2 |

## Sieci neuronowe vs GARCH

- Sieci neuronowe wygrywają: **12**
- GARCH wygrywają: **3**
- Brak istotnej różnicy: **15**

## LSTM vs GRU (bezpośrednie porównanie)

| Ticker | p-value | Wynik |
|--------|---------|-------|
| AAPL | 0.3515 | brak różnicy |
| NVDA | 0.0000 | **GRU** |
| JPM | 0.0000 | **GRU** |
| XOM | 0.2714 | brak różnicy |
| SCCO | 0.4464 | brak różnicy |

## Szczegółowe wyniki

| Ticker | Model 1 | Model 2 | DM stat | p-value | Istotne? | Lepszy |
|--------|---------|---------|---------|---------|----------|--------|
| AAPL | GARCH(1,1) | EGARCH(1,1) | +9.592 | 0.0000 | TAK | EGARCH(1,1) |
| AAPL | GARCH(1,1) | GJR-GARCH(1,1,1) | -1.006 | 0.3145 | NIE | brak |
| AAPL | GARCH(1,1) | LSTM | +3.324 | 0.0009 | TAK | LSTM |
| AAPL | GARCH(1,1) | GRU | +2.958 | 0.0031 | TAK | GRU |
| AAPL | EGARCH(1,1) | GJR-GARCH(1,1,1) | -3.828 | 0.0001 | TAK | EGARCH(1,1) |
| AAPL | EGARCH(1,1) | LSTM | +2.813 | 0.0049 | TAK | LSTM |
| AAPL | EGARCH(1,1) | GRU | +2.598 | 0.0094 | TAK | GRU |
| AAPL | GJR-GARCH(1,1,1) | LSTM | +3.517 | 0.0004 | TAK | LSTM |
| AAPL | GJR-GARCH(1,1,1) | GRU | +3.117 | 0.0018 | TAK | GRU |
| AAPL | LSTM | GRU | +0.932 | 0.3515 | NIE | brak |
| NVDA | GARCH(1,1) | EGARCH(1,1) | -1.506 | 0.1321 | NIE | brak |
| NVDA | GARCH(1,1) | GJR-GARCH(1,1,1) | -3.726 | 0.0002 | TAK | GARCH(1,1) |
| NVDA | GARCH(1,1) | LSTM | +1.753 | 0.0796 | NIE | brak |
| NVDA | GARCH(1,1) | GRU | +3.348 | 0.0008 | TAK | GRU |
| NVDA | EGARCH(1,1) | GJR-GARCH(1,1,1) | -1.613 | 0.1067 | NIE | brak |
| NVDA | EGARCH(1,1) | LSTM | +1.951 | 0.0510 | NIE | brak |
| NVDA | EGARCH(1,1) | GRU | +3.500 | 0.0005 | TAK | GRU |
| NVDA | GJR-GARCH(1,1,1) | LSTM | +2.443 | 0.0146 | TAK | LSTM |
| NVDA | GJR-GARCH(1,1,1) | GRU | +3.882 | 0.0001 | TAK | GRU |
| NVDA | LSTM | GRU | +4.923 | 0.0000 | TAK | GRU |
| JPM | GARCH(1,1) | EGARCH(1,1) | -3.126 | 0.0018 | TAK | GARCH(1,1) |
| JPM | GARCH(1,1) | GJR-GARCH(1,1,1) | +0.045 | 0.9644 | NIE | brak |
| JPM | GARCH(1,1) | LSTM | -3.397 | 0.0007 | TAK | GARCH(1,1) |
| JPM | GARCH(1,1) | GRU | +1.049 | 0.2941 | NIE | brak |
| JPM | EGARCH(1,1) | GJR-GARCH(1,1,1) | +1.456 | 0.1455 | NIE | brak |
| JPM | EGARCH(1,1) | LSTM | -3.128 | 0.0018 | TAK | EGARCH(1,1) |
| JPM | EGARCH(1,1) | GRU | +1.756 | 0.0790 | NIE | brak |
| JPM | GJR-GARCH(1,1,1) | LSTM | -3.454 | 0.0006 | TAK | GJR-GARCH(1,1,1) |
| JPM | GJR-GARCH(1,1,1) | GRU | +1.049 | 0.2944 | NIE | brak |
| JPM | LSTM | GRU | +4.585 | 0.0000 | TAK | GRU |
| XOM | GARCH(1,1) | EGARCH(1,1) | +3.317 | 0.0009 | TAK | EGARCH(1,1) |
| XOM | GARCH(1,1) | GJR-GARCH(1,1,1) | +2.027 | 0.0427 | TAK | GJR-GARCH(1,1,1) |
| XOM | GARCH(1,1) | LSTM | +1.805 | 0.0710 | NIE | brak |
| XOM | GARCH(1,1) | GRU | +2.271 | 0.0232 | TAK | GRU |
| XOM | EGARCH(1,1) | GJR-GARCH(1,1,1) | -3.916 | 0.0001 | TAK | EGARCH(1,1) |
| XOM | EGARCH(1,1) | LSTM | +1.617 | 0.1059 | NIE | brak |
| XOM | EGARCH(1,1) | GRU | +1.947 | 0.0515 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | LSTM | +1.706 | 0.0881 | NIE | brak |
| XOM | GJR-GARCH(1,1,1) | GRU | +2.120 | 0.0340 | TAK | GRU |
| XOM | LSTM | GRU | -1.100 | 0.2714 | NIE | brak |
| SCCO | GARCH(1,1) | EGARCH(1,1) | +4.438 | 0.0000 | TAK | EGARCH(1,1) |
| SCCO | GARCH(1,1) | GJR-GARCH(1,1,1) | -3.169 | 0.0015 | TAK | GARCH(1,1) |
| SCCO | GARCH(1,1) | LSTM | +1.592 | 0.1114 | NIE | brak |
| SCCO | GARCH(1,1) | GRU | +1.070 | 0.2847 | NIE | brak |
| SCCO | EGARCH(1,1) | GJR-GARCH(1,1,1) | -5.141 | 0.0000 | TAK | EGARCH(1,1) |
| SCCO | EGARCH(1,1) | LSTM | +1.259 | 0.2080 | NIE | brak |
| SCCO | EGARCH(1,1) | GRU | +0.772 | 0.4401 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | LSTM | +1.694 | 0.0902 | NIE | brak |
| SCCO | GJR-GARCH(1,1,1) | GRU | +1.155 | 0.2481 | NIE | brak |
| SCCO | LSTM | GRU | -0.761 | 0.4464 | NIE | brak |