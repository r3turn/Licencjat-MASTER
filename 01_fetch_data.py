import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from params import TICKERS, START_DATE, END_DATE, set_seed

# Reproducibility
set_seed()

# Utwórz foldery jeśli nie istnieją
os.makedirs("data/raw", exist_ok=True)
os.makedirs("charts/01_fetch_data", exist_ok=True)

# Pobierz dane
print(f"Pobieram dane dla: {TICKERS}")
print(f"Okres: {START_DATE} - {END_DATE}")

try:
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,  # Adjusted close jako 'Close'
        progress=True
    )
    if data.empty:
        raise ValueError("Pobrane dane są puste!")
except Exception as e:
    print(f"\nBŁĄD pobierania danych: {e}")
    print("Sprawdź połączenie z internetem lub spróbuj ponownie później.")
    sys.exit(1)

# Wyciągnij tylko Close prices
prices = data["Close"]
prices = prices.dropna()

# Zapisz
prices.to_parquet("data/raw/prices.parquet")
print(f"\nZapisano: data/raw/prices.parquet")
print(f"Shape: {prices.shape}")
print(f"Okres: {prices.index[0].date()} - {prices.index[-1].date()}")

# Wykres cen (znormalizowane do 100)
normalized = prices / prices.iloc[0] * 100

plt.figure(figsize=(12, 6))
for ticker in TICKERS:
    plt.plot(normalized.index, normalized[ticker], label=ticker, linewidth=0.8)

plt.title("Ceny akcji (znormalizowane do 100)")
plt.xlabel("Data")
plt.ylabel("Wartość (base=100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("charts/01_fetch_data/prices_normalized.png", dpi=150)
plt.close()

print(f"Wykres: charts/01_fetch_data/prices_normalized.png")
