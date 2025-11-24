import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ticker setup
TICKER_LIST = ['AAPL', 'NVDA', 'JPM', 'XOM', 'KGH.WA']
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

# data download
raw_data = yf.download(tickers=TICKER_LIST, start=START_DATE, end=END_DATE, auto_adjust=False)
print(raw_data.head)

print(raw_data.columns)
df_prices = raw_data['Adj Close'].ffill()
df_volume = raw_data['Volume'].ffill()