import pandas as pd
import numpy as np
from arch import arch_model
import os

BOLD_GREEN = "\033[1;92m"
RESET = "\033[0m"

# LOAD DATA
returns = pd.read_parquet("data/returns.parquet")
metrics_list = []

models_config = [
    {'name': 'GARCH',  'vol': 'Garch',  'p': 1, 'o': 0, 'q': 1},
    {'name': 'EGARCH', 'vol': 'EGarch', 'p': 1, 'o': 0, 'q': 1},
    {'name': 'GJR',    'vol': 'Garch',  'p': 1, 'o': 1, 'q': 1}
]

for ticker in returns.columns:
    y = returns[ticker] * 100 
    
    for config in models_config:
        try:
            model = arch_model(y, vol=config['vol'], p=config['p'], o=config['o'], q=config['q'], dist='Normal')
            res = model.fit(disp='off')
            
            metrics_list.append({
                'Ticker': ticker,
                'Model': config['name'],
                'AIC': res.aic,
                'BIC': res.bic,
                'LogLik': res.loglikelihood
            })
        except:
            pass

metrics_df = pd.DataFrame(metrics_list)

# CONSOLE LOG
desired_order = ['GJR', 'EGARCH', 'GARCH']
metrics_df['Model'] = pd.Categorical(metrics_df['Model'], categories=desired_order, ordered=True)
metrics_df = metrics_df.sort_values(by=['Ticker', 'Model'])

# Save to CSV
metrics_df.to_csv("data/garch_metrics.csv", index=False)

# CONSOLE VIEW
best_aic = metrics_df.groupby('Ticker')['AIC'].min()
best_bic = metrics_df.groupby('Ticker')['BIC'].min()
best_ll  = metrics_df.groupby('Ticker')['LogLik'].max()

print("\n" + "="*85)
# header
print(f"{'TICKER':<8} {'MODEL':<8} {'AIC (Min)':>12}      {'BIC (Min)':>12}      {'LogLik (Max)':>12}")
print("="*85)

current_ticker = ""

for index, row in metrics_df.iterrows():
    t = row['Ticker']
    m = row['Model']
    
    val_aic = row['AIC']
    val_bic = row['BIC']
    val_ll = row['LogLik']
    
    # AIC
    if val_aic == best_aic[t]:
        str_aic = f"{BOLD_GREEN}{val_aic:12.2f}{RESET}"
    else:
        str_aic = f"{val_aic:12.2f}"

    # BIC
    if val_bic == best_bic[t]:
        str_bic = f"{BOLD_GREEN}{val_bic:12.2f}{RESET}"
    else:
        str_bic = f"{val_bic:12.2f}"

    # LogLik
    if val_ll == best_ll[t]:
        str_ll = f"{BOLD_GREEN}{val_ll:12.2f}{RESET}"
    else:
        str_ll = f"{val_ll:12.2f}"

    # Separator
    if t != current_ticker and current_ticker != "":
        print("-" * 85)
    current_ticker = t
    
    print(f"{t:<8} {m:<8} {str_aic}     {str_bic}     {str_ll}")

print("="*85)
print(f"\n{BOLD_GREEN}CSV saved{RESET}")