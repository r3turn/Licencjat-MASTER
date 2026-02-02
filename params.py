# params.py - Konfiguracja projektu

# Instrumenty (zróżnicowanie sektorowe)
TICKERS = [
    "AAPL",   # Technologia (USA) - high beta
    "NVDA",   # Technologia (USA) - high beta
    "JPM",    # Finanse (USA) - sektor bankowy
    "XOM",    # Energia (USA) - sektor surowcowy
    "SCCO",   # Southern Copper - mining
]

# Okres badawczy
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

# Walk-forward validation (expanding window)
INITIAL_TRAIN_SIZE = 1000  # Pierwsze okno treningowe (dni) - ~4 lata
FORECAST_HORIZON = 1       # Prognoza 1 dzień do przodu
REFIT_EVERY = 50           # Re-trenuj model co X dni

# Sensitivity analysis (robustness check)
TRAIN_SIZE_SENSITIVITY = [500, 750, 1000, 1250, 1500]

# LSTM/GRU (na później)
WINDOW_SIZE = 20           # Lookback dla sieci
VAL_RATIO = 0.1            # Validation do early stopping

# Testy statystyczne
ARCH_NLAGS = 5             # Liczba lagów w ARCH test

# Reproducibility
SEED = 42


def set_seed(seed=SEED):
    """Ustaw seedy dla reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch nie zainstalowany
