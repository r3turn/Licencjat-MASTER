# params.py - Konfiguracja projektu

# Instrumenty (zróżnicowanie sektorowe)
TICKERS = [
    "AAPL",   # Technologia (USA)
    "NVDA",   # Technologia (USA)
    "JPM",    # Finanse (USA)
    "XOM",    # Energia (USA)
    "SCCO",   # Surowce (miedź)
]

# Okres badawczy
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

# Podział danych (chronologiczny, stały — bez nakładania się zbiorów)
TRAIN_RATIO = 0.8   # 80% — trenowanie modeli
VAL_RATIO   = 0.1   # 10% — walidacja (early stopping dla NN / propagacja stanu dla GARCH)
TEST_RATIO  = 0.1   # 10% — finalna, jednorazowa ocena modeli

# Sliding window dla sieci neuronowych (L = 30 dni lookback)
WINDOW_SIZE = 30

# Testy statystyczne
ARCH_NLAGS = 5

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
        pass
