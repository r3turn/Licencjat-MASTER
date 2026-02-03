# utils/walk_forward.py - Walk-Forward Validation dla modeli GARCH

import numpy as np
from arch import arch_model
from tqdm import tqdm


def walk_forward_garch(returns_series, model_config, initial_train_size, refit_every=50):
    """
    Walk-Forward Validation dla modeli rodziny GARCH.

    Expanding window: okno treningowe rośnie z każdym krokiem.
    Model jest re-fitowany co `refit_every` dni dla wydajności.

    Parameters:
    -----------
    returns_series : pd.Series lub np.array
        Szereg logarytmicznych stóp zwrotu (NIE przeskalowany)
    model_config : dict
        Konfiguracja arch_model, np. {'vol': 'GARCH', 'p': 1, 'q': 1}
    initial_train_size : int
        Początkowy rozmiar okna treningowego (np. 1000 dni)
    refit_every : int
        Co ile dni re-fitować model (domyślnie 50)

    Returns:
    --------
    dict z kluczami:
        - forecasts: np.array prognoz σ² (wariancji)
        - realized: np.array zrealizowanej zmienności (r²)
        - dates: indeks dat (jeśli pd.Series)
        - fit_count: ile razy model był fitowany
    """
    # Konwersja do numpy i skalowanie (*100 dla stabilności GARCH)
    if hasattr(returns_series, 'values'):
        y = returns_series.values * 100
        dates = returns_series.index[initial_train_size:]
    else:
        y = np.array(returns_series) * 100
        dates = None

    n = len(y)
    forecasts = []
    realized = []

    last_fit = None
    fit_count = 0

    # Progress bar
    pbar = tqdm(range(initial_train_size, n), desc="Walk-forward")

    for t in pbar:
        # Re-fit model co refit_every dni lub na początku
        if last_fit is None or (t - initial_train_size) % refit_every == 0:
            try:
                model = arch_model(
                    y[:t],
                    mean='Constant',
                    dist='t',  # t-distribution dla grubych ogonów
                    **model_config
                )
                last_fit = model.fit(disp='off', show_warning=False)
                fit_count += 1
            except Exception as e:
                # Jeśli fit się nie uda, użyj poprzedniego
                if last_fit is None:
                    raise RuntimeError(f"Pierwszy fit się nie powiódł: {e}")

        # Prognoza na okres t (1-step ahead)
        try:
            fcast = last_fit.forecast(horizon=1)
            variance_forecast = fcast.variance.values[-1, 0]

            # Przeskaluj z powrotem (/100)² = /10000
            sigma2_pred = variance_forecast / 10000

        except Exception:
            # Fallback: użyj ostatniej conditional variance
            sigma2_pred = (last_fit.conditional_volatility[-1] / 100) ** 2

        # Realized volatility = r²
        sigma2_real = (y[t] / 100) ** 2

        forecasts.append(sigma2_pred)
        realized.append(sigma2_real)

    return {
        'forecasts': np.array(forecasts),
        'realized': np.array(realized),
        'dates': dates,
        'fit_count': fit_count,
    }


def walk_forward_garch_variance_targeting(returns_series, model_config, initial_train_size, refit_every=50):
    """
    Walk-Forward z variance targeting (alternatywna parametryzacja).

    Variance targeting: ω jest obliczane z unconditional variance,
    co może poprawić zbieżność w trudnych przypadkach.
    """
    if hasattr(returns_series, 'values'):
        y = returns_series.values * 100
        dates = returns_series.index[initial_train_size:]
    else:
        y = np.array(returns_series) * 100
        dates = None

    n = len(y)
    forecasts = []
    realized = []

    last_fit = None
    fit_count = 0

    pbar = tqdm(range(initial_train_size, n), desc="Walk-forward (VT)")

    for t in pbar:
        if last_fit is None or (t - initial_train_size) % refit_every == 0:
            try:
                # Variance targeting dla lepszej zbieżności
                model = arch_model(
                    y[:t],
                    mean='Constant',
                    dist='t',
                    rescale=True,  # automatyczne skalowanie
                    **model_config
                )
                last_fit = model.fit(
                    disp='off',
                    show_warning=False,
                    options={'maxiter': 500}
                )
                fit_count += 1
            except Exception as e:
                if last_fit is None:
                    raise RuntimeError(f"Pierwszy fit się nie powiódł: {e}")

        try:
            fcast = last_fit.forecast(horizon=1)
            # Przy rescale=True, variance jest już w oryginalnej skali
            variance_forecast = fcast.variance.values[-1, 0]
            sigma2_pred = variance_forecast / 10000
        except Exception:
            sigma2_pred = (last_fit.conditional_volatility[-1] / 100) ** 2

        sigma2_real = (y[t] / 100) ** 2

        forecasts.append(sigma2_pred)
        realized.append(sigma2_real)

    return {
        'forecasts': np.array(forecasts),
        'realized': np.array(realized),
        'dates': dates,
        'fit_count': fit_count,
    }
