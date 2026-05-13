# utils/walk_forward.py - Walk-Forward Validation dla modeli GARCH

import numpy as np
from arch import arch_model
from tqdm import tqdm
from scipy.special import gamma as _gamma_fn


def _garch_step(vol_type, o_param, params, eps, last_sigma2):
    """
    Jedna iteracja rekursji GARCH — aktualizuje wariancję warunkową.

    Parametry i wartości są w przestrzeni skalowanej (×100).
    Obsługuje GARCH, GJR-GARCH i EGARCH.
    """
    omega = float(params['omega'])
    beta  = float(params['beta[1]'])

    if vol_type == 'EGARCH':
        alpha = float(params.get('alpha[1]', 0.0))
        gamma = float(params.get('gamma[1]', 0.0))
        nu    = float(params.get('nu', 8.0))
        sigma = np.sqrt(max(last_sigma2, 1e-10))
        z = eps / sigma
        # E[|z|] dla standaryzowanego rozkładu t(nu)
        E_abs_z = (2.0 * np.sqrt(nu - 2.0) * _gamma_fn((nu + 1.0) / 2.0) /
                   (_gamma_fn(nu / 2.0) * (nu - 1.0) * np.sqrt(np.pi)))
        log_h = (omega
                 + beta  * np.log(max(last_sigma2, 1e-10))
                 + alpha * (abs(z) - E_abs_z)
                 + gamma * z)
        return np.exp(np.clip(log_h, -30, 30))
    else:
        # GARCH(1,1) lub GJR-GARCH(1,1,1)
        alpha = float(params['alpha[1]'])
        gamma = float(params.get('gamma[1]', 0.0)) if o_param > 0 else 0.0
        I_neg = 1.0 if eps < 0.0 else 0.0
        h = omega + (alpha + gamma * I_neg) * eps ** 2 + beta * last_sigma2
        return max(h, 1e-10)


def walk_forward_garch(returns_series, model_config, initial_train_size, refit_every=50):
    """
    Walk-Forward Validation dla modeli rodziny GARCH.

    Expanding window: okno treningowe rośnie z każdym krokiem.
    Parametry re-estymowane co `refit_every` dni; między re-fitami
    wariancja warunkowa aktualizowana jest codziennie przez rekursję GARCH
    z bieżącą obserwacją y[t-1] — prawidłowy one-step-ahead forecast.

    Parameters:
    -----------
    returns_series : pd.Series lub np.array
        Szereg logarytmicznych stóp zwrotu (NIE przeskalowany)
    model_config : dict
        Konfiguracja arch_model, np. {'vol': 'GARCH', 'p': 1, 'q': 1}
    initial_train_size : int
        Początkowy rozmiar okna treningowego (np. 1000 dni)
    refit_every : int
        Co ile dni re-estymować parametry (domyślnie 50)

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
    realized  = []

    last_fit      = None
    fit_count     = 0
    last_params   = None
    last_mu       = 0.0
    last_sigma2   = None   # wariancja warunkowa w przestrzeni skalowanej

    vol_type = model_config.get('vol', 'GARCH').upper()
    o_param  = model_config.get('o', 0)

    pbar = tqdm(range(initial_train_size, n), desc="Walk-forward")

    for t in pbar:
        # Re-estymacja parametrów co refit_every dni
        if last_fit is None or (t - initial_train_size) % refit_every == 0:
            try:
                model = arch_model(
                    y[:t],
                    mean='Constant',
                    dist='t',
                    **model_config
                )
                last_fit    = model.fit(disp='off', show_warning=False)
                fit_count  += 1
                last_params = last_fit.params
                last_mu     = float(last_params.get('mu', 0.0))
                # Ostatnia wariancja warunkowa z dopasowanego modelu (σ²_{t-1})
                last_sigma2 = float(np.asarray(last_fit.conditional_volatility)[-1]) ** 2
            except Exception as e:
                if last_fit is None:
                    raise RuntimeError(f"Pierwszy fit się nie powiódł: {e}")

        # Aktualizacja rekursją GARCH: σ²_t = f(ε_{t-1}, σ²_{t-1})
        eps         = y[t - 1] - last_mu
        last_sigma2 = _garch_step(vol_type, o_param, last_params, eps, last_sigma2)

        # Prognoza w oryginalnej skali
        sigma2_pred = last_sigma2 / 10000

        # Realized volatility = r²
        sigma2_real = (y[t] / 100) ** 2

        forecasts.append(sigma2_pred)
        realized.append(sigma2_real)

    return {
        'forecasts': np.array(forecasts),
        'realized':  np.array(realized),
        'dates':     dates,
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
