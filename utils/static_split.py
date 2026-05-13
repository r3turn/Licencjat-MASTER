# utils/static_split.py - Ewaluacja modeli GARCH z podziałem 80/10/10

import numpy as np
from arch import arch_model
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
        E_abs_z = (2.0 * np.sqrt(nu - 2.0) * _gamma_fn((nu + 1.0) / 2.0) /
                   (_gamma_fn(nu / 2.0) * (nu - 1.0) * np.sqrt(np.pi)))
        log_h = (omega
                 + beta  * np.log(max(last_sigma2, 1e-10))
                 + alpha * (abs(z) - E_abs_z)
                 + gamma * z)
        return np.exp(np.clip(log_h, -30, 30))
    else:
        alpha = float(params['alpha[1]'])
        gamma = float(params.get('gamma[1]', 0.0)) if o_param > 0 else 0.0
        I_neg = 1.0 if eps < 0.0 else 0.0
        h = omega + (alpha + gamma * I_neg) * eps ** 2 + beta * last_sigma2
        return max(h, 1e-10)


def static_split_garch(returns_series, model_config, train_ratio=0.8, val_ratio=0.1):
    """
    Ewaluacja modeli GARCH z podziałem 80/10/10.

    Procedura:
    1. Estymacja parametrów na zbiorze treningowym (80%).
    2. Propagacja rekursji GARCH przez zbiór walidacyjny (10%) bez zbierania prognoz
       — aktualizacja stanu warunkowej wariancji.
    3. Prognozy 1-step-ahead metodą sliding window na zbiorze testowym (10%).

    Parametry modelu pozostają niezmienione od momentu estymacji na zbiorze treningowym
    (analogia do sieci neuronowych: wagi ustalone po treningu).

    Parameters
    ----------
    returns_series : pd.Series
        Szereg logarytmicznych stóp zwrotu.
    model_config : dict
        Konfiguracja arch_model, np. {'vol': 'GARCH', 'p': 1, 'q': 1}.
    train_ratio : float
        Udział zbioru treningowego (domyślnie 0.8).
    val_ratio : float
        Udział zbioru walidacyjnego (domyślnie 0.1).

    Returns
    -------
    dict z kluczami: forecasts, realized, dates, train_size, val_size, test_size
    """
    if hasattr(returns_series, 'values'):
        y_orig    = returns_series.values
        all_dates = returns_series.index
    else:
        y_orig    = np.array(returns_series)
        all_dates = None

    n = len(y_orig)
    y = y_orig * 100  # skalowanie (*100) dla stabilności numerycznej GARCH

    train_size = int(n * train_ratio)
    val_size   = int(n * val_ratio)
    test_start = train_size + val_size

    print(f"  Podział: train={train_size}, val={val_size}, test={n - test_start}")

    vol_type = model_config.get('vol', 'GARCH').upper()
    o_param  = model_config.get('o', 0)

    # 1. Estymacja parametrów na zbiorze treningowym
    model = arch_model(y[:train_size], mean='Constant', dist='t', **model_config)
    fit   = model.fit(disp='off', show_warning=False)

    params      = fit.params
    mu          = float(params.get('mu', 0.0))
    last_sigma2 = float(np.asarray(fit.conditional_volatility)[-1]) ** 2

    # 2. Propagacja rekursji przez zbiór walidacyjny (aktualizacja stanu)
    for t in range(train_size, test_start):
        eps         = y[t - 1] - mu
        last_sigma2 = _garch_step(vol_type, o_param, params, eps, last_sigma2)

    # 3. Prognozy 1-step-ahead na zbiorze testowym (sliding window)
    forecasts = []
    realized  = []

    for t in range(test_start, n):
        eps         = y[t - 1] - mu
        last_sigma2 = _garch_step(vol_type, o_param, params, eps, last_sigma2)

        forecasts.append(last_sigma2 / 10000)       # oryginalna skala
        realized.append((y[t] / 100) ** 2)          # r²

    dates = all_dates[test_start:] if all_dates is not None else None

    return {
        'forecasts':  np.array(forecasts),
        'realized':   np.array(realized),
        'dates':      dates,
        'train_size': train_size,
        'val_size':   val_size,
        'test_size':  n - test_start,
    }
