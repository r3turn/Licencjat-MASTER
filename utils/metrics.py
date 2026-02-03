# utils/metrics.py - Metryki oceny prognoz zmienności

import numpy as np
from scipy import stats


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

    Standardowa miara błędu prognozy.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.

    Bardziej odporna na outliers niż RMSE.
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred, epsilon=1e-10):
    """
    Mean Absolute Percentage Error.

    Uwaga: problematyczna gdy y_true bliskie zeru.
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def qlike(y_true, y_pred, epsilon=1e-10):
    """
    QLIKE Loss - standard w literaturze o zmienności.

    Asymetryczna funkcja straty - karze niedoszacowanie
    zmienności bardziej niż przeszacowanie.

    QLIKE = mean(log(σ²_pred) + σ²_true / σ²_pred)

    Mniejsze = lepsze.
    """
    y_pred = np.maximum(y_pred, epsilon)  # unikaj log(0)
    return np.mean(np.log(y_pred) + y_true / y_pred)


def diebold_mariano_test(errors_1, errors_2, h=1):
    """
    Test Diebolda-Mariano dla porównania prognoz.

    H0: Oba modele mają równą dokładność prognoz
    H1: Modele różnią się dokładnością

    Parameters:
    -----------
    errors_1 : array - błędy prognozy modelu 1
    errors_2 : array - błędy prognozy modelu 2
    h : int - horyzont prognozy (dla korekty autokowariancji)

    Returns:
    --------
    dm_stat : float - statystyka testowa DM
    p_value : float - wartość p (dwustronna)
    """
    # Różnica strat (squared errors)
    d = errors_1 ** 2 - errors_2 ** 2
    n = len(d)

    # Średnia różnica
    mean_d = np.mean(d)

    # Autokowariancja dla korekty (Newey-West style)
    gamma_0 = np.var(d, ddof=1)

    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0
        gamma_sum += 2 * gamma_k

    # Wariancja z korektą
    var_d = (gamma_0 + gamma_sum) / n

    # Statystyka DM
    if var_d > 0:
        dm_stat = mean_d / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        dm_stat = 0
        p_value = 1.0

    return dm_stat, p_value


def calculate_all_metrics(y_true, y_pred):
    """
    Oblicz wszystkie metryki naraz.

    Returns:
    --------
    dict z kluczami: rmse, mae, mape, qlike
    """
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'qlike': qlike(y_true, y_pred),
    }
