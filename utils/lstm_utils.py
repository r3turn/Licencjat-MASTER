# utils/lstm_utils.py - LSTM dla prognozowania zmienności
#
# Zawiera:
# - Architekturę modelu LSTM
# - Trenowanie z early stopping
# - Ewaluację z podziałem 80/10/10 i sliding window na zbiorze testowym

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMVolatility(nn.Module):
    """
    LSTM do prognozowania zmienności (wariancji warunkowej).

    Architektura:
    - Warstwa LSTM przetwarzająca sekwencję stóp zwrotu
    - Dropout przed warstwą wyjściową
    - Warstwa FC + Softplus (gwarantuje dodatnią wariancję)
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return self.softplus(out).squeeze(-1)


def train_lstm(model, X_train, y_train, X_val, y_val,
               epochs=100, batch_size=32, learning_rate=0.001,
               patience=15, device='cpu', verbose=True):
    """
    Trenuj LSTM z early stopping na zbiorze walidacyjnym.

    Parameters
    ----------
    model : LSTMVolatility
    X_train, y_train : np.array — dane treningowe
    X_val, y_val : np.array — dane walidacyjne (early stopping)
    epochs : int
    batch_size : int
    learning_rate : float
    patience : int — liczba epok bez poprawy przed zatrzymaniem
    device : str
    verbose : bool

    Returns
    -------
    model : wytrenowany model
    train_losses, val_losses : list
    """
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss     = float('inf')
    best_model_state  = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} — train: {epoch_train_loss:.6f}, val: {val_loss:.6f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def static_split_lstm(returns_series, window_size, train_ratio=0.8, val_ratio=0.1,
                      hidden_size=32, num_layers=1, dropout=0.1,
                      epochs=100, batch_size=32, learning_rate=0.001,
                      patience=15, device='cpu', seed=42):
    """
    Ewaluacja LSTM z podziałem 80/10/10 i sliding window na zbiorze testowym.

    Procedura:
    1. Normalizacja z użyciem statystyk zbioru treningowego (zapobiega data leakage).
    2. Trenowanie na 80% train — zbiór walidacyjny (10%) wyłącznie dla early stopping.
    3. Prognozy 1-step-ahead metodą sliding window na 10% test (model nie jest re-trenowany).

    Dla każdego dnia t w zbiorze testowym:
        wejście:  ostatnie window_size stóp zwrotu [r_{t-L}, ..., r_{t-1}]
        wyjście:  prognoza σ²_t

    Parameters
    ----------
    returns_series : pd.Series
    window_size : int — długość sekwencji wejściowej (L)
    train_ratio, val_ratio : float
    hidden_size, num_layers, dropout : architektura LSTM
    epochs, batch_size, learning_rate, patience : parametry trenowania
    device : 'cpu' lub 'cuda'
    seed : int

    Returns
    -------
    dict z kluczami: forecasts, realized, dates, train_size, val_size, test_size
    """
    from params import set_seed
    set_seed(seed)

    if hasattr(returns_series, 'values'):
        returns   = returns_series.values
        all_dates = returns_series.index
    else:
        returns   = np.array(returns_series)
        all_dates = None

    n          = len(returns)
    train_size = int(n * train_ratio)
    val_size   = int(n * val_ratio)
    test_start = train_size + val_size

    print(f"  Podział: train={train_size}, val={val_size}, test={n - test_start}")

    # 1. Normalizacja — parametry wyłącznie ze zbioru treningowego
    train_mean   = np.mean(returns[:train_size])
    train_std    = np.std(returns[:train_size]) + 1e-8
    returns_norm = (returns - train_mean) / train_std

    # 2. Sekwencje treningowe — cel w zbiorze treningowym
    X_train_list, y_train_list = [], []
    for i in range(window_size, train_size):
        X_train_list.append(returns_norm[i - window_size:i])
        y_train_list.append(returns_norm[i] ** 2)
    X_train = np.array(X_train_list).reshape(-1, window_size, 1)
    y_train = np.array(y_train_list)

    # Sekwencje walidacyjne — cel w zbiorze walidacyjnym
    X_val_list, y_val_list = [], []
    for i in range(train_size, test_start):
        X_val_list.append(returns_norm[i - window_size:i])
        y_val_list.append(returns_norm[i] ** 2)
    X_val = np.array(X_val_list).reshape(-1, window_size, 1)
    y_val = np.array(y_val_list)

    # 3. Trenowanie z early stopping
    model = LSTMVolatility(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
    model, _, _ = train_lstm(model, X_train, y_train, X_val, y_val,
                             epochs=epochs, batch_size=batch_size,
                             learning_rate=learning_rate, patience=patience,
                             device=device, verbose=True)

    # 4. Sliding window predictions na zbiorze testowym
    model.eval()
    forecasts = []
    realized  = []

    for i in range(test_start, n):
        input_seq = returns_norm[i - window_size:i].reshape(1, window_size, 1)
        X_pred    = torch.FloatTensor(input_seq).to(device)

        with torch.no_grad():
            sigma2_norm = model(X_pred).item()

        # Przeskalowanie do oryginalnej skali: var_orig = var_norm * std²
        forecasts.append(sigma2_norm * (train_std ** 2))
        realized.append(returns[i] ** 2)

    dates = all_dates[test_start:] if all_dates is not None else None

    return {
        'forecasts':  np.array(forecasts),
        'realized':   np.array(realized),
        'dates':      dates,
        'train_size': train_size,
        'val_size':   val_size,
        'test_size':  n - test_start,
    }
