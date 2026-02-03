# utils/lstm_utils.py - LSTM dla prognozowania zmienności
#
# Zawiera:
# - Architekturę modelu LSTM
# - Przygotowanie sekwencji
# - Trenowanie z early stopping
# - Walk-forward validation

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class LSTMVolatility(nn.Module):
    """
    LSTM do prognozowania zmienności (wariancji).

    Architektura:
    - LSTM layer(s) przetwarzający sekwencję zwrotów
    - Fully connected layer na wyjściu
    - Softplus activation zapewniająca dodatnią wariancję
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM - batch_first=True oznacza input shape (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout przed warstwą wyjściową
        self.dropout = nn.Dropout(dropout)

        # Warstwa wyjściowa: hidden_size -> 1 (prognoza wariancji)
        self.fc = nn.Linear(hidden_size, 1)

        # Softplus zapewnia dodatnią wariancję (gładsze niż ReLU)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # LSTM output shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Bierzemy tylko ostatni timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Dropout + FC + Softplus
        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, 1)
        out = self.softplus(out)  # zapewnia > 0

        return out.squeeze(-1)  # (batch,)


def prepare_sequences(returns, window_size):
    """
    Przygotuj sekwencje dla LSTM.

    Parameters:
    -----------
    returns : np.array
        Szereg zwrotów (1D)
    window_size : int
        Długość sekwencji wejściowej (lookback)

    Returns:
    --------
    X : np.array, shape (n_samples, window_size, 1)
        Sekwencje wejściowe (zwroty)
    y : np.array, shape (n_samples,)
        Targety (r² - realized variance)
    """
    n = len(returns)
    X, y = [], []

    for i in range(window_size, n):
        # Input: ostatnie window_size zwrotów
        X.append(returns[i - window_size:i])
        # Target: r² następnego dnia (realized variance proxy)
        y.append(returns[i] ** 2)

    X = np.array(X).reshape(-1, window_size, 1)  # (samples, seq, features)
    y = np.array(y)

    return X, y


def train_lstm(model, X_train, y_train, X_val, y_val,
               epochs=100, batch_size=32, learning_rate=0.001,
               patience=10, device='cpu', verbose=False):
    """
    Trenuj LSTM z early stopping.

    Parameters:
    -----------
    model : LSTMVolatility
    X_train, y_train : np.array - dane treningowe
    X_val, y_val : np.array - dane walidacyjne
    epochs : int - max liczba epok
    batch_size : int
    learning_rate : float
    patience : int - ile epok bez poprawy przed zatrzymaniem
    device : str - 'cpu' lub 'cuda'
    verbose : bool - czy printować postęp

    Returns:
    --------
    model : wytrenowany model
    train_losses : list
    val_losses : list
    """
    model = model.to(device)

    # Konwersja do tensorów
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # DataLoader dla mini-batch training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer i loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)

        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {epoch_train_loss:.6f}, Val: {val_loss:.6f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Przywróć najlepszy model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def walk_forward_lstm(returns_series, window_size, initial_train_size,
                      refit_every=250, val_ratio=0.1,
                      hidden_size=32, num_layers=1, dropout=0.1,
                      epochs=100, batch_size=32, learning_rate=0.001,
                      patience=10, device='cpu', seed=42):
    """
    Walk-Forward Validation dla LSTM.

    Expanding window: okno treningowe rośnie z każdym krokiem.
    Model jest re-trenowany co `refit_every` dni.

    UWAGA: refit_every dla LSTM powinno być większe niż dla GARCH
    (zalecane: 250+) ze względu na czas trenowania.

    Parameters:
    -----------
    returns_series : pd.Series lub np.array
        Szereg logarytmicznych stóp zwrotu
    window_size : int
        Lookback window dla LSTM (np. 20)
    initial_train_size : int
        Początkowy rozmiar okna treningowego (np. 1000)
    refit_every : int
        Co ile dni re-trenować model (domyślnie 250)
    val_ratio : float
        Procent danych treningowych na walidację (0.1 = 10%)
    hidden_size, num_layers, dropout : hiperparametry LSTM
    epochs, batch_size, learning_rate, patience : parametry trenowania
    device : 'cpu' lub 'cuda'
    seed : int - dla reproducibility

    Returns:
    --------
    dict z kluczami:
        - forecasts: np.array prognoz σ²
        - realized: np.array zrealizowanej zmienności (r²)
        - dates: indeks dat
        - fit_count: ile razy model był trenowany
    """
    from params import set_seed

    # Konwersja do numpy
    if hasattr(returns_series, 'values'):
        returns = returns_series.values
        dates = returns_series.index[initial_train_size:]
    else:
        returns = np.array(returns_series)
        dates = None

    n = len(returns)
    forecasts = []
    realized = []

    model = None
    fit_count = 0

    # Skalowanie danych (zapisujemy parametry z pierwszego fitu)
    train_mean = None
    train_std = None

    # Progress bar
    pbar = tqdm(range(initial_train_size, n), desc="Walk-forward LSTM")

    for t in pbar:
        # Re-fit model co refit_every dni lub na początku
        if model is None or (t - initial_train_size) % refit_every == 0:
            set_seed(seed)  # reproducibility przy każdym treningu

            # Dane treningowe do momentu t
            train_returns = returns[:t]

            # Normalizacja (fit na train, transform na wszystko)
            train_mean = np.mean(train_returns)
            train_std = np.std(train_returns) + 1e-8
            train_returns_norm = (train_returns - train_mean) / train_std

            # Przygotuj sekwencje
            X_all, y_all = prepare_sequences(train_returns_norm, window_size)

            # Target scaling: y to r², skalujemy przez std²
            # Ale ponieważ train_returns_norm ma std~1, y_all jest już w dobrej skali

            # Podział na train/val (ostatnie val_ratio% na walidację)
            n_train = len(X_all)
            n_val = max(1, int(n_train * val_ratio))
            n_train_actual = n_train - n_val

            X_train, y_train = X_all[:n_train_actual], y_all[:n_train_actual]
            X_val, y_val = X_all[n_train_actual:], y_all[n_train_actual:]

            # Inicjalizuj i trenuj model
            model = LSTMVolatility(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

            model, _, _ = train_lstm(
                model, X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, patience=patience,
                device=device, verbose=False
            )

            fit_count += 1
            pbar.set_postfix({'fits': fit_count})

        # Prognoza na okres t
        model.eval()

        # Sekwencja wejściowa: ostatnie window_size zwrotów przed t
        input_seq = returns[t - window_size:t]
        input_seq_norm = (input_seq - train_mean) / train_std

        # Konwersja do tensora
        X_pred = torch.FloatTensor(input_seq_norm.reshape(1, window_size, 1))
        X_pred = X_pred.to(next(model.parameters()).device)

        with torch.no_grad():
            # Prognoza w skali znormalizowanej
            sigma2_pred_norm = model(X_pred).item()

        # Przeskaluj z powrotem: variance w oryginalnej skali
        # Jeśli r_norm = r / std, to r² = r_norm² * std²
        sigma2_pred = sigma2_pred_norm * (train_std ** 2)

        # Realized variance
        sigma2_real = returns[t] ** 2

        forecasts.append(sigma2_pred)
        realized.append(sigma2_real)

    return {
        'forecasts': np.array(forecasts),
        'realized': np.array(realized),
        'dates': dates,
        'fit_count': fit_count,
    }
