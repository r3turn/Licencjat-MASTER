import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

# --- KOLORY DLA BAJERU ---
GREEN = "\033[92m"
CYAN = "\033[96m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# --- KONFIGURACJA HIPERPARAMETRÓW ---
WINDOW_SIZE = 20   # Patrzymy 20 dni wstecz (ok. 1 miesiąc giełdowy)
TRAIN_SPLIT = 0.70 # 70% danych na trening
VAL_SPLIT = 0.15   # 15% na walidację (sprawdzanie w trakcie)
TEST_SPLIT = 0.15  # 15% na ostateczny test

# ==========================================
# 1. HARDWARE CHECK (Czy jesteś gotowy?)
# ==========================================
print(f"\n{GREEN}🔍 SPRAWDZANIE AKCELERACJI SPRZĘTOWEJ...{RESET}")

device = torch.device("cpu") # Domyślnie
hw_name = "CPU (Procesor)"

if torch.cuda.is_available():
    device = torch.device("cuda")
    hw_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    print(f"   ✅ Wykryto stację roboczą! Używam: {GREEN}{hw_name}{RESET}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    hw_name = "Apple Metal (MPS)"
    print(f"   ✅ Wykryto Maca M-Series! Używam: {GREEN}{hw_name}{RESET}")
else:
    print(f"   ⚠️  Brak GPU. Będę liczył na: {YELLOW}{hw_name}{RESET}")

print(f"   -> PyTorch version: {torch.__version__}")
print("-" * 50)

# ==========================================
# 2. PRZYGOTOWANIE DANYCH (TENSORY)
# ==========================================
print(f"{GREEN}📂 Wczytuję dane z data/returns.parquet...{RESET}")

if not os.path.exists("data/returns.parquet"):
    print(f"{RED}❌ BŁĄD: Brak pliku data/returns.parquet!{RESET}")
    exit()

df = pd.read_parquet("data/returns.parquet")
os.makedirs("data/tensors", exist_ok=True)

# Funkcja: Sliding Window (Cięcie danych na kawałki)
# X = Sekwencja 20 dni (np. pon-pt)
# y = Zmienność w dniu 21 (proxy: kwadrat zwrotu)
def create_sequences(data, window):
    X, y = [], []
    # Lecimy od początku do końca, zostawiając miejsce na okno
    for i in range(len(data) - window):
        # Wejście: sekwencja o długości 'window'
        seq_x = data[i : i+window]
        
        # Cel (Target): Zmienność w następnym dniu (t+1)
        # Używamy "Squared Returns" jako proxy dla zmienności (Standard w literaturze)
        target = data[i+window] ** 2
        
        X.append(seq_x)
        y.append(target)
        
    return np.array(X), np.array(y)

print(f"{GREEN}⚙️  Przetwarzanie spółek na Tensory PyTorch...{RESET}")

for ticker in df.columns:
    print(f"   -> Processing: {CYAN}{ticker}{RESET}...")
    
    # 1. Pobieramy surowe zwroty
    raw_data = df[ticker].values.reshape(-1, 1)
    
    # 2. Podział Chronologiczny (Bez tasowania! To szereg czasowy)
    total_len = len(raw_data)
    train_idx = int(total_len * TRAIN_SPLIT)
    val_idx = int(total_len * (TRAIN_SPLIT + VAL_SPLIT))
    
    train_raw = raw_data[:train_idx]
    val_raw = raw_data[train_idx:val_idx]
    test_raw = raw_data[val_idx:]
    
    # 3. Skalowanie (Fitujemy TYLKO na treningu, żeby nie oszukiwać!)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)
    
    # 4. Tworzenie sekwencji (To tu powstaje "trójwymiarowa kostka")
    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)
    
    # 5. Konwersja na PyTorch Tensors
    # Zamieniamy numpy array na torch.Tensor (float32 to standard dla GPU)
    tensors = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        
        # Zapisujemy parametry skalera, żeby potem odwrócić wynik (dla wykresów)
        "scaler_mean": scaler.mean_[0],
        "scaler_scale": scaler.scale_[0]
    }
    
    # 6. Zapis do pliku .pt (PyTorch Binary)
    save_path = f"data/tensors/{ticker}.pt"
    torch.save(tensors, save_path)

# Podsumowanie dla uspokojenia nerwów
sample_ticker = df.columns[0]
# NA TO (dodajemy weights_only=False):
sample_tensor = torch.load(f"data/tensors/{sample_ticker}.pt", weights_only=False)
print("\n" + "="*50)
print(f"{GREEN}🚀 SUKCES! Tensory gotowe w folderze data/tensors/{RESET}")
print("="*50)
print(f"Sprzęt wykryty na teraz: {hw_name}")
print(f"Przykładowy kształt danych (X_train) dla {sample_ticker}:")
print(f"👉 {sample_tensor['X_train'].shape}")
print(f"   (Próbki, Długość Okna, Cechy) -> Oczekiwane: (ok. 3600, 20, 1)")
print("="*50)