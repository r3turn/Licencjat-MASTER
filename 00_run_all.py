import subprocess
import sys
import os

scripts = [
    "01_fetch_data.py",
    "02_preprocessing.py",
    "03_garch.py",
    "04_egarch.py",
    "05_gjr_garch.py",
    # "06_lstm.py",
    # "07_gru.py",
    # "08_comparison.py",
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"URUCHAMIAM: {script}")
    print('='*50)

    if not os.path.exists(script):
        print(f"\nBŁĄD: {script} nie istnieje!")
        sys.exit(1)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nBŁĄD w {script}! Przerywam.")
        sys.exit(1)

print(f"\n{'='*50}")
print("WSZYSTKO UKOŃCZONE")
print('='*50)
