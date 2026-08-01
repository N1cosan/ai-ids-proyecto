"""
diagnose_bot.py

Diagnostico rapido: por que la clase Bot se sobreajusta tanto
(precision 1.00 en train, 0.20 en test real).

Hipotesis: el trafico de Bot en este dataset es muy homogeneo (pocos
patrones distintos, muy repetidos), lo que favorece memorizacion en
vez de generalizacion.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "full_clean_v4.csv"

df = pd.read_csv(DATA_PATH)

for label in ["Bot", "Web Attack \x96 Brute Force", "Web Attack \x96 XSS"]:
    subset = df[df["Label"] == label]
    if len(subset) == 0:
        # el caracter especial puede variar segun encoding, probamos contains
        subset = df[df["Label"].str.contains(label.split()[0], na=False)]
    n_total = len(subset)
    n_exact_dup = subset.drop(columns=["Label"]).duplicated().sum()
    print(f"\n{label}: {n_total} muestras totales, {n_exact_dup} duplicados exactos "
          f"({100*n_exact_dup/n_total:.1f}%)")