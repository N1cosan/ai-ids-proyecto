"""
merge_web_attacks.py

Fusiona 'Web Attack Brute Force' y 'Web Attack XSS' en una sola clase
'Web Attack', porque el modelo las confunde demasiado entre si (son
estructuralmente muy parecidas: ambas trafico HTTP de bajo volumen).

Se pierde el detalle de subtipo especifico, pero se espera ganar mucha
mas confiabilidad en el nivel que realmente importa para el IDS: "esto
es un ataque web" vs "esto es normal".
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "full_clean_v4.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "full_clean_v4_merged.csv"

df = pd.read_csv(DATA_PATH)

print("Distribucion antes de fusionar:")
print(df["Label"].value_counts())

mask_web = df["Label"].str.contains("Web Attack", na=False)
n_before = mask_web.sum()
df.loc[mask_web, "Label"] = "Web Attack"

print(f"\nFusionadas {n_before} filas en la clase unica 'Web Attack'")
print("\nDistribucion final:")
print(df["Label"].value_counts())

df.to_csv(OUT_PATH, index=False)
print(f"\nGuardado en: {OUT_PATH}")