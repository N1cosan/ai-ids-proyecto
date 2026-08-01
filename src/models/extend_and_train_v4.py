"""
extend_and_train_v4.py

Amplia el dataset de entrenamiento (v3 -> v4) incorporando Thursday y
Friday del CIC-IDS2017, para cubrir tipos de ataque que el modelo v3
no conocia en absoluto: Web Attacks (Brute Force, XSS, SQL Injection),
Infiltration, Bot, PortScan y DDoS.

Decision de diseno: Infiltration tiene solo 36 muestras en total en
el dataset combinado (menos que MIN_SAMPLES_PER_CLASS=50) -- igual
que paso con Heartbleed en el dataset original, es matematicamente
imposible entrenar/validar esa clase de forma confiable. Se excluye
del entrenamiento y se documenta como limitacion conocida, no se
fuerza su inclusion.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MIN_SAMPLES_PER_CLASS = 50

NEW_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

EXISTING_V3_PATH = PROCESSED_DIR / "full_clean_v3.csv"


def load_and_clean(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            print(f"AVISO: no se encontro {p.name}, se omite.")
            continue
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()
    combined = combined.drop_duplicates()
    return combined


if __name__ == "__main__":
    print("Cargando dataset v3 existente (Monday+Tuesday+Wednesday)...")
    df_v3 = pd.read_csv(EXISTING_V3_PATH)
    print(f"v3: {df_v3.shape}")

    print("\nCargando y limpiando Thursday + Friday...")
    df_new = load_and_clean([RAW_DIR / f for f in NEW_FILES])
    print(f"Thursday+Friday: {df_new.shape}")

    print("\nCombinando en dataset v4...")
    df_v4 = pd.concat([df_v3, df_new], ignore_index=True)
    df_v4 = df_v4.drop_duplicates()
    print(f"v4 combinado (antes de filtrar clases raras): {df_v4.shape}")

    print(f"\nDistribucion completa de clases:\n{df_v4['Label'].value_counts()}\n")

    counts = df_v4["Label"].value_counts()
    rare_classes = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if rare_classes:
        print(f"Excluyendo clases con menos de {MIN_SAMPLES_PER_CLASS} muestras: {rare_classes}")
        df_v4 = df_v4[~df_v4["Label"].isin(rare_classes)]

    print(f"\nShape final v4: {df_v4.shape}")
    print(f"Distribucion final:\n{df_v4['Label'].value_counts()}\n")

    out_path = PROCESSED_DIR / "full_clean_v4.csv"
    df_v4.to_csv(out_path, index=False)
    print(f"Dataset v4 guardado en: {out_path}")