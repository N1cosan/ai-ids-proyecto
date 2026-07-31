"""
load.py

Carga y limpieza de los CSV del dataset CIC-IDS2017.

Este dataset tiene tres problemas conocidos que resolvemos aqui:
1. Los nombres de columna traen espacios al inicio/final (ej. ' Label').
2. Existen valores Infinity y NaN en columnas como 'Flow Bytes/s' y
   'Flow Packets/s' (ocurren cuando la duracion del flujo es 0).
3. Los tipos de dato a veces se infieren mal por los valores anteriores.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Carga un CSV crudo de CIC-IDS2017 y normaliza nombres de columnas."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, low_memory=False)

    # Quita espacios sobrantes en los nombres de columna
    df.columns = df.columns.str.strip()

    return df


def load_multiple_csv(filepaths: list[str | Path]) -> pd.DataFrame:
    """Carga y concatena varios CSV (ej. Monday + Wednesday)."""
    dfs = [load_csv(fp) for fp in filepaths]
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataframe:
    - Reemplaza Infinity / -Infinity por NaN
    - Elimina filas con NaN (son una fraccion muy pequena del total)
    - Elimina duplicados exactos
    - Normaliza la columna Label (quita espacios)
    """
    df = df.copy()

    # Label puede tener espacios tambien en sus valores
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()

    # Reemplazar inf/-inf por NaN para poder eliminarlos de forma uniforme
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna()
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after_dup = len(df)

    print(f"Filas originales:           {n_before}")
    print(f"Filas tras quitar NaN/Inf:  {n_after_na} (-{n_before - n_after_na})")
    print(f"Filas tras quitar duplic.:  {n_after_dup} (-{n_after_na - n_after_dup})")

    return df


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Devuelve el conteo de clases en la columna Label."""
    return df["Label"].value_counts()


if __name__ == "__main__":
    # Ejemplo de uso: carga Monday + Wednesday, limpia, y muestra resumen
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

    files = [
        RAW_DIR / "Monday-WorkingHours.pcap_ISCX.csv",
        RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
    ]

    print("Cargando archivos...")
    df = load_multiple_csv(files)
    print(f"Shape original: {df.shape}\n")

    print("Limpiando...")
    df_clean = clean_dataframe(df)
    print(f"\nShape final: {df_clean.shape}\n")

    print("Distribucion de clases (Label):")
    print(get_label_distribution(df_clean))

    # Guardar version procesada
    OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "monday_wednesday_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nGuardado en: {out_path}")
