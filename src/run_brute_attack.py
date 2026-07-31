#!/usr/bin/env python3
"""
Lanzador del ataque de fuerza bruta contra DoS Slowhttptest
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# Añadir el directorio src al path para poder importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brute_evasion import evade_slowhttptest

# ============================================================
# RUTAS – AJUSTA SI ES NECESARIO
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # carpeta raíz ai-ids

MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model_v1.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "monday_wednesday_clean.csv")

# Si tus archivos están en otra ubicación, cámbialo aquí:
# MODEL_PATH = "../models/rf_model_v1.joblib"
# DATASET_PATH = "../data/processed_cicids2017.csv"


def main():
    print("=" * 60)
    print("  BRUTE-FORCE EVASION – DoS Slowhttptest")
    print("=" * 60)

    # 1. Cargar modelo
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encuentra el modelo en:\n  {MODEL_PATH}")
        print("\nBusca el archivo .joblib y actualiza MODEL_PATH.")
        return

    print(f"[*] Cargando modelo: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[+] Modelo cargado.")

    # 2. Cargar dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] No se encuentra el dataset en:\n  {DATASET_PATH}")
        print("\nBusca el CSV procesado y actualiza DATASET_PATH.")
        return

    print(f"[*] Cargando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Dataset cargado → {len(df):,} filas")

    # 3. Buscar muestras de DoS Slowhttptest
    label_col = None
    for cand in ["Label", "label", "Attack", "attack", "Class"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        print("[ERROR] No encontré la columna de etiqueta (Label).")
        print("Columnas disponibles:", list(df.columns)[:15], "...")
        return

    # Buscar variantes del nombre del ataque
    mask = df[label_col].astype(str).str.contains("Slowhttptest|slowhttptest|Slow HTTP", case=False, na=False)
    attack_samples = df[mask]

    if attack_samples.empty:
        print(f"[ERROR] No hay muestras de DoS Slowhttptest (columna '{label_col}').")
        print("Valores únicos de Label:", df[label_col].unique()[:20])
        return

    print(f"[+] Encontradas {len(attack_samples)} muestras de Slowhttptest")

    # Tomamos la primera (puedes cambiar a .sample(1) si quieres aleatoria)
    sample_row = attack_samples.iloc[0]

    # Features = todas las columnas excepto la etiqueta
    feature_names = [c for c in df.columns if c != label_col]
    sample_values = sample_row[feature_names].values.astype(float)

    print(f"\n[*] Lanzando evasión sobre 1 muestra de '{sample_row[label_col]}'...")
    print("-" * 60)

    # 4. Ejecutar evasión
    result = evade_slowhttptest(
        model=model,
        sample=sample_values,
        feature_names=feature_names,
        max_iters=120,
        verbose=True
    )

    # 5. Resultados finales
    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES")
    print("=" * 60)

    if result["success"]:
        print("STATUS: [!!!] ATAQUE EXITOSO – El modelo clasifica como BENIGN")
    else:
        print("STATUS: La evasión NO logró engañar completamente al modelo")

    print(f"Predicción original   : {result['original_pred']}  (conf {result['original_conf']:.4f})")
    print(f"Predicción adversaria : {result['adv_pred']}  (conf {result['adv_conf']:.4f})")
    print(f"P(BENIGN) final       : {result['benign_prob']:.4f}")
    print("=" * 60)
    print("[*] Proceso terminado.")


if __name__ == "__main__":
    main()