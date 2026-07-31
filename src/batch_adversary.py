#!/usr/bin/env python3
"""
Batch Adversarial Generator – DoS Slowhttptest
Procesa N muestras (tú eliges cuántas), mide tasa de evasión
y guarda las que lograron engañar al modelo.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from brute_evasion import evade_slowhttptest

# ============================================================
# CONFIGURACIÓN – CAMBIA AQUÍ
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model_v1.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "monday_wednesday_clean.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "slowhttptest_adversarial.csv")

N_SAMPLES = 40          # ← Cuántas muestras procesar (empieza con 30-50)
MAX_ITERS = 80          # ← Iteraciones del hill-climbing
SAVE_ADVERSARIAL = True # Guardar las que lograron evadir


def main():
    print("=" * 60)
    print("  BATCH ADVERSARIAL GENERATOR – DoS Slowhttptest")
    print("=" * 60)

    # 1. Modelo
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Modelo no encontrado:\n  {MODEL_PATH}")
        return
    print("[*] Cargando modelo...")
    model = joblib.load(MODEL_PATH)
    print("[+] Modelo cargado.")

    # 2. Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset no encontrado:\n  {DATASET_PATH}")
        return
    print("[*] Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Dataset: {len(df):,} filas")

    # 3. Filtrar Slowhttptest
    label_col = "Label" if "Label" in df.columns else "label"
    mask = df[label_col].astype(str).str.contains("Slowhttptest|slowhttptest", case=False, na=False)
    attack_df = df[mask].copy()
    print(f"[+] Muestras Slowhttptest disponibles: {len(attack_df)}")

    if len(attack_df) == 0:
        print("[ERROR] No se encontraron muestras de Slowhttptest.")
        return

    # Tomar solo N muestras
    n = min(N_SAMPLES, len(attack_df))
    samples = attack_df.sample(n=n, random_state=42).reset_index(drop=True)
    feature_names = [c for c in df.columns if c != label_col]

    print(f"\n[*] Procesando {n} muestras (max_iters={MAX_ITERS})...")
    print("-" * 60)

    results = []
    adversarial_rows = []
    success_count = 0

    for i, row in samples.iterrows():
        sample_values = row[feature_names].values.astype(float)

        print(f"[{i+1:02d}/{n}] ", end="", flush=True)

        result = evade_slowhttptest(
            model=model,
            sample=sample_values,
            feature_names=feature_names,
            max_iters=MAX_ITERS,
            verbose=False
        )

        if result["success"]:
            success_count += 1
            status = "EVADIDO "
        else:
            status = "RESISTENTE"

        print(f"{status} | Orig: {result['original_conf']:.2f} → BENIGN: {result['benign_prob']:.2f}")

        results.append(result)

        if result["success"] and SAVE_ADVERSARIAL:
            adv_row = dict(zip(feature_names, result["x_adversarial"]))
            adv_row[label_col] = "BENIGN_ADVERSARIAL"
            adversarial_rows.append(adv_row)

    # 4. Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"Muestras procesadas : {n}")
    print(f"Evasiones exitosas  : {success_count}")
    print(f"Tasa de evasión     : {100 * success_count / n:.1f}%")

    if results:
        avg_orig = np.mean([r["original_conf"] for r in results])
        avg_benign = np.mean([r["benign_prob"] for r in results])
        print(f"Confianza orig avg  : {avg_orig:.3f}")
        print(f"P(BENIGN) avg       : {avg_benign:.3f}")

    # 5. Guardar
    if SAVE_ADVERSARIAL and adversarial_rows:
        adv_df = pd.DataFrame(adversarial_rows)
        adv_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[+] Guardado: {OUTPUT_PATH}")
        print(f"    ({len(adversarial_rows)} muestras adversarias)")
    else:
        print("\n[!] No se generaron muestras adversarias.")

    print("=" * 60)
    print("[*] Proceso terminado.")


if __name__ == "__main__":
    main()