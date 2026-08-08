"""
diagnose_fp_hosts_fixed.py

Correccion del diagnostico anterior: diagnose_and_tune_fp.py fallo en
identificar los hosts porque 'Src IP'/'Dst IP' no sobrevivieron al
CSV de holdout v6 (se filtraron al combinar con CIC-IDS2017, que no
tiene esas columnas).

Este script reconstruye el MISMO split de evaluacion de BENIGN de
Neris usado en build_ctu13_augmented_dataset_v6.py (mismo
random_state=42, misma fraccion 0.30), pero esta vez conservando
'Src IP' y 'Dst IP' para poder identificar que host normal esta
involucrado en cada falso positivo.

Uso:
    python diagnose_fp_hosts_fixed.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
EVAL_FRACTION = 0.30

MODEL_DIR = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\models")
MODEL_ROUND = "v6_ctu13_augmented"

NERIS_LABELED_CSV = Path(r"D:\ctu13_output\ctu13_labeled.csv")

RELIABLE_NORMAL = {
    "147.32.84.170": "Stribrek",
    "147.32.84.164": "Grill",
    "147.32.84.134": "Jist",
}
LESS_RELIABLE_NORMAL = {
    "147.32.87.36": "CVUT-WebServer",
    "147.32.80.9": "CVUT-DNS-Server",
    "147.32.87.11": "MatLab-Server",
}


def host_group(src_ip, dst_ip):
    for ip in (src_ip, dst_ip):
        if ip in RELIABLE_NORMAL:
            return f"Confiable ({RELIABLE_NORMAL[ip]})"
    for ip in (src_ip, dst_ip):
        if ip in LESS_RELIABLE_NORMAL:
            return f"Menos confiable ({LESS_RELIABLE_NORMAL[ip]})"
    return "Otro/desconocido (no deberia pasar)"


def main():
    print("[*] Cargando modelo v6...")
    model = joblib.load(MODEL_DIR / f"rf_model_{MODEL_ROUND}.joblib")
    label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{MODEL_ROUND}.joblib")
    feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{MODEL_ROUND}.joblib")

    print(f"[*] Cargando CSV original de Neris (con Src IP/Dst IP): {NERIS_LABELED_CSV}")
    df_fam = pd.read_csv(NERIS_LABELED_CSV, low_memory=False)
    benign_rows = df_fam[df_fam["Label_CTU13"] == "BENIGN"].copy()
    print(f"[+] {len(benign_rows):,} filas BENIGN totales de Neris")

    # Reconstruir EXACTAMENTE el mismo split usado en build_ctu13_augmented_dataset_v6.py
    _, benign_eval = train_test_split(
        benign_rows, test_size=EVAL_FRACTION, random_state=RANDOM_STATE
    )
    print(f"[+] {len(benign_eval):,} filas en el holdout de evaluacion (deberia coincidir con 258,826)")

    X = benign_eval[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    benign_eval = benign_eval[valid_mask].reset_index(drop=True)
    X = X.reset_index(drop=True)

    pred_idx = model.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    benign_eval["pred"] = pred_labels

    benign_eval["host_group"] = benign_eval.apply(
        lambda r: host_group(str(r["Src IP"]), str(r["Dst IP"])), axis=1
    )

    print("\n" + "=" * 80)
    print("  Falsos positivos por host normal involucrado")
    print("=" * 80)

    total_por_grupo = benign_eval["host_group"].value_counts()
    fp_mask = benign_eval["pred"] != "BENIGN"
    fp_por_grupo = benign_eval[fp_mask]["host_group"].value_counts()

    print(f"\n{'Grupo de host':<40} {'Total':>10} {'Falsos positivos':>18} {'% FP':>8}")
    print("-" * 80)
    for grupo in total_por_grupo.index:
        total = total_por_grupo[grupo]
        fp = fp_por_grupo.get(grupo, 0)
        print(f"{grupo:<40} {total:>10,} {fp:>18,} {100*fp/total:>7.1f}%")

    print("\n" + "=" * 80)
    n_total_fp = fp_mask.sum()
    print(f"Total falsos positivos en este holdout: {n_total_fp:,} de {len(benign_eval):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
