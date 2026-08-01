"""
verify_generalization.py

Verificacion de generalizacion del modelo final (v3_round3) contra
datos verdaderamente nunca vistos: Thursday y Friday del dataset
CIC-IDS2017 (nunca usados en ningun entrenamiento anterior).

Estos dos dias son la prueba mas dura que le hemos hecho al modelo:
traen tipos de ataque que el modelo NUNCA vio en absoluto (ni en train
ni en el adversarial training):
  - Thursday: Web Attacks (Brute Force, XSS, SQL Injection), Infiltration
  - Friday:   Botnet, PortScan, DDoS

Mismos dos chequeos que ya usamos con Tuesday.csv:
1. Trafico BENIGN nuevo -> tasa de falsos positivos real.
2. Ataques nunca vistos -> que porcentaje pasa desapercibido como BENIGN
   vs que porcentaje al menos genera una alerta (aunque la etiqueta
   exacta este mal, porque el modelo no fue entrenado para esas clases).

Uso:
    python src/models/verify_generalization.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
RAW_DIR = BASE_DIR / "data" / "raw"

MODEL_VERSION = "robust_v3_round3"

# Archivos de Thursday y Friday -- nunca usados en ningun entrenamiento
FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


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

    if not frames:
        raise FileNotFoundError(
            "No se encontro ninguno de los archivos esperados en data/raw. "
            "Revisa que los hayas descargado y colocado ahi."
        )

    combined = pd.concat(frames, ignore_index=True)

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()
    combined = combined.drop_duplicates()
    return combined


if __name__ == "__main__":
    print(f"Cargando modelo en produccion: rf_model_{MODEL_VERSION}.joblib ...")
    model = joblib.load(MODEL_DIR / f"rf_model_{MODEL_VERSION}.joblib")
    label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{MODEL_VERSION}.joblib")
    feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{MODEL_VERSION}.joblib")

    print("Cargando y combinando Thursday + Friday (nunca vistos por el modelo)...")
    df = load_and_clean([RAW_DIR / f for f in FILES])
    print(f"Shape combinado: {df.shape}")
    print(f"\nDistribucion de clases:\n{df['Label'].value_counts()}\n")

    # === CHEQUEO 1: trafico BENIGN nunca visto ===
    benign = df[df["Label"] == "BENIGN"]
    X_benign = benign[feature_columns]
    preds = label_encoder.inverse_transform(model.predict(X_benign))

    n_total = len(preds)
    n_correct = (preds == "BENIGN").sum()
    fp_rate = 100 * (1 - n_correct / n_total)

    print("=" * 60)
    print("CHEQUEO 1: Trafico BENIGN de Thursday/Friday (nunca visto)")
    print("=" * 60)
    print(f"Total muestras BENIGN evaluadas: {n_total}")
    print(f"Correctamente clasificadas: {n_correct} ({100*n_correct/n_total:.2f}%)")
    print(f"Falsos positivos: {n_total - n_correct} ({fp_rate:.2f}%)")
    if n_total - n_correct > 0:
        wrong = preds[preds != "BENIGN"]
        print("\nDistribucion de los falsos positivos:")
        print(pd.Series(wrong).value_counts())

    # === CHEQUEO 2: tipos de ataque totalmente nuevos ===
    new_attack_labels = [l for l in df["Label"].unique() if l != "BENIGN"]
    print(f"\n{'='*60}")
    print(f"CHEQUEO 2: Ataques nunca vistos ni entrenados: {new_attack_labels}")
    print(f"{'='*60}")

    for label in new_attack_labels:
        subset = df[df["Label"] == label]
        X_subset = subset[feature_columns]
        preds_subset = label_encoder.inverse_transform(model.predict(X_subset))

        n = len(preds_subset)
        n_benign = (preds_subset == "BENIGN").sum()
        n_flagged = n - n_benign

        print(f"\n{label} ({n} muestras):")
        print(f"  Marcado como BENIGN (pasa desapercibido): {n_benign} ({100*n_benign/n:.1f}%)")
        print(f"  Marcado como algun ataque conocido (genera alerta): {n_flagged} ({100*n_flagged/n:.1f}%)")
        if n_flagged > 0:
            print(f"  Distribucion: {pd.Series(preds_subset[preds_subset != 'BENIGN']).value_counts().to_dict()}")