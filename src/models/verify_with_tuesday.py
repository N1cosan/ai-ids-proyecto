"""
verify_with_tuesday.py

Verificacion rigurosa de generalizacion usando Tuesday.csv -- un dia
del dataset que el modelo NUNCA vio en ninguna forma (ni en train.py,
ni en el adversarial training). Esta es la prueba honesta que faltaba.

Dos chequeos distintos:

1. TRAFICO BENIGNO DE TUESDAY (verdaderamente nunca visto):
   Mide el falso-positivo real del modelo: de todo el trafico normal
   de este dia nuevo, que porcentaje clasifica correctamente como
   BENIGN. Esta es la prueba de generalizacion que nos faltaba.

2. ATAQUES NUEVOS (FTP-Patator, SSH-Patator):
   El modelo NUNCA fue entrenado para reconocer estos tipos de ataque
   (no estaban en Monday/Wednesday). No es un fallo del modelo si no
   los clasifica como su categoria real -- eso esta fuera de su
   alcance de diseno. Lo que reportamos aqui es simplemente que
   porcentaje de ese trafico nuevo termina marcado como BENIGN (lo
   cual seria un ataque real pasando desapercibido) vs marcado como
   alguna de las clases de DoS conocidas (al menos genera una alerta,
   aunque la etiqueta exacta este equivocada).
"""

from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
TUESDAY_PATH = BASE_DIR / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

MODEL_VERSION = "robust_round2"


def load_and_clean_tuesday() -> pd.DataFrame:
    df = pd.read_csv(TUESDAY_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    import numpy as np
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


if __name__ == "__main__":
    print("Cargando modelo robusto en produccion...")
    model = joblib.load(MODEL_DIR / f"rf_model_{MODEL_VERSION}.joblib")
    label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{MODEL_VERSION}.joblib")
    feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{MODEL_VERSION}.joblib")

    print("Cargando y limpiando Tuesday.csv (nunca visto por el modelo)...")
    df = load_and_clean_tuesday()
    print(f"Shape: {df.shape}")
    print(f"\nDistribucion de clases en Tuesday:\n{df['Label'].value_counts()}\n")

    benign_tuesday = df[df["Label"] == "BENIGN"]
    X_benign = benign_tuesday[feature_columns]
    preds = model.predict(X_benign)
    pred_labels = label_encoder.inverse_transform(preds)

    n_correct = (pred_labels == "BENIGN").sum()
    n_total = len(pred_labels)
    false_positive_rate = 100 * (1 - n_correct / n_total)

    print("=" * 60)
    print("CHEQUEO 1: Trafico BENIGN de Tuesday (nunca visto)")
    print("=" * 60)
    print(f"Total muestras BENIGN evaluadas: {n_total}")
    print(f"Correctamente clasificadas como BENIGN: {n_correct} ({100*n_correct/n_total:.2f}%)")
    print(f"Falsos positivos (BENIGN marcado como ataque): {n_total - n_correct} ({false_positive_rate:.2f}%)")

    if n_total - n_correct > 0:
        wrong = pred_labels[pred_labels != "BENIGN"]
        print("\nDistribucion de los falsos positivos (a que los clasifico mal):")
        print(pd.Series(wrong).value_counts())

    new_attack_labels = [l for l in df["Label"].unique() if l != "BENIGN"]
    print(f"\n{'='*60}")
    print(f"CHEQUEO 2: Ataques nuevos, fuera del alcance de entrenamiento: {new_attack_labels}")
    print(f"{'='*60}")

    for label in new_attack_labels:
        subset = df[df["Label"] == label]
        X_subset = subset[feature_columns]
        preds = model.predict(X_subset)
        pred_labels_subset = label_encoder.inverse_transform(preds)

        n = len(pred_labels_subset)
        n_marked_benign = (pred_labels_subset == "BENIGN").sum()
        n_marked_attack = n - n_marked_benign

        print(f"\n{label} ({n} muestras, tipo de ataque NUNCA visto en entrenamiento):")
        print(f"  Marcado como BENIGN (pasa desapercibido): {n_marked_benign} ({100*n_marked_benign/n:.1f}%)")
        print(f"  Marcado como algun tipo de ataque conocido (genera alerta, "
              f"aunque la etiqueta este mal): {n_marked_attack} ({100*n_marked_attack/n:.1f}%)")
        if n_marked_attack > 0:
            print("  Distribucion de que lo clasifico:")
            print(f"  {pd.Series(pred_labels_subset[pred_labels_subset != 'BENIGN']).value_counts().to_dict()}")