"""
sanity_check_robust_model.py

Chequeo rapido (no es una prueba rigurosa de generalizacion) para
confirmar que el modelo robusto (post adversarial training) no colapso
en su capacidad de clasificar el trafico original correctamente.

AVISO METODOLOGICO: el modelo fue entrenado con TODO el dataset limpio
(no se reservo un holdout separado para esto), asi que este chequeo NO
prueba generalizacion a datos nunca vistos -- solo detecta fallas
catastroficas obvias (ej. que prediga todo como una sola clase). Para
una verificacion rigurosa, lo ideal es probar con datos de otro dia
(ej. Tuesday.csv, que trae ataques distintos y trafico nunca visto).
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "full_clean_v4.csv"

ROUND = "v4_base"

model = joblib.load(MODEL_DIR / f"rf_model_{ROUND}.joblib")
label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{ROUND}.joblib")
feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{ROUND}.joblib")

df = pd.read_csv(DATA_PATH)
df = df[df["Label"] != "Heartbleed"]

sample = pd.concat(
    [g.sample(min(len(g), 3000), random_state=1) for _, g in df.groupby("Label")],
    ignore_index=True,
)

X = sample[feature_columns]
y_true = label_encoder.transform(sample["Label"])
y_pred = model.predict(X)

print(f"=== Chequeo de sanidad: modelo robusto ({ROUND}) ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))