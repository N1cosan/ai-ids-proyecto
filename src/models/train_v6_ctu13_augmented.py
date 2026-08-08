"""
train_v6_ctu13_augmented.py

Reentrena con el dataset v6: CIC-IDS2017 + Bot diversificado (Neris,
Rbot, Virut) + BENIGN real de CTU-13 (Neris), corrigiendo el problema
de variable confusora que tenia v5 (donde el origen de los datos
predecia perfectamente la etiqueta).

Mismos hiperparametros que las versiones anteriores.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42

DATA_PATH = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\data\processed\full_clean_v6_ctu13_augmented.csv")
MODEL_DIR = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\models")


def main():
    print("Cargando dataset v6...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Shape: {df.shape}")

    if "Familia_CTU13" in df.columns:
        df = df.drop(columns=["Familia_CTU13"])

    X = df.drop(columns=["Label"])
    y_raw = df["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("\nMapeo de clases:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i} -> {cls}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    print("\nEntrenando Random Forest v6...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Reporte de clasificacion (modelo v6, holdout interno) ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("=== Matriz de confusion (modelo v6) ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    )
    print(cm_df)

    joblib.dump(model, MODEL_DIR / "rf_model_v6_ctu13_augmented.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v6_ctu13_augmented.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns_v6_ctu13_augmented.joblib")

    print(f"\nModelo v6 guardado en: {MODEL_DIR / 'rf_model_v6_ctu13_augmented.joblib'}")


if __name__ == "__main__":
    main()
