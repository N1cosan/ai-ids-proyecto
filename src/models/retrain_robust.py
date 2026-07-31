"""
retrain_robust.py

Reentrena el modelo combinando el dataset original limpio con los
ejemplos adversarios generados por generate_adversarial.py.

Los ejemplos adversarios se agregan con su etiqueta REAL (ej. 'DoS Hulk'),
no como BENIGN -- la idea es que el modelo aprenda que esos patrones
perturbados siguen siendo ataques, y deje de ser tan facil de evadir
con cambios pequenos en unas pocas features.

Guarda el modelo nuevo con sufijo _v2 para no sobreescribir el v1
(asi podemos comparar ambos lado a lado si hace falta).
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 50


def filter_rare_classes(df: pd.DataFrame, label_col: str = "Label",
                         min_samples: int = MIN_SAMPLES_PER_CLASS) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_samples].index.tolist()
    if rare_classes:
        print(f"Excluyendo clases con menos de {min_samples} muestras: {rare_classes}")
        df = df[~df[label_col].isin(rare_classes)]
    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"
    ADV_PATH = BASE_DIR / "data" / "processed" / "adversarial_samples.csv"
    MODEL_DIR = BASE_DIR / "models"

    print("Cargando dataset original...")
    df_original = pd.read_csv(DATA_PATH)
    df_original = filter_rare_classes(df_original)
    print(f"Original: {df_original.shape}")

    print("Cargando ejemplos adversarios...")
    df_adv = pd.read_csv(ADV_PATH)
    print(f"Adversarios: {df_adv.shape}")

    df_combined = pd.concat([df_original, df_adv], ignore_index=True)
    print(f"\nDataset combinado: {df_combined.shape}")
    print(df_combined["Label"].value_counts())

    X = df_combined.drop(columns=["Label"])
    y_raw = df_combined["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    print("\nEntrenando Random Forest v2 (con adversarial training)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Reporte de clasificacion (modelo v2) ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("=== Matriz de confusion (modelo v2) ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    )
    print(cm_df)

    joblib.dump(model, MODEL_DIR / "rf_model_v2.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v2.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns_v2.joblib")

    print(f"\nModelo v2 guardado en: {MODEL_DIR / 'rf_model_v2.joblib'}")
    print("\nSiguiente paso: correr generate_adversarial.py apuntando al")
    print("modelo v2 para medir si la tasa de evasion realmente bajo.")