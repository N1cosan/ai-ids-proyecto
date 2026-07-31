"""
train.py

Entrena un modelo Random Forest multi-clase para detectar y clasificar
tipos de ataque en trafico de red (dataset CIC-IDS2017).

Decisiones de diseno:
- Multi-clase (no binario): el objetivo del producto es explicar QUE tipo
  de ataque se detecto, no solo si hay o no anomalia.
- 'Heartbleed' se excluye del entrenamiento: con solo 11 muestras totales
  es imposible hacer un split train/test estratificado confiable. Se
  documenta como limitacion conocida, a resolver cuando se sumen mas dias
  de datos (ej. Tuesday, que trae mas variedad de ataques).
- class_weight='balanced' para compensar el desbalance entre BENIGN
  (mayoria) y las clases de ataque (minoria).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42

# Clases excluidas por falta de muestras suficientes para entrenar/validar
MIN_SAMPLES_PER_CLASS = 50


def load_processed_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def filter_rare_classes(df: pd.DataFrame, label_col: str = "Label",
                         min_samples: int = MIN_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """Elimina clases con muy pocas muestras (no se pueden validar bien)."""
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_samples].index.tolist()

    if rare_classes:
        print(f"Excluyendo clases con menos de {min_samples} muestras: {rare_classes}")
        df = df[~df[label_col].isin(rare_classes)]

    return df


def split_features_labels(df: pd.DataFrame, label_col: str = "Label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder: LabelEncoder):
    y_pred = model.predict(X_test)

    print("\n=== Reporte de clasificacion ===")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("=== Matriz de confusion ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    )
    print(cm_df)


def get_feature_importances(model, feature_names, top_n=15) -> pd.Series:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos procesados...")
    df = load_processed_data(DATA_PATH)
    print(f"Shape: {df.shape}")

    df = filter_rare_classes(df)
    print(f"Shape tras filtrar clases raras: {df.shape}")
    print(f"\nDistribucion final de clases:\n{df['Label'].value_counts()}\n")

    X, y_raw = split_features_labels(df)

    # Codificar labels a numeros (Random Forest trabaja mejor asi,
    # y ademas queda guardado el mapeo para poder interpretar despues)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("Mapeo de clases:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i} -> {cls}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    print("\nEntrenando Random Forest...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, label_encoder)

    print("\n=== Top 15 features mas importantes ===")
    print(get_feature_importances(model, X.columns))

    # Guardar modelo, encoder, y lista de columnas usadas
    joblib.dump(model, MODEL_DIR / "rf_model_v1.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v1.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns_v1.joblib")

    print(f"\nModelo guardado en: {MODEL_DIR / 'rf_model_v1.joblib'}")