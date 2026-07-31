"""
extend_and_train_v3.py

Amplia el dataset de entrenamiento incorporando Tuesday.csv (fuerza
bruta FTP/SSH), y reentrena un modelo v3 multi-clase que ahora cubre:
- BENIGN
- DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest (de Wednesday)
- FTP-Patator, SSH-Patator (de Tuesday, NUEVO)

Este es el modelo que de verdad cumple la propuesta de valor original
del producto: detectar fuerza bruta SSH, no solo DoS.

IMPORTANTE: este script entrena desde el dataset LIMPIO original (sin
los ejemplos adversarios previos). El adversarial training habria que
repetirlo despues sobre este modelo ampliado -- es un paso pendiente
aparte, no incluido aqui.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 50


def load_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"  {n_before} -> {len(df)} filas tras limpieza")
    return df


def filter_rare_classes(df: pd.DataFrame, label_col: str = "Label",
                         min_samples: int = MIN_SAMPLES_PER_CLASS) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"Excluyendo clases con menos de {min_samples} muestras: {rare}")
        df = df[~df[label_col].isin(rare)]
    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "models"

    files = {
        "Monday": RAW_DIR / "Monday-WorkingHours.pcap_ISCX.csv",
        "Wednesday": RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
        "Tuesday": RAW_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv",
    }

    dfs = []
    for name, path in files.items():
        print(f"\nCargando {name}...")
        df = load_csv(path)
        df = clean_dataframe(df)
        dfs.append(df)

    print("\nCombinando los tres dias...")
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    print(f"Shape combinado (tras deduplicar entre dias): {df_combined.shape}")

    df_combined = filter_rare_classes(df_combined)
    print(f"\nDistribucion final de clases:\n{df_combined['Label'].value_counts()}\n")

    out_path = PROCESSED_DIR / "full_clean_v3.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Dataset combinado guardado en: {out_path}")

    X = df_combined.drop(columns=["Label"])
    y_raw = df_combined["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("Mapeo de clases:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i} -> {cls}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    print("\nEntrenando Random Forest v3 (con cobertura de fuerza bruta)...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Reporte de clasificacion (modelo v3) ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("=== Matriz de confusion (modelo v3) ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    )
    print(cm_df)

    joblib.dump(model, MODEL_DIR / "rf_model_v3.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v3.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns_v3.joblib")

    print(f"\nModelo v3 guardado en: {MODEL_DIR / 'rf_model_v3.joblib'}")
    print("\nPENDIENTE: repetir el proceso de auditoria adversaria (generate_adversarial.py")
    print("+ iterative_adversarial_training.py) sobre este modelo v3 ampliado.")