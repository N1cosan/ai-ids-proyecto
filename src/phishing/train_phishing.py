"""
train_phishing.py
-------------------
Entrena el clasificador base de la Capa de Texto (Anti-Phishing):
TF-IDF + Regresión Logística sobre mensajes etiquetados como
phishing/legítimo.

Uso:
    python -m src.phishing.train_phishing --data data/phishing/mensajes_co.csv

El CSV de entrada debe tener (mínimo) estas columnas:
    texto   -> contenido del mensaje (str)
    label   -> 1 = phishing, 0 = legítimo
    canal   -> "whatsapp" | "email" | "sms"   (opcional, no se usa aún
                para entrenar, pero se guarda para análisis futuro)

Salida (en --outdir, por defecto models/phishing/):
    phishing_pipeline.joblib   -> Pipeline sklearn completo (TF-IDF + LR)
    metrics.json                -> métricas de evaluación
    training_report.txt         -> reporte legible (classification_report)

Por qué TF-IDF + Regresión Logística para el MVP:
    - Rápido de entrenar y de servir en CPU (no requiere GPU).
    - Coeficientes interpretables por término -> permite explicar
      "por qué" el modelo marcó el mensaje (ver explain.py).
    - Buena línea base bien documentada en detección de phishing/spam
      antes de pasar a modelos más pesados (embeddings, transformers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Permite ejecutar tanto como módulo (python -m src.phishing.train_phishing)
# como script suelto (python train_phishing.py) sin romper el import.
try:
    from src.phishing.features import clean_for_tfidf
except ImportError:  # ejecución directa dentro de la carpeta phishing/
    from features import clean_for_tfidf

RANDOM_STATE = 42


def cargar_dataset(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    columnas_requeridas = {"texto", "label"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Al CSV le faltan columnas requeridas: {faltantes}")

    df = df.dropna(subset=["texto", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["texto"].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


def construir_pipeline(max_features: int = 8000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),      # unigrama + bigrama: captura frases como "cuenta bloqueada"
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # datasets de phishing suelen estar desbalanceados
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def entrenar(df: pd.DataFrame, test_size: float = 0.2):
    X = df["texto"].apply(clean_for_tfidf)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    reporte = classification_report(y_test, y_pred, target_names=["legitimo", "phishing"])
    matriz = confusion_matrix(y_test, y_pred).tolist()
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = None  # ocurre si el test set queda con una sola clase (datasets muy chicos)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc": auc,
        "confusion_matrix": {"labels": ["legitimo", "phishing"], "matrix": matriz},
    }
    return pipeline, reporte, metrics


def guardar_artefactos(pipeline: Pipeline, reporte: str, metrics: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, outdir / "phishing_pipeline.joblib")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(outdir / "training_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Reporte de entrenamiento: Capa de Texto Anti-Phishing ===\n\n")
        f.write(reporte)
        f.write(f"\nROC-AUC: {metrics['roc_auc']}\n")


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de phishing (TF-IDF + LR)")
    parser.add_argument("--data", required=True, help="Ruta al CSV con columnas texto,label[,canal]")
    parser.add_argument("--outdir", default="models/phishing", help="Carpeta de salida para el modelo")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Cargando dataset desde {args.data} ...")
    df = cargar_dataset(args.data)
    print(f"Mensajes cargados: {len(df)}  (phishing={int(df['label'].sum())}, legitimos={int((df['label']==0).sum())})")

    if len(df) < 20:
        print(
            "AVISO: el dataset tiene muy pocos ejemplos para un modelo confiable. "
            "Esto es válido para probar que el pipeline corre, NO para producción. "
            "Ver sección de datasets en el README del módulo.",
            file=sys.stderr,
        )

    pipeline, reporte, metrics = entrenar(df, test_size=args.test_size)

    print("\n" + reporte)
    print(f"ROC-AUC: {metrics['roc_auc']}")

    outdir = Path(args.outdir)
    guardar_artefactos(pipeline, reporte, metrics, outdir)
    print(f"\nModelo guardado en: {outdir / 'phishing_pipeline.joblib'}")


if __name__ == "__main__":
    main()
