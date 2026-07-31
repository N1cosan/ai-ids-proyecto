"""
explain.py

Genera explicaciones para las predicciones del modelo usando SHAP.

Objetivo: en vez de solo decir "esto es un ataque tipo X", generar un
reporte del estilo "Bloqueado: coincide en un 98% con un patron de
DoS Hulk, principalmente por el tiempo entre paquetes (Flow IAT Mean)
y el tamano maximo de paquete (Max Packet Length)".

Usamos shap.TreeExplainer porque esta optimizado para modelos basados
en arboles (Random Forest, XGBoost, etc.) y es mucho mas rapido que el
explainer generico (KernelExplainer) para este tipo de modelo.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap


class IDSExplainer:
    def __init__(self, model_path: str | Path, encoder_path: str | Path,
                 feature_columns_path: str | Path):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_columns = joblib.load(feature_columns_path)

        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, X_row: pd.DataFrame, top_n: int = 5) -> dict:
        X_row = X_row[self.feature_columns]

        predicted_class_idx = self.model.predict(X_row)[0]
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        predicted_proba = self.model.predict_proba(X_row)[0][predicted_class_idx]

        shap_values = self.explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class_idx][0]
        else:
            class_shap_values = shap_values[0, :, predicted_class_idx]

        feature_contributions = pd.Series(
            class_shap_values, index=self.feature_columns
        )

        top_features = feature_contributions.sort_values(ascending=False).head(top_n)

        return {
            "predicted_class": predicted_class,
            "confidence": round(float(predicted_proba) * 100, 2),
            "top_contributing_features": [
                {
                    "feature": feat,
                    "value": float(X_row[feat].values[0]),
                    "shap_contribution": round(float(val), 4),
                }
                for feat, val in top_features.items()
            ],
        }

    def generate_report(self, X_row: pd.DataFrame) -> str:
        result = self.explain_instance(X_row)

        if result["predicted_class"] == "BENIGN":
            return (
                f"Trafico normal (confianza: {result['confidence']}%). "
                f"No se tomo ninguna accion."
            )

        features_txt = ", ".join(
            f"{f['feature']} ({f['value']:.2f})"
            for f in result["top_contributing_features"][:3]
        )

        return (
            f"ALERTA: Trafico bloqueado. "
            f"Coincide en un {result['confidence']}% con un patron de "
            f"'{result['predicted_class']}'. "
            f"Principales factores: {features_txt}."
        )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    MODEL_DIR = BASE_DIR / "models"
    DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"

    print("Cargando modelo y explainer...")
    explainer = IDSExplainer(
        model_path=MODEL_DIR / "rf_model_v1.joblib",
        encoder_path=MODEL_DIR / "label_encoder_v1.joblib",
        feature_columns_path=MODEL_DIR / "feature_columns_v1.joblib",
    )

    print("Cargando datos de ejemplo...")
    df = pd.read_csv(DATA_PATH)

    for label in df["Label"].unique():
        if label == "Heartbleed":
            continue

        sample = df[df["Label"] == label].sample(1, random_state=42)
        X_sample = sample.drop(columns=["Label"])

        print(f"\n--- Ejemplo real: {label} ---")
        report = explainer.generate_report(X_sample)
        print(report)