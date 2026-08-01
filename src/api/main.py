"""
main.py

API con FastAPI para servir el modelo de deteccion de intrusiones.

Endpoints:
- GET  /health          -> healthcheck simple
- POST /predict         -> recibe un flujo de trafico (features) y devuelve
                           la clasificacion + explicacion (estilo XAI)

Para correr:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

(ejecutar desde la carpeta raiz del proyecto, ai-ids/)
"""
from fastapi import BackgroundTasks
from src.api.alertas_telegram import enviar_alerta_telegram
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

app = FastAPI(
    title="AI-IDS API",
    description="Deteccion de intrusiones explicable (XAI) sobre trafico de red",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

# --- Carga del modelo, una sola vez al iniciar el servidor ---
# Usamos el modelo robusto (post adversarial training, ronda 2) en vez
# del v1 original, que demostramos vulnerable a evasion (80-100%).
MODEL_VERSION = "robust_round2"

model = joblib.load(MODEL_DIR / "rf_model_robust_v3_round3.joblib")
label_encoder = joblib.load(MODEL_DIR / "label_encoder_robust_v3_round3.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns_robust_v3_round3.joblib")
explainer = shap.TreeExplainer(model)


class TrafficFlow(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_attack: bool
    top_contributing_features: list
    report: str


def build_report(predicted_class: str, confidence: float, top_features: list) -> str:
    if predicted_class == "BENIGN":
        return f"Trafico normal (confianza: {confidence}%). No se tomo ninguna accion."

    features_txt = ", ".join(
        f"{f['feature']} ({f['value']:.2f})" for f in top_features[:3]
    )
    return (
        f"ALERTA: Trafico bloqueado. Coincide en un {confidence}% con un "
        f"patron de '{predicted_class}'. Principales factores: {features_txt}."
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_classes": list(label_encoder.classes_)}


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: TrafficFlow, background_tasks: BackgroundTasks):
    input_features = flow.features

    missing = set(feature_columns) - set(input_features.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan {len(missing)} features requeridas. Ejemplo: {list(missing)[:5]}",
        )

    X_row = pd.DataFrame([input_features])[feature_columns]

    predicted_class_idx = int(model.predict(X_row)[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = round(float(model.predict_proba(X_row)[0][predicted_class_idx]) * 100, 2)

    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        class_shap_values = shap_values[predicted_class_idx][0]
    else:
        class_shap_values = shap_values[0, :, predicted_class_idx]

    contributions = pd.Series(class_shap_values, index=feature_columns)
    top_features = contributions.sort_values(ascending=False).head(5)

    top_contributing_features = [
        {
            "feature": feat,
            "value": float(X_row[feat].values[0]),
            "shap_contribution": round(float(val), 4),
        }
        for feat, val in top_features.items()
    ]

    report = build_report(predicted_class, confidence, top_contributing_features)
    background_tasks.add_task(enviar_alerta_telegram, {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_attack": predicted_class != "BENIGN",
        "top_contributing_features": top_contributing_features,
        "report": report,
    })

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        is_attack=predicted_class != "BENIGN",
        top_contributing_features=top_contributing_features,
        report=report,
    )