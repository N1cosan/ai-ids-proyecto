"""
app.py

Dashboard de Streamlit para el AI-IDS.

Simula trafico de red en "tiempo real" leyendo filas del dataset procesado
y enviandolas una por una al endpoint /predict de la API FastAPI. Muestra:
- Un feed en vivo de alertas (con la explicacion SHAP)
- Un contador de trafico normal vs. ataques
- Un grafico de tipos de ataque detectados

IMPORTANTE: la API (uvicorn) debe estar corriendo antes de iniciar esto:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Para correr el dashboard (en otra terminal, con el venv activo):
    streamlit run dashboard/app.py
"""

import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"

st.set_page_config(page_title="AI-IDS Dashboard", layout="wide")

st.title("🛡️ AI-IDS — Panel de Deteccion de Intrusiones")
st.caption("Simulacion de trafico en tiempo real sobre el dataset CIC-IDS2017")


def check_api() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


if not check_api():
    st.error(
        "No se pudo conectar con la API en http://localhost:8000. "
        "Asegurate de tener corriendo: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` "
        "en otra terminal, y recarga esta pagina."
    )
    st.stop()


@st.cache_data
def load_sample_data(n_per_class: int = 20) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["Label"] != "Heartbleed"]

    parts = []
    for label, group in df.groupby("Label"):
        parts.append(group.sample(min(len(group), n_per_class), random_state=7))

    combined = pd.concat(parts, ignore_index=True)
    return combined.sample(frac=1, random_state=7).reset_index(drop=True)


df_sample = load_sample_data()

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "stats" not in st.session_state:
    st.session_state.stats = {"BENIGN": 0}
if "running" not in st.session_state:
    st.session_state.running = False
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0


col_a, col_b, col_c = st.columns([1, 1, 3])
with col_a:
    if st.button("▶️ Iniciar simulacion"):
        st.session_state.running = True
with col_b:
    if st.button("⏸️ Detener"):
        st.session_state.running = False
with col_c:
    speed = st.slider("Velocidad (segundos entre flujos)", 0.1, 2.0, 0.5)


col_stats, col_alerts = st.columns([1, 2])

stats_placeholder = col_stats.empty()
chart_placeholder = col_stats.empty()
alerts_placeholder = col_alerts.empty()


def render_stats():
    with stats_placeholder.container():
        st.subheader("Resumen")
        total = sum(st.session_state.stats.values())
        st.metric("Flujos procesados", total)
        for label, count in sorted(st.session_state.stats.items(), key=lambda x: -x[1]):
            st.write(f"**{label}**: {count}")

    with chart_placeholder.container():
        if len(st.session_state.stats) > 1:
            chart_df = pd.DataFrame(
                list(st.session_state.stats.items()), columns=["Tipo", "Conteo"]
            )
            st.bar_chart(chart_df.set_index("Tipo"))


def render_alerts():
    with alerts_placeholder.container():
        st.subheader("Feed de alertas (mas recientes primero)")
        if not st.session_state.alerts:
            st.info("Aun no hay alertas. Inicia la simulacion para ver resultados.")
        for alert in reversed(st.session_state.alerts[-15:]):
            if alert["is_attack"]:
                st.error(alert["report"])
            else:
                st.success(alert["report"])


render_stats()
render_alerts()


if st.session_state.running:
    row = df_sample.iloc[st.session_state.row_idx % len(df_sample)]
    true_label = row["Label"]
    features = row.drop("Label").to_dict()

    try:
        response = requests.post(API_URL, json={"features": features}, timeout=5)
        response.raise_for_status()
        result = response.json()

        st.session_state.alerts.append(result)
        pred_class = result["predicted_class"]
        st.session_state.stats[pred_class] = st.session_state.stats.get(pred_class, 0) + 1

    except requests.exceptions.RequestException as e:
        st.warning(f"Error llamando a la API: {e}")

    st.session_state.row_idx += 1
    time.sleep(speed)
    st.rerun()