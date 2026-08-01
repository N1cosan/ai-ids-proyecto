"""
alertas_telegram.py — Notificaciones de Telegram para el AI-IDS de red.

Adaptado del modulo equivalente de THE TRUTH ENGINE (phishing), pero
usando los campos que ya devuelve /predict en este proyecto:
predicted_class, confidence, is_attack, top_contributing_features, report.

Reutiliza las mismas credenciales de Telegram (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID) que el modulo de phishing, definidas en .env -- las
alertas de ambos sistemas llegan al mismo chat salvo que se configure
un bot distinto para este proyecto.
"""

from __future__ import annotations

import html
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Solo alertar si la confianza del modelo supera este umbral, para no
# saturar el chat con predicciones de baja confianza.
CONFIANZA_MINIMA_ALERTA = 70.0


async def enviar_alerta_telegram(resultado: dict) -> Optional[dict]:
    """
    Recibe el dict devuelto por el endpoint /predict del IDS.

    Devuelve:
      - dict de respuesta de Telegram  -> exito
      - None                           -> no se debe alertar (benigno o baja confianza)
      - {"error": "..."}               -> se intento alertar pero fallo
    """
    is_attack = bool(resultado.get("is_attack", False))
    confidence = float(resultado.get("confidence", 0))

    if not is_attack or confidence < CONFIANZA_MINIMA_ALERTA:
        return None

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] Token o chat_id no configurados (revisa .env)")
        return {"error": "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados"}

    predicted_class = resultado.get("predicted_class", "?")
    report = resultado.get("report", "")
    top_features = resultado.get("top_contributing_features") or []

    features_txt = "\n".join(
        f"- {html.escape(str(f.get('feature')))}: {f.get('value')}"
        for f in top_features[:5]
    ) or "- (sin features destacadas)"

    mensaje = (
        f"<b>🚨 AI-IDS: Ataque detectado</b>\n\n"
        f"<b>Tipo:</b> {html.escape(str(predicted_class))}\n"
        f"<b>Confianza:</b> {confidence}%\n\n"
        f"<b>Reporte:</b>\n{html.escape(str(report))}\n\n"
        f"<b>Features principales:</b>\n{features_txt}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje[:4000],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[telegram] Error al enviar la alerta: {e}")
        return {"error": str(e)}