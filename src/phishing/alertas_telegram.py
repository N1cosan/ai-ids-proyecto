"""
alertas_telegram.py
--------------------
Envía alertas a Telegram cuando el análisis es phishing
o sospechoso con score >= 50.
Los secretos se leen de variables de entorno / .env
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
import httpx

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ETIQUETAS_ALERTA = {"phishing"}
SCORE_MINIMO_SOSPECHOSO = 50


async def enviar_alerta_telegram(
    resultado: dict,
    texto_original: str = "",
) -> Optional[dict]:
    etiqueta = resultado.get("etiqueta", "")
    score = float(resultado.get("score", 0))

    debe_alertar = (
        etiqueta in ETIQUETAS_ALERTA
        or (etiqueta == "sospechoso" and score >= SCORE_MINIMO_SOSPECHOSO)
    )
    if not debe_alertar:
        return None

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] Token o chat_id no configurados (revisa .env)")
        return None

    canal = resultado.get("canal", "?")
    motivos = resultado.get("motivos") or []
    explicacion = resultado.get("explicacion") or ""
    resumen = resultado.get("resumen") or f"[{etiqueta}] score={score}"

    preview = (texto_original[:180] + "...") if len(texto_original) > 180 else texto_original
    motivos_txt = "\n".join(f"- {m}" for m in motivos) or "- (sin motivos detallados)"

    mensaje = (
        f"<b>{resumen}</b>\n\n"
        f"<b>Canal:</b> {canal}\n"
        f"<b>Score:</b> {score}/100\n\n"
        f"<b>Motivos:</b>\n{motivos_txt}\n\n"
        f"<b>Mensaje analizado:</b>\n<code>{preview}</code>\n\n"
        f"{explicacion}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje[:4000],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
