"""
breach_checker.py
-------------------
Consulta la API publica y gratuita de XposedOrNot (breach-analytics)
para saber si un email aparecio en filtraciones de datos conocidas.

Diseno de privacidad (importante, no cambiar sin pensarlo dos veces):
  - NUNCA se guarda el email en Neon ni en logs.
  - El cache es SOLO en memoria del proceso (se pierde en cada
    restart/deploy -- eso es intencional, no un bug).
  - La clave del cache es un hash SHA-256 del email normalizado, no el
    email en texto plano -- asi ni siquiera en memoria queda legible
    si alguien inspecciona el proceso.

Manejo de limites (100 consultas/dia en el tier gratuito de XposedOrNot):
  - Cache de 24h: si ya se consulto ese email hoy, se devuelve el
    resultado cacheado sin volver a llamar a la API externa.
  - Si XposedOrNot responde 429 (limite agotado), se devuelve un
    resultado especial que el endpoint traduce en un mensaje amigable,
    no un error crudo.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

XPOSEDORNOT_URL = "https://api.xposedornot.com/v1/breach-analytics"
TIMEOUT_SEGUNDOS = 8.0
CACHE_TTL_SEGUNDOS = 24 * 60 * 60  # 24 horas

# Cache en memoria: { hash_email: (timestamp_guardado, resultado_dict) }
_cache: dict[str, tuple[float, dict]] = {}


def _hash_email(email: str) -> str:
    email_norm = email.strip().lower()
    return hashlib.sha256(email_norm.encode("utf-8")).hexdigest()


def _leer_cache(email_hash: str) -> Optional[dict]:
    entry = _cache.get(email_hash)
    if not entry:
        return None
    guardado_en, resultado = entry
    if time.time() - guardado_en > CACHE_TTL_SEGUNDOS:
        del _cache[email_hash]
        return None
    return resultado


def _guardar_cache(email_hash: str, resultado: dict) -> None:
    _cache[email_hash] = (time.time(), resultado)


@dataclass
class ResultadoBreach:
    encontrado: bool
    desde_cache: bool
    limite_agotado: bool = False
    breaches: list[dict] | None = None
    total_breaches: int = 0
    riesgo: str | None = None
    error: str | None = None


def _parsear_respuesta_xposedornot(data: dict) -> ResultadoBreach:
    exposed = data.get("ExposedBreaches")
    if not exposed or not exposed.get("breaches_details"):
        return ResultadoBreach(encontrado=False, desde_cache=False, total_breaches=0)

    detalles = exposed["breaches_details"]
    metrics = data.get("BreachMetrics") or {}
    riesgo_info = (metrics.get("risk") or [{}])[0]

    breaches_resumidos = [
        {
            "nombre": b.get("breach"),
            "fecha": b.get("xposed_date"),
            "datos_expuestos": b.get("xposed_data", "").split(";") if b.get("xposed_data") else [],
            "riesgo_password": b.get("password_risk"),
            "registros_expuestos": b.get("xposed_records"),
        }
        for b in detalles
    ]

    return ResultadoBreach(
        encontrado=True,
        desde_cache=False,
        breaches=breaches_resumidos,
        total_breaches=len(breaches_resumidos),
        riesgo=riesgo_info.get("risk_label"),
    )


def check_breach(email: str) -> ResultadoBreach:
    """Punto de entrada principal. Nunca lanza excepcion -- cualquier
    fallo (red, 429, parseo) se devuelve como ResultadoBreach con
    error seteado, para que el endpoint decida como responder."""
    email_hash = _hash_email(email)

    cacheado = _leer_cache(email_hash)
    if cacheado is not None:
        resultado = ResultadoBreach(**cacheado)
        resultado.desde_cache = True
        return resultado

    try:
        with httpx.Client(timeout=TIMEOUT_SEGUNDOS) as client:
            resp = client.get(XPOSEDORNOT_URL, params={"email": email})
    except Exception as e:
        return ResultadoBreach(encontrado=False, desde_cache=False, error=f"Error de red: {e}")

    if resp.status_code == 429:
        return ResultadoBreach(encontrado=False, desde_cache=False, limite_agotado=True)

    if resp.status_code != 200:
        return ResultadoBreach(encontrado=False, desde_cache=False, error=f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception as e:
        return ResultadoBreach(encontrado=False, desde_cache=False, error=f"Respuesta invalida: {e}")

    resultado = _parsear_respuesta_xposedornot(data)

    # Solo cacheamos resultados exitosos (no errores ni 429, para
    # reintentar esos casos en la siguiente consulta)
    _guardar_cache(email_hash, {
        "encontrado": resultado.encontrado,
        "desde_cache": False,
        "breaches": resultado.breaches,
        "total_breaches": resultado.total_breaches,
        "riesgo": resultado.riesgo,
    })

    return resultado