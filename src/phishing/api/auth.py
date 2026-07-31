"""
auth.py — autenticación simple por API Key
La clave se lee de TRUTH_ENGINE_API_KEY (variable de entorno / .env)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

API_KEY = os.getenv("TRUTH_ENGINE_API_KEY", "")


async def verificar_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API Key del servidor no configurada (revisa .env)",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida o ausente")
    return True
