"""
pwned_checker.py
------------------
Verifica si una contraseña aparece en filtraciones conocidas usando
Pwned Passwords (Have I Been Pwned) con k-Anonymity: solo se envían
los primeros 5 caracteres del hash SHA-1, nunca la contraseña ni el
hash completo.

Privacidad (regla dura, no negociable):
  - La contraseña NUNCA se loggea ni se guarda en ningun lado (ni en
    Neon, ni en variables persistentes, ni en el request log de FastAPI).
  - Solo el prefijo de 5 caracteres del hash SHA-1 sale de este proceso.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import httpx

PWNED_API_URL = "https://api.pwnedpasswords.com/range/{prefix}"
TIMEOUT_SEGUNDOS = 6.0


@dataclass
class ResultadoPassword:
    veces_filtrada: int = 0
    error: Optional[str] = None


def check_password(password: str) -> ResultadoPassword:
    """Nunca lanza excepcion ni loggea la contraseña recibida."""
    if not password:
        return ResultadoPassword(error="Contraseña vacia")

    sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]

    try:
        with httpx.Client(timeout=TIMEOUT_SEGUNDOS) as client:
            resp = client.get(
                PWNED_API_URL.format(prefix=prefix),
                headers={"Add-Padding": "true"},  # mitiga ataques de timing/tamaño
            )
    except Exception as e:
        return ResultadoPassword(error=f"Error de red: {e}")

    if resp.status_code != 200:
        return ResultadoPassword(error=f"HTTP {resp.status_code}")

    for line in resp.text.splitlines():
        try:
            suf, count = line.strip().split(":")
        except ValueError:
            continue
        if suf == suffix:
            return ResultadoPassword(veces_filtrada=int(count))

    return ResultadoPassword(veces_filtrada=0)
