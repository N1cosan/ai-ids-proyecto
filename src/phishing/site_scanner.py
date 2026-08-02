"""
site_scanner.py
------------------
Escaneo basico de seguridad de un sitio web: valida el certificado SSL
y revisa la presencia de headers de seguridad HTTP recomendados
(HSTS, CSP, X-Frame-Options, X-Content-Type-Options).

No hace nada intrusivo ni de fuerza bruta -- solo una conexion HTTPS
normal (la misma que haria un navegador) y lectura de metadatos
publicos del certificado y los headers de respuesta. Pensado para que
un dueno de sitio pueda auto-revisar su propia configuracion, no como
herramienta de pentesting.

Diseno de privacidad / seguridad:
  - No se guarda el dominio consultado en ningun lado (mismo criterio
    que breach_checker.py / ip_checker.py).
  - Timeout corto para no dejar conexiones colgadas si el sitio no
    responde.
  - Nunca sigue mas de un puñado de redirecciones (evita loops).
"""
from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

TIMEOUT_SEGUNDOS = 8.0
DIAS_ALERTA_EXPIRACION = 15  # si el cert expira en menos de esto, se marca como advertencia

# Pesos para el score 0-100 (mismo patron que features.py / url_analyzer.py:
# transparente y facil de ajustar despues con datos reales).
PESO_SSL_VALIDO = 40
PESO_HSTS = 20
PESO_CSP = 20
PESO_XFRAME = 20


@dataclass
class ChecklistItem:
    clave: str
    nombre: str
    ok: bool
    detalle: str


@dataclass
class ResultadoScan:
    dominio: str
    score: int = 0
    riesgo: str = "desconocido"  # "bajo" | "medio" | "alto" | "desconocido"
    checklist: list[ChecklistItem] = field(default_factory=list)
    error: Optional[str] = None


def _normalizar_dominio(url_o_dominio: str) -> tuple[str, str]:
    """Devuelve (dominio, url_https) a partir de lo que haya escrito el
    usuario, sea 'ejemplo.com', 'http://ejemplo.com' o
    'https://ejemplo.com/pagina'."""
    valor = url_o_dominio.strip()
    if not valor.startswith(("http://", "https://")):
        valor = "https://" + valor
    parsed = urlparse(valor)
    dominio = parsed.netloc or parsed.path
    dominio = dominio.split(":")[0]  # quita puerto si viene
    return dominio, f"https://{dominio}"


def _verificar_ssl(dominio: str) -> ChecklistItem:
    try:
        contexto = ssl.create_default_context()
        with socket.create_connection((dominio, 443), timeout=TIMEOUT_SEGUNDOS) as sock:
            with contexto.wrap_socket(sock, server_hostname=dominio) as ssock:
                cert = ssock.getpeercert()

        no_despues = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        dias_restantes = (no_despues - datetime.now(timezone.utc)).days

        if dias_restantes < 0:
            return ChecklistItem(
                clave="ssl_valido",
                nombre="Certificado SSL valido",
                ok=False,
                detalle=f"El certificado ya vencio (hace {abs(dias_restantes)} dias).",
            )
        if dias_restantes < DIAS_ALERTA_EXPIRACION:
            return ChecklistItem(
                clave="ssl_valido",
                nombre="Certificado SSL valido",
                ok=True,
                detalle=f"Valido, pero vence pronto (en {dias_restantes} dias). Renuevalo con tiempo.",
            )
        return ChecklistItem(
            clave="ssl_valido",
            nombre="Certificado SSL valido",
            ok=True,
            detalle=f"Valido, vence en {dias_restantes} dias.",
        )
    except ssl.SSLCertVerificationError as e:
        return ChecklistItem(
            clave="ssl_valido",
            nombre="Certificado SSL valido",
            ok=False,
            detalle=f"El certificado no es valido para este dominio: {e.verify_message if hasattr(e, 'verify_message') else e}",
        )
    except Exception as e:
        return ChecklistItem(
            clave="ssl_valido",
            nombre="Certificado SSL valido",
            ok=False,
            detalle=f"No se pudo verificar el certificado ({e}).",
        )


def _verificar_headers(url: str) -> tuple[list[ChecklistItem], Optional[str]]:
    try:
        with httpx.Client(timeout=TIMEOUT_SEGUNDOS, follow_redirects=True, max_redirects=5) as client:
            resp = client.get(url, headers={"User-Agent": "TheTruthEngine-Scanner/1.0"})
    except Exception as e:
        return [], f"No se pudo conectar al sitio: {e}"

    headers = {k.lower(): v for k, v in resp.headers.items()}
    items = []

    hsts = headers.get("strict-transport-security")
    items.append(ChecklistItem(
        clave="hsts",
        nombre="HSTS (Strict-Transport-Security)",
        ok=bool(hsts),
        detalle="Presente: fuerza siempre HTTPS, evita ataques de downgrade." if hsts
                else "Ausente: el sitio no le exige al navegador usar siempre HTTPS.",
    ))

    csp = headers.get("content-security-policy")
    items.append(ChecklistItem(
        clave="csp",
        nombre="CSP (Content-Security-Policy)",
        ok=bool(csp),
        detalle="Presente: limita de donde se puede cargar contenido (mitiga XSS)." if csp
                else "Ausente: sin esta cabecera hay menos proteccion contra inyeccion de scripts.",
    ))

    xframe = headers.get("x-frame-options")
    items.append(ChecklistItem(
        clave="x_frame_options",
        nombre="X-Frame-Options",
        ok=bool(xframe),
        detalle="Presente: protege contra clickjacking (que el sitio se cargue oculto en un iframe ajeno)." if xframe
                else "Ausente: el sitio podria ser embebido en un iframe malicioso para engañar usuarios.",
    ))

    xcontent = headers.get("x-content-type-options")
    items.append(ChecklistItem(
        clave="x_content_type_options",
        nombre="X-Content-Type-Options",
        ok=bool(xcontent),
        detalle="Presente: evita que el navegador adivine tipos de archivo de forma insegura." if xcontent
                else "Ausente (informativo, no afecta el score).",
    ))

    return items, None


def scan_site(url_o_dominio: str) -> ResultadoScan:
    """Punto de entrada principal. Nunca lanza excepcion -- cualquier
    fallo se refleja en el campo error del resultado."""
    if not url_o_dominio or not url_o_dominio.strip():
        return ResultadoScan(dominio="", error="Escribe un dominio o URL para escanear.")

    dominio, url = _normalizar_dominio(url_o_dominio)

    item_ssl = _verificar_ssl(dominio)
    items_headers, error_headers = _verificar_headers(url)

    if error_headers and not item_ssl.ok:
        # Si ni SSL ni la conexion HTTP funcionaron, el sitio no es alcanzable
        return ResultadoScan(dominio=dominio, error=f"No se pudo escanear el sitio: {error_headers}")

    checklist = [item_ssl] + items_headers

    score = 0
    if item_ssl.ok:
        score += PESO_SSL_VALIDO
    por_clave = {i.clave: i for i in items_headers}
    if por_clave.get("hsts") and por_clave["hsts"].ok:
        score += PESO_HSTS
    if por_clave.get("csp") and por_clave["csp"].ok:
        score += PESO_CSP
    if por_clave.get("x_frame_options") and por_clave["x_frame_options"].ok:
        score += PESO_XFRAME

    if score >= 80:
        riesgo = "bajo"
    elif score >= 40:
        riesgo = "medio"
    else:
        riesgo = "alto"

    return ResultadoScan(
        dominio=dominio,
        score=score,
        riesgo=riesgo,
        checklist=checklist,
        error=error_headers,  # informativo: headers fallaron pero SSL si respondio
    )
