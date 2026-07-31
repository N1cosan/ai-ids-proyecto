"""
url_analyzer.py
-----------------
Señales de URL/dominio: acortadores, homógrafos, imitación de marcas
colombianas, TLDs poco comunes para banca, IPs literales, exceso de
subdominios, etc.

No requiere librerías externas (solo stdlib) para mantener el MVP
simple de instalar. Si más adelante se quiere resolución DNS/WHOIS en
vivo, eso se añade en un archivo aparte (por ejemplo url_reputation.py)
para no acoplar la lógica offline con llamadas de red.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.parse import urlparse

# Captura URLs con esquema/www, y además cualquier "dominio.tld" plausible
# (2-24 letras de TLD) seguido opcionalmente de una ruta, para no depender
# de una lista fija de extensiones (necesario para detectar acortadores
# como bit.ly, t.co, etc. cuyo TLD no es .com/.co "tradicional").
URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9][a-z0-9\-]*)*\.[a-z]{2,24}(?:/\S*)?)",
    re.IGNORECASE,
)

ACORTADORES = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly",
    "rebrand.ly", "shorturl.at", "acortaurl.com", "bit.do", "rb.gy",
}

TLDS_SOSPECHOSOS = {".xyz", ".top", ".click", ".shop", ".site", ".online", ".live", ".icu", ".fit"}

# Dominios legítimos conocidos (whitelist mínima) — usados como referencia
# para detectar imitaciones, NO como lista exhaustiva de "todo lo demás
# es phishing".
DOMINIOS_LEGITIMOS_CO = [
    "bancolombia.com", "davivienda.com", "bancodebogota.com", "bbva.com.co",
    "nequi.com.co", "daviplata.com", "dian.gov.co", "claro.com.co",
    "movistar.co", "tigo.com.co", "servientrega.com", "coordinadora.com",
    "wompi.co", "pse.com.co",
]

MARCAS_CLAVE = [
    "bancolombia", "davivienda", "nequi", "daviplata", "dian", "claro",
    "movistar", "tigo", "servientrega", "pse", "bbva",
]


@dataclass
class UrlSignal:
    url: str
    dominio: str
    es_ip_literal: bool = False
    usa_acortador: bool = False
    tld_sospechoso: bool = False
    num_subdominios: int = 0
    marca_en_subdominio_o_ruta: str | None = None
    posible_homografo_de: str | None = None
    similitud_maxima: float = 0.0
    indicadores: list[str] = field(default_factory=list)
    score: float = 0.0


def extract_urls(texto: str) -> list[str]:
    return [m.group(0).rstrip(").,;") for m in URL_PATTERN.finditer(texto)]


def _dominio_de(url: str) -> str:
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url.lower()


def _similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def analyze_url(url: str) -> UrlSignal:
    dominio = _dominio_de(url)
    dominio_sin_puerto = dominio.split(":")[0]

    señal = UrlSignal(url=url, dominio=dominio_sin_puerto)

    # IP literal en vez de dominio
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", dominio_sin_puerto):
        señal.es_ip_literal = True
        señal.indicadores.append("El enlace usa una dirección IP en vez de un dominio")

    # Acortadores
    if dominio_sin_puerto in ACORTADORES:
        señal.usa_acortador = True
        señal.indicadores.append("El enlace usa un acortador que oculta el destino real")

    # TLD sospechoso
    for tld in TLDS_SOSPECHOSOS:
        if dominio_sin_puerto.endswith(tld):
            señal.tld_sospechoso = True
            señal.indicadores.append(f"El dominio usa una extensión poco común para banca/gobierno ({tld})")
            break

    # Subdominios excesivos (ej: bancolombia.seguridad.verificar-ahora.xyz)
    partes = dominio_sin_puerto.split(".")
    señal.num_subdominios = max(len(partes) - 2, 0)
    if señal.num_subdominios >= 2:
        señal.indicadores.append("El dominio tiene muchos subdominios, común para disfrazar el sitio real")

    # Marca colombiana mencionada en el dominio pero NO es el dominio oficial
    for marca in MARCAS_CLAVE:
        if marca in dominio_sin_puerto:
            es_oficial = any(dominio_sin_puerto.endswith(d) for d in DOMINIOS_LEGITIMOS_CO)
            if not es_oficial:
                señal.marca_en_subdominio_o_ruta = marca
                señal.indicadores.append(
                    f"El dominio menciona '{marca}' pero no corresponde al dominio oficial de esa entidad"
                )
            break

    # Homógrafos / typosquatting: comparar contra dominios legítimos conocidos
    mejor_similitud = 0.0
    mejor_match = None
    for legit in DOMINIOS_LEGITIMOS_CO:
        legit_base = legit.split(".")[0]
        dominio_base = dominio_sin_puerto.split(".")[0].replace("-", "")
        sim = _similitud(dominio_base, legit_base)
        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_match = legit
    señal.similitud_maxima = round(mejor_similitud, 3)
    # Alta similitud pero no es exactamente el dominio oficial => posible homógrafo
    if mejor_match and 0.75 <= mejor_similitud < 1.0 and dominio_sin_puerto != mejor_match:
        señal.posible_homografo_de = mejor_match
        señal.indicadores.append(
            f"El dominio se parece mucho a '{mejor_match}' pero no es igual (posible imitación)"
        )

    # Score agregado de esta URL
    pesos = {
        "es_ip_literal": 0.40,
        "usa_acortador": 0.45,
        "tld_sospechoso": 0.20,
        "marca_en_subdominio_o_ruta": 0.35,
        "posible_homografo_de": 0.40,
    }
    score = 0.0
    if señal.es_ip_literal:
        score += pesos["es_ip_literal"]
    if señal.usa_acortador:
        score += pesos["usa_acortador"]
    if señal.tld_sospechoso:
        score += pesos["tld_sospechoso"]
    if señal.marca_en_subdominio_o_ruta:
        score += pesos["marca_en_subdominio_o_ruta"]
    if señal.posible_homografo_de:
        score += pesos["posible_homografo_de"]
    if señal.num_subdominios >= 2:
        score += 0.10
    señal.score = min(score, 1.0)
    return señal


def analyze_urls_in_text(texto: str, urls_extra: list[str] | None = None) -> list[UrlSignal]:
    """Extrae URLs del texto (más las pasadas explícitamente en
    `urls_extra`, útil cuando el canal ya las separa, como en payloads
    de email) y las analiza todas."""
    urls = extract_urls(texto)
    if urls_extra:
        urls.extend(u for u in urls_extra if u not in urls)
    return [analyze_url(u) for u in urls]
