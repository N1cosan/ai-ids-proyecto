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
 
try:
    from src.phishing.features import normalize_text
except ImportError:  # ejecución directa dentro de la carpeta phishing/
    from features import normalize_text
 
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
 
# Dominios institucionales colombianos (gobierno, educación, fuerzas
# militares) que legítimamente usan estructuras con varios subdominios
# (ej: jprmpalsjrioseco@cendoj.ramajudicial.gov.co). No se les aplica
# la penalización por "muchos subdominios" para no marcar como
# sospechosas comunicaciones institucionales reales.
TLDS_INSTITUCIONALES_CO = {".gov.co", ".edu.co", ".mil.co"}
 
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
 
# Marcas/entidades que además de las colombianas se usan seguido para
# suplantar el NOMBRE VISIBLE del remitente en phishing (casas de
# apuestas, servicios internacionales). Esta lista es más amplia que
# MARCAS_CLAVE a propósito: para el chequeo de "nombre visible del
# remitente vs. dominio real" conviene cubrir más marcas, porque acá
# no estamos comparando contra una whitelist de dominios legítimos
# (que sí requeriría mantenerla actualizada) sino solo detectando que
# el nombre y el dominio del MISMO correo no coinciden entre sí.
MARCAS_SUPLANTABLES = set(MARCAS_CLAVE) | {
    "1xbet", "betplay", "wplay", "rushbet", "codere", "stake", "bet365",
    "netflix", "amazon", "apple", "microsoft", "paypal", "google",
    "instagram", "facebook", "whatsapp", "soat", "efecty", "baloto",
}
 
# Patrón para extraer "Nombre visible <email@dominio>" de encabezados
# tipo "De:"/"From:" — frecuentes cuando alguien pega un correo
# reenviado o citado (el caso más común que vamos a recibir).
PATRON_REMITENTE = re.compile(
    r"(?:de|from)\s*:\s*([^<>\n]{1,60}?)\s*<\s*([a-z0-9_.+-]+@[a-z0-9.-]+\.[a-z]{2,24})\s*>",
    re.IGNORECASE,
)
 
 
@dataclass
class SenalRemitente:
    nombre_visible: str
    email: str
    dominio: str
    marca_detectada: str | None = None
    es_spoofing: bool = False
    indicadores: list[str] = field(default_factory=list)
 
 
def analizar_remitente_en_texto(texto: str) -> list[SenalRemitente]:
    """Busca patrones 'De: Marca <correo@dominio>' en el texto y detecta
    'display name spoofing': cuando el NOMBRE VISIBLE del remitente
    menciona una marca conocida pero el DOMINIO real del correo no
    corresponde a esa marca (ej: 'De: 1xbet <campaigns@ciravor.com>').
 
    Es una técnica de phishing muy común porque la mayoría de clientes
    de correo muestran el nombre visible más grande que la dirección
    real. A diferencia de otras señales de este módulo, esta NO
    depende de saber cuál es el dominio "oficial" de la marca — solo
    compara el nombre y el dominio dentro del MISMO correo, así que es
    una comparación 100% verificable con el propio texto."""
    resultados: list[SenalRemitente] = []
    for match in PATRON_REMITENTE.finditer(texto):
        nombre_visible = match.group(1).strip().strip("\"'")
        email = match.group(2).strip().lower()
        dominio = email.split("@")[-1]
 
        nombre_norm = normalize_text(nombre_visible)
        marca_en_nombre = next(
            (m for m in MARCAS_SUPLANTABLES if re.search(rf"\b{re.escape(m)}\b", nombre_norm)),
            None,
        )
 
        señal = SenalRemitente(nombre_visible=nombre_visible, email=email, dominio=dominio)
        if marca_en_nombre and marca_en_nombre not in dominio:
            señal.marca_detectada = marca_en_nombre
            señal.es_spoofing = True
            señal.indicadores.append(
                f"El remitente se muestra como '{nombre_visible}' pero el correo real "
                f"({dominio}) no corresponde a esa marca — posible suplantación de remitente"
            )
        resultados.append(señal)
    return resultados
 
 
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
    es_institucional_co = any(dominio_sin_puerto.endswith(tld) for tld in TLDS_INSTITUCIONALES_CO)
    partes = dominio_sin_puerto.split(".")
    señal.num_subdominios = max(len(partes) - 2, 0)
    if señal.num_subdominios >= 2 and not es_institucional_co:
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
    if señal.num_subdominios >= 2 and not es_institucional_co:
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