"""
ip_checker.py
--------------
Consulta la ubicacion, proveedor de internet, y señales basicas de
reputacion (proxy/VPN/datacenter) de una direccion IP, usando la API
publica y gratuita de ip-api.com (no requiere API key, limite de 45
consultas/minuto por IP de origen del servidor -- Render).

Diseno (mismo criterio de privacidad que breach_checker.py):
  - No se guarda la IP consultada en ningun lado (ni Neon, ni logs).
  - Cache en memoria de 1 hora por IP consultada, para no golpear el
    limite de la API externa si alguien refresca varias veces seguidas.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx

IP_API_URL = "http://ip-api.com/json/{ip}"
IP_API_FIELDS = "status,message,country,regionName,city,isp,org,as,proxy,hosting,query"
TIMEOUT_SEGUNDOS = 5.0
CACHE_TTL_SEGUNDOS = 60 * 60  # 1 hora

# Cache en memoria: { ip: (timestamp_guardado, resultado_dict) }
_cache: dict[str, tuple[float, dict]] = {}


def _leer_cache(ip: str) -> Optional[dict]:
    entry = _cache.get(ip)
    if not entry:
        return None
    guardado_en, resultado = entry
    if time.time() - guardado_en > CACHE_TTL_SEGUNDOS:
        del _cache[ip]
        return None
    return resultado


def _guardar_cache(ip: str, resultado: dict) -> None:
    _cache[ip] = (time.time(), resultado)


@dataclass
class ResultadoIP:
    ip: str
    pais: Optional[str] = None
    ciudad: Optional[str] = None
    isp: Optional[str] = None
    organizacion: Optional[str] = None
    es_proxy_o_vpn: bool = False
    es_datacenter: bool = False
    riesgo: str = "desconocido"   # "bajo" | "medio" | "alto" | "desconocido"
    desde_cache: bool = False
    error: Optional[str] = None


def obtener_ip_cliente(request) -> str:
    """Extrae la IP real del visitante detras de proxies/CDN (Render
    corre detras de Cloudflare, asi que request.client.host suele ser
    la IP interna del proxy, no la del usuario -- hay que mirar
    X-Forwarded-For primero)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # X-Forwarded-For puede traer una cadena "ip_cliente, proxy1, proxy2"
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def _calcular_riesgo(es_proxy_o_vpn: bool, es_datacenter: bool) -> str:
    if es_proxy_o_vpn:
        return "alto"
    if es_datacenter:
        return "medio"
    return "bajo"


def check_ip(ip: str) -> ResultadoIP:
    """Punto de entrada principal. Nunca lanza excepcion -- cualquier
    fallo se devuelve como ResultadoIP con error seteado."""
    cacheado = _leer_cache(ip)
    if cacheado is not None:
        resultado = ResultadoIP(**cacheado)
        resultado.desde_cache = True
        return resultado

    try:
        url = IP_API_URL.format(ip=ip)
        with httpx.Client(timeout=TIMEOUT_SEGUNDOS) as client:
            resp = client.get(url, params={"fields": IP_API_FIELDS})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return ResultadoIP(ip=ip, error=f"Error de red: {e}")

    if data.get("status") != "success":
        return ResultadoIP(ip=ip, error=data.get("message", "IP no valida o no localizable"))

    es_proxy_o_vpn = bool(data.get("proxy"))
    es_datacenter = bool(data.get("hosting"))

    resultado = ResultadoIP(
        ip=data.get("query", ip),
        pais=data.get("country"),
        ciudad=data.get("regionName") and data.get("city") and f"{data['city']}, {data['regionName']}" or data.get("city"),
        isp=data.get("isp"),
        organizacion=data.get("org"),
        es_proxy_o_vpn=es_proxy_o_vpn,
        es_datacenter=es_datacenter,
        riesgo=_calcular_riesgo(es_proxy_o_vpn, es_datacenter),
    )

    _guardar_cache(ip, {
        "ip": resultado.ip,
        "pais": resultado.pais,
        "ciudad": resultado.ciudad,
        "isp": resultado.isp,
        "organizacion": resultado.organizacion,
        "es_proxy_o_vpn": resultado.es_proxy_o_vpn,
        "es_datacenter": resultado.es_datacenter,
        "riesgo": resultado.riesgo,
        "desde_cache": False,
    })

    return resultado
