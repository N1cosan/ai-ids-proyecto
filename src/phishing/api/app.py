"""
FastAPI — THE TRUTH ENGINE (Capa de Texto / Anti-Phishing)
POST /analyze -> analyze_message() + alerta Telegram si corresponde
Protegido con header X-API-Key
"""
from __future__ import annotations
 
from typing import Any, Dict, List, Optional
 
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
 
from src.phishing.detector import analyze_message, PhishingDetector, DEFAULT_MODEL_PATH
from src.phishing.explain import generar_explicacion, generar_resumen_corto
from src.phishing.alertas_telegram import enviar_alerta_telegram
from src.phishing.db import init_db, guardar_analisis, listar_ultimos
from src.phishing.api.auth import verificar_api_key
from src.phishing.breach_checker import check_breach
from src.phishing.ip_checker import check_ip, obtener_ip_cliente
 
import time
from collections import defaultdict
from fastapi import Request

RATE_LIMIT_MAX = 5          # consultas
RATE_LIMIT_VENTANA = 60     # segundos

_rate_limit_tracker: dict[str, list[float]] = defaultdict(list)


def verificar_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "desconocido"
    ahora = time.time()

    _rate_limit_tracker[ip] = [t for t in _rate_limit_tracker[ip] if ahora - t < RATE_LIMIT_VENTANA]

    if len(_rate_limit_tracker[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demasiadas consultas. Maximo {RATE_LIMIT_MAX} por minuto.",
        )

    _rate_limit_tracker[ip].append(ahora)
app = FastAPI(
    title="THE TRUTH ENGINE — Anti-Phishing",
    description="Detección de phishing e ingeniería social orientada a Colombia (texto).",
    version="0.1.3",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
 
 
@app.on_event("startup")
def startup():
    init_db()
    # Precargar el modelo ML para evitar 503 en la primera petición
    try:
        print("[startup] Cargando modelo de phishing...")
        detector = PhishingDetector(model_path=DEFAULT_MODEL_PATH)
        detector._cargar_modelo()
        print("[startup] Modelo cargado correctamente.")
    except Exception as e:
        print(f"[startup] Error al cargar modelo: {e}")
 
 
# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    texto: str = Field(..., min_length=1, description="Mensaje a analizar")
    canal: str = Field(default="whatsapp", description="whatsapp | sms | email")
    remitente: Optional[str] = Field(default=None, description="Número o email del remitente")
    urls: Optional[List[str]] = Field(default=None, description="URLs ya extraídas (opcional)")
    alertar: bool = Field(
        default=True,
        description="Si True, envía Telegram cuando sea phishing/sospechoso",
    )
 
 
class AnalyzeResponse(BaseModel):
    score: float
    etiqueta: str
    motivos: List[str]
    indicadores: Dict[str, Any]
    canal: str
    remitente: Optional[str] = None
    prob_modelo_ml: Optional[float] = None
    urls: Optional[List[Any]] = None
    explicacion: Optional[str] = None
    resumen: Optional[str] = None
    telegram_enviado: bool = False
    telegram_error: Optional[str] = None
    analizado_por_llm: bool = False
    llm_categoria: Optional[str] = None

class BreachCheckResponse(BaseModel):
    encontrado: bool
    desde_cache: bool
    limite_agotado: bool = False
    total_breaches: int = 0
    riesgo: Optional[str] = None
    breaches: Optional[List[Dict[str, Any]]] = None
    mensaje: str
 
# ---------------------------------------------------------------------------
# Manejadores globales de excepciones
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Error de validación en el cuerpo de la petición",
            "errors": exc.errors(),
        },
    )
 
 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Evita filtrar stack traces al cliente en producción
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc),
        },
    )
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "servicio": "THE TRUTH ENGINE — Anti-Phishing",
        "version": "0.1.3",
        "docs": "/docs",
        "endpoints": ["POST /analyze (requiere X-API-Key)", "GET /health"],
    }
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    _auth: bool = Depends(verificar_api_key),
):
    print("🚩 PASO 1: Petición recibida y autorizada en /analyze")
 
    # Validación de canal
    if req.canal not in {"whatsapp", "email", "sms"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="canal debe ser 'whatsapp', 'email' o 'sms'",
        )
 
    # Análisis principal
    try:
        print("🚩 PASO 2: Entrando a la función analyze_message()...")
        result = analyze_message(
            texto=req.texto,
            remitente=req.remitente,
            canal=req.canal,
            urls=req.urls,
        )
        print("🚩 PASO 3: analyze_message() terminó correctamente!")
    except FileNotFoundError as e:
        print(f"❌ ERROR CRÍTICO - ARCHIVO NO ENCONTRADO: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modelo o recurso no disponible: {e}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al analizar el mensaje: {e}",
        )
 
    # Explicación y resumen
    explicacion = generar_explicacion(result)
    resumen = generar_resumen_corto(result)
    result["explicacion"] = explicacion
    result["resumen"] = resumen
 
    # Telegram (nunca debe tumbar la respuesta principal)
    telegram_enviado = False
    telegram_error: Optional[str] = None
 
    if req.alertar:
        try:
            tg = await enviar_alerta_telegram(result, texto_original=req.texto)
            telegram_enviado = tg is not None
        except Exception as e:
            telegram_error = str(e)
            print(f"[telegram] Error al enviar: {e}")
 
    # Guardar en SQLite
 # Guardar en SQLite
    try:
        guardar_analisis(
            texto=req.texto,
            canal=result.get("canal") or req.canal,
            remitente=result.get("remitente") or req.remitente,
            score=result.get("score"),
            etiqueta=result.get("etiqueta"),
            prob_modelo_ml=result.get("prob_modelo_ml"),
            motivos=result.get("motivos") or [],
            indicadores=result.get("indicadores") or {},
            telegram_enviado=telegram_enviado,
            telegram_error=telegram_error,
            analizado_por_llm=result.get("analizado_por_llm", False),
            llm_categoria=result.get("llm_categoria"),
        )
    except Exception as e:
        print(f"[db] Error al guardar análisis: {e}")
 
    return {
        "score": result.get("score"),
        "etiqueta": result.get("etiqueta"),
        "motivos": result.get("motivos") or [],
        "indicadores": result.get("indicadores") or {},
        "canal": result.get("canal") or req.canal,
        "remitente": result.get("remitente") or req.remitente,
        "prob_modelo_ml": result.get("prob_modelo_ml"),
        "urls": result.get("urls"),
        "explicacion": explicacion,
        "resumen": resumen,
        "telegram_enviado": telegram_enviado,
        "telegram_error": telegram_error,
        "analizado_por_llm": result.get("analizado_por_llm", False),
        "llm_categoria": result.get("llm_categoria"),
    }

@app.get("/check-breach", response_model=BreachCheckResponse)
def check_breach_endpoint(
    email: str,
    request: Request,
    _auth: bool = Depends(verificar_api_key),
):
    verificar_rate_limit(request)

    resultado = check_breach(email)

    if resultado.limite_agotado:
        return BreachCheckResponse(
            encontrado=False,
            desde_cache=False,
            limite_agotado=True,
            mensaje="Se agoto el limite diario de consultas del servicio de brechas. Intenta de nuevo mas tarde.",
        )

    if resultado.error:
        return BreachCheckResponse(
            encontrado=False,
            desde_cache=False,
            mensaje="No se pudo completar la verificacion en este momento. Intenta de nuevo.",
        )

    if not resultado.encontrado:
        return BreachCheckResponse(
            encontrado=False,
            desde_cache=resultado.desde_cache,
            mensaje="Buenas noticias: no encontramos tu correo en ninguna filtracion conocida.",
        )

    return BreachCheckResponse(
        encontrado=True,
        desde_cache=resultado.desde_cache,
        total_breaches=resultado.total_breaches,
        riesgo=resultado.riesgo,
        breaches=resultado.breaches,
        mensaje=f"Tu correo aparecio en {resultado.total_breaches} filtracion(es) conocida(s).",
    ) 
 
@app.get("/historial")
def historial(limit: int = 20, _auth: bool = Depends(verificar_api_key)):
    return {"items": listar_ultimos(limit)}
 
 
@app.get("/estadisticas")
def estadisticas(_auth: bool = Depends(verificar_api_key)):
    """Resumen rápido de los análisis guardados."""
    from src.phishing.db import get_connection
 
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS total FROM analisis")
        total = cur.fetchone()["total"]

        cur.execute("""
            SELECT etiqueta, COUNT(*) as cantidad
            FROM analisis
            GROUP BY etiqueta
        """)
        por_etiqueta = cur.fetchall()

        cur.execute("SELECT COUNT(*) AS cantidad FROM analisis WHERE telegram_enviado = 1")
        telegram_enviados = cur.fetchone()["cantidad"]

        cur.execute("SELECT ROUND(AVG(score), 1) AS promedio FROM analisis")
        score_promedio = cur.fetchone()["promedio"]

        cur.execute("""
            SELECT timestamp, texto, score, etiqueta
            FROM analisis
            ORDER BY id DESC
            LIMIT 1
        """)
        ultimo = cur.fetchone()
 
        return {
            "total_analisis": total,
            "por_etiqueta": {row["etiqueta"]: row["cantidad"] for row in por_etiqueta},
            "telegram_enviados": telegram_enviados,
            "score_promedio": score_promedio,
            "ultimo_analisis": {
                "timestamp": ultimo["timestamp"] if ultimo else None,
                "texto": (ultimo["texto"][:80] + "...") if ultimo and len(ultimo["texto"]) > 80 else (ultimo["texto"] if ultimo else None),
                "score": ultimo["score"] if ultimo else None,
                "etiqueta": ultimo["etiqueta"] if ultimo else None,
            } if ultimo else None,
        }
    finally:
        conn.close()