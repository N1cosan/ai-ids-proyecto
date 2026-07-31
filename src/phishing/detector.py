"""
detector.py
------------
Punto de entrada del módulo: carga el modelo entrenado y expone
analyze_message(), que combina:

  1. Probabilidad del modelo ML (TF-IDF + Regresión Logística)
  2. Señales de contenido basadas en reglas (features.py)
  3. Señales de URL/dominio (url_analyzer.py)

en un score final 0-100 con etiqueta y motivos explicables.

Diseño pensado para SaaS: esta función es pura (no hace I/O de red,
no manda alertas). La capa de FastAPI/Telegram que viene después solo
llama a analyze_message() y decide qué hacer con el resultado
(loggear, alertar, bloquear, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib

try:
    from src.phishing.features import clean_for_tfidf, extract_content_signals
    from src.phishing.url_analyzer import analyze_urls_in_text
except ImportError:  # ejecución directa dentro de la carpeta phishing/
    from features import clean_for_tfidf, extract_content_signals
    from url_analyzer import analyze_urls_in_text

DEFAULT_MODEL_PATH = Path("models/phishing/phishing_pipeline.joblib")

# Umbrales para la etiqueta final (score 0-100)
UMBRAL_PHISHING = 60           # antes 70
UMBRAL_SOSPECHOSO = 40         # se mantiene

# Cómo se combinan las 3 fuentes de evidencia en el score final.
# El modelo ML pesa más porque generaliza mejor con más datos,
# pero las reglas explícitas garantizan que patrones conocidos
# (ej: "Bancolombia" + "clave dinámica" + acortador) no dependan
# 100% de que el dataset de entrenamiento los haya visto.
PESO_MODELO_ML = 0.40          # antes 0.55
PESO_REGLAS_CONTENIDO = 0.35   # antes 0.25
PESO_REGLAS_URL = 0.25         # antes 0.20

@dataclass
class ResultadoAnalisis:
    texto: str
    canal: str
    remitente: str | None
    score: float                       # 0-100
    etiqueta: str                      # "phishing" | "sospechoso" | "legitimo"
    motivos: list[str] = field(default_factory=list)
    indicadores: dict = field(default_factory=dict)
    prob_modelo_ml: float | None = None
    urls_analizadas: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canal": self.canal,
            "remitente": self.remitente,
            "score": round(self.score, 1),
            "etiqueta": self.etiqueta,
            "motivos": self.motivos,
            "indicadores": self.indicadores,
            "prob_modelo_ml": self.prob_modelo_ml,
            "urls": [u.__dict__ for u in self.urls_analizadas],
        }


class PhishingDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipeline = None

    def _cargar_modelo(self):
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo en {self.model_path}. "
                    "Corre primero: python -m src.phishing.train_phishing --data <csv>"
                )
            self._pipeline = joblib.load(self.model_path)
        return self._pipeline

    def analyze_message(
        self,
        texto: str,
        remitente: str | None = None,
        canal: str = "whatsapp",
        urls: list[str] | None = None,
    ) -> ResultadoAnalisis:
        if canal not in {"whatsapp", "email", "sms"}:
            raise ValueError("canal debe ser 'whatsapp', 'email' o 'sms'")

        # 1. Modelo ML
        pipeline = self._cargar_modelo()
        texto_limpio = clean_for_tfidf(texto)
        prob_ml = float(pipeline.predict_proba([texto_limpio])[0][1])

        # 2. Reglas de contenido
        señales_contenido = extract_content_signals(texto)

        # 3. Reglas de URL
        señales_url = analyze_urls_in_text(texto, urls_extra=urls)
        score_url = max((u.score for u in señales_url), default=0.0)

        # Combinar en score final 0-100
        score_final = (
            PESO_MODELO_ML * prob_ml
            + PESO_REGLAS_CONTENIDO * señales_contenido.score_reglas
            + PESO_REGLAS_URL * score_url
        ) * 100
        score_final = min(max(score_final, 0.0), 100.0)

        if score_final >= UMBRAL_PHISHING:
            etiqueta = "phishing"
        elif score_final >= UMBRAL_SOSPECHOSO:
            etiqueta = "sospechoso"
        else:
            etiqueta = "legitimo"

        motivos = self._construir_motivos(señales_contenido, señales_url, prob_ml)

        return ResultadoAnalisis(
            texto=texto,
            canal=canal,
            remitente=remitente,
            score=score_final,
            etiqueta=etiqueta,
            motivos=motivos,
            indicadores={
                "categorias_contenido": señales_contenido.categorias_activas(),
                "marca_mencionada": señales_contenido.marca_detectada,
                "solicita_credenciales": señales_contenido.tiene_solicitud_credenciales,
                "usa_urgencia": señales_contenido.tiene_urgencia,
            },
            prob_modelo_ml=round(prob_ml, 3),
            urls_analizadas=señales_url,
        )

    @staticmethod
    def _construir_motivos(señales_contenido, señales_url, prob_ml) -> list[str]:
        motivos = []
        if señales_contenido.marca_detectada:
            motivos.append(f"Menciona la marca/entidad '{señales_contenido.marca_detectada}'")
        if señales_contenido.tiene_urgencia:
            motivos.append("Usa lenguaje de urgencia o presión")
        if señales_contenido.tiene_solicitud_credenciales:
            motivos.append("Solicita datos sensibles, clave u OTP")
        if señales_contenido.tiene_gancho_premio:
            motivos.append("Ofrece un premio, bono o beneficio como gancho")
        for u in señales_url:
            motivos.extend(u.indicadores)
        if prob_ml >= 0.7:
            motivos.append("El modelo de lenguaje entrenado reconoce un patrón típico de phishing")
        return motivos


# Instancia por defecto para uso simple tipo función (analyze_message())
_detector_default: PhishingDetector | None = None


def analyze_message(
    texto: str,
    remitente: str | None = None,
    canal: str = "whatsapp",
    urls: list[str] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Función de conveniencia: usa un detector cacheado a nivel de
    módulo para no recargar el modelo en cada llamada (importante
    cuando esto se sirva desde FastAPI)."""
    global _detector_default
    if _detector_default is None or Path(model_path) != _detector_default.model_path:
        _detector_default = PhishingDetector(model_path=model_path)
    resultado = _detector_default.analyze_message(texto, remitente=remitente, canal=canal, urls=urls)
    return resultado.to_dict()
