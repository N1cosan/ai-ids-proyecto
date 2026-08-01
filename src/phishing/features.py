"""
features.py
------------
Señales de CONTENIDO (texto) para detección de phishing / ingeniería
social, orientadas a patrones frecuentes en Colombia.
 
Estas señales se usan en dos lugares:
  1. Como reglas explicables independientes del modelo ML (detector.py
     las combina con la probabilidad del clasificador).
  2. Opcionalmente como columnas extra de features si más adelante se
     quiere pasar de "solo TF-IDF" a un modelo con features mixtas.
 
Todo el texto de entrada se asume ya "crudo" (sin normalizar); las
funciones de limpieza viven aquí para que train_phishing.py y
detector.py usen exactamente la misma normalización.
"""
 
from __future__ import annotations
 
import re
import unicodedata
from dataclasses import dataclass, field
 
 
# ---------------------------------------------------------------------
# Normalización de texto (compartida por entrenamiento e inferencia)
# ---------------------------------------------------------------------
 
def normalize_text(texto: str) -> str:
    """Minúsculas, sin tildes, espacios colapsados. NO elimina URLs
    (eso lo hace url_analyzer) porque la presencia de una URL es en sí
    una señal."""
    if not texto:
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto)
    return texto
 
 
# ---------------------------------------------------------------------
# Léxicos orientados a Colombia
# ---------------------------------------------------------------------
 
URGENCIA = [
    "bloqueara", "bloqueado", "bloqueada", "sera bloqueada", "será bloqueada",
    "cuenta bloqueada", "suspendida", "suspension", "suspensión",
    "vence hoy", "vence en", "ultima oportunidad", "última oportunidad",
    "actua ahora", "actúa ahora", "de inmediato", "urgente", "24 horas", "hoy mismo",
    "antes de que", "para evitar el bloqueo", "pago pendiente", "mora",
    "deuda pendiente", "reportado", "central de riesgo",
    # Seguros / SOAT / DIAN / paquetes
    "poliza activada", "póliza activada", "seguro activado", "seguro no solicitado",
    "quedo activada", "quedó activada", "quedo activado", "quedó activado",
    "cancelar poliza", "cancelar póliza", "cancelar seguro",
    "paquete retenido", "envio retenido", "envío retenido", "aduana",
    "pago pendiente dian", "sancion dian", "sanción dian", "embargo dian",
    "soat vencido", "soat barato", "renovar soat",
]
 
# Suplantación de marcas / entidades típicas en Colombia
MARCAS_CO = [
    "bancolombia", "davivienda", "banco de bogota", "banco de bogotá", "bbva",
    "banco popular", "banco caja social", "colpatria", "scotiabank",
    "nequi", "daviplata", "pse", "aval", "redeban", "efecty", "baloto",
    "dian", "policia nacional", "policía nacional", "fiscalia", "fiscalía",
    "migracion colombia", "migración colombia",
    "servientrega", "coordinadora", "envia", "envía", "interrapidisimo",
    "interrapidísimo", "tcc",
    "claro", "movistar", "tigo", "wom", "une",
    "eps", "sisben", "sisbén", "coljuegos",
    "soat",  # útil como “marca” de estafa de seguros
]
 
SOPORTE_TECNICO = [
    "soporte tecnico", "soporte técnico", "mesa de ayuda", "equipo de seguridad",
    "departamento de fraude", "verificacion de cuenta", "verificación de cuenta",
    "actualizar datos",
    # Operadores y soporte falso de WhatsApp / bancos
    "soporte claro", "soporte movistar", "soporte tigo", "soporte wom",
    "asesor bancolombia", "asesor nequi", "asesor davivienda",
    "area de seguridad", "área de seguridad", "seguridad bancaria",
    "proteccion de cuenta", "protección de cuenta",
]
 
# Solicitud directa de datos sensibles / credenciales
SOLICITUD_DATOS = [
    # Claves y OTP clásicos
    "numero de tarjeta", "código otp", "codigo otp", "clave dinamica", "clave dinámica",
    "clave de acceso", "contrasena", "contraseña", "cvv",
    "fecha de vencimiento de tarjeta", "numero de cuenta", "número de cuenta",
    "confirmar tus datos", "valida tu cuenta", "valide su cuenta",
    "verifica tu cuenta", "actualiza tus datos", "clave de tu banco",
    "token de seguridad", "valide sus datos", "valide sus datos aqui", "valide sus datos aquí",
    "ingrese sus datos", "ingresa tus datos", "actualizar datos",
    "confirme su cedula", "confirme su cédula", "confirme su clave",
    "verifique sus datos", "actualice sus datos", "actualice aqui", "actualice aquí",
    "valide aqui", "valide aquí",
    # Nequi / “acepta el pago” / “te envié por error”
    "acepta el pago", "acepta el cobro", "aceptar el pago", "aceptar el cobro",
    "oprime aceptar", "presiona aceptar", "dale en aceptar",
    "te envie por error", "te envié por error", "envie por equivocacion",
    "envié por equivocación", "transferencia por error", "te cayeron por error",
    "devuelveme el dinero", "devuélveme el dinero", "devuelve el dinero",
    "reembolsame", "reembólsame", "regresame el dinero", "regresa el dinero",
    "confirma la transferencia", "autoriza el pago", "autoriza el cobro",
    "clave dinamica nequi", "clave dinámica nequi", "codigo nequi", "código nequi",
    # Seguros / SOAT no solicitados
    "cancele aqui", "cancele aquí", "cancelar aqui", "cancelar aquí",
    "no la solicito", "no lo solicito", "no la solicite", "no lo solicite",
    "si no la solicito", "si no la solicitó",
]
WHATSAPP_ESTAFAS = [
    # Estafa familiar clásica
    "hola mama", "hola papa", "hola mamá", "hola papá",
    "se me daño el celular", "se me dañó el celular", "se me rompio el celular",
    "se me rompió el celular", "se me dano el celular",
    "este es mi nuevo numero", "este es mi nuevo número",
    "cambie de numero", "cambié de número", "perdi el celular", "perdí el celular",
    "necesito que me hagas un favor", "necesito un favor urgente",
    "transferencia urgente", "prestame", "préstame", "consigname", "consígame",
    "envia el dinero a esta cuenta", "envía el dinero a esta cuenta",
    "me robaron el celular", "me asaltaron", "estoy en un apuro",
    "necesito plata urgente", "me puedes prestar", "me puedes consignar",
    # Robo / suplantación de cuenta de WhatsApp
    "codigo de verificacion", "código de verificación",
    "codigo de seguridad de whatsapp", "código de seguridad de whatsapp",
    "soporte de whatsapp", "soporte whatsapp", "whatsapp support",
    "intentan clonar tu cuenta", "clonar tu whatsapp", "verificar tu cuenta de whatsapp",
    "te enviamos un codigo", "te enviamos un código",
    "ingresa el codigo", "ingresa el código", "envia el codigo", "envía el código",
]
 
PREMIOS_GANCHOS = [
    "ganaste", "has sido seleccionado", "has sido seleccionada",
    "reclama tu premio", "sorteo", "regalo", "beneficiario",
    "subsidio", "bono", "devolucion de saldo", "devolución de saldo",
    "promocion exclusiva", "promoción exclusiva", "oferta limitada",
]
 
_ALL_LEXICONS = {
    "urgencia": URGENCIA,
    "marca_suplantada": MARCAS_CO,
    "soporte_tecnico_falso": SOPORTE_TECNICO,
    "solicitud_datos_sensibles": SOLICITUD_DATOS,
    "estafa_whatsapp": WHATSAPP_ESTAFAS,
    "premio_gancho": PREMIOS_GANCHOS,
}
 
 
def _find_matches(texto_norm: str, lexicon: list[str]) -> list[str]:
    return [term for term in lexicon if term in texto_norm]
 
 
@dataclass
class ContentSignals:
    """Resultado de analizar el contenido textual de un mensaje."""
 
    texto_normalizado: str
    matches: dict = field(default_factory=dict)          # categoria -> [terminos]
    marca_detectada: str | None = None
    tiene_solicitud_credenciales: bool = False
    tiene_urgencia: bool = False
    tiene_gancho_premio: bool = False
    score_reglas: float = 0.0                              # 0..1 aporte de reglas
 
    def categorias_activas(self) -> list[str]:
        return [cat for cat, terms in self.matches.items() if terms]
 
 
def extract_content_signals(texto: str) -> ContentSignals:
    """Punto de entrada principal de este archivo. Recibe texto crudo,
    devuelve señales estructuradas listas para explain.py y detector.py."""
    texto_norm = normalize_text(texto)
 
    matches = {cat: _find_matches(texto_norm, lex) for cat, lex in _ALL_LEXICONS.items()}
 
    marcas = matches["marca_suplantada"]
    marca_detectada = marcas[0] if marcas else None
 
    tiene_solicitud = bool(matches["solicitud_datos_sensibles"])
    tiene_urgencia = bool(matches["urgencia"])
    tiene_premio = bool(matches["premio_gancho"])
 
    # Score de reglas simple y transparente (no reemplaza al modelo ML,
    # se combina con él en detector.py). Pesos elegidos para que la
    # combinación "marca + urgencia + solicitud de datos" (el patrón
    # más común de phishing bancario en CO) ya sea alto por sí sola.
# Score de reglas simple y transparente (no reemplaza al modelo ML,
    # se combina con él en detector.py). Pesos elegidos para que la
    # combinación "marca + urgencia + solicitud de datos" (el patrón
    # más común de phishing bancario en CO) ya sea alto por sí sola.
    pesos = {
        "marca_suplantada": 0.20,
        "urgencia": 0.20,
        "solicitud_datos_sensibles": 0.30,
        "soporte_tecnico_falso": 0.15,
        "estafa_whatsapp": 0.50,   # sube de 0.35 a 0.50
        "premio_gancho": 0.15,
    }
    score = sum(pesos[cat] for cat, terms in matches.items() if terms)
    score = min(score, 1.0)
 
    return ContentSignals(
        texto_normalizado=texto_norm,
        matches=matches,
        marca_detectada=marca_detectada,
        tiene_solicitud_credenciales=tiene_solicitud,
        tiene_urgencia=tiene_urgencia,
        tiene_gancho_premio=tiene_premio,
        score_reglas=score,
    )
 
 
def clean_for_tfidf(texto: str) -> str:
    """Limpieza usada específicamente para vectorizar con TF-IDF.
    Reemplaza URLs por un token especial (la URL en sí la analiza
    url_analyzer.py) para no perder la señal "contiene enlace" pero sin
    que el vocabulario del TF-IDF explote con dominios únicos."""
    texto_norm = normalize_text(texto)
    texto_norm = re.sub(r"https?://\S+|www\.\S+", " __url__ ", texto_norm)
    texto_norm = re.sub(r"[^a-z0-9ñ_\s]", " ", texto_norm)
    texto_norm = re.sub(r"\s+", " ", texto_norm).strip()
    return texto_norm