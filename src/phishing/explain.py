"""
explain.py
-----------
Convierte el resultado de detector.analyze_message() en un texto de
alerta legible en español de Colombia, para mostrarlo a un usuario
final o a un admin en un dashboard/Telegram.

Se mantiene separado de detector.py a propósito: el "qué se detectó"
(detector) no debería mezclarse con el "cómo se lo explico a alguien"
(explain), para poder cambiar el tono/formato sin tocar la lógica de
detección.
"""

from __future__ import annotations

EMOJI_POR_ETIQUETA = {
    "phishing": "🚨",
    "sospechoso": "⚠️",
    "legitimo": "✅",
}

TITULO_POR_ETIQUETA = {
    "phishing": "Mensaje probablemente PHISHING",
    "sospechoso": "Mensaje SOSPECHOSO — revisar con cuidado",
    "legitimo": "No se detectaron señales claras de phishing",
}

RECOMENDACION_POR_ETIQUETA = {
    "phishing": (
        "No hagas clic en enlaces ni entregues claves, códigos OTP o datos "
        "de tu tarjeta. Si dice ser tu banco, entra directamente desde la "
        "app oficial o llama a la línea que aparece en el reverso de tu "
        "tarjeta, nunca al número que trae el mensaje."
    ),
    "sospechoso": (
        "Antes de responder o hacer clic, verifica por un canal oficial "
        "(app, línea telefónica conocida, página oficial escrita a mano en "
        "el navegador, no copiada del mensaje)."
    ),
    "legitimo": (
        "No se identificaron patrones típicos de phishing, pero mantén la "
        "regla general: ningún banco o entidad seria te pedirá tu clave "
        "completa u OTP por WhatsApp, SMS o correo."
    ),
}


def _formatear_motivos(motivos: list[str]) -> str:
    if not motivos:
        return "  (No se identificaron indicadores específicos)"
    return "\n".join(f"  • {m}" for m in motivos)


def generar_explicacion(resultado: dict, incluir_score: bool = True) -> str:
    """Recibe el dict devuelto por analyze_message() y arma un texto
    de alerta en español (CO)."""
    etiqueta = resultado["etiqueta"]
    score = resultado["score"]
    canal = resultado.get("canal", "mensaje")
    motivos = resultado.get("motivos", [])

    emoji = EMOJI_POR_ETIQUETA.get(etiqueta, "ℹ️")
    titulo = TITULO_POR_ETIQUETA.get(etiqueta, "Resultado del análisis")
    recomendacion = RECOMENDACION_POR_ETIQUETA.get(etiqueta, "")

    partes = [f"{emoji} {titulo}"]
    if incluir_score:
        partes.append(f"Nivel de riesgo: {score}/100 · Canal: {canal}")
    partes.append("")
    partes.append("Motivos detectados:")
    partes.append(_formatear_motivos(motivos))
    partes.append("")
    partes.append(f"Recomendación: {recomendacion}")

    return "\n".join(partes)


def generar_resumen_corto(resultado: dict) -> str:
    """Versión de una sola línea, útil para logs o notificaciones
    compactas (por ejemplo, listas en un dashboard)."""
    etiqueta = resultado["etiqueta"]
    score = resultado["score"]
    emoji = EMOJI_POR_ETIQUETA.get(etiqueta, "ℹ️")
    motivos = resultado.get("motivos") or []
    principal_motivo = motivos[0] if motivos else "sin indicadores específicos"
    return f"{emoji} [{etiqueta.upper()} · {score}/100] {principal_motivo}"
