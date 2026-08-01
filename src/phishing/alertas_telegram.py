async def enviar_alerta_telegram(
    resultado: dict,
    texto_original: str = "",
) -> Optional[dict]:
    """
    Devuelve:
      - dict de respuesta de Telegram  → éxito
      - None                          → no se debe alertar o no hay credenciales
      - {"error": "..."}              → se intentó alertar pero falló
    """
    etiqueta = resultado.get("etiqueta", "")
    score = float(resultado.get("score", 0))

    debe_alertar = (
        etiqueta in ETIQUETAS_ALERTA
        or (etiqueta == "sospechoso" and score >= SCORE_MINIMO_SOSPECHOSO)
    )
    if not debe_alertar:
        return None

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] Token o chat_id no configurados (revisa .env)")
        return {"error": "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados"}

    canal = resultado.get("canal", "?")
    motivos = resultado.get("motivos") or []
    explicacion = resultado.get("explicacion") or ""
    resumen = resultado.get("resumen") or f"[{etiqueta}] score={score}"

    raw_preview = (texto_original[:180] + "...") if len(texto_original) > 180 else texto_original
    preview = html.escape(raw_preview)
    motivos_txt = "\n".join(f"- {html.escape(str(m))}" for m in motivos) or "- (sin motivos detallados)"

    mensaje = (
        f"<b>{html.escape(str(resumen))}</b>\n\n"
        f"<b>Canal:</b> {html.escape(str(canal))}\n"
        f"<b>Score:</b> {score}/100\n\n"
        f"<b>Motivos:</b>\n{motivos_txt}\n\n"
        f"<b>Mensaje analizado:</b>\n<code>{preview}</code>\n\n"
        f"{html.escape(str(explicacion))}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje[:4000],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[telegram] Error al enviar la alerta: {e}")
        return {"error": str(e)}