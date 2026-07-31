# THE TRUTH ENGINE — Capa de Texto (Anti-Phishing)

MVP de detección explicable de phishing / ingeniería social en correo,
SMS y WhatsApp, orientado a patrones de estafa comunes en Colombia.

Este módulo vive en `src/phishing/` y **no modifica ni depende** del
AI-IDS de red (`src/models`, `src/atacks`, `src/data`, etc.). Son dos
productos en el mismo repo, con módulos separados.

---

## 1. Estructura y responsabilidad de cada archivo

```
src/phishing/
├── __init__.py          Punto de entrada del paquete, docstring general
├── features.py           Normalización de texto + señales de CONTENIDO
│                          basadas en reglas (léxicos CO: urgencia, marcas
│                          suplantadas, solicitud de credenciales, estafas
│                          típicas de WhatsApp, ganchos de premio).
│                          También expone clean_for_tfidf(), usada tanto
│                          en entrenamiento como en inferencia para que
│                          ambos vean el texto igual.
├── url_analyzer.py        Señales de URL/dominio: acortadores, TLDs
│                          sospechosos, IPs literales, exceso de
│                          subdominios, marca mencionada en un dominio que
│                          no es el oficial, y detección de homógrafos por
│                          similitud de texto contra una whitelist mínima
│                          de dominios legítimos CO.
├── train_phishing.py     Script de entrenamiento: carga CSV, limpia,
│                          vectoriza con TF-IDF (uni+bigramas), entrena
│                          Regresión Logística balanceada, calcula
│                          métricas y guarda el pipeline con joblib.
├── detector.py            Carga el modelo entrenado y expone
│                          analyze_message(texto, remitente, canal, urls)
│                          -> combina probabilidad del modelo ML + reglas
│                          de contenido + reglas de URL en un score 0-100,
│                          etiqueta y lista de motivos.
└── explain.py              Convierte el resultado de analyze_message()
                             en un texto de alerta legible en español de
                             Colombia (para dashboard, Telegram, o un
                             usuario final).

data/
└── mensajes_co_ejemplo.csv   CSV mínimo (20 filas) solo para probar que
                                el pipeline corre de punta a punta. NO es
                                un dataset de entrenamiento real.

requirements-phishing.txt     Dependencias específicas de este módulo.
```

`audio_spoof/` (deepfake de voz) queda **fuera de alcance** de este
módulo; se anota como línea futura pero no se implementa nada aquí.

---

## 2. Cómo correr el entrenamiento (con datos de ejemplo)

```bash
pip install -r requirements-phishing.txt

python -m src.phishing.train_phishing \
    --data data/mensajes_co_ejemplo.csv \
    --outdir models/phishing
```

Esto genera:
- `models/phishing/phishing_pipeline.joblib` — pipeline TF-IDF + LR
- `models/phishing/metrics.json`
- `models/phishing/training_report.txt`

**Importante:** con 20 filas el modelo no es confiable, es solo para
verificar que el código corre. Ver sección 4 para datasets reales.

## 3. Cómo usar el detector

```python
from src.phishing.detector import analyze_message
from src.phishing.explain import generar_explicacion

resultado = analyze_message(
    "Bancolombia: su cuenta sera bloqueada, valide aqui bit.ly/segur-bancol",
    canal="sms",
)

print(resultado["etiqueta"])   # "phishing" | "sospechoso" | "legitimo"
print(resultado["score"])       # 0-100
print(resultado["motivos"])     # lista de razones en español

print(generar_explicacion(resultado))
```

Por defecto `analyze_message` busca el modelo en
`models/phishing/phishing_pipeline.joblib`; puedes pasar
`model_path="otra/ruta.joblib"` si lo entrenaste en otro lugar.

---

## 4. Datasets reales recomendados (para reemplazar el CSV de ejemplo)

El CSV de ejemplo es solo para probar el pipeline. Para un modelo
usable en producción se necesita un dataset real, idealmente con
ejemplos en español y, si es posible, específicos de Colombia/LatAm:

- **SMS Phishing / Smishing datasets en Kaggle** — buscar "SMS phishing
  dataset" o "smishing dataset"; varios incluyen mensajes en inglés que
  sirven como base pero requieren traducción/adaptación a modismos CO.
- **Phishing email corpora públicos** (ej. Nazario phishing corpus,
  disponible en repositorios académicos) — mayormente en inglés,
  sirven para patrones estructurales (urgencia, solicitud de datos)
  más que para vocabulario en español.
- **Datos propios recolectados con consentimiento**: la fuente más
  valiosa a mediano plazo es que la propia startup reciba reportes de
  usuarios (mensajes reenviados marcados como "esto me pareció
  sospechoso") — eso sí refleja el español y las marcas colombianas
  reales.
- **Aumentar el léxico de `features.py`** con casos reales reportados
  por los primeros usuarios/pilotos es, en la práctica, tan importante
  como conseguir un CSV grande, porque el modelo ML se apoya en esas
  mismas reglas mientras el dataset es pequeño.

Al conseguir un dataset real, solo hace falta que el CSV tenga las
columnas `texto,label[,canal]` — el resto del pipeline no cambia.

---

## 5. Falsos positivos: criterios y cómo evitarlos

Mensajes legítimos que **sí** pueden activar señales de reglas (y por
eso el score combina reglas + modelo ML, en vez de decidir solo con
reglas):

- Notificaciones reales de bancos ("su extracto está disponible", "pago
  recibido") — mencionan la marca pero no piden credenciales ni usan
  urgencia amenazante. `features.py` distingue "menciona marca" de
  "solicita credenciales"; ambas señales juntas pesan mucho más que
  cualquiera por separado.
- Recordatorios legítimos de vencimiento de factura (Claro, Tigo,
  Movistar) que no incluyen enlace ni solicitud de datos.
- Códigos OTP genuinos enviados por el banco al propio usuario (no piden
  reenviar el código, solo lo informan) — la regla de
  `solicitud_datos_sensibles` busca frases como "confirma tu clave" o
  "envía el código", no la sola presencia de la palabra "código".

Recomendaciones para reducir falsos positivos en producción:

1. **Mantener el umbral "sospechoso" (40) separado de "phishing" (70)**
   para no bloquear de una mensajes ambiguos; "sospechoso" debería
   generar una alerta suave, no una acción automática.
2. **Revisar periódicamente `metrics.json`** (matriz de confusión) para
   ver si el modelo empieza a marcar como phishing comunicaciones
   reales frecuentes de un remitente conocido, y ajustar el léxico o
   reentrenar.
3. **Permitir marcar remitentes/dominios de confianza** (lista blanca
   por cliente/empresa) antes de llevar esto a producción como SaaS —
   no implementado aún en este MVP, queda anotado como siguiente paso.

## 6. Limitaciones conocidas del MVP (documentar para el equipo/usuarios)

- El modelo se entrena con TF-IDF, que no captura significado más allá
  de palabras/n-gramas: mensajes de phishing con vocabulario nuevo o
  muy distinto al dataset de entrenamiento pueden pasar desapercibidos.
- La whitelist de dominios legítimos en `url_analyzer.py` es mínima y
  manual; no reemplaza una verificación real de reputación de dominio
  (WHOIS, edad del dominio, certificados) — eso queda para una fase
  posterior (`url_reputation.py` con llamadas de red, fuera de este
  MVP).
- No hay detección de imágenes/adjuntos (ej. capturas de pantalla de
  supuestos bancos) — esta capa es solo texto.
- El dataset de ejemplo (20 filas) es insuficiente para producción;
  las métricas que arroje `train_phishing.py` con ese CSV no deben
  usarse como referencia de calidad real del sistema.
- No incluye aún manejo de remitente verificado (SPF/DKIM en email,
  número verificado en WhatsApp Business) — el parámetro `remitente`
  se recibe y se guarda en el resultado, pero todavía no se usa como
  señal.
