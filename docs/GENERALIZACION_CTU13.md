# Generalización externa del IDS: validación con CTU-13

## Motivación

Todas las versiones del modelo hasta `rf_model_robust_v4_round3` fueron
entrenadas **y evaluadas** exclusivamente dentro de CIC-IDS2017. Esto deja
una pregunta sin responder: ¿el modelo generaliza a tráfico de red real que
nunca vio, generado por otro malware, en otra red, capturado con otra
metodología? Un modelo puede lucir excelente en su propio dataset y fallar
por completo ante datos externos.

Este documento resume una prueba de generalización externa usando
**CTU-13** (Stratosphere Labs, capturas de botnets reales de 2011),
completamente independiente de CIC-IDS2017.

## Hallazgo 1 — `rf_model_robust_v4_round3` no generaliza a malware externo

Se procesó el tráfico completo del Scenario 1 de CTU-13 (malware Neris,
6GB, 3,857,394 flujos) con el mismo pipeline de extracción de features
(CICFlowMeter) que generó el dataset de entrenamiento original, y se
etiquetó con el ground truth oficial de Stratosphere.

| Métrica | Resultado |
|---|---|
| Flujos Bot reales evaluados | 10,619 |
| Detectados como "Bot" exacto | 0 (0.0%) |
| Detectados como cualquier ataque | 49 (0.5%) |
| Falsos negativos | 10,570 (99.5%) |
| Falsos positivos sobre BENIGN real (862,753 flujos) | 49 (0.01%) |

**Conclusión:** el modelo tiene especificidad excelente (99.99% en BENIGN
externo), pero prácticamente no detecta el malware Neris. La causa raíz,
confirmada comparando distribuciones de features clave, es que el modelo
memorizó la firma específica del bot Ares (click-fraud vía HTTP, ráfagas
rápidas, payloads sustanciales) presente en CIC-IDS2017, en vez de un
patrón general de "comportamiento de botnet". Neris usa C2 vía IRC, con
comportamiento de red casi opuesto (conexiones mayormente inactivas,
paquetes sin payload). Ejemplos de features con órdenes de magnitud
distintos: `Fwd IAT Mean` 42x más lento en CTU-13, `Destination Port`
mediana 8080 (proxy HTTP) en CIC-IDS2017 vs. 53 (DNS) en CTU-13.

## Intento de corrección — v5 (fallido, con lección metodológica importante)

Se diversificó la clase "Bot" agregando muestras reales de tres familias de
malware de CTU-13 (Neris, Rbot, Virut), reservando un holdout del 30% de
cada familia nunca visto en entrenamiento.

**Error cometido:** se agregaron muestras "Bot" de CTU-13 sin ningún
"BENIGN" de la misma fuente. El origen de los datos (CTU-13 vs
CIC-IDS2017) se convirtió en una variable confusora perfectamente
correlacionada con la etiqueta.

| Métrica (sobre holdout de Neris, nunca visto) | v5 |
|---|---|
| Detección de Bot | 100.0% |
| Falsos positivos sobre BENIGN real | **100.00%** (colapso total) |

El modelo no aprendió "comportamiento de botnet" — aprendió a reconocer
"esto viene de la captura de CTU-13", marcando absolutamente todo como
ataque sin importar la etiqueta real.

## Corrección — v6

Se agregó también una porción de BENIGN real de CTU-13 (Neris, 603,927
filas de entrenamiento / 258,826 de holdout, mismo split 70/30
disciplinado), rompiendo la correlación perfecta entre origen y etiqueta.

| Familia | N (holdout, nunca visto) | Detección Bot |
|---|---|---|
| Neris | 3,186 | 97.9% |
| Rbot | 79 | 98.7% |
| Virut | 558 | 99.6% |

| Métrica | v4_round3 | v5 | **v6** |
|---|---|---|---|
| Detección Bot (Neris) | 0.0% | 100.0% | **97.9%** |
| Falsos positivos BENIGN (Neris) | 0.01% | 100.00% (colapso) | **5.70%** |

Ninguna otra clase (DDoS, PortScan, fuerza bruta, etc.) se degradó —
todas se mantienen en 0.99–1.00 de precision/recall.

### Diagnóstico de los falsos positivos restantes

Se investigó si el 5.7% de falsos positivos se concentraba en los hosts
"normales" marcados oficialmente por Stratosphere como menos confiables
(CVUT-WebServer, CVUT-DNS-Server, MatLab-Server). **La hipótesis no se
sostuvo**: los hosts con ground truth más sólido (Stribrek, Grill, Jist —
workstations reales de usuario) mostraron tasas de falsos positivos más
altas (12.8%–16.8%) que el host de mayor volumen y etiqueta "menos
confiable" (CVUT-DNS-Server, 5.4%). Esto indica que los falsos positivos
reflejan confusión genuina del modelo entre comportamiento normal de
usuario y patrones de botnet, no ruido de etiquetado.

### Ajuste de umbral (calibración post-hoc, sin reentrenar)

| Umbral P(Bot) | Falsos positivos | Detección Bot |
|---|---|---|
| 0.5 (por defecto) | 5.70% | 98.1% |
| 0.7 | 4.51% | 96.8% |
| **0.8 (recomendado)** | **3.52%** | **94.7%** |
| 0.9 | 2.28% | 89.8% |

**Punto de operación recomendado: umbral 0.8** — reduce los falsos
positivos en 38% relativo (5.7% → 3.5%) a cambio de solo 3.4 puntos de
recall, manteniendo una mejora enorme frente al 0% original.

## Conclusión general

Diversificar el entrenamiento con múltiples familias de malware reales
mejora sustancial y genuinamente la generalización de detección de Bot
(0% → ~95-98% según umbral), a costa de un aumento medible pero razonable
en falsos positivos (0.01% → 3.5-5.7%). Este es un trade-off legítimo de
precision/recall, no un artefacto — documentado con datos externos reales,
nunca vistos en entrenamiento, y con la disciplina metodológica de
holdouts separados en cada etapa.

## Limitaciones conocidas y trabajo futuro

- El reentrenamiento v6 **no repite** las rondas de adversarial training
  que sí tiene `rf_model_robust_v4_round3` — queda como trabajo futuro
  aplicar ese proceso sobre v6 si se justifica.
- Solo se incorporó BENIGN externo de una red (Neris/CTU-13). Sumar
  BENIGN de más redes reduciría aún más el riesgo de que el origen de
  los datos actúe como variable confusora residual.
- Solo se probaron 3 de las 13 familias de malware de CTU-13. Ampliar a
  más familias (Sogou, Murlo, NSIS.ay, Menti) diversificaría aún más el
  aprendizaje, aunque implica procesar capturas mucho más pesadas
  (90-120GB en algunos casos).
- El tráfico "Background" de CTU-13 (~77% del total capturado, sin
  ground truth confiable) se excluyó completamente de esta evaluación,
  siguiendo la metodología oficial de Stratosphere -- no se inventó
  ninguna etiqueta para datos sin verificar.

## Artefactos generados

- `prepare_ctu13_csv.py` — alinea columnas de CICFlowMeter al esquema
  exacto del modelo (78 features, incluyendo el bug histórico de
  `Fwd Header Length` duplicada).
- `label_ctu13.py` — etiqueta Bot/BENIGN por IP según ground truth
  oficial de Stratosphere, con filtro opcional de tráfico DNS.
- `build_ctu13_augmented_dataset_v6.py` — construye el dataset de
  entrenamiento aumentado con split train/eval disciplinado por familia.
- `train_v6_ctu13_augmented.py` — reentrenamiento.
- `evaluate_v6_holdout.py`, `diagnose_fp_hosts_fixed.py` — evaluación y
  diagnóstico sobre el holdout nunca visto.
- Modelos: `rf_model_v6_ctu13_augmented.joblib` (+ encoder y columnas).
