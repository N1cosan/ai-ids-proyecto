## ## Verificación de generalización: Thursday + Friday (nunca vistos)

**Fecha:** 31/07/2026

**Script:** `src/models/verify_generalization.py`

**Modelo evaluado:** `rf_model_robust_v3_round3.joblib`

**Dataset:** Thursday-Morning-WebAttacks, Thursday-Afternoon-Infilteration,

Friday-Morning, Friday-Afternoon-PortScan, Friday-Afternoon-DDoS

(1,017,939 filas combinadas, nunca usadas en ningún entrenamiento)

### Resultado 1: Generalización en tráfico normal (confirmado, buena noticia)

Sobre 795,104 muestras BENIGN verdaderamente nuevas:

- Correctamente clasificadas: 99.90%

- Falsos positivos: 0.10% (776 casos, mayoría confundidos con DoS Hulk)

Consistente con el resultado ya visto en Tuesday.csv (99.67%). El modelo

generaliza el tráfico normal de forma estable across distintos días.

### Resultado 2: Vacío de cobertura ante tipos de ataque nunca vistos

| Ataque | % pasa desapercibido | % genera alguna alerta |

|---|---|---|

| Web Attack (Brute Force) | 95.5% | 4.5% |

| Web Attack (XSS) | 94.6% | 5.4% |

| Web Attack (SQL Injection) | 81.0% | 19.0% |

| Infiltration | 100.0% | 0.0% |

| Bot | 100.0% | 0.0% |

| PortScan | 99.8% | 0.2% |

| DDoS | 43.5% | 56.5% |

**No es una falla del modelo ni de la mitigación adversaria** — es alcance:

el modelo solo fue entrenado con DoS (Hulk, GoldenEye, slowloris,

Slowhttptest) y fuerza bruta (FTP-Patator, SSH-Patator). Nunca vio

ejemplos de Web Attacks, Infiltration, Bot o PortScan/DDoS.

**Hallazgo notable:** DDoS es el único caso donde una mayoría (56.5%)

sí dispara alguna alerta, aunque siempre bajo la etiqueta incorrecta

"DoS Hulk" — atribuible a que ambos comparten un patrón estructural

de volumen alto de paquetes en poco tiempo.

**Hallazgo crítico:** Infiltration y Bot pasan 100% desapercibidos.

Son los tipos de ataque estructuralmente más distintos a los que el

modelo conoce (movimiento lateral / comunicación con C2 vs. volumen

de tráfico), y representan el vacío de cobertura más serio detectado

hasta ahora.

### Relevancia de negocio

El producto actualmente cubre bien: DoS, fuerza bruta (SSH/FTP), y

parcialmente DDoS (por coincidencia estructural con DoS Hulk, con

etiqueta incorrecta). NO cubre: ataques web (XSS, SQLi, brute force

web), infiltración, ni tráfico de botnets.

### Siguiente paso recomendado

Igual que se hizo con Tuesday.csv → fuerza bruta, el camino directo

es incorporar Thursday+Friday al dataset de entrenamiento como nuevas

clases, y repetir el ciclo completo (entrenar, medir sanidad, correr

adversarial training) para un modelo v4 con cobertura ampliada.

Nota: Infiltration solo tiene 36 muestras totales en todo el dataset,

por lo que probablemente quede excluida del entrenamiento por el mismo

motivo que Heartbleed (MIN_SAMPLES_PER_CLASS) — limitación a documentar,

no a forzar.

