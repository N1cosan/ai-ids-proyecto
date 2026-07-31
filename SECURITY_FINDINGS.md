### Plan de mitigacion (pendiente)

- [ ] Reentrenar el modelo incluyendo las muestras adversarias generadas
  ```
  como ejemplos adicionales de la clase de ataque correspondiente

  (adversarial training / data augmentation)
  ```

- [ ] Evaluar si un ensemble con distintos tipos de modelo (ej. Random
  ```
  Forest + Gradient Boosting) reduce la superficie de evasion comun
  ```

- [ ] ### Pendientes
  - [ ] Ejecutar la ronda 3 (el script se detuvo en la ronda 2,
        posiblemente por un valor de configuracion incorrecto al copiar
        el archivo)
  - [x] Verificar generalizacion en trafico verdaderamente nunca visto
        (Tuesday.csv) -- ver seccion siguiente
  - [ ] Evaluar si un ensemble con distintos tipos de modelo reduce aun
        mas la superficie de evasion
  - [ ] Repetir este mismo proceso de auditoria (generar adversarios +
        medir) periodicamente conforme el modelo evolucione, no como
        un chequeo de una sola vez
  ## Verificacion con datos nunca vistos (Tuesday.csv)
  **Fecha:** 30/07/2026
  **Script:** `src/models/verify_with_tuesday.py`
  **Dataset:** Tuesday-WorkingHours.pcap_ISCX.csv (nunca usado en ningun
  entrenamiento anterior, ni original ni adversario)
  ### Resultado 1: Generalizacion en trafico normal (buena noticia)
  Sobre 412,476 muestras BENIGN verdaderamente nuevas:
  - Correctamente clasificadas: 99.67%
  - Falsos positivos: 0.33% (1377 casos, mayoria confundidos con DoS Hulk)
  Esto confirma que el modelo generaliza razonablemente bien en el caso
  general, no solo memorizo el dataset de entrenamiento original.
  ### Resultado 2: VACIO DE COBERTURA CRITICO (no es un bug, es alcance)
  El modelo NO detecta ataques de fuerza bruta:
  - FTP-Patator: 100% de las muestras pasaron desapercibidas como BENIGN
  - SSH-Patator: 99.7% de las muestras pasaron desapercibidas como BENIGN
  Esto NO es una falla de la mitigacion adversaria ni una vulnerabilidad
  nueva -- es que el modelo unicamente fue entrenado con Monday (benigno)
  y Wednesday (ataques DoS: Hulk, GoldenEye, slowloris, Slowhttptest).
  Nunca vio ejemplos de fuerza bruta, y los patrones de trafico de este
  tipo de ataque (intentos repetidos de conexion, timing de logins
  fallidos) son estructuralmente distintos a los de DoS.
  **Relevancia de negocio:** el caso de uso original del producto
  ("coincide en un 98% con un ataque de fuerza bruta sobre SSH") no
  esta cubierto actualmente por el modelo en produccion.
  ### Siguiente paso recomendado
  Incorporar Tuesday.csv al dataset de entrenamiento para extender la
  cobertura del modelo a fuerza bruta (FTP-Patator, SSH-Patator) como
  clases nuevas, y repetir todo el proceso (entrenamiento, explicabilidad,
  auditoria adversaria) para el modelo ampliado.

