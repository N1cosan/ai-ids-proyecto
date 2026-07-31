"""
generate_adversarial.py

Genera ejemplos adversarios a escala usando busqueda hill-climbing
(query-based, apta para modelos no diferenciables como Random Forest).

IMPORTANTE sobre la etiqueta: los ejemplos adversarios se guardan con su
etiqueta ORIGINAL real (ej. 'DoS Hulk'), NO como 'BENIGN'. La idea del
adversarial training es enseñarle al modelo: "este patron perturbado
SIGUE siendo un ataque, no te dejes enganar por estos cambios pequeños".

Algoritmo (por muestra):
1. Partir de un ejemplo real de ataque, correctamente clasificado.
2. En cada iteracion, elegir un subconjunto aleatorio de varias features
   y perturbarlas todas a la vez en una direccion aleatoria (perturbar
   una sola feature casi nunca cambia el voto de un arbol individual,
   asi que la busqueda se estanca; con varias features a la vez si se
   cruzan limites de decision reales del ensamble).
3. Aceptar el cambio si no empeora la confianza en la clase correcta
   (permite movimientos laterales para explorar, ya que predict_proba
   de un Random Forest es discreto -- fraccion de arboles que votan).
   Si llevamos muchas iteraciones sin mejora real, agrandamos el paso
   para escapar de mesetas planas.
4. Parar si se logra evasion (el modelo predice otra clase) o si se
   alcanza el limite de iteraciones.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

RANDOM_STATE = 42
MAX_ITERATIONS = 400
STEP_FRACTION = 0.05  # perturbar en pasos de 5% del valor original
N_SAMPLES_PER_CLASS = 30  # cuantos ejemplos de ataque intentar evadir por clase


def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "rf_model_v1.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder_v1.joblib")
    feature_columns = joblib.load(model_dir / "feature_columns_v1.joblib")
    return model, label_encoder, feature_columns


def hill_climb_attack(model, x: np.ndarray, true_class_idx: int,
                       feature_columns: list, rng: np.random.Generator,
                       max_iter: int = MAX_ITERATIONS,
                       n_features_per_step: int = 8) -> tuple[np.ndarray, bool, int]:
    """
    Intenta evadir la deteccion perturbando x mediante hill-climbing.

    NOTA DE DISENO: un Random Forest vota por arbol; predict_proba es la
    fraccion de arboles que eligen cada clase. Perturbar UNA sola feature
    a la vez rara vez cambia el voto de un arbol (cada arbol usa varias
    features en su recorrido), asi que la busqueda se estanca. Por eso
    perturbamos un subconjunto de varias features en simultaneo en cada
    iteracion -- esto se acerca mas a como realmente se cruzan los
    limites de decision de un ensamble de arboles.

    Devuelve (x_adversarial, evadio_exitosamente, iteraciones_usadas)
    """
    x_current = x.copy()
    n_features = len(feature_columns)
    df_cols = feature_columns  # para pasarle nombres de columna al modelo

    def get_proba(vec):
        return model.predict_proba(pd.DataFrame([vec], columns=df_cols))[0][true_class_idx]

    def get_pred(vec):
        return model.predict(pd.DataFrame([vec], columns=df_cols))[0]

    current_proba = get_proba(x_current)
    stall_counter = 0

    for it in range(max_iter):
        k = min(n_features_per_step, n_features)
        feat_indices = rng.choice(n_features, size=k, replace=False)

        step_scale = STEP_FRACTION * (1 + stall_counter // 10)

        best_candidate = None
        best_proba = current_proba

        for _ in range(6):
            candidate = x_current.copy()
            for feat_idx in feat_indices:
                direction = rng.choice([1, -1])
                original_value = candidate[feat_idx]
                step = abs(original_value) * step_scale
                if step == 0:
                    step = 0.01
                candidate[feat_idx] = max(original_value + direction * step, 0)

            candidate_proba = get_proba(candidate)

            if candidate_proba <= best_proba:
                best_proba = candidate_proba
                best_candidate = candidate

        if best_candidate is not None and best_proba < current_proba:
            stall_counter = 0
        else:
            stall_counter += 1

        if best_candidate is not None:
            x_current = best_candidate
            current_proba = best_proba

        if get_pred(x_current) != true_class_idx:
            return x_current, True, it + 1

    return x_current, False, max_iter


def generate_adversarial_dataset(df: pd.DataFrame, model, label_encoder,
                                  feature_columns: list) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    adversarial_rows = []

    attack_labels = [l for l in df["Label"].unique() if l not in ("BENIGN", "Heartbleed")]

    for label in attack_labels:
        true_class_idx = label_encoder.transform([label])[0]
        subset = df[df["Label"] == label]
        n = min(N_SAMPLES_PER_CLASS, len(subset))
        samples = subset.sample(n, random_state=RANDOM_STATE)

        n_success = 0
        print(f"\nGenerando adversarios para clase '{label}' ({n} muestras)...")

        for _, row in samples.iterrows():
            x = row[feature_columns].values.astype(float)

            original_pred = model.predict(pd.DataFrame([x], columns=feature_columns))[0]
            if original_pred != true_class_idx:
                continue

            x_adv, success, iters = hill_climb_attack(
                model, x, true_class_idx, feature_columns, rng
            )

            if success:
                n_success += 1
                adv_row = dict(zip(feature_columns, x_adv))
                adv_row["Label"] = label
                adversarial_rows.append(adv_row)

        print(f"  Evasiones exitosas: {n_success}/{n} ({100*n_success/max(n,1):.1f}%)")

    return pd.DataFrame(adversarial_rows)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    MODEL_DIR = BASE_DIR / "models"
    DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "adversarial_samples.csv"

    print("Cargando modelo y datos...")
    model, label_encoder, feature_columns = load_artifacts(MODEL_DIR)
    df = pd.read_csv(DATA_PATH)

    print("Generando ejemplos adversarios (esto puede tardar varios minutos)...")
    adv_df = generate_adversarial_dataset(df, model, label_encoder, feature_columns)

    print(f"\n{'='*60}")
    print(f"Total de ejemplos adversarios generados: {len(adv_df)}")
    if len(adv_df) > 0:
        print(adv_df["Label"].value_counts())
        adv_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nGuardado en: {OUTPUT_PATH}")
    else:
        print("No se generaron ejemplos adversarios exitosos.")