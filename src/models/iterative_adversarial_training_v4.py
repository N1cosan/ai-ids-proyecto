"""
iterative_adversarial_training.py
Entrenamiento adversario iterativo (varias rondas) combinado con
sobremuestreo de los ejemplos adversarios.
DECISION METODOLOGICA IMPORTANTE:
Las muestras de ataque se dividen en dos grupos desde el inicio:
- adv_train_pool: se usa SOLO para generar ejemplos adversarios de
  entrenamiento (el modelo los ve durante el fit).
- adv_eval_pool: se usa SOLO para medir la tasa de evasion (el modelo
  NUNCA los ve durante el entrenamiento).
Si midieramos la evasion sobre las mismas muestras que usamos para
generar ejemplos de entrenamiento, estariamos midiendo si el modelo
"memorizo" esos casos puntuales, no si generalizo una defensa real.
Esta separacion es la unica forma honesta de saber si mejoro de verdad.
Loop por ronda:
1. Generar ejemplos adversarios contra el modelo ACTUAL, usando
   solo adv_train_pool.
2. Acumular esos ejemplos en un pool de entrenamiento adversario
   (crece ronda a ronda).
3. Sobremuestrear ese pool (repetirlo OVERSAMPLE_FACTOR veces) al
   combinarlo con el dataset original limpio.
4. Reentrenar un modelo nuevo desde cero con esa combinacion.
5. Medir la tasa de evasion del modelo nuevo contra adv_eval_pool
   (nunca antes visto).
6. Repetir.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from generate_adversarial import hill_climb_attack

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 50
N_ROUNDS = 3
OVERSAMPLE_FACTOR = 25
N_ADV_SAMPLES_PER_CLASS_TRAIN = 60
N_EVAL_SAMPLES_PER_CLASS = 15


def filter_rare_classes(df, label_col="Label", min_samples=MIN_SAMPLES_PER_CLASS):
    counts = df[label_col].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"Excluyendo clases raras: {rare}")
        df = df[~df[label_col].isin(rare)]
    return df


def split_attack_pools(df: pd.DataFrame, label_col: str = "Label"):
    """Separa cada clase de ataque en pool de entrenamiento (70%) y de evaluacion (30%)."""
    benign = df[df[label_col] == "BENIGN"]
    attacks = df[df[label_col] != "BENIGN"]
    train_parts, eval_parts = [], []
    for label, group in attacks.groupby(label_col):
        tr, ev = train_test_split(group, test_size=0.3, random_state=RANDOM_STATE)
        train_parts.append(tr)
        eval_parts.append(ev)
    adv_train_pool = pd.concat(train_parts, ignore_index=True)
    adv_eval_pool = pd.concat(eval_parts, ignore_index=True)
    return benign, adv_train_pool, adv_eval_pool


def generate_round_adversarials(pool: pd.DataFrame, model, label_encoder,
                                 feature_columns: list, n_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in pool["Label"].unique():
        true_idx = label_encoder.transform([label])[0]
        subset = pool[pool["Label"] == label]
        n = min(n_per_class, len(subset))
        samples = subset.sample(n, random_state=seed)
        n_success = 0
        for _, row in samples.iterrows():
            x = row[feature_columns].values.astype(float)
            pred = model.predict(pd.DataFrame([x], columns=feature_columns))[0]
            if pred != true_idx:
                continue
            x_adv, success, _ = hill_climb_attack(
                model, x, true_idx, feature_columns, rng, max_iter=160
            )
            if success:
                n_success += 1
                r = dict(zip(feature_columns, x_adv))
                r["Label"] = label
                rows.append(r)
        print(f" {label:20s}: {n_success}/{n} evasiones al generar")
    return pd.DataFrame(rows)


def measure_evasion_rate(pool: pd.DataFrame, model, label_encoder,
                          feature_columns: list, n_per_class: int, seed: int = 999) -> float:
    """Mide evasion SOLO sobre el pool de evaluacion, nunca visto en entrenamiento."""
    rng = np.random.default_rng(seed)
    total_success, total_tested = 0, 0
    for label in pool["Label"].unique():
        true_idx = label_encoder.transform([label])[0]
        subset = pool[pool["Label"] == label]
        n = min(n_per_class, len(subset))
        samples = subset.sample(n, random_state=seed)
        n_success, n_tested = 0, 0
        for _, row in samples.iterrows():
            x = row[feature_columns].values.astype(float)
            pred = model.predict(pd.DataFrame([x], columns=feature_columns))[0]
            if pred != true_idx:
                continue
            n_tested += 1
            _, success, _ = hill_climb_attack(
                model, x, true_idx, feature_columns, rng, max_iter=140
            )
            if success:
                n_success += 1
        total_success += n_success
        total_tested += n_tested
        rate = 100 * n_success / max(n_tested, 1)
        print(f" {label:20s}: {n_success}/{n_tested} ({rate:.1f}%)")
    global_rate = 100 * total_success / max(total_tested, 1)
    return global_rate


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR / "data" / "processed" / "full_clean_v4_merged.csv"
    MODEL_DIR = BASE_DIR / "models"
    print("Cargando dataset original...")
    df = pd.read_csv(DATA_PATH)
    df = filter_rare_classes(df)
    print("Separando pools de ataque (70% train-adv / 30% eval-adv, nunca mezclados)...")
    benign_df, adv_train_pool, adv_eval_pool = split_attack_pools(df)
    print(f"BENIGN: {len(benign_df)} | adv_train_pool: {len(adv_train_pool)} | adv_eval_pool: {len(adv_eval_pool)}")
    base_clean_df = pd.concat([benign_df, adv_train_pool], ignore_index=True)
    print("\n=== Entrenando modelo base (ronda 0, sin adversarios) ===")
    X0 = base_clean_df.drop(columns=["Label"])
    y0_raw = base_clean_df["Label"]
    label_encoder = LabelEncoder()
    y0 = label_encoder.fit_transform(y0_raw)
    feature_columns = list(X0.columns)
    current_model = train_model(X0, y0)
    print("\nMidiendo evasion BASE (ronda 0) contra adv_eval_pool (holdout, nunca visto)...")
    rate = measure_evasion_rate(adv_eval_pool, current_model, label_encoder, feature_columns, N_EVAL_SAMPLES_PER_CLASS)
    print(f"\n>>> Tasa de evasion ronda 0 (baseline): {rate:.1f}%\n")
    history = [("ronda_0_baseline", rate)]
    cumulative_adv_pool = pd.DataFrame()
    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"RONDA {round_num}: generando adversarios contra el modelo actual...")
        print(f"{'='*60}")
        new_adv = generate_round_adversarials(
            adv_train_pool, current_model, label_encoder, feature_columns,
            N_ADV_SAMPLES_PER_CLASS_TRAIN, seed=RANDOM_STATE + round_num,
        )
        print(f"Nuevos ejemplos adversarios generados esta ronda: {len(new_adv)}")
        cumulative_adv_pool = pd.concat([cumulative_adv_pool, new_adv], ignore_index=True)
        oversampled_adv = pd.concat([cumulative_adv_pool] * OVERSAMPLE_FACTOR, ignore_index=True)
        combined_df = pd.concat([base_clean_df, oversampled_adv], ignore_index=True)
        print(f"Dataset combinado para reentrenar: {combined_df.shape} "
              f"(adversarios acumulados: {len(cumulative_adv_pool)}, x{OVERSAMPLE_FACTOR} sobremuestreo)")
        X = combined_df.drop(columns=["Label"])
        y = label_encoder.transform(combined_df["Label"])
        print(f"Reentrenando modelo ronda {round_num}...")
        current_model = train_model(X, y)
        print(f"\nMidiendo evasion ronda {round_num} contra adv_eval_pool (holdout)...")
        rate = measure_evasion_rate(adv_eval_pool, current_model, label_encoder, feature_columns, N_EVAL_SAMPLES_PER_CLASS)
        print(f"\n>>> Tasa de evasion ronda {round_num}: {rate:.1f}%\n")
        history.append((f"ronda_{round_num}", rate))
        joblib.dump(current_model, MODEL_DIR / f"rf_model_robust_v4_round{round_num}.joblib")
        joblib.dump(label_encoder, MODEL_DIR / f"label_encoder_robust_v4_round{round_num}.joblib")
        joblib.dump(feature_columns, MODEL_DIR / f"feature_columns_robust_v4_round{round_num}.joblib")
    print(f"\n{'='*60}")
    print("RESUMEN DE TODAS LAS RONDAS (tasa de evasion sobre holdout nunca visto)")
    print(f"{'='*60}")
    for name, rate in history:
        print(f" {name:20s}: {rate:.1f}%")