"""
measure_evasion.py

Mide la tasa de evasion (hill-climbing) contra un modelo especifico.
Permite comparar v1 (original) vs v2 (con adversarial training) usando
el MISMO ataque, para confirmar con datos si la mitigacion funciono.

Uso:
    python src/models/measure_evasion.py v1
    python src/models/measure_evasion.py v2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from generate_adversarial import hill_climb_attack


def load_artifacts(model_dir: Path, version: str):
    model = joblib.load(model_dir / f"rf_model_{version}.joblib")
    label_encoder = joblib.load(model_dir / f"label_encoder_{version}.joblib")
    feature_columns = joblib.load(model_dir / f"feature_columns_{version}.joblib")
    return model, label_encoder, feature_columns


def measure(df, model, label_encoder, feature_columns, n_per_class: int = 30):
    rng = np.random.default_rng(123)
    results = {}

    attack_labels = [l for l in df["Label"].unique() if l not in ("BENIGN", "Heartbleed")]

    for label in attack_labels:
        true_class_idx = label_encoder.transform([label])[0]
        subset = df[df["Label"] == label]
        n = min(n_per_class, len(subset))
        samples = subset.sample(n, random_state=123)

        n_success = 0
        n_tested = 0

        for _, row in samples.iterrows():
            x = row[feature_columns].values.astype(float)
            pred = model.predict(pd.DataFrame([x], columns=feature_columns))[0]
            if pred != true_class_idx:
                continue
            n_tested += 1

            _, success, _ = hill_climb_attack(model, x, true_class_idx, feature_columns, rng)
            if success:
                n_success += 1

        rate = 100 * n_success / max(n_tested, 1)
        results[label] = (n_success, n_tested, rate)
        print(f"{label:20s}: {n_success}/{n_tested} evasiones ({rate:.1f}%)")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("v1", "v2"):
        print("Uso: python measure_evasion.py [v1|v2]")
        sys.exit(1)

    version = sys.argv[1]

    BASE_DIR = Path(__file__).resolve().parents[2]
    MODEL_DIR = BASE_DIR / "models"
    DATA_PATH = BASE_DIR / "data" / "processed" / "monday_wednesday_clean.csv"

    print(f"Midiendo tasa de evasion contra el modelo {version}...")
    model, label_encoder, feature_columns = load_artifacts(MODEL_DIR, version)
    df = pd.read_csv(DATA_PATH)

    print(f"\n{'='*60}")
    results = measure(df, model, label_encoder, feature_columns)
    print(f"{'='*60}")

    total_success = sum(r[0] for r in results.values())
    total_tested = sum(r[1] for r in results.values())
    print(f"\nTASA DE EVASION GLOBAL ({version}): "
          f"{total_success}/{total_tested} ({100*total_success/max(total_tested,1):.1f}%)")