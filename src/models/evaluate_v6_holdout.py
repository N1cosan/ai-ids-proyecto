"""
evaluate_v6_holdout.py

Prueba real de generalizacion para el modelo v6: mide deteccion de Bot
Y tasa de falsos positivos sobre BENIGN, ambos usando el holdout que
se aparto ANTES de cualquier entrenamiento (nunca visto), desglosado
por familia de malware y por origen.

A diferencia de evaluate_v5_holdout.py, este holdout SI incluye BENIGN
real de CTU-13 (Neris), porque v6 corrigio el problema de v5 (variable
confusora: origen de los datos = proxy perfecto de la etiqueta).

Uso:
    python evaluate_v6_holdout.py
"""

from pathlib import Path
import pandas as pd
import joblib

MODEL_DIR = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\models")
MODEL_ROUND = "v6_ctu13_augmented"

HOLDOUT_CSV = Path(r"D:\ctu13_output\ctu13_holdout_eval_v6.csv")


def main():
    print("[*] Cargando modelo v6...")
    model = joblib.load(MODEL_DIR / f"rf_model_{MODEL_ROUND}.joblib")
    label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{MODEL_ROUND}.joblib")
    feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{MODEL_ROUND}.joblib")

    print(f"\n[*] Cargando holdout v6 (nunca visto en entrenamiento): {HOLDOUT_CSV}")
    df = pd.read_csv(HOLDOUT_CSV, low_memory=False)
    print(f"[+] {len(df):,} filas cargadas")

    print("\n" + "=" * 95)
    print("  RESULTADOS -- Holdout v6 NUNCA VISTO en entrenamiento (por familia y etiqueta real)")
    print("=" * 95)

    print(f"\n{'Familia':<10} {'Etiqueta real':<10} {'N':>10} {'Pred Bot exacto':>16} {'%':>7} "
          f"{'Pred no-BENIGN':>15} {'%':>7}")
    print("-" * 95)

    resumen = []
    for (familia, label_real), grupo in df.groupby(["Familia_CTU13", "Label"]):
        X = grupo[feature_columns].copy()
        X = X.replace([float("inf"), float("-inf")], pd.NA)
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]

        y_pred_idx = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_idx)

        n = len(y_pred)
        n_bot_exacto = (y_pred == "Bot").sum()
        n_no_benign = (y_pred != "BENIGN").sum()

        print(f"{familia:<10} {label_real:<10} {n:>10,} {n_bot_exacto:>16,} "
              f"{100*n_bot_exacto/n:>6.1f}% {n_no_benign:>15,} {100*n_no_benign/n:>6.1f}%")

        resumen.append({"familia": familia, "label_real": label_real, "n": n,
                         "bot_exacto": n_bot_exacto, "no_benign": n_no_benign})

    print("-" * 95)

    print("\n" + "=" * 95)
    print("  LECTURA -- Deteccion de Bot real (recall) y Falsos positivos sobre BENIGN real")
    print("=" * 95)
    for r in resumen:
        if r["label_real"] == "Bot":
            print(f"  [{r['familia']}] Bot detectado (recall): {r['no_benign']:,}/{r['n']:,} "
                  f"({100*r['no_benign']/r['n']:.1f}%) -- exacto 'Bot': {r['bot_exacto']:,} "
                  f"({100*r['bot_exacto']/r['n']:.1f}%)")
        else:
            print(f"  [{r['familia']}] Falsos positivos sobre BENIGN real: {r['no_benign']:,}/{r['n']:,} "
                  f"({100*r['no_benign']/r['n']:.2f}%)")

    print("\n" + "=" * 95)
    print("  COMPARACION RESUMEN -- v4_round3 vs v5 vs v6 (sobre Neris, mismo tipo de prueba)")
    print("=" * 95)
    print(f"{'Modelo':<15} {'Deteccion Bot (Neris)':>25} {'Falsos positivos BENIGN (Neris)':>35}")
    print(f"{'v4_round3':<15} {'0.0%':>25} {'0.01%':>35}")
    print(f"{'v5':<15} {'100.0%':>25} {'100.00% (colapso)':>35}")
    neris_bot = next((r for r in resumen if r["familia"] == "Neris" and r["label_real"] == "Bot"), None)
    neris_benign = next((r for r in resumen if r["familia"] == "Neris" and r["label_real"] == "BENIGN"), None)
    if neris_bot and neris_benign:
        print(f"{'v6':<15} {100*neris_bot['no_benign']/neris_bot['n']:>24.1f}% "
              f"{100*neris_benign['no_benign']/neris_benign['n']:>34.2f}%")
    print("=" * 95)


if __name__ == "__main__":
    main()
