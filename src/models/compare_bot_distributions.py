"""
compare_bot_distributions.py

Diagnostico: compara la distribucion de las features mas importantes
del modelo entre el Bot de CIC-IDS2017 (con el que se entreno) y el
Bot de CTU-13 (malware Neris, sobre el que el modelo generalizo mal).

La hipotesis a confirmar/descartar: el modelo aprendio firmas muy
especificas del bot Ares (C2 por HTTP/click-fraud, presente en
CIC-IDS2017) que no se parecen al comportamiento de red de Neris
(C2 por IRC), y por eso los valores de las features clave caen en
rangos completamente distintos.

Uso:
    python compare_bot_distributions.py
(ajusta las rutas si tu estructura difiere)
"""

from pathlib import Path
import pandas as pd
import joblib

MODEL_DIR = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\models")
MODEL_ROUND = "robust_v4_round3"

CIC_IDS2017_PATH = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\data\processed\full_clean_v4_merged.csv")
CTU13_LABELED_PATH = Path(r"D:\ctu13_output\ctu13_labeled.csv")

TOP_N_FEATURES = 15


def main():
    print("[*] Cargando modelo para obtener ranking de importancia de features...")
    model = joblib.load(MODEL_DIR / f"rf_model_{MODEL_ROUND}.joblib")
    feature_columns = joblib.load(MODEL_DIR / f"feature_columns_{MODEL_ROUND}.joblib")

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(TOP_N_FEATURES)
    print(f"\n[+] Top {TOP_N_FEATURES} features mas importantes del modelo:")
    print(top_features.to_string())

    print(f"\n[*] Cargando Bot de CIC-IDS2017: {CIC_IDS2017_PATH}")
    df_cic = pd.read_csv(CIC_IDS2017_PATH, low_memory=False)
    df_cic.columns = df_cic.columns.str.strip()
    bot_cic = df_cic[df_cic["Label"] == "Bot"]
    print(f"[+] {len(bot_cic):,} filas de Bot en CIC-IDS2017")

    print(f"\n[*] Cargando Bot de CTU-13: {CTU13_LABELED_PATH}")
    df_ctu = pd.read_csv(CTU13_LABELED_PATH, low_memory=False)
    bot_ctu = df_ctu[df_ctu["Label_CTU13"] == "Bot"]
    print(f"[+] {len(bot_ctu):,} filas de Bot en CTU-13")

    print("\n" + "=" * 100)
    print(f"  COMPARACION -- Top {TOP_N_FEATURES} features: Bot CIC-IDS2017 (train) vs Bot CTU-13 (test)")
    print("=" * 100)

    rows = []
    for feat in top_features.index:
        if feat not in bot_cic.columns or feat not in bot_ctu.columns:
            continue
        cic_vals = pd.to_numeric(bot_cic[feat], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
        ctu_vals = pd.to_numeric(bot_ctu[feat], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()

        rows.append({
            "feature": feat,
            "importancia": round(top_features[feat], 4),
            "CIC_mean": round(cic_vals.mean(), 2),
            "CIC_median": round(cic_vals.median(), 2),
            "CTU_mean": round(ctu_vals.mean(), 2),
            "CTU_median": round(ctu_vals.median(), 2),
        })

    comp_df = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print(comp_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("Interpretacion: si CIC_mean/median y CTU_mean/median estan en ordenes de")
    print("magnitud distintos (10x, 100x, o signos opuestos) para las features de mayor")
    print("importancia, es evidencia directa de que el modelo aprendio rangos numericos")
    print("especificos del bot de entrenamiento (Ares) que Neris simplemente no reproduce.")
    print("=" * 100)


if __name__ == "__main__":
    main()
