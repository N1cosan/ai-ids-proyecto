"""
build_ctu13_augmented_dataset_v6.py

Correccion sobre v5: v5 solo agrego muestras 'Bot' de CTU-13 sin
ningun 'BENIGN' de la misma fuente. Eso convirtio el origen de los
datos (CTU-13 vs CIC-IDS2017) en una variable confusora perfectamente
correlacionada con la etiqueta -- el modelo v5 aprendio a reconocer
"esto viene de la red de CTU-13" en vez de "esto es trafico de
botnet", y por eso marco el 100% del BENIGN real de CTU-13 como
ataque (falsos positivos).

v6 corrige esto agregando tambien una porcion de BENIGN real de
CTU-13 (de Neris, la unica familia de la que tenemos BENIGN real
capturado) al entrenamiento, con el mismo split disciplinado 70/30
train/eval que ya se aplico a Bot. Asi el origen de los datos deja de
predecir la etiqueta por si solo, y el modelo tiene que aprender la
diferencia real de comportamiento dentro de la misma red.

Uso:
    python build_ctu13_augmented_dataset_v6.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
EVAL_FRACTION = 0.30

CIC_IDS2017_PATH = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\data\processed\full_clean_v4_merged.csv")

# (nombre_familia, ruta_csv_etiquetado, tiene_benign)
NEW_SOURCES = [
    ("Neris", Path(r"D:\ctu13_output\ctu13_labeled.csv"), True),
    ("Rbot",  Path(r"D:\ctu13_rbot_bot_output\rbot_only_labeled.csv"), False),
    ("Virut", Path(r"D:\ctu13_virut_output\virut_labeled.csv"), False),
]

OUT_TRAIN_PATH = Path(r"C:\Users\nicko\Documents\ai-ids-proyecto\ai-ids\data\processed\full_clean_v6_ctu13_augmented.csv")
OUT_EVAL_PATH = Path(r"D:\ctu13_output\ctu13_holdout_eval_v6.csv")


def split_and_tag(df_class, family_name, label_value):
    train_part, eval_part = train_test_split(
        df_class, test_size=EVAL_FRACTION, random_state=RANDOM_STATE
    )
    train_part = train_part.copy()
    eval_part = eval_part.copy()
    train_part["Label"] = label_value
    eval_part["Label"] = label_value
    train_part["Familia_CTU13"] = family_name
    eval_part["Familia_CTU13"] = family_name
    return train_part, eval_part


def main():
    print(f"[*] Cargando dataset base CIC-IDS2017: {CIC_IDS2017_PATH}")
    df_cic = pd.read_csv(CIC_IDS2017_PATH, low_memory=False)
    df_cic.columns = df_cic.columns.str.strip()
    print(f"[+] {len(df_cic):,} filas")

    model_feature_cols = [c for c in df_cic.columns if c != "Label"]

    train_parts = []
    eval_parts = []

    for family_name, path, has_benign in NEW_SOURCES:
        if not path.exists():
            print(f"\n[ERROR] No se encontro el CSV de {family_name}: {path}")
            continue

        print(f"\n[*] Procesando familia: {family_name} ({path})")
        df_fam = pd.read_csv(path, low_memory=False)

        bot_rows = df_fam[df_fam["Label_CTU13"] == "Bot"].copy()
        faltantes = [c for c in model_feature_cols if c not in bot_rows.columns]
        if faltantes:
            print(f"    [ERROR] Faltan columnas: {faltantes}. Se omite esta familia.")
            continue
        bot_rows = bot_rows[model_feature_cols].copy()

        bot_train, bot_eval = split_and_tag(bot_rows, family_name, "Bot")
        print(f"    Bot    -> train: {len(bot_train):,} | eval: {len(bot_eval):,}")
        train_parts.append(bot_train)
        eval_parts.append(bot_eval)

        if has_benign:
            benign_rows = df_fam[df_fam["Label_CTU13"] == "BENIGN"].copy()
            benign_rows = benign_rows[model_feature_cols].copy()
            benign_train, benign_eval = split_and_tag(benign_rows, family_name, "BENIGN")
            print(f"    BENIGN -> train: {len(benign_train):,} | eval: {len(benign_eval):,}")
            train_parts.append(benign_train)
            eval_parts.append(benign_eval)

    if not train_parts:
        print("\n[ERROR] No se pudo procesar ninguna familia nueva. Abortando.")
        return

    new_train = pd.concat(train_parts, ignore_index=True)
    new_eval = pd.concat(eval_parts, ignore_index=True)

    df_cic_for_train = df_cic.copy()
    df_cic_for_train["Familia_CTU13"] = "CIC-IDS2017"
    new_train_for_concat = new_train[model_feature_cols + ["Label", "Familia_CTU13"]]

    df_augmented = pd.concat([df_cic_for_train, new_train_for_concat], ignore_index=True)

    print(f"\n{'='*70}")
    print("  DATASET DE ENTRENAMIENTO AUMENTADO (v6, con BENIGN de CTU-13)")
    print(f"{'='*70}")
    print(f"Shape final: {df_augmented.shape}")
    print("\nDistribucion de clases:")
    print(df_augmented["Label"].value_counts().to_string())
    print(f"\nOrigen de las filas Bot:")
    print(df_augmented[df_augmented["Label"] == "Bot"]["Familia_CTU13"].value_counts().to_string())
    print(f"\nOrigen de las filas BENIGN:")
    print(df_augmented[df_augmented["Label"] == "BENIGN"]["Familia_CTU13"].value_counts().to_string())

    df_augmented.to_csv(OUT_TRAIN_PATH, index=False)
    print(f"\n[+] Guardado dataset de entrenamiento: {OUT_TRAIN_PATH}")

    print(f"\n{'='*70}")
    print("  HOLDOUT DE EVALUACION v6 (nunca visto en entrenamiento)")
    print(f"{'='*70}")
    print(f"Total: {len(new_eval):,} filas")
    print(new_eval.groupby(["Familia_CTU13", "Label"]).size().to_string())

    new_eval.to_csv(OUT_EVAL_PATH, index=False)
    print(f"\n[+] Guardado holdout de evaluacion: {OUT_EVAL_PATH}")


if __name__ == "__main__":
    main()
