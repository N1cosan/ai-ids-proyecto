"""
prepare_ctu13_csv.py

Toma el CSV crudo generado por CICFlowMeter sobre el .pcap de CTU-13
y lo transforma para que tenga EXACTAMENTE las mismas 78 columnas de
features (mismo nombre, mismo orden) que espera rf_model_robust_v4_round3.

Dos casos especiales manejados aqui, no triviales de hacer a mano:

1. Naming: CICFlowMeter (version nueva) usa nombres ligeramente
   distintos a los del CIC-IDS2017 original con el que se entreno el
   modelo (ej. 'Total Fwd Packet' vs 'Total Fwd Packets', 'Dst Port'
   vs 'Destination Port', 'CWR Flag Count' vs 'CWE Flag Count' -- esta
   ultima es una inconsistencia historica del dataset original, no un
   error nuestro).

2. Columna duplicada: el CIC-IDS2017 original tenia 'Fwd Header Length'
   repetida dos veces en el CSV (bug conocido del dataset). Al leerlo
   con pandas, la segunda copia se renombro automaticamente a
   'Fwd Header Length.1'. El modelo aprendio a esperar AMBAS columnas
   (mismo valor duplicado). El CICFlowMeter nuevo ya no tiene ese bug
   -- solo trae la columna una vez -- asi que la duplicamos aqui
   manualmente para replicar la estructura exacta de entrenamiento.

Se conservan ademas Src IP, Dst IP, Src Port, Dst Port y Timestamp
(sin renombrar aparte) en el CSV de salida, para poder hacer el
etiquetado Bot/BENIGN por IP en el siguiente paso del pipeline.

Uso:
    python prepare_ctu13_csv.py <csv_entrada_cicflowmeter> <csv_salida>
"""

import sys
from pathlib import Path
import pandas as pd

# Mapeo: nombre en el CSV de CICFlowMeter -> nombre esperado por el modelo
RENAME_MAP = {
    "Dst Port": "Destination Port",
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

# Los 78 features exactos que espera el modelo, en el orden guardado
# en feature_columns_robust_v4_round3.joblib
MODEL_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Columnas de metadata que conservamos aparte (no son features del modelo,
# pero las necesitamos para el etiquetado por IP en el siguiente paso)
METADATA_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]


def main():
    if len(sys.argv) != 3:
        print("Uso: python prepare_ctu13_csv.py <csv_entrada> <csv_salida>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f"[*] Leyendo: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[+] {len(df):,} filas, {len(df.columns)} columnas originales")

    # 1. Renombrar segun el mapeo
    df = df.rename(columns=RENAME_MAP)

    # 2. Duplicar Fwd Header Length -> Fwd Header Length.1
    if "Fwd Header Length" not in df.columns:
        print("[ERROR] No se encontro 'Fwd Header Length' tras el renombrado. Abortando.")
        sys.exit(1)
    df["Fwd Header Length.1"] = df["Fwd Header Length"]

    # 3. Verificar que las 78 columnas del modelo ya existen
    faltantes = [c for c in MODEL_FEATURES if c not in df.columns]
    if faltantes:
        print(f"[ERROR] Siguen faltando columnas tras el mapeo: {faltantes}")
        sys.exit(1)
    print("[+] Las 78 columnas del modelo estan presentes tras el renombrado.")

    # 4. Armar el CSV de salida: metadata (para etiquetado despues) + las 78 features, en orden
    metadata_presentes = [c for c in METADATA_COLS if c in df.columns]
    label_presente = ["Label"] if "Label" in df.columns else []

    out_cols = metadata_presentes + MODEL_FEATURES + label_presente
    df_out = df[out_cols]

    df_out.to_csv(out_path, index=False)
    print(f"[+] Guardado: {out_path}")
    print(f"    {len(df_out):,} filas, {len(df_out.columns)} columnas "
          f"({len(metadata_presentes)} metadata + {len(MODEL_FEATURES)} features"
          f"{' + Label' if label_presente else ''})")


if __name__ == "__main__":
    main()
