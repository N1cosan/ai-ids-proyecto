"""
label_ctu13.py

Etiqueta el CSV ya preparado (columnas alineadas al modelo) de un
escenario de CTU-13, usando las IPs oficiales del ground truth
publicado por Stratosphere Labs para cada escenario especifico.

Metodologia (la misma que usa Stratosphere para etiquetar sus propios
netflows, aplicada aqui a nivel de flujo de CICFlowMeter):

1. Si el flujo involucra alguna IP infectada (como Src IP o Dst IP)
   -> Bot
2. Si no, y el flujo involucra alguno de los hosts normales
   verificados -> BENIGN
3. Cualquier otro caso ("Background": trafico de origen no verificado,
   ni confirmado malicioso ni confirmado benigno) -> se EXCLUYE de la
   evaluacion/entrenamiento. Forzar este trafico a BENIGN inventaria
   ground truth que no tenemos.

FILTRO DNS: dentro de los flujos etiquetados 'Bot', se excluyen
ademas los que son puramente consultas DNS (puerto 53 en Src o Dst).
Motivo: el ground truth de CTU-13 etiqueta TODO el trafico del host
infectado como Bot, incluyendo actividad mundana como resolucion DNS
que no es en si misma la señal de C2/comportamiento malicioso. Incluir
esos flujos en el entrenamiento diluye la señal real que queremos que
el modelo aprenda. Se puede desactivar con --no-dns-filter.

Uso:
    python label_ctu13.py <csv_entrada> <csv_salida> <escenario> [--no-dns-filter]

Escenarios soportados: botnet42 (Neris), botnet52 (Rbot), botnet46 (Virut)
"""

import sys
from pathlib import Path
import pandas as pd

# Configuracion de ground truth por escenario, segun los README
# oficiales de Stratosphere (mcfp.felk.cvut.cz/publicDatasets/).
SCENARIOS = {
    "botnet42": {  # Scenario 1, Neris
        "infected_ips": {"147.32.84.165"},  # SARUMAN
        "normal_ips": {
            "147.32.84.170",  # Stribrek
            "147.32.84.164",  # Grill
            "147.32.84.134",  # Jist
            "147.32.87.36",   # CVUT-WebServer (nota oficial: menos confiable)
            "147.32.80.9",    # CVUT-DNS-Server (nota oficial: menos confiable)
            "147.32.87.11",   # MatLab-Server (nota oficial: menos confiable)
        },
    },
    "botnet52": {  # Scenario 11, Rbot
        "infected_ips": {
            "147.32.84.165",  # SARUMAN
            "147.32.84.191",  # SARUMAN1
            "147.32.84.192",  # SARUMAN2
        },
        "normal_ips": {
            "147.32.84.170",  # Stribrek
            "147.32.84.134",  # Jist
            "147.32.84.164",  # Grill
            "147.32.87.36",   # CVUT-WebServer
            "147.32.80.9",    # CVUT-DNS-Server
            "147.32.87.11",   # MatLab-Server
        },
    },
    "botnet46": {  # Scenario 5, Virut
        "infected_ips": {"147.32.84.165"},  # SARUMAN
        "normal_ips": {
            "147.32.84.170",  # Stribrek
            "147.32.84.134",  # Jist
            "147.32.84.164",  # Grill
            "147.32.87.36",   # CVUT-WebServer
            "147.32.80.9",    # CVUT-DNS-Server
            "147.32.87.11",   # MatLab-Server
        },
    },
}

DNS_PORT = "53"


def label_row(src_ip: str, dst_ip: str, infected_ips: set, normal_ips: set) -> str:
    if src_ip in infected_ips or dst_ip in infected_ips:
        return "Bot"
    if src_ip in normal_ips or dst_ip in normal_ips:
        return "BENIGN"
    return "Background"


def main():
    if len(sys.argv) < 4:
        print("Uso: python label_ctu13.py <csv_entrada> <csv_salida> <escenario> [--no-dns-filter]")
        print(f"Escenarios disponibles: {list(SCENARIOS.keys())}")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    scenario = sys.argv[3]
    apply_dns_filter = "--no-dns-filter" not in sys.argv

    if scenario not in SCENARIOS:
        print(f"[ERROR] Escenario '{scenario}' no configurado. "
              f"Disponibles: {list(SCENARIOS.keys())}")
        sys.exit(1)

    cfg = SCENARIOS[scenario]
    infected_ips = cfg["infected_ips"]
    normal_ips = cfg["normal_ips"]

    print(f"[*] Leyendo: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    print(f"[+] {len(df):,} filas cargadas")

    if "Src IP" not in df.columns or "Dst IP" not in df.columns:
        print("[ERROR] El CSV no tiene columnas 'Src IP' / 'Dst IP'. "
              "Revisa que uses el CSV generado por prepare_ctu13_csv.py.")
        sys.exit(1)

    print(f"[*] Etiquetando por IP (escenario: {scenario}, infectadas: {infected_ips})...")
    df["Label_CTU13"] = df.apply(
        lambda row: label_row(str(row["Src IP"]), str(row["Dst IP"]), infected_ips, normal_ips),
        axis=1,
    )

    print("\nDistribucion de etiquetas (antes de filtro DNS):")
    print(df["Label_CTU13"].value_counts())

    df_labeled = df[df["Label_CTU13"] != "Background"].copy()
    n_excluded_bg = len(df) - len(df_labeled)
    print(f"\n[+] Excluidas {n_excluded_bg:,} filas de trafico 'Background' "
          f"(sin ground truth confiable)")

    if apply_dns_filter:
        src_port_col = "Src Port" if "Src Port" in df_labeled.columns else None
        dst_port_col = "Destination Port" if "Destination Port" in df_labeled.columns else None

        is_bot = df_labeled["Label_CTU13"] == "Bot"
        is_dns = pd.Series(False, index=df_labeled.index)
        if src_port_col:
            is_dns |= (df_labeled[src_port_col].astype(str) == DNS_PORT)
        if dst_port_col:
            is_dns |= (df_labeled[dst_port_col].astype(str) == DNS_PORT)

        dns_bot_mask = is_bot & is_dns
        n_dns_excluded = dns_bot_mask.sum()
        df_labeled = df_labeled[~dns_bot_mask].copy()
        print(f"[+] Excluidos {n_dns_excluded:,} flujos 'Bot' que eran solo consultas DNS "
              f"(puerto {DNS_PORT}) -- no representan la señal de C2 real")
    else:
        print("[!] Filtro DNS desactivado (--no-dns-filter)")

    n_bot = (df_labeled["Label_CTU13"] == "Bot").sum()
    n_benign = (df_labeled["Label_CTU13"] == "BENIGN").sum()
    print(f"\n[+] Quedan {len(df_labeled):,} filas para evaluacion/entrenamiento "
          f"(Bot: {n_bot:,}, BENIGN: {n_benign:,})")

    df_labeled.to_csv(out_path, index=False)
    print(f"\n[+] Guardado: {out_path}")


if __name__ == "__main__":
    main()
