import joblib
import pandas as pd
import numpy as np


def test_evasion_all_v2():
    # 1. Cargar modelo, encoder y columnas
    try:
        model = joblib.load('models/rf_model_v1.joblib')
        le = joblib.load('models/label_encoder_v1.joblib')
        feature_cols = joblib.load('models/feature_columns_v1.joblib')
        print("[+] Modelo, encoder y columnas cargados correctamente.\n")
    except Exception as e:
        print(f"[-] Error cargando archivos del modelo: {e}")
        return

    # 2. Cargar dataset
    try:
        df = pd.read_csv('data/processed/monday_wednesday_clean.csv')
        print(f"[+] Dataset cargado ({len(df)} filas)\n")
    except Exception as e:
        print(f"[-] Error cargando dataset: {e}")
        return

    # ============================================
    # PERFILES DE ATAQUE (adaptados a cada técnica)
    # ============================================
    perfiles = {
        'DoS Hulk': {
            'descripcion': 'Reduce tamaños de paquetes y varianza (ataque de volumen)',
            'cambios': {
                'Bwd Packet Length Max': 200.0,
                'Total Length of Bwd Packets': 500.0,
                'Packet Length Variance': 8000.0,
                'Bwd Packet Length Mean': 120.0,
                'Bwd Packet Length Std': 40.0,
                'Max Packet Length': 350.0,
                'Average Packet Size': 180.0,
                'Packet Length Mean': 140.0,
                'Packet Length Std': 60.0,
            }
        },
        'DoS GoldenEye': {
            'descripcion': 'Reduce tamaños y estandariza longitudes de paquetes',
            'cambios': {
                'Bwd Packet Length Max': 250.0,
                'Total Length of Bwd Packets': 800.0,
                'Bwd Packet Length Std': 60.0,
                'Max Packet Length': 400.0,
                'Packet Length Variance': 12000.0,
                'Average Packet Size': 200.0,
            }
        },
        'DoS Slowhttptest': {
            'descripcion': 'Suaviza tiempos de llegada (IAT) y reduce picos de tamaño',
            'cambios': {
                'Flow IAT Mean': 80000.0,
                'Fwd IAT Mean': 70000.0,
                'Bwd IAT Mean': 70000.0,
                'Flow IAT Std': 15000.0,
                'Bwd Packet Length Max': 300.0,
                'Max Packet Length': 450.0,
                'Packet Length Variance': 15000.0,
            }
        },
        'DoS slowloris': {
            'descripcion': 'Normaliza intervalos de tiempo (ataque de conexiones lentas)',
            'cambios': {
                'Flow IAT Mean': 60000.0,
                'Flow IAT Max': 120000.0,
                'Fwd IAT Mean': 55000.0,
                'Bwd IAT Mean': 55000.0,
                'Flow Duration': 3000000.0,
                'Active Mean': 50000.0,
                'Idle Mean': 100000.0,
                'Bwd Packet Length Max': 250.0,
            }
        }
    }

    resultados = []

    for ataque, perfil in perfiles.items():
        print("=" * 65)
        print(f" Ataque: {ataque}")
        print(f" Perfil: {perfil['descripcion']}")
        print("=" * 65)

        # Buscar fila real del ataque
        mask = df['Label'] == ataque
        if mask.sum() == 0:
            mask = df['Label'].astype(str).str.contains(ataque.replace('DoS ', ''), case=False, na=False)

        if mask.sum() == 0:
            print(f"[-] No se encontraron filas de '{ataque}'\n")
            continue

        row = df[mask].iloc[0].copy()
        print(f"[+] Fila real seleccionada: {row['Label']}")

        X = row.drop('Label')
        X = X.reindex(feature_cols, fill_value=0)
        df_attack = pd.DataFrame([X])

        # Predicción original
        pred_num = model.predict(df_attack)[0]
        pred_name = le.inverse_transform([pred_num])[0]
        conf = model.predict_proba(df_attack)[0].max()

        print(f"\n[Original]")
        print(f"Predicción : {pred_name}")
        print(f"Confianza  : {conf*100:.2f}%")

        # Aplicar perfil de evasión específico
        df_evasion = df_attack.copy()
        for col, valor in perfil['cambios'].items():
            if col in df_evasion.columns:
                df_evasion[col] = valor

        # Predicción con disfraz
        pred_eva_num = model.predict(df_evasion)[0]
        pred_eva_name = le.inverse_transform([pred_eva_num])[0]
        conf_eva = model.predict_proba(df_evasion)[0].max()

        print(f"\n[Con disfraz adaptativo]")
        print(f"Predicción : {pred_eva_name}")
        print(f"Confianza  : {conf_eva*100:.2f}%")

        # Veredicto
        if pred_eva_name == 'BENIGN' and pred_name != 'BENIGN':
            print("\n[!!!] ÉXITO → El ataque EVADIÓ el modelo")
            resultado = "VULNERABLE"
        elif pred_name == 'BENIGN':
            print("\n[?] El modelo no detectó el ataque original")
            resultado = "NO DETECTADO"
        else:
            print("\n[-] El modelo RESISTIÓ la evasión")
            resultado = "RESISTENTE"

        resultados.append({
            'Ataque': ataque,
            'Original': pred_name,
            'Conf_Orig': f"{conf*100:.1f}%",
            'Evasión': pred_eva_name,
            'Conf_Eva': f"{conf_eva*100:.1f}%",
            'Resultado': resultado
        })
        print()

    # Resumen final
    print("\n" + "=" * 65)
    print(" RESUMEN FINAL - EVASIÓN CON PERFILES ADAPTATIVOS")
    print("=" * 65)
    resumen = pd.DataFrame(resultados)
    print(resumen.to_string(index=False))
    print()


if __name__ == "__main__":
    test_evasion_all_v2()