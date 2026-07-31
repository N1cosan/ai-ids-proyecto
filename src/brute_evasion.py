#!/usr/bin/env python3
"""
Motor de Brute-Force / Black-Box Evasion para DoS Slowhttptest
CICIDS2017 + Random Forest
"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
from copy import deepcopy


# Features más discriminativas para Slowhttptest
SLOWHTTPTEST_FEATURES = [
    "Flow Duration",
    "Flow IAT Std", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
    "Active Mean", "Active Min", "Active Max",
    "Idle Mean", "Idle Max", "Idle Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Mean", "Bwd IAT Std",
    "Average Packet Size", "Packet Length Mean", "Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
]

PERTURB_RANGES = {
    "Flow Duration":       (-0.65, 0.90),
    "Flow IAT Std":        (-0.75, 1.30),
    "Flow IAT Mean":       (-0.55, 1.00),
    "Flow IAT Max":        (-0.65, 1.10),
    "Flow IAT Min":        (-0.45, 0.70),
    "Active Mean":         (-0.70, 1.60),
    "Active Min":          (-0.55, 1.10),
    "Active Max":          (-0.60, 1.30),
    "Idle Mean":           (-0.80, 1.60),
    "Idle Max":            (-0.70, 1.40),
    "Idle Min":            (-0.50, 1.10),
    "Fwd IAT Mean":        (-0.60, 1.10),
    "Fwd IAT Std":         (-0.70, 1.30),
    "Average Packet Size": (-0.45, 0.55),
    "Packet Length Std":   (-0.55, 0.90),
    "Flow Bytes/s":        (-0.65, 0.80),
    "Flow Packets/s":      (-0.55, 0.70),
}

def get_feature_indices(feature_names, target_features):
    idx = {}
    for f in target_features:
        if f in feature_names:
            idx[f] = list(feature_names).index(f)
    return idx

def clip_realistic(x, feature_names):
    x = x.copy()
    for i, name in enumerate(feature_names):
        if any(k in str(name) for k in ["Duration", "IAT", "Active", "Idle",
                                         "Bytes", "Packets", "Length", "Size"]):
            x[i] = max(0.0, float(x[i]))
    return x

def random_perturbation(x, feature_idx, feature_names, intensity=1.0):
    x_new = x.copy()
    for fname, idx in feature_idx.items():
        if fname not in PERTURB_RANGES:
            continue
        low, high = PERTURB_RANGES[fname]
        factor = 1.0 + np.random.uniform(low * intensity, high * intensity)
        x_new[idx] = x[idx] * factor
    return clip_realistic(x_new, feature_names)

def evaluate_benign_prob(model, x, feature_names, benign_label="BENIGN"):
    """Devuelve P(BENIGN)."""
    # Convertir a DataFrame para evitar el warning de feature names
    X = pd.DataFrame([x], columns=feature_names)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if benign_label in classes:
        return float(proba[classes.index(benign_label)])
    return float(proba[0])

def hill_climb(model, x_orig, feature_idx, feature_names,
               max_iters=100, n_candidates=15, intensity=0.90):
    best_x = x_orig.copy()
    best_score = evaluate_benign_prob(model, best_x, feature_names)
    history = [(0, best_score)]

    for it in range(1, max_iters + 1):
        candidates = []
        for _ in range(n_candidates):
            cand = random_perturbation(best_x, feature_idx, feature_names, intensity)
            score = evaluate_benign_prob(model, cand, feature_names)
            candidates.append((score, cand))

        # También desde el original (exploración)
        cand_from_orig = random_perturbation(x_orig, feature_idx, feature_names, intensity * 1.25)
        score_from_orig = evaluate_benign_prob(model, cand_from_orig, feature_names)
        candidates.append((score_from_orig, cand_from_orig))

        candidates.sort(key=lambda t: t[0], reverse=True)
        top_score, top_x = candidates[0]

        if top_score > best_score:
            best_score = top_score
            best_x = top_x
            history.append((it, best_score))
            if best_score > 0.70:
                intensity *= 0.93

        if best_score >= 0.93:
            break

    return best_x, best_score, history

def evade_slowhttptest(model, sample, feature_names,
                      benign_label="BENIGN", max_iters=120, verbose=True):
    x = np.asarray(sample, dtype=float).copy()
    feature_idx = get_feature_indices(feature_names, SLOWHTTPTEST_FEATURES)

    if not feature_idx:
        raise ValueError("No se encontraron features de Slowhttptest. Revisa los nombres de columnas.")

    # Evaluación original
    # Evaluación original
# Evaluación original
    X_orig = pd.DataFrame([x], columns=feature_names)
    orig_proba = model.predict_proba(X_orig)[0]
    orig_pred = model.classes_[np.argmax(orig_proba)]
    orig_conf = float(np.max(orig_proba))
    orig_benign_p = evaluate_benign_prob(model, x, feature_names, benign_label)

    if verbose:
        print(f"[Original] Predicción : {orig_pred}")
        print(f"           Confianza  : {orig_conf:.4f}")
        print(f"           P(BENIGN)  : {orig_benign_p:.4f}")

    # Optimización
    x_adv, final_benign_p, history = hill_climb(
        model, x, feature_idx, feature_names,
        max_iters=max_iters, n_candidates=16, intensity=0.95
    )

    X_adv = pd.DataFrame([x_adv], columns=feature_names)
    adv_proba = model.predict_proba(X_adv)[0]
    adv_pred = model.classes_[np.argmax(adv_proba)]
    adv_conf = float(np.max(adv_proba))

    if verbose:
        print(f"\n[Adversario] Predicción : {adv_pred}")
        print(f"             Confianza  : {adv_conf:.4f}")
        print(f"             P(BENIGN)  : {final_benign_p:.4f}")
        print(f"             Pasos útiles: {len(history)}")

        # Top cambios
        deltas = []
        for fname, idx in feature_idx.items():
            if abs(x_adv[idx] - x[idx]) > 1e-6:
                rel = (x_adv[idx] - x[idx]) / (abs(x[idx]) + 1e-9)
                deltas.append((fname, x[idx], x_adv[idx], rel))
        deltas.sort(key=lambda t: abs(t[3]), reverse=True)

        print("\nTop cambios relativos:")
        for fname, old, new, rel in deltas[:10]:
            print(f"  {fname:28s}: {old:12.2f} → {new:12.2f}  ({rel:+.1%})")

    return {
        "x_original": x,
        "x_adversarial": x_adv,
        "original_pred": str(orig_pred),
        "original_conf": orig_conf,
        "adv_pred": str(adv_pred),
        "adv_conf": adv_conf,
        "benign_prob": final_benign_p,
        "history": history,
        "success": (str(adv_pred) == benign_label) or (final_benign_p > 0.55)
    }