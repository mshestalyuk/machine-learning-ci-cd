"""
Trening modelu predykcji ryzyka zmiany.

- Model: HistGradientBoostingClassifier (drzewa wzmacniane gradientowo).
- Kalibracja: prawdopodobieństwa kalibrowane (sigmoid) aby score 0..1 był
  interpretowalny jako szacunek prawdopodobieństwa defektu.
- Zapis: model + mediany cech (baseline do wyjaśnień) + metryki -> model/.

Uruchom: python src/train_model.py --data data/changes.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS, TARGET


def train(data_path: str, out_dir: str, seed: int = 42) -> dict:
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=1.0,
        random_state=seed,
    )
    # Kalibracja prawdopodobieństw na walidacji krzyżowej
    model = CalibratedClassifierCV(base, method="sigmoid", cv=4)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "defect_rate": round(float(y.mean()), 4),
    }

    # Mediany cech jako baseline do lokalnych wyjaśnień (ablacja jednej cechy)
    baseline = X_train.median(numeric_only=True).to_dict()

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "risk_model.pkl"))
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/changes.csv")
    ap.add_argument("--out", default="model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    metrics = train(args.data, args.out, args.seed)
    print("Metryki ewaluacji (zbiór testowy):")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
