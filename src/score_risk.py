"""
Scoring ryzyka zmiany na podstawie wyuczonego modelu.

Wejście:  features.json (z extract_features.py)
Model:    model/risk_model.pkl + model/baseline.json
Wyjście:
  - score (0..1) i poziom ryzyka LOW/MEDIUM/HIGH
  - top czynniki podbijające ryzyko (lokalna ablacja względem mediany)
  - raport Markdown (do komentarza w PR / GITHUB_STEP_SUMMARY)
  - wyjścia GitHub Actions (risk_score, risk_level, gate_failed)

Progi konfigurowalne przez zmienne środowiskowe:
  RISK_THRESHOLD_MEDIUM (domyślnie 0.30)
  RISK_THRESHOLD_HIGH   (domyślnie 0.60)
  RISK_FAIL_ON          (np. "HIGH" -> job pada przy HIGH; pusty = nie blokuj)
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, FEATURE_LABELS


def _level(score: float, t_med: float, t_high: float) -> str:
    if score >= t_high:
        return "HIGH"
    if score >= t_med:
        return "MEDIUM"
    return "LOW"


def _emoji(level: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[level]


def explain(model, feats: dict, baseline: dict, score: float, top_k: int = 4):
    """
    Lokalne wyjaśnienie metodą ablacji jednej cechy: dla każdej cechy
    podmieniamy jej wartość na medianę z treningu i mierzymy spadek score.
    Dodatni wkład = cecha podbija ryzyko tej zmiany.
    """
    row = pd.DataFrame([{k: feats[k] for k in FEATURE_COLUMNS}])
    contribs = []
    for col in FEATURE_COLUMNS:
        if col not in baseline:
            continue
        perturbed = row.copy()
        perturbed.at[0, col] = baseline[col]
        base_score = float(model.predict_proba(perturbed)[:, 1][0])
        contrib = score - base_score  # >0: cecha zwiększa ryzyko
        contribs.append((col, contrib, feats[col]))
    contribs.sort(key=lambda x: x[1], reverse=True)
    return [c for c in contribs if c[1] > 0.005][:top_k]


def build_report(score, level, top_factors, meta, thresholds) -> str:
    pct = round(score * 100, 1)
    lines = [
        f"## {_emoji(level)} Ocena ryzyka zmiany: **{level}** ({pct}%)",
        "",
        f"Oszacowane prawdopodobieństwo, że zmiana wprowadzi defekt: **{pct}%**.",
        "",
        "| Czynnik podbijający ryzyko | Wartość | Wkład |",
        "|---|---|---|",
    ]
    if top_factors:
        for col, contrib, val in top_factors:
            label = FEATURE_LABELS.get(col, col)
            lines.append(f"| {label} | `{val}` | +{round(contrib * 100, 1)} pp |")
    else:
        lines.append("| _Brak wyraźnych czynników ryzyka_ | – | – |")
    lines += [
        "",
        f"<sub>Progi: MEDIUM ≥ {thresholds[0]:.2f}, HIGH ≥ {thresholds[1]:.2f} · "
        f"zakres `{meta.get('base','?')[:9]}..{meta.get('head','?')[:9]}`</sub>",
    ]
    return "\n".join(lines)


def set_gha_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"{name}={value}\n")


def append_step_summary(md: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(md + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.json")
    ap.add_argument("--model", default="model/risk_model.pkl")
    ap.add_argument("--baseline", default="model/baseline.json")
    ap.add_argument("--report", default="risk_report.md")
    args = ap.parse_args()

    t_med = float(os.environ.get("RISK_THRESHOLD_MEDIUM", "0.30"))
    t_high = float(os.environ.get("RISK_THRESHOLD_HIGH", "0.60"))
    fail_on = os.environ.get("RISK_FAIL_ON", "").strip().upper()

    with open(args.features) as f:
        feats = json.load(f)
    meta = feats.pop("_meta", {})
    with open(args.baseline) as f:
        baseline = json.load(f)
    model = joblib.load(args.model)

    row = pd.DataFrame([{k: feats[k] for k in FEATURE_COLUMNS}])
    score = float(model.predict_proba(row)[:, 1][0])
    level = _level(score, t_med, t_high)
    top_factors = explain(model, feats, baseline, score)

    report = build_report(score, level, top_factors, meta, (t_med, t_high))
    with open(args.report, "w") as f:
        f.write(report)

    # Konsola
    print(report)
    print()

    # Wyjścia GitHub Actions
    set_gha_output("risk_score", f"{score:.4f}")
    set_gha_output("risk_level", level)
    append_step_summary(report)

    gate_failed = bool(fail_on) and (
        (fail_on == "HIGH" and level == "HIGH")
        or (fail_on == "MEDIUM" and level in ("MEDIUM", "HIGH"))
    )
    set_gha_output("gate_failed", "true" if gate_failed else "false")

    if gate_failed:
        print(f"::error::Bramka ryzyka: poziom {level} >= {fail_on} — blokuję.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
