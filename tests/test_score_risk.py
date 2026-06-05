import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS  # pythonpath = ["src"]
from score_risk import _level, build_report, explain


def test_level_thresholds():
    assert _level(0.10, 0.30, 0.60) == "LOW"
    assert _level(0.29, 0.30, 0.60) == "LOW"
    assert _level(0.30, 0.30, 0.60) == "MEDIUM"  # próg włącznie
    assert _level(0.45, 0.30, 0.60) == "MEDIUM"
    assert _level(0.60, 0.30, 0.60) == "HIGH"  # próg włącznie
    assert _level(0.95, 0.30, 0.60) == "HIGH"


def test_build_report_contains_level_factors_and_percent():
    meta = {"base": "abcdef123456", "head": "123456abcdef"}
    top = [("lines_added", 0.12, 804), ("is_fix", 0.05, 1)]
    md = build_report(0.956, "HIGH", top, meta, (0.30, 0.60))

    assert "HIGH" in md
    assert "95.6%" in md
    assert "Dodane linie" in md  # czytelna etykieta z FEATURE_LABELS
    assert "+12.0 pp" in md  # wkład czynnika sformatowany jako punkty proc.
    assert "abcdef123" in md  # skrócony zakres base..head


def test_build_report_without_factors():
    md = build_report(0.10, "LOW", [], {}, (0.30, 0.60))

    assert "LOW" in md
    assert "Brak" in md  # linia "Brak wyraźnych czynników ryzyka"


class DummyModel:
    """Atrapa modelu: prawdopodobieństwo zależy wyłącznie od `lines_added`.

    Dzięki temu ablacja `lines_added` (podmiana na niższą medianę) musi obniżyć
    score, czyli dać dodatni wkład tej cechy.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        la = X["lines_added"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-(la - 100.0) / 50.0))
        return np.column_stack([1.0 - p, p])


def test_explain_identifies_dominant_feature():
    feats = {k: 0 for k in FEATURE_COLUMNS}
    feats["lines_added"] = 800  # bardzo duża zmiana -> wysokie ryzyko
    baseline = {k: 0 for k in FEATURE_COLUMNS}
    baseline["lines_added"] = 50  # mediana znacznie niższa

    model = DummyModel()
    score = float(model.predict_proba(pd.DataFrame([feats]))[:, 1][0])

    top = explain(model, feats, baseline, score, top_k=4)

    assert top, "oczekiwano co najmniej jednego czynnika podbijającego ryzyko"
    assert top[0][0] == "lines_added"  # czynnik dominujący
    assert top[0][1] > 0  # dodatni wkład = zwiększa ryzyko
    assert top[0][2] == 800  # zwracana jest faktyczna wartość cechy


def test_explain_returns_empty_when_change_matches_baseline():
    """Gdy cechy = baseline, żaden czynnik nie podbija ryzyka (wkłady ~0)."""
    feats = {k: 0 for k in FEATURE_COLUMNS}
    feats["lines_added"] = 50
    baseline = {k: 0 for k in FEATURE_COLUMNS}
    baseline["lines_added"] = 50

    model = DummyModel()
    score = float(model.predict_proba(pd.DataFrame([feats]))[:, 1][0])

    top = explain(model, feats, baseline, score, top_k=4)

    assert top == []