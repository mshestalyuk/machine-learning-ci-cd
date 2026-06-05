import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, FEATURE_LABELS, TARGET
from generate_dataset import generate

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"


def test_feature_set_is_well_formed():
    assert len(FEATURE_COLUMNS) == 15
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS), "powtórzone nazwy cech"
    assert TARGET == "introduced_defect"
    assert TARGET not in FEATURE_COLUMNS, "etykieta nie może być cechą"


def test_every_feature_has_human_label():
    """build_report używa FEATURE_LABELS — każda cecha musi mieć etykietę."""
    missing = [c for c in FEATURE_COLUMNS if c not in FEATURE_LABELS]
    assert not missing, f"brak czytelnych etykiet dla: {missing}"


def test_generated_dataset_matches_feature_contract():
    """Generator danych musi produkować dokładnie cechy + etykietę używane w treningu."""
    df = generate(n=64, seed=0)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"generator nie tworzy cechy: {col}"
    assert TARGET in df.columns
    # train_model robi df[FEATURE_COLUMNS] — sprawdzamy, że ta selekcja się powiedzie.
    assert list(df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_baseline_covers_all_features():
    """explain() iteruje po baseline; brak cechy = cicho pominięte wyjaśnienie."""
    baseline = json.loads((MODEL_DIR / "baseline.json").read_text())
    missing = [c for c in FEATURE_COLUMNS if c not in baseline]
    assert not missing, f"baseline.json nie pokrywa cech: {missing}"


def test_committed_model_expects_exactly_the_feature_columns():
    """Zapisany model musi oczekiwać tylu cech, ile definiuje kontrakt."""
    model = joblib.load(MODEL_DIR / "risk_model.pkl")
    assert getattr(model, "n_features_in_", len(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_model_scores_risky_change_higher_than_safe_one():
    """Sanity zachowania modelu: duża, rozproszona zmiana bez testów, poza godzinami
    pracy i od niedoświadczonego autora powinna dostać wyższy score niż mała,
    przetestowana zmiana doświadczonego autora."""
    model = joblib.load(MODEL_DIR / "risk_model.pkl")

    risky = {
        "lines_added": 800, "lines_deleted": 200, "total_churn": 1000,
        "files_changed": 12, "num_directories": 6, "entropy": 0.9,
        "num_commits": 8, "author_experience": 2, "author_recent_changes": 15,
        "test_files_changed": 0, "test_ratio": 0.0, "is_fix": 1,
        "hour_of_day": 3, "day_of_week": 6, "off_hours": 1,
    }
    safe = {
        "lines_added": 20, "lines_deleted": 5, "total_churn": 25,
        "files_changed": 2, "num_directories": 1, "entropy": 0.1,
        "num_commits": 1, "author_experience": 300, "author_recent_changes": 2,
        "test_files_changed": 1, "test_ratio": 0.5, "is_fix": 0,
        "hour_of_day": 11, "day_of_week": 2, "off_hours": 0,
    }
    X_risky = pd.DataFrame([{k: risky[k] for k in FEATURE_COLUMNS}])
    X_safe = pd.DataFrame([{k: safe[k] for k in FEATURE_COLUMNS}])

    score_risky = float(model.predict_proba(X_risky)[:, 1][0])
    score_safe = float(model.predict_proba(X_safe)[:, 1][0])

    assert score_risky > score_safe