"""
Generator syntetycznego zbioru danych do predykcji ryzyka zmiany (change-risk).

Cechy odzwierciedlają literaturę JIT defect prediction (Kamei i in., 2013):
rozmiar zmiany, rozproszenie (entropia), historia/doświadczenie autora,
pokrycie testami oraz kontekst czasowy wdrożenia.

Etykieta `introduced_defect` (0/1) jest generowana z modelu logitowego,
w którym prawdopodobieństwo defektu rośnie wraz z ryzykownymi cechami.
Dzięki temu zależności są realistyczne i wyuczalne, ale nietrywialne.
"""

import argparse
import json

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Cechy rozmiaru zmiany ---
    lines_added = rng.gamma(shape=2.0, scale=40.0, size=n).astype(int)
    lines_deleted = rng.gamma(shape=1.5, scale=20.0, size=n).astype(int)
    files_changed = (1 + rng.poisson(lam=3.0, size=n)).astype(int)
    num_directories = np.minimum(
        files_changed, 1 + rng.poisson(lam=1.5, size=n)
    ).astype(int)

    # Entropia zmiany: jak bardzo zmiana jest rozproszona po katalogach.
    # Wyższa entropia = trudniejsza do przejrzenia = bardziej ryzykowna.
    entropy = (num_directories / files_changed) * rng.uniform(0.5, 1.0, size=n)

    # --- Cechy historii / autora ---
    num_commits = (1 + rng.poisson(lam=2.0, size=n)).astype(int)
    author_experience = rng.gamma(shape=2.0, scale=60.0, size=n).astype(int)
    author_recent_changes = rng.poisson(lam=5.0, size=n).astype(int)

    # --- Cechy jakościowe ---
    test_files_changed = rng.binomial(files_changed, p=0.25)
    test_ratio = test_files_changed / files_changed

    # Słowa kluczowe sugerujące poprawkę błędu -> historycznie bardziej ryzykowne
    is_fix = rng.binomial(1, p=0.30, size=n)

    # --- Kontekst czasowy ---
    hour_of_day = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)  # 0=pon ... 6=niedz
    off_hours = (((hour_of_day < 8) | (hour_of_day > 18)) | (day_of_week >= 5)).astype(int)

    total_churn = lines_added + lines_deleted

    # --- Logit ryzyka: suma wkładów ryzykownych czynników ---
    logit = (
        -3.0
        + 0.0024 * total_churn                  # więcej zmian -> wyżej
        + 0.14 * files_changed                  # rozproszenie po plikach
        + 2.4 * entropy                         # rozproszenie po katalogach
        + 1.2 * off_hours                       # wdrożenia poza godzinami
        + 0.7 * is_fix                          # poprawki bywają ryzykowne
        - 2.2 * test_ratio                      # testy obniżają ryzyko
        - 0.006 * author_experience             # doświadczenie obniża ryzyko
        + 0.07 * author_recent_changes          # zmęczenie / wiele zmian naraz
        + 0.10 * num_commits                    # PR z wieloma commitami
    )
    logit += rng.normal(0, 0.30, size=n)        # szum nieredukowalny

    prob = _sigmoid(logit)
    introduced_defect = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "total_churn": total_churn,
            "files_changed": files_changed,
            "num_directories": num_directories,
            "entropy": entropy.round(4),
            "num_commits": num_commits,
            "author_experience": author_experience,
            "author_recent_changes": author_recent_changes,
            "test_files_changed": test_files_changed,
            "test_ratio": test_ratio.round(4),
            "is_fix": is_fix,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "off_hours": off_hours,
            "introduced_defect": introduced_defect,
        }
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/changes.csv")
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    rate = df["introduced_defect"].mean()
    print(json.dumps({"rows": len(df), "defect_rate": round(float(rate), 4)}, indent=2))
    print(f"Zapisano: {args.out}")


if __name__ == "__main__":
    main()
