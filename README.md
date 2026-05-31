# ML Risk Assessment dla CI/CD

Automatyczna ocena ryzyka zmiany (pull requesta) z użyciem modelu uczenia
maszynowego, zintegrowana z **GitHub Actions**. Dla każdego PR model szacuje
prawdopodobieństwo, że zmiana wprowadzi defekt/awarię, wystawia poziom ryzyka
(`LOW` / `MEDIUM` / `HIGH`), komentuje PR z listą czynników ryzyka i — opcjonalnie
— blokuje merge.

Podejście opiera się na linii badań *Just-In-Time (JIT) defect prediction*
(Kamei i in., 2013): ryzyko zmiany przewidujemy z metryk samej zmiany, a nie
z analizy całego repozytorium.

## Jak to działa

```
pull_request ──► extract_features.py ──► features.json
                       │ (git diff + git log)
                       ▼
                 score_risk.py ──► score 0..1 + poziom + czynniki
                       │ (model/risk_model.pkl)
                       ▼
        komentarz w PR  +  podsumowanie joba  +  (opcjonalna bramka)
```

### Cechy wejściowe (15)

| Grupa | Cechy |
|---|---|
| Rozmiar | `lines_added`, `lines_deleted`, `total_churn`, `files_changed`, `num_directories` |
| Rozproszenie | `entropy` (rozkład zmian po plikach, 0..1) |
| Historia | `num_commits`, `author_experience`, `author_recent_changes` |
| Jakość | `test_files_changed`, `test_ratio`, `is_fix` |
| Kontekst | `hour_of_day`, `day_of_week`, `off_hours` |

### Model

- `HistGradientBoostingClassifier` (drzewa wzmacniane gradientowo)
- kalibracja prawdopodobieństw (`CalibratedClassifierCV`, sigmoid) — score jest
  interpretowalny jako szacunek prawdopodobieństwa defektu
- wyjaśnialność: lokalna **ablacja jednej cechy** (podmiana na medianę
  treningową) wskazuje, które cechy najmocniej podbiły ryzyko danego PR

## Szybki start (lokalnie)

```bash
pip install -r requirements.txt

# 1. zbuduj dane i wytrenuj model
python src/generate_dataset.py --n 10000 --out data/changes.csv
python src/train_model.py --data data/changes.csv --out model

# 2. oceń bieżącą gałąź względem main
python src/extract_features.py --base main --head HEAD --out features.json
python src/score_risk.py --features features.json \
       --model model/risk_model.pkl --baseline model/baseline.json
```

## Konfiguracja (zmienne środowiskowe)

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `RISK_THRESHOLD_MEDIUM` | `0.30` | próg poziomu MEDIUM |
| `RISK_THRESHOLD_HIGH` | `0.60` | próg poziomu HIGH |
| `RISK_FAIL_ON` | `""` | `HIGH` lub `MEDIUM` → bramka blokuje job; puste → tylko raport |

Ustawia się je w `.github/workflows/risk-assessment.yml` (sekcja `env`).

## Workflowy

- **`risk-assessment.yml`** — uruchamiany na każdym PR; ocenia ryzyko, komentuje,
  opcjonalnie blokuje.
- **`model-retrain.yml`** — retrening (ręczny / cotygodniowy), commituje nowy
  artefakt modelu.

## Wdrożenie na własnych danych

Generator syntetyczny (`generate_dataset.py`) służy do startu i demonstracji.
W produkcji zastąp go ekstraktorem historii własnego repo:

1. Przejdź historię (`git log`) i policz te same cechy dla każdego commita scalonego.
2. Etykietę `introduced_defect` ustal przez powiązanie commita z późniejszą
   awarią — np. metodą **SZZ** (commit wskazany przez `git blame` przy poprawce),
   czerwonym buildem CI po merge'u, albo incydentem z systemu zgłoszeń.
3. Wytrenuj na realnych danych — reszta pipeline'u pozostaje bez zmian.

## Ograniczenia

- Model uczony na danych syntetycznych ma walor demonstracyjny; metryki
  (ROC-AUC ≈ 0.74) odzwierciedlają wbudowaną zależność, nie realne repo.
- Ocena bazuje na metrykach zmiany, nie na semantyce kodu — to sygnał
  wspomagający review, a nie jego zamiennik.
```