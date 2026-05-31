"""Wspólna definicja cech używanych przez model ryzyka."""

# Kolejność i zestaw cech MUSZĄ być identyczne w treningu i scoringu.
FEATURE_COLUMNS = [
    "lines_added",
    "lines_deleted",
    "total_churn",
    "files_changed",
    "num_directories",
    "entropy",
    "num_commits",
    "author_experience",
    "author_recent_changes",
    "test_files_changed",
    "test_ratio",
    "is_fix",
    "hour_of_day",
    "day_of_week",
    "off_hours",
]

TARGET = "introduced_defect"

# Czytelne nazwy do raportu w PR
FEATURE_LABELS = {
    "lines_added": "Dodane linie",
    "lines_deleted": "Usunięte linie",
    "total_churn": "Łączna zmiana (churn)",
    "files_changed": "Zmienione pliki",
    "num_directories": "Liczba katalogów",
    "entropy": "Rozproszenie zmiany (entropia)",
    "num_commits": "Liczba commitów",
    "author_experience": "Doświadczenie autora",
    "author_recent_changes": "Ostatnie zmiany autora",
    "test_files_changed": "Zmienione pliki testowe",
    "test_ratio": "Udział testów w zmianie",
    "is_fix": "Poprawka błędu (fix)",
    "hour_of_day": "Godzina wdrożenia",
    "day_of_week": "Dzień tygodnia",
    "off_hours": "Poza godzinami pracy",
}
