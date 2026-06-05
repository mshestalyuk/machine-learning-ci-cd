import os
import subprocess
from datetime import datetime

import pytest

from extract_features import extract  # dostępne dzięki pythonpath = ["src"]


def _git(args, cwd, env=None):
    """Uruchom komendę git w danym katalogu z deterministyczną tożsamością autora."""
    e = os.environ.copy()
    e.update(
        {
            "GIT_AUTHOR_NAME": "Tester",
            "GIT_AUTHOR_EMAIL": "tester@example.com",
            "GIT_COMMITTER_NAME": "Tester",
            "GIT_COMMITTER_EMAIL": "tester@example.com",
        }
    )
    if env:
        e.update(env)
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=e,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _rev(cwd, ref="HEAD"):
    return subprocess.check_output(
        ["git", "rev-parse", ref], cwd=cwd, text=True
    ).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Świeże repo git z jednym commitem bazowym; CWD ustawione na to repo."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(["init", "-q", "-b", "main"], r)
    (r / "README.md").write_text("base\n")
    _git(["add", "."], r)
    _git(["commit", "-q", "-m", "base"], r)
    monkeypatch.chdir(r)
    return r


def _commit_files(repo_dir, files: dict, message: str, date: str | None = None):
    for name, content in files.items():
        (repo_dir / name).write_text(content)
    _git(["add", "."], repo_dir)
    env = None
    if date is not None:
        env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    _git(["commit", "-q", "-m", message], repo_dir, env=env)


def test_entropy_even_split_is_one(repo):
    """Dwa pliki z równym churnem -> entropia znormalizowana = 1.0."""
    base = _rev(repo)
    ten_lines = "".join(f"line{i}\n" for i in range(10))
    _commit_files(repo, {"a.py": ten_lines, "b.py": ten_lines}, "two files")
    feats = extract(base, _rev(repo))

    assert feats["files_changed"] == 2
    assert feats["lines_added"] == 20
    assert feats["entropy"] == pytest.approx(1.0, abs=0.01)


def test_entropy_single_file_is_zero(repo):
    """Zmiana w jednym pliku -> entropia = 0 (brak rozproszenia)."""
    base = _rev(repo)
    _commit_files(repo, {"only.py": "x = 1\n"}, "single file")
    feats = extract(base, _rev(repo))

    assert feats["files_changed"] == 1
    assert feats["entropy"] == 0.0


def test_test_ratio_and_detection(repo):
    """Plik testowy rozpoznany; test_ratio = liczba testów / wszystkie pliki."""
    base = _rev(repo)
    _commit_files(
        repo,
        {"module.py": "x = 1\n", "test_module.py": "def test_x():\n    assert True\n"},
        "code + test",
    )
    feats = extract(base, _rev(repo))

    assert feats["files_changed"] == 2
    assert feats["test_files_changed"] == 1
    assert feats["test_ratio"] == pytest.approx(0.5)


def test_is_fix_detected_from_message(repo):
    """Słowo kluczowe 'fix' w wiadomości commita ustawia is_fix = 1."""
    base = _rev(repo)
    _commit_files(repo, {"patch.py": "y = 2\n"}, "fix: correct regression in patch")
    feats = extract(base, _rev(repo))

    assert feats["is_fix"] == 1


def test_is_fix_zero_for_feature_message(repo):
    """Zwykła zmiana funkcjonalna -> is_fix = 0."""
    base = _rev(repo)
    _commit_files(repo, {"feature.py": "z = 3\n"}, "add new feature module")
    feats = extract(base, _rev(repo))

    assert feats["is_fix"] == 0


def _commit_at_local(repo_dir, files, message, local_dt: datetime):
    """Zacommituj ze znaczkiem czasu odpowiadającym podanej dacie w czasie LOKALNYM.

    extract_features liczy hour/day przez datetime.fromtimestamp (czas lokalny),
    więc budujemy epoch z naiwnej daty lokalnej (local -> epoch). Dzięki rundzie
    'local -> epoch -> local' test jest deterministyczny w KAŻDEJ strefie czasowej
    i nie wymaga time.tzset() (którego nie ma na Windowsie).
    """
    epoch = int(local_dt.timestamp())  # interpretacja w lokalnej strefie czasowej
    _commit_files(repo_dir, files, message, date=f"{epoch} +0000")


def test_off_hours_weekend(repo):
    """Commit w niedzielę o 03:00 czasu lokalnego -> off_hours = 1."""
    base = _rev(repo)
    # 2024-01-07 to niedziela; 03:00 to poza godzinami pracy
    _commit_at_local(repo, {"night.py": "w = 4\n"}, "weekend change",
                     datetime(2024, 1, 7, 3, 0))
    feats = extract(base, _rev(repo))

    assert feats["hour_of_day"] == 3
    assert feats["day_of_week"] == 6  # niedziela (poniedziałek=0)
    assert feats["off_hours"] == 1


def test_business_hours_weekday(repo):
    """Commit w środę o 11:00 czasu lokalnego -> off_hours = 0."""
    base = _rev(repo)
    # 2024-01-10 to środa; 11:00 to godziny pracy
    _commit_at_local(repo, {"day.py": "v = 5\n"}, "daytime change",
                     datetime(2024, 1, 10, 11, 0))
    feats = extract(base, _rev(repo))

    assert feats["hour_of_day"] == 11
    assert feats["day_of_week"] == 2  # środa
    assert feats["off_hours"] == 0


def test_off_hours_rule_is_internally_consistent(repo):
    """off_hours musi wynikać z hour_of_day/day_of_week wg reguły z kodu —
    działa niezależnie od strefy czasowej maszyny."""
    base = _rev(repo)
    _commit_files(repo, {"x.py": "q = 1\n"}, "any change")
    feats = extract(base, _rev(repo))

    h, d = feats["hour_of_day"], feats["day_of_week"]
    expected = 1 if (h < 8 or h > 18 or d >= 5) else 0
    assert feats["off_hours"] == expected