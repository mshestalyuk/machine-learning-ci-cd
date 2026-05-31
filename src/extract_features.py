"""
Ekstrakcja cech z kontekstu zmiany (PR / commit) w środowisku CI.

Źródła:
- git diff <base>..<head>  -> rozmiar, pliki, katalogi, entropia, testy
- git log                  -> liczba commitów, doświadczenie autora
- commit message           -> wykrycie poprawki (fix)
- czas commita             -> godzina, dzień tygodnia, poza godzinami

W GitHub Actions bazę/head bierzemy z eventu PR. Lokalnie można podać
--base i --head ręcznie. Wynik zapisywany jest jako JSON.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime

from features import FEATURE_COLUMNS

FIX_PATTERN = re.compile(
    r"\b(fix|fixes|fixed|bug|bugfix|hotfix|patch|repair|issue|regression)\b",
    re.IGNORECASE,
)
TEST_PATTERN = re.compile(r"(test|spec|__tests__|\.test\.|\.spec\.)", re.IGNORECASE)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def _resolve_refs(base: str | None, head: str | None) -> tuple[str, str]:
    """Ustal SHA bazy i head — z argumentów lub ze środowiska GitHub Actions."""
    head = head or os.environ.get("GITHUB_SHA") or "HEAD"
    if not base:
        base_ref = os.environ.get("GITHUB_BASE_REF")  # np. "main" w PR
        if base_ref:
            base = _run(["git", "merge-base", f"origin/{base_ref}", head]) or f"origin/{base_ref}"
        else:
            base = _run(["git", "rev-parse", f"{head}~1"]) or f"{head}~1"
    return base, head


def extract(base: str, head: str) -> dict:
    # --- diff numstat: dodane/usunięte linie per plik ---
    numstat = _run(["git", "diff", "--numstat", f"{base}..{head}"])
    lines_added = lines_deleted = files_changed = test_files = 0
    dirs: set[str] = set()
    per_file_churn: list[int] = []

    for row in numstat.splitlines():
        parts = row.split("\t")
        if len(parts) != 3:
            continue
        add_s, del_s, path = parts
        add = int(add_s) if add_s.isdigit() else 0
        rem = int(del_s) if del_s.isdigit() else 0
        lines_added += add
        lines_deleted += rem
        files_changed += 1
        per_file_churn.append(add + rem)
        d = os.path.dirname(path) or "."
        dirs.add(d)
        if TEST_PATTERN.search(path):
            test_files += 1

    files_changed = max(files_changed, 1)
    num_directories = max(len(dirs), 1)
    total_churn = lines_added + lines_deleted

    # --- entropia rozproszenia churn po plikach (znormalizowana 0..1) ---
    entropy = 0.0
    if total_churn > 0 and len(per_file_churn) > 1:
        import math

        ps = [c / total_churn for c in per_file_churn if c > 0]
        h = -sum(p * math.log(p, 2) for p in ps)
        entropy = h / math.log(len(per_file_churn), 2)  # 0..1

    # --- liczba commitów w PR ---
    log_count = _run(["git", "rev-list", "--count", f"{base}..{head}"])
    num_commits = int(log_count) if log_count.isdigit() else 1

    # --- autor + doświadczenie (commity autora w historii do head) ---
    author = _run(["git", "log", "-1", "--format=%ae", head])
    exp = 0
    if author:
        exp_s = _run(["git", "rev-list", "--count", f"--author={author}", head])
        exp = int(exp_s) if exp_s.isdigit() else 0
    # ostatnie zmiany autora (ostatnie 30 dni)
    recent_s = _run(
        ["git", "rev-list", "--count", f"--author={author}", "--since=30.days", head]
    ) if author else ""
    author_recent_changes = int(recent_s) if recent_s.isdigit() else 0

    # --- czy poprawka? (wiadomości commitów w zakresie) ---
    msgs = _run(["git", "log", "--format=%s%n%b", f"{base}..{head}"])
    is_fix = 1 if FIX_PATTERN.search(msgs) else 0

    # --- kontekst czasowy commita head ---
    ts = _run(["git", "log", "-1", "--format=%ct", head])
    if ts.isdigit():
        dt = datetime.fromtimestamp(int(ts))
    else:
        dt = datetime.now()
    hour_of_day = dt.hour
    day_of_week = dt.weekday()  # 0=pon
    off_hours = 1 if (hour_of_day < 8 or hour_of_day > 18 or day_of_week >= 5) else 0

    feats = {
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "total_churn": total_churn,
        "files_changed": files_changed,
        "num_directories": num_directories,
        "entropy": round(entropy, 4),
        "num_commits": num_commits,
        "author_experience": exp,
        "author_recent_changes": author_recent_changes,
        "test_files_changed": test_files,
        "test_ratio": round(test_files / files_changed, 4),
        "is_fix": is_fix,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "off_hours": off_hours,
    }
    # gwarancja kompletności i kolejności
    return {k: feats[k] for k in FEATURE_COLUMNS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="SHA/ref bazy")
    ap.add_argument("--head", default=None, help="SHA/ref head")
    ap.add_argument("--out", default="features.json")
    args = ap.parse_args()

    base, head = _resolve_refs(args.base, args.head)
    feats = extract(base, head)
    feats["_meta"] = {"base": base, "head": head}

    with open(args.out, "w") as f:
        json.dump(feats, f, indent=2)
    print(f"Wyodrębniono cechy ({base[:9]}..{head[:9]}) -> {args.out}")
    print(json.dumps({k: v for k, v in feats.items() if k != "_meta"}, indent=2))


if __name__ == "__main__":
    main()
