"""
Verify that every number the README claims comes from a recorded eval run.

The README marks its measured sections with

    <!-- numbers:start -->  ...  <!-- numbers:end -->

Every numeric token inside those blocks must appear somewhere in
results/raw/*.json, at one of the precisions the report renders. Structural
numbers — ports, config values, counts — are whitelisted explicitly below, so
adding a new claim means either measuring it or declaring it.

    python scripts/check_readme_numbers.py

Exit code 1 if any claim is unverifiable. Wire it into CI and the README
cannot drift away from the evidence.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
READMES = ["README.md", "README.ru.md"]

BLOCK = re.compile(r"<!--\s*numbers:start\s*-->(.*?)<!--\s*numbers:end\s*-->", re.S)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# Numbers that describe the setup rather than a measurement. Each one must be
# traceable to config/default.yaml, docker-compose.yml or eval/cases.yaml.
STRUCTURAL = {
    # config/default.yaml
    "800", "120", "20", "5", "60", "6000", "0.02", "0.15", "0.60", "60.0",
    # ports and versions
    "6333", "4840", "502", "3000", "8000", "11434", "1.12.4", "16", "3.12",
    # eval set and corpus
    "13", "11", "2", "4", "1", "0", "3", "51",
}


def collect_values() -> set[str]:
    """Every number recorded by an eval run, at the precisions we render."""
    values: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            number = float(node)
            for precision in (0, 1, 2, 3, 4, 6):
                values.add(f"{number:.{precision}f}")
            values.add(str(node))
            # Percentages, for prose that says "drops 9.1%"
            for precision in (0, 1, 2):
                values.add(f"{number * 100:.{precision}f}")

    for path in sorted(RAW_DIR.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return values


def check(path: Path, allowed: set[str], required: bool) -> list[str]:
    if not path.exists():
        # README.md must exist; a translation is optional until it is written.
        return [f"{path.name}: file not found"] if required else []
    text = path.read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    if not blocks:
        return [f"{path.name}: no <!-- numbers:start --> block found"]

    problems: list[str] = []
    for block in blocks:
        # Skip fenced code: commands and config samples are not claims.
        without_code = re.sub(r"```.*?```", "", block, flags=re.S)
        # Skip the placeholder rows of an unfilled table.
        for token in NUMBER.findall(without_code):
            normalised = token.replace(",", ".")
            if normalised in STRUCTURAL or normalised in allowed:
                continue
            problems.append(f"{path.name}: {token!r} is not in results/raw and is not structural")
    return problems


def main() -> int:
    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.json")):
        print(f"FAIL  no eval runs in {RAW_DIR.relative_to(ROOT)} — run `python -m eval.run` first")
        return 1

    allowed = collect_values()
    problems: list[str] = []
    for name in READMES:
        problems.extend(check(ROOT / name, allowed, required=(name == "README.md")))

    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        print(f"\n{len(problems)} unverifiable claim(s).")
        return 1

    runs = sorted(p.stem for p in RAW_DIR.glob("*.json"))
    present = [n for n in READMES if (ROOT / n).exists()]
    print(f"PASS  every number in {', '.join(present)} traces to results/raw ({', '.join(runs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
