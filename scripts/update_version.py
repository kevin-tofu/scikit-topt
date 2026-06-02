#!/usr/bin/env python3
"""
Update version strings across docs/conf.py, pyproject.toml, CITATION.cff
and optionally paper/paper.md.

Usage:
    python scripts/update_version.py --version 0.3.5
    python scripts/update_version.py --version 0.3.5 --update-paper

You may pass "v0.3.5" or "0.3.5"; the script normalizes where needed:
- Numeric form (no leading v) is written to pyproject.toml, docs/conf.py, and CITATION.cff.
- Prefixed form ("v...") is written to paper/paper.md (only when --update-paper is given).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import re
import sys


@dataclass
class Replacement:
    pattern: str
    repl: str
    description: str


def apply_replacements(
    path: Path,
    replacements: list[Replacement],
    verbose: bool = False,
) -> list[str]:
    """Apply regex replacements; return descriptions that did not match.

    When ``verbose`` is True, each replacement is printed as:
        {path}: {description}: '{before}' -> '{after}'
    """
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for rep in replacements:
        pattern = re.compile(rep.pattern, flags=re.MULTILINE)

        def _repl(match: re.Match) -> str:
            before = match.group(0)
            after = match.expand(rep.repl)
            if verbose:
                print(f"{path}: {rep.description}: '{before}' -> '{after}'")
            return after

        updated, count = pattern.subn(_repl, text)
        if count == 0:
            missing.append(rep.description)
        text = updated
    path.write_text(text, encoding="utf-8")
    return missing


def build_tasks(new_version: str) -> dict[Path, list[Replacement]]:
    plain = new_version.lstrip("vV")
    prefixed = new_version if new_version.lower().startswith("v") else f"v{plain}"

    tasks: dict[Path, list[Replacement]] = {
        Path("docs/conf.py"): [
            Replacement(
                r"^(release\s*=\s*[\"']).+?([\"'])",
                rf"\g<1>{plain}\g<2>",
                "docs/conf.py release",
            )
        ],
        Path("pyproject.toml"): [
            Replacement(
                r"^(version\s*=\s*\").+?(\")",
                rf"\g<1>{plain}\g<2>",
                "pyproject.toml version fields",
            )
        ],
        Path("CITATION.cff"): [
            Replacement(
                r"^(version:\s*)v?[0-9A-Za-z.\-]+",
                rf"\g<1>{plain}",
                "CITATION.cff version",
            )
        ],
    }

    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update Scikit-Topt version strings.")
    parser.add_argument("--version", required=True, help="New version (e.g., 0.3.5 or v0.3.5).")
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also update version mention in paper/paper.md (default: off).",
    )
    args = parser.parse_args(argv)

    tasks = build_tasks(args.version)
    missing_messages: list[str] = []

    for path, reps in tasks.items():
        if not path.exists():
            missing_messages.append(f"{path} (file missing)")
            continue
        missing_messages.extend(
            apply_replacements(path, reps, verbose=True)
        )

    if missing_messages:
        sys.stderr.write("Some patterns were not updated:\n")
        for msg in missing_messages:
            sys.stderr.write(f"  - {msg}\n")
        return 1

    print(f"Updated version strings to {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
