#!/usr/bin/env python3
"""Fail the build if a term that must not be published appears anywhere in the tree.

This repository becomes public. Everything under version control, plus everything not
yet ignored, is a candidate for publication — so both are scanned:

    git ls-files  ∪  git ls-files --others --exclude-standard

**The sensitive terms are not in this file.** A checker that ships its own list of
things-that-must-not-be-published publishes them the moment the repository goes public,
which defeats the check. They live in `tools/redaction-terms.local.txt`, which is
gitignored; this file carries only patterns that are safe to read in public.

Term file format: one pattern per line, `#` comments and blank lines ignored. A line is
a regular expression, matched case-sensitively. Wrap a plain word in `\\b` so that a
project code name does not also match it as a substring of an ordinary word — the reason
a bare `NDA` is useless is that it matches "standard", "boundary" and "ndarray".

    uv run python tools/check_redaction.py

Exits 0 when clean, 1 on any match, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMS_FILE = REPO_ROOT / "tools" / "redaction-terms.local.txt"

# Patterns safe to state in public, applied on top of the local term file.
#
# Case-sensitive and word-bounded on purpose: "NDA" as a substring appears inside
# "standard", "boundary", "standalone" and "np.ndarray", and a check that reports those
# is a check nobody reads.
PUBLIC_PATTERNS: tuple[str, ...] = (
    r"\bNDA\b",
    r"\bnon-disclosure\b",
    r"\bconfidential\b",
)

# The gate's own machinery necessarily contains the words it searches for. Excluding it
# is a hole, and a knowingly chosen one: these two files are what the check is made of,
# and they are reviewed as such.
SELF_REFERENTIAL = frozenset(
    {
        Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "tests/test_redaction.py",
    }
)

# Binary and vendored content: a font or an image has no readable terms, and matching
# bytes inside one would only produce noise.
SKIP_SUFFIXES = frozenset(
    {".ttf", ".otf", ".woff", ".woff2", ".png", ".jpg", ".jpeg", ".pdf", ".ico", ".lock"}
)
SKIP_DIRS = ("fonts/", ".venv/", "out/")


@dataclass(frozen=True)
class Finding:
    path: str
    line_number: int
    pattern: str
    line: str

    def __str__(self) -> str:
        excerpt = self.line.strip()[:100]
        return f"{self.path}:{self.line_number}: matches /{self.pattern}/ — {excerpt}"


def load_patterns(terms_file: Path = TERMS_FILE) -> list[str]:
    """Public patterns plus whatever the local term file adds."""
    patterns = list(PUBLIC_PATTERNS)
    if terms_file.is_file():
        for line in terms_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
    return patterns


def publishable_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Everything tracked, plus everything untracked and not ignored."""
    paths: set[str] = set()
    for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        )
        paths.update(line for line in result.stdout.splitlines() if line)

    return sorted(
        repo_root / path
        for path in paths
        if path not in SELF_REFERENTIAL
        and Path(path).suffix.lower() not in SKIP_SUFFIXES
        and not path.startswith(SKIP_DIRS)
    )


def scan(paths: list[Path], patterns: list[str], repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Every line of every file that matches any pattern."""
    compiled = [(pattern, re.compile(pattern)) for pattern in patterns]
    findings: list[Finding] = []

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Not text, or gone since git listed it. Nothing readable to leak.
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, regex in compiled:
                if regex.search(line):
                    findings.append(
                        Finding(
                            path=path.relative_to(repo_root).as_posix(),
                            line_number=line_number,
                            pattern=pattern,
                            line=line,
                        )
                    )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check-redaction",
        description="Fail if a term that must not be published appears in the tree.",
    )
    parser.add_argument("--terms", type=Path, default=TERMS_FILE, help="local term file")
    parser.add_argument("--quiet", action="store_true", help="print findings only")
    args = parser.parse_args(argv)

    # `REPO_ROOT` is passed rather than left to the default so it is resolved at call
    # time — a default argument would freeze the value at import.
    patterns = load_patterns(args.terms)
    files = publishable_files(REPO_ROOT)
    findings = scan(files, patterns, repo_root=REPO_ROOT)

    if not args.quiet:
        source = "public patterns" + (f" + {args.terms.name}" if args.terms.is_file() else "")
        print(f"check-redaction: {len(patterns)} pattern(s) ({source}), {len(files)} file(s)")
        if not args.terms.is_file():
            # Not an error — a fresh clone has no local terms — but worth saying, since a
            # silent pass here would look identical to a real one.
            print(f"  note: {args.terms} not found; only public patterns applied")

    for finding in findings:
        print(finding, file=sys.stderr)

    if findings:
        print(f"\nFAILED: {len(findings)} match(es)", file=sys.stderr)
        return 1
    if not args.quiet:
        print("clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
