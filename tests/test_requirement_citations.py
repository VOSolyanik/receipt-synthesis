"""No requirement number from the consumer's requirements document survives in this tree.

This repository is published. That document is not, and never will be — it lives in a
neighbouring repository and is authoritative for its own contents. So a citation of the
form `Ф-` plus a number resolves to nothing for the only reader who matters here: someone
who clones this repo and has no way to look the number up. It reads as a precise reference
and carries no information at all.

The second failure is worse, because it is silent. Requirement numbers are renumbered.
When that happens the citation still parses, still looks authoritative, and now points at
a different requirement — the file starts lying without anything about it looking wrong,
which is exactly the class of mistake `tasks/lessons.md` exists to stop. Restating the
requirement in substance, on its own authority, cannot rot that way: it is either right or
visibly arguable.

Hence a test rather than a one-off cleanup. An agent reading the requirements document
naturally cites its numbers back into whatever it writes, so the references return unless
something fails the build when they do.

Deliberately NOT part of the redaction gate (`tools/check_redaction.py`,
`tests/test_redaction.py`). That gate fails because something must never be published;
this one fails because a citation cannot be resolved by whoever reads it. They are
different problems with different fixes — redact versus restate — and putting both behind
one gate produces a failure message that no longer tells anyone which of the two they hit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# U+0424 is CYRILLIC CAPITAL LETTER EF, the prefix every one of those identifiers carries.
# It is built from its code point rather than written out so that THIS FILE contains no
# text matching its own pattern and can be scanned along with every other file. The obvious
# alternative — excluding this path from the scan — would make the one file nobody checks
# the one file that talks about requirement numbers.
REQUIREMENT_ID = re.compile(chr(0x0424) + r"-\d")

# `git ls-files` lists fonts and rendered images too. There is nothing citable inside a
# glyph table, and matching bytes in one would only produce noise.
BINARY_SUFFIXES = frozenset(
    {".ttf", ".otf", ".woff", ".woff2", ".png", ".jpg", ".jpeg", ".pdf", ".ico"}
)


def tracked_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Everything under version control — precisely what publishing the repo hands over."""
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return sorted(
        repo_root / line
        for line in result.stdout.splitlines()
        if line and Path(line).suffix.lower() not in BINARY_SUFFIXES
    )


def citations(paths: list[Path], repo_root: Path = REPO_ROOT) -> list[str]:
    """Every line of every file that cites a requirement by number."""
    found: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Not text, or gone since git listed it. Nothing readable to cite with.
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if REQUIREMENT_ID.search(line):
                relative = path.relative_to(repo_root).as_posix()
                found.append(f"{relative}:{line_number}: {line.strip()[:100]}")
    return found


# ------------------------------------------------------------------ the gate --


def test_no_tracked_file_cites_a_requirement_by_number():
    found = citations(tracked_files())

    assert not found, (
        "A requirement number is unresolvable for anyone outside the requirements "
        "document, and points at the wrong requirement once it is renumbered. State what "
        "the requirement says instead:\n" + "\n".join(found)
    )


# ------------------------------------------------------- the gate is not vacuous --


def test_a_planted_citation_is_caught(tmp_path):
    """A checker that searches for nothing passes forever. Plant one and see it reported."""
    planted = tmp_path / "notes.yaml"
    planted.write_text(
        f"ok: fine\nnote: {chr(0x0424)}-999 requires the jurisdiction\n", encoding="utf-8"
    )

    found = citations([planted], repo_root=tmp_path)

    assert len(found) == 1
    assert found[0].startswith("notes.yaml:2:")


def test_the_file_the_citations_came_from_is_in_the_scanned_set():
    """`config/labelling-schema.yaml` is where the numbering was. If it ever drops out of
    the scanned set the gate goes green for the wrong reason."""
    paths = {path.relative_to(REPO_ROOT).as_posix() for path in tracked_files()}

    assert "config/labelling-schema.yaml" in paths


# ------------------------------------------------- this file does not trip itself --


def test_this_test_does_not_match_its_own_pattern():
    """A test searching for a pattern its own source contains passes or fails for a reason
    that has nothing to do with the tree it is checking.

    Solved by construction rather than by an exclusion list: the prefix is spelled
    `chr(0x0424)` and the planted example is interpolated the same way, so no line of this
    source matches, and the file needs no special case — it is scanned like any other
    tracked file, and a real citation added here would fail the gate above."""
    own_source = Path(__file__).resolve()

    assert citations([own_source], repo_root=own_source.parent) == []
