"""No address into the consumer's requirements document survives in this tree.

Two forms of address are covered, because they are one defect wearing two hats: a
requirement identifier (the Cyrillic letter EF, a hyphen and a number) and a section
reference (the section sign and a number).

Why an unresolvable address is a defect, and not merely untidy. This repository is
published. That document is not, and never will be — it lives in a neighbouring repository
and is authoritative for its own contents. So an address into it resolves to nothing for
the only reader who matters here: someone who clones this repo and has no way to look it
up. It is worse than saying nothing, because it does not read as nothing — it reads as a
precise pointer to a rule that must exist, so a reader takes the sentence on trust and
stops asking what the rule actually says. The address buys the reader's attention and then
hands back a dead end.

The second failure is rot, and it takes a different shape for each form. A requirement
identifier rots silently: renumber the document and the citation still parses, still looks
authoritative, and now points at a different requirement — the file starts lying without
anything about it looking wrong, which is exactly the class of mistake `tasks/lessons.md`
exists to stop. A section reference rots loudly, since a reorganized document is more
likely to lose the section than to renumber it into something else. That difference
weakens one argument against section references and leaves the first one — unresolvable to
every reader outside the document — untouched. Restating a requirement in substance, on
its own authority, cannot rot either way: it is either right or visibly arguable.

Hence a test rather than a one-off cleanup. An agent reading the requirements document
naturally cites its addresses back into whatever it writes, so they return unless something
fails the build when they do. This gate caught a citation introduced by the repository's
own owner within a day of being written; the discipline it replaces would not have.

Deliberately not part of the redaction gate (`tools/check_redaction.py`,
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

# Both patterns are assembled from code points rather than written out, so that this file
# contains no text matching its own patterns and can be scanned along with every other
# file. The obvious alternative — excluding this path from the scan — would make the one
# file nobody checks the one file that talks about requirement and section numbers.
ADDRESSES = (
    # U+0424 CYRILLIC CAPITAL LETTER EF, the prefix every requirement identifier carries.
    re.compile(chr(0x0424) + r"-\d"),
    # U+00A7 SECTION SIGN followed by a number. Deliberately not narrowed any further: the
    # sign does not appear anywhere in this repository's prose about its own sections, so
    # there is nothing legitimate for a narrower pattern to spare. Public statute would be
    # the one real exception, and this tree does not cite it that way either — see
    # `config/fiscal-rules.yaml`, which writes "Tax Code art. 193.1". A future jurisdiction
    # whose law is conventionally cited with the sign (German ustG, say) will trip this
    # gate; the fix then is to spell the statute out the way fiscal-rules.yaml already
    # does, which is what an outside reader can resolve anyway.
    re.compile(chr(0x00A7) + r"\s*\d"),
)

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
    """Every line of every file that addresses the requirements document by number."""
    found: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Not text, or gone since git listed it. Nothing readable to cite with.
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(address.search(line) for address in ADDRESSES):
                relative = path.relative_to(repo_root).as_posix()
                found.append(f"{relative}:{line_number}: {line.strip()[:100]}")
    return found


# ------------------------------------------------------------------ the gate --


def test_no_tracked_file_addresses_the_requirements_document_by_number():
    found = citations(tracked_files())

    assert not found, (
        "A requirement number or section reference is unresolvable for anyone outside the "
        "requirements document, and it survives the document being renumbered or "
        "reorganized while no longer pointing at what it claims to. State what the "
        "requirement says instead:\n" + "\n".join(found)
    )


# ------------------------------------------------------- the gate is not vacuous --


def test_a_planted_requirement_number_is_caught(tmp_path):
    """A checker that searches for nothing passes forever. Plant one and see it reported."""
    planted = tmp_path / "notes.yaml"
    planted.write_text(
        f"ok: fine\nnote: {chr(0x0424)}-999 requires the jurisdiction\n", encoding="utf-8"
    )

    found = citations([planted], repo_root=tmp_path)

    assert len(found) == 1
    assert found[0].startswith("notes.yaml:2:")


def test_a_planted_section_reference_is_caught(tmp_path):
    """The half added second, planted in both spacings it is written with."""
    planted = tmp_path / "notes.yaml"
    planted.write_text(
        f"ok: fine\n"
        f"tight: per {chr(0x00A7)}4.3 this type has no line items\n"
        f"spaced: per {chr(0x00A7)} 12 the amount is gross\n",
        encoding="utf-8",
    )

    found = citations([planted], repo_root=tmp_path)

    assert len(found) == 2
    assert found[0].startswith("notes.yaml:2:")
    assert found[1].startswith("notes.yaml:3:")


def test_the_file_the_citations_came_from_is_in_the_scanned_set():
    """`config/labelling-schema.yaml` is where the numbering was. If it ever drops out of
    the scanned set the gate goes green for the wrong reason."""
    paths = {path.relative_to(REPO_ROOT).as_posix() for path in tracked_files()}

    assert "config/labelling-schema.yaml" in paths


# ------------------------------------------------- this file does not trip itself --


def test_this_test_does_not_match_its_own_patterns():
    """A test searching for a pattern its own source contains passes or fails for a reason
    that has nothing to do with the tree it is checking.

    Solved by construction rather than by an exclusion list: both prefixes are spelled as
    `chr(...)` and the planted examples are interpolated the same way, so no line of this
    source matches, and the file needs no special case — it is scanned like any other
    tracked file, and a real citation added here would fail the gate above."""
    own_source = Path(__file__).resolve()

    assert citations([own_source], repo_root=own_source.parent) == []
