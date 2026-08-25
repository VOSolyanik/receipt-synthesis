"""The redaction gate: nothing that must not be published is in the tree.

Two halves, and the second matters as much as the first. A checker that reports "clean"
because it searches for nothing at all reports "clean" forever, and the repository is
public by then.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# `tools/` is put on the import path by tests/conftest.py.
from check_redaction import (
    PUBLIC_PATTERNS,
    TERMS_FILE,
    Finding,
    load_patterns,
    main,
    publishable_files,
    scan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the gate --


def test_the_tree_is_clean():
    assert scan(publishable_files(), load_patterns()) == []


def test_the_gate_exits_zero():
    assert main(["--quiet"]) == 0


def test_it_scans_both_tracked_and_untracked_files():
    """Untracked-but-not-ignored files are one `git add -A` away from being published,
    so leaving them out would check the wrong set."""
    paths = {path.relative_to(REPO_ROOT).as_posix() for path in publishable_files()}

    assert "README.md" in paths, "tracked files are not being scanned"
    assert "config/policy.yaml" in paths
    assert not any(path.startswith("out/") for path in paths), "generated output is ignored"


# ----------------------------------------------------- the gate is not vacuous --


def test_it_finds_a_planted_term(tmp_path):
    planted = tmp_path / "notes.md"
    planted.write_text("harmless line\nthis one mentions an NDA clause\n", encoding="utf-8")

    findings = scan([planted], load_patterns(), repo_root=tmp_path)

    assert len(findings) == 1
    assert findings[0].line_number == 2
    assert isinstance(findings[0], Finding)


def test_it_reports_every_occurrence_not_just_the_first(tmp_path):
    planted = tmp_path / "notes.md"
    planted.write_text("NDA\nfine\nNDA again\n", encoding="utf-8")

    assert len(scan([planted], load_patterns(), repo_root=tmp_path)) == 2


def test_a_planted_term_makes_the_gate_fail(tmp_path, monkeypatch, capsys):
    """End to end, through `main`, so the exit code is what is asserted — that is what a
    hook or a ci step reads."""
    planted = tmp_path / "leak.md"
    planted.write_text("mentions a confidential arrangement\n", encoding="utf-8")

    monkeypatch.setattr("check_redaction.publishable_files", lambda *_, **__: [planted])
    monkeypatch.setattr("check_redaction.REPO_ROOT", tmp_path)

    assert main(["--quiet"]) == 1
    assert "leak.md" in capsys.readouterr().err


# ------------------------------------------------------------- pattern quality --


@pytest.mark.parametrize(
    "innocent",
    ["standard", "boundary", "standalone", "np.ndarray", "calendar", "secondary"],
)
def test_word_bounded_patterns_do_not_match_ordinary_words(innocent, tmp_path):
    """`NDA` as a bare substring matches "sta-NDA-rd", "bou-NDA-ry" and "np.-NDA-rray".
    A gate that reports those is a gate that gets ignored, so every pattern is anchored
    on word boundaries and matched case-sensitively."""
    planted = tmp_path / "code.py"
    planted.write_text(f"value = {innocent}\n", encoding="utf-8")

    assert scan([planted], load_patterns(), repo_root=tmp_path) == []


def test_the_repository_domain_vocabulary_is_not_a_forbidden_term(tmp_path):
    """"benefits" is what this generator is about — it is in the README, in
    architecture.md and throughout policy.yaml. Listing it would light the gate up on
    every run, and a gate that is always red is a gate nobody reads."""
    planted = tmp_path / "policy.yaml"
    planted.write_text("# a benefits package, benefit categories\n", encoding="utf-8")

    assert scan([planted], load_patterns(), repo_root=tmp_path) == []


def test_public_patterns_are_safe_to_read_in_public():
    """The list in the tracked file may only contain generic words. Anything specific —
    a project code name, an employer, a person — belongs in the gitignored term file,
    because this file is published along with everything else."""
    for pattern in PUBLIC_PATTERNS:
        assert pattern.startswith(r"\b") and pattern.endswith(r"\b")


def test_local_terms_extend_the_public_ones(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("# a comment\n\n\\bProjectCodeName\\b\n", encoding="utf-8")

    patterns = load_patterns(terms)

    assert set(PUBLIC_PATTERNS) <= set(patterns)
    assert r"\bProjectCodeName\b" in patterns
    assert not any(pattern.startswith("#") for pattern in patterns), "comments leaked in"


def test_the_local_term_file_is_never_committed():
    """It holds the things that must not be published; committing it publishes them."""
    tracked = subprocess.run(
        ["git", "ls-files", str(TERMS_FILE.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == "", "the local term file is tracked by git"


def test_the_checker_does_not_report_itself():
    """It necessarily contains the words it searches for."""
    paths = {path.relative_to(REPO_ROOT).as_posix() for path in publishable_files()}
    assert "tools/check_redaction.py" not in paths
