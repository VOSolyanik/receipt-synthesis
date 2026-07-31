"""A limitation is a property of the corpus; a figure is a measurement of one run.

🔴 THE SPLIT THIS MODULE GUARDS. While the two shared a record, every change to the draw forced
rewriting the description of a property that had not changed — and `known_limitations` had
accumulated six "re-measured for version N" notes saying so. A limitation now describes itself
qualitatively and REFERENCES a profile; a profile is self-describing, and adding one rewrites
nothing.

The tests below hold the boundary in both directions: no limitation may quote a run's counts, and
no profile may exist without saying which run it measured.
"""

from __future__ import annotations

import re

import pytest
import yaml

from receipt_synth.config import CONFIG_DIR

CONTRACT = yaml.safe_load((CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8"))
LIMITATIONS = CONTRACT["known_limitations"]
PROFILES = CONTRACT["run_profiles"]

# Counts of a corpus look like "36 of 143" or a bare three-digit total. Two things that look like
# them are NOT them, and both are exempted by name rather than by a loose pattern:
#
#   an OBSERVATION about real documents — 👁 "1 of 3 real receipts", "1 of 8 confirmations" — which
#   is evidence and belongs with the limitation it supports;
#   a VOCABULARY property — how many distinct strings a template set can produce — which is a
#   property of config/generation.yaml and does not move when a seed does.
_OBSERVATION = re.compile(r"👁[^.]{0,120}?\d+ of \d+")
_RUN_COUNT = re.compile(r"\b\d+ of \d{2,}\b")


def body_of(limitation: dict) -> str:
    return " ".join(
        str(value) for key, value in limitation.items()
        if key not in ("id", "name", "see_profile")
    )


# ------------------------------------------------------- the limitations side --


def test_no_limitation_carries_a_run_figure_key():
    """`share_in_reference_run` is gone from every entry. A consumer that parsed it finds
    `how_a_consumer_detects_it` and `see_profile` instead, which is declared at the version key."""
    assert LIMITATIONS, "no limitations — this test would assert nothing"

    carrying = [k["id"] for k in LIMITATIONS if "share_in_reference_run" in k]
    assert not carrying, f"{carrying} still carry the figure key"


def test_every_limitation_describes_itself_and_points_at_a_profile():
    """Qualitative first, reference second. An entry with a reference and no description would have
    moved the problem rather than solved it: a consumer reading it would still have to open a
    profile to learn what the limitation IS."""
    for limitation in LIMITATIONS:
        assert limitation["see_profile"] == "run_profiles", limitation["id"]
        described = limitation.get("how_a_consumer_detects_it", "")
        assert len(described) > 200, (
            f"{limitation['id']} points at a profile without describing itself in "
            f"{len(described)} characters"
        )


def test_no_limitation_quotes_a_count_from_a_run():
    """The boundary, checked on the prose rather than on the key names — moving a figure into a
    differently-named key would satisfy the test above and defeat the purpose.

    Its denominator is every entry, and its exemptions are named: an observation about real
    documents is evidence, not a measurement of this corpus.
    """
    offenders = {}
    for limitation in LIMITATIONS:
        body = body_of(limitation)
        without_observations = _OBSERVATION.sub("", body)
        hits = sorted(set(_RUN_COUNT.findall(without_observations)))
        if hits:
            offenders[limitation["id"]] = hits

    assert not offenders, (
        f"{len(offenders)} of {len(LIMITATIONS)} limitations quote a run count: {offenders}. "
        "A count of one corpus belongs in a profile; a limitation states the property."
    )


def test_the_observation_exemption_exempts_something():
    """The test above would be weaker if nothing used the exemption — a filter that never fires
    hides the fact that it is there. 👁 KL-05 and KL-09 both rest on counts of REAL documents."""
    using = [k["id"] for k in LIMITATIONS if _OBSERVATION.search(body_of(k))]

    assert using, (
        "no limitation carries an observed count, so the exemption in the sweep above removes "
        "nothing and its narrowness is untested"
    )


# ---------------------------------------------------------- the profiles side --


def test_exactly_one_profile_is_authoritative_and_the_prose_names_it():
    """🔴 THE SENTENCE THAT STOPS TWO SETS OF NUMBERS READING AS A CONTRADICTION.

    Until the production run there was no delivered corpus, so the invariant was that NOTHING here
    was authoritative. A dataset has now been generated and handed over, and the invariant inverts
    rather than disappears: EXACTLY ONE profile describes it, and the prose says which. Zero would
    leave a consumer without an answer; two would leave them with the contradiction this key exists
    to prevent — and a flag nobody restated in prose would be a fact only a parser could find.
    """
    profiles = PROFILES["profiles"]
    assert profiles, "no profiles recorded"

    authoritative = [p["id"] for p in profiles if p["authoritative"]]
    assert len(authoritative) == 1, (
        f"{len(authoritative)} of {len(profiles)} profiles are authoritative ({authoritative}); "
        "a consumer asking which numbers describe its corpus must get exactly one answer"
    )
    assert authoritative[0] in PROFILES["authoritative_profile"], (
        f"{authoritative[0]} carries the flag and `authoritative_profile` does not name it"
    )


def test_the_authoritative_profile_can_be_checked_against_a_corpus():
    """A profile that cannot say WHICH corpus it describes will one day be read as describing a
    different one, and the delivered corpus is the case where that costs something.

    The dataset is not committed — it is reproduced from the seed — so identity is the only thing
    standing between "figures for the corpus you hold" and "figures for some run of that command".
    Every other profile is an example and needs no such handle.
    """
    profile = next(p for p in PROFILES["profiles"] if p["authoritative"])
    identity = profile["corpus_identity"]

    for key in ("manifest_sha256", "images_sha256", "labels_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", str(identity[key])), (
            f"{profile['id']}.corpus_identity.{key} is not a SHA-256 digest"
        )
    assert len(identity["how_to_recompute"]) > 200, (
        f"{profile['id']} states digests without saying how they were computed, so nobody can "
        "reproduce one to compare against"
    )


@pytest.mark.parametrize("profile", PROFILES["profiles"], ids=lambda p: p["id"])
def test_every_profile_names_the_run_it_measured(profile):
    """A figure that cannot say which corpus it describes will one day be read as describing a
    different one. Identity is the command and the seed; both are required."""
    assert profile["command"].startswith("generate-dataset ")
    assert f"--seed {profile['seed']}" in profile["command"]
    assert profile["generator_version_at_measurement"]
    assert isinstance(profile["authoritative"], bool)


def test_a_command_that_cannot_be_run_says_so_where_it_is_recorded():
    """🔴 `--split` IS REQUIRED, SO A RECORDED COMMAND WITHOUT ONE NO LONGER EXECUTES.

    A `command` here has two jobs: it says what was run, and it is the handle by which a reader
    reproduces the run. Requiring the flag put those in conflict for the profiles taken before the
    partition existed — their commands are ACCURATE AND NOT RUNNABLE, and neither resolution is
    available: inventing a fraction would make a run claim a partition it never had, and deleting
    the profile would destroy the record that the run happened.

    So the tension is recorded rather than resolved, and this holds the CLASS rather than the three
    entries that have it today. A profile added later whose command omits the flag is either a run
    that inherited a default — in which case the value is knowable and belongs in the command — or
    one that predates the flag, in which case it has to say so. Silence is the only outcome ruled
    out, because silence is what leaves a reader pasting an unexecutable line into a shell.
    """
    profiles = PROFILES["profiles"]
    assert profiles, "no profiles, so this test would assert nothing"

    unexplained = [
        p["id"] for p in profiles
        if "--split" not in p["command"]
        and len(str(p.get("the_command_predates_the_split_flag", ""))) < 100
    ]
    assert not unexplained, (
        f"{len(unexplained)} of {len(profiles)} profiles record a command that will be refused "
        f"for want of --split and do not say why: {unexplained}"
    )

    # Both sides of the split have to be occupied, or the check above is satisfied by a shape
    # nobody is in. A file where every command carried the flag would pass it vacuously.
    runnable = [p["id"] for p in profiles if "--split" in p["command"]]
    historical = [p["id"] for p in profiles if "the_command_predates_the_split_flag" in p]
    assert runnable, "no profile records a runnable command, so the exemption exempts everything"
    assert historical, (
        "no profile carries the historical annotation, so the branch that permits an unrunnable "
        "command is untested"
    )


@pytest.mark.parametrize("profile", PROFILES["profiles"], ids=lambda p: p["id"])
def test_every_distribution_in_a_profile_carries_its_denominator(profile):
    """The discipline that survives the move. A block of counts with no denominator reads as
    exhaustive and never is — so every mapping of counts inside a profile carries a `note` saying
    what it was measured over."""
    blocks = {
        key: value for key, value in profile.items()
        if isinstance(value, dict) and any(isinstance(v, int) for v in value.values())
    }
    assert blocks, f"{profile['id']} records no distribution at all"

    for key, block in blocks.items():
        assert "note" in block, f"{profile['id']}.{key} has counts and no denominator"
        assert re.search(r"\bdenominator\b|\bof \d+\b|\bZero of\b", block["note"], re.IGNORECASE), (
            f"{profile['id']}.{key} has a note that does not state what it was measured over"
        )


def test_a_profile_can_be_added_without_touching_a_limitation():
    """The point of the whole split, asserted as a property of the shape rather than as a promise:
    a limitation references the SECTION, never a profile id, so adding `RP-03` changes nothing in
    `known_limitations`. Were a limitation to name a profile, step 10 would have to rewrite every
    entry that mentioned the old one."""
    ids = [p["id"] for p in PROFILES["profiles"]]
    assert ids, "no profiles, so this test would assert nothing"

    for limitation in LIMITATIONS:
        assert limitation["see_profile"] == "run_profiles"
        # THE WHOLE ENTRY, INCLUDING ITS `name`. The first version scanned only the body, and a
        # mutation putting a profile id into the name survived — the id would then be just as much
        # of an edit for step 10 to make, and just as invisible.
        everywhere = " ".join(str(value) for value in limitation.values())
        assert not any(pid in everywhere for pid in ids), (
            f"{limitation['id']} names a specific profile, so adding one would force an edit here"
        )
