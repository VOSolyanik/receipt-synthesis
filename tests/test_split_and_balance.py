"""The train / validation partition, and the dimensions of the balance report added with it.

Two properties are load-bearing here and neither is obvious from reading the output:

* the partition's unit is the PERSONA, because a claim's verdict can depend on that persona's
  earlier claims through the cumulative annual limit. A per-claim partition would make a
  validation label a function of training data;
* the partition is seeded INDEPENDENTLY of the generator's own stream, so adding it changed no
  pixel and no drawn value. That is what lets an already-generated corpus be partitioned.

The report's own rule is tested here too, on every dimension it prints: a figure carries its
denominator, and something with no data is ABSENT rather than zero.
"""

from __future__ import annotations

import random
from datetime import date
from decimal import Decimal

import pytest

from receipt_synth.assembler import (
    DEFAULT_TRAIN_FRACTION,
    MIN_DOCUMENTS_PER_TARGET_CLASS,
    Dataset,
    assign_splits,
    balance_report,
)
from receipt_synth.schemas import (
    Capture,
    ClaimGroundTruth,
    DocGroundTruth,
    DocType,
    Split,
    Verdict,
)


def personas(count: int) -> list[str]:
    return [f"p{index + 1:03d}" for index in range(count)]


def a_document(doc_id, *, doc_type=DocType.FISCAL_RECEIPT, split=None,
               currency="UAH", language="uk", capture=Capture.SCREENSHOT):
    return DocGroundTruth(
        doc_id=doc_id, source_file=f"{doc_id}.png", doc_type=doc_type,
        language=language, currency=currency, amount=Decimal("10.00"),
        date=date(2026, 6, 15), counterparty="Аптека", line_items=[],
        has_qr=True, qr_is_fiscal=True, has_fiscal_number=True, capture=capture,
        content_bbox=(0.0, 0.0, 10.0, 10.0), split=split,
    )


def a_claim(claim_id, persona_id, doc_ids, *, verdict=Verdict.COVERED, split=None):
    return ClaimGroundTruth(
        claim_id=claim_id, persona_id=persona_id, category="vitamins_nutrition",
        documents=list(doc_ids), verdict=verdict, split=split,
    )


# ------------------------------------------------------------- the assignment --


def test_the_requested_fraction_is_honoured_by_rounding():
    """HAND-COMPUTED from `round(n × fraction)` and nothing else.

    20 × 0.85 = 17.0 exactly, so 17 train and 3 validation.
     7 × 0.85 =  5.95, which rounds to 6, so 6 and 1.
    """
    for count, fraction, expected_train in ((20, 0.85, 17), (7, 0.85, 6), (4, 0.5, 2)):
        split = assign_splits(personas(count), seed=42, train_fraction=fraction)
        train = sum(1 for side in split.values() if side is Split.TRAIN)

        assert len(split) == count, "a persona was lost or duplicated by the partition"
        assert train == expected_train, (count, fraction)
        assert count - train == count - expected_train


def test_a_tie_rounds_the_way_python_rounds_and_that_is_pinned_deliberately():
    """⚠️ `round` IS BANKER'S ROUNDING, so 10 × 0.25 = 2.5 becomes 2 and not 3.

    Pinned rather than left to be discovered. The choice barely matters for a partition, but an
    unpinned tie is exactly the kind of thing a later tidy-up "fixes" into `int(x + 0.5)`, which
    would silently move one persona across the boundary of every run whose product lands on a half.
    """
    split = assign_splits(personas(10), seed=1, train_fraction=0.25)

    assert sum(1 for side in split.values() if side is Split.TRAIN) == 2


def test_the_same_seed_and_the_same_personas_give_the_same_partition():
    ids = personas(30)

    assert assign_splits(ids, seed=7, train_fraction=0.85) == assign_splits(
        ids, seed=7, train_fraction=0.85
    )


def test_a_different_seed_gives_a_different_partition():
    """Otherwise the seed is decorative and every run would validate on the same personas."""
    ids = personas(30)

    assert assign_splits(ids, seed=7, train_fraction=0.85) != assign_splits(
        ids, seed=8, train_fraction=0.85
    )


def test_the_partition_consumes_from_no_generator_it_does_not_own():
    """🔴 THE PROPERTY THAT MAKES THE PARTITION A LABELLING RATHER THAN A CHANGE.

    `assign_splits` builds its own generator from the seed. Were it to draw from a stream somebody
    else is using, every value taken after it would shift, and a corpus generated before the
    partition existed could never be reproduced from the same seed.

    BOTH STREAMS ARE CHECKED, and the second is the one that matters. A caller-owned `Random`
    instance is the obvious case and also the one that cannot really happen — the function takes no
    generator. The reachable mistake is `random.shuffle(...)`, which silently uses the MODULE-GLOBAL
    generator, and a test that watched only an instance would report clean while the global stream
    moved under every other caller in the process.
    """
    instance = random.Random(99)
    reference = random.Random(99)
    random.seed(20260803)
    global_reference = random.Random()
    global_reference.seed(20260803)

    assign_splits(personas(50), seed=99, train_fraction=0.85)

    assert instance.random() == reference.random(), (
        "the partition drew from a generator instance it does not own"
    )
    assert random.random() == global_reference.random(), (
        "the partition drew from the MODULE-GLOBAL generator — `random.shuffle` rather than "
        "`Random(...).shuffle` — which moves the stream for every other caller in the process"
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_a_fraction_that_is_not_a_partition_is_refused(fraction):
    """0 and 1 are refused as firmly as 1.5: everything on one side is not a partition, and a run
    asking for one has a mistyped argument rather than an unusual intention."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        assign_splits(personas(10), seed=1, train_fraction=fraction)


def test_a_run_too_small_to_partition_gets_an_empty_side_rather_than_a_forced_one():
    """🔴 NO SIDE IS TOPPED UP, and WHICH side comes out empty follows the fraction.

    One persona at the default is a single side. It used to be the training one, because the
    default used to be 0.85; at 0.5 `round(1 × 0.5)` is 0 and the persona lands on validation
    instead. The property under test is unchanged and is not about which side wins: forcing a
    persona across would satisfy the SHAPE of a partition while producing a side of one person —
    worse than nothing, because it looks like something.
    """
    split = assign_splits(personas(1), seed=1, train_fraction=DEFAULT_TRAIN_FRACTION)

    assert len(set(split.values())) == 1, "a single persona cannot occupy two sides"


def test_the_default_fraction_is_not_the_convention_borrowed_from_training():
    """🔴 THE DEFAULT IS 0.5 AND THE REASON IS THAT NOTHING IS TRAINED ON THIS DATASET.

    85/15 belongs to tasks where a model LEARNS on the larger side, and the larger side is large
    because learning consumes examples. Here both sides answer a different question: the partition
    guards against fitting the MEASUREMENT — whoever uses this corpus inspects documents, finds
    where extraction errs and adjusts, and a figure does not count on the documents that were
    inspected and tuned against. Inspection needs a few dozen documents; measurement wants as many
    as the corpus allows.

    Pinned as a test rather than left in a comment because a borrowed convention is exactly the
    kind of number that creeps back in during an unrelated edit, carrying an authority it never
    earned. Changing it should require saying so here.
    """
    assert DEFAULT_TRAIN_FRACTION == 0.5


def test_the_default_gives_the_measurement_side_at_least_half():
    """The PROPERTY behind the digit above, so the two tests fail for different reasons.

    A drift back toward a training convention would raise the train share above a half, and this
    catches that without pinning any particular value — 0.5, 0.4 and 0.3 all pass, 0.85 does not.
    The binding constraint on how large validation must be is not a convention at all: a per-class
    figure needs MIN_DOCUMENTS_PER_TARGET_CLASS on the side it is measured on, and the thinnest
    target class runs near 8% of documents.
    """
    validation_fraction = 1 - DEFAULT_TRAIN_FRACTION

    assert validation_fraction >= DEFAULT_TRAIN_FRACTION, (
        f"the default sends {DEFAULT_TRAIN_FRACTION:.0%} to a side that trains nothing, leaving "
        f"{validation_fraction:.0%} to carry every per-class figure"
    )


def test_the_cli_default_is_the_assembler_default_rather_than_its_own_copy():
    """Two spellings of one decision drift apart, and this one would drift silently: a run would
    partition differently from a direct call to `generate_dataset`, and nothing would say so."""
    from receipt_synth.cli import build_parser

    parsed = build_parser().parse_args(["--seed", "1"])

    assert parsed.split == DEFAULT_TRAIN_FRACTION


# ------------------------------------------------------------ report: the split --


def a_dataset(*, split=True, verdicts=(Verdict.COVERED, Verdict.PARTIALLY_COVERED)):
    """Two personas, one claim each, two documents each — enough to have two sides."""
    documents, claims = [], []
    assignment = {"p001": Split.TRAIN, "p002": Split.VALIDATION} if split else {}
    for persona_id, verdict in zip(("p001", "p002"), verdicts, strict=True):
        side = assignment.get(persona_id)
        doc_ids = [f"{persona_id}_c1_d1", f"{persona_id}_c1_d2"]
        documents += [
            a_document(doc_ids[0], doc_type=DocType.INVOICE, split=side),
            a_document(doc_ids[1], doc_type=DocType.PAYMENT_CONFIRMATION, split=side),
        ]
        claims.append(a_claim(f"{persona_id}_c1", persona_id, doc_ids, verdict=verdict, split=side))
    return Dataset(
        seed=1, personas=[], claims=claims, documents=documents,
        split=assignment, train_fraction=0.5 if split else None,
    )


def test_the_report_names_a_verdict_a_side_does_not_have():
    """The half of the split block that matters. The sizes alone are the reassuring half; this is
    the one that says the partition is unusable for a class — and since the partition is NOT
    stratified, it is a real outcome rather than a hypothetical."""
    report = balance_report(a_dataset())

    assert "THIS SIDE CONTAINS NO partially_covered" in report
    assert "THIS SIDE CONTAINS NO covered" in report
    assert "not stratified" in report


def test_an_unpartitioned_dataset_says_so_rather_than_printing_an_empty_split():
    """NOT COMPUTED and EMPTY are different states. A dataset assembled by hand has no partition at
    all, which is not a partition that put everything on one side."""
    report = balance_report(a_dataset(split=False))

    assert "NOT COMPUTED" in report
    assert a_dataset(split=False).as_manifest()["split"] is None


def test_an_empty_side_is_declared_EMPTY_and_never_as_a_share_of_zero():
    """🔴 The rule the capture block already follows, applied to the partition. `0.0%` says a
    measurement was taken; EMPTY says there is no side here. A run of one persona has the second."""
    dataset = Dataset(
        seed=1, personas=[], claims=[a_claim("p001_c1", "p001", ["d1"], split=Split.TRAIN)],
        documents=[a_document("d1", split=Split.TRAIN)],
        split={"p001": Split.TRAIN}, train_fraction=0.85,
    )
    line = next(
        ln for ln in balance_report(dataset).splitlines() if ln.strip().startswith("validation")
    )

    assert "EMPTY" in line
    assert "0.0%" not in line


def test_the_manifest_carries_the_decision_and_not_the_id_lists():
    """One statement of where a record went, and it is on the record. A list of ids here beside a
    `split` field there would be two statements of one fact, and the first edit to either makes
    them disagree with nothing to notice it."""
    manifest = a_dataset().as_manifest()
    block = manifest["split"]

    assert block["unit"] == "persona"
    assert block["stratified"] is False
    assert block["train_fraction_requested"] == 0.5
    assert block["realized"]["train"] == {"personas": 1, "claims": 1, "documents": 2}
    assert block["realized"]["validation"] == {"personas": 1, "claims": 1, "documents": 2}
    for side in block["realized"].values():
        assert not any(isinstance(value, list) for value in side.values())


# -------------------------------------------------------- report: the dimensions --


def test_a_document_class_no_archetype_can_build_is_ABSENT_and_carries_no_threshold():
    """🔴 ABSENT AND BELOW-THRESHOLD ARE DIFFERENT FINDINGS. `DocType` names classes the registry
    cannot build; a count of 0 beside a minimum would invite somebody to close a shortfall that no
    run of any size can close. Asserted in both directions — the buildable class below the minimum
    must still be flagged, or this test would pass for a report that flagged nothing."""
    report = balance_report(a_dataset())
    lines = {ln.split()[0]: ln for ln in report.splitlines() if ln.startswith("  ")}

    assert "ABSENT" in lines["act"]
    assert str(MIN_DOCUMENTS_PER_TARGET_CLASS) not in lines["act"]
    assert f"BELOW {MIN_DOCUMENTS_PER_TARGET_CLASS}" in lines["invoice"]


def test_the_per_class_minimum_is_reported_and_never_acted_on():
    """🔴 A POSITIONAL QUESTION, ANSWERED WITH `ast` RATHER THAN BY RUNNING ANYTHING.

    The threshold must not reach the generation path. A generator that resized a run or reweighted
    a draw until its own report looked healthy produces a report that cannot fail, and a report that
    cannot fail is one nobody reads. Reading the source is the only instrument that can see this:
    no output of a passing run distinguishes "the corpus met the threshold" from "the corpus was
    made to meet it".
    """
    import ast
    import inspect

    from receipt_synth import assembler

    tree = ast.parse(inspect.getsource(assembler))
    users = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Name) and inner.id == "MIN_DOCUMENTS_PER_TARGET_CLASS"
            for inner in ast.walk(node)
        )
    }

    assert users == {"_document_class_lines"}, (
        f"the per-class minimum is read by {sorted(users)}; it may be READ BY THE REPORT ONLY. "
        "Anything on the generation path that consults it is tuning the corpus to its own target."
    )


@pytest.mark.parametrize("title", ["Currency", "Language"])
def test_a_dimension_with_one_value_says_so(title):
    """`UAH 100.0%` on its own reads as a balanced distribution that happens to have one member.
    A corpus of a single currency does not exercise the dimension at all, and a per-value figure
    computed on it is the corpus average under another name."""
    report = balance_report(a_dataset())
    index = next(i for i, ln in enumerate(report.splitlines()) if ln.startswith(f"{title} —"))
    block = "\n".join(report.splitlines()[index : index + 4])

    assert "ONE VALUE ACROSS THE WHOLE CORPUS" in block
    assert title.lower() in block


def test_a_dimension_with_two_values_is_not_flagged():
    """The negative half. Without it the flag above could be unconditional and the test would not
    know — a warning that is always printed is a warning nobody reads."""
    dataset = a_dataset()
    dataset.documents[0].currency = "EUR"
    report = balance_report(dataset)
    index = next(i for i, ln in enumerate(report.splitlines()) if ln.startswith("Currency —"))
    block = "\n".join(report.splitlines()[index : index + 4])

    assert "ONE VALUE" not in block
    assert "EUR" in block and "UAH" in block


def test_a_finding_about_the_corpus_never_wears_the_self_contradiction_marker():
    """🔴 `!!` IS A RESERVED WORD IN THIS REPORT AND MEANS ONE THING: the report contradicts itself.

    Rows that do not sum to their header, a verdict realized that the planner cannot draw, claims
    missing that nothing accounts for — none of which should ever fire. A FINDING ABOUT THE CORPUS
    is a different statement and is written in capitals and plain English instead: ABSENT, EMPTY,
    BELOW n, ONE VALUE, THIS SIDE CONTAINS NO x. Those are EXPECTED to fire.

    The two merged once, and it was not caught by reading. Marking a below-threshold class with
    `!!` broke `test_the_verdict_table_accounts_for_every_claim_it_counts`, whose whole assertion is
    that a healthy run's report carries no `!!` — an assertion that means nothing the moment the
    sigil also marks ordinary findings. This test exists so the collision cannot come back quietly.
    """
    report = balance_report(a_dataset())

    assert "!!" not in report, (
        "a corpus finding is marked `!!`, which is reserved for the report contradicting itself; "
        "use a capitalised word instead"
    )
    for finding in ("ABSENT", f"BELOW {MIN_DOCUMENTS_PER_TARGET_CLASS}",
                    "ONE VALUE ACROSS THE WHOLE CORPUS", "THIS SIDE CONTAINS NO"):
        assert finding in report, (
            f"{finding!r} is missing, so the assertion above passes for a report that says nothing"
        )


def test_every_dimension_of_an_empty_corpus_reports_absence_rather_than_a_distribution():
    """A report over nothing must not read as a report over something. Every dimension is checked,
    with the denominator stated — five blocks, not the one that happened to be looked at."""
    report = balance_report(Dataset(seed=1, personas=[], claims=[], documents=[]))

    for title in ("Currency", "Language"):
        assert f"{title} — 0 document(s)" in report
        assert "ABSENT — this run produced no documents" in report
    assert "Document classes — 0 document(s)" in report
    assert "Capture channels — 0 document(s)" in report
    assert "Train / validation split — NOT COMPUTED" in report
