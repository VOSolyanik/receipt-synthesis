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
    MIN_DOCUMENTS_PER_TARGET_CLASS,
    Dataset,
    _absence_probability,
    assign_splits,
    balance_report,
)
from receipt_synth.policy_engine import (
    insufficient_evidence_causes,
    insufficient_evidence_causes_min_run_size,
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

    One persona at a half is a single side: `round(1 × 0.5)` is 0, so the persona lands on
    VALIDATION. Forcing it across would satisfy the SHAPE of a partition while producing a side of
    one person — worse than nothing, because it looks like something. The 0.5 is a chosen input,
    not an inherited one; there is no default left to inherit.

    🔴 THE ASSERTION IS THE SIDE, NOT THE COUNT OF SIDES, AND THE OLD FORM WAS VACUOUS. It read
    `len(set(split.values())) == 1`, which is TRUE OF ANY IMPLEMENTATION: one persona goes in, so
    one entry comes out, so one distinct value comes out. A top-up that clamps `train_size` to at
    least 1 — the exact defect the docstring warns about — survived it, because clamping still
    yields a single-entry dict. There is no fixed point outside the thing under test in a count
    derived from an input of size one; the SIDE is that fixed point, computed on paper from
    `round(1 × 0.5) = 0`, and a clamp moves it to `train` immediately.
    """
    split = assign_splits(personas(1), seed=1, train_fraction=0.5)

    assert split == {"p001": Split.VALIDATION}, (
        "the one persona was not left on the side the fraction puts it on — a run too small to "
        f"partition got a forced side instead of an empty one: {split}"
    )


def test_the_split_fraction_cannot_be_inherited_from_a_default():
    """🔴 WHAT THE DELETED `DEFAULT_TRAIN_FRACTION == 0.5` TEST WAS PROTECTING, AT ITS ROOT.

    That test pinned a digit so that 85/15 — a convention whose premise is absent here, since
    nothing is trained on this dataset — could not creep back during an unrelated edit carrying an
    authority it never earned. The constant is gone, so the digit cannot be pinned. But the digit
    was never the thing: what it guarded is that A PARTITION NOBODY DECLARED MUST NOT BE APPLIED,
    and having no default at all is the strictly stronger form — there is no number left to creep
    back INTO.

    Checked at EVERY entry point rather than at the one that held the constant, because a default
    written as a bare literal in a signature is the same defect and less visible than a named
    constant was. The old test could not have caught that at all: it compared one constant against
    one digit and said nothing about a second spelling elsewhere.
    """
    import inspect

    from receipt_synth.assembler import generate_dataset

    for entry in (assign_splits, generate_dataset):
        parameter = inspect.signature(entry).parameters["train_fraction"]
        assert parameter.default is inspect.Parameter.empty, (
            f"{entry.__name__} defaults train_fraction to {parameter.default!r}, so a run can be "
            "performed without the partition ever having been declared"
        )


def test_the_guidance_on_what_to_pass_outlived_the_constant():
    """🔴 THE ARGUMENT FOR A HALF DID NOT DISAPPEAR WITH THE DEFAULT — IT CHANGED JOBS.

    The deleted `1 - default >= default` test held the PROPERTY behind the digit: the measurement
    side never smaller than the development side, because nothing is trained on this dataset and a
    per-class figure needs MIN_DOCUMENTS_PER_TARGET_CLASS on the side it is MEASURED on, so the
    thinnest class sets the floor. With no default there is no fraction of ours to hold that
    property against — the caller's is not ours to constrain, and a consumer that really does train
    has every reason to pass something else.

    So the same reasoning is held where it now has to work: as GUIDANCE READ BEFORE CHOOSING. A
    required flag whose help says only "a number between 0 and 1" would have removed the default
    and lost the argument with it, which is the failure this test exists to make loud. `--help` is
    the copy that ships with the tool, so it is the one a test can reach.

    `_actions` rather than `format_help()`: the guidance has to be checked as the SPLIT flag's
    text, and a substring search over the whole help would pass on a `0.5` printed by any other
    flag added later.
    """
    from receipt_synth.cli import build_parser

    guidance = next(
        action.help for action in build_parser()._actions if action.dest == "split"
    )

    assert "0.5" in guidance, "the help no longer says what to pass"
    assert "thinnest" in guidance, "the help names a number without the argument that produced it"
    assert "required" in guidance, "the help does not say the decision has to be made"


def test_a_run_that_declares_no_split_is_refused_rather_than_defaulted(capsys):
    """The command-line half of the same property, and the half a user meets.

    Replaces a test that asserted the parser's default equalled the assembler's, which guarded
    against two spellings of one decision drifting apart. With no default anywhere there is exactly
    one spelling — the caller's — so that drift cannot occur, and what is worth holding instead is
    that the OMISSION IS REFUSED rather than filled in silently.

    The message is asserted as well as the exit: argparse raises `SystemExit` for a mistyped flag
    too, so the raise alone would stay green if `--split` went back to being optional.
    """
    from receipt_synth.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--seed", "1"])
    assert "--split" in capsys.readouterr().err

    parsed = build_parser().parse_args(["--seed", "1", "--split", "0.4"])
    assert parsed.split == 0.4


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


# --------------------------------- report: insufficient_evidence run-size guideline --


def a_run_realizing_one_cause(size, *, realized_cause="payment_precedes_subject"):
    """`size` built claims: one `insufficient_evidence` claim carrying only `realized_cause`, so
    the other declared cause is realized zero times, and `size - 1` ordinary `covered` filler
    claims so the run reaches the requested total. Documents are irrelevant to the cause block
    and left empty, as `test_every_dimension_of_an_empty_corpus...` already established is safe."""
    claims = [
        ClaimGroundTruth(
            claim_id="c001", persona_id="p001", category="vitamins_nutrition",
            documents=[], verdict=Verdict.INSUFFICIENT_EVIDENCE, imperfection=[realized_cause],
        )
    ]
    claims += [
        a_claim(f"c{index:03d}", f"p{index:03d}", [], verdict=Verdict.COVERED)
        for index in range(2, size + 1)
    ]
    return Dataset(seed=1, personas=[], claims=claims, documents=[])


def test_an_unrealized_cause_below_the_guideline_reads_as_run_too_small():
    """The brief's own example: a 36-claim run with zero `amount_mismatch` must not read as this
    policy violated. `payment_precedes_subject` is the one drawn, so `amount_mismatch` alone sits
    at zero — below the guideline, that is expected sampling variance, not a defect."""
    guideline = insufficient_evidence_causes_min_run_size()
    report = balance_report(a_run_realizing_one_cause(guideline - 1))
    line = next(ln for ln in report.splitlines() if ln.strip().startswith("amount_mismatch"))

    assert f"RUN TOO SMALL — {guideline - 1} built claim(s) < guideline {guideline}" in line
    assert "INVESTIGATE THE MECHANISM" not in line


def test_an_unrealized_cause_at_the_guideline_reads_as_worth_investigating():
    """At or above the guideline the same zero count is worth investigating, and the report says
    so in different words — the two readings must never share a sentence."""
    guideline = insufficient_evidence_causes_min_run_size()
    report = balance_report(a_run_realizing_one_cause(guideline))
    line = next(ln for ln in report.splitlines() if ln.strip().startswith("amount_mismatch"))

    assert (
        f"INVESTIGATE THE MECHANISM — {guideline} built claim(s), at or above "
        f"guideline {guideline}"
    ) in line
    assert "RUN TOO SMALL" not in line


def test_the_finding_prints_the_probability_its_own_guideline_is_derived_at():
    """🔴 THE FINDING USED TO ASSERT MORE THAN THE DERIVATION SUPPORTS. It read "LIKELY A
    DESIGN/MECHANISM DEFECT", i.e. better than even odds, while policy.yaml derives the guideline
    at the 95% level — so a zero AT the guideline is a ~5% event and the word was wrong by an
    order of magnitude, in the direction that costs somebody an investigation.

    Asserted against the ARITHMETIC rather than against the printed string: (1 − p)^N for the
    rarest declared cause, computed here from policy.yaml on both sides of the comparison — the
    guideline is the smallest N putting that under 5%, so the run one claim SHORT of it must sit
    at or above 5% and the guideline itself below. That pair is what pins the number to the
    threshold; a report that printed a figure drifting from the file would fail one of the two.
    """
    guideline = insufficient_evidence_causes_min_run_size()
    rarest = min(insufficient_evidence_causes().values())

    at_guideline = _absence_probability(rarest, guideline)
    one_short = _absence_probability(rarest, guideline - 1)

    assert at_guideline < 0.05 <= one_short, (
        f"guideline {guideline} is not the smallest run size putting the rarest cause's absence "
        f"below 5%: {one_short:.4f} at {guideline - 1}, {at_guideline:.4f} at {guideline}"
    )

    report = balance_report(a_run_realizing_one_cause(guideline))
    line = next(ln for ln in report.splitlines() if ln.strip().startswith("amount_mismatch"))
    assert f"a zero is a {at_guideline:.1%} event" in line

    # And the sample DOES explain it one claim lower, which is the other half of the sentence.
    smaller = balance_report(a_run_realizing_one_cause(guideline - 1))
    line = next(ln for ln in smaller.splitlines() if ln.strip().startswith("amount_mismatch"))
    assert f"a zero is a {one_short:.0%} event" in line


def test_a_realized_cause_carries_no_run_size_finding():
    """The guideline only speaks about a cause that realized zero. The cause that DID occur must
    not carry either finding, however small the run."""
    guideline = insufficient_evidence_causes_min_run_size()
    report = balance_report(a_run_realizing_one_cause(guideline - 1))
    line = next(
        ln for ln in report.splitlines() if ln.strip().startswith("payment_precedes_subject")
    )

    assert "RUN TOO SMALL" not in line
    assert "DESIGN/MECHANISM DEFECT" not in line
