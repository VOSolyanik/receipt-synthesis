"""The pipeline ends: persona, plan, degradation, and the run as a whole.

The claim these tests exist to defend is the one the whole tool rests on — that a seed
determines the output. It is checked at each stage and again on the finished dataset,
because a single unseeded draw anywhere makes the promise false everywhere.
"""

from __future__ import annotations

import json
import random
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from receipt_synth.assembler import balance_report, generate_dataset
from receipt_synth.claim_planner import (
    ARCHETYPES,
    REALIZABLE_VERDICTS,
    archetypes_for,
    documentable_categories,
    draw_partially_covered_cause,
    draw_verdict,
    plan_claim,
    plan_claims,
    plannable_categories,
    unrealizable_verdicts,
    why_no_claim,
)
from receipt_synth.cli import main
from receipt_synth.config import load_policy
from receipt_synth.content_builder import MAX_LINE_ITEMS, is_valid_rnokpp
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import Ledger
from receipt_synth.schemas import (
    Capture,
    ClaimGroundTruth,
    Country,
    DocType,
    Verdict,
    VerdictBasis,
)

SEED = 20260803


# ----------------------------------------------------------------- personas --


def persona(seed: int = SEED, country: Country = Country.UA):
    return generate_persona(random.Random(seed), persona_id="p001", country=country)


def test_persona_is_deterministic_under_seed():
    assert persona() == persona()


def test_personas_differ_between_seeds():
    names = {persona(seed).full_name for seed in range(20)}
    assert len(names) > 1


def test_ukrainian_persona_carries_a_valid_rnokpp():
    assert is_valid_rnokpp(persona().tax_id)


def test_tax_id_agrees_with_the_person_it_belongs_to():
    """The identifier is not merely checksum-correct: its first five digits are the
    birth date and the ninth digit's parity is the sex. A persona whose name and
    identifier disagreed would be internally inconsistent evidence."""
    for seed in range(30):
        result = persona(seed)
        days = int(result.tax_id[:5])
        birth = date(1899, 12, 31).toordinal() + days
        age_in_2026 = 2026 - date.fromordinal(birth).year
        assert 20 <= age_in_2026 <= 65


def test_persona_speaks_and_is_paid_in_something_plausible():
    result = persona(country=Country.UA)
    assert result.home_currencies == ["UAH"]
    assert result.languages == ["uk"]
    assert result.location.country is Country.UA


def test_categories_are_drawn_within_the_configured_count():
    spec = load_policy()["persona_categories"]["count_per_persona"]
    known = {entry["id"] for entry in load_policy()["categories"]}

    for seed in range(30):
        categories = persona(seed).benefit_categories
        assert set(categories) <= known
        assert len(set(categories)) == len(categories), "a category was drawn twice"
        assert spec["min"] <= len(categories) <= spec["max"] + 1


def test_medical_insurance_dominates_in_ukraine():
    """policy.yaml gives it 0.95 for UA. If the probability block were ignored the share
    would fall to whatever a uniform draw produces, and the dataset would be skewed in a
    way nothing else reports."""
    share = sum("medical_insurance" in persona(seed).benefit_categories for seed in range(200))
    assert share > 170


# -------------------------------------------------------------------- plans --


def test_plan_is_deterministic_under_seed():
    subject = persona()
    assert plan_claim(
        random.Random(1), persona=subject, claim_id="c1", ledger=Ledger()
    ) == plan_claim(random.Random(1), persona=subject, claim_id="c1", ledger=Ledger())


def test_plan_dates_fall_inside_the_active_period():
    """A date outside the window drives `insufficient_evidence`. Landing outside it by
    accident would attach that document to a `covered` label."""
    period = load_policy()["period"]
    start = date.fromisoformat(str(period["start"]))
    end = date.fromisoformat(str(period["end"]))

    subject = persona()
    for seed in range(50):
        issued = plan_claim(
            random.Random(seed), persona=subject, claim_id="c1", ledger=Ledger()
        ).issued_at
        assert start <= issued.date() <= end


def test_plan_picks_a_category_the_persona_holds():
    subject = persona()
    plan = plan_claim(random.Random(3), persona=subject, claim_id="c1", ledger=Ledger())
    assert plan.category in subject.benefit_categories


def test_plan_picks_an_archetype_that_can_carry_the_category():
    subject = persona()
    plan = plan_claim(random.Random(3), persona=subject, claim_id="c1", ledger=Ledger())
    assert plan.category in plan.archetype.categories
    assert plan.archetype.country is subject.location.country


def test_a_category_no_archetype_covers_is_refused():
    subject = persona()
    unsupported = next(
        category
        for category in subject.benefit_categories
        if not archetypes_for(Country.UA, category)
    )
    with pytest.raises(ValueError):
        plan_claim(
            random.Random(1), persona=subject, claim_id="c1",
            category=unsupported, ledger=Ledger(),
        )


def test_a_category_the_persona_does_not_hold_is_refused():
    subject = persona()
    absent = next(
        entry["id"]
        for entry in load_policy()["categories"]
        if entry["id"] not in subject.benefit_categories
    )
    with pytest.raises(ValueError):
        plan_claim(
            random.Random(1), persona=subject, claim_id="c1",
            category=absent, ledger=Ledger(),
        )


@pytest.mark.parametrize(
    "verdict",
    [
        Verdict.NOT_PROOF_OF_PAYMENT,
        Verdict.INSUFFICIENT_EVIDENCE,
        Verdict.PARTIALLY_PAID,
        Verdict.REJECTED,
    ],
)
def test_verdicts_no_archetype_can_carry_are_refused_not_faked(verdict):
    """Explicit over silent. A planner that accepted one of these and produced an ordinary
    basket would write a wrong label rather than fail. The message has to say what each
    actually needs, because the reason differs: two are content mechanisms, one needs
    document types that do not exist, and `rejected` needs a basket builder that draws no
    covered line at all — which `policy_engine` can already label and nothing can yet
    build."""
    with pytest.raises(NotImplementedError) as raised:
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1",
            verdict=verdict, ledger=Ledger(),
        )
    assert verdict.value in str(raised.value)
    assert len(str(raised.value)) > 60, "the message has to say what the verdict needs"


def test_the_planner_realizes_exactly_the_verdicts_the_engine_can_be_asked_for():
    assert set(REALIZABLE_VERDICTS) == {Verdict.COVERED, Verdict.PARTIALLY_COVERED}
    assert set(unrealizable_verdicts()) == {
        Verdict.NOT_PROOF_OF_PAYMENT,
        Verdict.INSUFFICIENT_EVIDENCE,
        Verdict.PARTIALLY_PAID,
        Verdict.REJECTED,
    }


def test_a_drawn_verdict_is_always_one_that_can_be_built():
    drawn = {draw_verdict(random.Random(seed)) for seed in range(200)}
    assert drawn == set(REALIZABLE_VERDICTS), "both realizable verdicts must be reachable"


def test_a_realizable_verdict_with_no_share_cannot_be_drawn_from():
    """The combination policy.yaml and this planner must never be in at once. `rejected`
    is declared with no share, so the day something builds one, whoever adds it to
    `REALIZABLE_VERDICTS` has to give it a share too — otherwise `random.choices` would be
    handed `None` as a weight. Named here rather than left to surface as a TypeError inside
    the standard library.
    """
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            claim_planner,
            "REALIZABLE_VERDICTS",
            (*REALIZABLE_VERDICTS, Verdict.REJECTED),
        )
        with pytest.raises(ValueError, match="no share"):
            claim_planner.draw_verdict(random.Random(1))


def test_the_drawn_mix_is_the_target_mix_renormalized_over_the_realizable_subset():
    """verdict_mix gives covered 0.50 and partially_covered 0.20; of the four that cannot
    be built, three carry 0.10 each and `rejected` carries no share at all. Renormalized
    over the two that can:

        covered            0.50 / 0.70 = 0.714…
        partially_covered  0.20 / 0.70 = 0.286…

    A member with no share must not touch that arithmetic — the draw is over the
    realizable subset, and `rejected` is not in it.

    Over 4000 draws the realized share of `covered` should sit near 0.714. The window is
    wide (±0.04) on purpose: this asserts the weights are the policy's, not that a
    pseudo-random draw hits a mean.
    """
    rng = random.Random(20260803)
    draws = [draw_verdict(rng) for _ in range(4000)]
    share = draws.count(Verdict.COVERED) / len(draws)
    assert abs(share - 0.5 / 0.7) < 0.04


def test_the_partially_covered_cause_is_drawn_from_the_policy():
    """partially_covered_causes: mixed_items 0.65, limit_exhausted 0.35."""
    rng = random.Random(11)
    draws = [draw_partially_covered_cause(rng) for _ in range(4000)]
    assert set(draws) == {"mixed_items", "limit_exhausted"}
    assert abs(draws.count("mixed_items") / len(draws) - 0.65) < 0.04


def test_a_mixed_items_claim_is_planned_with_a_coverage_target_the_builder_can_use():
    plan = plan_claim(
        random.Random(4), persona=persona(), claim_id="c1", ledger=Ledger(),
        verdict=Verdict.PARTIALLY_COVERED, cause="mixed_items",
    )
    assert plan.cause == "mixed_items"
    assert plan.coverage_target is not None
    assert Decimal(0) < plan.coverage_target < Decimal(1)
    assert plan.item_count is None


def test_a_limit_exhausted_claim_is_planned_as_a_basket_the_balance_cannot_absorb():
    """vitamins_nutrition has a 12000 limit. Against an untouched balance the planner
    must ask for a basket big enough to overrun it, and never for one longer than a shop
    would print."""
    plan = plan_claim(
        random.Random(4), persona=persona(), claim_id="c1",
        verdict=Verdict.PARTIALLY_COVERED, cause="limit_exhausted",
        category="vitamins_nutrition", ledger=Ledger(),
    )
    assert plan.cause == "limit_exhausted"
    assert plan.coverage_target is None
    assert 2 <= plan.item_count <= MAX_LINE_ITEMS


def test_an_unknown_partially_covered_cause_is_refused():
    with pytest.raises(ValueError):
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
            verdict=Verdict.PARTIALLY_COVERED, cause="wishful_thinking",
        )


def test_a_cause_without_a_partially_covered_verdict_is_refused():
    with pytest.raises(ValueError):
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
            verdict=Verdict.COVERED, cause="mixed_items",
        )


# ------------------------------------------------------------- many claims --


def test_claims_for_one_persona_come_out_in_date_order():
    """Cumulative limits bind in date order, so the plans have to be in it: a persona
    whose claims arrived unordered would exhaust its balance on whichever claim happened
    to be planned last rather than on the one that happened last."""
    plans = list(
        plan_claims(random.Random(5), persona=persona(), count=6, ledger=Ledger())
    )
    dates = [plan.issued_at for plan in plans]
    assert dates == sorted(dates)
    assert len({plan.claim_id for plan in plans}) == len(plans)


def test_planning_stops_when_the_persona_has_no_balance_left():
    """policy.yaml assigns no verdict to a claim that reimburses nothing, so the planner
    must not produce one. With the whole vitamins_nutrition limit already reimbursed — the
    only category any registered archetype can document — there is nothing left to plan."""
    subject = persona()
    ledger = Ledger()
    for category_id in subject.benefit_categories:
        ledger.record(subject.persona_id, category_id, Decimal("100000"))

    assert list(plan_claims(random.Random(5), persona=subject, count=5, ledger=ledger)) == []


def test_an_exhausted_category_is_refused_on_both_paths():
    """`plannable_categories` filters exhausted categories out of the *drawn* path only.
    A caller naming a category explicitly went around that filter, and the engine would
    then refuse to label the claim the planner had just produced — so the guard has to sit
    where both paths pass through it."""
    subject = persona()
    ledger = Ledger()
    ledger.record(subject.persona_id, "vitamins_nutrition", Decimal("12000"))

    with pytest.raises(ValueError, match="balance left"):
        plan_claim(
            random.Random(1), persona=subject, claim_id="c1",
            category="vitamins_nutrition", ledger=ledger,
        )


def test_plan_claims_is_deterministic_under_seed():
    subject = persona()
    first = list(plan_claims(random.Random(9), persona=subject, count=4, ledger=Ledger()))
    second = list(plan_claims(random.Random(9), persona=subject, count=4, ledger=Ledger()))
    assert first == second


def test_registered_archetypes_declare_what_they_can_carry():
    known = {entry["id"] for entry in load_policy()["categories"]}
    for archetype in ARCHETYPES.values():
        assert archetype.categories, f"{archetype.slug} carries no category"
        assert set(archetype.categories) <= known, f"{archetype.slug} names an unknown category"


def test_documentable_categories_is_the_intersection():
    subject = persona()
    for category in documentable_categories(subject):
        assert category in subject.benefit_categories
        assert archetypes_for(Country.UA, category)


def test_plannable_categories_narrows_documentable_ones_by_balance():
    """The two questions the planner asks are different: which categories a template can
    carry, and which of those still have money left. Merging them behind an optional
    ledger is what let the balance guard hold on some call paths and not on others."""
    subject = persona()
    ledger = Ledger()
    assert plannable_categories(subject, ledger) == documentable_categories(subject)

    for category in documentable_categories(subject):
        ledger.record(subject.persona_id, category, Decimal("100000"))
    assert plannable_categories(subject, ledger) == []
    assert documentable_categories(subject), "templates did not change"


def test_the_planner_will_not_plan_without_a_ledger():
    """Required, not optional. A caller who omitted it used to get a silent full balance
    and a claim the oracle then refused, with the failure landing a stage from its cause.
    A caller with no history passes `Ledger()` and says so."""
    with pytest.raises(TypeError):
        plan_claim(random.Random(1), persona=persona(), claim_id="c1")


def test_every_verdict_is_either_realizable_or_named_unrealizable():
    """The enum hole. `unrealizable_verdicts` used to iterate `verdict_mix`, so a verdict
    added to `Verdict` but absent from the mix belonged to neither list: no test would
    reach it, and the planner would raise `KeyError` out of its reasons table instead of a
    `NotImplementedError` saying what the verdict needs."""
    assert set(REALIZABLE_VERDICTS) | set(unrealizable_verdicts()) == set(Verdict)
    assert not set(REALIZABLE_VERDICTS) & set(unrealizable_verdicts())


def test_a_verdict_with_no_recorded_reason_still_fails_cleanly():
    """The fallback that keeps the promise above. Simulated by asking for a realizable
    verdict with its reason removed, because there is no sixth enum member to add."""
    from receipt_synth import claim_planner

    reasons = dict(claim_planner._UNREALIZABLE_REASONS)
    reasons.pop(Verdict.PARTIALLY_PAID)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "_UNREALIZABLE_REASONS", reasons)
        with pytest.raises(NotImplementedError, match="no mechanism registered"):
            plan_claim(
                random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
                verdict=Verdict.PARTIALLY_PAID,
            )


# --------------------------------------------------------------- degradation --


def image_and_boxes():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 160, 3), dtype=np.uint8)
    boxes = {"total": (10.0, 20.0, 40.0, 12.0), "date": (60.0, 150.0, 50.0, 12.0)}
    return image, boxes


def test_degradation_is_deterministic_under_seed():
    image, boxes = image_and_boxes()
    first = degrade(image, boxes, seed=7)
    second = degrade(image, boxes, seed=7)

    assert np.array_equal(first.image, second.image)
    assert first.field_bboxes == second.field_bboxes


def test_degradation_differs_between_seeds():
    image, boxes = image_and_boxes()
    assert not np.array_equal(degrade(image, boxes, seed=7).image,
                              degrade(image, boxes, seed=8).image)


def test_degradation_changes_the_image():
    image, boxes = image_and_boxes()
    assert not np.array_equal(degrade(image, boxes, seed=7).image, image)


def test_degradation_accepts_a_seed_wider_than_a_c_int():
    """Augraphy seeds OpenCV through `cv2.setRNGSeed`, which takes a signed C int, while
    the pipeline derives seeds from a 64-bit generator. The fold happens in the degrader
    so no caller has to know."""
    image, boxes = image_and_boxes()
    assert degrade(image, boxes, seed=2**63 - 1).field_bboxes == boxes


def test_every_box_survives_degradation():
    """A dropped box would leave a document whose labels name a field the annotation no
    longer locates."""
    image, boxes = image_and_boxes()
    assert set(degrade(image, boxes, seed=7).field_bboxes) == set(boxes)


def test_boxes_stay_whole_pixels_through_degradation():
    """Albumentations normalizes coordinates to [0, 1] and back, which turns 26 into
    25.999999217689037. A box indexes an image, so the rounding the renderer applied has
    to survive the trip."""
    image, boxes = image_and_boxes()
    for box in degrade(image, boxes, seed=7).field_bboxes.values():
        assert all(float(value).is_integer() for value in box)


def test_non_geometric_degradation_leaves_boxes_where_they_were():
    image, boxes = image_and_boxes()
    assert degrade(image, boxes, seed=7).field_bboxes == boxes


def test_uncalibrated_capture_modes_are_refused():
    image, boxes = image_and_boxes()
    for capture in (Capture.PHOTO, Capture.SCAN):
        with pytest.raises(NotImplementedError):
            degrade(image, boxes, seed=7, capture=capture)


# ------------------------------------------------------------- the whole run --


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    return generate_dataset(seed=SEED, out_dir=out), out


def test_run_produces_one_document_per_claim(dataset):
    result, _ = dataset
    assert len(result.personas) == 1
    assert len(result.claims) == 1
    assert len(result.documents) == 1


def test_images_and_labels_are_written(dataset):
    result, out = dataset
    document = result.documents[0]

    assert (out / "images" / document.source_file).is_file()
    assert (out / "labels" / f"{document.doc_id}.json").is_file()
    assert (out / "labels" / f"{result.claims[0].claim_id}.claim.json").is_file()
    assert (out / "ground_truth.json").is_file()


def test_the_claim_points_at_the_document_that_was_written(dataset):
    result, _ = dataset
    assert result.claims[0].documents == [result.documents[0].doc_id]
    assert result.claims[0].verdict is Verdict.COVERED
    assert result.documents[0].doc_type is DocType.FISCAL_RECEIPT


def test_labels_state_the_invariants_that_were_built(dataset):
    """The end-to-end statement: what the ground truth says is what content_builder
    guaranteed, carried unchanged through rendering and degradation."""
    result, _ = dataset
    document = result.documents[0]

    assert sum(item.qty * item.price for item in document.line_items) == document.amount
    assert all(item.covered for item in document.line_items)
    assert document.synthetic is True
    assert document.generator_version


def test_a_covered_claim_carries_the_policy_engines_answer(dataset):
    """The fields the engine derives, no longer unset. A fully covered claim reads 1.0,
    depends on the documents alone, and says why in words."""
    result, _ = dataset
    claim = result.claims[0]

    assert claim.covered_fraction == 1.0
    assert claim.reimbursable_amount == result.documents[0].amount
    assert claim.verdict_basis == [VerdictBasis.DOCUMENTS]
    assert claim.imperfection == []
    assert claim.policy_trace == [
        f"category={claim.category} ok",
        "period ok",
        "coverage 100% (all line items covered)",
    ]


# ---------------------------------------------------- many claims, one run --


@pytest.fixture(scope="module")
def multi_claim_dataset(tmp_path_factory):
    """Enough personas × claims that the cumulative-limit mechanism actually fires."""
    out = tmp_path_factory.mktemp("multi")
    return generate_dataset(seed=SEED, out_dir=out, personas=4, claims_per_persona=8), out


def test_a_persona_can_file_several_claims(multi_claim_dataset):
    result, _ = multi_claim_dataset
    assert len(result.claims) > len(result.personas)
    assert len(result.documents) == len(result.claims)


def test_the_single_claim_default_is_still_reachable(dataset):
    result, _ = dataset
    assert len(result.claims) == 1


def test_every_claim_of_a_persona_is_in_date_order(multi_claim_dataset):
    result, _ = multi_claim_dataset
    for persona_record in result.personas:
        dates = [
            document.date
            for claim, document in zip(result.claims, result.documents, strict=True)
            if claim.persona_id == persona_record.persona_id
        ]
        assert dates == sorted(dates)


def test_the_limit_flag_is_recomputable_from_the_dataset_alone(multi_claim_dataset):
    """The ledger, re-derived independently of the engine that produced the labels.

    Walk the claims of each persona in order, add up what each one is reimbursed —
    `min(covered on the document, what remains of the annual limit)` — and check that
    `limit_exhausted` is flagged on exactly the claims where that `min` bound, and that
    the `reimbursable_amount` written to the label file is that same number. This is the
    property the whole ledger exists to enforce, checked on what landed on disk.
    """
    from collections import defaultdict

    from receipt_synth.policy_engine import annual_limit, covered_total

    result, _ = multi_claim_dataset
    spent: dict[tuple[str, str], Decimal] = defaultdict(lambda: Decimal(0))

    for claim, document in zip(result.claims, result.documents, strict=True):
        key = (claim.persona_id, claim.category)
        limit = annual_limit(claim.category)
        remaining = limit - spent[key]
        covered = covered_total(claim.category, document.line_items)
        reimbursed = min(covered, remaining)

        assert ("limit_exhausted" in claim.imperfection) is (reimbursed < covered)
        assert claim.reimbursable_amount == reimbursed
        spent[key] += reimbursed
        assert spent[key] <= limit


def test_a_limit_exhausted_claim_occurs_and_says_it_depends_on_account_state(
    multi_claim_dataset,
):
    """The mechanism that cannot exist with one claim per persona. Its verdict_basis is
    the whole reason the distinction between the two causes exists: nobody can read a
    remaining annual balance off a receipt."""
    result, _ = multi_claim_dataset
    exhausted = [c for c in result.claims if "limit_exhausted" in c.imperfection]

    assert exhausted, "no claim exercised the cumulative limit"
    for claim in exhausted:
        assert claim.verdict is Verdict.PARTIALLY_COVERED
        assert VerdictBasis.ACCOUNT_STATE in claim.verdict_basis
        assert any("annual limit" in line for line in claim.policy_trace)


def test_a_mixed_items_claim_occurs_and_says_it_depends_on_the_documents_alone(
    multi_claim_dataset,
):
    result, _ = multi_claim_dataset
    mixed = [
        c for c in result.claims
        if c.imperfection == ["mixed_items"]
    ]

    assert mixed, "no claim carried a non-covered line"
    for claim in mixed:
        assert claim.verdict is Verdict.PARTIALLY_COVERED
        assert claim.verdict_basis == [VerdictBasis.DOCUMENTS]
        assert 0 < claim.covered_fraction < 1


def test_the_per_line_covered_flags_agree_with_the_claim_fraction(multi_claim_dataset):
    """The end-to-end statement for the oracle: the claim label a consumer reads is
    recomputable from the per-line labels in the same file."""
    result, _ = multi_claim_dataset
    for claim, document in zip(result.claims, result.documents, strict=True):
        # True for every claim, including the limit-bound ones: `covered_fraction` stays
        # a property of the line items, and the limit is recorded elsewhere in the label.
        covered = sum(
            Decimal(str(item.qty)) * Decimal(str(item.price))
            for item in document.line_items
            if item.covered
        )
        total = sum(
            Decimal(str(item.qty)) * Decimal(str(item.price)) for item in document.line_items
        )
        assert abs(float(covered / total) - claim.covered_fraction) < 1e-6


def test_the_dataset_labels_are_reproducible_from_the_documents_alone(multi_claim_dataset):
    """`evaluate_claims` is the batch entry point, and this is what keeps it honest.

    The assembler cannot use it — claim *k*'s documents are sized against the balance left
    by claims 1…k−1, so planning, building and evaluating have to interleave. That leaves
    two paths through the same rules, and two paths drift. So: take the finished run, hand
    every claim's documents to `evaluate_claims`, and require the labels that come back to
    be the ones the assembler wrote. It is also what a consumer does when it wants to
    check the dataset it was given.
    """
    from receipt_synth.policy_engine import ClaimInput, evaluate_claims

    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}
    inputs = [
        ClaimInput(
            claim_id=claim.claim_id,
            persona_id=claim.persona_id,
            category=claim.category,
            documents=tuple(by_id[doc_id] for doc_id in claim.documents),
        )
        for claim in result.claims
    ]

    for claim, evaluation in zip(result.claims, evaluate_claims(inputs), strict=True):
        assert evaluation.verdict is claim.verdict, claim.claim_id
        assert evaluation.fraction_as_label() == claim.covered_fraction, claim.claim_id
        assert evaluation.reimbursable == claim.reimbursable_amount, claim.claim_id
        assert list(evaluation.verdict_basis) == claim.verdict_basis, claim.claim_id
        assert list(evaluation.imperfection) == claim.imperfection, claim.claim_id
        assert list(evaluation.policy_trace) == claim.policy_trace, claim.claim_id


def test_the_balance_report_names_the_verdicts_it_could_not_generate(multi_claim_dataset):
    """A report that renormalized silently would print a tidy table over 70% of the target
    mix and look balanced. Naming the missing 30% is the point of the report at this
    stage."""
    result, _ = multi_claim_dataset
    report = balance_report(result)

    assert "NOT GENERATED IN THIS RUN" in report
    for verdict in unrealizable_verdicts():
        assert verdict.value in report
    assert "CONDITIONAL" in report
    assert "30.0% of the target mix is absent" in report
    for verdict in REALIZABLE_VERDICTS:
        assert verdict.value in report
    assert "mixed_items" in report and "limit_exhausted" in report


def test_a_verdict_with_no_share_is_named_and_does_not_enter_the_arithmetic(
    multi_claim_dataset,
):
    """`rejected` is in `verdict_mix` with no share yet, and the report has to survive that
    twice over.

    It must not print a percentage for it — there is none, and `0.0%` would read as a
    decision. And the 30% it reports as absent is now only what the THREE share-carrying
    unrealizable verdicts account for, so the report has to say that the figure is a lower
    bound. Silently summing a `None` as zero would leave the same 30% on the page as a
    complete answer.
    """
    result, _ = multi_claim_dataset
    report = balance_report(result)

    assert "rejected (no share declared yet)" in report
    assert "rejected (0.0%)" not in report
    assert "30.0% of the target mix is absent" in report
    assert "LOWER BOUND" in report
    assert "no share for rejected" in report


def test_the_cause_table_counts_claims_and_says_so(multi_claim_dataset):
    """policy.yaml splits the `partially_covered` bucket per claim, summing to 1.0.
    Counting occurrences puts a claim carrying both causes in two rows, and the printed
    shares can then never converge on 65/35 however long the run — a number that cannot
    reach its target reads as a miss rather than as a category error."""
    result, _ = multi_claim_dataset
    report = balance_report(result)

    partial = [c for c in result.claims if c.verdict is Verdict.PARTIALLY_COVERED]
    assert f"partially_covered by cause — {len(partial)} claim(s)" in report
    assert "per claim, not per occurrence" in report

    # The three rows partition the bucket: each claim is counted once.
    only_mixed = sum(1 for c in partial if c.imperfection == ["mixed_items"])
    only_limit = sum(1 for c in partial if c.imperfection == ["limit_exhausted"])
    both = sum(1 for c in partial if len(c.imperfection) == 2)
    assert only_mixed + only_limit + both == len(partial)
    assert "both causes on one claim" in report


def test_the_report_separates_a_binding_limit_from_a_builder_shortfall(multi_claim_dataset):
    """The two directions of target-vs-realized are different events. A `covered` target
    overruled by the ledger is the limit mechanism working; a `partially_covered` target
    that came out `covered` is a basket sized from an estimate that fell short. One
    counter for both would report a defect as a feature."""
    result, _ = multi_claim_dataset
    report = balance_report(result)
    pairs = list(zip(result.plans, result.claims, strict=True))

    overruled = [1 for p, c in pairs if p.verdict is Verdict.COVERED and p.verdict is not c.verdict]
    shortfall = [
        1 for p, c in pairs
        if p.verdict is Verdict.PARTIALLY_COVERED and c.verdict is Verdict.COVERED
    ]
    assert overruled or shortfall, "this run exercised neither direction"

    if overruled:
        assert "the oracle overruling" in report
    if shortfall:
        assert "builder" in report and "shortfall" in report


def test_the_report_states_how_many_claims_were_ordered_and_how_many_were_built(
    multi_claim_dataset,
):
    """The number a dataset size gets quoted from. Four personas × eight claims orders
    thirty-two; planning stops early once a persona's only documentable category has its
    limit spent, so fewer get built. Opening with the built count alone presents the
    shortfall as though it had never been asked for."""
    result, _ = multi_claim_dataset
    report = balance_report(result)

    ordered, built = result.claims_ordered, len(result.claims)
    skipped = sum(result.claims_skipped.values())

    assert ordered == 4 * 8
    assert built < ordered, "this run did not exercise the early stop"
    assert skipped == ordered - built, "the missing claims are not all accounted for"
    assert f"Claims — {ordered} ordered, {built} built, {skipped} not built" in report
    for reason in result.claims_skipped:
        assert reason in report


def test_every_skipped_claim_is_attributed_to_a_named_reason(multi_claim_dataset):
    """`UNATTRIBUTED` exists so that an unexplained shortfall says so instead of being
    folded into the likeliest bucket. It should be empty on a healthy run."""
    from receipt_synth.assembler import UNATTRIBUTED
    from receipt_synth.claim_planner import NO_ARCHETYPE, NO_BALANCE_LEFT

    result, _ = multi_claim_dataset
    assert set(result.claims_skipped) <= {NO_BALANCE_LEFT, NO_ARCHETYPE, UNATTRIBUTED}
    assert result.claims_skipped.get(UNATTRIBUTED, 0) == 0


def test_the_reason_a_persona_runs_out_is_the_one_reported(multi_claim_dataset):
    """Re-derived from the finished run rather than taken from the report: a persona that
    built fewer claims than were ordered must have no plannable category left, and the
    reason must be the spent-balance one rather than the missing-archetype one — every
    persona in a run holds a documentable category by construction."""
    from receipt_synth.claim_planner import NO_BALANCE_LEFT
    from receipt_synth.policy_engine import Ledger, annual_limit, covered_total

    result, _ = multi_claim_dataset
    for persona_record in result.personas:
        pairs = [
            (claim, document)
            for claim, document in zip(result.claims, result.documents, strict=True)
            if claim.persona_id == persona_record.persona_id
        ]
        if len(pairs) == 8:
            continue  # this persona built everything it was asked for

        ledger = Ledger()
        for claim, document in pairs:
            covered = covered_total(claim.category, document.line_items)
            remaining = ledger.remaining(claim.persona_id, claim.category)
            ledger.record(claim.persona_id, claim.category, min(covered, remaining))

        assert why_no_claim(persona_record, ledger) == NO_BALANCE_LEFT
        assert documentable_categories(persona_record)
        for category in documentable_categories(persona_record):
            assert ledger.spent(persona_record.persona_id, category) == annual_limit(category)


def _verdict_table(report: str) -> list[str]:
    """The indented rows between the "Verdict balance" header and whatever follows it."""
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Verdict balance"))
    rows = []
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        rows.append(line.strip())
    return rows


def test_the_verdict_table_accounts_for_every_claim_it_counts(multi_claim_dataset):
    """The rows have to sum to the header. Iterating `verdict_mix` and skipping the
    unrealizable verdicts meant that a claim the engine emitted as one would vanish from
    the table while still counting in the total, and nothing would say so."""
    result, _ = multi_claim_dataset
    report = balance_report(result)

    counted = {}
    for line in _verdict_table(report):
        name, count = line.split()[0], int(line.split()[1])
        counted[Verdict(name)] = count
    assert sum(counted.values()) == len(result.claims)
    assert set(counted) >= set(REALIZABLE_VERDICTS), "a realizable verdict has no row"
    assert "!!" not in report, "the report flagged a discrepancy in its own arithmetic"


def test_a_verdict_the_planner_cannot_draw_is_still_given_a_row():
    """Unreachable today — the planner draws only two verdicts and the engine emits no
    others. It is the silence that would be the defect, so the row is asserted directly on
    a hand-built Dataset rather than waited for."""
    from receipt_synth.assembler import Dataset

    claim = ClaimGroundTruth(
        claim_id="p001_c1", persona_id="p001", category="vitamins_nutrition",
        documents=["p001_c1_d1"], verdict=Verdict.INSUFFICIENT_EVIDENCE,
    )
    report = balance_report(
        Dataset(seed=1, personas=[], claims=[claim], documents=[], claims_ordered=1)
    )

    assert "insufficient_evidence" in report
    assert "realized but not realizable" in report


def test_a_dataset_without_plans_still_reports():
    """`Dataset.plans` defaults to empty. Raising on the comparison would make the report
    unavailable exactly when someone assembled a Dataset by hand to look at it."""
    from receipt_synth.assembler import Dataset

    report = balance_report(Dataset(seed=1, personas=[], claims=[], documents=[]))
    assert "comparison skipped" in report


def test_the_multi_claim_run_is_reproducible(tmp_path):
    first = generate_dataset(
        seed=SEED, out_dir=tmp_path / "a", personas=3, claims_per_persona=5
    )
    second = generate_dataset(
        seed=SEED, out_dir=tmp_path / "b", personas=3, claims_per_persona=5
    )
    assert first.as_manifest() == second.as_manifest()
    assert balance_report(first) == balance_report(second)


def test_at_least_one_claim_is_refused(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(seed=SEED, out_dir=tmp_path, claims_per_persona=0)


def test_manifest_is_valid_json_and_declares_its_provenance(dataset):
    _, out = dataset
    manifest = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    assert manifest["synthetic"] is True
    assert manifest["seed"] == SEED
    assert manifest["generator_version"]
    assert len(manifest["documents"]) == 1


def test_bboxes_in_the_written_labels_index_the_written_image(dataset):
    from PIL import Image

    result, out = dataset
    document = result.documents[0]
    with Image.open(out / "images" / document.source_file) as image:
        width, height = image.size

    assert document.field_bboxes
    for name, (x, y, box_width, box_height) in document.field_bboxes.items():
        assert x >= 0 and y >= 0, f"{name} starts outside the image"
        assert x + box_width <= width, f"{name} runs past the right edge"
        assert y + box_height <= height, f"{name} runs past the bottom edge"


def test_the_same_seed_reproduces_the_dataset(tmp_path):
    """The promise the repository is built on: it ships the generator and the seed, not
    the data."""
    first = generate_dataset(seed=SEED, out_dir=tmp_path / "a")
    second = generate_dataset(seed=SEED, out_dir=tmp_path / "b")

    assert first.as_manifest() == second.as_manifest()
    name = first.documents[0].source_file
    assert (tmp_path / "a" / "images" / name).read_bytes() == (
        tmp_path / "b" / "images" / name
    ).read_bytes()


def test_a_different_seed_produces_a_different_dataset(tmp_path):
    other = generate_dataset(seed=SEED + 1, out_dir=tmp_path / "c")
    assert other.documents[0].amount != 0


def test_line_items_do_not_repeat_a_printed_name(tmp_path):
    """A cash register lists an article once and states how many. The same name twice at
    two prices is not a receipt."""
    for seed in range(SEED, SEED + 5):
        result = generate_dataset(seed=seed, out_dir=tmp_path / f"s{seed}")
        names = [item.name for item in result.documents[0].line_items]
        assert len(names) == len(set(names))


def test_cli_runs_and_reports(tmp_path, capsys):
    assert main(["--seed", str(SEED), "--out", str(tmp_path)]) == 0
    assert "receipt-synth" in capsys.readouterr().out
    assert (tmp_path / "ground_truth.json").is_file()


def test_cli_requires_a_seed(capsys):
    """A default seed would let a run look reproducible without anyone having recorded
    what to reproduce it with."""
    with pytest.raises(SystemExit):
        main(["--out", "out"])
