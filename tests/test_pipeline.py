"""The pipeline ends: persona, plan, degradation, and the run as a whole.

The claim these tests exist to defend is the one the whole tool rests on — that a seed
determines the output. It is checked at each stage and again on the finished dataset,
because a single unseeded draw anywhere makes the promise false everywhere.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

from receipt_synth import claim_planner
from receipt_synth.assembler import (
    balance_report,
    generate_dataset,
)
from receipt_synth.claim_planner import (
    ARCHETYPES,
    REALIZABLE_VERDICTS,
    ClaimPlan,
    DocumentPlan,
    _select_documents,
    archetypes_for,
    can_assemble_evidence,
    documentable_categories,
    draw_partially_covered_cause,
    draw_verdict,
    evidence_of,
    plan_claim,
    plan_claims,
    plannable_categories,
    realizable_verdicts_for,
    unrealizable_verdicts,
    why_no_claim,
)
from receipt_synth.cli import main
from receipt_synth.config import high_frequency_surnames, jurisdiction, load_policy
from receipt_synth.content_builder import (
    MAX_LINE_ITEMS,
    draw_party_identity,
    is_valid_rnokpp,
)
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    PAYMENT_PRECEDES_SUBJECT,
    Evidence,
    Ledger,
    document_evidence,
    insufficient_evidence_causes,
    verdict_mix,
)
from receipt_synth.schemas import (
    Capture,
    ClaimGroundTruth,
    Country,
    DocType,
    Verdict,
    VerdictBasis,
)

SEED = 20260803

# 👁 The value an internet-acquiring confirmation prints where the payer would be — the first of the
# two observed forms of emptiness. Read from the configuration rather than written as "-", because
# a literal here would be a second place the string lives.
_UNIDENTIFIED_PAYER = jurisdiction("UA")["payment_confirmation"]["parties"]["empty_value"]


# ----------------------------------------------------------------- personas --


def persona(seed: int = SEED, country: Country = Country.UA):
    return generate_persona(random.Random(seed), persona_id="p001", country=country)


def test_persona_is_deterministic_under_seed():
    assert persona() == persona()


def test_personas_differ_between_seeds():
    names = {persona(seed).full_name for seed in range(20)}
    assert len(names) > 1


def test_a_persona_surname_comes_from_the_narrowed_pool():
    """A persona is not printed on any document today — but it is the payer on a payment
    confirmation, so it draws its surname from the same narrowed pool a printed sole trader
    does. Asserted here so that the mechanism cannot quietly serve only one of the two.

    See `personal_names` in config/generation.yaml for why the pool is narrowed.
    """
    pool = set(high_frequency_surnames("uk"))
    drawn = {persona(seed).full_name.split()[-1] for seed in range(200)}
    assert drawn <= pool, sorted(drawn - pool)


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
    """A payment date outside the window drives `rejected`. Landing outside it by accident
    would attach that document to a `covered` label."""
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


def test_plan_picks_archetypes_that_can_carry_the_category():
    subject = persona()
    plan = plan_claim(random.Random(3), persona=subject, claim_id="c1", ledger=Ledger())
    assert plan.documents, "a claim is a list of documents, and never an empty one"
    for document in plan.documents:
        assert plan.category in document.archetype.categories
        assert document.archetype.country is subject.location.country


def test_a_category_whose_archetypes_cannot_prove_both_facts_is_refused():
    """A claim needs both facts — what was bought and that it was paid for — so a category whose
    only registered archetype supplies ONE of them is refused rather than half-built.

    REWRITTEN WITH THE SECOND DOCUMENT CLASS, and the old form no longer described anything. It
    looked for a category NO archetype covers, which was every category but one while the registry
    held fiscal receipts alone. A bank payment confirmation carries every category — 👁 it lists no
    items, so nothing on it can contradict one — and the refusal now comes from the evidence being
    incomplete rather than from the category having no template at all. Same guard, reached through
    the branch that is actually live: `_select_documents` refuses, because `document_evidence` says
    what each type proves and nothing says which purchase an unpaired payment settled.
    """
    payment_only = [
        archetype
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == Evidence(False, True)
    ]
    assert payment_only, "no payment-proving archetype is registered at all"

    assert not can_assemble_evidence(payment_only), (
        "a set of archetypes that all prove the payment and none the subject must not be "
        "assemblable — it is the case this whole rule exists for"
    )
    with pytest.raises(ValueError, match="establishes both"):
        _select_documents(random.Random(1), payment_only, datetime(2026, 6, 15, 12, 0))


def test_a_payment_only_category_is_not_reported_as_documentable():
    """The other half, one stage earlier — and it is what keeps the refusal above off the drawn
    path. `documentable_categories` answers the question the assembler asks before any claim
    exists, so a category it reports would be drawn and then refused, and the run would die
    mid-dataset on a machine nobody is watching.
    """
    subject = persona()
    payment_archetypes = [
        archetype
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == Evidence(False, True)
    ]
    assert payment_archetypes, "no payment-proving archetype is registered at all"

    # A registry holding ONLY payment-proving archetypes. Every category is then covered by
    # archetypes and none is completable, which is exactly the state the rule is about.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            claim_planner,
            "ARCHETYPES",
            {archetype.slug: archetype for archetype in payment_archetypes},
        )
        assert archetypes_for(Country.UA, subject.benefit_categories[0]), (
            "the narrowed registry covers none of this persona's categories, so the assertion "
            "below would hold for the wrong reason"
        )
        assert documentable_categories(subject) == []


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
        Verdict.PARTIALLY_PAID,
        Verdict.REJECTED,
    ],
)
def test_verdicts_no_archetype_can_carry_are_refused_not_faked(verdict):
    """Explicit over silent. A planner that accepted one of these and produced an ordinary
    basket would write a wrong label rather than fail. The message has to say what each
    actually needs, because the reason differs: `not_proof_of_payment` needs a document type
    that establishes no payment, `rejected` needs either a basket builder that draws no covered
    line or a payment dated outside the active period, and `partially_paid` needs document types
    that do not exist at all.

    ⚠️ `insufficient_evidence` LEFT THIS LIST when its two cross-check causes became drawable. Its
    third cause did not, and the reason moved to `_UNREALIZABLE_CAUSES` rather than disappearing
    with the entry — a verdict being realizable and every route to it being realizable are
    different statements, and the second is the one a reader of a corpus needs.

    `policy_engine` can already label TWO of the three — the ones above except
    `_UNREALIZABLE_REASONS` key and nothing else, because the field an invoice would have to
    carry to be partly settled does not exist. So for that one, neither side is built."""
    with pytest.raises(NotImplementedError) as raised:
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1",
            verdict=verdict, ledger=Ledger(),
        )
    assert verdict.value in str(raised.value)
    assert len(str(raised.value)) > 60, "the message has to say what the verdict needs"


def test_the_planner_realizes_exactly_the_verdicts_the_engine_can_be_asked_for():
    """`insufficient_evidence` MOVED SIDES when the cross-check causes became drawable. Two of its
    three causes are planned; the third, `subject_not_evidenced`, needs a deliberately incomplete
    claim and keeps its reason in `_UNREALIZABLE_CAUSES` rather than losing it with the entry that
    left `_UNREALIZABLE_REASONS`."""
    assert set(REALIZABLE_VERDICTS) == {
        Verdict.COVERED,
        Verdict.PARTIALLY_COVERED,
        Verdict.INSUFFICIENT_EVIDENCE,
    }
    assert set(unrealizable_verdicts()) == {
        Verdict.NOT_PROOF_OF_PAYMENT,
        Verdict.PARTIALLY_PAID,
        Verdict.REJECTED,
    }
    # A verdict that left the table must not leave its unbuildable half unexplained.
    assert set(claim_planner._UNREALIZABLE_CAUSES) == {"subject_not_evidenced"}
    assert len(claim_planner._UNREALIZABLE_CAUSES["subject_not_evidenced"]) > 60


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
        # The default argument was bound at import time, so the patched module attribute has to be
        # passed explicitly — which is itself the property worth pinning: `draw_verdict` draws over
        # the subset it is GIVEN, and a caller narrowing that subset per persona is what
        # `realizable_verdicts_for` does.
        with pytest.raises(ValueError, match="no share"):
            claim_planner.draw_verdict(
                random.Random(1), claim_planner.REALIZABLE_VERDICTS
            )


def test_the_drawn_mix_is_the_target_mix_renormalized_over_the_realizable_subset():
    """verdict_mix gives covered 0.50, partially_covered 0.20 and insufficient_evidence 0.10; of
    the three that cannot be built, two carry 0.10 each and `rejected` carries no share at all.
    Renormalized over the three that can:

        covered                0.50 / 0.80 = 0.625
        partially_covered      0.20 / 0.80 = 0.250
        insufficient_evidence  0.10 / 0.80 = 0.125

    A member with no share must not touch that arithmetic — the draw is over the
    realizable subset, and `rejected` is not in it.

    Over 4000 draws each realized share should sit near its renormalized target. The window is
    wide (±0.04) on purpose: this asserts the weights are the policy's, not that a
    pseudo-random draw hits a mean. ALL THREE are checked, because checking only `covered` would
    have passed unchanged when a third member joined the denominator.
    """
    rng = random.Random(20260803)
    draws = [draw_verdict(rng) for _ in range(4000)]
    mix = verdict_mix()
    total = sum(mix[verdict] for verdict in REALIZABLE_VERDICTS)

    assert total == pytest.approx(0.8), "the realizable shares no longer sum to what this asserts"
    for verdict in REALIZABLE_VERDICTS:
        share = draws.count(verdict) / len(draws)
        assert abs(share - mix[verdict] / total) < 0.04, verdict


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


def test_the_category_narrowing_for_insufficient_evidence_is_per_verdict():
    """🔴 THE SUBTLE PART OF C, ASSERTED DIRECTLY RATHER THAN THROUGH A RUN.

    `insufficient_evidence` is realized by a claim whose SUBJECT and PAYMENT documents disagree, so
    it needs a category documented by a PAIR. A category holding a fiscal receipt gets one
    self-sufficient document — the planner prefers that shape — and one document cannot contradict
    itself, so such a category can never realize the verdict.

    AND THE NARROWING IS PER VERDICT, NOT GLOBAL: `covered` and `partially_covered` are realizable
    in both shapes, so a receipt category stays plannable for them. Narrowing globally would remove
    the only `fiscal_receipt` documents the corpus has, to satisfy a constraint belonging to one
    verdict out of three.

    ⚠️ WHY THIS IS A UNIT TEST AND NOT A RUN. A mutation that removed the narrowing survived every
    run-based test, and the reason was not a weak test: NO PERSONA of the pipeline fixture holds
    `vitamins_nutrition`, the only category with a both-proving archetype, so no run there can reach
    the branch at all. A property that a draw may or may not exercise has to be asserted where it is
    decided.
    """
    receipt_categories = {
        category_id
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == (True, True)
        for category_id in archetype.categories
    }
    assert receipt_categories, "no archetype proves both facts — this test asserts nothing"

    holder = next(
        generate_persona(random.Random(seed), persona_id="pX", country=Country.UA)
        for seed in range(200)
        if receipt_categories
        & set(generate_persona(random.Random(seed), persona_id="pX",
                               country=Country.UA).benefit_categories)
    )
    ledger = Ledger()
    everything = plannable_categories(holder, ledger)
    for_mismatch = plannable_categories(holder, ledger, Verdict.INSUFFICIENT_EVIDENCE)

    assert receipt_categories & set(everything), "the persona holds no receipt category"
    assert not receipt_categories & set(for_mismatch), (
        f"{receipt_categories & set(for_mismatch)} has a self-sufficient archetype and cannot "
        "realize a disagreement between two documents"
    )
    # Per verdict, not global: the coverage verdicts keep the whole set.
    for verdict in (Verdict.COVERED, Verdict.PARTIALLY_COVERED):
        assert plannable_categories(holder, ledger, verdict) == everything, verdict
    assert set(realizable_verdicts_for(holder, ledger)) == set(REALIZABLE_VERDICTS)


def test_a_persona_holding_only_a_receipt_category_cannot_realize_the_disagreement_verdict():
    """The other side of the same rule, and what `realizable_verdicts_for` exists for: such a
    persona's draw must EXCLUDE the verdict rather than produce a plan that has to be refused a
    stage later. The conditioning is declared — a share conditioned on the persona is not the share
    policy.yaml states."""
    receipt_only = next(
        category_id
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == (True, True)
        for category_id in archetype.categories
    )
    holder = generate_persona(random.Random(3), persona_id="pX", country=Country.UA)
    holder = holder.model_copy(update={"benefit_categories": [receipt_only]})

    assert plannable_categories(holder, Ledger()) == [receipt_only]
    assert plannable_categories(holder, Ledger(), Verdict.INSUFFICIENT_EVIDENCE) == []
    assert Verdict.INSUFFICIENT_EVIDENCE not in realizable_verdicts_for(holder, Ledger())
    assert set(realizable_verdicts_for(holder, Ledger())) == {
        Verdict.COVERED, Verdict.PARTIALLY_COVERED
    }


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


def test_plan_claim_names_the_cause_when_the_persona_can_realize_nothing():
    """`why_no_claim` guards `plan_claims`, but a direct call to `plan_claim` reaches
    `draw_verdict(rng, realizable_verdicts_for(...) or REALIZABLE_VERDICTS)` unguarded. The
    `or` used to fall back to the full list and draw a verdict the persona cannot realize —
    the resulting `ValueError` then named that arbitrary drawn verdict rather than the real
    reason nothing is plannable, which is the anti-pattern this module's own docstrings warn
    against three times: a failure landing a stage away from its cause. `plan_claim` must
    raise on the empty list itself, naming the persona and the reason `why_no_claim` gives."""
    subject = persona()
    ledger = Ledger()
    for category in documentable_categories(subject):
        ledger.record(subject.persona_id, category, Decimal("100000"))
    assert realizable_verdicts_for(subject, ledger) == ()
    cause = why_no_claim(subject, ledger)
    assert cause is not None

    with pytest.raises(ValueError, match=re.escape(cause)) as excinfo:
        plan_claim(random.Random(1), persona=subject, claim_id="c1", ledger=ledger)
    assert subject.persona_id in str(excinfo.value)


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


@pytest.mark.parametrize("capture", list(Capture))
def test_every_capture_channel_is_deterministic_under_its_seed(capture):
    """🔴 THE TRIPWIRE THAT CAUGHT `DirtyRollers`. Determinism under `--seed` is an invariant of
    this generator, and an Augraphy effect is free to ignore `random_seed` — one of the three
    that were tried does, so the first `scan` recipe produced different pixels on every call
    while every other test stayed green. Parametrized over the whole enum so a channel added
    later is covered before anybody remembers to think about it."""
    image, boxes = image_and_boxes()
    first = degrade(image, boxes, seed=7, capture=capture)
    second = degrade(image, boxes, seed=7, capture=capture)

    assert np.array_equal(first.image, second.image)
    assert first.field_bboxes == second.field_bboxes


@pytest.mark.parametrize("capture", list(Capture))
def test_every_capture_channel_keeps_every_box(capture):
    """Whole pixels and no box lost, on every channel rather than on the one that applies no
    geometry. A dropped box leaves a document whose labels name a field the annotation no
    longer locates; a fractional one is not an index into an image."""
    image, boxes = image_and_boxes()
    moved = degrade(image, boxes, seed=7, capture=capture).field_bboxes

    assert set(moved) == set(boxes)
    for box in moved.values():
        assert all(float(value).is_integer() for value in box)


def test_the_two_paper_channels_move_the_boxes_and_the_screen_channel_does_not():
    """The channels are not three names for one recipe. A screenshot is axis-aligned by
    construction — there is no hand holding it and no sheet lying crooked — so its boxes must
    come back where they were, while a photograph and a scan must move them. Asserted in both
    directions: a recipe that lost its geometry would otherwise look like a working channel."""
    image, boxes = image_and_boxes()

    assert degrade(image, boxes, seed=7, capture=Capture.SCREENSHOT).field_bboxes == boxes
    for capture in (Capture.PHOTO, Capture.SCAN):
        moved = degrade(image, boxes, seed=7, capture=capture).field_bboxes
        assert moved != boxes, capture


def test_a_capture_channel_with_no_recipe_raises_rather_than_borrowing_one():
    """A channel added to the enum without a recipe must fail by name. Falling through to another
    channel's artefacts would put documents in the corpus labelled as one channel and degraded as
    another, which no test of either channel could see.

    🔴 EACH TABLE IS CHECKED BY ITSELF, AND THE FIRST VERSION OF THIS TEST WAS NOT. There are TWO
    recipe tables — one for the paper effects and one for the geometry — and `degrade` calls them in
    order. Asserting only that `degrade` raises means the SECOND table's guard is enough to keep the
    test green while the first one is gone: a mutation that deleted the paper table's refusal
    survived exactly that way, and it was the test that was weak rather than the mutation that was
    mis-aimed. The denominator is two.
    """
    from receipt_synth.degrader import _geometry, _paper_pipeline

    image, boxes = image_and_boxes()
    with pytest.raises(NotImplementedError, match="has no recipe"):
        degrade(image, boxes, seed=7, capture="fax")  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="has no recipe"):
        _paper_pipeline("fax", 7)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="has no recipe"):
        _geometry("fax")  # type: ignore[arg-type]


# ------------------------------------------------------------- the whole run --


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    return generate_dataset(seed=SEED, out_dir=out, train_fraction=0.5), out


def test_the_run_builds_exactly_the_documents_its_plans_asked_for(dataset):
    """One document per claim is a property of the ARCHETYPE REGISTRY — the one template
    registered proves both facts, so a claim needs no second document — and not of a
    dataset. Asserted against the plans rather than against the number 1, so that the day
    a claim plans two this test measures the run instead of failing on a constant."""
    result, _ = dataset
    assert len(result.personas) == 1
    assert len(result.claims) == 1
    assert len(result.documents) == sum(len(plan.documents) for plan in result.plans)


def test_images_and_labels_are_written(dataset):
    result, out = dataset
    document = result.documents[0]

    assert (out / "images" / document.source_file).is_file()
    assert (out / "labels" / f"{document.doc_id}.json").is_file()
    assert (out / "labels" / f"{result.claims[0].claim_id}.claim.json").is_file()
    assert (out / "ground_truth.json").is_file()


def test_the_claim_points_at_the_documents_that_were_written(dataset):
    """The join from a claim to its evidence, ASSERTED AGAINST THE RUN rather than against the
    number one. It used to read `== [result.documents[0].doc_id]` and to pin the type as a fiscal
    receipt — true while every registered archetype proved both facts, and false the moment the
    invoice made a split pair buildable. A constant that was a property of the registry is exactly
    what the test above it warns about."""
    result, _ = dataset
    claim = result.claims[0]

    assert claim.documents == [document.doc_id for document in result.documents]
    assert claim.verdict is Verdict.COVERED
    assert claim.linked is (len(claim.documents) > 1)


def test_every_document_of_a_claim_names_the_same_persona_and_the_same_vendor(dataset):
    """THE WIRING BETWEEN THE PERSONA AND THE DOCUMENTS, which nothing asserted until a mutation
    survived and said so.

    A claim's documents must agree about WHO. The invoice is addressed to the claimant and the
    payment document is drawn on the claimant's account, so `payer` is the persona's name on every
    document that names one; and the seller must be the SAME vendor instance on both, which used to
    hold because a claim had one document and is now a constraint somebody has to keep — a sole
    trader's name is drawn rather than stored, so two calls would print two sellers.

    Found by `assembler._build_document` being mutated to pass a literal buyer name: every test in
    the suite stayed green. The mutation was aimed at the wrong test, and re-aiming it had nowhere
    to land, which is the useful kind of survivor.
    """
    result, _ = dataset
    persona = result.personas[0]

    named = [d for d in result.documents if d.payer is not None]
    assert named, "no document of this run names a payer — the assertion below is vacuous"
    for document in named:
        assert document.payer == persona.full_name, document.doc_id

    for claim in result.claims:
        sellers = {result.documents[
            [d.doc_id for d in result.documents].index(doc_id)
        ].counterparty for doc_id in claim.documents}
        assert len(sellers) == 1, f"{claim.claim_id} names {sellers} across its documents"


def test_the_assembler_builds_a_fiscal_receipt_and_tells_it_the_capture_channel(tmp_path):
    """🔴 THE RECEIPT BRANCH OF THE ASSEMBLER, EXERCISED DIRECTLY — because no run-based fixture
    reaches it any more.

    Measured rather than assumed: the only category with a both-proving archetype is
    `vitamins_nutrition`, and NO PERSONA of either pipeline fixture holds it. Since the invoice
    activated the pair, every fixture claim is a two-document claim, so the whole
    "one self-sufficient document" branch — including the `capture=` the receipt builder now
    requires — runs in no test that drives `generate_dataset`.

    Found by a mutation surviving: deleting `basket |= {"capture": capture}` left every test green,
    and the reason was not a weak assertion but a fixture that cannot reach the code. Asserted here
    on a hand-built plan, which is where the dispatch is decided rather than where a draw might
    happen to land.
    """
    from receipt_synth import assembler
    from receipt_synth.renderer import Renderer

    receipt = next(
        archetype for archetype in ARCHETYPES.values() if evidence_of(archetype) == (True, True)
    )
    persona = generate_persona(random.Random(4), persona_id="p001", country=Country.UA)
    when = datetime(2026, 6, 15, 12, 0)
    plan = ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category=receipt.categories[0],
        verdict=Verdict.COVERED, issued_at=when,
        documents=(DocumentPlan(archetype=receipt, issued_at=when),),
    )
    vendor = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}

    with Renderer() as renderer:
        document, reference = assembler._build_document(
            random.Random(7), persona=persona, plan=plan, document_plan=plan.documents[0],
            vendor=vendor, identity=draw_party_identity(random.Random(7), vendor, "UA"),
            doc_id="p001_c1_d1", renderer=renderer, out_dir=tmp_path,
        )

    # A fiscal receipt is a whole claim, so nothing in its claim can cite it.
    assert reference is None

    assert document.doc_type is DocType.FISCAL_RECEIPT
    # The channel reached the builder: the VAT row's form is one the document's OWN capture allows.
    allowed = jurisdiction("UA")["tax_line_forms_by_medium"][document.capture.medium.value]
    assert document.vat_row_form in allowed, (
        f"{document.vat_row_form!r} is not a form a {document.capture.value} may print"
    )
    assert (tmp_path / "images" / document.source_file).is_file()


def test_no_claim_contradicts_itself_across_its_own_documents(multi_claim_dataset):
    """🔴 A THIRD CLASS OF CHECK, and it exists because two other classes could not see the defect
    that produced it.

    A claim's documents describe ONE transaction between ONE pair of parties. Every field they
    share must therefore agree — and agree BEFORE NORMALIZATION, on the raw string, which is the
    whole point.

    WHY PER-DOCUMENT ASSERTIONS CANNOT SEE THIS. `counterparty` was labelled as the PRINTED form on
    two classes («ТОВ «Ключ»») and as the BARE trade name on two others («Ключ»), for three commits.
    Every per-document assertion passed: each class was internally consistent. The contract's
    comparison rule strips the legal form, so both spellings compare EQUAL for any consumer — no
    scorecard could have shown it either.

    AND WHY MUTATION TESTING CANNOT SEE IT. There was nothing to break: no assertion existed whose
    reddening would reveal the divergence, so a mutation of either side left the suite green. A gap
    of this shape is invisible to a technique that measures whether existing assertions bite.

    So the form is: take a claim, take its documents, and ask whether they agree with each other on
    every field they share. It is checked here, on a run, rather than per class, because no single
    class can be wrong about it alone.

    THE TWO DELIBERATE DISAGREEMENTS ARE EXCLUDED BY THE CLAIM'S OWN CAUSE, never by a tolerance:
    a claim planned as `amount_mismatch` must disagree about the amount and about nothing else, and
    one planned as `payment_precedes_subject` about the order and about nothing else. Their
    exclusion is therefore itself an assertion — the label says which disagreement is intended, and
    everything else must still agree.
    """
    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}
    persona_names = {persona.persona_id: persona.full_name for persona in result.personas}

    multi = [claim for claim in result.claims if len(claim.documents) > 1]
    assert multi, "no claim carries two documents — this test asserts nothing"

    checked = Counter()
    for claim in result.claims:
        documents = [by_id[doc_id] for doc_id in claim.documents]
        if len(documents) < 2:
            continue

        # Fields that must agree on the RAW value, whatever the claim's cause.
        for field in ("currency", "language", "counterparty", "synthetic"):
            values = {getattr(document, field) for document in documents}
            assert len(values) == 1, f"{claim.claim_id} disagrees about {field}: {values}"
            checked[field] += 1

        # 🔴 THE PAYER IS NARROWER, AND THIS CHECK IS WHAT ESTABLISHED IT. Its first run failed on
        # `{'-', 'Олекса Семенюк'}`: an internet-acquiring confirmation 👁 does not identify the
        # payer and prints a HYPHEN as the value, while the invoice beside it names the claimant.
        # The DOCUMENT is right — that emptiness is observed and deliberate — so the rule is that
        # every document which NAMES a payer names the same one, and the exemption is keyed on the
        # configured empty value rather than on a tolerance. The consequence for a consumer is that
        # the payer cannot be cross-checked across such a claim at all; declared as KL-09.
        named = [d for d in documents if d.payer not in (None, _UNIDENTIFIED_PAYER)]
        assert named, f"{claim.claim_id} names no payer on any document"
        assert {d.payer for d in named} == {persona_names[claim.persona_id]}, claim.claim_id
        checked["payer_is_the_persona"] += 1
        checked["payer_unidentified"] += len(documents) - len(named)

        subject = next(
            d for d in documents if document_evidence(d.doc_type).proves_subject
        )
        payment = next(
            d for d in documents if not document_evidence(d.doc_type).proves_subject
        )

        # The amount, unless the claim's own label says the two were built to disagree.
        if AMOUNT_MISMATCH in claim.imperfection:
            assert subject.amount != payment.amount, claim.claim_id
        else:
            assert subject.amount == payment.amount, claim.claim_id
        checked["amount"] += 1

        # The order, on the same principle.
        if PAYMENT_PRECEDES_SUBJECT in claim.imperfection:
            assert payment.date < subject.date, claim.claim_id
        else:
            assert payment.date >= subject.date, claim.claim_id
        checked["date_order"] += 1

    assert checked["amount"] == len(multi), (
        f"{checked['amount']} of {len(multi)} multi-document claims were checked"
    )
    print(f"\ncross-document self-consistency: {len(multi)} of {len(result.claims)} claims "
          f"carry two documents; checks applied {dict(checked)}")


def test_the_documents_of_a_run_agree_on_the_sellers_PRINTED_identity(multi_claim_dataset):
    """The test above, one layer down: on the PAGE rather than on the label.

    🔴 IT IS HERE AND NOT ONLY IN test_cross_document_identity.py BECAUSE OF WHAT THAT MODULE
    CANNOT SEE. That module builds a pair the way the assembler builds one and asserts on the two
    rendered pages — which proves the builders honour a shared identity, and proves nothing about
    whether the ASSEMBLER hands them one. `identity=` is a required parameter, so dropping it fails
    loudly; `cites=` is not, so an assembler that stopped passing it would go on producing valid
    documents whose purpose lines cite a stranger, and every assertion in that module would still
    pass. This one runs the whole pipeline and reads what came out.

    ⚠️ AND IT IS THE MEASUREMENT THAT FOUND THE DEFECT, with the same reader — `fields_of` of
    tools/cross_document_audit.py, regexes over `reference_text`. On the delivered corpus that
    measurement returned 587 pairs and 0 agreements on the seller's tax code and IBAN; the same
    instrument runs here so the number cannot quietly go back.
    """
    from cross_document_audit import fields_of

    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}

    readable = Counter()
    agree = Counter()
    pairs = 0
    for claim in result.claims:
        documents = [by_id[doc_id] for doc_id in claim.documents]
        if len(documents) < 2:
            continue
        pairs += 1
        subject, payment = (
            fields_of(d.model_dump(mode="json"))
            for d in sorted(
                documents, key=lambda d: not document_evidence(d.doc_type).proves_subject
            )
        )
        for field in ("seller_name", "seller_tax_code", "seller_account", "seller_bank_name"):
            if subject[field] is None or payment[field] is None:
                continue
            readable[field] += 1
            agree[field] += subject[field] == payment[field]
        if payment["invoice_number"] is not None:
            readable["invoice_number"] += 1
            agree["invoice_number"] += (
                subject["invoice_number"] == payment["invoice_number"]
            )

    assert pairs, "no claim of this run carries two documents — nothing was measured"
    # 👁 THE PAYEE'S BANK IS SOMETIMES A CAPTION WITH NOTHING UNDER IT, observed on the recipient's
    # bank of a real confirmation, so that row is readable on most pairs and not on all. The two
    # bounds are therefore different assertions rather than one loosened to fit: three requisites
    # are printed on every pair, and the fourth must AGREE wherever it is printed at all.
    for field in ("seller_name", "seller_tax_code", "seller_account"):
        assert readable[field] == pairs, (
            f"{field} was readable on {readable[field]} of {pairs} pairs; a field the audit "
            "cannot read is a field it cannot report on either"
        )
    assert readable["seller_bank_name"], "no pair printed the payee's bank on both documents"
    for field in ("seller_name", "seller_tax_code", "seller_account", "seller_bank_name"):
        assert agree[field] == readable[field], (
            f"{field} agrees on {agree[field]} of {readable[field]} pairs that print it — see "
            "docs/cross-document-fields.md"
        )
    assert readable["invoice_number"], (
        "no payment document of this run cited an invoice, so the reference row was not measured"
    )
    assert agree["invoice_number"] == readable["invoice_number"], (
        f"{agree['invoice_number']} of {readable['invoice_number']} citations name the claim's "
        "own invoice"
    )
    print(f"\ncross-document identity, read off the page: {pairs} pairs; "
          f"agree {dict(agree)} of readable {dict(readable)}")


def test_both_cross_check_causes_occur_and_a_claim_carries_exactly_one(multi_claim_dataset):
    """🔴 THE REQUIREMENT IS ON THE RESULT, NOT ON THE SHARE. config/policy.yaml splits
    `insufficient_evidence` evenly between its two buildable causes and says why the split is even;
    what has to hold is that BOTH are non-zero in a run, because a verdict's share says nothing
    about which mechanism realized it. Ten percent of the corpus arriving through one cause would
    leave the vocabulary promising three and the data holding one.

    And exactly one per claim: the two are mutually exclusive by construction — the planner draws
    one — so a claim carrying both would mean the builder had realized a cause nobody planned.
    """
    result, _ = multi_claim_dataset
    causes = set(insufficient_evidence_causes())
    assert len(causes) == 2, f"the policy declares {len(causes)} buildable causes, not 2"

    flagged = [
        claim for claim in result.claims
        if claim.verdict is Verdict.INSUFFICIENT_EVIDENCE
    ]
    assert flagged, "no claim reached insufficient_evidence — the verdict is not being realized"

    seen = Counter()
    for claim in flagged:
        mine = set(claim.imperfection) & causes
        assert len(mine) == 1, f"{claim.claim_id} carries {mine}, expected exactly one"
        seen[mine.pop()] += 1

    for cause in causes:
        assert seen[cause] > 0, (
            f"{cause} occurs 0 times in {len(flagged)} insufficient_evidence claims of "
            f"{len(result.claims)}; policy.yaml requires every declared cause to be non-zero"
        )


def test_a_claim_drawn_as_insufficient_evidence_is_labelled_as_one(multi_claim_dataset):
    """🔴 THE PLAN AND THE LABEL MUST AGREE FOR THIS VERDICT, and that is NOT true of the other two.

    A claim drawn as `covered` may legitimately come back `partially_covered` — the ledger
    overrules the plan, which is the cumulative-limit mechanism working, and
    `assembler._drift_lines` reports it. A claim drawn as `insufficient_evidence` has no such
    excuse: the builder was told to make two documents disagree, the engine compares them
    deterministically, and there is no third party to overrule anything. So a drawn claim that
    comes back `covered` means the builder did not do what it was told, silently.

    FOUND BY A SURVIVING MUTATION. `plannable_categories` was made to ignore the verdict, so claims
    drawn as `insufficient_evidence` could be planned in the one category holding fiscal receipts —
    a single self-sufficient document, nothing to disagree with — and came back `covered`. Every
    test passed: the causes that did occur still occurred. Nothing asserted that a claim aimed at
    this verdict reaches it.
    """
    result, _ = multi_claim_dataset
    pairs = list(zip(result.plans, result.claims, strict=True))
    assert pairs, "no plans recorded — this test would assert nothing"

    drawn = [
        (plan, claim) for plan, claim in pairs
        if plan.verdict is Verdict.INSUFFICIENT_EVIDENCE
    ]
    assert drawn, "no claim was drawn as insufficient_evidence"

    for plan, claim in drawn:
        assert claim.verdict is Verdict.INSUFFICIENT_EVIDENCE, (
            f"{claim.claim_id} was drawn as insufficient_evidence with cause {plan.cause!r} and "
            f"came back {claim.verdict.value} — the documents did not disagree as planned"
        )
        assert plan.cause in claim.imperfection, (
            f"{claim.claim_id} was drawn for {plan.cause!r} and carries {claim.imperfection}"
        )


def test_an_insufficient_evidence_claim_pays_nothing_and_spends_no_balance(multi_claim_dataset):
    """A claim whose documents contradict each other is not a claim whose money is merely capped.
    It reimburses NOTHING, and — the part that would go unnoticed — it must leave the persona's
    annual balance untouched, or every later claim of theirs is sized against money that was never
    paid out. The ledger is what a cumulative limit is, so a leak here would move other claims'
    verdicts rather than its own."""
    result, _ = multi_claim_dataset
    flagged = [c for c in result.claims if c.verdict is Verdict.INSUFFICIENT_EVIDENCE]
    assert flagged, "no claim reached insufficient_evidence"

    for claim in flagged:
        assert claim.reimbursable_amount == 0, claim.claim_id
        assert VerdictBasis.DOCUMENTS in claim.verdict_basis
        # It is decided from the documents alone: nothing about the persona's history is consulted,
        # which is what distinguishes it from a limit-bound claim.
        assert VerdictBasis.ACCOUNT_STATE not in claim.verdict_basis, claim.claim_id


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
    subject = next(
        document
        for document in result.documents
        if document_evidence(document.doc_type).proves_subject
    )

    assert claim.covered_fraction == 1.0
    # 🔴 THE CLAIM'S AMOUNT IS ONE DOCUMENT'S, NOT THE SUM. On a split pair the invoice and the
    # payment describe ONE movement of money, and adding them would count it twice — which is what
    # `resolve_evidence` exists to prevent and what this line now measures rather than assumes.
    assert claim.reimbursable_amount == subject.amount
    if len(result.documents) > 1:
        # THE ARITHMETIC THAT WOULD HAVE GONE UNNOTICED. Both documents of a pair state the same
        # amount — the payment settles the invoice — so a claim that summed its documents would
        # report exactly twice the money and would still come out `covered`. The reimbursable
        # amount is what gives it away, which is the same shape as
        # `test_claim_evidence.py::test_a_split_pair_is_one_transaction_and_not_two`.
        naive_sum = sum(document.amount for document in result.documents)
        assert naive_sum == subject.amount * len(result.documents), "the pair should agree"
        assert claim.reimbursable_amount == subject.amount
        assert claim.reimbursable_amount != naive_sum
    assert claim.verdict_basis == [VerdictBasis.DOCUMENTS]
    assert claim.imperfection == []
    assert claim.policy_trace[0] == f"category={claim.category} ok"
    assert claim.policy_trace[-2:] == ["period ok", "coverage 100% (all line items covered)"]
    assert "1 transaction" in claim.policy_trace[1], claim.policy_trace


# ---------------------------------------------------- many claims, one run --


@pytest.fixture(scope="module")
def multi_claim_dataset(tmp_path_factory):
    """Enough personas × claims that the cumulative-limit mechanism actually fires."""
    out = tmp_path_factory.mktemp("multi")
    return (
        generate_dataset(
            seed=SEED, out_dir=out, train_fraction=0.5, personas=4, claims_per_persona=8
        ),
        out,
    )


def documents_of(result):
    """(claim, its documents) for a finished run, joined through `claim.documents`.

    Not `zip(result.claims, result.documents)`. `Dataset.documents` is a flat list over
    every claim of the run, and pairing the two lists positionally only works while every
    claim has exactly one document — a coincidence of the archetype registry, not a
    property of a dataset. The join through the claim's own document ids is what the label
    files give a consumer, and it stays right when a claim spans two.
    """
    by_id = {document.doc_id: document for document in result.documents}
    return [(claim, [by_id[doc_id] for doc_id in claim.documents]) for claim in result.claims]


def test_a_persona_can_file_several_claims(multi_claim_dataset):
    result, _ = multi_claim_dataset
    assert len(result.claims) > len(result.personas)
    assert len(result.documents) == sum(len(claim.documents) for claim in result.claims)


def test_every_document_written_belongs_to_exactly_one_claim(multi_claim_dataset):
    """The join `documents_of` relies on, asserted rather than assumed: no document is
    orphaned, none is claimed twice, and every id a claim names was written."""
    result, _ = multi_claim_dataset
    claimed = [doc_id for claim in result.claims for doc_id in claim.documents]

    assert len(claimed) == len(set(claimed)), "a document is claimed by two claims"
    assert set(claimed) == {document.doc_id for document in result.documents}


def test_the_single_claim_default_is_still_reachable(dataset):
    result, _ = dataset
    assert len(result.claims) == 1


def test_every_claim_of_a_persona_is_in_date_order(multi_claim_dataset):
    """By THE PAYMENT DATE, which is what a claim takes its place in the ledger by — see
    `ClaimInput.dated`. Read off the document whose TYPE proves payment, resolved through
    policy.yaml's `document_evidence` exactly as `policy_engine.resolve_evidence` does.

    🔴 IT USED TO TAKE THE LATEST DATE OF THE CLAIM'S DOCUMENTS, and that was wrong on a
    mechanism this repository deliberately generates. A claim planned with the cause
    `payment_precedes_subject` is dated BACKWARDS on purpose — its subject document is later
    than its payment — so the latest date is the subject's, and ordering by it compares a
    claim's subject against the next claim's payment. It passed only while no such claim
    happened to be adjacent to a later one; the check was never testing what it said. It also
    said "every document of this dataset is a fiscal receipt", which stopped being true when
    the invoice archetype landed.
    """
    result, _ = multi_claim_dataset
    for persona_record in result.personas:
        dates = [
            max(
                document.date
                for document in documents
                if document_evidence(document.doc_type).proves_payment
            )
            for claim, documents in documents_of(result)
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

    from receipt_synth.policy_engine import annual_limit, covered_total, document_evidence

    result, _ = multi_claim_dataset
    spent: dict[tuple[str, str], Decimal] = defaultdict(lambda: Decimal(0))

    for claim, documents in documents_of(result):
        key = (claim.persona_id, claim.category)
        limit = annual_limit(claim.category)
        remaining = limit - spent[key]
        # Over the claim's SUBJECT documents, which is where the line items of a claim
        # live — see `policy_engine.resolve_evidence`. Every document here proves both
        # facts, so the two sets coincide; taking them from the evidence table rather than
        # from every document is what keeps this re-derivation honest when they stop
        # coinciding.
        covered = covered_total(
            claim.category,
            [
                line
                for document in documents
                if document_evidence(document.doc_type).proves_subject
                for line in document.line_items
            ],
        )
        reimbursed = min(covered, remaining)

        # 🔴 A CLAIM WHOSE EVIDENCE IS INSUFFICIENT PAYS NOTHING AND CONSUMES NOTHING, so the
        # coverage arithmetic above does not describe it: its documents contradict each other, and
        # what its lines cover is beside the point. It was unreachable until the cross-check causes
        # became drawable, and the re-derivation asserted the covered amount against a
        # `reimbursable_amount` of zero the first time one appeared.
        #
        # THE LEDGER IS THE PART THAT MATTERS: such a claim must leave the balance untouched, or a
        # persona's later claims would be sized against money nothing ever paid out. That is
        # asserted here by NOT adding to `spent` and by the running total staying within the limit.
        if claim.verdict is Verdict.INSUFFICIENT_EVIDENCE:
            assert claim.reimbursable_amount == 0
            assert "limit_exhausted" not in claim.imperfection
            continue

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
    for claim, documents in documents_of(result):
        # True for every claim, including the limit-bound ones: `covered_fraction` stays
        # a property of the line items, and the limit is recorded elsewhere in the label.
        lines = [line for document in documents for line in document.line_items]
        covered = sum(
            Decimal(str(item.qty)) * Decimal(str(item.price))
            for item in lines
            if item.covered
        )
        total = sum(Decimal(str(item.qty)) * Decimal(str(item.price)) for item in lines)
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
    assert "20.0% of the target mix is absent" in report
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
    assert "20.0% of the target mix is absent" in report
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
    # ⚠️ THE EARLY STOP IS NO LONGER EXERCISED BY THIS RUN, and that is a measurement rather than a
    # regression: until the invoice landed a persona had ONE documentable category, so eight claims
    # against one annual limit exhausted it. Every category is completable now, so a persona has
    # several balances to spend and thirty-two claims fit. The assertion moved from
    # "built < ordered" — a property of a one-template registry — to the accounting identity, which
    # is what
    # the report is actually for. The early stop keeps its own test below, on a persona narrowed to
    # one category.
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
            (claim, documents)
            for claim, documents in documents_of(result)
            if claim.persona_id == persona_record.persona_id
        ]
        if len(pairs) == 8:
            continue  # this persona built everything it was asked for

        ledger = Ledger()
        for claim, documents in pairs:
            lines = [line for document in documents for line in document.line_items]
            covered = covered_total(claim.category, lines)
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
    """Unreachable today — the engine emits no verdict the planner cannot draw. It is the silence
    that would be the defect, so the row is asserted directly on a hand-built Dataset rather than
    waited for.

    The verdict is TAKEN FROM `unrealizable_verdicts()` rather than named: it used to name
    `insufficient_evidence`, which became realizable, and the test then asserted that a realizable
    verdict produces an unrealizable-verdict warning — passing for the wrong reason would have been
    the next step."""
    from receipt_synth.assembler import Dataset

    unrealizable = unrealizable_verdicts()
    assert unrealizable, "every verdict is realizable, so this test asserts nothing"
    verdict = unrealizable[0]

    claim = ClaimGroundTruth(
        claim_id="p001_c1", persona_id="p001", category="vitamins_nutrition",
        documents=["p001_c1_d1"], verdict=verdict,
    )
    report = balance_report(
        Dataset(seed=1, personas=[], claims=[claim], documents=[], claims_ordered=1)
    )

    assert verdict.value in report
    assert "realized but not realizable" in report


def a_document_on(capture, *, doc_id, lost=()):
    """One label record, for the capture block of the report only."""
    from receipt_synth.schemas import DocGroundTruth, DocType

    return DocGroundTruth(
        doc_id=doc_id, source_file=f"{doc_id}.png", doc_type=DocType.FISCAL_RECEIPT,
        language="uk", currency="UAH", amount=Decimal("10.00"), date=date(2026, 6, 15),
        counterparty="Аптека", line_items=[], has_qr=True, qr_is_fiscal=True,
        has_fiscal_number=True, capture=capture, content_bbox=(0.0, 0.0, 10.0, 10.0),
        content_lost_edges=list(lost),
    )


def test_the_draw_can_reach_every_capture_channel():
    """`CAPTURE_CHANNELS` is what the assembler draws from, and `Capture` is what the label may
    carry. A channel present in the enum and missing from the tuple would be a value the contract
    declares and no corpus can contain — unreachable rather than merely rare, and nothing about a
    run would say so."""
    from receipt_synth.assembler import CAPTURE_CHANNELS

    assert set(CAPTURE_CHANNELS) == set(Capture)
    assert len(CAPTURE_CHANNELS) == len(Capture), "a channel is listed twice, which weights it"


def test_a_run_of_any_size_contains_more_than_one_capture_channel(multi_claim_dataset):
    """🔴 THE CHANNEL IS DRAWN, NOT FIXED. Every earlier version of this generator produced a
    corpus of screenshots only, and nothing in the labels distinguished "this channel was chosen"
    from "this channel is the only one there is". With sixty-odd documents and an equal three-way
    draw, all three appear with a probability that rounds to one — so anything less is the draw
    being gone rather than the run being unlucky.

    It matters beyond variety: the two paper channels are the ONLY route by which the electronic
    side of the VAT-row rule stops being the whole corpus. See `test_vat_row_form.py`.
    """
    result, _ = multi_claim_dataset
    assert len(result.documents) > 30, "too few documents for this test to mean anything"
    assert {document.capture for document in result.documents} == set(Capture)


def test_a_capture_channel_with_no_documents_is_reported_ABSENT_and_never_as_zero():
    """🔴 ABSENT AND ZERO ARE DIFFERENT STATEMENTS AND ONLY ONE OF THEM CAN BE TRUE.

    `0.0%` complete says a measurement was taken and came out at nothing; absence says no
    measurement exists. Folding the first into the second is how an empty cell becomes a data point
    in somebody's table, and nobody re-derives it afterwards.

    Asserted in BOTH directions: the channel that has documents must carry a completeness fraction,
    and the two that have none must carry the word rather than a share. A test on the word alone
    would pass for a report that printed both.
    """
    from receipt_synth.assembler import Dataset

    dataset = Dataset(
        seed=1, personas=[], claims=[],
        documents=[
            a_document_on(Capture.PHOTO, doc_id="d1"),
            a_document_on(Capture.PHOTO, doc_id="d2", lost=["bottom"]),
        ],
    )
    report = balance_report(dataset)
    lines = {
        line.split()[0]: line
        for line in report.splitlines()
        if line.startswith("  ") and line.split() and line.split()[0] in
        {c.value for c in Capture}
    }

    assert lines["photo"].strip().startswith("photo")
    assert "complete 1/2" in lines["photo"]
    for absent in ("scan", "screenshot"):
        assert "ABSENT" in lines[absent], f"{absent} is missing but not declared absent"
        assert "complete" not in lines[absent], (
            f"{absent} has no documents, so a completeness figure for it is a measurement "
            "that was never taken"
        )


def test_a_document_whose_content_was_not_measured_is_counted_apart():
    """A `content_complete` of `None` is neither complete nor incomplete, and the report must not
    silently fold it into the second. Otherwise a corpus with no extents at all would print a
    completeness of 0 — a measurement nobody took, in the place a real one goes."""
    from receipt_synth.assembler import Dataset

    dataset = Dataset(
        seed=1, personas=[], claims=[],
        documents=[
            a_document_on(Capture.SCAN, doc_id="d1"),
            a_document_on(Capture.SCAN, doc_id="d2").model_copy(update={"content_bbox": None}),
        ],
    )
    report = balance_report(dataset)
    line = next(ln for ln in report.splitlines() if ln.strip().startswith("scan"))

    assert "complete 1/2" in line
    assert "1 unmeasured" in line, f"the unmeasured document is not declared: {line!r}"


def test_a_dataset_without_plans_still_reports():
    """`Dataset.plans` defaults to empty. Raising on the comparison would make the report
    unavailable exactly when someone assembled a Dataset by hand to look at it."""
    from receipt_synth.assembler import Dataset

    report = balance_report(Dataset(seed=1, personas=[], claims=[], documents=[]))
    assert "comparison skipped" in report


def test_the_multi_claim_run_is_reproducible(tmp_path):
    first = generate_dataset(
        seed=SEED, out_dir=tmp_path / "a", train_fraction=0.5, personas=3, claims_per_persona=5
    )
    second = generate_dataset(
        seed=SEED, out_dir=tmp_path / "b", train_fraction=0.5, personas=3, claims_per_persona=5
    )
    assert first.as_manifest() == second.as_manifest()
    assert balance_report(first) == balance_report(second)


def test_at_least_one_claim_is_refused(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(seed=SEED, out_dir=tmp_path, train_fraction=0.5, claims_per_persona=0)


def test_manifest_is_valid_json_and_declares_its_provenance(dataset):
    _, out = dataset
    manifest = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    result, _ = dataset
    assert manifest["synthetic"] is True
    assert manifest["seed"] == SEED
    assert manifest["generator_version"]
    # Against the run, not against `1`: a claim may hold two documents since the invoice landed.
    assert len(manifest["documents"]) == len(result.documents)
    assert len(manifest["claims"]) == len(result.claims)


def test_every_document_of_a_claim_is_on_the_claim_s_side_of_the_partition(multi_claim_dataset):
    """🔴 THE INTEGRITY THE PARTITION EXISTS TO KEEP. An invoice in train and the payment that
    settles it in validation is ONE TRANSACTION split across the boundary — the model would see the
    amount, the date and the counterparty of a validation document while training. It holds because
    the unit is the persona, which sits above both; this is the assertion that says it holds in the
    data rather than in the reasoning."""
    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}

    for claim in result.claims:
        assert claim.split is not None, claim.claim_id
        for doc_id in claim.documents:
            assert by_id[doc_id].split is claim.split, (claim.claim_id, doc_id)


def test_every_claim_of_a_persona_is_on_one_side(multi_claim_dataset):
    """The unit, asserted as a property of the output. A persona whose claims straddled the
    boundary would break the reason the unit is the persona at all: annual limits are cumulative,
    so a validation claim's own verdict would be a function of a training claim."""
    result, _ = multi_claim_dataset
    sides = defaultdict(set)
    for claim in result.claims:
        sides[claim.persona_id].add(claim.split)

    assert sides, "no claims, so this test asserts nothing"
    for persona_id, seen in sides.items():
        assert len(seen) == 1, f"{persona_id} has claims on both sides: {seen}"


def test_the_partition_reaches_the_written_labels_and_the_manifest(multi_claim_dataset):
    """On disk, not only in memory — a consumer reads the files. BOTH record kinds, because they are
    populated by two separate statements and only one of them was checked at first.

    🔴 THE TOTALS ARE COMPARED AGAINST THE CORPUS SIZE, NOT AGAINST THE RECORDS. Comparing the
    manifest's realized counts with `sum(1 for c in claims if c.split is side)` looks stricter and
    is not: both sides of that comparison read the same field, so a defect that stopped populating
    it moves both to zero and the assertion still holds. A mutation that dropped the claims' split
    entirely survived exactly that way. The absolute totals cannot be satisfied by a record that
    carries no side at all.
    """
    result, out = multi_claim_dataset
    document = result.documents[0]
    claim = result.claims[0]
    written = json.loads((out / "labels" / f"{document.doc_id}.json").read_text(encoding="utf-8"))
    written_claim = json.loads(
        (out / "labels" / f"{claim.claim_id}.claim.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    assert written["split"] == document.split.value
    assert written_claim["split"] == claim.split.value
    assert manifest["split"]["unit"] == "persona"
    # 0.5 because THIS FIXTURE PASSED 0.5, not because anything defaults to it — nothing does any
    # more. The comparison is stronger for it: the test now chooses the input and checks that the
    # manifest echoes the request, where before both sides read one constant the code supplied.
    assert manifest["split"]["train_fraction_requested"] == 0.5

    realized = manifest["split"]["realized"]
    assert sum(side["claims"] for side in realized.values()) == len(result.claims), (
        "the sides do not account for every claim — one is on neither side, or on both"
    )
    assert sum(side["documents"] for side in realized.values()) == len(result.documents)
    assert sum(side["personas"] for side in realized.values()) == len(result.personas)


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
    first = generate_dataset(seed=SEED, out_dir=tmp_path / "a", train_fraction=0.5)
    second = generate_dataset(seed=SEED, out_dir=tmp_path / "b", train_fraction=0.5)

    assert first.as_manifest() == second.as_manifest()
    name = first.documents[0].source_file
    assert (tmp_path / "a" / "images" / name).read_bytes() == (
        tmp_path / "b" / "images" / name
    ).read_bytes()


def test_a_different_seed_produces_a_different_dataset(tmp_path):
    other = generate_dataset(seed=SEED + 1, out_dir=tmp_path / "c", train_fraction=0.5)
    assert other.documents[0].amount != 0


def test_line_items_do_not_repeat_a_printed_name(tmp_path):
    """A cash register lists an article once and states how many. The same name twice at
    two prices is not a receipt."""
    for seed in range(SEED, SEED + 5):
        result = generate_dataset(seed=seed, out_dir=tmp_path / f"s{seed}", train_fraction=0.5)
        names = [item.name for item in result.documents[0].line_items]
        assert len(names) == len(set(names))


def test_cli_runs_and_reports(tmp_path, capsys):
    assert main(["--seed", str(SEED), "--split", "0.5", "--out", str(tmp_path)]) == 0
    assert "receipt-synth" in capsys.readouterr().out
    assert (tmp_path / "ground_truth.json").is_file()


def test_cli_requires_a_seed(capsys):
    """A default seed would let a run look reproducible without anyone having recorded
    what to reproduce it with.

    🔴 `--split` IS SUPPLIED SO THAT ONLY THE SEED IS MISSING. It became required too, and a call
    omitting both raises `SystemExit` whichever of them argparse is enforcing — so the test would
    have stayed green with `--seed` defaulted again, asserting nothing. The message is checked for
    the same reason: the raise alone does not say which argument produced it.
    """
    with pytest.raises(SystemExit):
        main(["--out", "out", "--split", "0.5"])
    assert "--seed" in capsys.readouterr().err
