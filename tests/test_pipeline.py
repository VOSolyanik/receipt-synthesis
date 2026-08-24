"""The pipeline ends: persona, plan, degradation, and the run as a whole.

The claim these tests exist to defend is the one the whole tool rests on — that a seed
determines the output. It is checked at each stage and again on the finished dataset,
because a single unseeded draw anywhere makes the promise false everywhere.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

import cv2
import numpy as np
import pytest
from PIL import Image

from receipt_synth import claim_planner, persona_generator
from receipt_synth.assembler import (
    SYNTHETIC_DATA_MARKER,
    _write_png,
    balance_report,
    generate_dataset,
)
from receipt_synth.claim_planner import (
    ARCHETYPES,
    REALIZABLE_VERDICTS,
    STATES_AN_INSTALMENT_TERM,
    ClaimPlan,
    DocumentPlan,
    _select_documents,
    archetypes_for,
    can_assemble_evidence,
    documentable_categories,
    draw_insufficient_evidence_cause,
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
from receipt_synth.config import (
    archetype_draw_weights,
    high_frequency_surnames,
    jurisdiction,
    load_policy,
    load_vendors,
    partial_payment_schedules,
)
from receipt_synth.content_builder import (
    MAX_LINE_ITEMS,
    draw_party_identity,
    is_valid_rnokpp,
)
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    COUNTERPARTY_MISMATCH,
    OUTSIDE_PERIOD,
    PAYMENT_PRECEDES_SUBJECT,
    SUBJECT_MISMATCH,
    SUBJECT_NOT_EVIDENCED,
    Evidence,
    Ledger,
    active_period,
    document_evidence,
    insufficient_evidence_causes,
    insufficient_evidence_causes_min_run_size,
    rejected_routes,
    reporting_currency,
    verdict_mix,
)
from receipt_synth.schemas import (
    Capture,
    ClaimGroundTruth,
    Country,
    DocGroundTruth,
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
    """⚠️ two calls in one process cannot see a clock dependence, which is why the two tests below
    exist beside this one. Both calls here happen at the same instant, so a value anchored on
    `datetime.now()` is identical in both — and one was: `Faker.date_of_birth` drew the persona's
    birth date, the РНОКПП encodes it, and this assertion held every day while the identifier
    changed at midnight."""
    assert persona() == persona()


def test_the_birth_date_the_identifier_encodes_is_anchored_on_THE_BENEFIT_PERIOD():
    """🔴 the discriminating test, and the mutation it exists for is the code that was there.

    A birth date anchored on the clock is invisible to every same-process assertion and moves the
    corpus at midnight. Measured, not argued: two production runs of the identical command two
    hours apart across midnight moved 36 of 96 personas' `tax_id` and 352 of 1141 images, because
    `Faker.date_of_birth` takes its offset from the seed and its anchor from `datetime.now()`.

    So the property asserted is not "deterministic" — that one passes either way. It is which
    anchor: move `policy.yaml`'s benefit period and the identifier must move with it. Under the
    clock-anchored draw this patch changes nothing at all and the assertion fails immediately.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            persona_generator, "active_period", lambda: (date(1990, 1, 1), date(1990, 12, 31))
        )
        moved = persona()

    assert moved.tax_id != persona().tax_id, (
        "the identifier did not move when the benefit period did, so the birth date it encodes "
        "is anchored on something other than the period — a clock, most likely"
    )


def test_the_drawn_birth_date_lands_inside_the_declared_age_band():
    """The positive half, computed outside the code: the РНОКПП's first five digits are days since
    the epoch, so the birth date can be read back off the identifier and checked against the band
    the period implies. Without this the test above is satisfied by an anchor that moves with the
    period and puts personas at any age at all."""
    reference = active_period()[1]
    oldest = reference - timedelta(days=round(60 * 365.2425))
    youngest = reference - timedelta(days=round(22 * 365.2425))

    for seed in range(50):
        days = int(persona(seed).tax_id[:5])
        born = date(1900, 1, 1) + timedelta(days=days - 1)
        assert oldest <= born <= youngest, (seed, born, oldest, youngest)


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
    """A payment date outside the window drives `rejected`. Landing outside it by accident would
    attach that document to a `covered` label, which is the defect this guards.

    ⚠️ `rejected` plans are excluded, not exempted. They leave the window on purpose, and the
    biconditional — outside if and only if `rejected` — is asserted over a run sized from the draw
    in `test_only_a_rejected_plan_dates_its_payment_outside_the_benefit_period`. Here the excluded
    plans are counted, so a change that made every plan `rejected` would empty this loop rather
    than pass it.
    """
    period = load_policy()["period"]
    start = date.fromisoformat(str(period["start"]))
    end = date.fromisoformat(str(period["end"]))

    subject = persona()
    plans = [
        plan_claim(random.Random(seed), persona=subject, claim_id="c1", ledger=Ledger())
        for seed in range(50)
    ]
    covered_by_the_period = [p for p in plans if p.verdict is not Verdict.REJECTED]
    assert covered_by_the_period, "every plan aimed at `rejected` — nothing was checked"

    for plan in covered_by_the_period:
        assert start <= plan.issued_at.date() <= end, plan.verdict


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
    only registered archetype supplies one of them is refused rather than half-built.

    Rewritten with the second document class, and the old form no longer described anything. It
    looked for a category no archetype covers, which was every category but one while the registry
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

    # A registry holding only payment-proving archetypes. Every category is then covered by
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


def test_a_verdict_no_archetype_can_carry_is_refused_not_faked():
    """Explicit over silent. A planner that accepted a verdict it cannot build and produced an
    ordinary basket would write a wrong label rather than fail. The message has to say what the
    verdict actually needs, not merely that it is unavailable.

    ⚠️ driven by a patched registry, because no unrealizable verdict is left to parametrize over:
    `insufficient_evidence` went when its cross-check causes became drawable, `rejected` when its
    period route did, `not_proof_of_payment` when `EvidenceIntent.PAYMENT_GAP` landed, and
    `partially_paid` when an archetype learned to print an instalment term. The capability is not
    obsolete with them — the enum grows, and the next member added before its mechanism exists is
    who this protects — so patching is what keeps the branch under test without carrying an
    unrealizable verdict nothing needs.

    The reason table is patched alongside the subset, because the two are a pair: a verdict removed
    from `REALIZABLE_VERDICTS` with no entry beside it gets the fallback message, which is the
    next test's subject and not this one's.
    """
    verdict = Verdict.PARTIALLY_PAID
    narrowed = tuple(v for v in claim_planner.REALIZABLE_VERDICTS if v is not verdict)
    assert len(narrowed) == len(claim_planner.REALIZABLE_VERDICTS) - 1, (
        "the patch removed nothing — this test would assert against the live registry"
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "REALIZABLE_VERDICTS", narrowed)
        patch.setattr(
            claim_planner,
            "_UNREALIZABLE_REASONS",
            {verdict: "needs an archetype that prints what a payment can be part of"},
        )
        with pytest.raises(NotImplementedError) as raised:
            plan_claim(
                random.Random(1), persona=persona(), claim_id="c1",
                verdict=verdict, ledger=Ledger(),
            )
    assert verdict.value in str(raised.value)
    assert len(str(raised.value)) > 60, "the message has to say what the verdict needs"


def test_the_planner_realizes_exactly_the_verdicts_the_engine_can_be_asked_for():
    """`insufficient_evidence` moved sides when the cross-check causes became drawable, `rejected`
    when the planner learned to date a payment outside the benefit period, and `partially_paid`
    when an archetype learned to print the instalment term a payment can settle one of. Every
    member of the enum is now realizable, which is what makes the second assertion below the
    interesting one: it is empty, and it has never been empty before.

    `_UNREALIZABLE_ROUTES` is where a route to a realizable verdict explains itself, and it is
    asserted empty now: `rejected` became buildable by what was bought as well as by when it was
    paid the day `_draw_basket` accepted a coverage target of zero, so every route to every
    realizable verdict is drawn — a promise the corpus keeps for the first time, and one this
    test exists to catch anyone quietly breaking.
    """
    assert set(REALIZABLE_VERDICTS) == set(Verdict), (
        "the planner no longer aims at every verdict; the balance report's absent-mix block and "
        "the tests that patch it are what a shrinking subset needs next"
    )
    assert set(unrealizable_verdicts()) == set()
    assert claim_planner._UNREALIZABLE_REASONS == {}, (
        "a verdict is named unrealizable while `REALIZABLE_VERDICTS` holds every member; one of "
        "the two tables is stale"
    )
    assert claim_planner._UNREALIZABLE_ROUTES == {}, (
        "the routes a realizable verdict cannot be reached by are declared here; an entry added "
        "or closed without this list moving is a corpus property nobody wrote down"
    )


def test_every_cause_the_policy_declares_a_share_for_can_actually_be_planned():
    """🔴 the other direction, and the one that would break silently. A share whose cause
    `plan_claim` refuses raises only on the run where the draw happens to land on it, which is a
    failure that arrives by luck rather than by test.

    Asserted by building a plan, not by looking the cause up in a table. Its first form checked
    membership of the unrealizable-route table, which the assertion two lines above had just
    pinned — a check that could not fail, which is the dominant defect class in this repository.
    The only thing that establishes a cause is plannable is planning it.

    A category is not named: the persona's own drawn categories are used, so this also covers the
    case where a cause's shape needs one the persona happens not to hold — `plan_claim` picks from
    `plannable_categories` for the verdict, and a cause that cannot be realized in any of them
    raises here.
    """
    causes = insufficient_evidence_causes()
    assert causes, "policy.yaml declares no cause — this test would assert nothing"

    for cause in causes:
        plan = plan_claim(
            random.Random(3), persona=persona(), claim_id="c1", ledger=Ledger(),
            verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=cause,
        )
        assert plan.cause == cause
        assert plan.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert plan.documents, f"{cause} planned a claim with no document"


def test_a_drawn_verdict_is_always_one_that_can_be_built():
    drawn = {draw_verdict(random.Random(seed)) for seed in range(200)}
    assert drawn == set(REALIZABLE_VERDICTS), "every realizable verdict must be reachable"


def test_a_realizable_verdict_with_no_share_cannot_be_drawn_from():
    """The combination policy.yaml and this planner must never be in at once: a verdict something
    can build, declared with no share. `random.choices` would be handed `None` as a weight, and
    the failure is named here rather than left to surface as a TypeError inside the standard
    library.

    ⚠️ driven by a patched mix. policy.yaml declared exactly this combination as long as nothing
    drew `rejected`; every declared member now carries a number, so the guard is untestable — and
    untested — without patching one back out. Whoever adds the next verdict to
    `REALIZABLE_VERDICTS` before giving it a share is who this still protects.
    """
    from receipt_synth import claim_planner

    undeclared = dict(verdict_mix())
    undeclared[Verdict.REJECTED] = None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "verdict_mix", lambda: undeclared)
        with pytest.raises(ValueError, match="no share"):
            claim_planner.draw_verdict(random.Random(1), REALIZABLE_VERDICTS)


def test_the_drawn_mix_is_the_target_mix_renormalized_over_the_realizable_subset():
    """verdict_mix gives covered 0.40, partially_covered 0.20 and 0.10 to each of the four
    negatives. Every member is now realizable, so the subset sums to 1.00 and the renormalization
    is the identity:

        covered                0.40 / 1.00 = 0.400
        partially_covered      0.20 / 1.00 = 0.200
        insufficient_evidence  0.10 / 1.00 = 0.100
        not_proof_of_payment   0.10 / 1.00 = 0.100
        partially_paid         0.10 / 1.00 = 0.100
        rejected               0.10 / 1.00 = 0.100

    ⚠️ the denominator moved twice without any share moving — 0.80, then 0.90 when
    `not_proof_of_payment` became realizable, now 1.00 with `partially_paid`. Each verdict carried
    its declared share throughout and each one is drawn slightly rarer after every move. That is
    the conditioning this table exists to make visible, and it is why policy.yaml's
    `insufficient_evidence_causes_min_run_size` moved 75 → 85 → 90 across the same revisions.

    🔴 the identity is the weakest state this test has ever been in, and it is worth saying so:
    while the subset was a strict subset, a renormalization bug showed up as a wrong share here.
    It cannot today — dividing by 1.00 hides the operation — so what still bites is the ±0.04
    window on each declared share, and what would catch the next regression is this file's other
    assertion that the subset equals the enum.

    Over 4000 draws each realized share should sit near its target. The window is wide (±0.04) on
    purpose: this asserts the weights are the policy's, not that a pseudo-random draw hits a mean.
    Every member is checked, because checking only `covered` would pass unchanged when a member
    joins the denominator.
    """
    rng = random.Random(20260803)
    draws = [draw_verdict(rng) for _ in range(4000)]
    mix = verdict_mix()
    total = sum(mix[verdict] for verdict in REALIZABLE_VERDICTS)

    assert total == pytest.approx(1.0), "the realizable shares no longer sum to what this asserts"
    for verdict in REALIZABLE_VERDICTS:
        share = draws.count(verdict) / len(draws)
        assert abs(share - mix[verdict] / total) < 0.04, verdict


def test_the_partially_covered_cause_is_drawn_from_the_policy():
    """partially_covered_causes: mixed_items 0.65, limit_exhausted 0.35."""
    rng = random.Random(11)
    draws = [draw_partially_covered_cause(rng) for _ in range(4000)]
    assert set(draws) == {"mixed_items", "limit_exhausted"}
    assert abs(draws.count("mixed_items") / len(draws) - 0.65) < 0.04


def test_every_insufficient_evidence_cause_is_reachable_by_the_draw():
    """insufficient_evidence_causes: subject_not_evidenced 0.34, amount_mismatch 0.33,
    payment_precedes_subject 0.33 — thirds to the nearest hundredth, and policy.yaml says why
    equality is the position rather than the default.

    🔴 the reachability is the assertion, the shares are the check on it. A cause the draw cannot
    reach is a bucket of the corpus nothing fills, and it fails silently: every claim still gets a
    label, the balance report still adds up, and one mechanism of three is simply never exercised.
    So every declared cause must come out of the generator, and each within a wide window of its
    share — wide because this asserts the weights are the policy's, not that a pseudo-random draw
    hits a mean.
    """
    shares = insufficient_evidence_causes()
    assert sum(shares.values()) == pytest.approx(1.0), shares

    rng = random.Random(20260803)
    draws = Counter(draw_insufficient_evidence_cause(rng) for _ in range(4000))
    assert set(draws) == set(shares), (
        f"{sorted(set(shares) - set(draws))} declared in policy.yaml and never drawn"
    )
    for cause, share in shares.items():
        assert abs(draws[cause] / 4000 - share) < 0.04, (cause, draws[cause])


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
    """Cumulative limits bind in date order, so the plans that spend have to be in it: a persona
    whose claims arrived unordered would exhaust its balance on whichever claim happened
    to be planned last rather than on the one that happened last.

    ⚠️ A `rejected` plan is not in that sequence, and it is dropped rather than tolerated. Its
    payment is displaced a whole benefit period out of the window, so it sorts before or after
    everything; it also reimburses nothing, which is why the order it breaks is one that does not
    describe it. The property that survives is the one the order exists for — the claims that
    consume a balance are ordered by the date they consume it on.
    """
    plans = list(
        plan_claims(random.Random(5), persona=persona(), count=6, ledger=Ledger())
    )
    spending = [plan for plan in plans if plan.verdict is not Verdict.REJECTED]
    assert spending, "every plan aimed at `rejected` — no order was checked"

    dates = [plan.issued_at for plan in spending]
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


# ------------------------------------------------- a run large enough to bind --
#
# 🔴 Why a planner-only run exists at all, because the alternative is what this replaced. Whether
# every declared cause is realized is a property of the draw, and the rarest of them lands on about
# one claim in twenty-four. The rendered fixture below builds ~32 claims, so a cause missing from it
# is an ordinary outcome and not a defect — config/policy.yaml says exactly that, and names the run
# size at which a zero stops being sampling noise. A suite that enforced at 32 what the policy
# declares unmeasurable below 75 would go red on the next change to the seed stream, for nothing.
#
# Planning is cheap: no template, no browser, no image. So the existence questions are asked here,
# at a size derived from the draw rather than chosen, and the rendered fixture is left to answer the
# questions only a built document can — what the label of such a claim actually says.


def _run_size_for(rate: float, threshold: float = 1e-3) -> int:
    """How many planned claims make an event drawn at `rate` practically certain to occur.

    An event drawn at rate p is absent from N claims with probability (1 - p)^N, so
    N = ln(threshold) / ln(1 - p). The rate is always derived from policy.yaml by a caller below
    and never written down, so a share that moves resizes the run instead of quietly making a
    test weaker.

    `threshold` is 1e-3 rather than the 0.05 policy.yaml uses for its own guideline. That guideline
    tells a reader of one corpus when to be suspicious of a zero; this is a test, and a test that
    fails once in twenty runs of a shifted seed stream is a test nobody trusts.
    """
    return math.ceil(math.log(threshold) / math.log(1 - rate))


def _drawn_at(verdict: Verdict) -> float:
    """The share of built claims that aim at `verdict`, i.e. the mix renormalized over the
    realizable subset — the same conditioning `draw_verdict` applies and the balance report
    prints."""
    mix = verdict_mix()
    return mix[verdict] / sum(mix[realizable] for realizable in REALIZABLE_VERDICTS)


def _claims_for_certainty(threshold: float = 1e-3) -> int:
    """How many planned claims make the rarest declared cause practically certain to occur.

    Derived from the policy, never written down: the chance a claim aims at
    `insufficient_evidence`, split by `insufficient_evidence_causes`.
    """
    rate = _drawn_at(Verdict.INSUFFICIENT_EVIDENCE) * min(insufficient_evidence_causes().values())
    return _run_size_for(rate, threshold)


def _plans_from_many_personas(count: int) -> list:
    """`count` or more claim plans, over as many fresh personas as it takes.

    Each persona gets its own generator derived from one root, as `assembler` does, and its own
    empty `Ledger` — which is never recorded into, so no balance is ever exhausted and no persona
    stops short. That is the difference from a real run and it is the right one here: what is being
    measured is the draw, and a ledger binding would remove claims for a reason that has nothing to
    do with the cause under test.
    """
    root = random.Random(SEED)
    plans: list = []
    index = 0
    while len(plans) < count:
        index += 1
        rng = random.Random(root.getrandbits(64))
        subject = generate_persona(rng, persona_id=f"p{index:03d}", country=Country.UA)
        plans.extend(plan_claims(rng, persona=subject, count=8, ledger=Ledger()))
    return plans


def test_every_declared_cause_is_planned_in_a_run_large_enough_to_require_it():
    """🔴 the requirement is on the result, not on the share. A verdict's share says nothing about
    which mechanism realized it, so ten percent of a corpus arriving through one cause would leave
    the vocabulary promising three and the data holding one. What has to hold is that each declared
    cause occurs — and it is asked at a size where a zero cannot be luck.

    The run is planner-only. A cause the planner never draws is a bucket of the corpus nothing
    fills, and it fails silently: every claim still gets a label and the balance report still adds
    up.
    """
    causes = insufficient_evidence_causes()
    size = _claims_for_certainty()
    plans = _plans_from_many_personas(size)
    assert len(plans) >= size

    seen = Counter(
        plan.cause for plan in plans if plan.verdict is Verdict.INSUFFICIENT_EVIDENCE
    )
    for cause in causes:
        assert seen[cause] > 0, (
            f"{cause} was drawn 0 times in {len(plans)} planned claims, where its own share makes "
            f"absence a one-in-a-thousand event: {dict(seen)}"
        )


def test_the_evidence_gap_is_planned_for_exactly_one_cause_and_no_other():
    """The structural half, and it holds claim by claim rather than in aggregate: a plan carries
    `EVIDENCE_GAP` if and only if its cause is `subject_not_evidenced`.

    Both directions matter and they fail differently. A gap planned for a cross-check cause would
    build a claim with nothing to cross-check, which the engine would label `subject_not_evidenced`
    while the plan said otherwise; a cross-check shape planned for the gap cause would come back
    `covered` or contradicting, and the label would name a mechanism the corpus does not contain.
    """
    plans = _plans_from_many_personas(_claims_for_certainty())
    gap = claim_planner.EvidenceIntent.EVIDENCE_GAP

    for plan in plans:
        assert (plan.intent is gap) == (plan.cause == SUBJECT_NOT_EVIDENCED), (
            f"{plan.claim_id}: intent {plan.intent} with cause {plan.cause!r}"
        )
    assert any(plan.intent is gap for plan in plans), "no plan is an evidence gap"
    assert any(plan.intent is not gap for plan in plans), "every plan is an evidence gap"


def test_rejected_is_planned_in_a_run_large_enough_to_require_it():
    """The verdict `verdict_mix` gave a share to has to be one the planner actually aims at.

    A share without a draw fails silently in the direction that looks healthiest: every claim still
    gets a label, the balance report still sums to 1.0 over the rows it prints, and the bucket the
    policy sized simply stays empty — which is precisely the state this repository was in until
    now, and which nothing but a run of this size makes visible.

    Sized from the policy rather than picked: `rejected` is drawn on its renormalized share of
    built claims, so a run of `_run_size_for` that rate misses it once in a thousand shifted seed
    streams.
    """
    size = _run_size_for(_drawn_at(Verdict.REJECTED))
    plans = _plans_from_many_personas(size)
    assert len(plans) >= size

    rejected = [plan for plan in plans if plan.verdict is Verdict.REJECTED]
    assert rejected, (
        f"no claim aimed at `rejected` in {len(plans)} planned claims, where its share in "
        "verdict_mix makes absence a one-in-a-thousand event"
    )
    assert all(
        plan.cause in (OUTSIDE_PERIOD, claim_planner.ZERO_COVERAGE) for plan in rejected
    ), (
        "a `rejected` plan carries one of the two routes `rejected_routes` declares: "
        f"{sorted({str(plan.cause) for plan in rejected})}"
    )


def test_only_an_outside_period_rejected_plan_dates_its_payment_outside_the_period():
    """Both directions, claim by claim, because one route's label rests entirely on this date.

    A plan on the `outside_period` route whose payment stayed inside the window is a claim the
    engine labels `covered` while the plan says otherwise — the drift the balance report would
    report as a builder shortfall, on a mechanism that never fired. A plan of any other verdict —
    or of the zero-coverage route, whose date is deliberately ordinary — whose payment left the
    window is worse: the engine refuses it on the period before it ever reads the basket, so the
    mechanism the claim was built to exercise is absent from the corpus under a label that says
    it is there.

    🔴 the biconditional must never read "outside == rejected". That equality is the corpus's
    most expensive degeneracy — the payment date alone predicting the verdict — and the
    zero-coverage route exists precisely to break it. The equality below is per route, which is
    the statement that survives.

    ⚠️ the claim's date, not every document's. policy.yaml checks the period against the payment
    and against nothing else, so a subject document dated inside the window on a `rejected` claim
    is correct and expected — an invoice issued in December and settled in the next benefit year.
    """
    start, end = active_period()
    plans = _plans_from_many_personas(_run_size_for(_drawn_at(Verdict.REJECTED)))

    for plan in plans:
        outside = not start <= plan.issued_at.date() <= end
        assert outside == (
            plan.verdict is Verdict.REJECTED and plan.cause == OUTSIDE_PERIOD
        ), (
            f"{plan.claim_id}: {plan.verdict.value}/{plan.cause} claim dated "
            f"{plan.issued_at.date()} against the period {start}..{end}"
        )


def test_a_zero_coverage_plan_keeps_an_ordinary_date_and_an_empty_basket_target():
    """The route by what was bought, named explicitly: an in-window date and a coverage target
    of exactly zero, which is what `content_builder._draw_basket` reads as "no covered line".

    The date staying inside the window is not incidental — it is the property the route was
    opened for. A consumer reading only the calendar must have nothing to read on this claim.
    """
    start, end = active_period()
    plan = plan_claim(
        random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
        verdict=Verdict.REJECTED, cause=claim_planner.ZERO_COVERAGE,
    )
    assert plan.cause == claim_planner.ZERO_COVERAGE
    assert plan.coverage_target == Decimal(0)
    assert start <= plan.issued_at.date() <= end, plan.issued_at


def test_both_rejected_routes_occur_in_a_run_large_enough_to_require_them():
    """A declared route share that never fires fails silently in the healthiest-looking
    direction — every claim still gets a label and the bucket simply stays empty. Sized from the
    draw: each route is `rejected`'s renormalized share times its `rejected_routes` share, so a
    run of `_run_size_for` that rate misses either route once in a thousand shifted seed streams.
    """
    shares = rejected_routes()
    size = _run_size_for(_drawn_at(Verdict.REJECTED) * min(shares.values()))
    plans = _plans_from_many_personas(size)

    routes = {plan.cause for plan in plans if plan.verdict is Verdict.REJECTED}
    assert routes == {OUTSIDE_PERIOD, claim_planner.ZERO_COVERAGE}, (
        f"routes realized in {len(plans)} planned claims: {sorted(map(str, routes))}; "
        "`rejected_routes` declares two"
    )


def test_a_rejected_payment_falls_on_both_sides_of_the_benefit_period():
    """🔴 A corpus where every refused claim is late teaches "late", NOT "outside".

    The displacement draws its sign, and this is what that draw is for: were it fixed, the payment
    date of every `rejected` claim would sort after every other claim in the dataset, and a
    consumer could reach the verdict from an inequality that happens to hold here and holds
    nowhere else. Asserted on both tails rather than on a count, since the two are equally likely
    and a run this size misses either one about once in a thousand.
    """
    start, end = active_period()
    # Two halvings, not one: the sign draw splits the out-of-period route's dates, and that
    # route is itself `rejected_routes["outside_period"]` of the verdict's bucket.
    size = _run_size_for(_drawn_at(Verdict.REJECTED) * rejected_routes()[OUTSIDE_PERIOD] * 0.5)
    plans = _plans_from_many_personas(size)
    assert len(plans) >= size

    dates = [plan.issued_at.date() for plan in plans if plan.verdict is Verdict.REJECTED]
    assert [d for d in dates if d < start], f"no payment before {start}: {sorted(dates)[:5]}"
    assert [d for d in dates if d > end], f"no payment after {end}: {sorted(dates)[-5:]}"


def test_not_proof_of_payment_is_planned_in_a_run_large_enough_to_require_it():
    """The verdict `verdict_mix` has always given a share to has to be one the planner aims at.

    ⚠️ the share did not move and the bucket was empty, which is the failure mode this size is
    chosen against: 0.10 of the mix has been declared since the file was written, every report
    printed the number, and nothing filled it. Nothing about a run said so except the balance
    report's "not generated in this run" line.

    Sized from the policy rather than picked, exactly as the `rejected` test above is.
    """
    size = _run_size_for(_drawn_at(Verdict.NOT_PROOF_OF_PAYMENT))
    plans = _plans_from_many_personas(size)
    assert len(plans) >= size

    aimed = [plan for plan in plans if plan.verdict is Verdict.NOT_PROOF_OF_PAYMENT]
    assert aimed, (
        f"no claim aimed at `not_proof_of_payment` in {len(plans)} planned claims, where its "
        "share in verdict_mix makes absence a one-in-a-thousand event"
    )
    for plan in aimed:
        assert plan.cause is None, (
            f"{plan.claim_id} carries the cause {plan.cause!r}; this verdict has one slot and one "
            "way to fail it, so `imperfection` is empty and a plan must not name one"
        )
        assert len(plan.documents) == 1, (
            f"{plan.claim_id} plans {len(plan.documents)} documents; the claim is a subject "
            "document and nothing beside it"
        )
        evidence = evidence_of(plan.documents[0].archetype)
        assert evidence == Evidence(True, False), (
            f"{plan.claim_id} is evidenced by {plan.documents[0].archetype.slug}, which proves "
            f"{evidence} — a document proving the payment leaves nothing for this verdict"
        )


def test_partially_paid_is_planned_in_a_run_large_enough_to_require_it():
    """The last member of `verdict_mix` to get a mechanism, sized from the policy exactly as the
    two tests above are.

    ⚠️ its share has been declared since the file was written and nothing filled it, for longer
    than any other verdict's: the bucket needed an archetype to print something no archetype
    printed, so no reordering of the planner could have opened it. That is the state this size is
    chosen against.

    Every plan is checked for all three things the mechanism consists of — a schedule the
    configuration declares, no cause, and a subject document whose class can state the term —
    because a plan short of any one of them builds a claim the engine labels `amount_mismatch`,
    which is a wrong label rather than a failure.
    """
    size = _run_size_for(_drawn_at(Verdict.PARTIALLY_PAID))
    plans = _plans_from_many_personas(size)
    assert len(plans) >= size

    aimed = [plan for plan in plans if plan.verdict is Verdict.PARTIALLY_PAID]
    assert aimed, (
        f"no claim aimed at `partially_paid` in {len(plans)} planned claims, where its share in "
        "verdict_mix makes absence a one-in-a-thousand event"
    )
    for plan in aimed:
        assert plan.schedule in partial_payment_schedules(), (
            f"{plan.claim_id} names the schedule {plan.schedule!r}, which "
            "config/generation.yaml does not declare"
        )
        assert plan.cause is None, (
            f"{plan.claim_id} carries the cause {plan.cause!r}; this verdict has one mechanism "
            "and one way to reach it, so `imperfection` is empty and a plan must not name one"
        )
        assert len(plan.documents) == 2, (
            f"{plan.claim_id} plans {len(plan.documents)} documents; a payment can settle part of "
            "an obligation only where another document states the obligation"
        )
        subject = plan.subject_document
        assert subject.archetype.doc_type in STATES_AN_INSTALMENT_TERM, (
            f"{plan.claim_id} is subjected by {subject.archetype.slug}, whose class cannot print "
            "the term the verdict rests on"
        )


def test_only_a_partially_paid_plan_names_a_payment_schedule():
    """Both directions, claim by claim. A `partially_paid` plan without a schedule builds an
    ordinary agreeing pair and comes back `covered`; a plan of any other verdict carrying one
    prints an instalment term on a claim whose payment settles the whole invoice, and the corpus
    then contains the marker on a claim that is not partly paid — which is precisely what would
    teach a consumer to ignore it."""
    plans = _plans_from_many_personas(_run_size_for(_drawn_at(Verdict.PARTIALLY_PAID)))

    for plan in plans:
        assert (plan.schedule is not None) == (plan.verdict is Verdict.PARTIALLY_PAID), (
            f"{plan.claim_id}: {plan.verdict.value} claim with schedule {plan.schedule!r}"
        )


def test_every_declared_schedule_is_reached_by_the_draw():
    """A schedule config/generation.yaml declares and the draw never selects is a difficulty knob
    with no corpus behind it — the same defect as an archetype nothing renders. Sized from the
    rarest of them: the draw is uniform, so each is the verdict's rate over the number declared.
    """
    schedules = partial_payment_schedules()
    assert len(schedules) > 1, "one schedule is not a draw — this test would assert nothing"

    plans = _plans_from_many_personas(
        _run_size_for(_drawn_at(Verdict.PARTIALLY_PAID) / len(schedules))
    )
    drawn = {plan.schedule for plan in plans if plan.schedule is not None}

    assert drawn == set(schedules), f"declared {sorted(schedules)}, drawn {sorted(drawn)}"


def test_the_non_fiscal_slip_is_reached_by_the_draw_and_only_through_this_verdict():
    """🔴 the archetype this task exists for, asserted on both sides.

    It must be reached — a registered archetype the draw never selects is a template with a test
    suite and no corpus, which is what it was for as long as it was a mock-up. And it must be
    reached only as the whole evidence of a `not_proof_of_payment` claim: nothing settles a
    товарний чек, so a pair built from one would assert a settlement relation no observation
    supports. See `claim_planner._SETTLED_BY_A_PAYMENT`.

    The size is that of the verdict that carries it, halved for the draw between the two
    subject-only archetypes an eligible category offers.
    """
    slug = "ua_non_fiscal_receipt"
    assert slug in ARCHETYPES, "the archetype is not registered — this test asserts nothing"

    plans = _plans_from_many_personas(
        _run_size_for(_drawn_at(Verdict.NOT_PROOF_OF_PAYMENT) * 0.5)
    )
    carrying = [
        plan for plan in plans
        if any(document.archetype.slug == slug for document in plan.documents)
    ]
    assert carrying, f"{slug} is registered and no plan of this run draws it"
    for plan in carrying:
        assert plan.verdict is Verdict.NOT_PROOF_OF_PAYMENT, (
            f"{plan.claim_id} carries {slug} under the verdict {plan.verdict.value}"
        )
        assert len(plan.documents) == 1, f"{plan.claim_id} pairs {slug} with another document"


def test_the_subject_half_of_a_pair_is_always_a_document_a_payment_can_settle():
    """The same rule from the other end, and over the shape rather than over one archetype: every
    two-document plan's subject is of a class `_SETTLED_BY_A_PAYMENT` names.

    🔴 it is the assertion that would fail first if the pair branch went back to drawing from
    every subject-only archetype, which is what it did before this class landed and what the
    obvious reading of `document_evidence` still suggests. The pair would then print a payment
    purpose citing nothing, on about half the split claims of six categories.
    """
    unsettleable = [
        archetype
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == Evidence(True, False)
        and archetype.doc_type not in claim_planner._SETTLED_BY_A_PAYMENT
    ]
    assert unsettleable, (
        "every registered subject-only archetype can be settled by a payment, so this test "
        "cannot distinguish the narrowing from its absence"
    )

    plans = _plans_from_many_personas(_claims_for_certainty())
    pairs = [plan for plan in plans if len(plan.documents) == 2]
    assert pairs, "no plan of this run carries two documents"

    for plan in pairs:
        subject = plan.subject_document.archetype
        assert subject.doc_type in claim_planner._SETTLED_BY_A_PAYMENT, (
            f"{plan.claim_id} pairs a payment with {subject.slug}, which no payment settles"
        )


def test_the_payment_gap_is_planned_for_exactly_the_verdict_that_needs_it():
    """The structural half, claim by claim: a plan carries `PAYMENT_GAP` if and only if its verdict
    is `not_proof_of_payment`.

    Both directions fail differently, and neither is visible in a count. A gap planned for another
    verdict would drop that claim's payment document and the engine would answer
    `not_proof_of_payment` while the plan named something else; the verdict planned without the gap
    would come back `covered`, and the corpus would promise a bucket it does not contain.

    ⚠️ and it is the other gap's mirror image. `EVIDENCE_GAP` is the subject gap, asserted the same
    way against `subject_not_evidenced` above. Two members of one enum with opposite meanings are
    exactly the pair a later reader will confuse, so each is pinned to its own verdict.
    """
    plans = _plans_from_many_personas(_run_size_for(_drawn_at(Verdict.NOT_PROOF_OF_PAYMENT)))
    gap = claim_planner.EvidenceIntent.PAYMENT_GAP

    for plan in plans:
        assert (plan.intent is gap) == (plan.verdict is Verdict.NOT_PROOF_OF_PAYMENT), (
            f"{plan.claim_id}: intent {plan.intent} on a {plan.verdict.value} claim"
        )
    assert any(plan.intent is gap for plan in plans), "no plan is a payment gap"
    assert any(plan.intent is not gap for plan in plans), "every plan is a payment gap"


def test_a_cause_named_for_not_proof_of_payment_is_refused():
    """One slot of `document_evidence`, one way to fail it, so `imperfection` is empty on every
    such claim and a caller naming a cause has misread the verdict. Refused rather than dropped:
    a cause silently ignored would be a plan whose record says something the label never will."""
    with pytest.raises(ValueError, match="carries no cause"):
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
            verdict=Verdict.NOT_PROOF_OF_PAYMENT, cause=SUBJECT_NOT_EVIDENCED,
        )


def test_a_category_documented_by_a_receipt_alone_cannot_realize_not_proof_of_payment():
    """🔴 the narrowing that excludes nothing in the live registry, exercised against a hand-built
    one — which is the only way to exercise it and the reason it is written at all.

    A fiscal receipt proves its own payment, so a category documented by receipts alone leaves no
    gap for this verdict: every claim it can produce carries a payment document. Every Ukrainian
    category also carries the invoice today, so `plannable_categories` filters nothing out of the
    live registry and a test against it would assert the absence of an effect.

    Without the filter such a category would be drawn and then refused inside `_select_documents`,
    mid-run, on whichever claim happened to draw it — the failure landing a stage from its cause.
    """
    receipts = {
        slug: archetype
        for slug, archetype in ARCHETYPES.items()
        if evidence_of(archetype) == Evidence(True, True)
    }
    assert receipts, "no self-sufficient archetype is registered — this test asserts nothing"
    category_id = next(iter(next(iter(receipts.values())).categories))

    holder = generate_persona(random.Random(3), persona_id="pX", country=Country.UA)
    holder = holder.model_copy(update={"benefit_categories": [category_id]})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", receipts)
        assert plannable_categories(holder, Ledger()) == [category_id], (
            "the narrowed registry cannot document the category at all, so the assertion below "
            "would hold for the wrong reason"
        )
        assert plannable_categories(holder, Ledger(), Verdict.NOT_PROOF_OF_PAYMENT) == []
        assert Verdict.NOT_PROOF_OF_PAYMENT not in realizable_verdicts_for(holder, Ledger())
        # And the refusal one stage on, in the words of the branch that would have raised.
        with pytest.raises(ValueError, match="states the subject alone"):
            _select_documents(
                random.Random(1),
                list(receipts.values()),
                datetime(2026, 6, 15, 12, 0),
                intent=claim_planner.EvidenceIntent.PAYMENT_GAP,
            )


def test_a_subject_mismatch_plan_carries_a_payment_that_can_print_the_citation():
    """🔴 the shape half of the `subject_mismatch` mechanism: the cause needs a page with a
    purpose line, and the app-transaction screen has none — a plan that drew it would build a
    claim whose builder refuses `must_cite`, a stage away from the choice that broke it.

    Direct calls, sized against the mutation rather than the draw. A planner that dropped the
    narrowing hands the cause to the citation-less archetype only at that archetype's own draw
    weight — 8% today — so a sweep sized merely to contain the cause once passes such a mutation
    more often than not (measured: it did). The seed count is derived from that weight the way
    `_run_size_for` derives everything else: absence of a violation across it clears the
    once-in-a-thousand bar, from the weight as configured rather than as remembered.
    """
    pool = {
        slug: archetype
        for slug, archetype in ARCHETYPES.items()
        if evidence_of(archetype) == Evidence(False, True)
        or archetype.doc_type in claim_planner._SETTLED_BY_A_PAYMENT
    }
    uncitable = [
        slug for slug in pool
        if evidence_of(pool[slug]) == Evidence(False, True)
        and slug not in claim_planner._CITES_THE_SETTLED_DOCUMENT
    ]
    assert uncitable, (
        "every registered payment archetype can print a citation, so this test cannot "
        "distinguish the narrowing from its absence"
    )
    weights = archetype_draw_weights()
    seeds = _run_size_for(min(weights[slug] for slug in uncitable))

    holder = generate_persona(random.Random(3), persona_id="pX", country=Country.UA)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", pool)
        for seed in range(seeds):
            plan = plan_claim(
                random.Random(seed), persona=holder, claim_id=f"c{seed}", ledger=Ledger(),
                verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=SUBJECT_MISMATCH,
            )
            payments = [
                d for d in plan.documents if not evidence_of(d.archetype).proves_subject
            ]
            assert len(payments) == 1, plan.claim_id
            assert payments[0].archetype.slug in claim_planner._CITES_THE_SETTLED_DOCUMENT, (
                f"seed {seed}: {payments[0].archetype.slug} cannot print the citation the "
                "cause is realized by"
            )


def test_a_rejected_route_the_policy_does_not_declare_is_refused_by_name():
    """`rejected` has exactly the two routes `rejected_routes` declares, so a caller naming a
    third has to be told what the two are rather than handed an ordinary claim under its name.

    ⚠️ the call that used to sit here asserted the opposite: `cause="zero_coverage"` was refused
    while nothing could build the basket, and it builds now — that flip is deliberate and is
    covered by `test_a_zero_coverage_plan_keeps_an_ordinary_date_and_an_empty_basket_target`.
    """
    with pytest.raises(ValueError, match="rejected_routes"):
        plan_claim(
            random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
            verdict=Verdict.REJECTED, cause="basket_from_another_planet",
        )


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
    """🔴 the subtle part of C, asserted directly rather than through a run.

    `insufficient_evidence` is realized by a claim whose subject and payment documents disagree, so
    it needs a category documented by a pair. A category holding a fiscal receipt gets one
    self-sufficient document — the planner prefers that shape — and one document cannot contradict
    itself, so such a category can never realize the verdict.

    And the narrowing is per verdict, not global: `covered` and `partially_covered` are realizable
    in both shapes, so a receipt category stays plannable for them. Narrowing globally would remove
    the only `fiscal_receipt` documents the corpus has, to satisfy a constraint belonging to one
    verdict out of three.

    ⚠️ why this is a unit test and not a run. A mutation that removed the narrowing survived every
    run-based test, and the reason was not a weak test: no persona of the pipeline fixture holds
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
    persona's draw must exclude the verdict rather than produce a plan that has to be refused a
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
    # `rejected` is realizable for this persona, and the contrast is the point: it is the one
    # verdict whose mechanism is a date, so it asks nothing of the shape of the evidence and a
    # self-sufficient receipt realizes it as readily as a pair does.
    #
    # And so is `not_proof_of_payment`, which is a different contrast and a sharper one. It does
    # ask something of the shape of the evidence — a subject document with no payment beside it —
    # and this category satisfies that not because it has a receipt but because it also has the
    # invoice and the товарний чек. The receipt is useless to it: a document that proves its own
    # payment leaves no gap. So the verdict is here on the strength of the other archetypes of the
    # same category, which is exactly what `plannable_categories` narrows on.
    assert set(realizable_verdicts_for(holder, Ledger())) == {
        Verdict.COVERED,
        Verdict.PARTIALLY_COVERED,
        Verdict.REJECTED,
        Verdict.NOT_PROOF_OF_PAYMENT,
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
    """The fallback that keeps the promise above: a verdict outside `REALIZABLE_VERDICTS` and
    absent from the reasons table gets a `NotImplementedError` naming itself, never a `KeyError`
    out of the table.

    Simulated by narrowing the realizable subset, because the reasons table is empty now — every
    member of the enum is realizable — so there is no live combination to drive it with and no
    seventh member to add. The patch is the whole point: the state it fabricates is exactly the
    state whoever adds the next verdict to the enum will be in for one commit.
    """
    from receipt_synth import claim_planner

    verdict = Verdict.PARTIALLY_PAID
    narrowed = tuple(v for v in claim_planner.REALIZABLE_VERDICTS if v is not verdict)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "REALIZABLE_VERDICTS", narrowed)
        patch.setattr(claim_planner, "_UNREALIZABLE_REASONS", {})
        with pytest.raises(NotImplementedError, match="no mechanism registered"):
            plan_claim(
                random.Random(1), persona=persona(), claim_id="c1", ledger=Ledger(),
                verdict=verdict,
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
    """🔴 the tripwire that caught `DirtyRollers`. Determinism under `--seed` is an invariant of
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

    🔴 each table is checked by itself. There are two recipe tables — one for the paper effects
    and one for the geometry — and `degrade` calls them in order, so asserting only that `degrade`
    raises lets the second table's guard keep the test green while the first one is gone. A
    mutation deleting the paper table's refusal survives exactly that way. The denominator is two.
    """
    from receipt_synth.degrader import _geometry, _paper_pipeline

    image, boxes = image_and_boxes()
    with pytest.raises(NotImplementedError, match="has no recipe"):
        degrade(image, boxes, seed=7, capture="fax")  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="has no recipe"):
        _paper_pipeline("fax", 7)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="has no recipe"):
        _geometry("fax")  # type: ignore[arg-type]


# ------------------------------------------------------------- the shipped PNG marker --


def test_write_png_stamps_the_marker_as_a_text_chunk(tmp_path):
    """Pins the finding that sent `SYNTHETIC_DATA_MARKER` to ASCII hyphens: PIL's `PngInfo` encodes
    to Latin-1 and falls back to an `iTXt` chunk — silently — for anything that does not fit, so a
    marker spelled with an em dash would never raise and would still land in the wrong chunk.
    `.text` only surfaces what `tEXt`/`iTXt`/`zTXt` chunks PIL actually wrote, so reading it back is
    already a check that a `tEXt` chunk exists, not merely that the string round-trips."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "marker.png"

    _write_png(path, image)

    with Image.open(path) as written:
        assert written.text["Comment"] == SYNTHETIC_DATA_MARKER


def test_write_png_does_not_touch_a_pixel(tmp_path):
    """The marker is metadata appended after the pixel data; writing it must not alter a single
    value of the image `degrader.degrade` produced. Compared through `cv2.imread`, which is how
    every degraded array reaches disk in the real pipeline — see `_build_document`."""
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    path = tmp_path / "pixels.png"

    _write_png(path, image)

    # `_write_png` takes BGR (OpenCV's convention) and converts to RGB before saving with PIL;
    # `cv2.imread` reads it back as BGR, so round-tripping through it is the honest comparison.
    assert np.array_equal(cv2.imread(str(path)), image)


# ------------------------------------------------------------- the whole run --


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    return generate_dataset(seed=SEED, out_dir=out, train_fraction=0.5), out


def test_the_run_builds_exactly_the_documents_its_plans_asked_for(dataset):
    """One document per claim is a property of the archetype registry — the one template
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
    """The join from a claim to its evidence, asserted against the run rather than against the
    number one. Reading `== [result.documents[0].doc_id]` and pinning the type as a fiscal receipt
    holds only while every registered archetype proves both facts, and breaks the moment the
    invoice makes a split pair buildable — a constant that is really a property of the registry,
    which is what the test above it warns about."""
    result, _ = dataset
    claim = result.claims[0]

    assert claim.documents == [document.doc_id for document in result.documents]
    assert claim.verdict is Verdict.COVERED
    assert claim.linked is (len(claim.documents) > 1)


def test_every_document_of_a_claim_names_the_same_persona_and_the_same_vendor(dataset):
    """the wiring between the persona and the documents, which nothing asserted until a mutation
    survived and said so.

    A claim's documents must agree about who. The invoice is addressed to the claimant and the
    payment document is drawn on the claimant's account, so `payer` is the persona's name on every
    document that names one; and the seller must be the same vendor instance on both, which used to
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
    """🔴 the receipt branch of the assembler, exercised directly — because no run-based fixture
    reaches it any more.

    Measured rather than assumed: the only category with a both-proving archetype is
    `vitamins_nutrition`, and no persona of either pipeline fixture holds it. Since the invoice
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
    # The channel reached the builder: the VAT row's form is one the document's own capture allows.
    allowed = jurisdiction("UA")["tax_line_forms_by_medium"][document.capture.medium.value]
    assert document.vat_row_form in allowed, (
        f"{document.vat_row_form!r} is not a form a {document.capture.value} may print"
    )
    assert (tmp_path / "images" / document.source_file).is_file()


def test_no_claim_contradicts_itself_across_its_own_documents(multi_claim_dataset):
    """🔴 A third class of check, and it exists because two other classes could not see the defect
    that produced it.

    A claim's documents describe one transaction between one pair of parties. Every field they
    share must therefore agree — and agree before normalization, on the raw string, which is the
    whole point.

    Why per-document assertions cannot see this. `counterparty` was labelled as the printed form on
    two classes («ТОВ «Ключ»») and as the bare trade name on two others («Ключ»), for three commits.
    Every per-document assertion passed: each class was internally consistent. The contract's
    comparison rule strips the legal form, so both spellings compare equal for any consumer — no
    scorecard could have shown it either.

    And why mutation testing cannot see it. There was nothing to break: no assertion existed whose
    reddening would reveal the divergence, so a mutation of either side left the suite green. A gap
    of this shape is invisible to a technique that measures whether existing assertions bite.

    So the form is: take a claim, take its documents, and ask whether they agree with each other on
    every field they share. It is checked here, on a run, rather than per class, because no single
    class can be wrong about it alone.

    The three deliberate disagreements are excluded by the claim's own cause, never by a tolerance:
    a claim planned as `amount_mismatch` must disagree about the amount and about nothing else, one
    planned as `payment_precedes_subject` about the order and about nothing else, and one planned
    as `counterparty_mismatch` about the party and about nothing else. Their exclusion is therefore
    itself an assertion — the label says which disagreement is intended, and everything else must
    still agree. The three are the axes `cross_document_agreement` declares in config/policy.yaml.
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

        # Fields that must agree on the raw value, whatever the claim's cause. `counterparty` left
        # this list when it became an axis a claim can be built to fail — it is checked below,
        # against the claim's own cause, exactly as the amount and the order are.
        #
        # 🔴 Currency is the hard one and it is the oracle's rule, not this test's opinion:
        # `policy_engine._one_claim_one_currency` refuses to label a claim stated in two, because
        # coverage pools line items across documents and every cross-document axis compares
        # amounts between them. The planner pairs within a currency so this can never be reached.
        for field in ("currency", "synthetic"):
            values = {getattr(document, field) for document in documents}
            assert len(values) == 1, f"{claim.claim_id} disagrees about {field}: {values}"
            checked[field] += 1

        # 🔴 Language is not one of them any more, and the exception is asserted rather than
        # dropped. It sat in the tuple above while every document of the corpus was Ukrainian, and
        # the euro pair falsified it: a Ukrainian employee who buys from a foreign platform submits
        # that platform's English invoice and their own bank's Ukrainian confirmation of the
        # transfer. That is not a defect to be tolerated — it is the ordinary shape of a
        # cross-border claim, and a corpus in which every claim is monolingual would be missing it.
        #
        # ⛔ The licence is narrow and this is where it is stated: a claim may hold two languages
        # exactly when it is stated in a currency that is not the reporting one. A mixed-language
        # claim in hryvnias would be two domestic documents disagreeing, which nothing builds and
        # nothing would explain. Nothing about the contract changes — `language` is a per-document
        # field with a per-language reporting slice, and no statement of it is about a claim.
        languages = {document.language for document in documents}
        currencies = {document.currency for document in documents}
        assert len(languages) == 1 or currencies != {reporting_currency()}, (
            f"{claim.claim_id} disagrees about language: {languages}, and its documents are "
            f"stated in {currencies} — a domestic claim's documents share a language"
        )
        checked["language" if len(languages) == 1 else "language_cross_border"] += 1

        # 🔴 The payer is narrower, and this check is what established it. Its first run failed on
        # `{'-', 'Олекса Семенюк'}`: an internet-acquiring confirmation 👁 does not identify the
        # payer and prints a hyphen as the value, while the invoice beside it names the claimant.
        # The document is right — that emptiness is observed and deliberate — so the rule is that
        # every document which names a payer names the same one, and the exemption is keyed on the
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

        # The amount, unless the claim's own label says the two were built to disagree — or says
        # the payment settles one part of the subject, which is a third case and not a weaker
        # version of either. `partially_paid` is a claim whose documents agree: the payment is
        # exactly what the subject printed as an instalment, so the equality that holds is against
        # that field rather than against the total. Checking it against `amount` would have made
        # this sweep read the intended pair as a defect; skipping the claim would have left the
        # only shape whose two documents state different totals on purpose unchecked.
        if claim.verdict is Verdict.PARTIALLY_PAID:
            assert subject.instalment_amount is not None, claim.claim_id
            assert payment.amount == subject.instalment_amount, claim.claim_id
            assert payment.amount < subject.amount, claim.claim_id
            checked["amount_is_one_instalment"] += 1
        elif AMOUNT_MISMATCH in claim.imperfection:
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

        # The party, on the same principle. 🔴 the only field that differs on such a claim, which
        # is what makes the negative worth building: an invoice from one seller beside a payment to
        # another agrees about everything else, so a consumer cannot reach the label by any route
        # but the names.
        if COUNTERPARTY_MISMATCH in claim.imperfection:
            assert subject.counterparty != payment.counterparty, claim.claim_id
            assert subject.amount == payment.amount, (
                f"{claim.claim_id} is a party mismatch and must disagree about nothing else"
            )
        else:
            assert subject.counterparty == payment.counterparty, claim.claim_id
        checked["counterparty"] += 1

    assert checked["amount"] == len(multi), (
        f"{checked['amount']} of {len(multi)} multi-document claims were checked"
    )
    print(f"\ncross-document self-consistency: {len(multi)} of {len(result.claims)} claims "
          f"carry two documents; checks applied {dict(checked)}")


def test_the_documents_of_a_run_agree_on_the_sellers_PRINTED_identity(multi_claim_dataset):
    """The test above, one layer down: on the page rather than on the label.

    🔴 it is here and not only in test_cross_document_identity.py because of what that module
    cannot see. That module builds a pair the way the assembler builds one and asserts on the two
    rendered pages — which proves the builders honour a shared identity, and proves nothing about
    whether the assembler hands them one. `identity=` is a required parameter, so dropping it fails
    loudly; `cites=` is not, so an assembler that stopped passing it would go on producing valid
    documents whose purpose lines cite a stranger, and every assertion in that module would still
    pass. This one runs the whole pipeline and reads what came out.

    ⚠️ and it is the measurement that found the defect, with the same reader — `fields_of` of
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
    domestic_pairs = 0
    for claim in result.claims:
        documents = [by_id[doc_id] for doc_id in claim.documents]
        if len(documents) < 2:
            continue
        subject, payment = (
            fields_of(d.model_dump(mode="json"))
            for d in sorted(
                documents, key=lambda d: not document_evidence(d.doc_type).proves_subject
            )
        )
        # 🔴 The one claim whose pages name two sellers on purpose, and it leaves this sweep with an
        # assertion rather than with a skip: a claim planned as `counterparty_mismatch` was built
        # so that the payment went to another party, so its seller rows must disagree, and one that
        # agreed would be the negative silently not built. It is excluded from the denominator
        # afterwards because the rows below measure the opposite property — that a claim describing
        # one purchase names one seller by one set of numbers.
        if COUNTERPARTY_MISMATCH in claim.imperfection:
            assert subject["seller_name"] and payment["seller_name"], (
                f"{claim.claim_id}: a seller's name was not readable on both pages, so the "
                "disagreement below would assert nothing"
            )
            assert subject["seller_name"] != payment["seller_name"], claim.claim_id
            readable["seller_name_disagrees_on_purpose"] += 1
            continue
        # 🔴 The second deliberate exclusion, counted and verified rather than skipped: a claim
        # whose payment page is the app's transaction screen prints no party block at all — that
        # absence is the archetype's entire argument — so the requisite rows below have nothing
        # to read on it by design, not by defect. The discriminator is the box vocabulary
        # (`merchant_descriptor` is printed by that archetype and no other), and the branch
        # asserts the absence it excuses: a party requisite turning up on such a page would mean
        # the archetype stopped being the negative example it is registered as.
        payment_document = max(
            documents, key=lambda d: document_evidence(d.doc_type).proves_payment
        )
        if "merchant_descriptor" in payment_document.field_bboxes:
            for field in ("seller_name", "seller_tax_code", "seller_account"):
                assert payment[field] is None, (
                    f"{claim.claim_id}: the app transaction screen printed {field!r}, and its "
                    "whole registration rests on printing no party requisite at all"
                )
            readable["payment_prints_no_party_block"] += 1
            continue
        pairs += 1
        # 🔴 The third deliberate exclusion, and — like the two above — it asserts what it excuses.
        # A cross-border pair names a seller outside the Ukrainian register: it has no ЄДРПОУ and
        # its bank has no МФО, so neither page prints a tax code and there is nothing for the row
        # below to compare. Printing the code drawn for the claim would put an eight-digit
        # Ukrainian identifier under a foreign company's name, which is the defect this exclusion
        # exists to keep visible. The discriminator is the seller's own IBAN — a foreign account is
        # what says the party is foreign — and the two requisites that are printed stay strict.
        if subject["seller_tax_code"] is None and payment["seller_tax_code"] is None:
            assert subject["seller_account"] and not subject["seller_account"].startswith("UA"), (
                f"{claim.claim_id}: neither page printed a seller's tax code and the seller banks "
                "in Ukraine, so the absence is a defect rather than a foreign party"
            )
            readable["cross_border_prints_no_tax_code"] += 1
        else:
            domestic_pairs += 1
        for field in ("seller_name", "seller_tax_code", "seller_account", "seller_bank_name"):
            if subject[field] is None or payment[field] is None:
                continue
            readable[field] += 1
            agree[field] += subject[field] == payment[field]
        if payment["invoice_number"] is not None:
            # 🔴 And the citation has its own deliberate disagreement, which is not the party's:
            # a claim planned as `subject_mismatch` prints a purpose naming another рахунок, and
            # that wrong number is the negative. So it is asserted to disagree rather than counted
            # among the pairs that must agree — the same shape as the `counterparty_mismatch`
            # branch above, on the axis beside it.
            #
            # ⚠️ It was not asserted here until a redrawn run happened to contain one. The cause
            # lands on about one built claim in forty, this fixture builds a few dozen, and the
            # row read `agree == readable` for as long as no such claim was drawn — a gap in the
            # test that only a change of the seed stream could expose, and did.
            if SUBJECT_MISMATCH in claim.imperfection:
                assert subject["invoice_number"] is not None, (
                    f"{claim.claim_id}: the subject's own number was not readable, so the "
                    "disagreement below would assert nothing"
                )
                assert subject["invoice_number"] != payment["invoice_number"], (
                    f"{claim.claim_id} is a subject mismatch and its payment cites the claim's "
                    "own invoice, so the negative was not built"
                )
                readable["invoice_number_disagrees_on_purpose"] += 1
            else:
                readable["invoice_number"] += 1
                agree["invoice_number"] += (
                    subject["invoice_number"] == payment["invoice_number"]
                )

    assert pairs, "no claim of this run carries two documents — nothing was measured"
    # 👁 The payee's bank is sometimes a caption with nothing under it, observed on the recipient's
    # bank of a real confirmation, so that row is readable on most pairs and not on all. The two
    # bounds are therefore different assertions rather than one loosened to fit: three requisites
    # are printed on every pair, and the fourth must agree wherever it is printed at all.
    for field in ("seller_name", "seller_account"):
        assert readable[field] == pairs, (
            f"{field} was readable on {readable[field]} of {pairs} pairs; a field the audit "
            "cannot read is a field it cannot report on either"
        )
    # The tax code is measured over the pairs that have one — see the cross-border branch above,
    # where the denominator is split and the absence is asserted rather than tolerated.
    assert readable["seller_tax_code"] == domestic_pairs, (
        f"seller_tax_code was readable on {readable['seller_tax_code']} of {domestic_pairs} "
        "domestic pairs; a field the audit cannot read is a field it cannot report on either"
    )
    assert domestic_pairs, "no domestic pair in this run — the tax-code row measured nothing"
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


def test_a_built_claim_carries_exactly_one_insufficient_evidence_cause(multi_claim_dataset):
    """The declared causes are mutually exclusive by construction — the planner draws one, and a
    claim with no subject document has no pair to cross-check — so a claim carrying two would mean
    the builder had realized a cause nobody planned. That holds claim by claim, at any run size.

    ⚠️ whether every cause occurs is asked elsewhere, and that is not a weakening. Each cause
    lands on about one built claim in forty, so at this fixture's size a zero is ordinary
    sampling — config/policy.yaml says so and names the run size at which it stops being
    (`insufficient_evidence_causes_min_run_size`). Enforcing existence here would make the suite go
    red on the next change to the seed stream, for no defect;
    `test_every_declared_cause_is_planned_in_a_run_large_enough_to_require_it` asks the same
    question at a size where the answer means something. The clause below still enforces it if this
    fixture ever grows past the guideline.

    ⚠️ the denominator is derived from the policy, not written here. The count of declared causes
    moved from two to three when `subject_not_evidenced` gained a mechanism, and a test pinned to
    the number would have had to be edited for a change it is meant to cover.
    """
    result, _ = multi_claim_dataset
    causes = set(insufficient_evidence_causes())
    assert len(causes) > 1, "one declared cause makes 'exactly one per claim' a tautology"

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

    if len(result.claims) >= insufficient_evidence_causes_min_run_size():
        for cause in causes:
            assert seen[cause] > 0, (
                f"{cause} occurs 0 times in {len(flagged)} insufficient_evidence claims of "
                f"{len(result.claims)}, at or above the run size policy.yaml requires every "
                "declared cause to be non-zero at"
            )


def test_a_claim_drawn_as_insufficient_evidence_is_labelled_as_one(multi_claim_dataset):
    """🔴 the plan and the label must agree for this verdict, and that is not true of the other two.

    A claim drawn as `covered` may legitimately come back `partially_covered` — the ledger
    overrules the plan, which is the cumulative-limit mechanism working, and
    `assembler._drift_lines` reports it. A claim drawn as `insufficient_evidence` has no such
    excuse: the builder was told to make two documents disagree, the engine compares them
    deterministically, and there is no third party to overrule anything. So a drawn claim that
    comes back `covered` means the builder did not do what it was told, silently.

    Found by a surviving mutation. `plannable_categories` was made to ignore the verdict, so claims
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


def test_a_claim_evidenced_by_a_payment_alone_is_labelled_as_one(multi_claim_dataset):
    """🔴 the case the thesis rests on, read off the label a consumer receives: a claim carrying a
    bank document and nothing that says what was bought. No single document proves both facts, and
    until `EvidenceIntent.EVIDENCE_GAP` the corpus could not show a system meeting an absent
    subject — only two documents contradicting each other.

    What is checked is the claim as it is written to disk, not the plan: one document, of a type
    policy.yaml says proves no subject, `insufficient_evidence` with the cause
    `subject_not_evidenced`, nothing reimbursed, and no coverage fraction because there is no line
    to compute one from. `linked` is false: one document is not a linked claim.

    ⚠️ existence is not asserted at this size, and the reason is the policy's own. Such a claim
    arrives on about one built claim in twenty-four, so requiring one in ~32 would be requiring at
    n≈32 what config/policy.yaml declares unmeasurable below
    `insufficient_evidence_causes_min_run_size`. That the planner draws them is asserted where it
    binds, in `test_every_declared_cause_is_planned_in_a_run_large_enough_to_require_it` and
    `test_the_evidence_gap_is_planned_for_exactly_one_cause_and_no_other`; that the engine labels
    one correctly is asserted on a hand-built claim in
    `tests/test_claim_evidence.py::test_a_payment_with_nothing_saying_what_it_bought_is_insufficient_evidence`.
    What only a run can add is that the two meet — so the shape is checked on whatever this run
    produced, and required outright once a run is large enough for a zero to mean something.
    """
    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}

    gaps = [
        claim for claim in result.claims
        if not any(
            document_evidence(by_id[doc_id].doc_type).proves_subject
            for doc_id in claim.documents
        )
    ]
    if len(result.claims) >= insufficient_evidence_causes_min_run_size():
        assert gaps, (
            f"no claim of this {len(result.claims)}-claim run is evidenced by a payment alone, at "
            "or above the run size policy.yaml requires every declared cause to be non-zero at"
        )

    for claim in gaps:
        assert len(claim.documents) == 1, claim.claim_id
        assert claim.verdict is Verdict.INSUFFICIENT_EVIDENCE, claim.claim_id
        assert claim.imperfection == [SUBJECT_NOT_EVIDENCED], claim.claim_id
        assert claim.reimbursable_amount == 0, claim.claim_id
        assert claim.covered_fraction is None, claim.claim_id
        assert claim.linked is False, claim.claim_id
        document = by_id[claim.documents[0]]
        assert document.line_items == [], (
            f"{document.doc_id} proves no subject and yet lists items"
        )
        assert "no document states what was bought" in " ".join(claim.policy_trace)


def test_an_insufficient_evidence_claim_pays_nothing_and_spends_no_balance(multi_claim_dataset):
    """A claim whose documents contradict each other is not a claim whose money is merely capped.
    It reimburses nothing, and — the part that would go unnoticed — it must leave the persona's
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


def test_a_claim_evidenced_by_a_subject_alone_is_labelled_as_one(multi_claim_dataset):
    """🔴 the other gap, read off the label a consumer receives: a claim carrying a document that
    says what was bought and nothing that says the money moved. It is the case rule RC-08 rests on
    — an employee submits what they have, and the policy says it is not proof of payment.

    Checked on what landed on disk rather than on the plan: one document, of a type policy.yaml
    says proves no payment, `not_proof_of_payment` with an empty `imperfection` (one slot, one way
    to fail it), nothing reimbursed, `linked` false — and a `covered_fraction` that is a real
    number rather than `null`, which is the sharpest contrast with the subject gap. Such a claim
    has line items: the basket is ordinary and typically wholly covered, and none of it is payable.

    ⚠️ existence is not asserted at this size, and the reason is the same one the payment-alone
    test gives: at ~32 built claims a zero is ordinary sampling. That the planner draws them is
    asserted at a size derived from the mix, in
    `test_not_proof_of_payment_is_planned_in_a_run_large_enough_to_require_it`; that the engine
    labels one correctly is asserted per document type in
    `tests/test_claim_evidence.py::test_every_type_that_proves_no_payment_reaches_the_same_verdict_alone`.
    What only a run can add is that the two meet.
    """
    result, _ = multi_claim_dataset
    by_id = {document.doc_id: document for document in result.documents}

    unpaid = [
        claim for claim in result.claims
        if not any(
            document_evidence(by_id[doc_id].doc_type).proves_payment
            for doc_id in claim.documents
        )
    ]
    if len(result.claims) >= insufficient_evidence_causes_min_run_size():
        assert unpaid, (
            f"no claim of this {len(result.claims)}-claim run is evidenced by a subject document "
            "alone, at or above the run size policy.yaml calls large enough for a zero to mean "
            "something"
        )

    for claim in unpaid:
        assert len(claim.documents) == 1, claim.claim_id
        assert claim.verdict is Verdict.NOT_PROOF_OF_PAYMENT, claim.claim_id
        assert claim.imperfection == [], claim.claim_id
        assert claim.reimbursable_amount == 0, claim.claim_id
        assert claim.linked is False, claim.claim_id
        assert claim.verdict_basis == [VerdictBasis.DOCUMENTS], claim.claim_id
        document = by_id[claim.documents[0]]
        assert document.line_items, (
            f"{document.doc_id} states what was bought and lists nothing"
        )
        assert claim.covered_fraction is not None, (
            f"{claim.claim_id} has line items and no coverage fraction"
        )
        assert document.has_fiscal_number is False, (
            f"{document.doc_id} proves no payment and carries a fiscal number"
        )
        assert "no document proves payment" in " ".join(claim.policy_trace)


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
    # 🔴 The claim's amount is one document's, not the sum. On a split pair the invoice and the
    # payment describe one movement of money, and adding them would count it twice — which is what
    # `resolve_evidence` exists to prevent and what this line now measures rather than assumes.
    assert claim.reimbursable_amount == subject.amount
    if len(result.documents) > 1:
        # The arithmetic that would have gone unnoticed. Both documents of a pair state the same
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
    # "period ok" sits at index 1 since the 2026-08-17 reordering: the period is checked
    # right after the money-moved slot, before the evidence line is written.
    assert claim.policy_trace[1] == "period ok"
    assert claim.policy_trace[-1] == "coverage 100% (all line items covered)"
    assert "1 transaction" in claim.policy_trace[2], claim.policy_trace


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
    """By the payment date, which is what a claim takes its place in the ledger by — see
    `ClaimInput.dated`. Read off the document whose type proves payment, resolved through
    policy.yaml's `document_evidence` exactly as `policy_engine.resolve_evidence` does.

    🔴 it used to take the latest date of the claim's documents, and that was wrong on a
    mechanism this repository deliberately generates. A claim planned with the cause
    `payment_precedes_subject` is dated backwards on purpose — its subject document is later
    than its payment — so the latest date is the subject's, and ordering by it compares a
    claim's subject against the next claim's payment. It passed only while no such claim
    happened to be adjacent to a later one; the check was never testing what it said. It also
    said "every document of this dataset is a fiscal receipt", which stopped being true when
    the invoice archetype landed.

    ⚠️ and A `rejected` claim is out of the sequence altogether, for the same kind of reason and a
    different mechanism: its payment is displaced a whole benefit period out of the window on
    purpose, and it reimburses nothing, so it takes no place in the ledger this order exists to
    keep honest. Dropping it is not a weakening — the claims that consume a balance are still
    required to be ordered by the date they consume it on.

    🔴 and A `not_proof_of_payment` claim has no such date at all, which is a stronger statement
    than being out of the sequence: it carries no document of a payment-proving type, so there is
    nothing to read a payment date off. It is dropped by that property rather than by its verdict
    — the sequence is over payment dates and it has none — and the drop is asserted to be exactly
    those claims below, so a claim that lost its payment document for some other reason cannot
    slip out of the ordering unnoticed.
    """
    result, _ = multi_claim_dataset
    without_a_payment = [
        claim
        for claim, documents in documents_of(result)
        if not any(document_evidence(d.doc_type).proves_payment for d in documents)
    ]
    assert without_a_payment, (
        "no claim of this run lacks a payment document, so the exclusion below removes nothing "
        "and this test asserts the ordering of a set nobody narrowed"
    )
    for claim in without_a_payment:
        assert claim.verdict is Verdict.NOT_PROOF_OF_PAYMENT, (
            f"{claim.claim_id} has no payment document and is labelled {claim.verdict.value}"
        )

    excluded = {claim.claim_id for claim in without_a_payment}
    for persona_record in result.personas:
        dates = [
            max(
                document.date
                for document in documents
                if document_evidence(document.doc_type).proves_payment
            )
            for claim, documents in documents_of(result)
            if claim.persona_id == persona_record.persona_id
            and claim.verdict is not Verdict.REJECTED
            and claim.claim_id not in excluded
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
        # Over the claim's subject documents, which is where the line items of a claim
        # live — see `policy_engine.resolve_evidence`. Taking them from the evidence
        # table rather than from every document is what keeps this re-derivation honest
        # when the two sets stop coinciding.
        #
        # 🔴 Converted exactly where the label says a conversion applied, and by the
        # label's own rate — this loop is the consumer's half of the currency decision. A
        # document whose doc_id appears in the claim's `fx_rates` has its covered sum
        # multiplied by that recorded rate and quantized once, at the point
        # config/fx-rates.yaml declares (sum-then-convert-then-quantize, 0.01, half-up),
        # implemented here rather than imported from the engine: two independent
        # implementations of one declared constant is what the symmetry claim means. A
        # KeyError on the rate lookup is itself a finding — a foreign-currency document
        # whose claim label failed to prove the conversion it underwent.
        rate_of = {applied.doc_id: applied.rate for applied in claim.fx_rates}
        covered = Decimal(0)
        for document in documents:
            if not document_evidence(document.doc_type).proves_subject:
                continue
            doc_covered = covered_total(claim.category, list(document.line_items))
            if document.currency != "UAH":
                doc_covered = (doc_covered * rate_of[document.doc_id]).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            covered += doc_covered
        reimbursed = min(covered, remaining)

        # 🔴 A claim whose evidence is insufficient pays nothing and consumes nothing, so the
        # coverage arithmetic above does not describe it: its documents contradict each other, and
        # what its lines cover is beside the point. It was unreachable until the cross-check causes
        # became drawable, and the re-derivation asserted the covered amount against a
        # `reimbursable_amount` of zero the first time one appeared.
        #
        # The ledger is the part that matters: such a claim must leave the balance untouched, or a
        # persona's later claims would be sized against money nothing ever paid out. That is
        # asserted here by not adding to `spent` and by the running total staying within the limit.
        #
        # 🔴 A `rejected` claim is the second of that kind and reaches it differently: its evidence
        # is complete and its lines may cover the whole basket — `covered` above is a real number
        # for it — and the policy still pays nothing, because the payment fell outside the benefit
        # period. So the coverage arithmetic does not describe it either, and the ledger claim is
        # the same one: a claim outside the window consumes none of a balance that belongs to the
        # window. That neutrality is what lets the planner displace such a claim's date out of the
        # ascending order the other tests check.
        # 🔴 A `not_proof_of_payment` claim is the third of that kind and reaches zero by a third
        # route: it carries no payment document at all, so nothing on it says the money the lines
        # add up to ever moved. The other two have a payment whose relation to the purchase fails
        # a check. The ledger claim is identical for all three and is the part this test is about
        # — a claim that pays nothing consumes nothing.
        # 🔴 A `partially_paid` claim is the fourth, and the only one of the four whose documents
        # are in perfect order. Its evidence is complete, its pair agrees — the payment is exactly
        # the instalment the subject printed — and it still pays nothing, because policy.yaml has
        # not decided how much of a partly settled obligation is payable and the engine will not
        # invent a figure. The ledger claim is the same as for the other three, and it is the one
        # this test is about.
        if claim.verdict in (
            Verdict.INSUFFICIENT_EVIDENCE,
            Verdict.REJECTED,
            Verdict.NOT_PROOF_OF_PAYMENT,
            Verdict.PARTIALLY_PAID,
        ):
            assert claim.reimbursable_amount == 0
            assert "limit_exhausted" not in claim.imperfection
            continue

        assert ("limit_exhausted" in claim.imperfection) is (reimbursed < covered)
        assert claim.reimbursable_amount == reimbursed
        spent[key] += reimbursed
        assert spent[key] <= limit


# A fixed moment for the hand-built plans below — inside the benefit period, so nothing about
# these cases turns on the date.
WHEN = datetime(2026, 6, 3, 10, 15)


def _plan_for_schedule(schedule: str | None, cause: str | None = None) -> ClaimPlan:
    """A minimal plan, for the amount rule below. Built by hand rather than drawn: what is under
    test is what the assembler does with a plan's schedule, and a drawn plan would make the case
    depend on a seed landing on this verdict."""
    from receipt_synth.claim_planner import ARCHETYPES, DocumentPlan

    return ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category="sport",
        verdict=Verdict.PARTIALLY_PAID if schedule else Verdict.COVERED,
        documents=(DocumentPlan(archetype=ARCHETYPES["ua_invoice"], issued_at=WHEN),),
        issued_at=WHEN, cause=cause, schedule=schedule,
    )


def _subject(amount: str, instalment: str | None = None) -> DocGroundTruth:
    return DocGroundTruth(
        doc_id="p001_c1_d1", source_file="p001_c1_d1.png", doc_type=DocType.INVOICE,
        language="uk", currency="UAH", amount=Decimal(amount),
        instalment_amount=None if instalment is None else Decimal(instalment),
        date=WHEN.date(), counterparty="Клуб", line_items=[],
        has_qr=False, qr_is_fiscal=False, has_fiscal_number=False, capture=Capture.SCAN,
    )


def test_the_payment_of_a_partly_settled_claim_states_the_instalment_the_page_printed():
    """🔴 the half of the mechanism that lives in the assembler, tested where it is rather than
    through a rendered run. A payment that stated the invoice's total beside an invoice printing an
    instalment term is not a broken document and not an exception: it is a pair that agrees, so the
    engine labels it `covered`, the corpus silently contains zero claims of this verdict, and the
    only trace is a drift line in the balance report. That is the failure this asserts against.

    Read off the built document, not recomputed. The invoice divided its own total and printed the
    result; a second division here would round apart from it on the first total that does not
    divide evenly, and the payment would then disagree with the term beside it by a kopiyka —
    which the engine labels `amount_mismatch`. So a subject whose printed instalment is not the
    quotient of its own amount is used deliberately below, and the payment has to follow the page.
    """
    from receipt_synth.assembler import _amount_the_payment_states

    rng = random.Random(1)
    # 333.33 is not 1000.00 / 3 exactly, which is the point: a recomputation would say 333.33 too,
    # so the figure is skewed to 300.00 — a value only the document can supply.
    partly = _amount_the_payment_states(
        rng, _plan_for_schedule("quarterly"), _subject("1000.00", "300.00")
    )
    assert partly == Decimal("300.00")

    whole = _amount_the_payment_states(rng, _plan_for_schedule(None), _subject("1000.00"))
    assert whole == Decimal("1000.00"), "an ordinary pair states the same amount twice"


def test_a_plan_settled_in_parts_whose_subject_printed_no_term_is_refused():
    """The plan and the page have come apart, and the failure must land here rather than as a
    wrong label two stages later: the payment would state a part nothing on the page names, and the
    engine would answer `insufficient_evidence` with the cause `amount_mismatch` on a claim that
    was planned as a lawful partial settlement."""
    from receipt_synth.assembler import _amount_the_payment_states

    with pytest.raises(ValueError, match="instalment"):
        _amount_the_payment_states(
            random.Random(1), _plan_for_schedule("quarterly"), _subject("1000.00")
        )


def _plan_for_cause(cause: str | None) -> ClaimPlan:
    """A minimal `insufficient_evidence` plan, for the party rule below. Hand-built for the reason
    `_plan_for_schedule` is: what is under test is what the assembler does with a plan's cause, and
    a drawn plan would make the case depend on a seed landing on it."""
    from receipt_synth.claim_planner import ARCHETYPES, DocumentPlan

    return ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category="sport",
        verdict=Verdict.INSUFFICIENT_EVIDENCE if cause else Verdict.COVERED,
        documents=(
            DocumentPlan(archetype=ARCHETYPES["ua_invoice"], issued_at=WHEN),
            DocumentPlan(archetype=ARCHETYPES["ua_bank_payment_confirmation"], issued_at=WHEN),
        ),
        issued_at=WHEN, cause=cause,
    )


def test_the_payment_of_a_mismatched_party_claim_names_a_seller_the_invoice_does_not():
    """🔴 the half of this mechanism that lives in the assembler, tested where it is rather than
    through a rendered run — the lesson `partially_paid` left behind. A payment handed the claim's
    own vendor is not a broken document and not an exception: it is a pair that agrees, so the
    engine labels it `covered`, the corpus silently contains zero claims of this cause, and the
    only trace is a drift line in the balance report nobody diffs.

    Swept over seeds, not pinned to one. `sport` is served by five sellers in config/vendors.json,
    so a step that returned the claim's own vendor would still differ from it four times in five by
    luck at a single seed; the sweep is what makes the assertion about the code. The second half —
    that an ordinary claim gets the same object back — is the one that would take the whole corpus
    with it, since every other claim's two documents must name one seller.
    """
    from receipt_synth.assembler import _payee_the_payment_names, _pick_vendor

    mismatched = _plan_for_cause(COUNTERPARTY_MISMATCH)
    ordinary = _plan_for_cause(None)
    for seed in range(12):
        rng = random.Random(seed)
        vendor = _pick_vendor(rng, Country.UA, mismatched.category, mixed=False)

        other = _payee_the_payment_names(rng, mismatched, vendor, Country.UA)
        assert other["name"] != vendor["name"], (seed, vendor["name"])

        assert _payee_the_payment_names(rng, ordinary, vendor, Country.UA) is vendor, (
            f"seed {seed}: an ordinary claim's payment must name the claim's own seller"
        )


def test_the_payment_of_a_subject_mismatch_claim_cites_an_invoice_the_claim_does_not_hold():
    """🔴 the half of the `subject_mismatch` mechanism that lives in the assembler, tested where
    it is — the sibling of the payee sweep above, on the transaction's last dimension. A step
    that returned the claim's own reference would leave the pair agreeing, the engine answering
    `covered`, and the corpus with zero claims of the cause — the `partially_paid` lesson.

    Swept over seeds for the collision half — the drawn number must differ from the subject's own
    at every seed — and the ordinary half asserts the same object comes back, since every honest
    claim's citation must stay resolvable.
    """
    from receipt_synth.assembler import _reference_the_payment_cites
    from receipt_synth.content_builder import DocumentReference

    mismatched = _plan_for_cause(SUBJECT_MISMATCH)
    ordinary = _plan_for_cause(None)
    own = DocumentReference(number="1234", issued_at=WHEN)
    for seed in range(12):
        rng = random.Random(seed)
        other = _reference_the_payment_cites(rng, mismatched, own)
        assert other is not own
        assert other.number != own.number, (seed, other.number)

        assert _reference_the_payment_cites(rng, ordinary, own) is own, (
            f"seed {seed}: an ordinary claim's payment must cite the claim's own invoice"
        )


def test_a_forced_subject_mismatch_plan_comes_back_with_the_citation_on_the_page(tmp_path):
    """🔴 the plan → label loop for the subject axis, through the real assembler, builders and
    renderer. Three things have to conspire — the wrong reference drawn, the purpose formula
    forced to cite, and the engine reading both label ends — and a failure of any one leaves the
    claim `covered` here while every unit test above stays green.
    """
    from receipt_synth import assembler

    plan = ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category="sport",
        verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=SUBJECT_MISMATCH,
        documents=(
            DocumentPlan(archetype=ARCHETYPES["ua_invoice"], issued_at=WHEN),
            DocumentPlan(archetype=ARCHETYPES["ua_bank_payment_confirmation"], issued_at=WHEN),
        ),
        issued_at=WHEN,
    )

    def one_plan(rng, **kwargs):
        return iter([plan])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "plan_claims", one_plan)
        result = generate_dataset(
            seed=SEED, out_dir=tmp_path, train_fraction=0.5, personas=1, claims_per_persona=1
        )

    claim = result.claims[0]
    assert claim.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert claim.imperfection == [SUBJECT_MISMATCH]

    by_id = {document.doc_id: document for document in result.documents}
    subject = next(d for d in (by_id[i] for i in claim.documents) if d.line_items)
    payment = next(d for d in (by_id[i] for i in claim.documents) if not d.line_items)

    assert subject.document_code, "the invoice's own № is the axis's right-hand side"
    assert payment.cites_document_no, "the forced citation never reached the label"
    assert payment.cites_document_no != subject.document_code
    # The label is a structured copy of the page, not an extra fact: the cited number is printed
    # in the purpose line, where a reader of the image finds it.
    assert payment.cites_document_no in payment.payment_purpose


def test_the_party_a_document_names_follows_what_that_document_ESTABLISHES(tmp_path):
    """🔴 the other half of the mechanism: which document receives which party. The sweep above
    asserts only which party is drawn; the claim loop then has to hand the invoice's seller to the
    document that states what was bought and the second party to the one that proves the payment,
    and a loop that handed both documents the same party would leave the corpus with zero claims of
    this cause while every engine test stayed green — the `partially_paid` lesson exactly.

    🔴 and it must not depend on a draw landing on A 2.5% cause. The plan is forced — `plan_claims`
    is replaced by one hand-built plan — so this runs the real assembler loop, the real builders and
    the real renderer on a claim that is a party mismatch by construction, at any seed.

    The two sellers are fixed too, and that is what makes the assertion about the dispatch rather
    than about two names being different: `_pick_vendor` is stubbed to answer the claim's draw with
    the first seller and the payee's draw — the one that excludes a name — with the second. So an
    inverted dispatch prints two different names and still fails here, which it could not do if the
    assertion only said "the two disagree". Both entries are real rows of config/vendors.json, so
    the basket and the VAT status the builders derive from a profile stay the ones a real claim has.
    """
    from receipt_synth import assembler

    named = [entry for entry in load_vendors()["vendors"]["UA"]["sport"] if "name" in entry]
    assert len(named) >= 2, "config/vendors.json no longer offers two named sellers for sport"
    seller, payee = dict(named[0]), dict(named[1])
    assert seller["name"] != payee["name"]

    plan = ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category="sport",
        verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=COUNTERPARTY_MISMATCH,
        documents=(
            DocumentPlan(archetype=ARCHETYPES["ua_invoice"], issued_at=WHEN),
            DocumentPlan(archetype=ARCHETYPES["ua_bank_payment_confirmation"], issued_at=WHEN),
        ),
        issued_at=WHEN,
    )

    def one_plan(rng, **kwargs):
        return iter([plan])

    def fixed_vendors(rng, country, category, *, mixed, vat_payer=None, excluding_name=None):
        # The claim's own draw names nothing to exclude; the payee's draw excludes the seller. That
        # is the only difference between the two call sites, and it is what this stub keys on.
        return dict(payee) if excluding_name is not None else dict(seller)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "plan_claims", one_plan)
        patch.setattr(assembler, "_pick_vendor", fixed_vendors)
        result = generate_dataset(
            seed=SEED, out_dir=tmp_path, train_fraction=0.5, personas=1, claims_per_persona=1
        )

    claim = result.claims[0]
    by_id = {document.doc_id: document for document in result.documents}
    documents = [by_id[doc_id] for doc_id in claim.documents]
    assert len(documents) == 2, claim.documents

    for document in documents:
        expected = (
            seller["name"]
            if document_evidence(document.doc_type).proves_subject
            else payee["name"]
        )
        assert document.counterparty == expected, (
            f"{document.doc_id} is a {document.doc_type.value} and names "
            f"{document.counterparty!r}; the party a document names follows what it establishes"
        )

    # And the label the built pair earns, since the whole point of the dispatch is to produce it.
    assert claim.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert claim.imperfection == [COUNTERPARTY_MISMATCH]


def test_a_forced_zero_coverage_plan_comes_back_rejected_with_no_cause(tmp_path):
    """🔴 the plan → label loop for the route by what was bought, through the real assembler,
    builders and renderer — the `partially_paid` lesson applied on arrival rather than after the
    incident. A realizing step that silently stopped firing — a basket that drew one covered line
    after all — would come back `partially_covered` here, not `rejected`, and the corpus would
    hold zero claims of the route while every engine test stayed green.

    Forced, not drawn: the route is a 5% draw, so a run this size would miss it more often than
    not — `plan_claims` is replaced by one hand-built plan, which also pins the claim's date
    inside the window, the property the route exists for.
    """
    from receipt_synth import assembler

    plan = ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category="sport",
        verdict=Verdict.REJECTED, cause=claim_planner.ZERO_COVERAGE,
        documents=(
            DocumentPlan(archetype=ARCHETYPES["ua_invoice"], issued_at=WHEN),
            DocumentPlan(archetype=ARCHETYPES["ua_bank_payment_confirmation"], issued_at=WHEN),
        ),
        issued_at=WHEN, coverage_target=Decimal(0),
    )

    def one_plan(rng, **kwargs):
        return iter([plan])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "plan_claims", one_plan)
        result = generate_dataset(
            seed=SEED, out_dir=tmp_path, train_fraction=0.5, personas=1, claims_per_persona=1
        )

    claim = result.claims[0]
    assert claim.verdict is Verdict.REJECTED
    assert claim.imperfection == [], (
        "the zero-coverage route carries no cause; `outside_period` here means the date "
        "mechanism fired on a claim whose date was pinned inside the window"
    )
    assert claim.covered_fraction == 0.0
    assert claim.reimbursable_amount == 0.0

    by_id = {document.doc_id: document for document in result.documents}
    lines = [
        item for doc_id in claim.documents for item in by_id[doc_id].line_items
    ]
    assert lines, "no document of the claim lists what was bought"
    assert all(not item.covered for item in lines), (
        "a covered line on a zero-coverage claim is the realizing step not firing"
    )


def test_a_party_mismatch_cannot_be_planned_where_the_category_has_one_seller():
    """The failure lands where its cause is. Such a claim needs a category with at least two
    sellers, and `claim_planner` does not model vendors at all — so the refusal is here, naming the
    cause and the category, rather than surfacing as a pair that agrees and a label that drifted.

    ⚠️ unreachable against config/vendors.json, where every Ukrainian category carries five sellers
    or more, and driven by a patched vendor file for that reason."""
    from receipt_synth import assembler

    plan = _plan_for_cause(COUNTERPARTY_MISMATCH)
    only_one = {"vendors": {"UA": {plan.category: [
        {"name": "Sport Life", "legal_form": "TOV", "profile": "gym", "vat_payer": True},
    ]}}}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "load_vendors", lambda: only_one)
        vendor = assembler._pick_vendor(random.Random(1), Country.UA, plan.category, mixed=False)
        with pytest.raises(ValueError, match=COUNTERPARTY_MISMATCH) as refusal:
            assembler._payee_the_payment_names(random.Random(1), plan, vendor, Country.UA)

    # The vendor pool's own refusal is kept as the cause of this one rather than swallowed: it
    # names the seller that was excluded, which is the next thing a reader asks.
    assert "at least two sellers" in str(refusal.value)
    assert "Sport Life" in str(refusal.value.__cause__)


def test_a_partly_settled_claim_carries_its_term_on_the_rendered_page(multi_claim_dataset):
    """end to end, on documents that were actually rendered: the marker the verdict rests on is on
    the page and not only in the label, its box is there for a consumer to read it by, and the
    payment states exactly it.

    ⚠️ vacuous if this fixture happens to contain no such claim, and the honest place to say so is
    here. The fixture builds about 32 claims and this verdict is drawn on a tenth of them, so a run
    without one is a 3% event rather than a defect — too likely to assert against. Existence is
    established elsewhere, at a size derived from the policy:
    `test_partially_paid_is_planned_in_a_run_large_enough_to_require_it` plans enough claims that
    absence is a one-in-a-thousand event. What this test adds is the half that only a built
    document can answer.
    """
    result, _ = multi_claim_dataset

    # 🔴 The mechanism always realizes, and this is where that is asserted rather than assumed.
    # Nothing about a partly settled claim can be defeated by the draw — the term is printed, the
    # payment is read off it, the date is inside the window — so a plan aimed here that came back
    # under another verdict means the payment stopped following the page, which the balance report
    # would show only as a drift line nobody diffs.
    for plan, claim in zip(result.plans, result.claims, strict=True):
        if plan.verdict is Verdict.PARTIALLY_PAID:
            assert claim.verdict is Verdict.PARTIALLY_PAID, claim.claim_id

    for claim, documents in documents_of(result):
        if claim.verdict is not Verdict.PARTIALLY_PAID:
            continue
        subject = next(d for d in documents if document_evidence(d.doc_type).proves_subject)
        payment = next(d for d in documents if not document_evidence(d.doc_type).proves_subject)

        assert subject.instalment_amount is not None, claim.claim_id
        assert "instalment_amount" in subject.field_bboxes, (
            f"{subject.doc_id} carries the instalment in its label and nothing on the page points "
            "at it; a consumer reading the image could not reproduce this verdict"
        )
        assert payment.amount == subject.instalment_amount, claim.claim_id
        assert payment.amount < subject.amount, claim.claim_id
        assert claim.imperfection == [], claim.claim_id
        assert claim.reimbursable_amount == 0, claim.claim_id


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
    recomputable from the per-line labels in the same file.

    🔴 and a claim with no line at all reads `null`, never `0`. A claim planned as an evidence gap
    carries a payment document and nothing that lists items, so there is no fraction to recompute —
    and `0` would say the lines were read and none of them was covered, which is the label of a
    `rejected` claim.

    ⚠️ the branch is not required to be taken at this size. Such a claim arrives on about one built
    claim in twenty-four, and demanding one in ~32 would go red on a shifted seed stream for no
    defect. The property itself is pinned where it cannot be missed — on a hand-built claim in
    tests/test_claim_evidence.py, `test_a_payment_with_nothing_saying_what_it_bought_is_...` — and
    what this adds is that a run carries it through the assembler and into the label file unchanged.
    """
    result, _ = multi_claim_dataset
    for claim, documents in documents_of(result):
        # True for every claim, including the limit-bound ones: `covered_fraction` stays
        # a property of the line items, and the limit is recorded elsewhere in the label.
        lines = [line for document in documents for line in document.line_items]
        if not lines:
            assert claim.covered_fraction is None, (
                f"{claim.claim_id} carries no line item and reports a coverage fraction of "
                f"{claim.covered_fraction} — a fraction over nothing"
            )
            continue
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


def test_the_balance_report_says_nothing_is_missing_when_nothing_is(multi_claim_dataset):
    """🔴 the other side of the absent-mix block, and the state the report has never been in
    before. Every member of `verdict_mix` is realizable, so the whole mix is drawn and the block
    must be silent: a report still announcing a missing fraction over a complete corpus would send
    a reader looking for a bucket that is there, and the "conditional" warning would tell them not
    to read shares that are now unconditional.

    The rows themselves are asserted as before, so this is not merely an assertion of absence.
    """
    result, _ = multi_claim_dataset
    report = balance_report(result)

    assert not unrealizable_verdicts(), "a verdict is unrealizable — the block should have fired"
    assert "NOT GENERATED IN THIS RUN" not in report
    assert "CONDITIONAL" not in report
    assert "of the target mix is absent" not in report
    for verdict in Verdict:
        assert verdict.value in report
        assert f"({verdict_mix()[verdict]:.1%} of the full mix)" in report
    assert "mixed_items" in report and "limit_exhausted" in report


def test_the_balance_report_names_the_verdicts_it_could_not_generate(multi_claim_dataset):
    """A report that renormalized silently would print a tidy table over part of the target mix
    and look balanced. Naming what is absent is the point of the block — and the smaller that
    figure gets, the more a silent renormalization would look like the whole truth.

    ⚠️ driven by a patched registry: `partially_paid` was the last absent member and is now built,
    so no live case is left. The capability outlives the case — the next verdict added to the enum
    is absent from every corpus until its mechanism lands — so the block is exercised against a
    registry narrowed by one member.
    """
    from receipt_synth import assembler as assembler_module

    result, _ = multi_claim_dataset
    absent = Verdict.PARTIALLY_PAID
    narrowed = tuple(v for v in REALIZABLE_VERDICTS if v is not absent)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler_module, "REALIZABLE_VERDICTS", narrowed)
        patch.setattr(assembler_module, "unrealizable_verdicts", lambda: [absent])
        report = balance_report(result)

    assert "NOT GENERATED IN THIS RUN" in report
    assert absent.value in report
    assert "CONDITIONAL" in report
    # 10.0%: `partially_paid` is the one member the patch removes, and it carries 0.10 of the mix.
    assert "10.0% of the target mix is absent" in report
    for verdict in narrowed:
        assert verdict.value in report


def test_a_verdict_with_no_share_is_named_and_does_not_enter_the_arithmetic(
    multi_claim_dataset,
):
    """A member of `verdict_mix` may be declared with no share — `null`, which the loader returns
    as `None` — and the report has to survive that twice over: it must not print a percentage
    where there is none, since `0.0%` would read as a decision somebody took, and the absent
    fraction it prints is then only what the share-carrying absent verdicts account for, so it has
    to be called a lower bound. Silently summing a `None` as zero would leave a complete-looking
    figure on the page.

    ⚠️ driven by a patched mix: `rejected` carried no share in policy.yaml until the planner
    learned to aim at it. The capability is not obsolete with it:
    the loader still returns `None`, `verdict_mix` documents the case, and the next verdict
    declared before its mechanism exists arrives the same way. Patching is what keeps the branch
    under test without a `null` in the policy that nothing needs.

    ⚠️ the registry is patched too, since `partially_paid` became realizable. A share of `None` is
    only coherent on a verdict nothing draws — `draw_verdict` raises otherwise — and the absent-mix
    block that prints the lower bound is reached only when something is absent. So the fabricated
    state is the one that was live a commit ago: this member unrealizable and undeclared.
    """
    from receipt_synth import assembler as assembler_module

    result, _ = multi_claim_dataset
    undeclared = dict(verdict_mix())
    undeclared[Verdict.PARTIALLY_PAID] = None
    narrowed = tuple(v for v in REALIZABLE_VERDICTS if v is not Verdict.PARTIALLY_PAID)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("receipt_synth.assembler.verdict_mix", lambda: undeclared)
        patch.setattr(assembler_module, "REALIZABLE_VERDICTS", narrowed)
        patch.setattr(
            assembler_module, "unrealizable_verdicts", lambda: [Verdict.PARTIALLY_PAID]
        )
        report = balance_report(result)

    assert "partially_paid (no share declared yet)" in report
    assert "partially_paid (0.0%)" not in report
    # 0.0%, and it is the interesting number rather than a degenerate one: `partially_paid` is the
    # only absent member of the mix and the patch removes its share, so what the share-carrying
    # absent verdicts account for is nothing at all — which the report must still print as a lower
    # bound rather than as "the mix is fully covered".
    assert "0.0% of the target mix is absent" in report
    assert "LOWER BOUND" in report
    assert "no share for partially_paid" in report


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
    # ⚠️ The early stop is no longer exercised by this run, and that is a measurement rather than a
    # regression: until the invoice landed a persona had one documentable category, so eight claims
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

    ⚠️ the subset is patched now, and the test used to take its verdict from
    `unrealizable_verdicts()`. That list is empty — every member of the enum is realizable — so
    there is no verdict left to name and none to read off the planner either. Narrowing the
    registry is what supplies one, and the reason the branch is kept at all is unchanged: the enum
    grows, and the first run after a member is added and before its mechanism exists is exactly
    this state.
    """
    from receipt_synth import assembler as assembler_module
    from receipt_synth.assembler import Dataset

    verdict = Verdict.PARTIALLY_PAID
    narrowed = tuple(v for v in REALIZABLE_VERDICTS if v is not verdict)
    assert len(narrowed) == len(REALIZABLE_VERDICTS) - 1, "the patch removed nothing"

    claim = ClaimGroundTruth(
        claim_id="p001_c1", persona_id="p001", category="vitamins_nutrition",
        documents=["p001_c1_d1"], verdict=verdict,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler_module, "REALIZABLE_VERDICTS", narrowed)
        patch.setattr(assembler_module, "unrealizable_verdicts", lambda: [verdict])
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
    """`CAPTURE_DRAW_ORDER` is what the assembler walks when it draws, and `Capture` is what the
    label may carry. A channel present in the enum and missing from the tuple would be a value the
    contract declares and no draw can ever return — unreachable rather than merely rare, and
    nothing about a run would say so."""
    from receipt_synth.assembler import CAPTURE_DRAW_ORDER

    assert set(CAPTURE_DRAW_ORDER) == set(Capture)
    assert len(CAPTURE_DRAW_ORDER) == len(Capture), "a channel is listed twice, which weights it"


def test_a_run_of_any_size_contains_more_than_one_capture_channel(multi_claim_dataset):
    """🔴 the channel is drawn, not fixed. Every earlier version of this generator produced a
    corpus of screenshots only, and nothing in the labels distinguished "this channel was chosen"
    from "this channel is the only one there is". The draw is per class now (`capture_mix` in
    policy.yaml), and the run's majority classes each give their thinnest declared channel at
    least a fifth of the weight — so over fifty-odd documents every channel appears with a
    probability that rounds to one, and anything less is the draw being gone rather than the run
    being unlucky. (The per-class shape itself — a zero-weight channel never drawn, a screen-born
    archetype always a screenshot — is tests/test_capture_mix.py's job.)

    It matters beyond variety: the two paper channels are the only route by which the electronic
    side of the VAT-row rule stops being the whole corpus. See `test_vat_row_form.py`.
    """
    result, _ = multi_claim_dataset
    assert len(result.documents) > 30, "too few documents for this test to mean anything"
    assert {document.capture for document in result.documents} == set(Capture)


def test_a_capture_channel_with_no_documents_is_reported_ABSENT_and_never_as_zero():
    """🔴 absent and zero are different statements and only one of them can be true.

    `0.0%` complete says a measurement was taken and came out at nothing; absence says no
    measurement exists. Folding the first into the second is how an empty cell becomes a data point
    in somebody's table, and nobody re-derives it afterwards.

    Asserted in both directions: the channel that has documents must carry a completeness fraction,
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
    """🔴 the integrity the partition exists to keep. An invoice in train and the payment that
    settles it in validation is one transaction split across the boundary — the model would see the
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
    """On disk, not only in memory — a consumer reads the files. Both record kinds, because they are
    populated by two separate statements and only one of them was checked at first.

    🔴 the totals are compared against the corpus size, not against the records. Comparing the
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
    # 0.5 because this fixture passed 0.5, not because anything defaults to it — nothing does any
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

    🔴 `--split` is supplied so that only the seed is missing. It became required too, and a call
    omitting both raises `SystemExit` whichever of them argparse is enforcing — so the test would
    have stayed green with `--seed` defaulted again, asserting nothing. The message is checked for
    the same reason: the raise alone does not say which argument produced it.
    """
    with pytest.raises(SystemExit):
        main(["--out", "out", "--split", "0.5"])
    assert "--seed" in capsys.readouterr().err
