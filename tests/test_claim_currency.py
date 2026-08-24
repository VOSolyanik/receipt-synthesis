"""One claim, one currency — as a rule about which documents may be planned together.

`policy_engine._one_claim_one_currency` refuses to label a claim whose documents are stated in two
currencies, and `tests/test_claim_evidence.py` pins that refusal. This file is the other end of the
same fact: the planner must not build such a claim, because the refusal it would meet is not a
verdict — it is the oracle saying policy.yaml answers no question about the pair at all.

⚠️ every registry here is hand-built. The shipped one is checked separately, in
`test_the_shipped_registry_pairs_only_within_a_currency`, and these fixtures pin the mechanism: it
has to hold for any set of archetypes, not for the set that happens to be registered this week.
"""

from __future__ import annotations

import random
from datetime import datetime

import pytest

from receipt_synth.claim_planner import (
    _SUBJECT_LEAD_DAYS,
    ARCHETYPES,
    Archetype,
    EvidenceIntent,
    _pairable_subjects,
    _payment_archetypes,
    _select_documents,
    _settleable_instalment_subjects,
    _settleable_subjects,
    can_assemble_evidence,
    evidence_of,
)
from receipt_synth.policy_engine import Evidence
from receipt_synth.schemas import Country, DocType

CATEGORY = "professional_development"
ISSUED_AT = datetime(2026, 6, 15, 12, 0)


def _archetype(slug: str, doc_type: DocType, currency: str) -> Archetype:
    return Archetype(
        slug=slug,
        doc_type=doc_type,
        country=Country.UA,
        language="uk",
        currency=currency,
        categories=(CATEGORY,),
    )


UAH_INVOICE = _archetype("uah_invoice", DocType.INVOICE, "UAH")
UAH_TRANSFER = _archetype("uah_transfer", DocType.PAYMENT_CONFIRMATION, "UAH")
EUR_INVOICE = _archetype("eur_invoice", DocType.INVOICE, "EUR")
EUR_TRANSFER = _archetype("eur_transfer", DocType.PAYMENT_CONFIRMATION, "EUR")


# ================================================ the pair the planner refuses to build ==


def test_a_subject_no_payment_shares_a_currency_with_is_not_half_of_a_pair():
    """The case the rule exists for: both facts are covered, and by documents nothing may
    put in one claim. "A subject exists and a payment exists" is true here and
    `can_assemble_evidence` still has to answer no."""
    candidates = [EUR_INVOICE, UAH_TRANSFER]

    assert _pairable_subjects(candidates) == [EUR_INVOICE]
    assert _payment_archetypes(candidates) == [UAH_TRANSFER]
    assert _settleable_subjects(_pairable_subjects(candidates), [UAH_TRANSFER]) == []
    assert not can_assemble_evidence(candidates)


def test_the_planner_refuses_that_pair_rather_than_building_it():
    """And the refusal names the archetypes, because the repair is to the registry."""
    with pytest.raises(ValueError, match="no registered archetype"):
        _select_documents(random.Random(1), [EUR_INVOICE, UAH_TRANSFER], ISSUED_AT)


def test_a_pair_in_one_currency_is_buildable_again():
    """The same shape with the payment restated — nothing else about the registry moves."""
    assert can_assemble_evidence([EUR_INVOICE, EUR_TRANSFER])

    documents = _select_documents(random.Random(1), [EUR_INVOICE, EUR_TRANSFER], ISSUED_AT)

    assert [d.archetype.slug for d in documents] == ["eur_invoice", "eur_transfer"]


# ================================================================ the draw, two buckets ==


def test_every_pair_drawn_from_a_two_currency_registry_states_one_currency():
    """🔴 the property the oracle depends on. A registry holding both buckets draws from both,
    and no draw ever crosses them — the subject is drawn first and the payment from the
    subject's own currency, so a EUR invoice cannot pick up a UAH transfer at any seed."""
    candidates = [UAH_INVOICE, UAH_TRANSFER, EUR_INVOICE, EUR_TRANSFER]
    drawn = set()

    for seed in range(200):
        documents = _select_documents(random.Random(seed), candidates, ISSUED_AT)
        currencies = {d.archetype.currency for d in documents}
        assert len(currencies) == 1, (
            f"seed {seed} planned a claim in {sorted(currencies)}, which no policy can label"
        )
        drawn |= {tuple(d.archetype.slug for d in documents)}

    assert drawn == {
        ("uah_invoice", "uah_transfer"),
        ("eur_invoice", "eur_transfer"),
    }, "both buckets have to be reachable, or this test asserts nothing about the EUR one"


def test_the_narrowing_leaves_a_single_currency_registry_drawing_exactly_as_before():
    """⚠️ the compatibility claim, written down. The filter drops nothing when every archetype
    shares one currency, and the two draws happen in the order they always have — so a run over
    the UAH registry takes the same values from the generator as it did before the rule existed.
    Pinned by the seed stream itself: three values are taken — the lead, the subject, the payment
    — and what a fourth draw would return is what says the generator is in the same state."""
    candidates = [UAH_INVOICE, UAH_TRANSFER]
    rng = random.Random(7)

    _select_documents(rng, candidates, ISSUED_AT)

    reference = random.Random(7)
    reference.randint(0, _SUBJECT_LEAD_DAYS)
    reference.choice(candidates)
    reference.choice(candidates)
    assert rng.random() == reference.random()


# ============================================================ the composed instalment rule ==


def test_an_instalment_subject_no_payment_can_settle_is_not_plannable():
    """`partially_paid` needs a subject that states the arrangement and a payment that can
    settle a part of it. The currency narrowing composes with the class narrowing rather than
    replacing it, and either one alone would admit this registry."""
    candidates = [EUR_INVOICE, UAH_TRANSFER]

    assert _settleable_instalment_subjects(candidates) == []
    assert _settleable_instalment_subjects([EUR_INVOICE, EUR_TRANSFER]) == [EUR_INVOICE]


# ============================================================== the registry that ships ==


def test_the_shipped_registry_pairs_only_within_a_currency():
    """Every currency the registry states a subject in has a payment document beside it.

    ⛔ Not a restatement of the rule — it is the check that the registry can actually use every
    subject it holds. A subject archetype in a currency no payment document is stated in is a
    template that renders and never reaches a complete claim, which is a gap in the registry
    rather than in the planner.
    """
    for category in {c for a in ARCHETYPES.values() for c in a.categories}:
        candidates = [a for a in ARCHETYPES.values() if category in a.categories]
        payments = _payment_archetypes(candidates)
        unsettleable = set(_pairable_subjects(candidates)) - set(
            _settleable_subjects(_pairable_subjects(candidates), payments)
        )
        assert not unsettleable, (
            f"{sorted(a.slug for a in unsettleable)} state the subject of a "
            f"{category!r} claim in a currency no registered payment document is stated in"
        )


def test_a_currency_coherent_pair_is_also_a_single_vendor_pool():
    """🔴 the second constraint on a pair, and it is enforced somewhere else. `assembler` draws one
    seller per claim and refuses a plan whose documents name two pools; the planner does not model
    vendors at all. The two rules are independent and they must not disagree, so this asserts the
    registry satisfies both at once — a EUR pair split across an `EU` and a domestic pool would
    plan cleanly here and raise a stage later."""
    for category in {c for a in ARCHETYPES.values() for c in a.categories}:
        candidates = [a for a in ARCHETYPES.values() if category in a.categories]
        payments = _payment_archetypes(candidates)
        for subject in _settleable_subjects(_pairable_subjects(candidates), payments):
            pools = {
                payment.vendor_pool
                for payment in payments
                if payment.currency == subject.currency
            }
            assert pools == {subject.vendor_pool}, (
                f"{subject.slug} could be paired with a payment drawing its seller from "
                f"{sorted(str(p) for p in pools)}, and one claim names one seller"
            )


def test_a_self_contained_document_is_unaffected_by_the_currency_rule():
    """⛔ The rule is about pairs. A document proving both facts is the whole claim, so there is no
    second currency for it to disagree with — `eu_platform_receipt` goes on being drawn alone."""
    receipt = ARCHETYPES["eu_platform_receipt"]
    assert evidence_of(receipt) == Evidence(True, True)
    assert receipt.currency == "EUR"

    documents = _select_documents(
        random.Random(3), [receipt, UAH_TRANSFER], ISSUED_AT, intent=EvidenceIntent.COMPLETE
    )

    assert [d.archetype.slug for d in documents] == ["eu_platform_receipt"]
