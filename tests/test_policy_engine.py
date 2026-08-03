"""The oracle's own tests.

Every expected value here is computed by hand from `config/policy.yaml` and written down
with its arithmetic. None of it was obtained by running the engine and recording what it
printed — an oracle validated against itself is not an oracle. The constants the
arithmetic uses are asserted against the policy file in the first test below, so that a
change to the policy fails these tests loudly instead of quietly making them describe
something else.

The policy values these tests are derived from, as of policy.yaml version 1:

    coverage.full_threshold                     0.9999
    categories[vitamins_nutrition].annual_limit 12000
    categories[language_courses].annual_limit   12000
    categories[mental_health].annual_limit      25000
    period                                      2026-01-01 .. 2026-12-31
    reporting_currency                          UAH
    verdict_mix                                 covered 0.50, partially_covered 0.20,
                                                not_proof_of_payment 0.10,
                                                insufficient_evidence 0.10,
                                                partially_paid 0.10, rejected null

and, for the item-kind vocabulary:

    vitamins_nutrition.covered_items    vitamin_complex, mineral_supplement, nutritionist_visit
    vitamins_nutrition.excluded_items   medicine, medical_device, cosmetics, hygiene
    vitamins_nutrition.ambiguous_items  lab_test, sports_nutrition, probiotic
    sport.excluded_items                sport_apparel, sports_nutrition, food_drink
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from receipt_synth.config import category, load_policy
from receipt_synth.policy_engine import (
    OUTSIDE_PERIOD,
    ClaimInput,
    Ledger,
    PolicyGapError,
    active_period,
    annual_limit,
    coverage_of_kind,
    covered_fraction,
    covered_total,
    evaluate_claim,
    evaluate_claims,
    full_threshold,
    reporting_currency,
    resolved_coverage,
    verdict_for,
    verdict_mix,
)
from receipt_synth.schemas import (
    Capture,
    DocGroundTruth,
    DocType,
    LineItem,
    Verdict,
    VerdictBasis,
)

IN_PERIOD = date(2026, 6, 15)

# Kinds used to build test baskets, chosen so that the flag on the line and the bucket in
# policy.yaml agree — which the engine now insists on.
COVERED_KIND = "vitamin_complex"  # vitamins_nutrition.covered_items
EXCLUDED_KIND = "medical_device"  # vitamins_nutrition.excluded_items


def item(price: str, covered: bool, qty: int = 1, kind: str | None = None) -> LineItem:
    if kind is None:
        kind = COVERED_KIND if covered else EXCLUDED_KIND
    return LineItem(
        name=f"{kind} {price}",
        item_kind=kind,
        qty=Decimal(qty),
        price=Decimal(price),
        covered=covered,
        vat_letter="А",
    )


def document(
    items: list[LineItem],
    when: date = IN_PERIOD,
    doc_id: str = "d1",
    currency: str = "UAH",
) -> DocGroundTruth:
    return DocGroundTruth(
        doc_id=doc_id,
        source_file=f"{doc_id}.png",
        doc_type=DocType.FISCAL_RECEIPT,
        language="uk",
        currency=currency,
        amount=sum((i.qty * i.price for i in items), Decimal(0)),
        date=when,
        counterparty="Vendor",
        line_items=items,
        has_qr=True,
        qr_is_fiscal=True,
        has_fiscal_number=True,
        capture=Capture.SCREENSHOT,
    )


def evaluate(
    items: list[LineItem],
    *,
    category: str = "vitamins_nutrition",
    ledger: Ledger | None = None,
    persona_id: str = "p001",
    when: date = IN_PERIOD,
    currency: str = "UAH",
):
    return evaluate_claim(
        persona_id=persona_id,
        category=category,
        documents=[document(items, when=when, currency=currency)],
        ledger=ledger,
    )


# ------------------------------------------------- the policy these tests assume --


def test_the_constants_this_file_was_written_against():
    """Pin the policy values every expectation below was derived from.

    If policy.yaml changes, this fails first and says so, instead of the rest of the file
    quietly asserting arithmetic that no longer follows from the policy.
    """
    assert full_threshold() == Decimal("0.9999")
    assert annual_limit("vitamins_nutrition") == Decimal("12000")
    assert annual_limit("language_courses") == Decimal("12000")
    assert annual_limit("mental_health") == Decimal("25000")
    assert active_period() == (date(2026, 1, 1), date(2026, 12, 31))
    assert reporting_currency() == "UAH"
    assert load_policy()["limits"]["enforce_cumulative"] is True

    spec = category("vitamins_nutrition")
    assert COVERED_KIND in spec["covered_items"]
    assert EXCLUDED_KIND in spec["excluded_items"]


def test_verdict_mix_names_every_verdict_and_every_one_now_carries_a_share():
    """policy.yaml declares a share for all six, and the file's own rule is that the shares sum
    to 1.0 — the arithmetic is hand-computed here rather than summed twice:

        0.40 + 0.20 + 0.10 + 0.10 + 0.10 + 0.10 = 1.00

    ⚠️ `rejected` USED TO BE THE EXCEPTION, declared as `null` while nothing could build such a
    claim. It gained 0.10 when the planner learned to date a payment outside the benefit period,
    and `covered` gave up exactly that much — which is why the sum is unchanged and why the
    realizable subset still totals 0.80. The undecided-share MECHANISM is untouched and still
    tested, one test below: the next verdict declared before its mechanism exists arrives the
    same way.
    """
    mix = verdict_mix()
    assert set(mix) == set(Verdict), "every verdict is named in the mix"
    assert all(share is not None for share in mix.values()), (
        f"a verdict is declared with no share: {mix}"
    )
    assert mix[Verdict.COVERED] == pytest.approx(0.40)
    assert mix[Verdict.REJECTED] == pytest.approx(0.10)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_an_undecided_share_comes_back_as_none_and_never_as_zero():
    """The loader branch that survives `rejected` gaining a share, asserted on a patched policy
    because the file no longer contains a `null`.

    `None` rather than 0, and the difference is the whole point: a 0 share is a decision — "this
    verdict is deliberately never drawn" — it sums like any other weight, it renormalizes like any
    other weight, and nothing downstream could tell it apart from a share somebody chose. `None`
    is not a weight at all, so every reader has to say what it does with an undecided share. The
    balance report and `claim_planner.draw_verdict` both do, and both are tested against a mix
    patched the same way.
    """
    from receipt_synth import policy_engine

    policy = dict(load_policy())
    policy["verdict_mix"] = dict(policy["verdict_mix"], partially_paid=None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy_engine, "load_policy", lambda: policy)
        mix = verdict_mix()

    assert mix[Verdict.PARTIALLY_PAID] is None
    assert mix[Verdict.PARTIALLY_PAID] != 0
    assert set(mix) == set(Verdict)


# ------------------------------------------- coverage resolved from the policy --


def test_a_covered_kind_of_the_category_is_covered():
    assert coverage_of_kind("vitamins_nutrition", "vitamin_complex") is True
    assert coverage_of_kind("vitamins_nutrition", "nutritionist_visit") is True


def test_an_excluded_kind_of_the_category_is_not_covered():
    assert coverage_of_kind("vitamins_nutrition", "medicine") is False
    assert coverage_of_kind("vitamins_nutrition", "cosmetics") is False


def test_coverage_is_scoped_to_the_category_not_global_to_the_kind():
    """policy.yaml says so in a comment on `sport.excluded_items.sports_nutrition`: the
    same kind is ambiguous under `vitamins_nutrition`. A resolver keyed on the kind alone
    would answer one of the two wrongly and never notice."""
    assert coverage_of_kind("sport", "sports_nutrition") is False
    with pytest.raises(PolicyGapError):
        coverage_of_kind("vitamins_nutrition", "sports_nutrition")


def test_an_ambiguous_kind_has_no_coverage_answer():
    """`ambiguous_items` is the bucket policy.yaml deliberately leaves unanswered. The
    refusal lives in the oracle rather than in a convention `content_builder` follows, so
    that a future builder cannot quietly start drawing from it."""
    for kind in category("vitamins_nutrition")["ambiguous_items"]:
        with pytest.raises(PolicyGapError, match="ambiguous_items"):
            coverage_of_kind("vitamins_nutrition", kind)


def test_a_kind_from_another_category_has_no_coverage_answer():
    """`gym_membership` is a `sport` kind. Scored against `vitamins_nutrition` it belongs
    to no bucket, so there is no coverage answer to give — and an engine that computed a
    fraction over it would be reporting a number about nothing."""
    with pytest.raises(PolicyGapError, match="no bucket"):
        coverage_of_kind("vitamins_nutrition", "gym_membership")


def test_the_engine_refuses_a_basket_belonging_to_another_category():
    """The pass-through test. A claim filed under `mental_health` carrying pharmacy lines
    used to produce a contented `covered` at 100%: the engine summed a boolean somebody
    else had set and never asked the policy anything."""
    with pytest.raises(PolicyGapError):
        evaluate([item("500.00", True)], category="mental_health")


def test_a_line_whose_flag_contradicts_the_policy_is_refused():
    """`LineItem.covered` is an assertion, not an input. Preferring the policy would hide
    a builder bug; preferring the flag would make this module a pass-through. Exactly one
    of them is wrong and nothing in the ground truth can say which, so neither is used."""
    with pytest.raises(ValueError, match="covered_items"):
        resolved_coverage("vitamins_nutrition", [item("100.00", False, kind=COVERED_KIND)])
    with pytest.raises(ValueError, match="excluded_items"):
        resolved_coverage("vitamins_nutrition", [item("100.00", True, kind=EXCLUDED_KIND)])


def test_an_agreeing_basket_resolves_cleanly():
    items = [item("100.00", True), item("50.00", False)]
    assert resolved_coverage("vitamins_nutrition", items) == [True, False]


# ------------------------------------------------------------ covered_fraction --


def test_covered_fraction_is_covered_amount_over_total():
    """qty × price, summed:
        covered      1 × 300.00 + 2 × 100.00 = 500.00
        not covered  1 × 500.00              = 500.00
        total                                  1000.00
        fraction     500.00 / 1000.00        = 0.5
    """
    items = [item("300.00", True), item("100.00", True, qty=2), item("500.00", False)]
    assert covered_total("vitamins_nutrition", items) == Decimal("500.00")
    assert covered_fraction("vitamins_nutrition", items) == Decimal("0.5")


def test_covered_fraction_of_a_fully_covered_basket_is_one():
    items = [item("100.00", True), item("250.00", True, qty=2)]
    # 100.00 + 500.00 = 600.00 covered of 600.00 total.
    assert covered_fraction("vitamins_nutrition", items) == Decimal(1)


def test_covered_fraction_of_a_basket_with_nothing_covered_is_zero():
    items = [item("100.00", False), item("400.00", False)]
    assert covered_fraction("vitamins_nutrition", items) == Decimal(0)


def test_an_empty_basket_has_no_covered_fraction():
    """0/0 is not 0. A claim whose documents carry no line at all states an amount
    nothing accounts for, and the engine must say so rather than label it."""
    with pytest.raises(ValueError):
        covered_fraction("vitamins_nutrition", [])


def test_a_zero_amount_basket_has_no_covered_fraction():
    with pytest.raises(ValueError):
        covered_fraction("vitamins_nutrition", [item("0.00", True)])


# --------------------------------------------------------- the STRICT rule ----


def test_a_fully_covered_basket_is_covered():
    result = evaluate([item("100.00", True), item("250.00", True, qty=2)])
    assert result.verdict is Verdict.COVERED
    assert result.covered_fraction == Decimal(1)
    assert result.imperfection == ()
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)


def test_the_policys_own_example_a_three_hryvnia_item_on_a_thousand_hryvnia_receipt():
    """policy.yaml, `coverage`: "A 3 UAH carrier bag on a 1000 UAH pharmacy receipt is
    not reimbursed, and the label must say so."

        covered   997.00
        total    1000.00
        fraction    0.997   ->  a non-covered line exists  ->  partially_covered
    """
    result = evaluate([item("997.00", True), item("3.00", False)])
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.covered_fraction == Decimal("0.997")
    assert result.imperfection == ("mixed_items",)
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)


def test_one_kopiyka_not_covered_is_partially_covered_at_any_scale():
    """The STRICT rule, stated in policy.yaml as "ANY non-covered line item makes the claim
    partially covered, however small".

        covered   999.99
        total    1000.00
        fraction    0.99999

    The fraction is *above* full_threshold (0.9999). Under a numeric branch this would
    read `covered`; under the rule policy.yaml actually states it does not, because a line
    that is not reimbursed is on the document whatever it costs.
    """
    result = evaluate([item("999.99", True), item("0.01", False)])
    assert result.covered_fraction == Decimal("0.99999")
    assert result.covered_fraction > full_threshold()
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.imperfection == ("mixed_items",)


def test_a_non_covered_line_exactly_at_the_threshold_is_still_partially_covered():
    """    covered  9999.00
        total   10000.00
        fraction   0.9999  ==  full_threshold, and still partially covered.
    """
    result = evaluate([item("9999.00", True), item("1.00", False)])
    assert result.covered_fraction == full_threshold()
    assert result.verdict is Verdict.PARTIALLY_COVERED


def test_a_non_covered_line_cannot_be_diluted_away_by_adding_documents():
    """The case a per-document argument misses. `ClaimInput.documents` has no cardinality
    bound — docs/architecture.md#claim-level says a claim may span several — so a fraction
    branch can be walked past the threshold simply by attaching more covered documents,
    even while the whole claim stays inside a 12000 annual limit.

        ten documents of one covered line each   10 × 1190.00 = 11900.00
        one document with a single non-covered line of         1.00
        total                                                11901.00
        fraction  11900.00 / 11901.00 = 0.9999159734…

        full_threshold × total = 0.9999 × 11901.00 = 11899.80990
        11900.00 >= 11899.80990, so a numeric branch reads `covered`.

    11900.00 is below the 12000 limit, so nothing here is about the ledger: the claim is
    partially covered because one of its eleven lines is not reimbursed, full stop.
    """
    documents = [
        document([item("1190.00", True)], doc_id=f"d{index}") for index in range(10)
    ]
    documents.append(document([item("1.00", False)], doc_id="d10"))

    result = evaluate_claim(
        persona_id="p001", category="vitamins_nutrition", documents=documents
    )
    total = Decimal("11901.00")
    assert result.covered_fraction == Decimal("11900.00") / total
    assert Decimal("11900.00") >= full_threshold() * total, "a numeric branch says covered"
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.imperfection == ("mixed_items",)
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert result.reimbursable == Decimal("11900.00")
    assert result.policy_trace[-1] == "coverage 99.99% (1 of 11 line items not covered)"


def test_a_wholly_non_covered_basket_is_rejected():
    """    covered   0.00
        total   500.00
        fraction    0  ->  rejected

    policy.yaml's `coverage` block sends a zero covered fraction to `rejected`, and the
    enum now carries that member, so there is one answer where there used to be a slash.

    The whole claim, hand-derived:
        verdict             rejected — nothing on the document belongs to the category
        covered_fraction    0.00 / 500.00 = 0
        reimbursable        0.00 — the plan pays out on the covered amount, and there is none
        verdict_basis       ["documents"] — coverage is resolved from the line items alone,
                            which policy.yaml's `limits` block defines as the documents-only
                            basis; no ledger was consulted
        imperfection        () — the coverage route to `rejected` names no cause, because
                            the verdict says the whole of it. The other route to the same
                            verdict, a payment outside the period, does carry one; see
                            `test_the_two_routes_to_rejected_are_told_apart_by_their_cause`
    """
    result = evaluate([item("100.00", False), item("400.00", False)])
    assert result.verdict is Verdict.REJECTED
    assert result.covered_fraction == Decimal(0)
    assert result.reimbursable == Decimal("0.00")
    assert result.imperfection == ()
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — d1 (fiscal_receipt) proves both",
        "period ok",
        "coverage 0% (2 of 2 line items not covered)",
    )

    assert covered_fraction(
        "vitamins_nutrition", [item("100.00", False), item("400.00", False)]
    ) == Decimal(0)


def test_rejected_is_not_not_proof_of_payment():
    """Two outcomes, two members, and the pharmacy receipt is the case that separates them.

    A receipt listing nothing but medicines establishes payment perfectly well — it is
    about the wrong subject, which is `rejected`. `not_proof_of_payment` is a property of
    the document type, declared by `proves_payment: false` in `document_evidence`, and
    `verdict_for` sees only amounts, so it could not decide that question even if asked.
    """
    assert verdict_for(
        Decimal(0), Decimal("500.00"), every_line_covered=False
    ) is Verdict.REJECTED
    assert Verdict.REJECTED is not Verdict.NOT_PROOF_OF_PAYMENT
    assert Verdict.NOT_PROOF_OF_PAYMENT in Verdict


def test_a_rejected_claim_consumes_no_balance():
    """policy.yaml binds annual limits on cumulative *spend*, and the ledger records what a
    claim was reimbursed. A rejected claim is reimbursed nothing, so it cannot move the
    balance — and a persona is not punished for filing a claim the plan turned down."""
    ledger = Ledger()
    result = evaluate([item("500.00", False)], ledger=ledger)
    ledger.record("p001", "vitamins_nutrition", result.reimbursable)

    assert result.verdict is Verdict.REJECTED
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal("12000")


def test_the_threshold_is_a_self_check_on_a_fully_covered_claim():
    """`full_threshold` is kept, in the role its own comment gives it — absorbing
    arithmetic error, not absorbing items.

        0.9999 × 100.00 = 99.99, so a claim with no non-covered line whose covered amount
        came to 99.00 has lost 1.00 somewhere. That is a bug, and it raises instead of
        being rounded into a verdict.
    """
    assert verdict_for(
        Decimal("100.00"), Decimal("100.00"), every_line_covered=True
    ) is Verdict.COVERED
    with pytest.raises(ArithmeticError, match="full_threshold"):
        verdict_for(Decimal("99.00"), Decimal("100.00"), every_line_covered=True)


def test_the_threshold_is_never_consulted_when_a_line_is_not_covered():
    """Whatever the amounts say, a non-covered line decides the verdict on its own."""
    for covered in (Decimal("99.99"), Decimal("1.00")):
        assert verdict_for(
            covered, Decimal("100.00"), every_line_covered=False
        ) is Verdict.PARTIALLY_COVERED


def test_the_cheapest_real_article_is_still_absorbed_by_a_numeric_branch():
    """Why the structural rule is not merely tidier. The cheapest line the generator can
    print is a real article a human would see on the paper — a 15 UAH pen, at the low end
    of `stationery` in config/generation.yaml — and at that size the ratio is swallowed:

        covered  1000000.00
        not covered    15.00
        total    1000015.00
        fraction   0.999985000…
        full_threshold × total = 0.9999 × 1000015.00 = 999914.99850
        1000000.00 >= 999914.99850

    A numeric branch reads that as fully covered. The rule policy.yaml states does not,
    and this asserts the difference on `verdict_for` itself, with no ledger involved. The
    price is read from config rather than restated: it is data, and pinning it here would
    make an unrelated price edit fail this test instead of the rule it is about.
    """
    from receipt_synth.config import load_generation, price_range

    kinds = load_generation()["price_ranges"]["ranges"]
    cheapest = min(price_range(kind)[0] for kind in kinds)
    # Read from config rather than restated, so the worked example above stays an
    # illustration and this stays an assertion about `verdict_for`.
    assert Decimal(0) < cheapest < Decimal("100.00")

    covered, total = Decimal("1000000.00"), Decimal("1000000.00") + cheapest
    assert covered >= full_threshold() * total, "the numeric branch would say covered"
    assert verdict_for(covered, total, every_line_covered=True) is Verdict.COVERED
    assert verdict_for(covered, total, every_line_covered=False) is Verdict.PARTIALLY_COVERED


# ---------------------------------------------------------------- currency ----


def test_a_document_in_another_currency_is_refused_not_converted():
    """policy.yaml expresses limits in `reporting_currency`. Every archetype this
    generator has emits that currency, so a document in anything else means something
    upstream is wrong — and a converted amount would be a number in the ground truth that
    nothing in the dataset can prove."""
    with pytest.raises(ValueError, match="EUR"):
        evaluate([item("900.00", True)], currency="EUR")


def test_the_currency_check_names_the_reporting_currency():
    with pytest.raises(ValueError, match=reporting_currency()):
        evaluate([item("900.00", True)], currency="PLN")


def test_a_foreign_currency_document_cannot_consume_a_limit():
    """The failure this guard exists for: a 900.00 EUR document quietly taking 900.00 out
    of a 12000 UAH balance, with the trace printing UAH beside numbers that are not."""
    ledger = Ledger()
    with pytest.raises(ValueError):
        evaluate([item("900.00", True)], currency="EUR", ledger=ledger)
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal("12000")


# ------------------------------------------------------- cumulative limits ----


def test_a_claim_that_exactly_exhausts_the_limit_is_still_covered():
    """vitamins_nutrition limit 12000. Already reimbursed 9500.00, so 2500.00 remains.
    The claim's covered amount is exactly 2500.00 — it fits, nothing is clamped.

        reimbursable  min(2500.00, 2500.00) = 2500.00  ->  the clamp does not bind
        verdict       covered, from documents alone
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("9500.00"))
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal("2500.00")

    result = evaluate([item("1250.00", True, qty=2)], ledger=ledger)
    assert result.verdict is Verdict.COVERED
    assert result.reimbursable == Decimal("2500.00")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert result.imperfection == ()


def test_a_claim_that_exceeds_the_limit_becomes_partially_covered():
    """Already reimbursed 10500.00 of 12000, so 1500.00 remains.

        covered on the document  2000.00
        reimbursable             min(2000.00, 1500.00) = 1500.00  ->  the clamp binds
        verdict                  partially_covered, cause limit_exhausted
        verdict_basis            documents + account_state — no one can read a remaining
                                 balance off a receipt

    `covered_fraction` stays 1.0: it is a property of the line items, and every line on
    this document is covered. What the plan pays out is `reimbursable`, a separate number.
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("10500.00"))

    result = evaluate([item("1000.00", True, qty=2)], ledger=ledger)
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("1500.00")
    assert result.imperfection == ("limit_exhausted",)
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS, VerdictBasis.ACCOUNT_STATE)


def test_a_claim_beyond_a_fully_exhausted_limit_has_no_verdict_in_the_policy():
    """The gap this engine refuses to paper over.

    With 12000 of 12000 already reimbursed, a further claim reimburses nothing.
    policy.yaml assigns no verdict to that state, and none of the three candidates closes it.
    `rejected` is the policy not covering the claim on either of its two axes, and here both
    say it is covered — every line qualifies and the payment is inside the period; the only
    reason nothing is paid out is that the persona has already drawn the annual maximum.
    `not_proof_of_payment` is for a claim whose document types all carry
    `proves_payment: false`, and a fiscal receipt's does not. `partially_covered` would say
    some of the amount qualifies when none of it is payable. Inventing any of the three would
    put a label in the dataset that nothing specified, so the engine raises and `claim_planner`
    never plans one.
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("12000.00"))
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal(0)

    with pytest.raises(PolicyGapError):
        evaluate([item("500.00", True)], ledger=ledger)


def test_a_wholly_non_covered_basket_is_rejected_whatever_the_balance():
    """Coverage is decided from the documents, so a spent balance cannot change the answer.

    The limit gap in `_reimbursable` needs a positive covered amount to bind — a claim
    covering nothing asks nothing of the limit — so there is no second condition to weigh
    and no ambiguity about which of the two to report.
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("12000.00"))

    result = evaluate([item("500.00", False)], ledger=ledger)
    assert result.verdict is Verdict.REJECTED
    assert result.reimbursable == Decimal("0.00")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)


def test_a_mixed_basket_beyond_the_limit_carries_both_causes():
    """Already reimbursed 11000.00, so 1000.00 remains.

        covered on the document  1500.00
        not covered               500.00
        total                    2000.00
        covered_fraction   1500.00 / 2000.00 = 0.75  ->  a non-covered line exists
        reimbursable       min(1500.00, 1000.00) = 1000.00  ->  the clamp also binds
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("11000.00"))

    result = evaluate([item("1500.00", True), item("500.00", False)], ledger=ledger)
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.covered_fraction == Decimal("0.75")
    assert result.reimbursable == Decimal("1000.00")
    assert result.imperfection == ("mixed_items", "limit_exhausted")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS, VerdictBasis.ACCOUNT_STATE)


def test_limits_are_per_category():
    """mental_health has its own 25000. Spending the whole vitamins_nutrition limit
    leaves it untouched — and the basket has to be a mental_health one, because the engine
    resolves each kind against the category it is claimed under."""
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("12000.00"))

    result = evaluate(
        [item("900.00", True, kind="therapy_session")],
        ledger=ledger,
        category="mental_health",
    )
    assert result.verdict is Verdict.COVERED
    assert ledger.remaining("p001", "mental_health") == Decimal("25000")


def test_limits_are_per_persona():
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("12000.00"))

    result = evaluate([item("900.00", True)], ledger=ledger, persona_id="p002")
    assert result.verdict is Verdict.COVERED


def test_only_the_reimbursable_amount_consumes_the_balance():
    """The limit caps what the plan pays out, so a non-covered line cannot eat into it.

        covered   400.00 of a 1000.00 receipt
        the ledger moves by 400.00, not by 1000.00
    """
    ledger = Ledger()
    result = evaluate([item("400.00", True), item("600.00", False)], ledger=ledger)
    ledger.record("p001", "vitamins_nutrition", result.reimbursable)

    assert result.reimbursable == Decimal("400.00")
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal("11600.00")


# --------------------------------------------------------------- date order --


def claim(claim_id: str, when: date, items: list[LineItem]) -> ClaimInput:
    return ClaimInput(
        claim_id=claim_id,
        persona_id="p001",
        category="vitamins_nutrition",
        documents=(document(items, when=when, doc_id=f"{claim_id}_d1"),),
    )


def test_claims_are_processed_in_date_order_whatever_order_they_arrive_in():
    """Two claims of 8000.00 covered against a 12000 limit.

        March    8000.00 fits          -> covered,           remaining 4000.00
        September reimburses 4000.00   -> partially_covered, fraction 1.0

    The September claim is the one that overruns because it is later, not because of the
    order it was handed to the engine. Presenting them backwards must change nothing.
    """
    march = claim("c1", date(2026, 3, 1), [item("8000.00", True)])
    september = claim("c2", date(2026, 9, 1), [item("8000.00", True)])

    forwards = evaluate_claims([march, september])
    backwards = evaluate_claims([september, march])

    assert forwards[0].verdict is Verdict.COVERED
    assert forwards[1].verdict is Verdict.PARTIALLY_COVERED
    assert forwards[1].reimbursable == Decimal("4000.00")

    # Results come back aligned with the input, so the backwards run reports them swapped.
    assert backwards[0].verdict is Verdict.PARTIALLY_COVERED
    assert backwards[1].verdict is Verdict.COVERED
    assert backwards[0].reimbursable == Decimal("4000.00")


def test_claims_sharing_a_date_resolve_in_the_order_they_were_presented():
    """policy.yaml says nothing about ties, so the engine fixes one: the sort is stable,
    which means same-day claims keep the order the caller listed them in. Deterministic,
    and it does not depend on a claim id or on a hash."""
    first = claim("c1", date(2026, 5, 5), [item("8000.00", True)])
    second = claim("c2", date(2026, 5, 5), [item("8000.00", True)])

    results = evaluate_claims([first, second])
    assert results[0].verdict is Verdict.COVERED
    assert results[1].verdict is Verdict.PARTIALLY_COVERED

    swapped = evaluate_claims([second, first])
    assert swapped[0].verdict is Verdict.COVERED
    assert swapped[1].verdict is Verdict.PARTIALLY_COVERED


def test_evaluating_the_same_claims_twice_gives_the_same_answer():
    claims = [
        claim("c1", date(2026, 3, 1), [item("8000.00", True)]),
        claim("c2", date(2026, 9, 1), [item("8000.00", True)]),
    ]
    assert evaluate_claims(claims) == evaluate_claims(claims)


def test_a_claim_of_several_receipts_is_dated_by_the_earliest_of_them():
    """A claim may span documents, and the balance moves when the money did.

    Every document here is a fiscal receipt, so every one of them is a proof of payment
    and the earliest payment is simply the earliest document. Where that stops being true —
    an invoice dated before the payment that settles it — is
    `test_claim_evidence.test_a_claim_is_dated_by_its_proof_of_payment`.
    """
    early = document([item("100.00", True)], when=date(2026, 2, 1), doc_id="a")
    late = document([item("100.00", True)], when=date(2026, 11, 1), doc_id="b")
    subject = ClaimInput(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        documents=(late, early),
    )
    assert subject.dated == date(2026, 2, 1)


# ------------------------------------------------------ verdicts the planner --
# ------------------------------------------------------ cannot yet realize   --


def test_a_payment_outside_the_active_period_is_rejected():
    """The period is 2026-01-01..2026-12-31, so 2025-12-31 misses it by a day.

    `rejected` and not `insufficient_evidence`: nothing about this claim is unestablished —
    a fiscal receipt states what was bought and proves it was paid for — and the policy
    plainly does not cover a payment made outside its own window. `claim_planner` cannot yet
    plan this (it is a content mechanism, not coverage arithmetic), but the engine must
    classify it if it is handed one, and it must not consume the balance for a claim it pays
    nothing on.
    """
    result = evaluate([item("500.00", True)], when=date(2025, 12, 31))
    assert result.verdict is Verdict.REJECTED
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert result.reimbursable == Decimal(0)
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert any("period" in line for line in result.policy_trace)


def test_the_two_routes_to_rejected_are_told_apart_by_their_cause():
    """`rejected` has two mechanisms and one name, so the verdict alone cannot separate
    them and `imperfection` has to.

    A basket the category covers none of carries NO cause — the verdict says the whole of
    it. A payment outside the window carries `outside_period`, because "not covered" is
    true of both and only the cause says which sense of it applies.
    """
    by_basket = evaluate([item("500.00", False)], when=IN_PERIOD)
    by_date = evaluate([item("500.00", True)], when=date(2027, 1, 1))

    assert by_basket.verdict is by_date.verdict is Verdict.REJECTED
    assert by_basket.imperfection == ()
    assert by_date.imperfection == (OUTSIDE_PERIOD,)


def test_an_out_of_period_claim_still_reports_what_its_lines_cover():
    """`covered_fraction` is a property of the line items on every branch. Returning 0
    here would make a fully covered out-of-period claim indistinguishable, on that field,
    from one where nothing was covered at all — two different facts under one number.

        covered   997.00 of 1000.00 -> 0.997, regardless of the date

    All three come out `rejected`, and the third for two independent reasons at once; the
    point of the test is the fraction, which differs across all three.
    """
    mixed = evaluate([item("997.00", True), item("3.00", False)], when=date(2025, 12, 31))
    assert mixed.verdict is Verdict.REJECTED
    assert mixed.covered_fraction == Decimal("0.997")

    full = evaluate([item("1000.00", True)], when=date(2027, 1, 1))
    assert full.verdict is Verdict.REJECTED
    assert full.covered_fraction == Decimal(1)

    none = evaluate([item("1000.00", False)], when=date(2027, 1, 1))
    assert none.verdict is Verdict.REJECTED
    assert none.covered_fraction == Decimal(0)


def test_the_day_the_period_opens_and_the_day_it_closes_are_inside_it():
    for when in (date(2026, 1, 1), date(2026, 12, 31)):
        assert evaluate([item("500.00", True)], when=when).verdict is Verdict.COVERED


def test_a_period_failure_is_decided_before_coverage_is():
    """policy.yaml does not say which of the two wins when both apply, and the engine checks
    the period first: a document from outside the window is not evidence of anything in the
    period, so there is nothing for a coverage verdict to be about.

    Since both branches now answer `rejected`, the order is no longer visible in the
    verdict — it is visible in `imperfection` and in the trace. The period branch names its
    cause and argues no coverage at all; the coverage branch names no cause and states the
    fraction. Asserting the verdict alone here would have been vacuous.
    """
    result = evaluate([item("500.00", False)], when=date(2027, 1, 1))
    assert result.verdict is Verdict.REJECTED
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert not any("coverage" in line for line in result.policy_trace)


# -------------------------------------------------------------- policy_trace --


def test_the_trace_justifies_a_partially_covered_verdict():
    result = evaluate([item("997.00", True), item("3.00", False)])
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — d1 (fiscal_receipt) proves both",
        "period ok",
        "coverage 99.7% (1 of 2 line items not covered)",
    )


def test_the_trace_names_the_limit_when_the_limit_is_the_reason():
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("10500.00"))
    result = evaluate([item("1000.00", True, qty=2)], ledger=ledger)

    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — d1 (fiscal_receipt) proves both",
        "period ok",
        "coverage 100% (all line items covered)",
        "annual limit vitamins_nutrition 12000.00 UAH: 10500.00 already reimbursed, "
        "1500.00 remaining",
        "reimbursable 1500.00 UAH of 2000.00 covered — annual limit exhausted",
    )


def test_the_trace_says_nothing_about_the_balance_when_the_limit_did_not_bind():
    """The trace justifies the verdict. A limit that did not bind is not part of the
    justification, and printing the balance on every claim would suggest every verdict
    depends on account state when `verdict_basis` says it does not."""
    result = evaluate([item("100.00", True)])
    assert not any("limit" in line for line in result.policy_trace)


def test_the_trace_carries_no_coverage_line_for_an_out_of_period_claim():
    """Same principle: coverage did not justify this verdict, so it is not in the trace —
    even though `covered_fraction` still reports it as a fact about the lines."""
    result = evaluate([item("997.00", True), item("3.00", False)], when=date(2025, 1, 1))
    assert not any("coverage" in line for line in result.policy_trace)
    assert result.covered_fraction == Decimal("0.997")


def test_the_trace_does_not_round_a_partial_verdict_up_to_a_hundred_percent():
    """999.99 covered of 1000.00 is 99.999%, which to one decimal place is 100.0. The
    claim is `partially_covered`, so a trace reading "coverage 100%" would contradict the
    verdict it is supposed to justify. Three places is the first that does not."""
    result = evaluate([item("999.99", True), item("0.01", False)])
    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.policy_trace[-1] == "coverage 99.999% (1 of 2 line items not covered)"


def test_percent_still_formats_a_normal_fraction():
    """The formatting rule itself, exercised directly: one decimal place unless that would
    round a partial fraction up to 100% or a positive one down to 0%."""
    from receipt_synth import policy_engine

    assert policy_engine._percent(Decimal("0.5")) == "50%"
    assert policy_engine._percent(Decimal("0.999")) == "99.9%"
    assert policy_engine._percent(Decimal("0.999999")) == "99.9999%"


def test_percent_refuses_a_fraction_beyond_its_precision_bound():
    """`_percent` tries at most 9 decimal places before giving up. A fraction needing more
    than that to be told apart from 0%/100% used to fall through silently printing the
    wrong extreme — see the docstring. It must now fail loudly instead, and no basket this
    generator draws produces a fraction anywhere near this precise."""
    from receipt_synth import policy_engine

    with pytest.raises(AssertionError):
        policy_engine._percent(Decimal("0.999999999999"))


def test_the_trace_justifies_a_rejected_verdict_with_the_coverage_line():
    """The trace states what the verdict rested on, and for `rejected` that is coverage and
    nothing else — the period passed, and the limit was never asked anything."""
    result = evaluate([item("500.00", False)])
    assert result.verdict is Verdict.REJECTED
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — d1 (fiscal_receipt) proves both",
        "period ok",
        "coverage 0% (1 of 1 line items not covered)",
    )
    assert not any("limit" in line for line in result.policy_trace)


# ------------------------------------------------------------------- ledger --


def test_a_fresh_ledger_offers_the_whole_limit():
    assert Ledger().remaining("p001", "vitamins_nutrition") == Decimal("12000")


def test_the_ledger_never_reports_a_negative_balance():
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("20000.00"))
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal(0)


def test_the_ledger_refuses_a_negative_entry():
    with pytest.raises(ValueError):
        Ledger().record("p001", "vitamins_nutrition", Decimal("-1.00"))


def test_an_unknown_category_is_refused():
    with pytest.raises(KeyError):
        annual_limit("not_a_category")


def test_a_claim_with_no_documents_is_refused():
    with pytest.raises(ValueError):
        evaluate_claim(persona_id="p001", category="vitamins_nutrition", documents=[])
