"""A claim is a list of documents, and what that costs.

Every expected value here is derived by hand from `config/policy.yaml` and written down
with its arithmetic. None of it came from running the engine and recording what it
printed.

The policy values these tests are derived from, as of policy.yaml version 1:

    period                                      2026-01-01 .. 2026-12-31
    reporting_currency                          UAH
    categories[vitamins_nutrition].annual_limit 12000
    vitamins_nutrition.covered_items            vitamin_complex, mineral_supplement,
                                                nutritionist_visit
    vitamins_nutrition.excluded_items           medicine, medical_device, cosmetics, hygiene

    document_evidence
      fiscal_receipt        proves_subject true   proves_payment true
      payment_confirmation  proves_subject false  proves_payment true
      bank_statement        proves_subject false  proves_payment true
      invoice               proves_subject true   proves_payment false
      act                   proves_subject true   proves_payment false
      order_screenshot      proves_subject true   proves_payment false
      non_fiscal_receipt    proves_subject true   proves_payment false

`document_evidence` is what makes a claim's evidence resolvable into transactions, and
every shape below is read off that table rather than off a document's own opinion of
itself.
"""

from __future__ import annotations

import random
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from receipt_synth.claim_planner import (
    Archetype,
    ClaimPlan,
    DocumentPlan,
    evidence_of,
    plan_claim,
)
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    OUTSIDE_PERIOD,
    PAYMENT_PRECEDES_SUBJECT,
    SUBJECT_NOT_EVIDENCED,
    ClaimInput,
    Ledger,
    PolicyGapError,
    document_evidence,
    evaluate_claim,
    evaluate_claims,
    resolve_evidence,
)
from receipt_synth.schemas import (
    Capture,
    Country,
    DocGroundTruth,
    DocType,
    LineItem,
    Verdict,
    VerdictBasis,
)

COVERED_KIND = "vitamin_complex"  # vitamins_nutrition.covered_items
EXCLUDED_KIND = "medical_device"  # vitamins_nutrition.excluded_items

IN_PERIOD = date(2026, 6, 15)


def item(price: str, covered: bool = True, qty: int = 1, kind: str | None = None) -> LineItem:
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


def doc(
    doc_id: str,
    doc_type: DocType,
    *,
    amount: str,
    when: date = IN_PERIOD,
    items: list[LineItem] | None = None,
    currency: str = "UAH",
) -> DocGroundTruth:
    """One document label.

    `amount` is stated independently of `items` on purpose: a payment confirmation carries
    an amount and no lines, and the disagreement cases below need the two to be settable
    apart.
    """
    return DocGroundTruth(
        doc_id=doc_id,
        source_file=f"{doc_id}.png",
        doc_type=doc_type,
        language="uk",
        currency=currency,
        amount=Decimal(amount),
        date=when,
        counterparty="Vendor",
        line_items=items or [],
        has_qr=True,
        qr_is_fiscal=True,
        has_fiscal_number=True,
        capture=Capture.SCREENSHOT,
    )


def receipt(doc_id: str, items: list[LineItem], when: date = IN_PERIOD) -> DocGroundTruth:
    """A fiscal receipt — the one common type that proves both facts at once."""
    total = sum((i.qty * i.price for i in items), Decimal(0))
    return doc(doc_id, DocType.FISCAL_RECEIPT, amount=str(total), when=when, items=items)


def evaluate(documents, *, ledger: Ledger | None = None, category="vitamins_nutrition"):
    return evaluate_claim(
        persona_id="p001", category=category, documents=documents, ledger=ledger
    )


# ------------------------------------------ the policy these tests are read off --


def test_the_evidence_table_this_file_was_written_against():
    """Pin `document_evidence`. Every shape below is derived from these seven rows, so a
    change to the policy has to fail here rather than quietly making the rest of the file
    assert something that no longer follows from it."""
    assert document_evidence(DocType.FISCAL_RECEIPT) == (True, True)
    assert document_evidence(DocType.PAYMENT_CONFIRMATION) == (False, True)
    assert document_evidence(DocType.BANK_STATEMENT) == (False, True)
    assert document_evidence(DocType.INVOICE) == (True, False)
    assert document_evidence(DocType.ACT) == (True, False)
    assert document_evidence(DocType.ORDER_SCREENSHOT) == (True, False)
    assert document_evidence(DocType.NON_FISCAL_RECEIPT) == (True, False)


def test_every_document_type_the_generator_can_label_has_an_evidence_row():
    """`DocType` and the keys of `document_evidence` are the same vocabulary — schemas.py
    says so. A member without a row would reach the engine as a KeyError from inside a
    verdict branch rather than as a missing policy statement."""
    for doc_type in DocType:
        assert document_evidence(doc_type) is not None


# --------------------------------------------------- A: the money is not summed --


def test_an_invoice_and_its_payment_are_one_transaction_and_one_amount():
    """THE CLAIM THAT SPANS TWO DOCUMENTS, end to end.

    An invoice states what was bought; a payment confirmation states that it was paid.
    They are the same money, so the claim's amount is 1200.00 and NOT 2400.00.

        invoice d1   1 x 1200.00 vitamin_complex, covered      lines total 1200.00
        payment d2   amount 1200.00, no line items            the same money
        claim total  1200.00
        covered      1200.00
        fraction     1200.00 / 1200.00 = 1
        limit        vitamins_nutrition 12000, nothing spent -> the cap does not bind
        reimbursable min(1200.00, 12000) = 1200.00
    """
    invoice = doc(
        "c1_d1", DocType.INVOICE, amount="1200.00", when=date(2026, 6, 1),
        items=[item("1200.00")],
    )
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00", when=date(2026, 6, 5))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.COVERED
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("1200.00")
    assert result.imperfection == ()
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — subject c1_d1 (invoice), payment c1_d2 "
        "(payment_confirmation)",
        "period ok",
        "coverage 100% (all line items covered)",
    )


def test_adding_the_payment_to_the_invoice_would_double_the_claim():
    """The failure decision A exists to stop, stated as arithmetic rather than as prose.

    Summing every document's stated amount gives 2400.00 for the claim above — twice the
    money — and the verdict would still come out `covered`, so nothing downstream would
    notice. The reimbursable amount is what gives it away, and it is 1200.00.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00")

    result = evaluate([invoice, payment])

    naive_sum = invoice.amount + payment.amount
    assert naive_sum == Decimal("2400.00"), "the arithmetic this test is about"
    assert result.reimbursable == Decimal("1200.00")
    assert result.reimbursable * 2 == naive_sum


def test_two_fiscal_receipts_are_two_transactions_and_the_money_does_add():
    """The other side of A. Each fiscal receipt proves its own payment, so they are two
    movements of money and summing them is right.

        d1  1 x 500.00 covered
        d2  1 x 700.00 covered
        total 1200.00, covered 1200.00, fraction 1
    """
    result = evaluate([receipt("c1_d1", [item("500.00")]), receipt("c1_d2", [item("700.00")])])

    assert result.verdict is Verdict.COVERED
    assert result.reimbursable == Decimal("1200.00")
    assert result.covered_fraction == Decimal(1)


def test_the_subject_document_carries_the_lines_and_the_payment_carries_none():
    """`resolve_evidence` is what the amount rule is read off, so it is asserted directly:
    one transaction, whose subject and payment are different documents."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="900.00", items=[item("900.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="900.00")

    shape = resolve_evidence([invoice, payment])

    assert len(shape.transactions) == 1
    assert shape.transactions[0].subject is invoice
    assert shape.transactions[0].payment is payment
    assert shape.subject_documents == (invoice,)
    assert shape.payment_documents == (payment,)


# ------------------------------------------------ B: evidence completeness ------


def test_an_invoice_on_its_own_proves_no_payment():
    """B, and the reason it is a release condition. Before the check, this claim came back
    `covered` at 100% — every line qualifies — while nothing in it says the money moved.

        invoice 1 x 800.00 covered
        fraction 800.00 / 800.00 = 1, and it is still reported: the lines are a fact
        reimbursable 0.00 — the plan pays out on evidence, and there is none of payment
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="800.00", items=[item("800.00")])

    result = evaluate([invoice])

    assert result.verdict is Verdict.NOT_PROOF_OF_PAYMENT
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: no document proves payment — invoice",
    )


@pytest.mark.parametrize(
    "doc_type", [DocType.INVOICE, DocType.ACT, DocType.ORDER_SCREENSHOT, DocType.NON_FISCAL_RECEIPT]
)
def test_every_type_that_proves_no_payment_reaches_the_same_verdict_alone(doc_type):
    """Parametrized over `document_evidence` rather than over one example, because the
    verdict is a property of the TYPE and a rule stated for one type is a rule the others
    are free to break."""
    alone = doc("c1_d1", doc_type, amount="800.00", items=[item("800.00")])
    assert evaluate([alone]).verdict is Verdict.NOT_PROOF_OF_PAYMENT


def test_a_payment_with_nothing_saying_what_it_bought_is_insufficient_evidence():
    """The other half of B. Money moved and no document states what for.

    `covered_fraction` is None rather than 0: nothing derived a fraction, and 0 would say
    that nothing on the documents is covered — a different fact, and one this claim has no
    documents to support.
    """
    payment = doc("c1_d1", DocType.PAYMENT_CONFIRMATION, amount="800.00")

    result = evaluate([payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (SUBJECT_NOT_EVIDENCED,)
    assert result.covered_fraction is None
    assert result.reimbursable == Decimal("0.00")
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: no document states what was bought — payment_confirmation",
    )


def test_a_fiscal_receipt_alone_establishes_both_facts():
    """Why the completeness check is vacuous on the dataset that exists today, said out
    loud: the one registered archetype is the one common type that proves both."""
    shape = resolve_evidence([receipt("c1_d1", [item("500.00")])])
    assert shape.proves_subject and shape.proves_payment
    assert evaluate([receipt("c1_d1", [item("500.00")])]).verdict is Verdict.COVERED


# ------------------------------------------------ C, D: dates ------------------


def test_a_claim_is_dated_by_its_proof_of_payment():
    """C. A limit is consumed when money moves, and the earliest document is the invoice —
    the date of the obligation, not of the expense."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="100.00", when=date(2026, 2, 1),
                  items=[item("100.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="100.00", when=date(2026, 11, 1))

    subject = ClaimInput(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        documents=(invoice, payment),
    )
    assert subject.dated == date(2026, 11, 1)


def test_the_ledger_order_follows_the_payment_and_not_the_invoice():
    """C, where it changes an answer rather than a field.

    Two claims against the 12000 vitamins_nutrition limit, 8000.00 covered each:

        claim A  invoice 2026-01-05, payment 2026-12-01
        claim B  fiscal receipt 2026-06-01

    By the earliest DOCUMENT, A is first: A takes 8000.00, B is left 4000.00.
    By the PAYMENT, B is first: B takes 8000.00, A is left 4000.00.

    The two orderings disagree about which claim is limit-bound, so this is the assertion
    that decision C is implemented and not merely described.
    """
    claim_a = ClaimInput(
        claim_id="a", persona_id="p001", category="vitamins_nutrition",
        documents=(
            doc("a_d1", DocType.INVOICE, amount="8000.00", when=date(2026, 1, 5),
                items=[item("8000.00")]),
            doc("a_d2", DocType.PAYMENT_CONFIRMATION, amount="8000.00", when=date(2026, 12, 1)),
        ),
    )
    claim_b = ClaimInput(
        claim_id="b", persona_id="p001", category="vitamins_nutrition",
        documents=(receipt("b_d1", [item("8000.00")], when=date(2026, 6, 1)),),
    )

    first, second = evaluate_claims([claim_a, claim_b])

    assert second.verdict is Verdict.COVERED, "B paid in June, and June comes first"
    assert second.reimbursable == Decimal("8000.00")
    assert first.verdict is Verdict.PARTIALLY_COVERED
    assert first.reimbursable == Decimal("4000.00")
    assert "limit_exhausted" in first.imperfection


def test_a_december_invoice_paid_in_january_is_an_ordinary_claim():
    """D, and the case it dissolves. The period is 2026-01-01..2026-12-31 and the invoice
    is dated 2025-12-20 — outside it. The expense is January's, the claim is ordinary, and
    `insufficient_evidence` would be wrong.

        invoice  2025-12-20, 1 x 600.00 covered
        payment  2026-01-15, 600.00
        -> covered, reimbursable 600.00
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", when=date(2025, 12, 20),
                  items=[item("600.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", when=date(2026, 1, 15))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.COVERED
    assert result.reimbursable == Decimal("600.00")
    assert result.imperfection == ()


def test_a_payment_outside_the_period_is_insufficient_evidence():
    """The half of D that survives: the period is checked, on the payment date."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", when=date(2026, 12, 20),
                  items=[item("600.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", when=date(2027, 1, 5))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert result.covered_fraction == Decimal(1)
    assert any("2027-01-05" in line for line in result.policy_trace)


def test_a_payment_may_not_precede_the_document_it_settles():
    """D's second half, and it is a DIFFERENT defect from the period one — folding the two
    together is what made the December invoice look like a period failure.

        invoice 2026-06-10, payment 2026-06-01
        both dates are inside the period, and the order is still impossible
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", when=date(2026, 6, 10),
                  items=[item("600.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", when=date(2026, 6, 1))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (PAYMENT_PRECEDES_SUBJECT,)
    assert OUTSIDE_PERIOD not in result.imperfection, "both dates are inside the window"
    assert result.policy_trace == (
        "category=vitamins_nutrition ok",
        "evidence: 1 transaction — subject c1_d1 (invoice), payment c1_d2 "
        "(payment_confirmation)",
        "documents disagree: payment c1_d2 dated 2026-06-01 precedes c1_d1 dated 2026-06-10",
    )


def test_a_payment_on_the_same_day_as_the_invoice_is_in_order():
    """The boundary. Paying an invoice the day it is issued is ordinary, so the check is
    strict-before rather than before-or-equal."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", when=IN_PERIOD,
                  items=[item("600.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", when=IN_PERIOD)

    assert evaluate([invoice, payment]).verdict is Verdict.COVERED


# ------------------------------------------------ E: documents that disagree ----


def test_documents_stating_different_amounts_are_insufficient_evidence():
    """E. Both facts are present separately, and the claim does not establish that THIS
    payment paid for THIS subject.

        invoice  1 x 1200.00 covered
        payment  1000.00
        -> the two do not describe one transaction

    `covered_fraction` is still 1.0: every line of the invoice is covered, which is a fact
    about the lines and is true whatever the payment says.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00")

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (AMOUNT_MISMATCH,)
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert result.policy_trace[-1] == (
        "documents disagree: c1_d1 (invoice) states 1200.00 UAH, "
        "payment c1_d2 states 1000.00 UAH"
    )


def test_a_disagreement_is_not_a_flag_on_a_coverage_verdict():
    """The corruption E exists to close, stated as the assertion that would fail if the
    cause were hung off a coverage-determined verdict: a claim whose documents contradict
    each other must not come out `covered`, however well its lines qualify."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00")

    result = evaluate([invoice, payment])

    assert result.verdict is not Verdict.COVERED
    assert result.verdict is not Verdict.PARTIALLY_COVERED
    assert result.reimbursable == Decimal("0.00")


def test_both_disagreements_at_once_are_both_named():
    """One verdict, several causes — the shape `partially_covered` already sets. A
    consumer can know different things about each, so neither absorbs the other.

        invoice 2026-06-10 for 1200.00, payment 2026-06-01 for 1000.00
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", when=date(2026, 6, 10),
                  items=[item("1200.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00", when=date(2026, 6, 1))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert set(result.imperfection) == {AMOUNT_MISMATCH, PAYMENT_PRECEDES_SUBJECT}


def test_a_missing_document_and_a_contradiction_are_told_apart_by_their_cause():
    """E's distinguishing fact: whether a second document exists at all. Same verdict, and
    a consumer must be able to tell the two apart without parsing prose."""
    missing = evaluate([doc("c1_d1", DocType.PAYMENT_CONFIRMATION, amount="800.00")])
    contradicting = evaluate([
        doc("c2_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")]),
        doc("c2_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00"),
    ])

    assert missing.verdict is contradicting.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert missing.imperfection == (SUBJECT_NOT_EVIDENCED,)
    assert contradicting.imperfection == (AMOUNT_MISMATCH,)


def test_a_disagreeing_claim_consumes_no_balance():
    """It reimburses nothing, so it cannot move the ledger — the same rule an out-of-period
    claim follows."""
    ledger = Ledger()
    result = evaluate(
        [
            doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")]),
            doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00"),
        ],
        ledger=ledger,
    )
    ledger.record("p001", "vitamins_nutrition", result.reimbursable)

    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal("12000")


# ------------------------------------------------ shapes the policy cannot resolve --


def test_a_bank_statement_cannot_be_resolved_to_the_transaction_of_this_claim():
    """The case decision A names as the one to think hardest about.

    A statement lists several transactions and the label carries ONE amount for the whole
    document, so "the transaction in the statement that corresponds to this claim" is not
    something the ground truth can point at. Neither policy.yaml nor the labelling contract
    settles whether a statement produces one label or one per transaction, so the engine
    refuses instead of picking.
    """
    statement = doc("c1_d2", DocType.BANK_STATEMENT, amount="5000.00")
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", items=[item("600.00")])

    with pytest.raises(PolicyGapError, match="bank_statement"):
        evaluate([invoice, statement])


def test_a_receipt_and_an_invoice_together_have_no_derivable_pairing():
    """A fiscal receipt already proves its own subject, so an invoice beside it is either
    a second description of the same money or a second purchase with no payment of its
    own. `document_evidence` does not say which, and guessing would either double the
    claim or silently drop a document."""
    with pytest.raises(PolicyGapError, match="pair"):
        evaluate([
            receipt("c1_d1", [item("600.00")]),
            doc("c1_d2", DocType.INVOICE, amount="600.00", items=[item("600.00")]),
        ])


def test_one_invoice_against_two_payments_has_no_derivable_pairing():
    """Which payment settles the invoice is not derivable, and the answer decides the
    claim's amount."""
    with pytest.raises(PolicyGapError, match="pair"):
        evaluate([
            doc("c1_d1", DocType.INVOICE, amount="600.00", items=[item("600.00")]),
            doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00"),
            doc("c1_d3", DocType.PAYMENT_CONFIRMATION, amount="600.00"),
        ])


# ------------------------------------------------ coverage across documents ----


def test_a_non_covered_line_on_the_invoice_still_makes_the_claim_partial():
    """The strict rule is unchanged by evidence resolution.

        invoice  1 x 900.00 covered + 1 x 100.00 medical_device
        payment  1000.00
        total    1000.00, covered 900.00, fraction 0.9
    """
    invoice = doc(
        "c1_d1", DocType.INVOICE, amount="1000.00",
        items=[item("900.00"), item("100.00", covered=False)],
    )
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00")

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.covered_fraction == Decimal("0.9")
    assert result.imperfection == ("mixed_items",)
    assert result.reimbursable == Decimal("900.00")


def test_the_limit_binds_on_the_money_that_moved_once():
    """The consequence of A for the cumulative limit, which is where a doubled amount
    would have travelled silently.

        already reimbursed 11500.00 of 12000 -> 500.00 remains
        invoice 1 x 800.00 covered, payment 800.00
        reimbursable min(800.00, 500.00) = 500.00

    Had the claim been read as 1600.00 the reimbursable amount would still have been
    500.00 — the clamp hides it — but `covered_fraction` and the trace would have been
    computed off the doubled figure.
    """
    ledger = Ledger()
    ledger.record("p001", "vitamins_nutrition", Decimal("11500.00"))

    result = evaluate(
        [
            doc("c1_d1", DocType.INVOICE, amount="800.00", items=[item("800.00")]),
            doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="800.00"),
        ],
        ledger=ledger,
    )

    assert result.verdict is Verdict.PARTIALLY_COVERED
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("500.00")
    assert result.imperfection == ("limit_exhausted",)
    assert result.policy_trace[-1] == (
        "reimbursable 500.00 UAH of 800.00 covered — annual limit exhausted"
    )


def test_the_currency_check_still_sees_every_document():
    """`_check_currency` iterated every document before this step and has to keep doing
    so: the payment document is the one carrying the amount a limit is compared against,
    and it is the one with no line items to give it away."""
    with pytest.raises(ValueError, match="EUR"):
        evaluate([
            doc("c1_d1", DocType.INVOICE, amount="600.00", items=[item("600.00")]),
            doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", currency="EUR"),
        ])


# ================================================================================
# The plan: a claim is a list of documents
# ================================================================================


PAIR_REGISTRY = {
    "ua_invoice": Archetype(
        slug="ua_invoice", doc_type=DocType.INVOICE, country=Country.UA,
        language="uk", categories=("vitamins_nutrition",),
    ),
    "ua_transfer": Archetype(
        slug="ua_transfer", doc_type=DocType.PAYMENT_CONFIRMATION, country=Country.UA,
        language="uk", categories=("vitamins_nutrition",),
    ),
}
"""A registry with NO archetype that proves both facts.

The only way to reach the split-evidence path today: `templates/` holds one archetype and
it is a fiscal receipt, which proves both. Patched in rather than added to `ARCHETYPES`,
because registering an archetype no template and no builder can produce would break every
run rather than one test.
"""


def _persona():
    return generate_persona(random.Random(20260803), persona_id="p001", country=Country.UA)


def test_the_registered_archetype_yields_a_claim_of_exactly_one_document():
    """What the dataset contains today, asserted rather than assumed. One archetype is
    registered and it proves both facts, so a claim needs one document — and the planner
    prefers that shape wherever it exists."""
    plan = plan_claim(random.Random(3), persona=_persona(), claim_id="c1", ledger=Ledger())

    assert len(plan.documents) == 1
    assert plan.documents[0].archetype.slug == "ua_prro_receipt"
    assert evidence_of(plan.documents[0].archetype) == (True, True)


def test_a_registry_without_a_both_proving_archetype_plans_two_documents():
    """The mechanism this step exists for. With an invoice archetype and a transfer
    archetype and nothing that proves both, the planner assembles a claim out of the pair
    instead of producing half a claim and leaving the engine to find out."""
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
        plan = claim_planner.plan_claim(
            random.Random(3), persona=_persona(), claim_id="c1",
            category="vitamins_nutrition", ledger=Ledger(),
        )

    assert [d.archetype.slug for d in plan.documents] == ["ua_invoice", "ua_transfer"]
    assert plan.documents[0].issued_at <= plan.documents[1].issued_at, (
        "a payment may not precede what it settles"
    )
    assert plan.issued_at == plan.documents[1].issued_at, (
        "the claim is dated by its proof of payment"
    )


def test_only_one_document_of_a_plan_carries_the_basket():
    """Inventory item 5: the basket is sized ONCE for the claim, against what is left of
    the annual balance. A second document carrying line items would spend the balance
    twice and the `limit_exhausted` mechanism would be counted under a size it never had.
    """
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
        plan = claim_planner.plan_claim(
            random.Random(4), persona=_persona(), claim_id="c1",
            category="vitamins_nutrition", ledger=Ledger(),
            verdict=Verdict.PARTIALLY_COVERED, cause="limit_exhausted",
        )

    carriers = [d for d in plan.documents if evidence_of(d.archetype).proves_subject]
    assert len(carriers) == 1
    assert plan.subject_document is carriers[0]
    assert plan.item_count is not None, "this plan was sized to overrun the balance"


def test_a_plan_whose_documents_would_both_carry_a_basket_is_refused():
    """The guard behind the test above, exercised by building the plan it forbids. Two
    invoices state what was bought twice, and the claim-level basket has no way to say
    which of them spends it."""
    invoice = PAIR_REGISTRY["ua_invoice"]
    plan = ClaimPlan(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        verdict=Verdict.COVERED, issued_at=datetime(2026, 6, 15, 12, 0),
        documents=(
            DocumentPlan(archetype=invoice, issued_at=datetime(2026, 6, 1, 12, 0)),
            DocumentPlan(archetype=invoice, issued_at=datetime(2026, 6, 2, 12, 0)),
        ),
    )
    with pytest.raises(ValueError, match="sized once"):
        _ = plan.subject_document


def test_a_registry_that_cannot_establish_both_facts_is_refused():
    """An invoice archetype and nothing that proves payment. The planner draws `covered`
    and `partially_covered`, both of which need complete evidence, so building the claim
    anyway would mean planning a claim whose label is decided by what is missing."""
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", {"ua_invoice": PAIR_REGISTRY["ua_invoice"]})
        with pytest.raises(ValueError, match="both what was bought"):
            claim_planner.plan_claim(
                random.Random(3), persona=_persona(), claim_id="c1",
                category="vitamins_nutrition", ledger=Ledger(),
            )


def test_a_two_document_plan_is_deterministic_under_seed():
    """Including the lead between the subject document and its payment, which is drawn."""
    from receipt_synth import claim_planner

    def planned():
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
            return claim_planner.plan_claim(
                random.Random(11), persona=_persona(), claim_id="c1",
                category="vitamins_nutrition", ledger=Ledger(),
            )

    assert planned() == planned()


def test_the_claim_label_of_a_two_document_claim_is_linked():
    """`linked` is `len(documents) > 1` exactly, and it had never been true. Asserted on
    the ground-truth record the plan writes, not on the plan."""
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
        plan = claim_planner.plan_claim(
            random.Random(3), persona=_persona(), claim_id="p001_c1",
            category="vitamins_nutrition", ledger=Ledger(),
        )

    invoice = doc(
        "p001_c1_d1", DocType.INVOICE, amount="1200.00",
        when=plan.documents[0].issued_at.date(), items=[item("1200.00")],
    )
    payment = doc(
        "p001_c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00",
        when=plan.documents[1].issued_at.date(),
    )
    evaluation = evaluate_claim(
        persona_id="p001", category="vitamins_nutrition", documents=[invoice, payment]
    )
    label = plan.ground_truth(["p001_c1_d1", "p001_c1_d2"], evaluation)

    assert label.linked is True
    assert label.documents == ["p001_c1_d1", "p001_c1_d2"]
    assert label.verdict is Verdict.COVERED
    assert label.covered_fraction == 1.0
    assert label.reimbursable_amount == Decimal("1200.00")
    assert label.imperfection == []
    assert label.verdict_basis == [VerdictBasis.DOCUMENTS]
    assert label.policy_trace[1] == (
        "evidence: 1 transaction — subject p001_c1_d1 (invoice), "
        "payment p001_c1_d2 (payment_confirmation)"
    )


# ================================================================================
# The assembler: the document loop
# ================================================================================


def test_the_vendor_is_chosen_once_per_claim_however_many_documents_it_has(tmp_path):
    """Inventory item 7, checked on the loop rather than on the builder.

    A sole trader's name is DRAWN, so two calls to `_pick_vendor` for one claim would put
    two different sellers on two documents of one purchase. Counted here through a real
    run, so that the guarantee is a property of the loop and not of a call the loop
    happens not to make yet.
    """
    from receipt_synth import assembler

    calls: list[str] = []
    original = assembler._pick_vendor

    def counting(rng, country, category, *, mixed):
        calls.append(category)
        return original(rng, country, category, mixed=mixed)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "_pick_vendor", counting)
        result = assembler.generate_dataset(
            seed=20260803, out_dir=tmp_path, personas=2, claims_per_persona=3
        )

    assert calls, "this run built no claim"
    assert len(calls) == len(result.claims)


def test_document_ids_are_numbered_from_the_plan_and_not_fixed_at_one(tmp_path):
    """Inventory item 6. `_d1` was hardcoded; the id now comes from the position of the
    document in the plan, and the claim label points at exactly the documents written."""
    from receipt_synth.assembler import generate_dataset

    result = generate_dataset(seed=20260803, out_dir=tmp_path, personas=2, claims_per_persona=3)

    written = {document.doc_id for document in result.documents}
    for claim, plan in zip(result.claims, result.plans, strict=True):
        expected = [f"{claim.claim_id}_d{i}" for i in range(1, len(plan.documents) + 1)]
        assert claim.documents == expected
        assert set(expected) <= written
        assert claim.linked is (len(expected) > 1)


def test_a_planned_archetype_with_no_builder_fails_by_name():
    """The other assumption the loop used to carry: every document was built by
    `build_prro_receipt`, whatever the plan said. An archetype nothing can produce has to
    say so rather than be handed to the one builder that exists."""
    from receipt_synth import assembler

    plan = ClaimPlan(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        verdict=Verdict.COVERED, issued_at=datetime(2026, 6, 15, 12, 0),
        documents=(
            DocumentPlan(
                archetype=PAIR_REGISTRY["ua_invoice"], issued_at=datetime(2026, 6, 15, 12, 0)
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="ua_invoice"):
        assembler._build_document(
            random.Random(1), persona=_persona(), plan=plan,
            document_plan=plan.documents[0], vendor={}, doc_id="c1_d1",
            renderer=None, out_dir=Path("."),
        )
