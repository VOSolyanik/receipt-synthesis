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

from receipt_synth import claim_planner, policy_engine
from receipt_synth.claim_planner import (
    Archetype,
    ClaimPlan,
    DocumentPlan,
    archetypes_for,
    can_assemble_evidence,
    evidence_of,
    plan_claim,
)
from receipt_synth.config import load_policy
from receipt_synth.content_builder import PartyIdentity
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    COUNTERPARTY_MISMATCH,
    OUTSIDE_PERIOD,
    PARTIAL_PAYMENT_MARKER_FIELDS,
    PAYMENT_PRECEDES_SUBJECT,
    SUBJECT_NOT_EVIDENCED,
    AgreementAxis,
    ClaimInput,
    Ledger,
    PolicyGapError,
    active_period,
    cross_document_agreement,
    document_evidence,
    evaluate_claim,
    evaluate_claims,
    partial_payment_marker_fields,
    partial_payment_outside_the_period,
    resolve_evidence,
)
from receipt_synth.schemas import (
    Capture,
    Country,
    Direction,
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
    direction: Direction | None = None,
    instalment_amount: str | None = None,
    counterparty: str = "Vendor",
) -> DocGroundTruth:
    """One document label.

    `amount` is stated independently of `items` on purpose: a payment confirmation carries
    an amount and no lines, and the disagreement cases below need the two to be settable
    apart.

    `instalment_amount` is the printed marker of a partial settlement — what one part of this
    document's obligation comes to, where the document says it is settled in parts. `None` on
    every ordinary document, which is what makes the mismatch cases below still mismatch.

    `counterparty` DEFAULTS TO ONE VALUE FOR EVERY DOCUMENT, which is what an honest claim looks
    like: the invoice and the payment name one seller. The parameter exists so a pair can be built
    that does not — see the counterparty section below — and every other test in this file inherits
    the agreement rather than restating it.
    """
    return DocGroundTruth(
        doc_id=doc_id,
        source_file=f"{doc_id}.png",
        doc_type=doc_type,
        language="uk",
        currency=currency,
        amount=Decimal(amount),
        instalment_amount=None if instalment_amount is None else Decimal(instalment_amount),
        date=when,
        counterparty=counterparty,
        direction=direction,
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


def test_a_payment_document_carrying_lines_does_not_double_the_claim():
    """Decision A's guard, pinned on an input the design does not admit.

    Taking the lines from `subject_documents` rather than from every document is what keeps
    a payment document's lines out of the claim's amount — and NO set of documents this
    generator can build exercises it. The types that prove payment without proving the
    subject are `payment_confirmation` and `bank_statement`: the statement is refused
    outright by `resolve_evidence`, and the confirmation carries no line items, which the
    requirements state and the labelling contract records. A `fiscal_receipt` proves both
    facts, so in a self-contained claim `subject_documents` and `documents` are the same
    tuple. Every remaining mixture is refused. The branch is therefore unreachable BY
    CONSTRUCTION, not merely unexercised today.

    Which is why it is asserted here on a hand-built record instead of being waited for, the
    same way `test_a_verdict_the_planner_cannot_draw_is_still_given_a_row` asserts a report
    row nothing yet emits. The alternative was a comment saying the branch is defence in
    depth, and a comment does not go red when someone deletes what it describes — this file
    exists partly because that failure mode has already cost this project a day.

        invoice d1  1 x 1200.00 covered, lines total 1200.00
        payment d2  amount 1200.00, AND lines of its own worth 1200.00
        claim       total 1200.00, reimbursable 1200.00 — never 2400.00
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")])
    payment = doc(
        "c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00", items=[item("1200.00")]
    )

    result = evaluate([invoice, payment])

    assert result.reimbursable == Decimal("1200.00")
    assert result.covered_fraction == Decimal(1)
    assert result.verdict is Verdict.COVERED


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
    refusing it for its date would be wrong.

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


def test_a_payment_outside_the_period_is_rejected():
    """The half of D that survives: the period is checked, on the payment date.

        invoice 2026-12-20 (inside), payment 2027-01-05 (outside) — the payment decides

    `rejected`, and emphatically NOT `insufficient_evidence`. Every fact this claim rests
    on is established: the purchase is stated, the payment is proven, the two agree, and
    the invoice's own date is inside the window. Nothing is missing, so a verdict whose name
    means "a required fact was not established" does not describe it. The policy plainly
    does not cover this claim — by WHEN rather than by WHAT, which is the wrong-subject case
    turned ninety degrees — and `rejected` is the verdict for plainly not covered.

    `covered_fraction` is still 1.0: every line of the invoice belongs to the category, and
    that is a fact about the lines whatever the date says.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", when=date(2026, 12, 20),
                  items=[item("600.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00", when=date(2027, 1, 5))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.REJECTED
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,)
    assert any("2027-01-05" in line for line in result.policy_trace)


def test_a_payment_may_not_precede_the_document_it_settles():
    """D's second half, and it is a DIFFERENT defect from the period one — folding the two
    together is what made the December invoice look like a period failure. They now differ
    in the verdict as well as in the cause: an impossible order leaves the linkage
    unestablished (`insufficient_evidence`), while an out-of-window payment establishes
    everything and is simply not covered (`rejected`).

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


# ------------------------------------------ a payment that settles one part --
#
# 🔴 TWO CLAIMS WITH THE SAME PAIR OF NUMBERS AND DIFFERENT VERDICTS. Every case below is an
# invoice of 1200.00 beside a payment of less, and what decides the label is whether the INVOICE
# SAYS the obligation is settled in parts. The arithmetic is identical throughout on purpose:
# these tests are what stops `partially_paid` from being implemented as "the payment is smaller",
# which would relabel every low-side `amount_mismatch` and cost the corpus a negative it already
# has. config/policy.yaml, `partial_payment`, states the rule.


def test_a_payment_settling_one_instalment_is_partially_paid():
    """The lawful pair: an obligation of 1200.00 the invoice says is settled in parts of 300.00,
    and a payment of exactly 300.00.

    `covered_fraction` is still 1.0 — every line of the invoice is covered, which is a fact about
    the lines and says nothing about how much has been paid — and `reimbursable` is 0.00, because
    policy.yaml has not decided how much of a partly settled obligation is payable.

    NO CAUSE, for the reason `not_proof_of_payment` carries none: one mechanism, one way to reach
    it, nothing for a cause to distinguish.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00")

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.PARTIALLY_PAID
    assert result.imperfection == ()
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert result.policy_trace[-1] == (
        "partial settlement: c1_d1 states 1200.00 UAH settled in parts of 300.00, "
        "and payment c1_d2 states 300.00"
    )


def test_a_smaller_payment_without_the_marker_is_not_partially_paid():
    """🔴 THE DISCRIMINATOR, AND THE TEST THIS WHOLE BRANCH IS ON PROBATION FOR. The same two
    amounts as the case above — 1200.00 against 300.00 — with the instalment term absent from the
    invoice. The claim states no arrangement to pay in parts, so what it shows is a payment for an
    amount its subject document does not name: `insufficient_evidence`, cause `amount_mismatch`,
    exactly as before this verdict existed.

    A `partially_paid` implemented as "the payment is smaller" passes every other test in this
    section and fails this one. That is the whole of why it is here: the mismatch cause is a
    negative the corpus already contains, and a rule that swallowed it would leave the dataset
    with fewer distinctions than it had.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")])
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00")

    assert invoice.instalment_amount is None, "the marker is what this test removes"

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.verdict is not Verdict.PARTIALLY_PAID
    assert result.imperfection == (AMOUNT_MISMATCH,)


def test_a_payment_matching_no_part_of_a_stated_arrangement_is_a_mismatch():
    """The marker has to AGREE with the payment, not merely be present. An invoice settled in
    parts of 300.00 beside a payment of 250.00 is a payment for some third amount — the mismatch
    case again — and a rule that read only the presence of the term would label it a lawful
    instalment of an arrangement it does not fit."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="250.00")

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (AMOUNT_MISMATCH,)


def test_an_instalment_paid_before_its_invoice_is_still_insufficient_evidence():
    """The amount check is skipped for a partial settlement; the DATE check is not. A payment that
    precedes what it settles is an impossible order whether it pays a part or the whole, so the
    cause survives and the verdict with it — `partially_paid` describes a claim whose documents
    agree, and these do not."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", when=date(2026, 6, 10),
                  items=[item("1200.00")], instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00", when=date(2026, 6, 1))

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (PAYMENT_PRECEDES_SUBJECT,)
    assert AMOUNT_MISMATCH not in result.imperfection, (
        "the pair agrees about the amount — the part is what the invoice says it is"
    )


def test_a_partial_settlement_paid_outside_the_period_is_rejected():
    """🔴 THE ONE CLAIM TWO BRANCHES BOTH DESCRIBE, and policy.yaml decides which wins:
    `partial_payment.outside_the_period` says `rejected`, and this pins it.

    THE PRECEDENCE IS THE POINT, not the arithmetic. Both facts are proven, the pair agrees, and
    the invoice's term is printed — so `partially_paid` is true of the documents — while the
    payment fell outside the benefit window, so the plan does not cover the expense at all. The
    period question is prior: it asks whether the plan covers this claim, and `partially_paid` is
    a statement about a claim the plan does cover.

    ⚠️ NO GENERATED CLAIM REACHES IT. `claim_planner` names a schedule for a `partially_paid` plan
    and displaces the payment for a `rejected` one, never both, so this case exists in the
    consumer's world and not in the corpus. It is pinned here precisely because nothing else would
    catch a change to it: an engine that answered `partially_paid` would produce a corpus
    indistinguishable from this one.
    """
    outside = date(2025, 6, 15)
    start, end = active_period()
    assert not start <= outside <= end, "the date is inside the window — this proves nothing"

    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", when=date(2025, 6, 1),
                  items=[item("1200.00")], instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00", when=outside)

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.REJECTED
    assert result.verdict is not Verdict.PARTIALLY_PAID
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert partial_payment_outside_the_period() is Verdict.REJECTED, (
        "policy.yaml declares the other precedence; this test pins the engine to the file"
    )


def test_an_engine_ordered_against_the_declared_precedence_refuses_to_label():
    """The drift above, made to happen. The branch ORDER in `evaluate_claim` is this engine's
    answer to the precedence question, so a file declaring the other answer must stop the run
    rather than be silently overruled — a consumer reading that file would label such a claim
    `partially_paid` and score this dataset's `rejected` as a miss.

    Patched on the POLICY side because that is the side a consumer vendors."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", when=date(2025, 6, 1),
                  items=[item("1200.00")], instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00", when=date(2025, 6, 15))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "receipt_synth.policy_engine.partial_payment_outside_the_period",
            lambda: Verdict.PARTIALLY_PAID,
        )
        with pytest.raises(PolicyGapError, match="outside_the_period"):
            evaluate([invoice, payment])


def test_a_part_equal_to_the_whole_is_not_a_partial_settlement():
    """A "part" that comes to the whole obligation settles it, and a claim whose payment settles
    its invoice in full is an ordinary claim. Asserted because the guard is one comparison a
    reading of "the marker is present" would drop, and the resulting corpus would carry
    `partially_paid` on claims that were paid in full."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  instalment_amount="1200.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00")

    result = evaluate([invoice, payment])

    assert result.verdict is Verdict.COVERED
    assert result.reimbursable == Decimal("1200.00")


def test_a_partly_settled_claim_consumes_no_balance():
    """It pays out nothing, so it must leave the annual limit where it found it — otherwise a
    later claim of the same persona would be labelled against a balance this one never spent.

    The absolute the assertion is against is 12000 written out — the `vitamins_nutrition` limit
    this module's header records — rather than a figure read back through the code that spends it.
    """
    ledger = Ledger()
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00")

    result = evaluate([invoice, payment], ledger=ledger)
    ledger.record("p001", "vitamins_nutrition", result.reimbursable)

    assert result.verdict is Verdict.PARTIALLY_PAID
    assert ledger.spent("p001", "vitamins_nutrition") == Decimal("0.00")
    assert ledger.remaining("p001", "vitamins_nutrition") == Decimal(12000)


def test_the_marker_the_engine_reads_is_the_one_the_policy_declares():
    """🔴 A CONSUMER BUILDS ITS OWN ENGINE FROM policy.yaml. If this module's marker and the
    file's ever differ, the two engines label the same claim differently while each is correct
    about the field it read — the most expensive class of disagreement this repository has, because
    it looks like a measurement problem.

    Both directions are asserted: the names agree, and every name is a real field of the label
    record rather than a string nothing carries.
    """
    assert partial_payment_marker_fields() == PARTIAL_PAYMENT_MARKER_FIELDS
    assert PARTIAL_PAYMENT_MARKER_FIELDS, "an empty marker would make every pair a mismatch"
    for name in PARTIAL_PAYMENT_MARKER_FIELDS:
        assert name in DocGroundTruth.model_fields, f"{name} is not a label field"


def test_an_engine_reading_a_marker_the_policy_does_not_declare_refuses_to_label():
    """The drift above, made to happen. Patched on the POLICY side because that is the side a
    consumer vendors: an engine that went on labelling against a field the file no longer names
    would produce ground truth nobody can reproduce from the published spec."""
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  instalment_amount="300.00")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="300.00")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "receipt_synth.policy_engine.partial_payment_marker_fields", lambda: ("amount_due",)
        )
        with pytest.raises(PolicyGapError, match="amount_due"):
            evaluate([invoice, payment])


# ------------------------------- the axes are a policy parameter, not a list in the code --
#
# 🔴 WHAT THESE TESTS ARE ABOUT, and it is not the counterparty. `cross_document_agreement` in
# policy.yaml declares WHICH fields the two documents of a claim must agree on and WHAT it costs
# them not to; the engine reads that block. So the assertions below come in pairs — the axis
# declared and the same claim labelled, the axis withdrawn and the same claim passing — because a
# check that fires whatever the file says would be a check the file does not control.
#
# ⚠️ THE VARIANTS ARE BUILT HERE AND HANDED TO THE ENGINE. config/policy.yaml is never edited to
# make one of these pass: it is the file the rest of this suite reads its expectations from, and a
# test that moved it would be proving a property of its own edit. `policy_variant` patches the one
# function the engine reads the file through, so the accessor's own parsing and guards run on the
# variant exactly as they run on the file.


def policy_variant(patch, axes: list[dict]) -> None:
    """Hand `policy_engine` a policy identical to the shipped one but for its axis list."""
    variant = {**load_policy(), "cross_document_agreement": axes}
    patch.setattr(policy_engine, "load_policy", lambda: variant)


AMOUNT_AXIS = {"axis": "amount", "verdict": "insufficient_evidence", "cause": AMOUNT_MISMATCH}
COUNTERPARTY_AXIS = {
    "axis": "counterparty",
    "verdict": "insufficient_evidence",
    "cause": COUNTERPARTY_MISMATCH,
}


def a_pair_naming_two_parties() -> list[DocGroundTruth]:
    """An invoice from one party beside a payment to another, and ORDINARY IN EVERY OTHER RESPECT:
    the amounts agree to the kopiyka, both dates are inside the period and in order, and every line
    of the basket is covered. So a claim built from it is `covered` on every axis but this one,
    which is what makes it usable as the input to both directions below."""
    return [
        doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
            counterparty="Аптека АНЦ"),
        doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1200.00", counterparty="Подорожник"),
    ]


def test_a_payment_made_to_another_party_is_insufficient_evidence():
    """🔴 THE LINKAGE SLOT BROKEN BY WHO RATHER THAN BY HOW MUCH. Both documents are flawless and
    they agree about the money and the dates; the invoice was issued by one seller and the money
    went to another, so nothing establishes that THIS payment paid for THIS obligation.

    `covered_fraction` is still 1.0 — every line of the invoice is covered, which is a fact about
    the lines and is true whatever the payment names — and nothing is reimbursed.
    """
    result = evaluate(a_pair_naming_two_parties())

    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.imperfection == (COUNTERPARTY_MISMATCH,)
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert result.verdict_basis == (VerdictBasis.DOCUMENTS,), "both names are on the images"
    assert result.policy_trace[-1] == (
        "documents disagree: c1_d1 (invoice) names 'Аптека АНЦ', "
        "payment c1_d2 names 'Подорожник'"
    )


def test_the_same_pair_passes_when_the_policy_declares_no_counterparty_axis():
    """THE OTHER DIRECTION, AND THE ONE THAT MAKES THE FIRST MEAN SOMETHING. The identical
    documents, evaluated against a policy whose axis list is amount alone: nothing compares the two
    names, so the claim is an ordinary covered one.

    It fails if the comparison is reached by any route but the declared list — a leftover `if`, a
    predicate applied because it exists — which is exactly the state this block was lifted out of.
    """
    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [AMOUNT_AXIS])
        result = evaluate(a_pair_naming_two_parties())

    assert result.verdict is Verdict.COVERED
    assert result.imperfection == ()
    assert not any("names" in line for line in result.policy_trace), result.policy_trace


def test_the_verdict_a_disagreement_earns_is_read_from_the_policy():
    """The OUTCOME is a parameter too, not only the axis. The same pair, against a policy that
    declares the same axis with a different label, comes back carrying that label.

    ⚠️ NOBODY WOULD WRITE THIS POLICY, and it is not offered as one: a broken linkage is
    `insufficient_evidence` for the reason `cross_document_agreement` gives, and the shipped file
    says so. What this asserts is only that the engine takes the label FROM THE FILE — an engine
    that returned a constant would pass every other test in this section and fail this one.
    """
    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [{**COUNTERPARTY_AXIS, "verdict": "rejected"}])
        result = evaluate(a_pair_naming_two_parties())

    assert result.verdict is Verdict.REJECTED
    assert result.imperfection == (COUNTERPARTY_MISMATCH,)


def test_an_axis_the_engine_cannot_compare_is_refused_rather_than_ignored():
    """The asymmetry `cross_document_agreement` is built on. A declared axis nothing performs would
    let every claim that fails it be labelled as though its documents agreed — a silently wrong
    ground truth — so it raises where the file is read, naming what the engine can compare."""
    with pytest.MonkeyPatch.context() as patch:
        policy_variant(
            patch,
            [{"axis": "payment_form", "verdict": "insufficient_evidence", "cause": "whatever"}],
        )
        with pytest.raises(PolicyGapError, match="payment_form"):
            evaluate(a_pair_naming_two_parties())


def test_an_axis_declared_twice_is_refused():
    """A duplicate would put one cause into `imperfection` twice, and — if the two entries named
    different verdicts — make the label depend on which was read first."""
    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [COUNTERPARTY_AXIS, COUNTERPARTY_AXIS])
        with pytest.raises(PolicyGapError, match="twice"):
            evaluate(a_pair_naming_two_parties())


def test_two_failed_axes_declaring_different_verdicts_are_refused():
    """A claim may fail several axes at once, and policy.yaml states no precedence between the
    labels they declare. Picking either would be the engine deciding a policy question, and a
    consumer's engine picking the other would label the same claim differently.

    ⚠️ UNREACHABLE AGAINST THE SHIPPED FILE, where every axis declares `insufficient_evidence`, and
    asserted against a variant for that reason.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  counterparty="Аптека АНЦ")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00",
                  counterparty="Подорожник")

    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [AMOUNT_AXIS, {**COUNTERPARTY_AXIS, "verdict": "rejected"}])
        with pytest.raises(PolicyGapError, match="no precedence|different verdicts"):
            evaluate([invoice, payment])


def test_the_axes_are_applied_in_the_order_the_policy_declares_them():
    """`imperfection` follows the file, not the order the predicates happen to be written in. A
    claim failing two axes is reported the same way every run, and the way is the policy's.

    Asserted on a REVERSED variant rather than on the shipped order alone: against the file's own
    order the two are indistinguishable from a hardcoded sequence.
    """
    invoice = doc("c1_d1", DocType.INVOICE, amount="1200.00", items=[item("1200.00")],
                  counterparty="Аптека АНЦ")
    payment = doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="1000.00",
                  counterparty="Подорожник")

    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [AMOUNT_AXIS, COUNTERPARTY_AXIS])
        forwards = evaluate([invoice, payment])
    with pytest.MonkeyPatch.context() as patch:
        policy_variant(patch, [COUNTERPARTY_AXIS, AMOUNT_AXIS])
        backwards = evaluate([invoice, payment])

    assert forwards.imperfection == (AMOUNT_MISMATCH, COUNTERPARTY_MISMATCH)
    assert backwards.imperfection == (COUNTERPARTY_MISMATCH, AMOUNT_MISMATCH)


def test_the_shipped_policy_declares_the_axes_this_suite_was_written_against():
    """Pin `cross_document_agreement` as the tests above read it, the way this file pins
    `document_evidence` at its head: a change to the block has to fail here rather than quietly
    making a section assert something that no longer follows from the policy."""
    assert cross_document_agreement() == (
        AgreementAxis("amount", Verdict.INSUFFICIENT_EVIDENCE, AMOUNT_MISMATCH),
        AgreementAxis("date_order", Verdict.INSUFFICIENT_EVIDENCE, PAYMENT_PRECEDES_SUBJECT),
        AgreementAxis("counterparty", Verdict.INSUFFICIENT_EVIDENCE, COUNTERPARTY_MISMATCH),
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


def test_a_bank_statement_now_pairs_with_a_subject_document_like_any_other_payment():
    """What replaced the blanket refusal of a statement, and it is the same test inverted.

    A statement used to be refused outright here, on the ground that it lists several
    transactions while its label carried one amount for the whole document, so nothing pointed
    at the row the claim was about. Its label now describes ONE TRANSACTION — the row's amount,
    date, counterparty and direction, with `relevant_transaction` naming the row — so the
    claim's money is identified exactly as well as it is on a confirmation, and the type is an
    ordinary payment document to this function.

    Asserted on the SHAPE rather than on a verdict, because that is what changed: one
    transaction, the invoice as its subject and the statement as its payment.
    """
    statement = doc("c1_d2", DocType.BANK_STATEMENT, amount="600.00")
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", items=[item("600.00")])

    shape = resolve_evidence([invoice, statement])

    assert len(shape.transactions) == 1
    assert shape.transactions[0].subject is invoice
    assert shape.transactions[0].payment is statement
    assert evaluate([invoice, statement]).verdict is Verdict.COVERED


def test_a_credit_is_refused_because_money_arriving_proves_no_expense():
    """🔴 The invariant of the statement class, checked on the oracle rather than on the builder.

    Only a DEBIT can be proof of payment. A credit is a refund or a reversal, and a claim whose
    proof of payment is money ARRIVING is a claim proving that the money came back. policy.yaml
    assigns no verdict to that, so the engine refuses rather than inventing one.

    Built by hand because `content_builder.BankStatement` cannot produce it — it refuses to
    label a credit row — and this guard exists for what does not go through that builder: a trap
    archetype, or a consumer's own record. Same amount, same date, same parties as the test
    above; the direction is the only difference, and it is the whole difference.
    """
    refund = doc(
        "c1_d2", DocType.BANK_STATEMENT, amount="600.00", direction=Direction.CREDIT
    )
    invoice = doc("c1_d1", DocType.INVOICE, amount="600.00", items=[item("600.00")])

    with pytest.raises(PolicyGapError, match="credit"):
        evaluate([invoice, refund])


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

⚠️ IT IS NO LONGER THE ONLY WAY TO REACH THE SPLIT-EVIDENCE PATH. It was, while `templates/` held
fiscal receipts alone; the real `ua_invoice` archetype now reaches it in six categories of every
run. The fixture stays because it pins the SHAPE independently of the registry — the mechanism must
hold for any pair of archetypes, not only for the pair that happens to be registered — and because
it is the only way to exercise the path for `vitamins_nutrition`, where the real planner prefers
the receipt.
"""


def _persona():
    return generate_persona(random.Random(20260803), persona_id="p001", country=Country.UA)


def test_a_claims_shape_follows_the_evidence_its_category_can_assemble():
    """What a claim's document list is, ASSERTED AGAINST THE REGISTRY rather than against a number.

    It read `len(plan.documents) == 1` while every registered archetype proved both facts, and the
    invoice made that false in six categories of seven. The property was never "one document" — it
    is that the planner prefers a single document proving both facts wherever one is registered, and
    assembles a pair where none is. Both branches are checked here, with the category named, so
    neither can disappear unnoticed.
    """
    subject = _persona()
    both_categories = [
        category_id
        for category_id in subject.benefit_categories
        if any(
            evidence_of(archetype) == (True, True)
            for archetype in archetypes_for(Country.UA, category_id)
        )
    ]
    assert both_categories, "no category has an archetype proving both facts"

    for category_id in both_categories:
        plan = plan_claim(
            random.Random(3), persona=subject, claim_id="c1",
            category=category_id, ledger=Ledger(),
        )
        assert len(plan.documents) == 1, category_id
        assert evidence_of(plan.documents[0].archetype) == (True, True)

    split = [
        category_id
        for category_id in subject.benefit_categories
        if category_id not in both_categories
        and can_assemble_evidence(archetypes_for(Country.UA, category_id))
    ]
    assert split, (
        "no category is documented by a pair, so the invoice archetype activated nothing — "
        "this test would assert only the branch that always held"
    )
    for category_id in split:
        plan = plan_claim(
            random.Random(3), persona=subject, claim_id="c1",
            category=category_id, ledger=Ledger(),
        )
        assert len(plan.documents) == 2, category_id
        assert {evidence_of(d.archetype) for d in plan.documents} == {(True, False), (False, True)}
        # The subject is dated on or before the payment: an invoice is issued and then settled.
        assert plan.subject_document.issued_at <= plan.documents[-1].issued_at


    assert both_categories or split, "the persona holds no documentable category at all"


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


def test_the_same_candidates_build_a_pair_or_a_gap_depending_only_on_the_intent():
    """🔴 THE POINT OF `EvidenceIntent`, ASSERTED WHERE THE TWO CANNOT BE CONFUSED: one archetype
    list, two calls, two shapes. A payment document alone is what the cause `subject_not_evidenced`
    needs, and it must be REACHED BY ASKING — not by a subject archetype failing to turn up, which
    is what the refusal one test below still means.

    Both calls are made against the same `PAIR_REGISTRY` candidates, so nothing about the registry
    can explain the difference. Were the gap a fallback, the default call would have produced it
    too and this would fail on the first assertion.
    """
    candidates = list(PAIR_REGISTRY.values())
    issued_at = datetime(2026, 6, 15, 12, 0)

    complete = claim_planner._select_documents(random.Random(3), candidates, issued_at)
    assert {evidence_of(d.archetype) for d in complete} == {(True, False), (False, True)}

    gap = claim_planner._select_documents(
        random.Random(3), candidates, issued_at,
        intent=claim_planner.EvidenceIntent.EVIDENCE_GAP,
    )
    assert len(gap) == 1, [d.archetype.slug for d in gap]
    assert evidence_of(gap[0].archetype) == (False, True), (
        "a gap claim proves the payment and not the subject; a self-contained document would "
        "leave nothing for the cause to be about"
    )
    assert gap[0].issued_at == issued_at, "the claim is dated by its proof of payment"


def test_an_evidence_gap_needs_a_payment_archetype_and_says_so_when_there_is_none():
    """The gap is in the SUBJECT and nowhere else. Asked for one where only a subject archetype is
    registered, the planner refuses instead of returning an invoice on its own — that claim proves
    no payment at all, which is `not_proof_of_payment`, a different verdict nobody asked for here.
    """
    with pytest.raises(ValueError, match="proves the payment alone"):
        claim_planner._select_documents(
            random.Random(3),
            [PAIR_REGISTRY["ua_invoice"]],
            datetime(2026, 6, 15, 12, 0),
            intent=claim_planner.EvidenceIntent.EVIDENCE_GAP,
        )


def test_a_subject_not_evidenced_claim_is_planned_as_a_payment_and_nothing_else():
    """The plan the cause needs, built through `plan_claim` rather than through the selector, so
    that the route from a drawn cause to a shape is what is pinned.

    The category is one the registry CAN document completely — `PAIR_REGISTRY` holds an invoice —
    which is the whole distinction: the subject document is available and is deliberately not
    planned. A gap that only occurred where nothing else was possible would be a shortage wearing
    the name of a decision.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
        plan = claim_planner.plan_claim(
            random.Random(3), persona=_persona(), claim_id="c1",
            category="vitamins_nutrition", ledger=Ledger(),
            verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=SUBJECT_NOT_EVIDENCED,
        )

    assert plan.intent is claim_planner.EvidenceIntent.EVIDENCE_GAP
    assert [d.archetype.slug for d in plan.documents] == ["ua_transfer"]
    assert plan.issued_at == plan.documents[0].issued_at
    # Nothing was sized for a basket: there is no document to carry one.
    assert plan.coverage_target is None
    assert plan.item_count is None


def test_a_rejected_plan_is_refused_by_the_engine_on_the_period_and_on_nothing_else():
    """The loop closed: the planner's displaced date, read back by the oracle that labels it.

    The two halves were tested apart — the engine's period check on hand-built documents further
    up this file, the planner's draw in tests/test_pipeline.py — and each can be right while the
    pair is wrong. A plan that displaced the SUBJECT document's date instead of the payment's, or
    that displaced by a fortnight into a window edge, would satisfy both halves and produce a
    `covered` claim under a `rejected` target.

    🔴 EVERY LINE OF THE BASKET IS COVERED HERE, deliberately: `covered_fraction` comes back 1.0
    and the verdict is still `rejected`, which is the whole of the distinction the two routes to
    that verdict draw. Nothing about this claim is unestablished and nothing about it is
    non-covered — the policy does not cover it because of WHEN the money moved.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", PAIR_REGISTRY)
        plan = claim_planner.plan_claim(
            random.Random(3), persona=_persona(), claim_id="c1",
            category="vitamins_nutrition", ledger=Ledger(), verdict=Verdict.REJECTED,
        )

    start, end = active_period()
    assert plan.cause == OUTSIDE_PERIOD
    assert not start <= plan.issued_at.date() <= end, plan.issued_at

    subject, payment = plan.documents
    assert plan.issued_at == payment.issued_at, "the claim is dated by its proof of payment"

    result = evaluate([
        doc("c1_d1", DocType.INVOICE, amount="600.00", when=subject.issued_at.date(),
            items=[item("600.00")]),
        doc("c1_d2", DocType.PAYMENT_CONFIRMATION, amount="600.00",
            when=payment.issued_at.date()),
    ])

    assert result.verdict is Verdict.REJECTED
    assert result.imperfection == (OUTSIDE_PERIOD,)
    assert result.covered_fraction == Decimal(1)
    assert result.reimbursable == Decimal("0.00")
    assert any("falls outside" in line for line in result.policy_trace)


def test_a_gap_claim_refuses_its_subject_document_by_naming_the_intent():
    """Two refusals, one method, and they must not read alike. A COMPLETE plan with no carrier is
    inconsistent — something went missing — while a gap plan has none by design, and a caller told
    "0 documents state what was bought" would go looking for a template rather than at its own
    assumption that every claim has a subject.
    """
    transfer = PAIR_REGISTRY["ua_transfer"]
    gap = ClaimPlan(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        verdict=Verdict.INSUFFICIENT_EVIDENCE, cause=SUBJECT_NOT_EVIDENCED,
        intent=claim_planner.EvidenceIntent.EVIDENCE_GAP,
        issued_at=datetime(2026, 6, 15, 12, 0),
        documents=(DocumentPlan(archetype=transfer, issued_at=datetime(2026, 6, 15, 12, 0)),),
    )
    with pytest.raises(ValueError, match="ON PURPOSE") as deliberate:
        _ = gap.subject_document

    # The same documents WITHOUT the intent are a plan that lost its subject, and that message is
    # the other one. Asserted as a pair: a single message serving both cases is the defect.
    accidental = ClaimPlan(
        claim_id="c2", persona_id="p001", category="vitamins_nutrition",
        verdict=Verdict.COVERED, issued_at=datetime(2026, 6, 15, 12, 0),
        documents=gap.documents,
    )
    with pytest.raises(ValueError, match="sized once") as lost:
        _ = accidental.subject_document

    assert "ON PURPOSE" not in str(lost.value)
    assert "plans 0 documents" not in str(deliberate.value)


def test_a_registry_that_cannot_establish_both_facts_is_refused():
    """An invoice archetype and nothing that proves payment. `covered` and
    `partially_covered` both need complete evidence, so building the claim anyway would
    mean planning a claim whose label is decided by what is missing.

    `verdict` is passed explicitly here, naming a verdict this planner draws, so the test
    exercises `_select_documents`'s own refusal rather than `plan_claim`'s newer guard on an
    empty `realizable_verdicts_for` — that guard fires first, and correctly so, once nothing
    is documentable at all, which a registry this bare also triggers when `verdict` is left
    to be drawn."""
    from receipt_synth import claim_planner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim_planner, "ARCHETYPES", {"ua_invoice": PAIR_REGISTRY["ua_invoice"]})
        with pytest.raises(ValueError, match="both what was bought"):
            claim_planner.plan_claim(
                random.Random(3), persona=_persona(), claim_id="c1",
                category="vitamins_nutrition", ledger=Ledger(), verdict=Verdict.COVERED,
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

    🔴 THE STUB MIRRORS THE LIVE SIGNATURE KEYWORD FOR KEYWORD, AND THAT IS LOAD-BEARING RATHER
    THAN TIDINESS. `assembler._payee_the_payment_names` passes `excluding_name=` on every claim
    planned as `counterparty_mismatch`, so a stub one parameter short raises `TypeError` the
    moment the seed stream shifts such a claim into this profile — it was green by seed luck
    alone. Verified rather than assumed: at seed 2, same personas and claims, the run plans one
    such claim, and the short stub failed there with "counting() got an unexpected keyword
    argument 'excluding_name'".

    ⚠️ AND THAT SECOND DRAW IS NOT COUNTED, because it is not the call this test is about. It
    asks for a party the claim's own vendor is NOT — a deliberate second seller for the
    payment document — while what is asserted here is that the claim's OWN vendor is drawn
    once. Counting both would make the assertion below fail at seed 2 for a call that is
    correct.
    """
    from receipt_synth import assembler

    calls: list[str] = []
    original = assembler._pick_vendor

    def counting(rng, country, category, *, mixed, vat_payer=None, excluding_name=None):
        if excluding_name is None:
            calls.append(category)
        return original(
            rng, country, category, mixed=mixed, vat_payer=vat_payer,
            excluding_name=excluding_name,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "_pick_vendor", counting)
        result = assembler.generate_dataset(
            seed=20260803,
            out_dir=tmp_path,
            train_fraction=0.5,
            personas=2,
            claims_per_persona=3,
        )

    assert calls, "this run built no claim"
    assert len(calls) == len(result.claims)


def test_document_ids_are_numbered_from_the_plan_and_not_fixed_at_one(tmp_path):
    """Inventory item 6. `_d1` was hardcoded; the id now comes from the position of the
    document in the plan, and the claim label points at exactly the documents written.

    NOT ADDRESSABLE UNTIL A SECOND ARCHETYPE REGISTERS, and that is expected rather than a
    weakness here. With one document per plan the expected list is `["<claim>_d1"]`, which a
    hardcoded `_d1` also produces — so reverting the fix does not turn this red. It starts
    discriminating the moment a plan carries two documents. Do not "strengthen" it by
    asserting something a single-document plan can distinguish; there is nothing.
    """
    from receipt_synth.assembler import generate_dataset

    result = generate_dataset(
        seed=20260803, out_dir=tmp_path, train_fraction=0.5, personas=2, claims_per_persona=3
    )

    written = {document.doc_id for document in result.documents}
    for claim, plan in zip(result.claims, result.plans, strict=True):
        expected = [f"{claim.claim_id}_d{i}" for i in range(1, len(plan.documents) + 1)]
        assert claim.documents == expected
        assert set(expected) <= written
        assert claim.linked is (len(expected) > 1)


def test_a_planned_archetype_with_no_builder_fails_by_name():
    """The other assumption the loop used to carry: every document was built by
    `build_prro_receipt`, whatever the plan said. An archetype nothing can produce has to say so
    rather than be handed to the one builder that exists.

    ⚠️ THE SUBJECT OF THIS TEST HAD TO MOVE. It planned a `ua_invoice`, which had no builder; it has
    one now, so the archetype stopped being unproducible and the test began asserting that a real
    builder raises on an empty vendor — a different thing entirely, and one nothing needed. An `act`
    takes its place: 📄 a type policy.yaml gives evidence for and no template produces. The
    assertion below checks that it genuinely has no builder, so the day one lands this test says so
    instead of passing while measuring nothing.
    """
    from receipt_synth import assembler

    unbuilt = Archetype(
        slug="ua_act", doc_type=DocType.ACT, country=Country.UA,
        language="uk", categories=("vitamins_nutrition",),
    )
    assert unbuilt.slug not in assembler._BUILDERS, (
        "this archetype has a builder now, so the refusal below cannot be reached — pick a slug "
        "that genuinely has none"
    )
    plan = ClaimPlan(
        claim_id="c1", persona_id="p001", category="vitamins_nutrition",
        verdict=Verdict.COVERED, issued_at=datetime(2026, 6, 15, 12, 0),
        documents=(DocumentPlan(archetype=unbuilt, issued_at=datetime(2026, 6, 15, 12, 0)),),
    )
    with pytest.raises(NotImplementedError, match="ua_act"):
        assembler._build_document(
            random.Random(1), persona=_persona(), plan=plan,
            document_plan=plan.documents[0], vendor={},
            # Hand-built rather than drawn: the refusal must happen BEFORE anything reads either,
            # and a drawn identity would need a vendor this test deliberately does not supply.
            identity=PartyIdentity(
                tax_code="12345678", vat_number=None, bank_name="bank",
                bank_code="123456", account="UA000000000000000000000000000",
            ),
            doc_id="c1_d1", renderer=None, out_dir=Path("."),
        )
