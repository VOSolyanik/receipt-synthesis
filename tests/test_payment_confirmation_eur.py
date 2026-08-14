"""The euro payment confirmation — the domestic form with one axis moved.

WHY THE ARCHETYPE EXISTS, stated once here because every assertion below serves it: the corpus
had euros on `platform_receipt` alone, and that class proves both facts, so every euro document
was the whole of its claim. The conversion the oracle performs was therefore never exercised
across a PAIR, and never on a class a consumer extracts. This document is the payment half of a
euro pair.

The groups below are: what the page must SAY about its currency (a label recording EUR has to be
readable off the image), what the form does NOT grow to accommodate a foreign beneficiary, and
what the label carries.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal

import pytest

from receipt_synth.claim_planner import _CITES_THE_SETTLED_DOCUMENT, ARCHETYPES
from receipt_synth.config import jurisdiction
from receipt_synth.content_builder import (
    DocumentReference,
    build_payment_confirmation,
    draw_party_identity,
    words_to_amount_uk,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, DocType

SLUG = "ua_bank_payment_confirmation_eur"
CATEGORY = "professional_development"
BLOCK = jurisdiction("UA")["payment_confirmation"]

PLATFORM = {
    "name": "Coursera", "legal_form": "INC",
    "profile": "online_learning_platform", "vat_payer": False,
}
AMOUNT = Decimal("394.10")


def make(seed: int = 20260615, initiation: str = "transfer", **kwargs):
    """Built exactly as `assembler` builds it — the currency and the initiation mode are the
    partial in `_BUILDERS`, and the identity is drawn in the seller's own pool."""
    rng = random.Random(seed)
    return build_payment_confirmation(
        rng,
        issued_at=datetime(2026, 6, 15, 17, 41, 9),
        vendor=PLATFORM,
        identity=draw_party_identity(random.Random(seed + 1), PLATFORM, "EU"),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        amount=AMOUNT,
        currency="EUR",
        initiation=initiation,
        **kwargs,
    )


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as browser:
        yield browser


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    return renderer.render(
        SLUG, make().render_context(), tmp_path_factory.mktemp("fx") / "p.png"
    )


# ============================================================== the registration ==


def test_it_is_the_payment_half_of_a_euro_pair_and_takes_nothing_out_of_the_hryvnia_one():
    """🔴 THE BLAST-RADIUS QUESTION, ANSWERED THE OTHER WAY ROUND FROM THE PLATFORM RECEIPT.
    That archetype proves BOTH facts, so every category it carried lost its invoice-plus-payment
    pair — `_select_documents` prefers a self-contained document. This one proves the payment
    ALONE: it joins the payment pool of one category and removes nothing from it."""
    archetype = ARCHETYPES[SLUG]

    assert archetype.doc_type is DocType.PAYMENT_CONFIRMATION
    assert archetype.currency == "EUR"
    assert archetype.language == "uk"
    assert archetype.categories == (CATEGORY,)
    assert archetype.vendor_pool == "EU"
    assert SLUG in _CITES_THE_SETTLED_DOCUMENT, (
        "the same form prints the same purpose line, so it can cite the рахунок it settles"
    )


# ================================================ what the page says about the currency ==


def test_the_caption_names_the_currency_the_label_records(rendered):
    """⛔ A BARE FIGURE WOULD MAKE THE LABEL UNREADABLE FROM THE IMAGE. Every other document of
    this corpus prints one currency, so a number needs no code; this one does. The form is the
    observed «Сума (грн)» with the code that applies."""
    document = make()

    assert "(EUR)" in document.amount_caption
    assert "(EUR)" in document.fee_caption
    assert "(EUR)" in rendered.reference_text
    assert "грн" not in rendered.reference_text


def test_the_words_spell_euros_and_state_the_same_number(rendered):
    """The second, independent statement of the currency — and of the amount. The reading back
    is `words_to_amount_uk`, which is not the formatter re-run."""
    document = make()
    assert document.amount_in_words is not None, (
        "this seed has to draw the words, or the test asserts nothing about them"
    )

    assert "євро" in document.amount_in_words
    assert "копій" not in document.amount_in_words
    assert words_to_amount_uk(document.amount_in_words) == AMOUNT
    assert document.amount_in_words in rendered.reference_text


def test_a_caption_naming_hryvnias_is_never_drawn():
    """The two observed captions that name the domestic currency are a STATEMENT about the
    figure beside them, so a euro page cannot print either — at any seed."""
    domestic = set(BLOCK["captions_naming_the_domestic_currency"])

    for seed in range(200):
        document = make(seed)
        assert document.amount_caption not in domestic
        assert document.fee_caption not in domestic


# ============================================ what the form does NOT grow, and what it drops ==


def test_a_card_operation_in_a_foreign_currency_is_refused_rather_than_rendered():
    """👁 The card modes print an authorization code and a masked card — a domestic acquiring
    operation. What a bank executes against a foreign beneficiary's account is a transfer by
    account details, and nothing observed here says what the other document would look like."""
    with pytest.raises(ValueError, match="prints a card"):
        make(initiation="card")


def test_the_foreign_beneficiarys_bank_carries_no_domestic_bank_code(rendered):
    """📄 The МФО is assigned in the National Bank's register of participants, so a bank outside
    it has none — and the beneficiary's IBAN is what says the account is outside. The name is
    printed; the code is omitted rather than invented."""
    document = make()

    assert document.payee.account.startswith("DE")
    assert document.payee.bank is None or BLOCK["parties"]["bank_code_label"] not in (
        document.payee.bank
    )
    assert document.payer.account.startswith("UA"), (
        "the payer is a Ukrainian claimant at a Ukrainian bank — only the beneficiary is abroad"
    )


def test_the_page_prints_no_requisite_that_was_never_observed(rendered):
    """⛔ THE DECLARED NARROWING. A cross-border instruction adds a BIC, a correspondent bank and
    a charge option, and no such document was observed here — so the form prints the requisites
    it has evidence for and stops. This is what makes that a choice rather than an omission."""
    for absent in ("BIC", "SWIFT", "OUR", "SHA", "BEN", "Кореспондент"):
        assert absent not in rendered.reference_text


# ==================================================================== the label ==


def test_the_label_is_euro_and_ukrainian_and_still_proves_only_the_payment():
    document = make()
    truth = document.ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCAN, field_bboxes={}
    )

    assert truth.doc_type is DocType.PAYMENT_CONFIRMATION
    assert truth.currency == "EUR"
    assert truth.language == "uk"
    assert truth.amount == AMOUNT
    assert truth.line_items == [], (
        "a confirmation lists nothing — which is why it proves no subject"
    )
    assert truth.counterparty == "Coursera"


def test_the_fee_and_the_total_are_in_the_same_currency_as_the_transfer():
    """One page, one currency — exactly as one claim is. Nothing on the page says which of the
    three amounts a code applies to, and nothing has to."""
    for seed in range(50):
        document = make(seed)
        assert document.total_charged == document.transfer + document.fee


def test_it_can_cite_the_invoice_it_settles():
    """The purpose line is the form's, and the currency axis does not touch it — which is what
    lets a euro claim realize `subject_mismatch` exactly as a hryvnia one does."""
    document = make(
        must_cite=True,
        cites=DocumentReference(number="4417", issued_at=datetime(2026, 6, 10, 9, 0)),
    )

    assert document.cites_document_no == "4417"
    assert "4417" in (document.purpose or "")
