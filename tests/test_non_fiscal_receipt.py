"""The товарний чек: a page that reads like a fiscal receipt and carries no fiscal identity.

🔴 WHAT THIS CLASS IS FOR, AND WHAT THESE TESTS THEREFORE HAVE TO PROTECT. Every other archetype
of this corpus can be told apart from a fiscal receipt by its layout — an A4 sheet, a table of
transactions, a title in the wrong place. This one cannot: 📄 the tax service's rule is that its
content is the FISCAL RECEIPT'S OWN FORM less the fiscal number of the register and the wording
«ФІСКАЛЬНИЙ ЧЕК», so the basket, the arithmetic, the four totals and the columns are identical and
the whole difference is a set of requisites left out. A change that quietly printed one of them
back would destroy the only negative example the fiscality rule has, and no metric would move: the
document would still render, still label, still be read correctly field by field.

So the assertions below are mostly ABOUT WHAT IS ABSENT, checked on the rendered markup rather than
on the content object, because absence is a property of the page.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal

import pytest

from receipt_synth.claim_planner import ARCHETYPES, evidence_of
from receipt_synth.config import jurisdiction, load_policy, load_vendors
from receipt_synth.content_builder import (
    build_non_fiscal_receipt,
    draw_party_identity,
    line_items_total,
    resolve_vendor,
    validate_amount_in_words,
    validate_line_item_sum,
)
from receipt_synth.policy_engine import document_evidence
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, DocType
from test_renderer import printed_text

SLUG = "ua_non_fiscal_receipt"
SEED = 20260615
ISSUED_AT = datetime(2026, 6, 15, 17, 41, 9)

# THE ONLY KIND OF SELLER THIS DOCUMENT MAY HAVE: 📄 a registered ПДВ payer is obliged to use a
# cash register, so a slip is issued by a non-payer. Resolved once, as the assembler resolves a
# claim's vendor, because a sole trader's printed name is drawn rather than stored.
NON_PAYER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False},
    "UA",
)
PAYER = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}


def make(seed: int = SEED, vendor: dict = NON_PAYER, **kwargs):
    rng = random.Random(seed)
    return build_non_fiscal_receipt(
        rng,
        category_id=kwargs.pop("category_id", "vitamins_nutrition"),
        issued_at=ISSUED_AT,
        vendor=vendor,
        identity=draw_party_identity(random.Random(606), vendor, "UA"),
        address="м. Київ, вул. Хрещатик, 22",
        **kwargs,
    )


@pytest.fixture(scope="module")
def markup() -> str:
    """The built page, as HTML, WITH THE INLINED STYLESHEET STRIPPED — `test_renderer.printed_text`
    does the stripping and the reason is its own: the shared receipt stylesheet defines a
    `.vat-letter` rule and explains the fiscal requisites in Ukrainian comments, so a search over
    the whole document would find every word this file asserts is absent.

    Rendered through `build_html` rather than screenshotted: every question below is about what the
    page SAYS, and a browser adds nothing to that.
    """
    with Renderer() as renderer:
        return printed_text(renderer.build_html(SLUG, make().render_context()))


# --------------------------------------------------------------- who may issue it --


def test_a_registered_vat_payer_cannot_issue_one():
    """📄 A ПДВ payer is obliged to use a cash register, so a payer issuing a slip would be a page
    whose own requisites say it should not exist — and every VAT decision this builder makes (no
    «ПН» line, no letter on any line, no tax block) would then contradict the vendor record behind
    it.

    THE REFUSAL IS THE MECHANISM, not a comment: the caller chooses the vendor, and a builder that
    accepted a payer would print the contradiction silently on every such claim.
    """
    with pytest.raises(ValueError, match="cash register"):
        make(vendor=PAYER)


def test_the_archetype_carries_exactly_the_categories_that_have_a_non_payer_vendor():
    """🔴 THE REGISTRY AND config/vendors.json HAVE TO AGREE, and neither can be read off the
    other by a reader. The archetype declares six categories of seven; which six is a consequence
    of the rule above — a category served only by registered sellers cannot produce this document —
    and `medical_insurance` is served in config/vendors.json by insurers alone, every one of them
    registered.

    ⚠️ BOTH DIRECTIONS. A category dropped from the tuple while a non-payer sells there would
    silently remove a document class from that part of the corpus; a category added while every
    seller is registered would make `assembler._pick_vendor` raise mid-run, on the claim that
    happened to draw it.
    """
    vendors = load_vendors()["vendors"]["UA"]
    with_a_non_payer = {
        entry["id"]
        for entry in load_policy()["categories"]
        if any(not vendor["vat_payer"] for vendor in vendors.get(entry["id"], []))
    }
    assert with_a_non_payer, "no category has a non-payer vendor — this test asserts nothing"

    assert set(ARCHETYPES[SLUG].categories) == with_a_non_payer


def test_the_registration_says_what_the_document_proves_and_not_what_it_is_offered_as():
    """🔴 A SUBJECT-CLASS DOCUMENT. The claim shape it realizes is "submitted in place of a proof
    of payment", and that belongs to the PLANNER; registering the archetype as a payment class
    would contradict policy.yaml's `document_evidence` and relabel every claim carrying one.
    """
    assert evidence_of(ARCHETYPES[SLUG]) == (True, False)
    assert ARCHETYPES[SLUG].doc_type is DocType.NON_FISCAL_RECEIPT
    assert document_evidence(DocType.NON_FISCAL_RECEIPT) == (True, False)


# ------------------------------------------------------------ what is not on the page --


def test_the_page_carries_no_requisite_of_the_fiscal_form(markup):
    """🔴 THE ARCHETYPE IS ITS ABSENCES, and this is the test that says so. Each string below is a
    requisite of 📄 the published fiscal form that this document may not carry: the fiscal wording,
    the fiscal number of the register with either prefix, the «ЗН» factory serial, the online /
    offline marker, and the QR. A rendered slip carrying any of them is a fiscal receipt with a
    different title.

    THE PROBE IS THE CONFIGURED WORDING, not a literal typed here, so a jurisdiction that renames a
    requisite cannot leave this test looking for a string nothing prints.

    ⚠️ AND IT IS MATCHED AS A WHOLE WORD, which is not fussiness: «ЗН» is a substring of «ЗНИЖКА»,
    the discount label this document legitimately prints as one of 📄 the form's four amount lines.
    A plain `in` reported the slip as carrying a factory serial it does not have — a false finding
    on a correct page, which is the kind that gets a real assertion deleted.
    """
    rules = jurisdiction("UA")
    forbidden = [
        rules["receipt"]["title"],
        rules["receipt"]["registrars"]["prro"]["fiscal_number_label"],
        rules["receipt"]["registrars"]["rro"]["fiscal_number_label"],
        rules["identifiers"]["device_serial"]["label"],
        *rules["receipt"]["mode_markers"],
        # ⚠️ AND THE STRING THE CONTRACT CALLS THE NEGATIVE MARKER, for the opposite reason: no
        # public source shows «НЕ ФІСКАЛЬНИЙ ЧЕК» on a Ukrainian SALES document, so printing it
        # would be inventing an observation. See RC-08 — the decision is the author's and pending.
        rules["receipt"]["non_fiscal_marker"],
    ]
    letters = "A-Za-zА-ЯІЇЄҐа-яіїєґ"
    for wording in forbidden:
        found = re.search(
            rf"(?<![{letters}]){re.escape(wording)}(?![{letters}])", markup
        )
        assert not found, f"the slip prints {wording!r}, a requisite of the fiscal form"
    assert "<svg" not in markup, "the slip prints a QR — 📄 a requisite of the fiscal form"


def test_the_page_identifies_itself_positively(markup):
    """The other half, and the one that makes the absences readable: what stands in the fiscal
    title's slot is «ТОВАРНИЙ ЧЕК», which 📄 the tax service requires such a document to carry.
    A page with neither wording would be an unidentifiable document rather than this class."""
    title = jurisdiction("UA")["receipt"]["non_fiscal"]["title"]
    assert f'data-field="title">{title}<' in markup


def test_no_line_carries_a_vat_letter_and_no_tax_row_is_printed(markup):
    """A non-payer has assigned no rate group to anything, so 👁 the line ends with the amount —
    not a zero-rate letter, not "Без ПДВ". The LABEL and the PAGE have to agree about that: a
    letter in the ground truth that no reader can see would be a value extracted from nowhere."""
    document = make()
    assert document.line_items, "the slip lists nothing — nothing was checked"
    assert all(item.vat_letter is None for item in document.line_items)
    assert "vat-letter" not in markup
    assert "ПДВ" not in markup


def test_the_label_records_no_fiscality_of_any_kind():
    """The three flags a consumer classifies on, on a page that otherwise reads like a fiscal
    receipt. This is the record RC-08's rule would be scored against."""
    record = make().ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCAN, field_bboxes={}
    )
    assert record.doc_type is DocType.NON_FISCAL_RECEIPT
    assert record.has_fiscal_number is False
    assert record.has_qr is False
    assert record.qr_is_fiscal is False
    assert record.vat_row_form is None
    # 📄 The form this document follows names no buyer — the payer was standing at the counter —
    # so the class names one party and the label says so.
    assert record.payer is None
    assert record.counterparty == NON_PAYER["name"]


# ------------------------------------------------------------------ the money on it --


def test_the_money_on_the_page_is_the_money_in_the_label():
    """Three statements of one amount — the stored total, the line items, and the amount in words —
    checked against each other by the invariant validators rather than by re-deriving them here.

    ⛔ THIS ARCHETYPE DOES NOT BREAK INVARIANTS. It is a negative example about EVIDENCE, not a
    fraud archetype: its arithmetic is exactly as sound as a fiscal receipt's, which is what makes
    it hard — nothing about the numbers gives it away.
    """
    document = make()
    assert validate_line_item_sum(document.line_items, document.total)
    assert validate_amount_in_words(document.amount_in_words, document.total)
    assert document.amount_due == document.total
    assert document.total == line_items_total(document.line_items)


def test_the_payment_method_is_cash():
    """🔴 THE FIRST DOCUMENT OF THE CORPUS TO PRINT «ГОТІВКА», and it is a consequence rather than
    a preference: 📄 a card sale is a settlement operation that obliges the seller to use a
    register, so a slip issued without one records cash. Read from config — the second entry of
    `payment_method_labels`, which was unreachable until this class landed."""
    labels = jurisdiction("UA")["acquiring_block"]["payment_method_labels"]
    assert make().payment_method == labels[1]
    assert labels[1] != labels[0], "the two payment methods are one string — nothing distinguishes"


def test_a_mixed_basket_is_drawn_when_the_plan_asks_for_one():
    """The class takes the same basket knobs as every other subject document, and the coverage a
    basket comes to must not depend on which class carries it — see `content_builder._draw_basket`.

    Asserted through the KNOB rather than the arithmetic: what has to hold here is that this
    builder honours it at all, since a claim planned as `partially_covered` may be documented by
    any subject class the registry offers.
    """
    mixed = make(covered_only=False, coverage_target=Decimal("0.6"))
    assert any(not item.covered for item in mixed.line_items), "no non-covered line was drawn"
    assert any(item.covered for item in mixed.line_items), "no covered line was drawn"


def test_the_number_is_a_short_sequential_one_and_not_a_registers_identifier():
    """👁 A hand-kept book of товарні чеки is numbered sequentially. A ПРРО's eleven-character
    alphanumeric id would assert that a register produced the document, which is the one thing this
    class says did not happen."""
    rules = jurisdiction("UA")["receipt"]
    number = make().receipt_number
    assert re.fullmatch(rules["non_fiscal"]["number"]["pattern"], number), number
    assert not re.fullmatch(
        rules["receipt_number"]["alphanumeric"]["pattern"], number
    ), "the slip's number has a ПРРО's shape"


def test_two_runs_of_one_seed_build_the_same_slip():
    """Determinism under `--seed`, on the class rather than on the pipeline: every draw this
    builder makes goes through the generator it was handed."""
    assert make() == make()
