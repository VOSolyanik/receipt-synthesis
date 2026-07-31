"""The invoice: an OFFER TO PAY, and the subject document of the dominant pair.

🔴 WHAT THIS FILE MOSTLY ASSERTS IS AN ABSENCE, which is unusual and is the point. 📄 A Ukrainian
рахунок на оплату is not a primary accounting document — it proposes that the buyer pay, and the
fact of payment is established by a payment document. So the class must NOT carry a payment status
and must NOT carry an `amount_due`, and the consumer's requirement asking for the first of those is
recorded in the contract as a divergence rather than satisfied. An absence is easy to introduce by
accident and impossible to notice in a render, so it is tested rather than trusted.

Expected values are derived from config/policy.yaml, config/fiscal-rules.yaml and the arithmetic of
the basket. Nothing was copied out of a run.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal

import pytest
import yaml

from receipt_synth.config import CONFIG_DIR, category, jurisdiction, load_policy
from receipt_synth.content_builder import (
    Invoice,
    build_invoice,
    draw_party_identity,
    generate_rnokpp,
    is_valid_edrpou,
    is_valid_iban,
    is_valid_rnokpp,
    line_items_total,
    printed_legal_name,
    resolve_vendor,
    validate_amount_in_words,
    validate_line_item_sum,
)
from receipt_synth.renderer import TEMPLATES_DIR, Renderer
from receipt_synth.schemas import Capture, DocType

SLUG = "ua_invoice"
BLOCK = jurisdiction("UA")["invoice"]

DPI = 96
MM_PER_INCH = Decimal("25.4")

# A registered company and an unregistered sole trader. The pair exercises the one conditional this
# class has — 📄 a seller not registered for ПДВ prices without it, so the money columns lose their
# suffix and the tax line disappears. It is the same vendor property that decides a receipt's VAT
# block, which is why it is resolved once in the builder rather than twice in two templates.
PAYER = {
    "name": "Гімназія №1", "legal_form": "TOV", "profile": "language_school",
    "vat_payer": True,
}
PAYER_CATEGORY = "language_courses"
NON_PAYER = {"legal_form": "FOP", "profile": "therapy_practice", "vat_payer": False}
# Its own category, because a vendor sells what its profile sells: a therapy practice cannot
# invoice language lessons, and `build_invoice` says so rather than drawing something it lacks.
NON_PAYER_CATEGORY = "mental_health"

BUYER_NAME = "Ковальчук Олена Петрівна"
# A CHECKSUM-CORRECT РНОКПП, generated rather than typed — it is swept by the checksum test below,
# and a hand-typed ten-digit string failed it. The same fixture defect as in test_bank_statement.py,
# and the same fix: a persona's tax id is checksum-correct, so the fixture must be too.
BUYER_CODE = generate_rnokpp(random.Random(5))
WHEN = datetime(2026, 6, 3, 10, 15)


def make_invoice(
    seed: int = 20260603,
    vendor: dict = PAYER,
    category_id: str = PAYER_CATEGORY,
    coverage_target: str | None = None,
) -> Invoice:
    rng = random.Random(seed)
    resolved = resolve_vendor(rng, vendor, "UA")
    return build_invoice(
        rng,
        category_id=category_id,
        issued_at=WHEN,
        vendor=resolved,
        identity=draw_party_identity(rng, resolved, "UA"),
        buyer_name=BUYER_NAME,
        buyer_tax_id=BUYER_CODE,
        address="м. Київ, вул. Хрещатик, 22",
        covered_only=coverage_target is None,
        coverage_target=Decimal(coverage_target) if coverage_target else None,
    )


def label(invoice: Invoice, boxes: dict | None = None):
    return invoice.ground_truth(
        doc_id="c1_d1",
        source_file="c1_d1.png",
        capture=Capture.SCREENSHOT,
        field_bboxes=boxes or {},
    )


def printed_text(html: str) -> str:
    """The rendered page with its markup removed, for asserting what a reader sees."""
    return re.sub(r"<[^>]+>", " ", html)


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    invoice = make_invoice()
    output = tmp_path_factory.mktemp("invoice") / "invoice.png"
    return invoice, renderer.render(SLUG, invoice.render_context(), output)


# ------------------------------------------- what the class deliberately does NOT carry --


def test_the_label_carries_no_payment_status_under_any_name():
    """🔴 THE CENTRAL ABSENCE. 📄 An invoice is an offer to pay; the fact of payment is established
    by a payment document, and 👁 0 of 2 open invoices print any status.

    Asserted over the WHOLE record rather than by naming a field, because the failure to guard
    against is a status arriving under some other name — `paid`, `settled`, a boolean somewhere. The
    record's field set is fixed by `DocGroundTruth`, so this is a real sweep with a real
    denominator: every field of the model, checked for a value readable as a payment status.
    """
    record = label(make_invoice())
    fields = record.model_dump()

    assert fields, "no fields to sweep — this test would assert nothing"
    forbidden = [name for name in fields if any(
        token in name for token in ("paid", "status", "settle")
    )]
    assert not forbidden, (
        f"{forbidden} could be read as a payment status on a document that carries none, over "
        f"{len(fields)} fields checked"
    )


def test_the_label_carries_no_amount_due():
    """🔴 `amount_due` IS A RECEIPT REQUISITE — 📄 line 24 of the fiscal receipt form, where it
    differs from «СУМА» by the discount and the rounding. 👁 An invoice has ONE total block, so there
    is nothing for a second amount to differ from, and a populated field would be `amount` twice
    under two names — the failure the labelling contract exists to prevent."""
    record = label(make_invoice())

    assert record.amount_due is None
    assert record.fee is None
    assert record.total_charged is None
    assert record.direction is None
    assert record.relevant_transaction is None


def test_the_page_prints_no_payment_status_either(renderer):
    """The same absence on the RENDER, because a label and a page can disagree. Swept over the
    Ukrainian and Polish words a status would be written in — 📄 the consumer's own example is the
    Polish «Zapłacono», which is where that requirement came from."""
    for vendor, category_id in ((PAYER, PAYER_CATEGORY), (NON_PAYER, NON_PAYER_CATEGORY)):
        text = printed_text(
            renderer.build_html(
                SLUG, make_invoice(vendor=vendor, category_id=category_id).render_context()
            )
        )
        for token in ("Оплачено", "Не оплачено", "Zapłacono", "Сплачено", "ДО СПЛАТИ"):
            assert token not in text, f"{token!r} is printed on an invoice"


def test_no_line_item_carries_a_vat_letter():
    """⛔ THE OBSERVED TABLE HAS NO PER-LINE VAT LETTER COLUMN — it prices VAT-inclusive and states
    the tax once at the foot. A letter labelled and not printed would be a ground-truth value
    unreadable from the image, which is the rule that gave the bank statement its second money
    column. Checked on a REGISTERED payer, where a receipt would carry one on every line."""
    invoice = make_invoice(vendor=PAYER)

    assert invoice.vat_payer, "a non-payer carries no letters anyway — this would prove nothing"
    assert [item.vat_letter for item in invoice.line_items] == [None] * len(invoice.line_items)


# ------------------------------------------------------ what it does carry --


def test_the_label_is_the_total_the_lines_come_to():
    """The invariant every basket document shares, and the reason the invoice can be the subject
    document of a pair: its amount is derived from what was bought."""
    invoice = make_invoice()
    record = label(invoice)

    assert record.doc_type is DocType.INVOICE
    assert record.amount == line_items_total(invoice.line_items)
    assert validate_line_item_sum(record.line_items, record.amount)
    assert record.date == WHEN.date()


def test_the_counterparty_is_the_supplier_and_the_payer_is_the_claimant():
    """`counterparty` is the party opposite the claimant on every class; here that is the SUPPLIER.
    ⚠️ The buyer is the claimant — an invoice addressed to anybody else would evidence nothing about
    the persona filing the claim, the same reasoning that puts the persona on a statement's
    account."""
    record = label(make_invoice())

    assert record.counterparty == PAYER["name"]
    assert record.payer == BUYER_NAME


@pytest.mark.parametrize("seed", range(10))
def test_the_amount_in_words_states_the_total(seed):
    """📄 The words are obligatory in practice and are the same requisite written twice, which is
    what makes them a cross-check a consumer can run against the figure."""
    invoice = make_invoice(seed)
    context = invoice.render_context()

    assert validate_amount_in_words(context["amount_in_words"], invoice.total)
    if invoice.vat_payer:
        assert validate_amount_in_words(context["vat_in_words"], invoice.vat_total)


@pytest.mark.parametrize("seed", range(10))
def test_the_tax_is_the_tax_contained_within_the_total(seed):
    """👁 Prices are VAT-INCLUSIVE, so the tax is extracted from the gross rather than added on top —
    and the invoice's single «У т.ч. ПДВ» is the sum of what a receipt would print per rate. Derived
    from the letters before they are dropped, so the figure is the same one either class states."""
    invoice = make_invoice(seed)

    if not invoice.vat_payer:
        assert invoice.vat_total == 0
        return
    # 20% on a VAT-inclusive gross is gross/6, to the kopiyka, for a basket wholly at one rate.
    assert 0 < invoice.vat_total < invoice.total
    assert invoice.vat_total <= (invoice.total / 6).quantize(Decimal("0.01"))


@pytest.mark.parametrize("seed", range(10))
def test_every_identifier_printed_on_an_invoice_passes_its_own_checksum(seed):
    """A broken identifier in this repository has to be a labelled choice of a fraud archetype,
    never a side effect of a generator that did not bother."""
    invoice = make_invoice(seed)

    assert is_valid_iban(invoice.supplier.account)
    code = invoice.supplier.code
    assert len(code) in (8, 10), code
    assert (is_valid_rnokpp if len(code) == 10 else is_valid_edrpou)(code)
    assert is_valid_rnokpp(invoice.buyer.code)


def test_a_sole_trader_signs_in_their_own_name_and_states_no_post():
    """📄 The only two places a sole trader's invoice differs from a company's: the code register
    and the signature. A ФОП signs in their own hand and names no post; a company names the post
    of the authorized person."""
    trader = make_invoice(vendor=NON_PAYER, category_id=NON_PAYER_CATEGORY)
    company = make_invoice(vendor=PAYER)

    assert trader.signatory_post is None
    assert trader.signatory_name == trader.supplier.name
    assert company.signatory_post in BLOCK["signature"]["posts"]
    assert company.signatory_name != company.supplier.name


def test_a_non_payers_invoice_drops_the_tax_line_and_the_column_suffix(renderer):
    """📄 A seller not registered for ПДВ prices without it: one total line, no tax line, and money
    columns captioned without the «з ПДВ» suffix. The same vendor property that decides a receipt's
    VAT block."""
    payer = make_invoice(vendor=PAYER).render_context()
    non_payer = make_invoice(vendor=NON_PAYER, category_id=NON_PAYER_CATEGORY).render_context()

    assert payer["vat_total"] is not None
    assert payer["columns"]["price"] == BLOCK["columns"]["price"]
    assert non_payer["vat_total"] is None
    assert non_payer["columns"]["price"] == BLOCK["columns"]["price_no_vat"]

    text = printed_text(renderer.build_html(SLUG, non_payer))
    assert BLOCK["totals"]["single_label"] in text
    assert BLOCK["totals"]["vat_label"] not in text


def test_the_title_writes_its_date_in_words_and_the_body_in_digits(rendered):
    """👁 BOTH DATE SPELLINGS ON ONE PAGE — the title reads «від 3 червня 2026 р.» while the
    agreement line and any validity line use digits. That is a parsing case a corpus of receipts
    alone never presents, and it is reproduced because the observed invoice does it. The label
    carries the calendar date; `normalization.iso_date` settles the comparison."""
    invoice, _ = rendered
    title = invoice.render_context()["title"]
    month = BLOCK["long_date_months"][WHEN.month - 1]

    assert f"{WHEN.day} {month} {WHEN.year}" in title
    assert WHEN.strftime(jurisdiction("UA")["date_format"]) not in title
    assert label(invoice).date == WHEN.date()


# ------------------------------------------------------------------ the page --


def test_the_page_is_one_A4_sheet(rendered):
    _, result = rendered
    width = round(Decimal(BLOCK["page"]["width_mm"]) * DPI / MM_PER_INCH)
    height = round(Decimal(BLOCK["page"]["height_mm"]) * DPI / MM_PER_INCH)

    assert (result.width, result.height) == (width, height)


def test_the_stylesheet_renders_the_paper_the_configuration_declares():
    """The pixels in `<slug>.css` against the millimetres in config/fiscal-rules.yaml. Without this
    the two are unrelated numbers."""
    css = (TEMPLATES_DIR / f"{SLUG}.css").read_text(encoding="utf-8")
    width = round(Decimal(BLOCK["page"]["width_mm"]) * DPI / MM_PER_INCH)
    height = round(Decimal(BLOCK["page"]["height_mm"]) * DPI / MM_PER_INCH)

    assert f"width: {width}px;" in css
    assert f"min-height: {height}px;" in css


def test_the_specimen_payment_order_prints_the_recipient_a_second_time(renderer):
    """⚠️ 👁 1/1 — the classic 1С/BAS invoice carries a filled-in sample payment order above the
    title, so THE RECIPIENT'S NAME, CODE AND ACCOUNT APPEAR TWICE on one sheet.

    That is a genuine ambiguity for an extractor — which of two identical strings to return — and it
    is reproduced rather than smoothed away. Asserted by COUNTING occurrences in the rendered text,
    because "the block is present" would pass on a block that repeated nothing, which is exactly
    what the first draft of this test did.
    """
    invoice = make_invoice()
    text = printed_text(renderer.build_html(SLUG, invoice.render_context()))

    for value in (
        printed_legal_name(invoice.supplier.name, invoice.supplier.legal_form),
        invoice.supplier.code,
        invoice.supplier.account,
    ):
        assert text.count(value) == 2, (
            f"{value!r} appears {text.count(value)} times; the specimen payment order and the "
            "supplier block each print it once, and that duplication is the point"
        )
    # And the BUYER is named once — the specimen is a payment order TO the supplier, so only the
    # recipient's side is repeated. Without this the test would pass on a page that printed
    # everything twice.
    assert text.count(invoice.buyer.code) == 1


def test_the_indexed_line_item_boxes_follow_the_lines(rendered):
    """One set of boxes per line item, ordered down the page, so a consumer pairs box i with line
    item i without matching text. The same convention as the receipt's."""
    invoice, result = rendered

    tops = []
    for index in range(len(invoice.line_items)):
        assert f"item_{index}_name" in result.field_bboxes
        tops.append(result.field_bboxes[f"item_{index}_name"][1])
    assert tops == sorted(tops), "line items are not in document order"


def test_the_total_box_sits_below_every_line_item(rendered):
    _, result = rendered
    item_names = [
        name for name in result.field_bboxes if re.fullmatch(r"item_\d+_name", name)
    ]
    assert item_names, "no line-item boxes — this test would assert nothing"

    assert result.field_bboxes["total"][1] > max(
        result.field_bboxes[name][1] for name in item_names
    )


# ----------------------------------------------- the contract, on the two absences --


def contract_invoice() -> dict:
    contract = yaml.safe_load(
        (CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8")
    )
    return contract["document_types"]["invoice"]


def test_the_contract_records_the_payment_status_as_a_divergence_and_not_as_a_field():
    """🔴 The consumer's requirement asks this type for a field the document does not carry. The
    contract's job is to make that visible and to name both sides — never to satisfy it and never to
    inherit it by omission.

    Two files again, two independently maintained sides: the requirement is restated in the
    contract, and the generator emits NOTHING for it. A row that quietly acquired a generator
    counterpart would mean somebody had printed a status.

    The emptiness is asserted on the LIST rather than on a sentence. That column was prose until
    contract version 21 and this line read `== "none, and none is wanted"`, which pinned an
    editorial phrasing: rewording the cell reddened the test while adding a field to it did not.
    Now the two are the other way round, which is the direction that matters.
    """
    rows = contract_invoice()["prd_required_fields"]
    status_rows = [row for row in rows if "PAYMENT STATUS" in str(row["prd"]).upper()]

    assert len(status_rows) == 1, f"{len(status_rows)} of {len(rows)} rows mention a payment status"
    row = status_rows[0]
    assert row["status"] == "divergent"
    assert row["generator"] == []


def test_every_category_of_the_policy_can_be_invoiced():
    """The registry claims this archetype carries every category. That is a positive claim about
    BASKETS — unlike the bank classes, an invoice lists items — so it is checked by building one for
    each rather than asserted in a comment."""
    categories = [entry["id"] for entry in load_policy()["categories"]]
    assert len(categories) == 7, f"{len(categories)} categories — the sweep's denominator moved"

    from receipt_synth.claim_planner import ARCHETYPES

    assert set(ARCHETYPES["ua_invoice"].categories) == set(categories)
    for category_id in categories:
        # The FIRST vendor of each category, not a drawn one: the assertion is that every category
        # can be invoiced at all, and a draw would make a failure depend on the seed.
        invoice = make_invoice(vendor=_vendors_for(category_id)[0], category_id=category_id)
        assert invoice.line_items
        assert invoice.total > 0
        assert set(category(category_id)["covered_items"]) >= {
            item.item_kind for item in invoice.line_items
        }


def _vendors_for(category_id: str) -> list[dict]:
    from receipt_synth.config import load_vendors

    return load_vendors()["vendors"]["UA"][category_id]
