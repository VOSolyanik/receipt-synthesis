"""The cross-border invoice — the subject half of a euro claim.

THE ARCHETYPE'S REASON TO EXIST IS A SELECTION FACT, not a document one, and the first group below
is about that: a euro claim used to be a `platform_receipt`, which proves both facts and is
therefore the whole of its claim, so the oracle's conversion never ran across a pair and never
landed on a class a consumer extracts. The rest of the file is what the page states — an OFFER to
pay and never a record of payment — and what it deliberately does not.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

import pytest

from receipt_synth.claim_planner import (
    ARCHETYPES,
    STATES_AN_INSTALMENT_TERM,
    Evidence,
    _pairable_subjects,
    _payment_archetypes,
    _settleable_subjects,
    archetypes_for,
    evidence_of,
)
from receipt_synth.config import (
    eu_tax_treatment_shares,
    jurisdiction,
    load_fx_rates,
    load_generation,
    load_vendors,
    partial_payment_schedules,
    tax_on_top_rules,
)
from receipt_synth.content_builder import (
    build_eu_invoice,
    draw_party_identity,
    line_items_total,
    vendor_can_carry,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, Country, DocType

SLUG = "eu_invoice"
CATEGORY = "language_courses"
PLATFORM = {
    "name": "italki", "legal_form": "INC",
    "profile": "online_language_platform", "vat_payer": False,
}


def make(seed: int = 20260610, vendor: dict = PLATFORM, **kwargs):
    rng = random.Random(seed)
    return build_eu_invoice(
        rng,
        category_id=CATEGORY,
        issued_at=datetime(2026, 6, 10, 9, 0, 0),
        vendor=vendor,
        identity=draw_party_identity(random.Random(seed + 1), vendor, "EU"),
        buyer_name="Ковальчук Олена Петрівна",
        buyer_tax_id="2345678901",
        **kwargs,
    )


def form_of(document) -> str:
    """Which of the three tax forms this page took, read off what it stores — the same three
    names `eu_tax_treatment` in config/generation.yaml draws by."""
    if document.tax_amount is None:
        return "out_of_scope"
    return "reverse_charge" if document.tax_amount == 0 else "tax_on_top"


def seed_in_form(form: str) -> int:
    """A seed whose drawn treatment is `form` — found by scanning rather than pinned, so a share
    edit cannot silently leave a test asserting against the wrong form."""
    for seed in range(400):
        if form_of(make(seed)) == form:
            return seed
    raise AssertionError(f"no seed in 0..399 draws the {form!r} form — the draw is broken")


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as browser:
        yield browser


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    return renderer.render(
        SLUG, make().render_context(), tmp_path_factory.mktemp("eu-invoice") / "i.png"
    )


# ================================================ the registration, and why this category ==


def test_it_proves_the_subject_and_no_payment_like_every_invoice():
    archetype = ARCHETYPES[SLUG]

    assert archetype.doc_type is DocType.INVOICE
    assert evidence_of(archetype) == Evidence(True, False)
    assert archetype.currency == "EUR"
    assert archetype.language == "en"
    assert archetype.vendor_pool == "EU"


def test_the_category_it_is_registered_for_is_one_that_still_has_a_pair():
    """🔴 THE CHOICE THE WHOLE REGISTRATION TURNS ON. `_select_documents` PREFERS a document that
    proves both facts wherever one is registered, so a category carrying one never gets an
    invoice-plus-payment pair at any seed. `professional_development` — where the `EU` sellers
    already were — carries the two platform receipts; `language_courses` does not carry any
    such archetype, which is why the euro pair can be drawn there and only there.

    ⛔ THIS IS THE ASSERTION THAT WOULD HAVE CAUGHT THE MISTAKE. The pair was first registered
    for `professional_development`, rendered correctly in every test, and could not have reached
    a corpus."""
    assert ARCHETYPES[SLUG].categories == (CATEGORY,)

    for category in ARCHETYPES[SLUG].categories:
        candidates = archetypes_for(Country.UA, category)
        assert not [a for a in candidates if all(evidence_of(a))], (
            f"{category!r} carries an archetype that proves both facts, so a claim there is one "
            "document and this invoice would never be drawn into a pair"
        )


def test_the_euro_payment_document_beside_it_is_registered_for_the_same_category():
    """A subject archetype whose currency no payment document shares is a template that renders
    and never reaches a complete claim — `_settleable_subjects` would drop it silently."""
    candidates = archetypes_for(Country.UA, CATEGORY)
    payments = _payment_archetypes(candidates)
    settleable = _settleable_subjects(_pairable_subjects(candidates), payments)

    assert ARCHETYPES[SLUG] in settleable
    assert [p.slug for p in payments if p.currency == "EUR"] == [
        "ua_bank_payment_confirmation_eur"
    ]


def test_the_seller_pool_spans_both_baskets_and_both_trades():
    """🔴 THE POOL, NOT THE SELLER, IS WHAT HAS TO SPAN THE CATEGORY — the same shape as
    professional_development, where the marketplace carries the mixed basket the learning platform
    cannot. An online platform sells tuition and nothing printed, so it cannot carry a line
    `language_courses` excludes; the exam institutes can.

    ⚠️ AND THE POOL HAS TO SPAN BOTH TRADES, which is a measurement rather than a taste. With the
    platforms alone no euro basket could draw `language_exam` — 6 000–9 000 UAH a line, which four
    of the six domestic sellers draw — and a euro claim came out systematically smaller than a
    hryvnia one in the same category. Currency was then a proxy for the seller's trade, and a
    consumer could predict `limit_exhausted` from a currency code. The numbers are in
    config/vendors.json beside the sellers."""
    sellers = load_vendors()["vendors"]["EU"][CATEGORY]
    assert len(sellers) >= 2, "one seller would teach a consumer the name rather than the field"

    for seller in sellers:
        assert vendor_can_carry(seller, CATEGORY, mixed=False), seller["name"]
    assert [s["name"] for s in sellers if vendor_can_carry(s, CATEGORY, mixed=True)], (
        "no euro seller of this category can carry a mixed basket, so `partially_covered` by "
        "`mixed_items` is unreachable in euros"
    )

    profiles = load_vendors()["vendor_profiles"]
    priciest = "language_exam"
    assert [s["name"] for s in sellers if priciest in profiles[s["profile"]]], (
        f"no euro seller sells {priciest}, the category's most expensive covered kind — the "
        "domestic pool does, and the currency would predict the size of a claim"
    )


def test_every_kind_this_seller_sells_has_a_euro_price_that_nests_in_its_hryvnia_sibling():
    """The arithmetic `claim_planner` leans on, asserted for the kinds this archetype prices.
    `estimated_line_value` reads the UAH ranges and runs before an archetype is chosen, so a euro
    line worth less than its UAH sibling's floor would make an exhausted-limit plan miss."""
    generation = load_generation()
    rate = Decimal(str(load_fx_rates()["rates"]["EUR"]))
    eur = generation["price_ranges_eur"]["ranges"]
    profiles = load_vendors()["vendor_profiles"]
    sellers = load_vendors()["vendors"]["EU"][CATEGORY]

    for kind in {k for s in sellers for k in profiles[s["profile"]]}:
        assert kind in eur, f"{kind} has no euro price, so this seller cannot print a line of it"
        low, high = (Decimal(value) for value in eur[kind])
        uah_low, uah_high = (
            Decimal(value) for value in generation["price_ranges"]["ranges"][kind]
        )
        assert low * rate >= uah_low and high * rate <= uah_high, kind


# ============================================================ what the page states ==


def test_the_page_asks_for_money_and_never_records_that_it_moved(rendered):
    """🔴 THE CLASS'S WHOLE CONTENT. 📄 An invoice is an offer to pay; the fact of payment is
    established by the document beside it. A «Paid» mark here would make the class prove both
    facts and contradict `document_evidence` in config/policy.yaml."""
    text = rendered.reference_text

    assert "INVOICE" in text
    assert "Amount due" in text
    for absent in ("Paid", "Amount paid", "Payment method", "Receipt"):
        assert absent not in text


def test_the_currency_code_travels_with_the_headline_figure(rendered):
    document = make()

    assert "currency" in rendered.field_bboxes
    assert "EUR" in rendered.reference_text
    assert f"{document.total:.2f}" in rendered.reference_text.replace(",", "")


def test_the_number_the_payment_will_cite_is_printed_and_labelled(rendered):
    """The link between the two pages of one claim is a printed string on both ends — the
    invoice's own number, which the label carries as `document_code` and the payment's purpose
    line quotes."""
    document = make()

    assert "document_code" in rendered.field_bboxes
    assert document.number in rendered.reference_text
    assert document.reference.number == document.number
    assert f"Payment reference: {document.number}" in rendered.reference_text


def test_the_account_that_makes_it_payable_is_the_claims_own(rendered):
    """⚠️ Bank details are ordinary commercial content and not an Article 226 particular. They are
    printed because an offer to pay that names no account is not payable — and because the account
    is the CLAIM's, so the transfer beside it settles this obligation rather than resembling one."""
    document = make()

    assert document.iban.startswith("DE")
    assert document.iban in rendered.reference_text
    assert document.bank_name in rendered.reference_text


def test_no_seller_identification_at_any_rate_of_drawing(rendered):
    """⛔ No VAT identification number and no tax code, WHATEVER THE DRAWN TAX FORM — the absences
    the config block argues for are absences of the class, not of one form of its totals block."""
    assert "vat_amount" not in rendered.field_bboxes
    assert "seller_vat_number" not in rendered.field_bboxes
    assert "seller_tax_code" not in rendered.field_bboxes


# ============================================ the tax treatment — three forms of one block ==


def test_all_three_tax_forms_are_reachable_and_each_ones_arithmetic_holds():
    """🔴 THE FORM THE CORPUS LACKED BY CONSTRUCTION, plus the two no-tax forms kept reachable.
    The mass form adds the destination tax ON TOP, so the printed total EXCEEDS the line items —
    the honest foreign page a validator holding `Σ lines = total` flags falsely. Which form a
    seed draws is `eu_tax_treatment` in config/generation.yaml; each form's own invariants are
    asserted per document, and all three must occur or the draw is broken."""
    assert set(eu_tax_treatment_shares()) == {"tax_on_top", "out_of_scope", "reverse_charge"}

    seen = set()
    for seed in range(120):
        document = make(seed)
        form = form_of(document)
        seen.add(form)
        if form == "tax_on_top":
            assert document.tax_amount > 0
            assert document.total == document.subtotal + document.tax_amount
            assert document.vat_note is None, (
                "the out-of-scope sentence beside a charged rate would contradict the row"
            )
        elif form == "reverse_charge":
            assert document.tax_amount == Decimal("0.00")
            assert document.total == document.subtotal
            assert "reverse charge" in document.tax_label
            assert document.vat_note == tax_on_top_rules()["reverse_charge_note"]
        else:
            assert document.tax_label is None and document.tax_amount is None
            assert document.total == document.subtotal
            assert document.vat_note == jurisdiction("EU")["invoice"]["vat_note"]
    assert seen == {"tax_on_top", "out_of_scope", "reverse_charge"}


def test_the_rate_is_the_buyer_countrys_parameter_and_follows_a_relocation():
    """The treatment is DERIVED from the buyer's country — the persona's own axis, not a new
    field. Ukraine's 20% is the one figure this repository cites from published law; the other
    entries are ⛔ project parameters for the form, asserted here only to be what the config
    declares, never to be anybody's law."""
    rates = tax_on_top_rules()["rate_by_buyer_country"]
    seed = seed_in_form("tax_on_top")
    document = make(seed)

    assert rates["UA"] == 20.0
    assert document.tax_label == "VAT (20%)"
    assert document.tax_amount == (document.subtotal * Decimal("20") / 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert document.buyer_country == "Ukraine"

    # Same seed, same draw stream, one changed axis: the same form comes out — the treatment
    # draw sits at the same position — with the other country's parameter and exonym.
    relocated = make(seed, buyer_country=Country.DE)
    assert form_of(relocated) == "tax_on_top"
    assert relocated.buyer_country == "Germany"
    rate = Decimal(str(rates["DE"]))
    assert relocated.tax_label == f"VAT ({float(rate):g}%)"
    assert relocated.tax_amount == (relocated.subtotal * rate / 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def test_the_tax_on_top_page_prints_the_row_and_no_out_of_scope_sentence(renderer, tmp_path):
    """What is printed is what is labelled: the row's figure carries its own box, the total is
    the sum, and the foot is ABSENT — no rule over an empty block, no sentence contradicting the
    rate above it."""
    document = make(seed_in_form("tax_on_top"))
    page = renderer.render(SLUG, document.render_context(), tmp_path / "on-top.png")
    text = page.reference_text.replace(",", "")

    assert "tax_amount" in page.field_bboxes
    assert "vat_note" not in page.field_bboxes
    assert document.tax_label in page.reference_text
    assert f"{document.tax_amount:.2f}" in text
    assert f"{document.subtotal:.2f}" in text
    assert f"{document.total:.2f}" in text
    assert "VAT not charged" not in page.reference_text


def test_the_out_of_scope_page_is_the_old_page_exactly(renderer, tmp_path):
    """📄 Articles 44 and 59 place this supply outside the scope of EU VAT, so Article 226's tax
    particulars do not apply: no row, and the note names what decides it — the whole class's only
    form before the tax-on-top states landed, still reachable at its configured share."""
    document = make(seed_in_form("out_of_scope"))
    page = renderer.render(SLUG, document.render_context(), tmp_path / "out-of-scope.png")
    text = page.reference_text

    assert "vat_note" in page.field_bboxes
    assert "VAT not charged" in text
    assert "2006/112/EC" in text
    assert "tax_amount" not in page.field_bboxes


def test_the_reverse_charge_page_prints_a_zero_row_and_says_who_accounts(renderer, tmp_path):
    """The third form: the row is there, the figure is 0.00, and the foot carries the sentence —
    a page asserting a mechanism rather than omitting a block, which is a different document from
    either of the other two."""
    document = make(seed_in_form("reverse_charge"))
    page = renderer.render(SLUG, document.render_context(), tmp_path / "reverse.png")

    assert "tax_amount" in page.field_bboxes
    assert "vat_note" in page.field_bboxes
    assert document.tax_label in page.reference_text
    assert "0.00" in page.reference_text
    assert tax_on_top_rules()["reverse_charge_note"] in page.reference_text
    assert f"{document.total:.2f}" in page.reference_text.replace(",", "")


def test_the_seller_carries_its_own_registers_designation_and_no_ukrainian_code(rendered):
    """«italki Inc.» and not «INC «italki»» — the Ukrainian quotation marks are a rule about a
    Ukrainian firm's name. The bare mark is what the LABEL carries, on every class."""
    document = make()

    assert document.seller_display == "italki Inc."
    assert document.seller_name == "italki"
    assert "italki Inc." in rendered.reference_text


def test_the_due_date_never_falls_before_the_payment_that_settles_it():
    """🔴 A TERM BOUND BY THE CLAIM'S OWN MONEY. An offer whose date has passed is not the
    obligation the payment discharged — the Ukrainian invoice's validity line follows the same
    rule, and the drawn window is a FLOOR on the span rather than the whole of it."""
    settled = datetime(2026, 9, 30, 12, 0)

    for seed in range(50):
        document = make(seed, settled_at=settled)
        assert document.due_at.date() >= settled.date()
        # ...and without a settlement the window is the drawn one, inside the configured range.
        assert make(seed).due_at > make(seed).issued_at


# ================================================== the instalment term, and the label ==


def test_the_instalment_term_is_printed_only_when_the_plan_named_one(rendered):
    """⛔ NEVER DRAWN HERE. It decides the marker `policy_engine` reads to tell `partially_paid`
    from `amount_mismatch`, so a builder that drew it would be choosing a claim's verdict."""
    assert make().instalment_amount is None
    assert "instalment_amount" not in rendered.field_bboxes

    schedule = next(iter(partial_payment_schedules()))
    document = make(schedule=schedule)
    parts = partial_payment_schedules()[schedule]

    assert document.instalment_amount == (document.total / parts).quantize(Decimal("0.01"))
    assert document.render_context()["instalment"]["caption"]


def test_the_class_can_state_an_instalment_term_at_all():
    """`claim_planner` selects the subject of a `partially_paid` claim by CLASS, so an invoice
    archetype whose page could not print the term would be planned and then print nothing."""
    assert ARCHETYPES[SLUG].doc_type in STATES_AN_INSTALMENT_TERM
    assert "instalment_caption_format" in jurisdiction("EU")["invoice"]


def test_the_label_is_the_ukrainian_invoices_record_in_another_currency():
    document = make()
    truth = document.ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.DIGITAL_PDF, field_bboxes={}
    )

    assert truth.doc_type is DocType.INVOICE
    assert truth.currency == "EUR"
    assert truth.language == "en"
    assert truth.document_code == document.number
    assert truth.counterparty == "italki"
    assert truth.payer == "Ковальчук Олена Петрівна"
    assert truth.instalment_amount is None
    assert truth.has_qr is False
    assert truth.qr_is_fiscal is False
    assert truth.has_fiscal_number is False
    assert all(line.vat_letter is None for line in truth.line_items)


def test_the_label_amount_is_the_printed_total_and_the_tax_is_labelled_apart():
    """🔴 `amount` IS WHAT THE PAGE ASKS FOR — the payment document beside it is told exactly this
    figure (`assembler._amount_the_payment_states`), so the pair stays one transaction whichever
    form the totals block drew. `tax` is labelled apart so `amount = Σ line items + tax` is a
    checkable statement rather than a broken invariant, on every form."""
    for form in ("tax_on_top", "out_of_scope", "reverse_charge"):
        document = make(seed_in_form(form))
        truth = document.ground_truth(
            doc_id="d1", source_file="d1.png", capture=Capture.DIGITAL_PDF, field_bboxes={}
        )
        assert truth.amount == document.total, form
        assert truth.tax == document.tax_amount, form
        assert truth.amount == line_items_total(document.line_items) + (
            truth.tax or Decimal(0)
        ), form


def test_the_line_names_are_english():
    """The names come from the `en` template lists policy.yaml carries per kind — the language is
    data, and a Cyrillic name here would mean the draw fell back to the Ukrainian list."""
    for seed in range(20):
        for line in make(seed).line_items:
            assert line.name.isascii(), line.name
