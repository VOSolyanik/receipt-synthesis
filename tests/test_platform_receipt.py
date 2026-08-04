"""The platform receipt — the corpus's first English document and its first euro one.

The class's whole point is a pairing nothing else in the corpus has: it establishes BOTH
facts of `document_evidence` while carrying no fiscal identity of any jurisdiction, in a
currency the limits are not stated in. So the assertions below fall into three groups —
what the page deliberately lacks (the Article 226 absences, checked on the markup), what
the label carries (EUR, English, the buyer, the three false fiscality flags), and the
arithmetic that ties the euro price ranges to the hryvnia ones the planner sizes baskets
by. That last group is the one a config edit can silently break, and its docstring says
exactly which mechanism leans on it.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal

import pytest

from receipt_synth.claim_planner import ARCHETYPES, Evidence, evidence_of
from receipt_synth.config import load_fx_rates, load_generation, load_policy, load_vendors
from receipt_synth.content_builder import (
    build_platform_receipt,
    draw_party_identity,
    line_items_total,
    vendor_can_carry,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, DocType

SLUG = "eu_platform_receipt"
CATEGORY = "professional_development"

COURSE_PLATFORM = {
    "name": "Coursera", "legal_form": "INC",
    "profile": "online_learning_platform", "vat_payer": False,
}
MARKETPLACE = {
    "name": "Amazon", "legal_form": "SARL",
    "profile": "general_retailer", "vat_payer": False,
}


def make(seed: int = 20260615, vendor: dict = COURSE_PLATFORM, **kwargs):
    rng = random.Random(seed)
    return build_platform_receipt(
        rng,
        category_id=CATEGORY,
        issued_at=datetime(2026, 6, 15, 17, 41, 9),
        vendor=vendor,
        identity=draw_party_identity(random.Random(seed + 1), vendor, "UA"),
        buyer_name="Ковальчук Олена Петрівна",
        buyer_tax_id="2345678901",
        **kwargs,
    )


@pytest.fixture(scope="module")
def renderer():
    """One browser session for the whole module — a second sync Playwright in the same
    thread trips over the first one's event loop, besides costing a Chromium launch."""
    with Renderer() as browser:
        yield browser


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    return renderer.render(
        SLUG, make().render_context(), tmp_path_factory.mktemp("platform") / "p.png"
    )


# ================================================ the archetype and the policy ==


def test_the_registration_says_what_the_document_proves():
    """The second class that proves both facts, and the first that does so without being
    fiscal. Registered with its type's default and no override, like every archetype."""
    archetype = ARCHETYPES[SLUG]

    assert archetype.doc_type is DocType.PLATFORM_RECEIPT
    assert evidence_of(archetype) == Evidence(True, True)
    assert load_policy()["document_evidence"]["platform_receipt"] == {
        "proves_subject": True,
        "proves_payment": True,
    }


def test_the_blast_radius_is_one_category_and_the_seller_pool_is_the_eu_one():
    """`_select_documents` prefers a self-contained document wherever one is registered,
    so every category this archetype carried would lose its invoice-plus-payment pair for
    single-document verdicts and drop out of the pair-realized ones. The registry entry
    confines it to the platform class's natural home; widening this tuple is a
    distribution decision, and this test is what makes it a deliberate one."""
    archetype = ARCHETYPES[SLUG]

    assert archetype.categories == (CATEGORY,)
    assert archetype.vendor_pool == "EU"
    assert archetype.language == "en"


def test_every_eur_range_nests_inside_its_uah_sibling_at_the_vendored_rate():
    """🔴 THE ARITHMETIC THE PLANNER LEANS ON, pinned so an edit to either side goes red.

    `claim_planner` sizes a limit-exhausting basket with `estimated_line_value`, which
    reads the UAH ranges — it runs before a vendor or an archetype is chosen, so it
    cannot know the claim will price in euros. The sizing stays conservative exactly as
    long as a euro line is worth at least its UAH sibling's lower bound at the vendored
    rate; the upper bound keeps the basket inside the plausibility band the category was
    tuned to. Stated as the comment above `price_ranges_eur` in config/generation.yaml;
    verified here from the three files rather than trusted to stay prose.
    """
    generation = load_generation()
    rate = Decimal(str(load_fx_rates()["rates"]["EUR"]))
    uah = generation["price_ranges"]
    eur_ranges = generation["price_ranges_eur"]["ranges"]
    assert eur_ranges, "no euro ranges, so this test would assert nothing"

    for kind, (eur_low, eur_high) in eur_ranges.items():
        uah_low, uah_high = uah["ranges"].get(kind, uah["default"])
        assert Decimal(eur_low) * rate >= Decimal(uah_low), (
            f"{kind}: {eur_low} EUR × {rate} falls below the UAH floor {uah_low} — "
            "a limit-exhausting basket sized on the UAH ranges could then fail to overrun"
        )
        assert Decimal(eur_high) * rate <= Decimal(uah_high), (
            f"{kind}: {eur_high} EUR × {rate} exceeds the UAH ceiling {uah_high}"
        )


def test_the_mixed_carrier_is_the_marketplace_and_not_the_course_platform():
    """A mixed basket needs a non-covered line, and the learning-platform profile sells
    nothing `professional_development` excludes — so `_pick_vendor` must be able to
    reject it and still find a seller. The marketplace profile is that seller; were both
    to fail this, every `mixed_items` claim of the category would crash a stage away
    from the vendor choice that caused it."""
    assert not vendor_can_carry(COURSE_PLATFORM, CATEGORY, mixed=True)
    assert vendor_can_carry(COURSE_PLATFORM, CATEGORY, mixed=False)
    assert vendor_can_carry(MARKETPLACE, CATEGORY, mixed=True)

    pool = load_vendors()["vendors"]["EU"][CATEGORY]
    assert any(vendor_can_carry(v, CATEGORY, mixed=True) for v in pool), (
        "no EU vendor can carry a mixed basket — every mixed_items claim of "
        f"{CATEGORY} would raise inside _pick_vendor"
    )


# ========================================================== the page ==


def test_the_page_carries_none_of_the_particulars_it_disclaims(rendered):
    """📄 Article 226 of Directive 2006/112/EC read backwards, the non-fiscal slip's own
    method: a document declaring itself not a VAT invoice may lack the supplier's
    address, both parties' VAT identification numbers, and the rate and amount of tax —
    and this one lacks exactly those. Checked on the collected fields and the page text,
    not on the content object."""
    fields = set(rendered.field_bboxes)
    assert "seller_address" not in fields
    assert "seller_tax_code" not in fields
    assert "seller_vat_number" not in fields
    assert "vat_amount" not in fields
    assert "VAT" not in rendered.reference_text.replace("VAT invoice", "")


def test_the_page_declares_itself_not_a_vat_invoice(rendered):
    """🔴 The negative marker is a STRING WITH A POSITION — the other half of RC-08's
    argument, where the Ukrainian slip's marker was an absence."""
    assert "This is not a VAT invoice." in rendered.reference_text
    assert "not_a_tax_invoice_note" in rendered.field_bboxes


def test_the_currency_code_is_printed_and_boxed_once(rendered):
    """The corpus prints a code beside a number on no other class — one currency had
    nothing to distinguish. Here a bare figure would be ambiguous on the page exactly as
    it is in a label, so the code travels with every amount and is boxed beside the paid
    figure."""
    assert "currency" in rendered.field_bboxes
    assert "EUR" in rendered.reference_text


def test_the_money_on_the_page_is_the_money_in_the_label(rendered):
    document = make()
    printed_total = f"{document.total:.2f}".replace(",", "")
    assert printed_total in rendered.reference_text.replace(",", "")
    assert document.total == line_items_total(document.line_items)


# ========================================================== the label ==


def test_the_label_is_english_and_euro_and_fiscal_like_nothing():
    document = make()
    truth = document.ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )

    assert truth.doc_type is DocType.PLATFORM_RECEIPT
    assert truth.language == "en"
    assert truth.currency == "EUR"
    assert truth.has_qr is False
    assert truth.qr_is_fiscal is False
    assert truth.has_fiscal_number is False
    assert truth.counterparty == "Coursera"
    assert truth.payer == "Ковальчук Олена Петрівна"
    assert all(line.vat_letter is None for line in truth.line_items)


def test_the_line_names_are_english():
    """The names come from the `en` template lists policy.yaml carries per kind — the
    language is data, and a Cyrillic name here would mean the draw fell back to the
    Ukrainian catalogue."""
    document = make()
    for line in document.line_items:
        assert not any("Ѐ" <= ch <= "ӿ" for ch in line.name), line.name


def test_a_mixed_basket_is_drawn_when_the_plan_asks_for_one():
    document = make(
        vendor=MARKETPLACE, covered_only=False, coverage_target=Decimal("0.6")
    )
    flags = {line.covered for line in document.line_items}
    assert flags == {True, False}


def test_two_runs_of_one_seed_build_the_same_receipt():
    assert make() == make()


# ================================================ the domestic variant ==

UA_PLATFORM = {
    "name": "Prometheus", "legal_form": "TOV",
    "profile": "online_learning_platform", "vat_payer": True,
}


def make_ua(seed: int = 20260615, vendor: dict = UA_PLATFORM, **kwargs):
    rng = random.Random(seed)
    return build_platform_receipt(
        rng,
        category_id=CATEGORY,
        issued_at=datetime(2026, 6, 15, 17, 41, 9),
        vendor=vendor,
        identity=draw_party_identity(random.Random(seed + 1), vendor, "UA"),
        buyer_name="Ковальчук Олена Петрівна",
        buyer_tax_id="2345678901",
        address="м. Київ, вул. Хрещатик, 22",
        jurisdiction_code="UA",
        **kwargs,
    )


def test_the_domestic_variant_is_registered_beside_its_twin():
    archetype = ARCHETYPES["ua_platform_receipt"]
    assert archetype.doc_type is DocType.PLATFORM_RECEIPT
    assert archetype.categories == (CATEGORY,)
    assert archetype.vendor_pool is None  # the claimant's own pool — a domestic seller
    assert archetype.language == "uk"


def test_the_domestic_page_prints_the_requisites_the_eu_page_disclaims():
    """The third axis of the pair is LAW: the Ukrainian seller is a domestic company, so
    its legal name, address, identification code, ПН and the contained-VAT row are
    ordinary — each the exact particular the English page's footer licenses ITSELF to
    omit. One shared body, so the comparison is controlled by construction."""
    document = make_ua()
    context = document.render_context()

    assert context["seller"]["name"] == "ТОВ «Prometheus»"
    assert context["seller"]["address"] is not None
    assert context["seller"]["tax_code"] is not None
    assert context["seller"]["vat_number"] is not None
    assert context["vat"] is not None
    assert context["vat"]["label"] == "У т.ч. ПДВ 20%"
    assert context["not_a_tax_invoice_note"] is None
    assert context["currency"] == "UAH"


def test_the_domestic_vat_row_states_the_tax_contained_in_the_gross():
    """📄 «У т.ч. ПДВ» is the tax CONTAINED in a gross price, never added on top —
    gross × 20 / 120, exactly as the invoice states it. Known answer, computed on
    paper from the document's own total."""
    document = make_ua()
    expected = (document.total * Decimal("20") / Decimal("120")).quantize(Decimal("0.01"))
    assert document.vat_amount == expected


def test_the_domestic_label_is_ukrainian_and_hryvnia():
    truth = make_ua().ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.doc_type is DocType.PLATFORM_RECEIPT
    assert truth.language == "uk"
    assert truth.currency == "UAH"
    assert truth.counterparty == "Prometheus"
    assert all(line.vat_letter is None for line in truth.line_items)


def test_the_two_variants_share_one_field_set_plus_the_lawful_extras(renderer, tmp_path):
    """One body, two variants — so the Ukrainian page's boxes are the English page's plus
    exactly the requisites its jurisdiction adds: the seller's address and codes, and the
    VAT row; less the note only the English page prints. Field-set equality is what makes
    the pair a controlled comparison rather than two templates that resemble each other."""
    eu = renderer.render("eu_platform_receipt", make().render_context(), tmp_path / "eu.png")
    ua = renderer.render("ua_platform_receipt", make_ua().render_context(), tmp_path / "ua.png")

    eu_fields = {name for name in eu.field_bboxes if not name.startswith("item_")}
    ua_fields = {name for name in ua.field_bboxes if not name.startswith("item_")}
    assert ua_fields - eu_fields == {
        "seller_address", "seller_tax_code", "seller_vat_number", "vat_amount"
    }
    assert eu_fields - ua_fields == {"not_a_tax_invoice_note"}


def test_the_same_seed_builds_the_same_domestic_receipt():
    assert make_ua() == make_ua()
