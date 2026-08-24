"""VAT-payer status: the ПН line, and a VAT block, only where the seller is registered.

Two requisites of a Ukrainian receipt follow from whether the seller is registered for ПДВ
(податок на додану вартість — value added tax), and both were wrong on every rendered document
until this suite existed:

* the ПН line — the VAT-payer number. A registered seller prints it in addition to its ІД
  identification code, not instead of it: a payer carries one line more, never a different one.
  The number is twelve digits for a legal entity and the same ten-digit РНОКПП as its ІД for a
  sole trader, so the length follows the type of person rather than the prefix.
* the VAT block. A non-payer's line ends at the amount: no ПДВ-літера (the per-line VAT rate
  code), no tax-summary row, and no "Без ПДВ" in their place.

What this suite once asserted and why it was false. Its first version had the two identifier
lines mutually exclusive, and a payer printing ПН *instead of* ІД with twelve digits whatever
the seller was. That came from a published table of the form, which lists them as rows 4 and 5
with alternative example values. Real ПРРО output refuted it in two ways at once: a registered
company prints both lines together, and a registered sole trader's ПН is its ten-digit РНОКПП.
The tests that encoded the old reading were rewritten rather than removed, and each says what it
now pins.

Expectations come from the law and from config/fiscal-rules.yaml, never from what the builder
happens to emit: prefixes and lengths are read from the configuration.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

import pytest

from receipt_synth.config import jurisdiction, load_vendors
from receipt_synth.content_builder import (
    _build_tax_lines,
    build_prro_receipt,
    draw_party_identity,
    is_valid_edrpou,
    is_valid_rnokpp,
    resolve_vendor,
    vendor_is_vat_payer,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, LineItem

ISSUED_AT = datetime(2026, 8, 3, 14, 22, 51)

UA = jurisdiction("UA")
IDENTIFIERS = UA["identifiers"]

PAYER = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}

# The company non-payer exists as a fixture and not as a vendor entry, and the name is a
# documentation placeholder — «Приклад» is Ukrainian for "example", the same convention as the
# example.com payloads in config/fiscal-rules.yaml. A stored trading name here would assert
# that the firm bearing it is not registered for ПДВ, which is a claim about a real company's
# tax affairs; a placeholder asserts nothing and nothing is rendered into a dataset from it.
# config/vendors.json deliberately configures no company non-payer — see
# `$note_vat_payer_departures` — while the Положення still prescribes the requisite, so the
# builder has to produce it and this is where that is checked.
NON_PAYER_COMPANY = {
    "name": "Приклад",
    "legal_form": "TOV",
    "profile": "supplement_shop",
    "vat_payer": False,
}

# Both sole traders, differing only in status — which is the point: the status is a property
# of the vendor, so a ФОП on the general system prints «ПН» and a ФОП on the simplified
# system prints «ІД», and nothing about the legal form decides it.
NON_PAYER_SOLE_TRADER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False},
    "UA",
)
PAYER_SOLE_TRADER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": True},
    "UA",
)

# The non-payer used wherever the tests need one and its legal form is beside the point. A
# drawn sole trader, because that is the only unregistered seller config/vendors.json
# configures, so the tests exercise the variety the dataset actually contains.
NON_PAYER = NON_PAYER_SOLE_TRADER


def build(seed: int, vendor: dict, **kwargs):
    return build_prro_receipt(
        random.Random(seed),
        category_id="vitamins_nutrition",
        issued_at=ISSUED_AT,
        vendor=vendor,
        identity=draw_party_identity(random.Random(seed), vendor, "UA"),
        # The channel is stated at every call site: the builder has no default, because one
        # equal to the only live value hid a wiring break until a mutation survived.
        capture=kwargs.pop("capture", Capture.SCREENSHOT),
        **kwargs,
    )


# ------------------------------------------------------ the identifier lines --


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ("vendor", "id_register", "pn_length"),
    [(PAYER, "edrpou", 12), (PAYER_SOLE_TRADER, "rnokpp", 10)],
    ids=["company", "sole_trader"],
)
def test_a_registered_seller_prints_pn_IN_ADDITION_to_its_id_code(
    vendor, id_register, pn_length, seed
):
    """rewritten. The old assertion — a payer prints «ПН» of twelve digits and nothing else —
    was false twice over: it dropped the ІД line that a real registered company prints beside
    it, and it gave a sole trader's ПН twelve digits when 👁 it is the ten-digit РНОКПП.

    What is pinned now: both lines present, each carrying the length its own register defines,
    with the prefixes read from the configuration rather than restated here.
    """
    seller = build(seed, vendor).seller

    assert seller.tax_code_label == IDENTIFIERS[id_register]["label"] == "ІД"
    assert len(seller.tax_code) == IDENTIFIERS[id_register]["length"]

    assert seller.vat_number is not None, "a registered seller prints a ПН line"
    assert seller.vat_number_label == IDENTIFIERS["vat_number"]["label"] == "ПН"
    assert len(seller.vat_number) == pn_length
    assert seller.vat_number.isascii() and seller.vat_number.isdigit()


@pytest.mark.parametrize("seed", range(5))
def test_a_companys_pn_begins_with_its_id_code(seed):
    """👁 one observation, encoded as a construction. On the company receipt seen, the ІД value
    is the first eight digits of the twelve-digit ПН, so the builder appends to the ЄДРПОУ
    instead of drawing an unrelated number — the two printed lines then agree by construction
    rather than by luck, and a consumer cross-checking them cannot fail on an honest document.

    Only the length is certain; the prefix relation rests on a single document and no source
    states it as a requirement.
    """
    seller = build(seed, PAYER).seller

    assert is_valid_edrpou(seller.tax_code)
    assert seller.vat_number.startswith(seller.tax_code)
    assert len(seller.vat_number) == IDENTIFIERS["vat_number"]["legal_entity_length"]


def test_a_registered_sole_trader_prints_one_number_under_both_prefixes():
    """👁 For a sole trader the relation is identity, not prefix: the ПН *is* the РНОКПП. So the
    two lines carry the same ten digits, and anything assuming they differ is wrong."""
    seller = build(3, PAYER_SOLE_TRADER).seller

    assert seller.vat_number == seller.tax_code
    assert is_valid_rnokpp(seller.vat_number)
    assert seller.vat_number_label != seller.tax_code_label, "same number, different prefixes"


def test_a_non_payer_prints_its_id_code_and_no_pn():
    """The half that is unchanged in substance: no registration, no ПН line. What changed is
    that the ІД line is no longer evidence of non-registration — every seller prints one.
    """
    for vendor in (NON_PAYER_COMPANY, NON_PAYER_SOLE_TRADER):
        seller = build(3, vendor).seller
        assert seller.vat_number is None, "an unregistered seller has no VAT-payer number"
        assert seller.tax_code_label == "ІД"
        assert seller.tax_code

    assert is_valid_edrpou(build(3, NON_PAYER_COMPANY).seller.tax_code)
    assert is_valid_rnokpp(build(3, NON_PAYER_SOLE_TRADER).seller.tax_code)


def test_the_id_code_does_not_reveal_the_status():
    """🔴 The leak the old model had by construction. While «ІД» was printed only by non-payers,
    the prefix alone settled the seller's tax status — and a consumer could read registration
    off it without reading the ПН line or the VAT block. Now both varieties print «ІД» with the
    same prefix and the same length, so only the presence of ПН says anything.
    """
    payer, non_payer = build(3, PAYER).seller, build(3, NON_PAYER_COMPANY).seller

    assert payer.tax_code_label == non_payer.tax_code_label
    assert len(payer.tax_code) == len(non_payer.tax_code)
    assert (payer.vat_number is None) is not (non_payer.vat_number is None)


def test_a_non_payer_company_prints_its_edrpou_under_id():
    """A code path the law prescribes and no vendor entry produces. 📄 The Положення gives the
    «ІД» prefix to a seller's identification code, which for a company is its ЄДРПОУ, so the
    builder must render it correctly — but config/vendors.json configures no company non-payer
    on purpose (⛔ unobserved, and a stored trading name would assert a real firm's tax status;
    see `$note_vat_payer_departures` there). Hence a placeholder fixture rather than a vendor.
    """
    seller = build(3, NON_PAYER_COMPANY).seller
    assert seller.tax_code_label == IDENTIFIERS["edrpou"]["label"] == "ІД"
    assert is_valid_edrpou(seller.tax_code)


def test_the_status_is_not_derived_from_the_legal_form():
    """rewritten assertion, same property. It used to compare the two sole traders' prefixes,
    which only distinguished them while the lines were exclusive. The property it exists for is
    unchanged: two sellers of one legal form, differing only in `vat_payer`, must differ on the
    document — so a ФОП on the general system stays representable."""
    assert PAYER_SOLE_TRADER["legal_form"] == NON_PAYER_SOLE_TRADER["legal_form"]
    assert build(3, PAYER_SOLE_TRADER).seller.vat_number is not None
    assert build(3, NON_PAYER_SOLE_TRADER).seller.vat_number is None


def test_a_seller_carries_both_identifier_fields_each_independently_optional():
    """rewritten. The old version asserted that `Seller` had no `vat_number` field, on the
    mutual-exclusivity reading — so the model could not express the receipt a registered company
    actually prints. Both fields exist now and both are optional: `vat_number` is None for an
    unregistered seller, and `tax_code` is None for a receipt that omits the ІД line, which no
    vendor produces today but which real receipts may do.
    """
    fields = {name: f.type for name, f in build(3, PAYER).seller.__dataclass_fields__.items()}

    assert {"tax_code", "tax_code_label", "vat_number", "vat_number_label"} <= set(fields)
    for optional in ("tax_code", "vat_number"):
        assert "None" in str(fields[optional]), f"{optional} must be able to be absent"


def test_a_vendor_entry_without_a_status_is_refused():
    """Not defaulted from the legal form. A default would make the status a property of the
    form again — silently, for every entry nobody thought about."""
    with pytest.raises(ValueError, match="vat_payer"):
        vendor_is_vat_payer({"name": "X", "legal_form": "TOV", "profile": "pharmacy"})


def test_every_ukrainian_vendor_entry_declares_its_status():
    """Denominator stated: every entry of every UA category, so a category added without the
    flag fails here rather than at build time."""
    categories = load_vendors()["vendors"]["UA"]
    entries = [(name, entry) for name, block in categories.items() for entry in block]
    assert len(categories) == 7

    # A floor, not a count. This used to be `len(entries) == 44`, which reddened on every
    # legitimate vendor addition — including a correctly flagged one — and a test that fails
    # on correct work teaches its reader to edit the number rather than to look.
    #
    # What the count caught, and it is exactly one thing: the `missing` check below is empty
    # both when every entry declares the flag and when the comprehension found no entries at
    # all, so it passes vacuously if the vendor file is ever restructured beneath it. A count
    # noticed that; nothing else here did. Requiring every category to contribute keeps that
    # guard and drops the brittleness — the emptiness is what mattered, never the 44.
    empty = [name for name, block in categories.items() if not block]
    assert not empty, f"these categories yielded no vendor entries: {empty}"
    assert len(entries) >= len(categories)

    missing = [name for name, entry in entries if "vat_payer" not in entry]
    assert not missing, f"{len(missing)} of {len(entries)} entries state no vat_payer: {missing}"
    assert all(isinstance(entry["vat_payer"], bool) for _, entry in entries)


def test_both_statuses_are_reachable_in_the_category_that_renders():
    """`vitamins_nutrition` is the only category with a registered archetype, so if its
    vendors were all of one status the other variety would be absent from every dataset —
    which is the defect this commit exists to end, in the opposite direction."""
    statuses = {v["vat_payer"] for v in load_vendors()["vendors"]["UA"]["vitamins_nutrition"]}
    assert statuses == {True, False}


# ----------------------------------------------------------- the VAT block --


@pytest.mark.parametrize("seed", range(10))
def test_a_non_payer_carries_no_letter_on_any_line_and_no_tax_summary(seed):
    """👁 The observed line ends with the amount and nothing follows it; then the totals,
    with no ПДВ row anywhere. Neither the zero-rate letter «Г» nor "Без ПДВ" appears in
    place of the letter — both are permitted in writing and observed on no open sample."""
    receipt = build(seed, NON_PAYER)

    assert all(item.vat_letter is None for item in receipt.line_items)
    assert receipt.tax_lines == []
    assert receipt.render_context()["tax_lines"] == []


@pytest.mark.parametrize("seed", range(10))
def test_a_payer_carries_a_letter_on_every_line_and_a_tax_block(seed):
    receipt = build(seed, PAYER)

    assert all(item.vat_letter in UA["vat_letters"] for item in receipt.line_items)
    assert receipt.tax_lines
    assert sum(line.gross for line in receipt.tax_lines) == receipt.total


def test_a_null_letter_on_a_payers_receipt_is_still_a_builder_bug():
    """The guard is narrowed, not removed. A payer's line with no letter would silently
    leave turnover out of the tax block, which is the case the raise was written for."""
    line = LineItem(
        name="Вітамін D3", item_kind="vitamin_complex", qty=Decimal(1),
        price=Decimal("250.00"), covered=True, vat_letter=None,
    )
    with pytest.raises(ValueError, match="ПДВ-літера"):
        _build_tax_lines([line], vat_payer=True)


def test_a_letter_on_a_non_payers_receipt_is_a_builder_bug_too():
    """The other direction of the same guard: a seller with no VAT registration cannot have
    assigned a rate group to a line, so a letter there is the builder contradicting the
    document's own seller block."""
    line = LineItem(
        name="Вітамін D3", item_kind="vitamin_complex", qty=Decimal(1),
        price=Decimal("250.00"), covered=True, vat_letter="А",
    )
    with pytest.raises(ValueError, match="ПДВ-літера"):
        _build_tax_lines([line], vat_payer=False)


def test_the_zero_rate_letter_exists_but_is_not_the_non_payers_answer():
    """«Г» is the zero-rate group of a seller that is registered. Using it for a non-payer
    would print a "ПДВ Г=0,00%" row that no observed receipt carries.

    Second assertion replaced, the old one was near-vacuous. It read
    `"Г" not in {item.vat_letter for item in build(4, NON_PAYER).line_items}` — but for a
    non-payer every letter is already None, asserted more strongly one test above, so that
    check would have passed just as well had the builder emitted an arbitrary wrong letter.
    It could only ever fail on the single value it named.

    The mechanism by which «Г» could actually reach a line is `item_vat_letter`: the builder
    draws each line's letter from that mapping, so a kind pointed at «Г» is the whole of the
    failure mode. Asserting on the mapping reddens for any kind, which is what the name of
    this test claims to cover.
    """
    assert UA["vat_letters"]["Г"]["rate"] == 0.0

    letters_per_kind = UA["item_vat_letter"].values()
    reachable = {
        letter
        for value in letters_per_kind
        for letter in ([value] if isinstance(value, str) else value)
    }
    assert reachable, "no kind maps to any letter — the mapping was read at the wrong level"
    assert "Г" not in reachable, (
        f"a zero-rate letter is drawable by a line item: {sorted(reachable)}"
    )


# ---------------------------------------------------------- the total block --


@pytest.mark.parametrize("vendor", [PAYER, NON_PAYER], ids=["payer", "non_payer"])
@pytest.mark.parametrize("seed", range(5))
def test_amount_due_equals_the_total_while_discount_and_rounding_are_zero(seed, vendor):
    """`СУМА` and `ДО СПЛАТИ` are separate lines of the form and genuinely differ by
    `ЗНИЖКА` and `ЗАОКРУГЛЕННЯ`. Both are zero this version, so the two amounts coincide —
    and the contract states that the field therefore discriminates nothing yet."""
    receipt = build(seed, vendor)

    assert receipt.discount == Decimal(0)
    assert receipt.rounding == Decimal(0)
    assert receipt.amount_due == receipt.total


def test_amount_due_is_the_total_less_the_discount_plus_the_rounding():
    """🔴 the arithmetic and its signs, pinned while the inputs are still free.

    Every receipt the generator builds fixes `discount` and `rounding` at zero, so the
    derivation is invisible to every other test in this suite: replacing the whole expression
    with `return self.total` leaves them all green, and the sign of `rounding` — added, not
    subtracted — is unobservable too. `PrroReceipt.amount_due` claims to be derived so that the
    two amounts cannot drift apart, and a claim of protection nothing exercises is an untested
    assertion.

    So this test states the arithmetic directly, on a receipt whose two adjustments are set by
    hand. It needs no policy decision: what the policy has to settle is how a discount is
    distributed across covered and non-covered lines (RC-16), not what subtracting one does.

    Computed by hand, not from the code: 1 000.00 − 5.00 + 0.03 = 995.03.
    """
    receipt = replace(
        build(3, PAYER), total=Decimal("1000.00"),
        discount=Decimal("5.00"), rounding=Decimal("0.03"),
    )

    assert receipt.amount_due == Decimal("995.03")
    assert receipt.amount_due < receipt.total, "a discount lowers what is payable"
    assert replace(receipt, rounding=Decimal("-0.03")).amount_due == Decimal("994.97"), (
        "rounding is ADDED to the discounted total, so a negative rounding lowers it further"
    )
    # Quantized to the kopiyka, like every other amount that reaches a label:
    # 1 000.00 − 5.00 + 0.036 = 995.036 -> 995.04. Deliberately not a half-kopiyka tie — no
    # receipt can produce one, and pinning it here would assert a tie-breaking rule that
    # nothing in the generator relies on.
    assert replace(receipt, rounding=Decimal("0.036")).amount_due == Decimal("995.04")


def test_the_payment_row_states_what_is_payable_and_not_the_basket():
    """🔴 what is tendered is «ДО СПЛАТИ», and the two receipt templates disagreed about it. The
    fiscal body printed `total` on its «ГОТІВКА»/«КАРТКА» row while the товарний чек printed
    `amount_due` from the same basket — a difference no render could show and no test could see,
    because both adjustments are fixed at zero and the two amounts coincide on every document this
    generator builds. That is exactly the defect a discount draw would ship: a receipt whose payment
    row states an amount that does not settle its own last line.

    Read off the rendered page and on a receipt whose adjustments are set by hand — the arithmetic
    test above is set the same way, and for the same reason: an assertion over two equal numbers
    would be the tautology this file already had one of.
    """
    receipt = replace(
        build(7, PAYER), total=Decimal("1000.00"),
        discount=Decimal("5.00"), rounding=Decimal("0.03"),
    )
    with Renderer() as renderer:
        html = renderer.build_html("ua_prro_receipt", receipt.render_context())

    payable = f"995{receipt.decimal_separator}03"
    basket = f"1\u00a0000{receipt.decimal_separator}00"
    printed = re.search(r'data-field="paid_amount">([^<]*)<', html)
    assert printed is not None, "the payment row carries no box named `paid_amount`"
    assert printed.group(1).strip() == payable, (
        f"the payment row prints {printed.group(1)!r}; {payable} is payable and {basket} is only "
        "the basket"
    )


def test_the_total_block_carries_four_separate_values():
    context = build(6, PAYER).render_context()
    assert {"total", "discount", "rounding", "amount_due"} <= set(context)


def test_amount_due_reaches_the_label():
    receipt = build(6, PAYER)
    truth = receipt.ground_truth(
        doc_id="p001_c1_d1", source_file="x.png", capture="screenshot", field_bboxes={}
    )
    assert truth.amount_due == receipt.amount_due == truth.amount
