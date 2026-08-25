"""The bank payment confirmation: the second document class, and the first that proves one fact.

Every expectation here is computed from config/policy.yaml, config/fiscal-rules.yaml,
config/labelling-schema.yaml and the published law — never from what the builder happened to
produce. Where a published algorithm has a published example (ISO 13616 for the IBAN, Luhn for a
card number) the example is the oracle, because a checksum verified against its own implementation
verifies nothing.

What this module is mostly about is the three ways this class can be quietly wrong:

* the amount. Three amounts are printed, 📄 the norm names the smallest of the three as the amount
  of the operation, and 👁 the largest is the one in the biggest type. A generator that labelled
  the salient one would produce a corpus in which the wrong answer scores best.
* the date. Six concepts appear under ten captions on real documents, and three of them mean
  something other than the moment of debit. The contract says which are readable; these tests hold
  the generator to it, so the two files cannot drift apart.
* the fields that are not on it. RRN and the payment form were in an earlier field list, 👁 0 of 8
  documents carry either, and an absence is the one thing a rendering test cannot notice unless it
  is asked.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal

import pytest
import yaml

from receipt_synth.claim_planner import ARCHETYPES, evidence_of
from receipt_synth.config import (
    CONFIG_DIR,
    initiation_shares,
    jurisdiction,
    load_policy,
    payment_purposes,
)
from receipt_synth.content_builder import (
    build_payment_confirmation,
    draw_party_identity,
    generate_iban,
    iban_check_digits,
    is_valid_edrpou,
    is_valid_iban,
    is_valid_rnokpp,
    passes_luhn,
    printed_legal_name,
    unissuable_card_number,
    validate_amount_in_words,
)
from receipt_synth.renderer import TEMPLATES_DIR, Renderer
from receipt_synth.schemas import DocType

SLUG = "ua_bank_payment_confirmation"
BLOCK = jurisdiction("UA")["payment_confirmation"]

# A registered company and a sole trader. The payee's legal form decides which register its code
# comes from, and ⚠️ a sole trader's is ten digits — the length follows the kind of code and not
# the kind of party, which is the rule 👁 a sole-trader payee with a ten-digit code refuted the
# other reading of.
COMPANY = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
SOLE_TRADER = {"name": "Ковальчук О. С.", "legal_form": "FOP", "profile": "nutrition_practice",
               "vat_payer": False}

WHEN = datetime(2026, 4, 17, 11, 3, 9)


def make(seed: int = 20260417, vendor: dict = COMPANY, initiation: str | None = None,
         transfer: Decimal | None = None, **kwargs):
    return build_payment_confirmation(
        random.Random(seed),
        issued_at=WHEN,
        vendor=vendor,
        identity=draw_party_identity(random.Random(seed), vendor, "UA"),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        initiation=initiation,
        # Renamed from `transfer=` when the invoice landed: every payment-proving builder now takes
        # the claim's amount under the same keyword, so the assembler can hand it to whichever
        # archetype the plan chose. The dataclass field is still `transfer` — the distinction
        # between the amount of the operation and the total charged is the point of this class.
        amount=transfer,
        **kwargs,
    )


def test_the_label_carries_the_payees_bare_trade_name_and_the_page_prints_its_legal_form():
    """🔴 the label and the printed form are not the same string, and this class had them the same.

    config/labelling-schema.yaml makes the bare trading name authoritative under
    `normalization.party_name` — "this file makes the bare name authoritative" — because a legal
    form is a property of the seller's registration rather than of the merchant identity a claim is
    about. This class labelled the printed form, «ТОВ «Аптека АНЦ»», while a fiscal receipt of the
    same seller labelled «Аптека АНЦ».

    Nothing could see it. The contract's comparison strips the legal form, so both strings compare
    equal to any consumer; it took a cross-document test asking whether one claim agrees with itself
    to find two names for one merchant. Asserted here, on the class, so the fix has a guard of its
    own rather than depending on which document types a pipeline fixture happens to draw.
    """
    confirmation = make(vendor=COMPANY)
    record = truth(confirmation)

    assert record.counterparty == COMPANY["name"]
    assert confirmation.payee.name == printed_legal_name(
        COMPANY["name"], COMPANY["legal_form"]
    )
    assert record.counterparty != confirmation.payee.name, (
        "the label and the printed form coincide, so this test cannot tell them apart — pick a "
        "vendor whose legal form is printed"
    )


def many(count: int = 400, **kwargs):
    """A population of documents, one per seed. Distinct seeds, so each is an independent draw."""
    return [make(seed=seed, **kwargs) for seed in range(count)]


def truth(confirmation, doc_id: str = "p001_c1_d1"):
    return confirmation.ground_truth(
        doc_id=doc_id, source_file=f"{doc_id}.png", capture="screenshot", field_bboxes={}
    )


# ============================================================ the amounts ==


def test_the_labelled_amount_is_the_transfer_and_not_what_the_payer_paid():
    """📄 The National Bank's instruction on non-cash settlements makes the amount of the payment
    operation a requisite and does not name the fee at all, so the amount of this document is the
    transfer. A fee buys a banking service, which no benefit category covers.

    Asserted on a document whose fee is non-zero, because the two amounts coincide when it is
    zero and the assertion would then hold on a builder that labelled either one.
    """
    fee_bearing = next(c for c in many() if c.fee > 0)

    label = truth(fee_bearing)
    assert label.amount == fee_bearing.transfer
    assert label.total_charged == fee_bearing.transfer + fee_bearing.fee
    assert label.amount < label.total_charged, "the fee is non-zero, so the two must differ"
    assert label.fee == fee_bearing.fee


def test_the_total_is_derived_from_the_transfer_and_the_fee():
    """Three amounts, one of which is stored twice nowhere. `total_charged` is a property rather
    than a field so the three cannot drift; this exercises the arithmetic at a fee it was not
    drawn with, since every generated fee is a two-decimal value in a narrow band and the
    derivation would look right for the wrong reason at any of them."""
    confirmation = make()
    object.__setattr__(confirmation, "transfer", Decimal("1000.00"))
    object.__setattr__(confirmation, "fee", Decimal("12.34"))

    assert confirmation.total_charged == Decimal("1012.34")


def test_the_fee_is_non_zero_at_about_the_observed_rate():
    """👁 3 of 8 observed confirmations carry a non-zero fee — about 37%, which is not an edge
    case. Keeping it at zero was explicitly not allowed: a field that is always zero
    discriminates nothing, and it would hide the divergence between the norm's amount and the
    salient one, which is the whole reason this class is worth generating.

    The band is a sanity check on a draw, not a measurement, and it is wide on purpose: over 400
    documents at a declared rate of 0.37 the binomial standard deviation is about 2.4 points, so
    0.30 to 0.45 is roughly ±3 sd. What is load-bearing is the first assertion — that some fee is
    non-zero at all.
    """
    fees = [c.fee for c in many()]

    assert any(fee > 0 for fee in fees), "every fee is zero, so the field discriminates nothing"
    share = sum(1 for fee in fees if fee > 0) / len(fees)
    assert 0.30 <= share <= 0.45, f"non-zero fees are {share:.0%} of the corpus"


def test_the_amount_in_words_states_the_transfer():
    """👁 5 of 8 print the amount in words. It spells the transfer, which is 📄 the amount of the
    operation — the same requisite written twice rather than a second number."""
    spelled = next(c for c in many() if c.amount_in_words is not None and c.fee > 0)

    assert validate_amount_in_words(spelled.amount_in_words, spelled.transfer)
    assert not validate_amount_in_words(spelled.amount_in_words, spelled.total_charged)


def test_the_printed_total_is_the_largest_type_on_the_page(tmp_path):
    """🔴 the salience trap, asserted on the rendered page. 👁 On the one observed document that
    prints all three amounts the total is in the largest type — and 📄 it is not the amount of the
    operation. The divergence has to be visible, or an extractor that follows salience is being
    scored on a page where salience happens to agree with the norm.

    Measured as box height rather than as a font size in the stylesheet: what matters is what the
    image shows, and a rule can be overridden by another rule.
    """
    with_total = next(c for c in many() if c.prints_total and c.fee > 0)
    with Renderer() as renderer:
        result = renderer.render(SLUG, with_total.render_context(), tmp_path / "total.png")

    assert result.field_bboxes["total_charged"][3] > result.field_bboxes["amount"][3], (
        "the total is not printed larger than the operation amount, so the trap is not on the page"
    )


# ================================================== the date, and commit A ==


def contract_captions() -> tuple[list[str], list[str]]:
    """The captions the labelling contract accepts as the payment date, and those it refuses.

    Read from config/labelling-schema.yaml — the other side of the split that names the payment
    date. config/policy.yaml holds the concept and this file holds the mapping, and a generator
    printing a refused caption would be printing a document whose payment date the contract says
    cannot be read.
    """
    contract = yaml.safe_load((CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8"))
    block = contract["document_types"]["payment_confirmation"]["payment_date"]
    accept = [entry["caption"] for entry in block["accept_in_order"]]
    refuse = [entry["caption"] for entry in block["never"]] + [
        variant for entry in block["never"] for variant in entry["also_printed_as"]
    ]
    return accept, refuse


def test_every_printed_date_caption_is_one_the_contract_can_read():
    """the join between the two commits of this step, and neither file can hold it alone.

    The contract's preference order exists so two implementations do not read two different dates
    off one document. That is worth nothing if the generator prints a caption the contract refuses:
    the label would carry a date the contract says is not the payment date, and the dataset would
    disagree with its own specification while every file looked correct on its own.
    """
    accept, refuse = contract_captions()
    assert accept and refuse, "the contract names no captions — this test would assert nothing"

    printed = {c.date_caption for c in many()}
    assert printed, "no document printed a date caption"
    assert printed <= set(accept), (
        f"printed captions the contract cannot read: {sorted(printed - set(accept))}"
    )
    assert not printed & set(refuse), "a refused caption reached a document"


def test_the_configured_captions_are_exactly_the_ones_the_contract_accepts():
    """The other direction, and it catches the case the sweep above cannot: a generator printing
    only one accepted caption satisfies that test while two thirds of the contract's order is
    never exercised by any image. 👁 All three are observed on real documents."""
    accept, _ = contract_captions()

    assert BLOCK["date_captions"] == accept


def test_the_time_is_written_with_colons_and_not_the_receipt_dashes():
    """The jurisdiction's `time_format` separates with dashes, and that entry is 👁 an observation
    about ПРРО receipts — a till-printer quirk recorded under `verified_against_own_receipts`. A
    bank writes a time with colons. Inheriting the receipt's format put `11-03-09` on a bank
    document, which is the kind of defect that sits on every image of a class and no label
    reveals."""
    context = make().render_context()

    assert context["time"] == "11:03:09"
    assert jurisdiction("UA")["time_format"] == "%H-%M-%S", (
        "the receipt's dashed format has changed; this test's premise is that the two differ"
    )


# ============================ the conditional block: how it was initiated ==


@pytest.mark.parametrize("mode", sorted(BLOCK["initiation"]))
def test_each_initiation_mode_prints_exactly_the_fields_it_declares(mode):
    """one template with a conditional block, not two archetypes — asserted against the field sets
    config/fiscal-rules.yaml declares rather than against a list restated here, so the code and the
    configuration cannot drift.

    👁 3 of 8 real documents carry a masked card and an authorization code and a payment purpose at
    once, which is what refuted the earlier split into a quittance and a card slip. The axis that
    does vary is how the payment was initiated.
    """
    declared = BLOCK["initiation"][mode]
    confirmation = make(initiation=mode)

    assert (confirmation.auth_code is not None) is declared["prints_auth_code"]
    assert (confirmation.card_masked is not None) is declared["prints_card"]
    assert (confirmation.purpose is not None) is declared["prints_purpose"]
    if declared["identifies_payer"]:
        assert confirmation.payer.code is not None and confirmation.payer.account is not None
    else:
        # The first form of emptiness: a hyphen printed as the value. An extractor returns the
        # string, so the label carries it — telling it apart from a field that is simply absent is
        # the difference between two mechanisms a single "missing" label would merge.
        assert confirmation.payer.name == BLOCK["parties"]["empty_value"]
        assert confirmation.payer.code is None


def test_the_card_bearing_mode_carries_a_purpose_and_an_authorization_code_together():
    """The refutation stated as its own test, because it is the finding that removed a whole
    archetype from this step. If this ever fails, the two-archetype split is back."""
    confirmation = make(initiation="card")

    assert confirmation.card_masked and confirmation.auth_code and confirmation.purpose


def test_an_authorization_code_is_six_digits():
    """👁 4 of 4 card operations print six digits, which agrees with 📄 the published cashier's
    instruction. ⚠️ It is not a deduplication key at that width — six digits is a space of a
    million, unique only within an issuer and a window — and the document number is."""
    for confirmation in many(120, initiation="card"):
        assert re.fullmatch(r"[0-9]{6}", confirmation.auth_code)


def test_every_initiation_mode_is_drawn_when_none_is_named():
    """The shares in config/generation.yaml are 👁 observed frequencies, and a mode that is
    declared but never drawn is a field set no image carries."""
    drawn = {c.initiation for c in many()}

    assert drawn == set(initiation_shares()) == set(BLOCK["initiation"])


# ======================================= what is not on this document ======


def test_the_label_carries_no_line_items():
    """👁 8 of 8 confirmations carry no table of items, 📄 and a list of goods is not among the
    requisites. The empty list is the statement — this document lists nothing — which is why the
    field stays required rather than becoming nullable."""
    assert truth(make()).line_items == []


def test_neither_an_rrn_nor_a_payment_form_appears_anywhere_on_this_class():
    """👁 0 of 8 — both were in an earlier field list for this type and neither is on the document.
    An RRN is a requisite of the card operation as printed inside a fiscal receipt, where this
    repository does print it; a payment form (cash / card) is a receipt requisite too. Both were
    carried over by mistake, and an absence is the one thing a rendering test cannot notice unless
    it is asked for.

    Checked on the markup and on the label model together: a template that stopped printing them
    while the label still carried a field, or the reverse, would leave half the mistake in place.
    """
    markup = (TEMPLATES_DIR / f"{SLUG}.html").read_text(encoding="utf-8")
    body = re.sub(r"\{#.*?#\}", "", markup, flags=re.DOTALL)

    assert "rrn" not in body.lower(), "an RRN is not a requisite of a bank confirmation"
    assert "payment_method" not in body, "a payment form is a receipt requisite, not this one"

    label = truth(make())
    assert not hasattr(label, "rrn")
    assert not hasattr(label, "payment_method")


def test_a_qr_on_this_class_is_never_fiscal():
    """👁 2 of 8 carry a QR and both are marketing — an invitation to install an application. 👁 0
    of 8 carry a verification QR; a document is checked by its code as text. ✅ So this class is a
    stronger refutation of "a QR means the document is fiscal" than the fiscal case, where the QR
    at least addresses the tax service.

    Asserted over a document that has one, or the flag would be false for want of a QR.
    """
    with_qr = next(c for c in many() if c.qr_payload is not None)
    label = truth(with_qr)

    assert label.has_qr and not label.qr_is_fiscal
    assert not label.has_fiscal_number, "a bank confirmation carries no fiscal number"


def test_the_verification_footer_is_not_a_constant():
    """👁 3 of 8 — which directly refutes "every such document tells you how to check it". A
    generator that printed it always would make the field useless as a signal."""
    footers = [c.verification_footer for c in many()]

    assert any(footer is not None for footer in footers)
    assert any(footer is None for footer in footers)


# ================================================ identifiers and parties ==


def test_the_iban_check_digits_match_the_published_example():
    """📄 ISO 13616's own example — gb82 west 1234 5698 7654 32 — is the oracle. Verifying a
    checksum against this repository's own implementation of it verifies nothing, which is the
    lesson the ЄДРПОУ check digit cost a day to learn."""
    assert iban_check_digits("GB", "WEST12345698765432") == "82"
    assert is_valid_iban("GB82WEST12345698765432")
    assert not is_valid_iban("GB83WEST12345698765432")


def test_every_generated_iban_passes_its_own_checksum_and_carries_the_bank_code():
    """The length is 📄 fixed per country by the published registry and is read from
    config/fiscal-rules.yaml, so an account number of the wrong width fails here rather than being
    printed."""
    rules = jurisdiction("UA")["identifiers"]["iban_format"]
    rng = random.Random(4)

    for _ in range(200):
        bank_code = f"{rng.randrange(10 ** 6):06d}"
        iban = generate_iban(rng, bank_code)
        assert len(iban) == rules["length"]
        assert iban.startswith(f"{rules['country']}")
        assert iban[4:10] == bank_code
        assert is_valid_iban(iban)


def test_a_printed_card_number_could_not_be_an_issued_card():
    """🔴 publication, not realism. 👁 One observed document prints a recipient's card unmasked, so
    this generator must be able to print sixteen visible digits — and sixteen digits drawn freely
    satisfy the Luhn checksum one time in ten, at which point a published image carries a string
    that could be somebody's card. Failing a published checksum makes "this is not a card number" a
    property anyone can verify.

    The Luhn implementation is checked against a published test number first, or "nothing passes"
    would be satisfied by a checker that rejects everything.
    """
    assert passes_luhn("4111111111111111"), "the published test number should pass Luhn"
    assert not passes_luhn("4111111111111112")

    rng = random.Random(9)
    for _ in range(500):
        assert not passes_luhn(unissuable_card_number(rng))


def test_card_masking_varies_over_the_declared_schemes():
    """👁 six schemes over six documents, and ⛔ no norm was found for masking — the observed
    spread is what confirms the absence. A consumer whose pattern is "six digits, asterisks, four"
    matches a minority of the field, and a generator printing one scheme would teach the pattern.

    The unmasked scheme is asserted separately: it is the one a reader would call a bug.
    """
    printed = {c.card_masked for c in many(600, initiation="card")}
    schemes = BLOCK["card"]["masking_schemes"]

    assert len(printed) > 1
    assert any(re.fullmatch(r"[0-9]{16}|[0-9 ]{19}", card) for card in printed), (
        "no document printed an unmasked card, and 👁 one real document does"
    )
    assert any("·" in card for card in printed), "the dotted mask never appears"
    assert any(scheme["name_scheme"] for scheme in schemes if "name_scheme" in scheme)


def test_the_payees_code_comes_from_the_register_its_legal_form_belongs_to():
    """⚠️ the length follows the kind of code, not the kind of party. A company's ЄДРПОУ is eight
    digits and a sole trader's РНОКПП is ten — and a sole trader is a business, so a rule reading
    "ten digits, therefore a private individual" is false. 👁 A sole-trader payee with a ten-digit
    code was observed."""
    assert is_valid_edrpou(make(vendor=COMPANY).payee.code)
    assert is_valid_rnokpp(make(vendor=SOLE_TRADER).payee.code)


def test_a_captioned_field_can_be_empty_without_its_caption_disappearing(tmp_path):
    """👁 the second form of emptiness — a caption printed with nothing under it, observed on the
    recipient's bank among three such fields on one document. It is mechanically different from the
    hyphen: OCR returns nothing here and the string "-" there, so a label carrying only "field
    missing" would make accuracy on empty fields unmeasurable.

    The caption stays on the page and the value element goes, so no bounding box points at nothing.
    """
    empty = next(c for c in many() if c.payee.bank is None)
    with Renderer() as renderer:
        html = renderer.build_html(SLUG, empty.render_context())
        result = renderer.render(SLUG, empty.render_context(), tmp_path / "empty.png")

    assert BLOCK["parties"]["bank_label"] in html, "the caption vanished with its value"
    assert "payee_bank" not in result.field_bboxes, "an empty field left a box pointing at nothing"


# =========================================== the archetype and the policy ==


def test_the_type_proves_the_payment_and_not_the_subject():
    """confirmed, not changed. 👁 0 of 7 payment purposes name what was bought — they name an
    invoice, a delivery note or nothing but the movement of money — which is the strongest
    confirmation in this repository of the entry policy.yaml already carries. The dominant pair
    follows from it: an invoice proves the subject, this proves the payment."""
    entry = load_policy()["document_evidence"]["payment_confirmation"]

    assert entry == {"proves_subject": False, "proves_payment": True}


def test_the_archetype_is_registered_with_its_type_default_and_no_override():
    """No archetype-level evidence override exists, and this is the guard on that. An archetype
    whose evidence differed from its type's default could not be labelled correctly — the engine
    sees a document's type and not the archetype that produced it — so it must not be registered.
    """
    archetype = ARCHETYPES[SLUG]

    assert archetype.doc_type is DocType.PAYMENT_CONFIRMATION
    assert evidence_of(archetype) == (False, True)


def test_the_archetype_carries_every_category_the_policy_declares():
    """A document that lists nothing cannot contradict any category, so this archetype carries all
    of them — unlike a pharmacy receipt, which cannot print a gym membership. Checked against
    policy.yaml rather than against the literal tuple, so a category added to the policy cannot
    silently drop out of the registry."""
    declared = {entry["id"] for entry in load_policy()["categories"]}

    assert set(ARCHETYPES[SLUG].categories) == declared


def test_the_purpose_names_a_document_and_never_the_subject_of_the_expense():
    """👁 0 of 7. The purpose names an invoice by number, a generic category, or the movement of
    money itself. That is what `proves_subject: false` rests on, so it has to be true of what is
    printed and not only of what was observed elsewhere.

    Asserted as the absence of every printable line-item name of every category: if a purpose ever
    named one, this type would prove the subject on that document and the label would be wrong.
    """
    printable = {
        template.split("{")[0].strip().lower()
        for entry in load_policy()["categories"]
        for bucket in ("covered_items", "excluded_items", "ambiguous_items")
        for templates in entry.get(bucket, {}).values()
        for template in templates["uk"]
        if len(template.split("{")[0].strip()) > 6
    }
    assert printable, "no line-item names to check against"

    for confirmation in many():
        if confirmation.purpose is None:
            continue
        named = [name for name in printable if name in confirmation.purpose.lower()]
        assert not named, f"a payment purpose names what was bought: {named}"


def test_no_purpose_asserts_a_transfer_between_the_payers_own_accounts():
    """Every confirmation this generator builds pays a named firm — the payee block prints
    the vendor's name, code and IBAN — so a purpose saying «Переказ власних коштів», a
    transfer between the payer's own accounts, contradicts the page it is printed on.

    Measured before the repair, on the production corpus (RP-06): 54 of 261 confirmations
    printing a purpose carried exactly that formula — 55% of the "cites nothing" bucket —
    and `docs/cross-document-fields.md` blocked the `cites_subject_document` contract
    field on those 54, because the flag would have certified them as legitimately
    non-citing. The wording legitimately survives in one place, the bank statement's
    `credit_from_self` pool, whose counterparty is the holder.

    Pinned at the pool and on built documents: the pool is what the repair edited, the
    documents are what the defect was measured on.
    """
    for formula in payment_purposes("uk"):
        assert "власних коштів" not in formula.lower(), (
            f"the self-transfer purpose is back in the confirmation pool: {formula!r}"
        )
    for confirmation in many():
        if confirmation.purpose is None:
            continue
        assert "власних коштів" not in confirmation.purpose.lower(), (
            f"{confirmation.purpose!r} asserts a self-transfer on a page whose payee "
            f"block names {confirmation.payee.name!r}"
        )


# ------------------------------------------------- the citation, structured ----


def test_the_cited_number_in_the_label_is_the_one_the_purpose_prints():
    """`cites_document_no` is a structured copy of the page, never an extra fact: set exactly when
    the printed purpose names a рахунок, equal to the number it names, and a substring of the
    line a reader of the image finds it in."""
    from receipt_synth.content_builder import DocumentReference

    cited = DocumentReference(number="7411", issued_at=WHEN)
    seen_citing = seen_uncited = False
    for seed in range(40):
        page = make(seed, cites=cited)
        if page.cites_document_no is None:
            seen_uncited = True
            if page.purpose is not None:
                assert "7411" not in page.purpose, (
                    "the purpose names the рахунок and the label says it names none"
                )
            continue
        seen_citing = True
        assert page.cites_document_no == "7411"
        assert "7411" in page.purpose
    assert seen_citing and seen_uncited, (
        "the sweep no longer exercises both forms; widen the seed range"
    )


def test_must_cite_forces_the_citation_onto_the_page_at_every_seed():
    """The label-first knob of the `subject` axis: under `must_cite` no seed may draw an
    acquiring mode or a non-citing formula — a subject-mismatch claim whose page cites nothing
    would be the cause silently unrealized, which is the drift the knob exists to prevent."""
    from receipt_synth.content_builder import DocumentReference

    cited = DocumentReference(number="7411", issued_at=WHEN)
    for seed in range(30):
        page = make(seed, cites=cited, must_cite=True)
        assert page.purpose is not None, f"seed {seed} drew a mode with no purpose line"
        assert page.cites_document_no == "7411", f"seed {seed} drew a non-citing formula"
        assert "7411" in page.purpose


def test_must_cite_without_a_reference_is_refused():
    with pytest.raises(ValueError, match="must_cite"):
        make(1, must_cite=True)


def test_must_cite_against_a_named_purposeless_mode_is_refused():
    """The two knobs can contradict each other only if a caller names them both; the builder
    refuses rather than printing a citation the mode has no line for."""
    from receipt_synth.content_builder import DocumentReference

    with pytest.raises(ValueError, match="prints no purpose"):
        make(1, initiation="acquiring",
             cites=DocumentReference(number="7411", issued_at=WHEN), must_cite=True)


# ============================================================ determinism ==


def test_the_same_seed_builds_the_same_document():
    """What `--seed` promises, at the stage that introduces the most draws of any archetype so
    far — a mode, a bank, two accounts, a masking scheme, a signature path and eight independent
    presence draws."""
    first, second = make(seed=77), make(seed=77)

    assert first.render_context() == second.render_context()
    assert truth(first) == truth(second)


def test_the_signature_is_drawn_per_document():
    """A single fixed path would be a constant on every image of the class, and a constant is what
    a model learns instead of learning the field."""
    paths = {c.signature_path for c in many() if c.signature_path is not None}

    assert len(paths) > 1


def test_the_page_is_the_a4_sheet_the_configuration_declares(tmp_path):
    """📄 A4 is 210 × 297 mm (ISO 216), stated in `payment_confirmation.page` and rendered at
    96 dpi — the resolution a page is shown at on a screen, which is how such a document reaches a
    claimant. This is the counterpart of the receipts' width test, and it has to be a separate one:
    `receipt.widths_mm` lists the widths a thermal roll is sold in, and adding 210 to that list to
    make one test cover both classes would be a false statement about till rolls.

    Asserted on the rendered image and not only on the stylesheet, so a width that reached the page
    through a different rule than the one this reads still has to come out A4.
    """
    page = BLOCK["page"]
    expected_width = round(page["width_mm"] / 25.4 * 96)
    expected_height = round(page["height_mm"] / 25.4 * 96)

    with Renderer() as renderer:
        result = renderer.render(SLUG, make().render_context(), tmp_path / "a4.png")

    assert result.width == expected_width, f"the page is not {page['width_mm']} mm wide"
    # `min-height`, so a long payment purpose lengthens the sheet rather than being clipped. The
    # sheet is therefore at least A4 and never shorter, and a document that grew past one page is
    # a fact about the content rather than a defect in the paper.
    assert result.height >= expected_height


def test_a_stylesheet_and_a_template_exist_for_this_archetype():
    """Stated here as well as in the registry sweep, because this module names the slug in twenty
    assertions and a typo in it would make all of them silently test another document."""
    assert (TEMPLATES_DIR / f"{SLUG}.html").is_file()
    assert (TEMPLATES_DIR / f"{SLUG}.css").is_file()
