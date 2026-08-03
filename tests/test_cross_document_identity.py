"""Do the two documents of one claim agree — measured on the PAGE, not on the builder.

🔴 THE TRAP THIS MODULE EXISTS TO AVOID, stated first because it is the whole design. A test that
checked identifiers by reading `invoice.supplier.code` against `confirmation.payee.code` would be
comparing one object with itself: both fields are set from the same `PartyIdentity` instance, so
the assertion holds for every implementation that passes the instance along — including one that
passes it to a builder which then ignores it. That is the dominant defect class of this repository:
a check that cannot fail is indistinguishable from a passing check in every report.

So every assertion below reads `reference_text` — the printed characters of the rendered page,
recorded by the layout engine before rasterization — and pulls the value out with the same readers
`tools/cross_document_audit.py` uses to measure a corpus. That is the source the defect was
measured from: 587 invoice-plus-payment pairs of the delivered corpus, and the seller's tax code
and IBAN disagreed on 587 of them.

WHY THE READERS ARE SHARED WITH THE AUDIT TOOL, AND WHY THAT IS NOT CIRCULAR. A broken reader could
in principle make both the tool and this module wrong in the same way — so every row is asserted
against a KNOWN ANSWER as well as against the other document: the value read off each page must
equal the `PartyIdentity` the claim was built with, which is a value computed outside the reader.
A reader that returned a constant, or the wrong capture group, fails that immediately.

HOW THIS DIFFERS FROM `test_pipeline.test_no_claim_contradicts_itself_across_its_own_documents`,
which asks the same question one level up. That test compares LABEL fields across a generated run —
`counterparty`, `amount`, `payer`, the date order. None of the identifiers below is a label field,
so it could not see them, and did not: the divergence survived it for the whole life of the
archetype. Labels there, printed text here.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal

import pytest
from cross_document_audit import fields_of

from receipt_synth.content_builder import (
    build_bank_statement,
    build_invoice,
    build_payment_confirmation,
    draw_party_identity,
    resolve_vendor,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture

VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
SOLE_TRADER = {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": True}
CATEGORY = "vitamins_nutrition"
CLAIMANT = "Ковальчук Олена Петрівна"
CLAIMANT_CODE = "2345678901"
# 👁 An invoice is issued and then settled. The planner draws the lead; here it is pinned, because
# what these tests are about is what the two pages say and not when.
SUBJECT_AT = datetime(2026, 6, 3, 10, 15)
PAYMENT_AT = datetime(2026, 6, 11, 14, 33)

# `transfer` is the initiation mode that prints a purpose AND identifies the payer. The third mode,
# internet acquiring, prints neither — that absence is a row of the table in its own right and has
# its own test below, rather than being the mode every other test silently runs under.
IDENTIFYING_MODE = "transfer"

PAYMENT_CLASSES = ("payment_confirmation", "bank_statement")
# 🔴 FOUR SEEDS AND NOT ONE. See the `claims` fixture: a mutation survived the single-seed version
# of the seller rows because the bank name is drawn from a four-name list and coincided.
SEEDS = (20260603, 20260604, 20260605, 20260606)
PAYMENT_SLUGS = {
    "payment_confirmation": "ua_bank_payment_confirmation",
    "bank_statement": "ua_bank_statement",
}

# ------------------------------------------------------------ the rows of the table --
#
# One entry per row of docs/cross-document-fields.md that BOTH documents of a claim print, with the
# payment classes that print it. A row keyed on a field the audit reader does not know fails
# `test_every_field_the_reader_knows_is_accounted_for` below, and so does a field the reader gains
# without being placed here — the sweep is over the reader's vocabulary, not over today's list.
SELLER_ROWS = {
    "seller_name": PAYMENT_CLASSES,
    "seller_tax_code": PAYMENT_CLASSES,
    "seller_account": PAYMENT_CLASSES,
    "seller_bank_name": PAYMENT_CLASSES,
    # ⛔ A statement's counterparty block prints name, code, IBAN and bank and NO bank code, so
    # there is nothing on that class to compare. Narrower than the confirmation on purpose.
    "seller_bank_code": ("payment_confirmation",),
}
CLAIMANT_ROWS = {
    "payer_name": PAYMENT_CLASSES,
    "payer_tax_code": PAYMENT_CLASSES,
}
REFERENCE_ROWS = {"invoice_number": PAYMENT_CLASSES}

# What each seller row must equal on BOTH pages — the known answer, taken from the identity the
# claim was built with rather than from either document.
EXPECTED_FROM_IDENTITY = {
    "seller_tax_code": lambda identity: identity.tax_code,
    "seller_account": lambda identity: identity.account,
    "seller_bank_name": lambda identity: identity.bank_name,
    "seller_bank_code": lambda identity: identity.bank_code,
}


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


class Claim:
    """One claim's subject document and its payment document, built and rendered as a pair.

    The build order is the assembler's: one vendor, one identity, the invoice first because it
    fixes the amount, then the payment document told what it settles and what it cites. Anything
    else would be testing an arrangement no run produces.
    """

    def __init__(self, renderer, tmp_path, payment_class: str, seed: int, vendor: dict):
        rng = random.Random(seed)
        self.vendor = resolve_vendor(rng, dict(vendor), "UA")
        self.identity = draw_party_identity(rng, self.vendor, "UA")
        self.invoice = build_invoice(
            rng,
            category_id=CATEGORY,
            issued_at=SUBJECT_AT,
            vendor=self.vendor,
            identity=self.identity,
            buyer_name=CLAIMANT,
            buyer_tax_id=CLAIMANT_CODE,
            address="м. Київ, вул. Хрещатик, 22",
        )
        common = dict(
            issued_at=PAYMENT_AT,
            vendor=self.vendor,
            identity=self.identity,
            payer_name=CLAIMANT,
            payer_tax_id=CLAIMANT_CODE,
            amount=self.invoice.total,
            cites=self.invoice.reference,
        )
        if payment_class == "payment_confirmation":
            self.payment = build_payment_confirmation(
                rng, initiation=IDENTIFYING_MODE, **common
            )
        else:
            self.payment = build_bank_statement(rng, **common)

        self.subject_text = renderer.render(
            "ua_invoice", self.invoice.render_context(), tmp_path / f"{seed}-subject.png"
        ).reference_text
        self.payment_text = renderer.render(
            PAYMENT_SLUGS[payment_class],
            self.payment.render_context(),
            tmp_path / f"{seed}-payment.png",
        ).reference_text
        self.payment_class = payment_class

    def read(self) -> tuple[dict, dict]:
        """What each page says, read the way a consumer would have to read it."""
        subject = fields_of({"doc_type": "invoice", "reference_text": self.subject_text})
        payment = fields_of(
            {
                "doc_type": self.payment_class,
                "reference_text": self.payment_text,
                "relevant_transaction": self._relevant_transaction(),
            }
        )
        return subject, payment

    def _relevant_transaction(self) -> str | None:
        """Which row of a statement the label points at — `None` for a class with one operation."""
        if self.payment_class != "bank_statement":
            return None
        return self.payment.rows[self.payment.relevant_index].number


@pytest.fixture(scope="module")
def claims(renderer, tmp_path_factory):
    """Several claims per payment class, built once. Parametrized over the CLASSES rather than over
    the slugs: a second template for a class would not add a cross-document question.

    🔴 SEVERAL SEEDS AND NOT ONE, AND A MUTATION IS WHY. The first version pinned a single seed and
    a mutation that made a statement row draw its own bank name SURVIVED it: the bank comes from a
    four-name list, so a document drawing independently prints the claim's bank about a quarter of
    the time, and that seed was one of them. A field's CARDINALITY decides how many samples an
    equality assertion needs before it means anything — at four values, one sample is a coin that
    lands right too often — and the assertion looked identical either way. Four seeds put the
    survival probability of that same mutation under a percent, and the mutation now reddens.
    """
    out = tmp_path_factory.mktemp("cross-document")
    return {
        payment_class: [
            Claim(renderer, out, payment_class, seed=seed, vendor=VENDOR)
            for seed in SEEDS
        ]
        for payment_class in PAYMENT_CLASSES
    }


# ---------------------------------------------------- the seller is one party --


@pytest.mark.parametrize(
    ("field", "payment_class"),
    [(field, cls) for field, classes in SELLER_ROWS.items() for cls in classes],
)
def test_both_documents_of_a_claim_name_the_seller_by_the_same_requisite(
    field, payment_class, claims
):
    """One row of the seller block of docs/cross-document-fields.md, asserted on both pages.

    THE KNOWN ANSWER COMES FIRST. Each page's printed value is compared against the claim's
    `PartyIdentity` — a value neither document produced — and only then against the other page. An
    assertion of equality alone would also hold if the reader returned `None` from both, or the
    same wrong capture group from both; comparing against the identity is what makes it an
    absolute the defect cannot move.
    """
    read = 0
    for claim in claims[payment_class]:
        subject, payment = claim.read()
        if field in ("seller_bank_name", "seller_bank_code") and payment[field] is None:
            # 👁 A confirmation's payee bank is sometimes a CAPTION WITH NOTHING UNDER IT, which is
            # observed and deliberate. 🔴 BOTH FIELDS, NOT ONLY THE NAME: `payee.bank` is one
            # optional string, "name, Код банку code" or nothing at all, so the name and the code
            # are empty TOGETHER — a version of this guard naming only `seller_bank_name` skipped
            # the right claim for the wrong field and failed `seller_bank_code` outright the first
            # time a draw actually landed on the empty case for it. Every other requisite is
            # printed on every document of both classes, and a missing one there is a defect.
            continue
        read += 1

        assert subject[field] is not None, f"the invoice prints no {field}"
        assert payment[field] is not None, f"the {payment_class} prints no {field}"

        expected = EXPECTED_FROM_IDENTITY.get(field)
        if expected is not None:
            want = expected(claim.identity)
            assert subject[field] == want, f"invoice printed {subject[field]!r}, drew {want!r}"
            assert payment[field] == want, (
                f"{payment_class} printed {payment[field]!r}, drew {want!r}"
            )

        assert subject[field] == payment[field], (
            f"{field}: the invoice says {subject[field]!r} and the {payment_class} says "
            f"{payment[field]!r} — two documents of one claim, two sellers"
        )
    assert read, f"{field} was readable on no claim of {len(SEEDS)} — nothing was asserted"


def test_a_sole_traders_identifiers_agree_too_and_come_from_the_other_register(
    renderer, tmp_path
):
    """The register a code comes from follows the LEGAL FORM, and the claim-level draw must not have
    flattened that: a ФОП has no ЄДРПОУ, so an eight-digit code beside a sole trader's name would be
    an identifier no register could resolve to the party printed next to it.

    Run against a sole trader because the company case above cannot show it — both codes would be
    eight digits whether or not the rule survived.
    """
    claim = Claim(renderer, tmp_path, "payment_confirmation", seed=515, vendor=SOLE_TRADER)
    subject, payment = claim.read()

    assert len(claim.identity.tax_code) == 10, (
        f"a sole trader's code is a ten-digit РНОКПП, drew {claim.identity.tax_code!r}"
    )
    assert subject["seller_tax_code"] == claim.identity.tax_code
    assert payment["seller_tax_code"] == claim.identity.tax_code


def test_the_printed_bank_code_is_the_one_inside_the_printed_account(claims):
    """An IBAN carries its bank's МФО in the clear, so the bank code beside it is not an independent
    value. It is asserted on the PAGE rather than on the identity, because that is where the two
    would contradict each other: a document drawing its own bank while printing the claim's account
    is a defect no cross-document comparison catches and every reader of one page sees.
    """
    checked = 0
    for claim in claims["payment_confirmation"]:
        _, payment = claim.read()
        account, code = payment["seller_account"], payment["seller_bank_code"]
        if code is None:
            continue  # the payee bank block is empty on this one — 👁 observed, see above
        checked += 1
        assert account[4 : 4 + len(code)] == code, (
            f"the page prints bank code {code!r} beside account {account!r}, which carries "
            f"{account[4 : 4 + len(code)]!r}"
        )
    assert checked, "no confirmation printed a payee bank code — nothing was asserted"


def test_the_bank_identity_reader_agrees_with_what_the_document_was_built_with(claims):
    """`cross_document_audit.bank_identity_pairs` is the reader the new corpus-wide "one name, one
    code" axis (see that module) is built on — if IT misread the page, the axis could report a
    false disagreement across the whole corpus, or worse, a false agreement. Checked here against
    the KNOWN ANSWER, each document's own `bank_name`/`bank_code`, on the RENDERED page rather than
    the object the builder returned — the same discipline every other row of this module follows.

    On a bank statement this is also what proves the reader finds BOTH occurrences the new axis
    needs — the header AND the service-charge row — since `test_the_bank_charges_its_own_service
    _fee_on_every_statement` in test_bank_statement.py already covers their agreement at the
    builder level and this file exists to cover it on the page.
    """
    from cross_document_audit import bank_identity_pairs

    checked = 0
    for payment_class in PAYMENT_CLASSES:
        for claim in claims[payment_class]:
            pairs = bank_identity_pairs(payment_class, claim.payment_text)
            names = {name for name, _ in pairs}
            assert claim.payment.bank_name in names, (
                f"{payment_class}: no pair was read for the issuer {claim.payment.bank_name!r}, "
                f"read {pairs!r}"
            )
            if payment_class == "bank_statement":
                assert len(pairs) == 2, (
                    f"a statement names its issuer twice — the header and the service-charge "
                    f"row — read {pairs!r}"
                )
            for name, code in pairs:
                checked += 1
                if name == claim.payment.bank_name:
                    assert code == claim.payment.bank_code, (
                        f"{payment_class}: the page pairs {name!r} with {code!r}, the document's "
                        f"own bank_code is {claim.payment.bank_code!r}"
                    )
    assert checked, "no bank-identity pair was read off any rendered payment document"


# -------------------------------------------------- the claimant is one person --


@pytest.mark.parametrize(
    ("field", "payment_class"),
    [(field, cls) for field, classes in CLAIMANT_ROWS.items() for cls in classes],
)
def test_both_documents_of_a_claim_name_the_same_claimant(field, payment_class, claims):
    """The claimant's rows. These held before this change — both documents take the persona's own
    fields — and they are asserted anyway, because "it holds today" and "something checks it" are
    different states, and the seller's name held the same way until the day it did not.
    """
    for claim in claims[payment_class]:
        subject, payment = claim.read()
        assert subject[field] is not None and payment[field] is not None
        assert subject[field] == payment[field], (
            f"{field}: {subject[field]!r} / {payment[field]!r}"
        )


def test_an_unidentified_payer_is_an_absence_and_not_a_disagreement(renderer, tmp_path):
    """👁 On the internet-acquiring mode the confirmation does not identify the payer at all and
    prints a HYPHEN. The claimant then cannot be matched by name across the claim — which is a real
    property of the corpus, declared as KL-09, and not a defect for the row above to catch.

    It is asserted so that the row above stays honest: the parametrized test runs under the mode
    that DOES identify the payer, and without this test that choice would look like an oversight
    rather than the deliberate split of two cases.
    """
    claim = Claim(renderer, tmp_path, "payment_confirmation", seed=20260603, vendor=VENDOR)
    unidentified = build_payment_confirmation(
        random.Random(9),
        issued_at=PAYMENT_AT,
        vendor=claim.vendor,
        identity=claim.identity,
        payer_name=CLAIMANT,
        payer_tax_id=CLAIMANT_CODE,
        amount=claim.invoice.total,
        initiation="acquiring",
        cites=claim.invoice.reference,
    )
    text = renderer.render(
        PAYMENT_SLUGS["payment_confirmation"],
        unidentified.render_context(),
        tmp_path / "acquiring.png",
    ).reference_text
    payment = fields_of({"doc_type": "payment_confirmation", "reference_text": text})

    assert payment["payer_name"] != CLAIMANT
    assert CLAIMANT not in text, "this mode must not name the claimant anywhere on the page"
    # And the SELLER is still the claim's, which is what makes this an absence on one side rather
    # than a document about a different transaction.
    assert payment["seller_tax_code"] == claim.identity.tax_code


# ------------------------------------------- the payment cites the claim's invoice --


@pytest.mark.parametrize("payment_class", PAYMENT_CLASSES)
def test_where_the_payment_names_an_invoice_it_names_the_claims_own_invoice(
    payment_class, renderer, tmp_path
):
    """The reference row. Swept over seeds rather than pinned to one, because THE PURPOSE LINE IS
    DRAWN: some templates name no document at all and some name a ВН, so a single seed would test
    whichever case it happened to land on and go silently vacuous when the pool changed.

    Two things are asserted, and the second is what stops the first from being vacuous: every
    purpose that names a рахунок names THIS claim's invoice, and at least one seed produced such a
    purpose at all.
    """
    cited = 0
    for seed in range(12):
        claim = Claim(renderer, tmp_path, payment_class, seed=seed, vendor=VENDOR)
        _, payment = claim.read()
        if payment["invoice_number"] is None:
            continue
        cited += 1
        assert payment["invoice_number"] == claim.invoice.number, (
            f"seed {seed}: the {payment_class} cites invoice "
            f"{payment['invoice_number']!r}, the claim's invoice is {claim.invoice.number!r}"
        )
    assert cited, f"no {payment_class} in 12 seeds cited an invoice — this test asserted nothing"


def test_a_purpose_naming_a_delivery_note_does_not_carry_the_invoices_number(
    renderer, tmp_path
):
    """🔴 A ВН IS A DIFFERENT CLASS OF DOCUMENT, and no claim holds one. Both placeholders were
    `{invoice_no}` until the cross-document work, so filling the claim's invoice number here would
    have manufactured a reference resolving to the wrong class — and a linker matching on «№» alone
    would have scored a correct-looking hit on it.

    Swept for the same reason as the test above, and it asserts that the case OCCURS: a pool with
    no ВН template left would make this test pass while checking nothing.
    """
    seen = 0
    for seed in range(12):
        claim = Claim(renderer, tmp_path, "bank_statement", seed=seed, vendor=VENDOR)
        row = claim.payment.rows[claim.payment.relevant_index]
        if "ВН №" not in row.purpose:
            continue
        seen += 1
        assert claim.invoice.number not in row.purpose, (
            f"seed {seed}: a delivery note carries the invoice's own number — {row.purpose!r}"
        )
    assert seen, "no labelled row in 12 seeds named a ВН — this test asserted nothing"


def test_the_two_pages_write_the_invoices_date_in_different_scripts(renderer, tmp_path):
    """👁 THE INVOICE'S TITLE WRITES ITS DATE IN WORDS and a purpose line writes it in digits, so the
    reference cannot be resolved by string equality on the whole citation. That is what makes this
    row a resolvable one rather than another giveaway, and it is asserted rather than assumed
    because a later edit that printed digits in the title would remove the difficulty in silence.
    """
    for seed in range(12):
        claim = Claim(renderer, tmp_path, "payment_confirmation", seed=seed, vendor=VENDOR)
        digits = claim.invoice.issued_at.strftime("%d.%m.%Y")
        if digits not in claim.payment_text:
            continue
        title = re.search(r"Рахунок на оплату № \S+ від (.+?) р\.", claim.subject_text)
        assert title, "the invoice printed no title line"
        assert title.group(1) != digits, (
            f"the title writes the date as {title.group(1)!r}, the same form the payment uses"
        )
        assert str(claim.invoice.issued_at.year) in title.group(1)
        return
    pytest.fail("no confirmation in 12 seeds printed the invoice's date — nothing was asserted")


# ------------------------------------------- what must NOT agree, and completeness --


def test_only_the_labelled_row_of_a_statement_carries_the_claims_identity(claims):
    """A statement lists a dozen operations and the claim is ONE of them. If every row printed the
    claim's payee the page would not pose a problem — finding the right row is the task — so the
    identity must appear exactly once on it, and every other row must carry its own.
    """
    for claim in claims["bank_statement"]:
        text = claim.payment_text
        assert text.count(claim.identity.account) == 1, (
            f"the claim's IBAN appears {text.count(claim.identity.account)} times on the statement"
        )
        assert text.count(claim.identity.tax_code) == 1, (
            f"the claim's code appears {text.count(claim.identity.tax_code)} times on the statement"
        )
        # ⚠️ THE HOLDER MAY APPEAR TWICE AND NOBODY ELSE MAY. 👁 A credit can be a transfer from
        # ANOTHER ACCOUNT OF THE HOLDER'S, whose counterparty is the holder — so a page carrying
        # two of them prints one code twice, correctly. The first version of this line asserted
        # that every row's code was distinct; it passed on one seed and failed on the fourth, on a
        # statement with two «Переказ коштів між власними рахунками» rows. The property that is
        # actually true, and the one that matters for linking, is that a REPEAT means the holder.
        codes = [row.counterparty_code for row in claim.payment.rows]
        repeated = {code for code in codes if codes.count(code) > 1}
        assert repeated <= {CLAIMANT_CODE}, (
            f"{sorted(repeated)} appear on more than one row and are not the holder's own code — "
            "two rows naming one counterparty would make the labelled row ambiguous"
        )


@pytest.mark.parametrize("payment_class", PAYMENT_CLASSES)
def test_the_two_documents_agree_about_the_money(payment_class, claims):
    """The control row, and it is here to keep the module honest rather than to guard the amount —
    `policy_engine._cross_checks` compares the two exactly, and a run-level test already covers it.
    What it establishes for THIS module is that the pair really is one claim: a fixture that had
    quietly built two unrelated documents would fail here first.
    """
    for claim in claims[payment_class]:
        assert claim.invoice.total > Decimal(0)
        assert claim.payment.ground_truth(
            doc_id="c1_d2", source_file="c1_d2.png", capture=Capture.SCREENSHOT, field_bboxes={}
        ).amount == claim.invoice.total


def test_every_field_the_reader_knows_is_accounted_for():
    """🔴 THE SWEEP IS OVER THE READER'S VOCABULARY, NOT OVER TODAY'S ROWS. A field added to
    `cross_document_audit` — a new requisite that two classes print — would otherwise be measured
    by the tool and asserted by nothing, and the gap would be invisible: every test in this module
    would still pass.

    So the tool's field list and this module's tables are held equal. A new field must be placed in
    one of the tables above, which is a decision somebody makes rather than a default.
    """
    from cross_document_audit import FIELDS

    covered = set(SELLER_ROWS) | set(CLAIMANT_ROWS) | set(REFERENCE_ROWS)
    assert covered == set(FIELDS), (
        f"unasserted: {sorted(set(FIELDS) - covered)}; "
        f"not measured: {sorted(covered - set(FIELDS))}"
    )
