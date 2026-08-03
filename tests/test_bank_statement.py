"""The bank statement: one label for one row of a many-row document.

🔴 THE PROPERTY THIS FILE EXISTS FOR is that the label describes ONE TRANSACTION and says WHICH
one, while the document's own four turnover totals are printed and never labelled. Everything a
consumer scores comes off one row, so two things have to be true and neither is obvious from
reading the template: the labelled values are that row's, and the labelled BOXES are that row's
cells. The second is checked by geometry — box containment in the row's own rect — because "the
boxes point at the right row" is exactly the kind of claim that reads as verified and is not.

Every expected value here is derived from config/fiscal-rules.yaml, config/generation.yaml and the
arithmetic of the rows. Nothing was copied out of a run.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal
from string import Formatter

import pytest
import yaml

from receipt_synth.config import (
    CONFIG_DIR,
    bank_statement_count_range,
    bank_statement_share,
    jurisdiction,
    statement_purposes,
)
from receipt_synth.content_builder import (
    BankStatement,
    StatementRow,
    build_bank_statement,
    draw_party_identity,
    generate_rnokpp,
    is_valid_edrpou,
    is_valid_iban,
    is_valid_rnokpp,
    printed_legal_name,
    proves_payment_by_direction,
    resolve_vendor,
)
from receipt_synth.renderer import TEMPLATES_DIR, Renderer
from receipt_synth.schemas import Capture, Direction, DocType

SLUG = "ua_bank_statement"
BLOCK = jurisdiction("UA")["bank_statement"]

# 96 dpi, the resolution a PDF page is shown at on screen — the same scale the payment
# confirmation is rendered at, and the reason the two classes share it is that both reach a
# claimant as a screen rendering of a page rather than as a printed strip.
DPI = 96
MM_PER_INCH = Decimal("25.4")

def _px(key: str) -> int:
    """A declared paper dimension in millimetres, as the rendered pixel count.

    Rounded rather than truncated, which is what the browser does with a fractional CSS pixel:
    297 mm is 1122.5 px and the page is 1123 px wide. Truncating gave 1122 and the first version of
    the page-size test failed on its own arithmetic rather than on the render.
    """
    return round(Decimal(BLOCK["page"][key]) * DPI / MM_PER_INCH)


PAYER = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
HOLDER_NAME = "Ковальчук Олена Петрівна"
# A CHECKSUM-CORRECT РНОКПП, generated rather than typed. It is not decoration: on an own-account
# transfer the holder is the row's counterparty, so this value is swept by the checksum test below,
# and a hand-typed ten-digit string failed it — the fixture's defect, not the generator's.
HOLDER_CODE = generate_rnokpp(random.Random(3))
WHEN = datetime(2026, 5, 12, 14, 33)


def make_statement(seed: int = 20260512, vendor: dict = PAYER, amount: str | None = None):
    """One statement. `amount` pins the labelled transaction where a test needs to know it."""
    rng = random.Random(seed)
    resolved = resolve_vendor(rng, vendor, "UA")
    return build_bank_statement(
        rng,
        issued_at=WHEN,
        vendor=resolved,
        identity=draw_party_identity(rng, resolved, "UA"),
        payer_name=HOLDER_NAME,
        payer_tax_id=HOLDER_CODE,
        amount=Decimal(amount) if amount else None,
    )


def _common(rows: tuple[StatementRow, ...]) -> dict:
    """The non-row fields of a hand-built statement, for the two guard tests below.

    Hand-built because both guards refuse a statement `build_bank_statement` cannot produce, which
    is the point of them: they exist for what does not go through that builder — a trap archetype,
    or a consumer's own record.
    """
    return {
        "bank_name": "Б",
        "bank_code": "300001",
        "holder_name": HOLDER_NAME,
        "holder_code": HOLDER_CODE,
        "account": "UA123",
        "period_start": rows[0].at.date(),
        "period_end": rows[-1].at.date(),
        "issued_at": datetime(2026, 5, 20, 9, 0),
        "opening_balance": Decimal("500.00"),
        "rows": rows,
        "payee_trade_name": "Т",
        "decimal_separator": ",",
    }


def label(statement: BankStatement, boxes: dict | None = None):
    return statement.ground_truth(
        doc_id="c1_d1",
        source_file="c1_d1.png",
        capture=Capture.SCREENSHOT,
        field_bboxes=boxes or {},
    )


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    """One statement rendered once, with the object it was built from beside it.

    Both are needed together: every geometric assertion below compares a box against the row the
    builder decided was the relevant one, and a fixture returning only the render would leave the
    tests matching boxes against text.
    """
    statement = make_statement()
    output = tmp_path_factory.mktemp("statement") / "statement.png"
    return statement, renderer.render(SLUG, statement.render_context(), output)


# -------------------------------------------- the label is one transaction --


def test_the_label_carries_the_relevant_row_and_not_the_document():
    """🔴 The whole shape of the class. `amount`, `date`, `counterparty`, `payment_purpose` and
    `direction` are the values of ONE row, and `relevant_transaction` names it.

    The document's own summary is not in the record at all — checked in the test below rather than
    here, because "these five fields are the row's" and "the totals are nowhere" are two claims.
    """
    statement = make_statement()
    row = statement.relevant
    record = label(statement)

    assert record.doc_type is DocType.BANK_STATEMENT
    assert record.amount == row.amount
    assert record.date == row.at.date()
    # The bare trade name; the row prints the name with its legal form. See the test below.
    assert record.counterparty == statement.payee_trade_name
    assert record.payment_purpose == row.purpose
    assert record.direction is row.direction
    assert record.relevant_transaction == row.number


def test_the_four_turnover_totals_reach_no_label_field():
    """🔴 PRINTED AND NEVER LABELLED. The four most prominent numbers on the page are the opening
    and closing balances and the two turnovers, and no field of the record carries any of them.

    Stated as "no money field equals any of the four" rather than as "these fields are None",
    because the failure to guard against is a total arriving under a name that already exists —
    `total_charged` is the confirmation's «Загальна сума», one payment plus its fee, and reusing it
    for a period's turnover would give one key two meanings.
    """
    statement = make_statement()
    record = label(statement)

    totals = {
        statement.opening_balance,
        statement.closing_balance,
        statement.total_credit,
        statement.total_debit,
    }
    assert record.total_charged is None
    assert record.fee is None
    assert record.amount_due is None
    assert record.amount not in totals


def test_the_label_states_no_line_items_and_no_fiscal_marker():
    """A statement lists transactions, not the lines of a basket — which is why it establishes
    nothing about what was bought. 👁 No statement carried a QR or a fiscal number."""
    record = label(make_statement())

    assert record.line_items == []
    assert record.has_qr is False
    assert record.qr_is_fiscal is False
    assert record.has_fiscal_number is False


def test_the_holder_is_the_claimant_and_the_counterparty_is_the_payee():
    """⚠️ THE ACCOUNT IS THE CLAIMANT'S. A statement of anybody else's account evidences nothing
    about this claimant's money, which is why the header names the persona and not a company —
    even though the observed document was a company's."""
    statement = make_statement()
    record = label(statement)

    assert record.payer == HOLDER_NAME
    assert statement.holder_code == HOLDER_CODE
    # 🔴 THE BARE TRADE NAME, NOT THE PRINTED ONE. config/labelling-schema.yaml makes the bare name
    # authoritative; the ROW prints «ТОВ «Аптека АНЦ»» and the LABEL carries «Аптека АНЦ». This test
    # asserted the printed form until commit B, when a cross-document check found the confirmation
    # and the statement labelling one thing and a receipt of the same seller another.
    assert record.counterparty == PAYER["name"]
    assert statement.relevant.counterparty_name == printed_legal_name(
        PAYER["name"], PAYER["legal_form"]
    )


# --------------------------------------- which printed date the label's `date` is --
#
# 🔴 THE ORACLE IS SILENT ABOUT THIS, which is why it is tested here and specified in the contract.
# A date is a date: label the wrong one and every invariant in this repository still holds, every
# verdict still computes, and the defect surfaces at the far end as low field accuracy on `date` in
# a downstream scorecard — where it reads as an extraction error rather than as a specification gap.
# `document_types.bank_statement.payment_date` in config/labelling-schema.yaml is the specification;
# these three tests are what stop the page and the specification from drifting apart.


def contract_statement_payment_date() -> dict:
    """The statement's date mapping, resolved by path out of the labelling contract."""
    contract = yaml.safe_load(
        (CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8")
    )
    return contract["document_types"]["bank_statement"]["payment_date"]


def test_the_caption_the_contract_accepts_is_the_caption_the_page_prints():
    """The mapping is worth nothing unless it names a caption the document actually carries.

    TWO FILES, TWO SIDES, and that is what makes this a real check rather than a file compared
    with itself: config/fiscal-rules.yaml states what the template PRINTS and
    config/labelling-schema.yaml states what a consumer may READ AS THE PAYMENT DATE. Neither is
    derived from the other, so
    either one moving alone is the defect — the same property `tests/test_payment_date_split.py`
    holds between the policy and the contract for the confirmation.

    Asserted in both directions: exactly one caption is accepted, because 👁 the row carries exactly
    one date, and it is the column caption. A second accepted caption added without a second column
    would pass a subset check and is caught by the count.
    """
    accepted = [entry["caption"] for entry in contract_statement_payment_date()["accept"]]

    assert accepted == [BLOCK["columns"]["datetime"]], (
        "the contract accepts a caption the statement template does not print, or the template "
        f"prints one the contract does not accept: contract {accepted}, printed "
        f"{BLOCK['columns']['datetime']!r}"
    )


def test_the_contract_refuses_the_three_document_level_dates_the_page_also_prints():
    """The work of the mapping is on the refusal side, because the acceptance side has no contest.

    Each refusal must state its own mechanism and must not identify itself by naming another — a
    refusal written as "not the one above" moves when its sibling is edited, which is the defect
    `verdict_notes.definitions_name_their_own_slot` records for verdicts and which applies here for
    the same structural reason.
    """
    refused = contract_statement_payment_date()["never"]

    assert len(refused) == 3, "a date the page prints and the contract does not mention is unruled"
    captions = {entry["caption"] for entry in refused}
    for entry in refused:
        reason = entry["why"].strip()
        assert reason, f"{entry['caption']} is refused without a reason"
        borrowed = sorted(other for other in captions - {entry["caption"]} if other in reason)
        assert not borrowed, (
            f"the refusal of {entry['caption']!r} works by naming {borrowed}; each must be "
            "disqualified by its own meaning or an edit to one silently moves the other"
        )


@pytest.mark.parametrize("seed", range(12))
def test_the_labelled_date_is_the_rows_and_never_the_documents_own(seed):
    """The refusals, on the generator's side: the labelled `date` is the ROW's.

    The production timestamp is the strong case and it is an INVARIANT — the statement is made after
    its period closes, 👁 the observed one the following morning, so it is strictly later than every
    operation on the page and can never be any row's date. A record dated by the document would be
    dated days after the money moved, and the period rule reads this value.
    """
    statement = make_statement(seed)
    labelled = label(statement).date

    assert labelled == statement.relevant.at.date()
    assert statement.issued_at.date() > statement.period_end
    assert labelled < statement.issued_at.date()
    assert statement.period_start <= labelled <= statement.period_end


def test_the_last_operation_date_is_a_fact_about_the_table_and_not_about_the_claim():
    """«Дата останньої операції» is the header's most authoritative-looking date and is the wrong
    answer — it is the date of the LAST row, while the labelled row is usually not the last.

    ⚠️ AND THE MEASUREMENT CORRECTED THE CONTRACT. This test was written expecting the two dates to
    coincide on SOME seeds — the labelled row being last, or another row falling on the same day —
    and it found them distinct on 24 of 24. The reason is the trailing window: the period runs at
    least three days past the labelled transaction (`period_trail_days_range`), so with fifteen to
    twenty-five rows drawn across it, some row almost always falls later. The contract said "it may
    coincide" and now says what is true of this corpus instead.

    SO THE DISTINCTNESS IS A PROPERTY OF A DRAW PARAMETER AND NOT AN INVARIANT, which is why the
    count is asserted with its denominator rather than turned into a rule: narrow the trailing
    window and it stops holding, while the refusal in the contract — about WHICH FIELD IS READ —
    does not depend on it either way.
    """
    seeds = list(range(24))
    differ = 0
    for seed in seeds:
        statement = make_statement(seed)
        last = statement.rows[-1].at.date()
        # Always: the header's value is the last row's date, whatever the labelled row is.
        assert statement.render_context()["last_operation_on"] == last.strftime(
            jurisdiction("UA")["date_format"]
        )
        differ += label(statement).date != last

    assert differ == len(seeds), (
        f"the labelled date coincided with the last operation's on {len(seeds) - differ} of "
        f"{len(seeds)} seeds. That is not an error — the two MAY coincide — but the contract and "
        "this docstring both say the corpus does not exhibit it, and one of the three has to change"
    )


# ------------------------------------------------- the arithmetic of the page --


@pytest.mark.parametrize("seed", range(12))
def test_the_four_totals_reconcile_with_the_printed_rows(seed):
    """👁 The observed statement's four figures satisfy this exactly, so it is arithmetic and not a
    convention: what was there, plus what arrived, less what left. The totals are derived from the
    rows for that reason — a drawn closing balance could contradict the table it sits above, and
    nothing in a label would reveal it."""
    statement = make_statement(seed)

    debits = sum((r.amount for r in statement.rows if r.is_debit), Decimal(0))
    credits = sum((r.amount for r in statement.rows if not r.is_debit), Decimal(0))

    assert statement.total_debit == debits
    assert statement.total_credit == credits
    assert statement.closing_balance == statement.opening_balance + credits - debits
    assert statement.debit_count + statement.credit_count == len(statement.rows)


@pytest.mark.parametrize("seed", range(12))
def test_neither_balance_is_negative(seed):
    """An account that ends a period overdrawn is a different document — one with a credit line —
    and nothing observed supports it. The opening balance is DERIVED from the rows for exactly
    this: drawn independently, it produced closing balances of −6117 and −41898 on the first two
    renders."""
    statement = make_statement(seed)

    assert statement.opening_balance >= 0
    assert statement.closing_balance >= 0


@pytest.mark.parametrize("seed", range(12))
def test_the_labelled_amount_appears_on_no_other_row(seed):
    """What makes the label WELL-POSED, and a different decision from deferring decoys.

    A consumer identifies the row from the claim's other document — an invoice naming a seller and
    a total — so two rows carrying that amount would leave the ground truth pointing at one of two
    indistinguishable answers.
    """
    statement = make_statement(seed)
    others = [r.amount for i, r in enumerate(statement.rows) if i != statement.relevant_index]

    assert statement.relevant.amount not in others


@pytest.mark.parametrize("seed", range(12))
def test_every_statement_shows_both_directions(seed):
    """`direction` is a label field, so a page with no credit row would carry no visible instance
    of the distinction the field names. 👁 15 of 89 observed operations are credits, and at least
    one is generated whatever the share returns."""
    statement = make_statement(seed)

    assert statement.credit_count >= 1
    assert statement.debit_count >= 1


@pytest.mark.parametrize("seed", range(12))
def test_the_row_count_and_the_period_are_what_the_configuration_declares(seed):
    """The count comes from `bank_statement.row_count_range`, and every operation falls inside the
    period the header prints — the period is DERIVED from the rows, so a row outside it would be a
    header contradicting its own table."""
    low, high = bank_statement_count_range("row_count")
    statement = make_statement(seed)

    assert low <= len(statement.rows) <= high
    for row in statement.rows:
        assert statement.period_start <= row.at.date() <= statement.period_end


@pytest.mark.parametrize("seed", range(12))
def test_the_operations_are_printed_in_the_order_they_happened(seed):
    """👁 A statement is a chronological extract. It also decides WHERE the labelled row lands:
    ordering by time is what keeps it off the bottom of the page, together with the trailing days
    of the period."""
    statement = make_statement(seed)
    stamps = [row.at for row in statement.rows]

    assert stamps == sorted(stamps)


@pytest.mark.parametrize("seed", range(12))
def test_the_pointer_is_the_printed_number_of_the_labelled_row_and_is_unique(seed):
    """`relevant_transaction` is a POINTER INTO the document, not the identity OF one — which is
    what distinguishes it from `document_code`, left `None` here because 👁 the observed header
    carries no number of its own. A number repeated on two rows would point at both."""
    statement = make_statement(seed)
    numbers = [row.number for row in statement.rows]

    assert label(statement).relevant_transaction == statement.relevant.number
    assert len(set(numbers)) == len(numbers)


@pytest.mark.parametrize("seed", range(12))
def test_every_identifier_printed_on_a_statement_passes_its_own_checksum(seed):
    """Every IBAN and every counterparty code on the page is checksum-correct. A broken
    identifier in this repository has to be a labelled choice of a fraud archetype, never a side
    effect of a generator that did not bother — and this page prints more of them than any other
    archetype does, one set per row."""
    statement = make_statement(seed)

    assert is_valid_iban(statement.account)
    for row in statement.rows:
        assert is_valid_iban(row.counterparty_account)
        code = row.counterparty_code
        # Three lengths under one «Код» caption: 10 is a РНОКПП, 8 a ЄДРПОУ, and 6 the МФО of the
        # bank on its own service-charge row. ⚠️ The caption does not tell them apart; the length
        # does, and only partly — see the confirmation's finding on the same collision.
        assert len(code) in (6, 8, 10), code
        if len(code) == 10:
            assert is_valid_rnokpp(code)
        elif len(code) == 8:
            assert is_valid_edrpou(code)


# ------------------------------------- the invariant: only a debit proves payment --


def test_proves_payment_by_direction_admits_a_debit_and_refuses_a_credit():
    """🔴 The invariant as a validator. `None` passes, and that is a statement rather than a
    lenience: a class that prints no direction is a document about one movement of money, made
    because a payment was made, so there is nothing for the rule to be about."""
    assert proves_payment_by_direction(Direction.DEBIT) is True
    assert proves_payment_by_direction(None) is True
    assert proves_payment_by_direction(Direction.CREDIT) is False


def test_two_operations_may_not_carry_the_same_number():
    """The pointer has to point at ONE row. 👁 A drawn number repeated on two rows of one render
    before the numbering became a counter, and nothing else would have caught it: every labelled
    VALUE was still the right row's and only `relevant_transaction` was ambiguous.

    Built by hand, because `_draw_operation_numbers` cannot produce a collision — the trailing
    digits are one counter across the page. The guard is here for what does not go through it.
    """
    rows = tuple(
        StatementRow(
            number="777",
            at=datetime(2026, 5, 10 + index, 12, 0),
            amount=Decimal("100.00"),
            direction=Direction.DEBIT,
            purpose="п",
            counterparty_name="ТОВ «Т»",
            counterparty_code="12345678",
            counterparty_account="UA123",
            counterparty_bank="Б",
        )
        for index in range(2)
    )
    with pytest.raises(ValueError, match="same number"):
        BankStatement(**_common(rows), relevant_index=0)


def test_a_statement_cannot_be_built_with_a_credit_as_its_labelled_transaction():
    """The same invariant at construction. A credit is money ARRIVING — a refund, a reversal — and
    it evidences no expense whatever its amount, so a statement whose labelled row is a credit
    cannot support the claim it was built for.

    Built by hand, because `build_bank_statement` makes the labelled row a debit and offers no
    parameter to do otherwise. What is tested is the guard, not the builder: a trap archetype is
    where this is broken deliberately, and it will have to declare the break in the label.
    """
    rows = tuple(
        StatementRow(
            number=f"{index}",
            at=datetime(2026, 5, 10 + index, 12, 0),
            amount=Decimal("100.00"),
            direction=direction,
            purpose="п",
            counterparty_name="ТОВ «Т»",
            counterparty_code="12345678",
            counterparty_account="UA123",
            counterparty_bank="Б",
        )
        for index, direction in enumerate((Direction.DEBIT, Direction.CREDIT))
    )
    assert BankStatement(**_common(rows), relevant_index=0).relevant.is_debit
    with pytest.raises(ValueError, match="proof of payment"):
        BankStatement(**_common(rows), relevant_index=1)


def test_the_labelled_row_is_always_a_debit_over_many_seeds():
    """The builder's side of the same invariant, swept rather than asserted once: the guard above
    proves the refusal exists, this proves the builder never needs it."""
    for seed in range(40):
        statement = make_statement(seed)
        assert statement.relevant.direction is Direction.DEBIT


# ------------------------------------------------ the page, and where the boxes are --


def test_the_page_is_one_A4_sheet_the_long_way_round(rendered):
    """The image is exactly the paper `bank_statement.page` declares, and the row count is bounded
    so that it fits: a statement that overflowed onto a second sheet would be a document whose
    second page nobody has observed. ⛔ The orientation itself is not observed — it follows from
    seven columns, two of them free text."""
    _, result = rendered
    assert (result.width, result.height) == (_px("width_mm"), _px("height_mm"))


def test_the_stylesheet_renders_the_paper_the_configuration_declares():
    """The width and height in `<slug>.css` against the millimetres in config/fiscal-rules.yaml.
    Without this the two are unrelated numbers, and a page size edited in one place would leave
    the other stating a paper the image is not."""
    css = (TEMPLATES_DIR / f"{SLUG}.css").read_text(encoding="utf-8")

    assert f"width: {_px('width_mm')}px;" in css
    assert f"min-height: {_px('height_mm')}px;" in css


def test_every_row_of_every_statement_fits_the_declared_sheet(renderer, tmp_path):
    """The bound above, over the row counts that actually occur. `row_count_range` reaches 25, and
    25 rows of two-line cells are close to the page: a stylesheet loosened by a pixel per row would
    push the tallest statements onto a second sheet, which is the case a single render misses."""
    height = _px("height_mm")
    tallest = 0
    # ⚠️ A LIST OF SEEDS IS A MEASUREMENT OF ONE DRAW STREAM, and it goes stale whenever the stream
    # moves. Re-picked when the seller's identity became a claim-level draw: the previous set had
    # stopped reaching 25 rows, and the assertion below is what said so rather than the test quietly
    # exercising 24 for ever.
    for seed in (3, 5, 6, 16, 20, 21, 22):
        statement = make_statement(seed)
        result = renderer.render(SLUG, statement.render_context(), tmp_path / f"{seed}.png")
        tallest = max(tallest, len(statement.rows))
        assert result.height == height, f"seed {seed}, {len(statement.rows)} rows"
    assert tallest == bank_statement_count_range("row_count")[1], (
        "no seed in this set produced the maximum row count, so the bound was not exercised"
    )


def test_the_labelled_boxes_fall_on_the_labelled_row_and_on_no_other(rendered):
    """🔴 THE CLAIM THIS FILE MOST NEEDS TO PROVE, and it is proved by geometry rather than by
    reading the template.

    Every `<tr>` carries an indexed box of its own — `operation_<i>` over the whole row rect — so
    the labelled cells can be located INDEPENDENTLY of the mechanism that marked them: each
    labelled box must be contained in `operation_<k>` where k is the relevant index, and must not
    intersect any other row's rect. An off-by-one in `render_context` is exactly the defect that
    would otherwise pass every other test in this file, since the VALUES would still be the right
    row's.
    """
    statement, result = rendered
    k = statement.relevant_index
    mine = result.field_bboxes[f"operation_{k}"]

    for name in ("relevant_transaction", "date", "amount", "counterparty", "payment_purpose"):
        box = result.field_bboxes[name]
        # Contained horizontally and vertically, within ONE PIXEL. The slack is not measurement
        # error: every box is rounded to whole pixels independently, so a cell whose true top is
        # 473.6 and a row whose true top is 473.5 come back as 474 and 473. A tolerance wider than
        # the rounding would start admitting a genuinely wrong row, which is why the centre test
        # below carries the discrimination.
        assert box[0] >= mine[0] - 1 and box[0] + box[2] <= mine[0] + mine[2] + 1, name
        assert box[1] >= mine[1] - 1 and box[1] + box[3] <= mine[1] + mine[3] + 1, name

        # And the box's own centre is inside ITS row and inside no other. A row is over twenty
        # pixels tall, so this cannot be satisfied by rounding — it is what makes the assertion
        # about the right row rather than about a plausible neighbourhood.
        centre = box[1] + box[3] / 2
        for other in range(len(statement.rows)):
            theirs = result.field_bboxes[f"operation_{other}"]
            within = theirs[1] <= centre <= theirs[1] + theirs[3]
            assert within == (other == k), f"{name} centre in row {other}, relevant is {k}"


def test_the_labelled_amount_sits_in_the_debit_column(rendered):
    """Which of the two money columns the amount is in IS the direction — `direction` has no
    printed element of its own, so the two column headers carry boxes and the amount's own box is
    what a reader combines them with. A labelled amount in the credit column would be a page
    stating the opposite of the label."""
    statement, result = rendered
    debit = result.field_bboxes["column_debit"]
    credit = result.field_bboxes["column_credit"]
    amount = result.field_bboxes["amount"]
    centre = amount[0] + amount[2] / 2

    assert statement.relevant.direction is Direction.DEBIT
    assert debit[0] <= centre <= debit[0] + debit[2]
    assert not credit[0] <= centre <= credit[0] + credit[2]


def test_the_four_totals_are_printed_and_carry_boxes(rendered):
    """The other half of "printed and never labelled": they are on the page and locatable, which
    is what stops "find the relevant transaction" from being artificially easy. A box with no
    labelled value is the receipt's discount line again."""
    _, result = rendered

    for name in ("opening_balance", "closing_balance", "total_credit", "total_debit"):
        assert name in result.field_bboxes
        assert result.field_bboxes[name][2] > 0


def test_moving_the_label_to_another_row_changes_not_one_pixel(renderer, tmp_path):
    """⚠️ THE RELEVANT ROW MUST BE INDISTINGUISHABLE TO THE EYE, proved by rendering the same
    statement twice with the label on two different rows and comparing the two images BYTE FOR
    BYTE. If a stylesheet ever keyed on `data-field`, or the template gave the row a class, this
    goes red — and nothing else would notice, because every label would still be correct while the
    corpus measured "find the highlighted row".
    """
    statement = make_statement()
    elsewhere = next(
        index
        for index, row in enumerate(statement.rows)
        if row.is_debit and index != statement.relevant_index
    )
    moved = BankStatement(
        **{
            field: getattr(statement, field)
            for field in (
                "bank_name", "bank_code", "holder_name", "holder_code", "account",
                "period_start", "period_end", "issued_at", "opening_balance", "rows",
                "payee_trade_name", "decimal_separator",
            )
        },
        relevant_index=elsewhere,
    )

    first = renderer.render(SLUG, statement.render_context(), tmp_path / "a.png")
    second = renderer.render(SLUG, moved.render_context(), tmp_path / "b.png")

    assert first.image_path.read_bytes() == second.image_path.read_bytes()
    # And the boxes DID move, so the test above is comparing two genuinely different labels
    # rather than two renders of the same one.
    assert first.field_bboxes["amount"] != second.field_bboxes["amount"]


def test_the_stylesheet_never_selects_on_a_label_marker(rendered):
    """`data-field` marks the cells of one row. A CSS rule keyed on it would make that row look
    different from its neighbours, and the byte comparison above is the general proof — this is the
    cheap direct one, which names the mistake instead of merely detecting it."""
    css = (TEMPLATES_DIR / f"{SLUG}.css").read_text(encoding="utf-8")
    body = "\n".join(line for line in css.splitlines() if not line.lstrip().startswith("*"))

    assert "data-field" not in body


# ---------------------------------------------------------- what the page says --


def test_a_credit_row_from_the_holders_own_account_names_the_holder(renderer):
    """The defect a render caught and no label could have: the own-account transfer purpose was
    drawn against an ordinary vendor, printing a shop as the sender of a transfer between the
    reader's own accounts. The purpose pool and the counterparty are now chosen together."""
    self_purposes = statement_purposes("uk", "credit_from_self")
    seen = False
    for seed in range(30):
        for row in make_statement(seed).rows:
            if row.purpose in self_purposes:
                seen = True
                assert row.direction is Direction.CREDIT
                assert row.counterparty_name == HOLDER_NAME
    assert seen, (
        "no statement in 30 seeds carried an own-account transfer, so this test asserted "
        f"nothing — the share is {bank_statement_share('credit_from_self')}"
    )


def test_an_ordinary_row_never_carries_a_debit_purpose_on_a_credit(renderer):
    """👁 A credit row's purpose cannot be an «Оплата за…»: the money is arriving. Three pools, and
    the direction decides which one a row draws from."""
    debit_purposes = statement_purposes("uk", "debit")
    for seed in range(20):
        for row in make_statement(seed).rows:
            if not row.is_debit:
                assert not any(
                    row.purpose.startswith(t.split("{")[0]) for t in debit_purposes
                ), row.purpose


def test_the_bank_charges_its_own_service_fee_on_every_statement():
    """👁 1/1 statements carry a bank service charge — the one row whose counterparty is the issuer
    itself. Generated on every statement rather than drawn: 📄 the charge for account servicing
    recurs monthly, so its presence within a period is expected, and ⛔ one document cannot give a
    rate. It also gives the page a debit that is not a payment to a vendor.

    🔴 THE CODE, NOT ONLY THE NAME. This row used to draw its own МФО instead of printing the
    header's, so a delivered statement could name «АТ «Сенс Банк», код 686743» in the header and
    the same bank with code 399161 two lines later — three requisites of one bank on one page, two
    of them disagreeing. `counterparty_code` and the МФО inside `counterparty_account` (an IBAN
    carries it at `[4:10]`) both have to equal the document's own `bank_code`.
    """
    fee_purpose = statement_purposes("uk", "service_fee")[0]
    for seed in range(12):
        statement = make_statement(seed)
        fees = [row for row in statement.rows if row.purpose == fee_purpose]
        assert len(fees) == 1
        assert fees[0].direction is Direction.DEBIT
        assert fees[0].counterparty_name == statement.bank_name
        assert fees[0].counterparty_code == statement.bank_code, (
            f"seed {seed}: the header names bank code {statement.bank_code!r}, the service-charge "
            f"row prints {fees[0].counterparty_code!r} beside the same bank name"
        )
        assert fees[0].counterparty_account[4:10] == statement.bank_code, (
            f"seed {seed}: the service-charge row's own IBAN carries "
            f"{fees[0].counterparty_account[4:10]!r}, not the header's {statement.bank_code!r}"
        )


def test_a_statement_purpose_is_filled_only_from_a_document_reference():
    """🔴 The mechanical half of what `proves_subject: false` rests on: a purpose line is filled
    from an INVOICE NUMBER AND DATE and from nothing else — never from the placeholder vocabulary
    that prints merchandise on a receipt.

    WHAT THIS TEST DOES NOT DO, said plainly because the first version of it did nothing at all.
    "The purpose does not name the expense" is a property of the WORDS in config/generation.yaml,
    and a test reading them to check a claim about them compares the file with itself —
    the defect `lessons.md` records as moving the oracle and the subject together. The first version
    asserted that no `item_kind` slug appeared in a Ukrainian sentence, which is true of every
    Ukrainian sentence ever written, and would have passed whatever the pool said.

    What IS checkable is the agreement between the two sides: every template's placeholder set
    against the arguments `purpose_of` supplies. A template naming a fourth placeholder raises, and
    a builder that dropped one of the three raises — so the pool cannot start printing merchandise
    without this going red.

    ⚠️ `delivery_note_no` JOINED THE SET, AND IT IS NOT A LOOSENING. It names a ВН, a delivery note,
    which is a document class no claim holds — so it is filled from a draw while `invoice_no` on the
    labelled row is filled from the claim's own invoice. The two were one placeholder until the
    cross-document work, which meant a delivery note could be given an invoice's number; see
    docs/cross-document-fields.md and the note beside the templates in config/generation.yaml.
    """
    supplied = {"invoice_no", "invoice_date", "delivery_note_no"}
    for kind in ("debit", "credit", "credit_from_self", "service_fee"):
        for template in statement_purposes("uk", kind):
            named = {f for _, f, _, _ in Formatter().parse(template) if f}
            assert named <= supplied, (kind, template, named)

    # And the builder really does supply both, which is what makes the subset bound load-bearing
    # rather than an observation about strings.
    filled = {row.purpose for seed in range(6) for row in make_statement(seed).rows}
    assert any("№" in purpose for purpose in filled), "no purpose carried a document reference"
    assert not any("{" in purpose for purpose in filled), "a placeholder was left unfilled"


def test_the_document_states_the_period_and_the_last_operation_it_lists(rendered):
    """👁 The header prints the period and «Дата останньої операції», and both are DERIVED from the
    rows: a header stating a period its own table falls outside of is a contradiction a reader
    sees and no label records."""
    statement, _ = rendered
    context = statement.render_context()
    rules = jurisdiction("UA")

    assert context["last_operation_on"] == statement.rows[-1].at.strftime(rules["date_format"])
    assert context["title"] == BLOCK["header"]["title"].format(
        start=statement.period_start.strftime(rules["date_format"]),
        end=statement.period_end.strftime(rules["date_format"]),
    )
