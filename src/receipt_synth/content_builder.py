"""Concrete, valid content for a document — and the validators that say it is valid.

Three groups of things live here, in this order:

1. **Requisites.** Checksum-correct identifiers, an amount spelled out in Ukrainian,
   the VAT letter an item kind takes.
2. **The builder.** Assembles them into one ПРРО receipt (програмний реєстратор
   розрахункових операцій — the software cash register that replaced hardware РРО in
   Ukraine).
3. **The validators.** Each states one invariant as a question with a boolean answer.

The validators are not defensive programming, and their lack of callers on the honest
path is by design rather than an oversight. A truthful document satisfies its invariants
*by construction* — the total is computed as the sum of the line items, so the two cannot
disagree — and there the validators serve as the specification the test suite holds the
builder to. Their production callers arrive with the fraud and trap archetypes, which
state amounts that do **not** follow from their lines: something then has to establish
which invariant broke, so the imperfection can be named in the label instead of merely
rendered. See docs/architecture.md#3-content_builder.

Everything is deterministic: no function here reads a clock or an unseeded generator.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from string import Formatter

from faker import Faker

from receipt_synth.config import (
    acquirers,
    bank_codes,
    bank_statement_count_range,
    bank_statement_money_range,
    bank_statement_share,
    banks,
    category,
    eu_tax_treatment_shares,
    every_vendor,
    excluded_line_counts,
    fiscal_makers,
    high_frequency_surnames,
    initiating_systems,
    initiation_shares,
    invoice_count_range,
    invoice_share,
    jurisdiction,
    load_vendors,
    partial_payment_schedules,
    payment_confirmation_money_range,
    payment_confirmation_share,
    payment_purposes,
    phone_prefixes,
    placeholder_values,
    price_range,
    quantity_choices,
    statement_purposes,
    tax_on_top_rules,
    unprintable_item_kinds,
    vendor_profile,
)
from receipt_synth.schemas import (
    Capture,
    Country,
    Direction,
    DocGroundTruth,
    DocType,
    LineItem,
    Medium,
)

KOPIYKA = Decimal("0.01")


# =============================================================================
# Identifiers
# =============================================================================

# РНОКПП — реєстраційний номер облікової картки платника податків: the individual
# taxpayer number of a Ukrainian resident, ten digits, formerly called ІПН.
#
# Structure and check digit as publicly documented for the register (ДРФО):
#   digits 1–5   days elapsed from 1899-12-31 to the date of birth (1900-01-01 = 1)
#   digits 6–9   a sequence number; the parity of digit 9 encodes sex (even = female)
#   digit 10     check digit = (Σ wᵢ·dᵢ mod 11) mod 10 over digits 1–9,
#                with weights (-1, 5, 7, 9, 4, 6, 10, 5, 7)
_RNOKPP_EPOCH = date(1899, 12, 31)
_RNOKPP_WEIGHTS = (-1, 5, 7, 9, 4, 6, 10, 5, 7)

# ЄДРПОУ — код Єдиного державного реєстру підприємств та організацій України: the
# registry code of a legal entity, eight digits, printed on a receipt as "ІД".
#
# Check digit as publicly documented for the register, and as implemented by
# python-stdnum (`stdnum/ua/edrpou.py`), which is the reference this follows. Two details
# make it unlike the РНОКПП one, and both are load-bearing:
#
#   * the weight set is chosen by the FIRST DIGIT, not by a numeric range. The two
#     readings agree everywhere except at exactly 60000000; keying on the digit is what
#     the standard does, so 6xxxxxxx always takes the plain set.
#   * when the weighted sum leaves a remainder of 10, the calculation is repeated with
#     every weight raised by 2 — and the second remainder is reduced mod 10, so a
#     remainder of 10 becomes check digit 0. Such codes are real and in the register
#     (41761770, 25083040, …); treating them as unissuable rejects valid identifiers.
_EDRPOU_WEIGHTS_LOW = (1, 2, 3, 4, 5, 6, 7)  # first digit not in 3, 4, 5
_EDRPOU_WEIGHTS_MID = (7, 1, 2, 3, 4, 5, 6)  # first digit 3, 4 or 5
_EDRPOU_MID_FIRST_DIGITS = "345"


def _is_digits(value: str, length: int) -> bool:
    # `str.isdigit()` alone accepts non-ASCII digits such as "١"; an identifier that
    # rendered as Arabic-Indic numerals would be nonsense on a Ukrainian receipt.
    return len(value) == length and value.isascii() and value.isdigit()


def _rnokpp_length() -> int:
    """How many digits a РНОКПП has, from config rather than from a literal.

    The ПН path already reads its length out of config/fiscal-rules.yaml, and a literal
    here would have left the same fact stated twice in two different ways. It IS stated
    twice all the same — the weight table pins the body length too — so the two are
    cross-checked instead of one of them being trusted silently.
    """
    length = jurisdiction("UA")["identifiers"]["rnokpp"]["length"]
    body_length = len(_RNOKPP_WEIGHTS)
    if length != body_length + 1:
        raise ValueError(
            f"`identifiers.rnokpp.length` is {length}, but the published check-digit "
            f"algorithm weights {body_length} body digits — check config/fiscal-rules.yaml"
        )
    return length


def rnokpp_check_digit(body: str) -> int:
    """The tenth digit of a РНОКПП, given its first nine."""
    if not _is_digits(body, len(_RNOKPP_WEIGHTS)):
        raise ValueError(f"a РНОКПП body is nine digits, got {body!r}")
    return sum(w * int(d) for w, d in zip(_RNOKPP_WEIGHTS, body, strict=True)) % 11 % 10


def is_valid_rnokpp(value: str) -> bool:
    """Whether a string is a well-formed, checksum-correct РНОКПП."""
    length = _rnokpp_length()
    return _is_digits(value, length) and rnokpp_check_digit(value[: length - 1]) == int(
        value[length - 1]
    )


def generate_rnokpp(
    rng: random.Random,
    birth_date: date | None = None,
    female: bool | None = None,
) -> str:
    """A valid РНОКПП, optionally consistent with a persona's birth date and sex.

    Passing both is what keeps the identifier from merely passing its checksum while
    contradicting the person it belongs to.
    """
    if birth_date is None:
        # A working-age adult. The exact window is arbitrary; the point is that the
        # date is drawn from the seeded generator rather than from a clock.
        birth_date = _RNOKPP_EPOCH + timedelta(days=rng.randint(21_915, 39_082))

    days = (birth_date - _RNOKPP_EPOCH).days
    if not 1 <= days <= 99_999:
        raise ValueError(f"birth date outside the range a РНОКПП can encode: {birth_date}")

    # The parity of the ninth digit encodes sex — even for a woman, odd for a man.
    # Drawn freely when the caller has no opinion.
    parity = rng.randint(0, 1) if female is None else int(not female)
    sex_digit = rng.randrange(parity, 10, 2)

    body = f"{days:05d}{rng.randint(0, 999):03d}{sex_digit}"
    return body + str(rnokpp_check_digit(body))


def edrpou_check_digit(code: str) -> int:
    """The expected eighth digit of a ЄДРПОУ, derived from its first seven.

    Always a digit. The second pass is reduced mod 10, so a remainder of 10 on both
    passes yields 0 — a real and reasonably common outcome, not a sign that the code
    cannot exist.
    """
    if not _is_digits(code, 8):
        raise ValueError(f"a ЄДРПОУ is eight digits, got {code!r}")

    base = (
        _EDRPOU_WEIGHTS_MID if code[0] in _EDRPOU_MID_FIRST_DIGITS else _EDRPOU_WEIGHTS_LOW
    )
    remainder = sum(w * int(d) for w, d in zip(base, code[:7], strict=True)) % 11
    if remainder < 10:
        return remainder

    # Repeat with every weight raised by 2, then fold 10 down to 0.
    return sum((w + 2) * int(d) for w, d in zip(base, code[:7], strict=True)) % 11 % 10


def is_valid_edrpou(value: str) -> bool:
    """Whether a string is a well-formed, checksum-correct ЄДРПОУ."""
    return _is_digits(value, 8) and edrpou_check_digit(value) == int(value[7])


def generate_edrpou(rng: random.Random) -> str:
    """A valid ЄДРПОУ.

    Computed directly from the drawn prefix. This used to draw whole codes and reject
    invalid ones, on the reasoning that the weight set depended on the value of the
    complete code and so could not be resolved from seven digits — that reasoning was
    part of the same misreading that made `edrpou_check_digit` return None. The set
    depends on the first digit alone, so the eighth is a pure function of the first
    seven and there is nothing to reject.
    """
    # Note: excludes codes with a leading zero, which the register does issue. Widening
    # this is tracked separately — it changes which identifiers a given seed produces.
    prefix = f"{rng.randint(1_000_000, 9_999_999):07d}"
    return prefix + str(edrpou_check_digit(prefix + "0"))


# =============================================================================
# Amounts in words — Ukrainian
# =============================================================================

# Both гривня and копійка are feminine, so a numeral agreeing with them takes the
# feminine form. Only 1 and 2 differ between genders.
_ONES = {
    1: "один", 2: "два", 3: "три", 4: "чотири", 5: "п'ять",
    6: "шість", 7: "сім", 8: "вісім", 9: "дев'ять",
}
_ONES_FEMININE = {1: "одна", 2: "дві"}
_TEENS = {
    10: "десять", 11: "одинадцять", 12: "дванадцять", 13: "тринадцять", 14: "чотирнадцять",
    15: "п'ятнадцять", 16: "шістнадцять", 17: "сімнадцять", 18: "вісімнадцять",
    19: "дев'ятнадцять",
}
_TENS = {
    2: "двадцять", 3: "тридцять", 4: "сорок", 5: "п'ятдесят", 6: "шістдесят",
    7: "сімдесят", 8: "вісімдесят", 9: "дев'яносто",
}
_HUNDREDS = {
    1: "сто", 2: "двісті", 3: "триста", 4: "чотириста", 5: "п'ятсот", 6: "шістсот",
    7: "сімсот", 8: "вісімсот", 9: "дев'ятсот",
}

# (value, feminine, (one, few, many)). Тисяча is feminine, мільйон is masculine.
_SCALES = (
    (1_000_000, False, ("мільйон", "мільйони", "мільйонів")),
    (1_000, True, ("тисяча", "тисячі", "тисяч")),
)

_HRYVNIA_FORMS = ("гривня", "гривні", "гривень")
_KOPIYKA_FORMS = ("копійка", "копійки", "копійок")
# The euro is INDECLINABLE in Ukrainian — one form for all three counts — while the cent declines
# like any masculine noun. Both facts are ordinary grammar rather than anything about a document.
_EURO_FORMS = ("євро", "євро", "євро")
_CENT_FORMS = ("цент", "центи", "центів")

# What a Ukrainian document spells an amount in, per currency: the unit's forms, the subunit's,
# and the GENDER THE UNIT GOVERNS — гривня is feminine («одна гривня», «дві гривні») and євро is
# masculine («один євро», «два євро»). The gender belongs to the currency and not to the amount,
# which is why it is stored beside the words rather than passed by a caller.
_CURRENCY_WORDS: dict[str, tuple[tuple[str, str, str], tuple[str, str, str], bool]] = {
    "UAH": (_HRYVNIA_FORMS, _KOPIYKA_FORMS, True),
    "EUR": (_EURO_FORMS, _CENT_FORMS, False),
}

_MAX_UNITS = 1_000_000_000


def _plural_uk(n: int, one: str, few: str, many: str) -> str:
    """Slavic three-form agreement: 1 → one, 2–4 → few, everything else → many,
    except that 11–14 take `many` regardless of their last digit."""
    if 11 <= n % 100 <= 14:
        return many
    last = n % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def _group_words(n: int, feminine: bool) -> list[str]:
    """Words for a value in 1…999."""
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest])
    else:
        tens, units = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if units:
            words.append(_ONES_FEMININE[units] if feminine and units in _ONES_FEMININE
                         else _ONES[units])
    return words


def _int_to_words_uk(n: int, feminine: bool) -> list[str]:
    if n == 0:
        return ["нуль"]
    words: list[str] = []
    for value, scale_is_feminine, forms in _SCALES:
        count, n = divmod(n, value)
        if count:
            words += _group_words(count, scale_is_feminine)
            words.append(_plural_uk(count, *forms))
    if n:
        words += _group_words(n, feminine)
    return words


def amount_in_words_uk(amount: Decimal, currency: str = "UAH") -> str:
    """An amount as printed in words on a Ukrainian document.

    Units in words, subunits in digits, both declined to agree with their own count — the
    convention Ukrainian receipts and invoices follow:

        Decimal("2500.00")        -> "дві тисячі п'ятсот гривень 00 копійок"
        Decimal("394.10"), "EUR"  -> "триста дев'яносто чотири євро 10 центів"

    🔴 A UKRAINIAN DOCUMENT MAY STATE A FOREIGN AMOUNT, and this is the line where it stops being
    a hryvnia one. The words are what the page says the currency is, beside the caption that names
    the code — two independent statements of it, which is what a label recording `EUR` needs from
    an image that would otherwise carry a bare number.

    Raises on a currency this module has no words for, rather than defaulting to hryvnias: an
    amount spelled in the wrong currency is a document that contradicts its own caption.
    """
    if amount < 0:
        raise ValueError(f"a printed amount is not negative: {amount}")
    if amount != amount.quantize(KOPIYKA):
        raise ValueError(f"an amount has at most two decimal places: {amount}")
    if currency not in _CURRENCY_WORDS:
        raise ValueError(
            f"no Ukrainian unit words are stated for {currency!r}; spelling an amount in "
            f"{sorted(_CURRENCY_WORDS)} is what this function can do"
        )

    units, subunits = divmod(int(amount.scaleb(2)), 100)
    if units >= _MAX_UNITS:
        raise ValueError(f"amount too large to spell out: {amount}")

    unit_forms, subunit_forms, feminine = _CURRENCY_WORDS[currency]
    words = _int_to_words_uk(units, feminine=feminine)
    words.append(_plural_uk(units, *unit_forms))
    return f"{' '.join(words)} {subunits:02d} {_plural_uk(subunits, *subunit_forms)}"


# Reverse lookup, so that reading an amount back is an independent act rather than the
# formatter re-run. A validator built on the formatter would only confirm the formatter.
_WORD_VALUES: dict[str, int] = {"нуль": 0}
_WORD_VALUES |= {word: n for n, word in _ONES.items()}
_WORD_VALUES |= {word: n for n, word in _ONES_FEMININE.items()}
_WORD_VALUES |= {word: n for n, word in _TEENS.items()}
_WORD_VALUES |= {word: n * 10 for n, word in _TENS.items()}
_WORD_VALUES |= {word: n * 100 for n, word in _HUNDREDS.items()}

_SCALE_VALUES: dict[str, int] = {
    form: value for value, _, forms in _SCALES for form in forms
}

# Every currency's unit and subunit words at once, so that reading an amount back does not need to
# be told which currency it is in — the words themselves say. The alternations are sorted longest
# first so that a form which is a prefix of another cannot win the match.
_UNIT_FORMS = sorted(
    {form for unit, _, _ in _CURRENCY_WORDS.values() for form in unit}, key=len, reverse=True
)
_SUBUNIT_FORMS = sorted(
    {form for _, subunit, _ in _CURRENCY_WORDS.values() for form in subunit},
    key=len,
    reverse=True,
)

_AMOUNT_IN_WORDS_RE = re.compile(
    rf"^(?P<units>.+?)\s+(?P<unit>{'|'.join(_UNIT_FORMS)})"
    rf"\s+(?P<subunits>\d{{2}})\s+(?P<subunit>{'|'.join(_SUBUNIT_FORMS)})$"
)


def _words_to_int_uk(text: str) -> int:
    total = 0
    group = 0
    seen = False
    for word in text.split():
        if word in _SCALE_VALUES:
            total += (group or 1) * _SCALE_VALUES[word]
            group = 0
            seen = True
        elif word in _WORD_VALUES:
            group += _WORD_VALUES[word]
            seen = True
        else:
            raise ValueError(f"not a Ukrainian numeral: {word!r}")
    if not seen:
        raise ValueError("no numeral found")
    return total + group


def words_to_amount_uk(text: str) -> Decimal:
    """The amount a Ukrainian amount-in-words states, in whatever currency it names. Raises
    ``ValueError`` if the text is not one.

    ⛔ IT DOES NOT REPORT THE CURRENCY, and that is deliberate: this is the independent reading of
    a NUMBER, used to check that a printed figure and its printed words agree. Which currency the
    two are in is stated by the caption beside them and by the label, and a second answer here
    would be a second place for the two to disagree.
    """
    match = _AMOUNT_IN_WORDS_RE.match(text.strip())
    if match is None:
        raise ValueError(f"not an amount in words: {text!r}")
    units = _words_to_int_uk(match["units"])
    return (Decimal(units) + Decimal(match["subunits"]) / 100).quantize(KOPIYKA)


# =============================================================================
# VAT letters
# =============================================================================


def allowed_vat_letters(item_kind: str, country: str) -> tuple[str, ...]:
    """Every ПДВ-літера (VAT rate code printed per line) an item kind may carry.

    Usually one. A vitamin complex may be registered either as a medicinal product or
    as a dietary supplement, and real receipts show both — so the configuration gives a
    list, and the choice is made per document.
    """
    mapping = jurisdiction(country)["item_vat_letter"]
    value = mapping.get(item_kind, mapping["default"])
    return tuple(value) if isinstance(value, list) else (value,)


def vat_letter_for_kind(item_kind: str, country: str, rng: random.Random) -> str:
    """The VAT letter this item kind takes on this document."""
    letters = allowed_vat_letters(item_kind, country)
    return letters[0] if len(letters) == 1 else rng.choice(letters)


def vat_rate_for_letter(letter: str, country: str) -> float:
    """The percentage a VAT letter stands for. Raises ``KeyError`` for an unknown one."""
    return jurisdiction(country)["vat_letters"][letter]["rate"]


# =============================================================================
# Document content
# =============================================================================


@dataclass(frozen=True)
class Seller:
    """The party issuing the receipt.

    UP TO TWO IDENTIFIER LINES, AND THEY ARE NOT ALTERNATIVES. Each is optional on its own:

    * ``tax_code`` is the ІД — the seller's identification code. An eight-digit ЄДРПОУ for a
      legal entity, a ten-digit РНОКПП for a sole trader: the length follows the TYPE OF PERSON
      and not the prefix. ``None`` models a receipt that omits the line, which real receipts
      may do — 👁 only 1 of 3 real receipts carries one, and this generator prints it on every
      document anyway, as the superset. That overstatement is declared under
      ``known_limitations`` in config/labelling-schema.yaml; the axis of the variation is the
      ПРРО software provider rather than the seller, so the variant is a second template. See
      ``identifiers`` in config/fiscal-rules.yaml.
    * ``vat_number`` is the ПН — the VAT-payer number, and ``None`` unless the seller is
      registered. Twelve digits for a legal entity, of which the first eight are its ЄДРПОУ, so
      the two lines agree by construction — 👁 1/1, MEANING ONE DOCUMENT: only one observed
      receipt carries both lines, and no source states the relation as a requirement. For a sole
      trader it is THE SAME ten-digit РНОКПП the ІД line carries — one number under two
      prefixes, not two numbers.

    A registered payer therefore prints one line MORE than a non-payer, not a different one.
    An earlier version of this model had them mutually exclusive, on a published table of the
    form that lists them as rows 4 and 5 with alternative examples; real ПРРО output prints both
    together and refuted it.

    ``vat_payer`` is carried on the seller rather than looked up again at render time because it
    decides two requisites of the same document — whether ПН is printed at all, and whether the
    receipt has a VAT block — and the two must not be able to disagree.
    """

    name: str
    legal_form: str
    address: str
    vat_payer: bool  # платник ПДВ: registered for VAT, so the receipt carries a VAT block
    tax_code: str | None
    tax_code_label: str
    vat_number: str | None
    vat_number_label: str


@dataclass(frozen=True)
class TaxLine:
    """One row of the tax summary block at the foot of the receipt."""

    letter: str
    rate: float
    gross: Decimal  # the VAT-inclusive turnover taxed at this rate
    vat: Decimal  # the tax contained within it


@dataclass(frozen=True)
class Acquiring:
    """The card-acquiring block, printed when the receipt was paid by card.

    ``rrn`` (retrieval reference number) is the deduplication key: the same RRN in two
    files is the same payment claimed twice.
    """

    acquirer: str
    terminal_id: str
    operation: str
    card_masked: str  # ЕПЗ — електронний платіжний засіб, the masked card number
    auth_code: str
    rrn: str


@dataclass(frozen=True)
class PrroReceipt:
    """One Ukrainian fiscal receipt, complete but not yet rendered.

    NAMED AFTER THE ПРРО AND NO LONGER ONLY ONE. It also carries a classic hardware РРО
    receipt, which differs in the fiscal identity it prints and in nothing else this class
    models — see `build_prro_receipt`. The name is left alone deliberately: renaming it
    reaches every test module and is a rename rather than a change of behaviour, so it is
    recorded here as a naming defect instead of being fixed in the same commit as the
    templates.
    """

    seller: Seller
    issued_at: datetime
    title: str
    # Онлайн / Офлайн. `None` on a classic hardware РРО: 📄 the mode marker is line 31 of the
    # form and a ПРРО requisite, so a hardware receipt carries no such line.
    mode_marker: str | None
    receipt_number: str
    fiscal_device_number: str
    # The prefix printed before `fiscal_device_number` — «ФН ПРРО» or «ФН». Carried rather
    # than looked up in the template, because it follows the kind of registrar and a template
    # cannot know which one built the document. A literal in the markup is exactly what put
    # «ПН» on every sole trader's receipt.
    fiscal_number_label: str
    # ЗН — the factory serial of a hardware РРО, and `None` for a ПРРО, which has none.
    device_serial: str | None
    device_serial_label: str
    # The maker printed immediately after «ФІСКАЛЬНИЙ ЧЕК»: 📄 one requisite with the wording,
    # 👁 present on 11 of 11 open receipts. A software provider for a ПРРО, a device
    # manufacturer for a hardware РРО.
    provider_name: str
    line_items: list[LineItem]
    total: Decimal
    # СУМА and ДО СПЛАТИ are two different lines of the form (20 and 24) and genuinely differ
    # by these two. Both are zero in this version, so `amount_due` equals `total` — see
    # `amount_due` below for why the divergence is deferred rather than merely unimplemented.
    discount: Decimal  # ЗНИЖКА
    rounding: Decimal  # ЗАОКРУГЛЕННЯ — cash rounding to the nearest printable unit
    amount_in_words: str
    tax_lines: list[TaxLine]
    payment_method: str
    acquiring: Acquiring | None
    decimal_separator: str
    qr_payload: str
    footer: str
    # WHICH OF THE TWO 👁 OBSERVED FORMS the VAT summary row takes — a key of
    # `tax_line_label_forms` in config/fiscal-rules.yaml. `None` for a seller that is not
    # registered for ПДВ, whose receipt has no tax block at all. Chosen from the document's MEDIUM
    # rather than drawn freely; see `_draw_tax_line_form`.
    vat_row_form: str | None

    @property
    def amount_due(self) -> Decimal:
        """ДО СПЛАТИ — what the customer actually pays: the basket less any discount, plus
        cash rounding.

        DERIVED, NOT STORED, so that the two amounts cannot drift apart — and the derivation is
        exercised rather than merely asserted: every receipt this builder produces fixes both
        adjustments at zero, so the arithmetic and the sign of each term would be invisible to
        the whole suite were it not for
        ``test_amount_due_is_the_total_less_the_discount_plus_the_rounding``, which sets them by
        hand. It equals ``total`` for as long as both adjustments are zero, and that is
        deliberate: reaching a genuine
        divergence needs a rule for DISTRIBUTING a basket-level discount across covered and
        non-covered lines, because coverage is decided per line and the lines sum to
        ``total`` — and config/policy.yaml says nothing about how. Choosing here would wire an
        interpretation the policy does not contain into the ground truth.

        The consequence for a consumer is stated in config/labelling-schema.yaml, where it
        matters: while the two coincide the field DISCRIMINATES NOTHING, so its accuracy must
        not be reported as a metric.
        """
        return (self.total - self.discount + self.rounding).quantize(KOPIYKA)

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted.

        Formatting lives here rather than in the template so that the number format of a
        jurisdiction is decided once, in one place, instead of in twenty-four templates.
        """
        rules = jurisdiction("UA")
        return {
            "title": self.title,
            "mode_marker": self.mode_marker,
            "seller": self.seller,
            "seller_display": legal_name(self.seller),
            "date": self.issued_at.strftime(rules["date_format"]),
            "time": self.issued_at.strftime(rules["time_format"]),
            "receipt_number": self.receipt_number,
            "fiscal_device_number": self.fiscal_device_number,
            "fiscal_number_label": self.fiscal_number_label,
            "device_serial": self.device_serial,
            "device_serial_label": self.device_serial_label,
            "provider_name": self.provider_name,
            "items": [
                {
                    "name": item.name,
                    "qty": f"{item.qty:g}",
                    "price": self._amount(item.price),
                    "sum": self._amount((item.qty * item.price).quantize(KOPIYKA)),
                    "vat_letter": item.vat_letter,
                }
                for item in self.line_items
            ],
            # Label and amount separately, because the amount belongs in the same right-hand
            # column as every other amount on the paper. One string for the whole row is what
            # left the VAT amounts floating mid-line while the four totals beside them were
            # flush right.
            "tax_lines": [
                {
                    # Two placeholders for the rate, and a jurisdiction picks one:
                    # `rate` is the bare number, `rate_2dp` two decimals written with this
                    # document's own separator — the form observed on Ukrainian receipts.
                    "label": _tax_line_format(rules, self.vat_row_form).format(
                        name=rules["vat_letters"][line.letter]["name"],
                        letter=line.letter,
                        rate=line.rate,
                        rate_2dp=f"{line.rate:.2f}".replace(".", self.decimal_separator),
                    ),
                    "amount": self._amount(line.vat),
                }
                for line in self.tax_lines
            ],
            "total": self._amount(self.total),
            "discount": self._amount(self.discount),
            "rounding": self._amount(self.rounding),
            "amount_due": self._amount(self.amount_due),
            "totals_labels": rules["receipt"]["totals_labels"],
            "amount_in_words": self.amount_in_words,
            "payment_method": self.payment_method,
            "acquiring": self.acquiring,
            "acquiring_labels": {
                field_spec["key"]: field_spec["label"]
                for field_spec in rules["acquiring_block"]["fields"]
            },
            "qr_payload": self.qr_payload,
            "footer": self.footer,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this receipt.

        Nothing here is inferred from a rendered image — every value was decided before
        the document existed. Only the bounding boxes come from the renderer, and they
        come from the browser's layout engine rather than from OCR.
        """
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.FISCAL_RECEIPT,
            language="uk",
            currency="UAH",
            amount=self.total,
            amount_due=self.amount_due,
            date=self.issued_at.date(),
            counterparty=self.seller.name,
            line_items=self.line_items,
            has_qr=True,
            qr_is_fiscal=True,
            has_fiscal_number=True,
            capture=capture,
            field_bboxes=field_bboxes,
            # From the RENDERER, like the boxes: neither is decided by the content class, and both
            # describe the page that was produced from it.
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
            vat_row_form=self.vat_row_form,
        )


# --- item content ------------------------------------------------------------

# Everything a line item is made of — the vocabulary that fills `{brand}` and `{dose}`,
# what an article costs, how many of it a basket holds — is data and lives in
# `config/generation.yaml`. It used to live here, and it was the only Ukraine-and-pharmacy
# assumption left in the codebase.

_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# The legal form as printed before the name. A ФОП is printed WITHOUT quotes: a sole
# trader trades under a person's name, not under a firm name, and «ФОП «Прізвище І. Б.»»
# is not a form any Ukrainian document uses.
_LEGAL_FORM_PREFIX = {"TOV": "ТОВ", "FOP": "ФОП", "PRAT": "ПрАТ"}
_SOLE_TRADER = "FOP"

# The designation of a firm registered OUTSIDE the Ukrainian register, printed AFTER the name and
# without quotes — "Coursera Inc." The Ukrainian marks are a rule about how a UKRAINIAN firm name
# is written; applying them to a foreign one produces «INC «Coursera»», a form no register holds
# and no document prints. The forms are the ones config/vendors.json's `EU` block uses.
_LEGAL_FORM_SUFFIX = {
    "INC": "Inc.", "LLC": "LLC", "SARL": "S.à r.l.", "GMBH": "GmbH", "EV": "e.V.",
    # ⛔ AN EMPTY DESIGNATION IS A STATEMENT, and it is the honest one for a foreign institution
    # that trades under a bare name. The forms above are each a real entity's actual one — the
    # rule config/vendors.json states for a public mark applies to the form printed beside it —
    # and a body whose form this repository has not verified gets none rather than a plausible
    # guess. A wrong legal form is a checkable false claim about a real organization.
    "NONE": "",
}

# Jurisdictions that print a sole trader as surname plus initials — "Ковальчук О. С." —
# rather than as a full name.
#
# Legal forms whose holder trades under a natural person's name, so that no name is stored
# for them and one is drawn per vendor instance. A German `EK` is deliberately absent: it is
# a registered sole merchant who may equally trade under a business designation, so for that
# form the presence or absence of a stored name is what decides, not the form itself.
#
# Both belong beside `_LEGAL_FORM_PREFIX` above: all three are jurisdiction rules living in
# code, and they move out together when the non-UA templates land. Kept adjacent so the set
# is found at once rather than one member at a time.
_SURNAME_AND_INITIALS = frozenset({"UA"})
_PERSONAL_NAME_FORMS = frozenset({"FOP", "JDG", "EK", "AUTONOMO"})

# The longest receipt this builder will print. A cap rather than a preference: the planner
# sizes a basket upward when it needs one large enough to overrun an annual limit, and
# without a bound a large enough remaining balance would ask for a receipt no shop issues.
#
# Stays in code deliberately, unlike the item vocabulary next to it. It is a bound the
# planner needs a name for — `claim_planner._overrun_item_count` clamps to it — not a knob
# anyone tunes to change what a dataset looks like.
MAX_LINE_ITEMS = 20


def _placeholders(template: str) -> list[str]:
    # Sorted, because iterating a set would order the draws by a hash that varies
    # between interpreter runs — and determinism under --seed would quietly stop holding.
    return sorted({name for _, name, _, _ in Formatter().parse(template) if name})


def _fill_placeholders(
    template: str, item_kind: str, rng: random.Random, language: str = "uk"
) -> str:
    """A line-item name with its placeholders resolved.

    Raises ``KeyError`` through `config.placeholder_values` when a placeholder has no
    vocabulary. This used to return ``None`` and the caller skipped the template, which
    meant a name template nobody had filled in silently left the dataset's vocabulary
    narrower than the policy declares — and nothing anywhere said so.
    """
    return template.format(
        **{
            name: rng.choice(placeholder_values(item_kind, name, language))
            for name in _placeholders(template)
        }
    )


def sellable_kinds(catalogue: dict, vendor: dict) -> list[str]:
    """The item kinds of a bucket that this vendor sells AND this generator can print.

    Two filters, and they answer different questions. A pharmacy does not sell "Складання
    плану харчування": that is affinity, vendor data, declared per profile in
    `config/vendors.json`, and this function only intersects it with the bucket rather than
    deciding what a shop stocks. A pharmacy does sell shower gel, and this generator still
    cannot print one, because the template asks for a brand and no publicly known house
    makes both articles the kind's templates name: that is `unprintable_item_kinds` in
    `config/generation.yaml`, a declared and tested hole rather than a silent skip.

    Sorted, because the result is drawn from: iterating the profile's own set would order
    the draws by a hash that varies between interpreter runs.
    """
    sells = vendor_profile(vendor["profile"])
    unprintable = unprintable_item_kinds()
    return sorted(kind for kind in catalogue if kind in sells and kind not in unprintable)


def _faker_locale(country: str) -> str:
    """The Faker locale of a jurisdiction, e.g. ``UA -> uk_UA``.

    Composed rather than tabulated: a Faker locale identifier is ``language_COUNTRY``, and
    both halves are already in fiscal-rules.yaml — the block's own key and its `language`.
    A second table would be the same two facts written down again, and the copy is the one
    that goes stale.
    """
    return f"{jurisdiction(country)['language']}_{country}"


def personal_surname(rng: random.Random, fake: Faker, country: str, *, female: bool) -> str:
    """The surname of a person named on a document of this jurisdiction.

    Drawn from the narrowed high-frequency set in `config/generation.yaml` where the
    jurisdiction's language declares one, and from Faker's own pool where it does not. The
    reasoning is stated under `personal_names` in that file; the short form is that this
    dataset is published, and a rare surname on a rendered receipt points at whoever bears
    it while a surname carried by a hundred thousand people does not.

    ONE mechanism for both places a personal name is composed — a sole trader below, and a
    persona in `persona_generator`, whose name is not printed yet but will be as the payer
    on a payment confirmation. The exposure is the same in both, so the narrowing is not
    built twice.

    Takes the caller's `Faker` rather than making one, because both callers already have a
    seeded instance for the given name and a second instance would draw from a second seed.
    """
    pool = high_frequency_surnames(jurisdiction(country)["language"])
    if pool:
        return rng.choice(pool)
    return fake.last_name_female() if female else fake.last_name_male()


def sole_trader_name(rng: random.Random, country: str = "UA") -> str:
    """The printed name of a sole trader — composed, never stored.

    A curated list of personal names is a standing liability: every entry is an unverified
    claim about who trades under that name, and it has to be re-checked as the world
    changes. So the name is composed at build time rather than kept as a column in
    `config/vendors.json` — the surname by `personal_surname` above, the given name and
    patronymic by Faker, neither of which identifies anybody on its own.

    Ukrainian documents print a sole trader as surname plus initials — ``Ковальчук О. С.`` —
    which is both the convention and what a receipt shows; elsewhere the full name is
    printed. That branch is jurisdiction law living in code, like `_LEGAL_FORM_PREFIX`, and
    the two move out together.
    """
    # Faker carries its own generator, so it is seeded from ours rather than left to start
    # from a clock — the same arrangement `persona_generator` uses.
    fake = Faker(_faker_locale(country))
    fake.seed_instance(rng.getrandbits(64))

    female = rng.random() < 0.5
    first = fake.first_name_female() if female else fake.first_name_male()
    last = personal_surname(rng, fake, country, female=female)
    if country not in _SURNAME_AND_INITIALS:
        return f"{first} {last}"

    # По батькові — the patronymic, the second initial on a Ukrainian document.
    middle = fake.middle_name_female() if female else fake.middle_name_male()
    return f"{last} {first[0]}. {middle[0]}."


def resolve_vendor(rng: random.Random, vendor: dict, country: str = "UA") -> dict:
    """A vendor entry with its printed name settled, ready to be carried.

    An entry that STATES a name keeps it — that is a firm trading under a mark. An entry
    with NO name is one that trades under a natural person's, and the name is drawn here.
    Absence is the signal rather than the legal form, because the two do not coincide: a
    German `EK` is a registered sole merchant who may trade under a business designation
    ("Sonnen-Apotheke e.K.") just as readily as under their own name.

    Called ONCE per vendor instance — when the vendor is chosen for a claim — and the result
    is passed to every document of that claim. That ordering is the whole design: a personal
    name is now a draw rather than a constant, and a draw repeated per document would print
    two different sellers on two documents of one purchase. The constraint is older than the
    draw; what changed is that the type no longer enforces it, since a constant string could
    not differ from itself and a drawn one can.
    """
    if "name" in vendor:
        return vendor
    if vendor["legal_form"] not in _PERSONAL_NAME_FORMS:
        raise ValueError(
            f"vendor entry {vendor!r} states no name, but {vendor['legal_form']!r} is a "
            "legal form that trades under a mark rather than under a person's name. Only "
            f"{sorted(_PERSONAL_NAME_FORMS)} have their name drawn; everything else states "
            "one in config/vendors.json."
        )
    return {**vendor, "name": sole_trader_name(rng, country)}


def _vat_number(rng: random.Random, id_code: str, *, is_sole_trader: bool) -> str:
    """The ПН — VAT-payer number — of a registered seller, built FROM its ІД.

    A sole trader's ПН simply IS its РНОКПП: 👁 one ten-digit number appears under both
    prefixes, so there is nothing to draw.

    A legal entity's is twelve digits, and 👁 the first eight of them are its ЄДРПОУ on the one
    company receipt observed. So the remaining digits are drawn and appended rather than a
    fresh number being made up: independent numbers would contradict that document, and would
    also let a consumer's cross-check between the two printed lines fail on a document this
    generator calls honest. Derived rather than stored for the same reason `amount_due` is —
    two values that must agree should not be two values.

    The relation rests on a SINGLE observation, and no source states it as a requirement. What
    is certain is the length, which config/fiscal-rules.yaml holds.
    """
    if is_sole_trader:
        return id_code

    length = jurisdiction("UA")["identifiers"]["vat_number"]["legal_entity_length"]
    tail = length - len(id_code)
    if tail < 0:
        raise ValueError(
            f"a ПН of {length} digits cannot begin with a {len(id_code)}-digit ІД — check "
            "`identifiers` in config/fiscal-rules.yaml"
        )
    return id_code + f"{rng.randint(0, 10**tail - 1):0{tail}d}"


def vendor_is_vat_payer(vendor: dict) -> bool:
    """Whether this vendor is registered for ПДВ — податок на додану вартість, value added tax.

    Read from the vendor entry, never derived from ``legal_form``. It decides two printed
    requisites at once — which tax identifier the seller block carries and under which prefix,
    and whether the receipt has a VAT block at all — and both have exceptions in either
    direction: a ФОП on the general system is registered, a small company on the simplified
    system is not. A rule over the legal form would print a configuration the entry
    contradicts, which is the shape of the defect this replaced.

    A MISSING FLAG IS REFUSED RATHER THAN DEFAULTED. A default would quietly make the status a
    property of the legal form again for every entry nobody thought about — and the printed
    consequence would be a plausible-looking document rather than a failure.
    """
    if "vat_payer" not in vendor:
        raise ValueError(
            f"vendor entry {vendor!r} states no `vat_payer`. Whether the seller is registered "
            "for ПДВ decides its tax-identifier line and whether the receipt carries a VAT "
            "block at all, and it does not follow from the legal form — state it on the entry "
            "in config/vendors.json"
        )
    return bool(vendor["vat_payer"])


@dataclass(frozen=True)
class PartyIdentity:
    """Who a party IS on paper, drawn ONCE PER CLAIM and printed on every document of it.

    🔴 THE CONSTRAINT `resolve_vendor` SOLVES FOR THE NAME, SOLVED FOR THE NUMBERS TOO. That
    docstring says a per-document draw "would print two different sellers on two documents of one
    purchase", and it was right — but it fixed only the name, so `generate_edrpou` and
    `generate_iban` went on being called once per builder and every identifier of one seller
    differed between the two documents of the same claim. Measured on the delivered corpus: 587
    pairs, and the seller's tax code and IBAN disagreed on 587 of them, while the name agreed on
    587 of them. See docs/cross-document-fields.md.

    Why that is worse than an ordinary wrong value: **linking documents of one claim is what this
    corpus exists to pose as a problem**, and a field that never matches is not a hard instance of
    that problem — it is an unsolvable one. A system scored on it scores zero by construction, and
    the number says something about the generator rather than about the system. The name matching
    byte for byte is the same defect from the other side.

    ⚠️ `bank_name` AND `bank_code` BELONG TO THE IDENTITY AND NOT TO THE DOCUMENT, because
    `account` is built on `bank_code` — an IBAN carries its bank's МФО in the clear. A document
    drawing its own bank while printing the claim's account would print a bank code that
    contradicts the account beside it, which is a defect no cross-document check would catch and
    every reader of one page would see.

    Not to be confused with `Seller`, `Party` or `InvoiceParty`: those are the party blocks three
    document classes PRINT, each with its own captions and its own optional fields. This is the
    identity all three print, and it is drawn where a claim is assembled rather than where a page
    is composed.
    """

    tax_code: str
    vat_number: str | None
    bank_name: str
    bank_code: str
    account: str


def _draw_bank(rng: random.Random, country: str) -> tuple[str, str]:
    """A bank name and its code, together — the one draw plus the one lookup, in one place.

    Third occurrence of `rng.choice(banks(country)); bank_codes(country)[bank_name]`: once here for
    `draw_party_identity`, once for the payment confirmation's own issuer, once for a statement
    row's account. Extracted so a name can no longer be drawn without its code beside it — the same
    reason `bank_codes` exists at all, one call site short of covering every draw.

    THE DRAW ITSELF IS UNCHANGED: one `rng.choice` over `banks(country)`, same as before this
    existed, so a seeded run's output does not move.
    """
    bank_name = rng.choice(banks(country))
    return bank_name, bank_codes(country)[bank_name]


def draw_party_identity(
    rng: random.Random, vendor: dict, country: str = "UA"
) -> PartyIdentity:
    """Draw one party's identity — the register the code comes from follows the legal form.

    A ФОП has no ЄДРПОУ at all, so an eight-digit code beside a sole trader's name would be an
    identifier no register could resolve to the party printed next to it. The ПН is derived from
    the ІД rather than drawn, for the reason `_vat_number` gives, and is drawn here even for the
    classes that never print it: it costs one value from the generator and keeps the draw the same
    length whichever archetype the claim turns out to use.

    Called once per claim by the assembler, and once per ORDINARY STATEMENT ROW by
    `build_bank_statement` — those counterparties are different firms on purpose, and a fresh
    identity per row is what they need.
    """
    is_sole_trader = vendor["legal_form"] == _SOLE_TRADER
    tax_code = generate_rnokpp(rng) if is_sole_trader else generate_edrpou(rng)
    # The NAME is drawn; the CODE is looked up. See `_draw_bank` — a fresh draw here is exactly
    # the defect this function used to carry: the same real bank name coming back with a
    # different МФО on the next identity drawn for it.
    bank_name, bank_code = _draw_bank(rng, country)
    return PartyIdentity(
        tax_code=tax_code,
        vat_number=(
            _vat_number(rng, tax_code, is_sole_trader=is_sole_trader)
            if vendor_is_vat_payer(vendor)
            else None
        ),
        bank_name=bank_name,
        bank_code=bank_code,
        account=generate_iban(rng, bank_code, country),
    )


@dataclass(frozen=True)
class DocumentReference:
    """The document a payment's purpose line cites: its number, and the date it bears.

    🔴 A REFERENCE IS A CROSS-DOCUMENT FIELD OF ITS OWN, and it used to be drawn independently on
    each side — the invoice printed one number in its title, and the payment beside it cited a
    number drawn from `rng.randint(1, 9999)`, so the two never agreed on any pair of the delivered
    corpus. Passing the subject document's own reference makes the citation resolvable: it is
    embedded in free text on the payment side and written into a title on the subject side, and the
    DATE is spelled in words on one page and in digits on the other, so the two ends still have to
    be parsed and normalized before they can be compared. That is the difference between a field a
    system can earn a score on and one it cannot.

    ⛔ It is NOT passed to every purpose line. A statement's ordinary rows and a purpose naming a
    ВН — a delivery note, a different class of document — refer to documents that are not in the
    claim, and that is the whole reason a payment document establishes nothing about what was
    bought. See `_cited_number` and docs/cross-document-fields.md.
    """

    number: str
    issued_at: datetime


def vendor_can_carry(vendor: dict, category_id: str, *, mixed: bool) -> bool:
    """Whether this vendor can issue the receipt a plan asks for.

    Asked by the assembler before a vendor is chosen, so that a plan needing a non-covered
    line is never handed to a vendor that sells nothing non-covered. Honest profiles exist
    that cannot — a nutrition practice sells services and nothing the plan excludes — and
    the alternative to filtering here is a builder that fails on a vendor which never could.
    """
    spec = category(category_id)
    if not sellable_kinds(spec["covered_items"], vendor):
        return False
    return not mixed or bool(sellable_kinds(spec["excluded_items"], vendor))


def _minor(amount: Decimal) -> int:
    """An amount in hryvnias as a whole number of kopiykas.

    Configuration states prices on the same scale as `annual_limit` in policy.yaml, which
    is the scale a human reads. Whole minor units are what the price draw and the reprice
    clamp need, and converting here is what keeps that a detail of this module rather than
    something the file's reader has to hold in their head. Exact: `config.price_range`
    refuses a range finer than two decimal places, so nothing is rounded away.
    """
    return int(amount.scaleb(2))


def _build_line_item(
    item_kind: str,
    templates: list[str],
    rng: random.Random,
    *,
    covered: bool,
    vat_payer: bool,
    language: str = "uk",
    currency: str = "UAH",
) -> LineItem:
    names = [
        _fill_placeholders(template, item_kind, rng, language=language)
        for template in templates
    ]

    low, high = price_range(item_kind, currency)
    return LineItem(
        name=rng.choice(names),
        item_kind=item_kind,
        qty=Decimal(rng.choice(quantity_choices())),
        # Drawn in whole ten-kopiyka steps: retail prices do not end in arbitrary
        # kopiykas, and an exact integer keeps the sum exact. `randrange` is half-open, so
        # the configured `high` is the one price this draw cannot produce — see
        # `config.price_range`, which says where the bound IS inclusive. The same
        # ten-minor-unit grid serves the euro draw — one mechanism, two currencies, as the
        # `price_ranges_eur` comment declares.
        price=Decimal(rng.randrange(_minor(low), _minor(high), 10)) / 100,
        covered=covered,
        # A seller with no ПДВ registration has assigned no rate group to anything, so the
        # line carries no letter — and nothing takes its place: 👁 the line ends with the
        # amount. Not the zero-rate letter «Г», not "Без ПДВ". See the note at the head of
        # config/generation.yaml for the three sources that disagree about this.
        vat_letter=vat_letter_for_kind(item_kind, "UA", rng) if vat_payer else None,
    )


# How many draws to allow before accepting a shorter receipt. Reached only when a
# category has fewer distinct renderable names than the requested line count.
_DISTINCT_DRAW_LIMIT = 40


def _draw_distinct_items(
    rng: random.Random,
    kinds: list[str],
    catalogue: dict,
    count: int,
    *,
    covered: bool,
    vat_payer: bool,
    language: str = "uk",
    currency: str = "UAH",
) -> list[LineItem]:
    """Line items with distinct printed names.

    A receipt lists a product once and says how many; the same article appearing twice on
    one receipt at two different prices is not something a cash register produces. Kinds
    are still drawn with replacement — a pharmacy basket really can hold two different
    vitamins — it is the printed name that has to be unique.

    `language` selects which template list of the catalogue a name is drawn from —
    policy.yaml carries `uk` and `en` for every kind — and `currency` selects the price
    block, per `config.price_range`. Defaults keep every existing caller a Ukrainian one.
    """
    items: list[LineItem] = []
    seen: set[str] = set()

    # Scaled by the requested count: the flat bound was written for baskets of two to
    # four, and a basket sized to overrun an annual limit would otherwise run out of
    # attempts before it ran out of names.
    attempts = max(_DISTINCT_DRAW_LIMIT, _DISTINCT_DRAW_LIMIT * count // 4)
    for _ in range(attempts):
        if len(items) == count:
            break
        kind = rng.choice(kinds)
        item = _build_line_item(
            kind,
            catalogue[kind][language],
            rng,
            covered=covered,
            vat_payer=vat_payer,
            language=language,
            currency=currency,
        )
        if item.name not in seen:
            seen.add(item.name)
            items.append(item)
    return items


def estimated_line_value(category_id: str) -> Decimal:
    """Roughly what one covered line of this category is worth.

    Used by `claim_planner` to size a basket *before* it is drawn, when it needs one large
    enough to exceed a remaining annual balance. An estimate and nothing else: no label is
    ever derived from it, and a basket that misses the balance is reported as a miss
    rather than relabelled.

    Averaged over every covered kind of the category rather than over the ones the vendor
    sells, because the planner runs before a vendor is chosen. That widens the error — a
    pharmacy sells none of the four-figure service kinds the average includes — and the
    error is absorbed the same way as every other: a basket that fails to overrun the
    balance is labelled by what it turned out to be.
    """
    kinds = sorted(category(category_id)["covered_items"])
    if not kinds:
        raise ValueError(f"category {category_id!r} declares no covered item")

    ranges = [price_range(kind) for kind in kinds]
    mean_price = sum((low + high) for low, high in ranges) / (2 * len(ranges))
    quantities = quantity_choices()
    mean_qty = Decimal(sum(quantities)) / len(quantities)
    return (mean_price * mean_qty).quantize(KOPIYKA)


def _repriced(item: LineItem, line_total: Decimal, currency: str = "UAH") -> LineItem:
    """The same line, priced so it comes to about ``line_total``.

    Clamped into the item kind's own range — in the basket's own currency — and rounded to
    ten minor units, so that hitting a coverage target cannot print a 4 UAH
    blood-pressure monitor. The clamp is why the realized coverage only approaches the
    target — which is enough, because the target only has to land the claim on the right
    side of `full_threshold`.
    """
    low, high = price_range(item.item_kind, currency)
    kopiykas = int((line_total / item.qty * 100).to_integral_value(rounding=ROUND_HALF_UP))
    kopiykas = min(max(kopiykas - kopiykas % 10, _minor(low)), _minor(high))
    return item.model_copy(update={"price": Decimal(kopiykas) / 100})


def _excluded_ceiling(kinds: list[str], currency: str = "UAH") -> Decimal:
    """The most one non-covered line may cost, over the kinds available.

    The sizing bound for the loop below: at qty 1 no non-covered line can be repriced
    above this without leaving the range its item kind is plausible in.
    """
    return max(price_range(kind, currency)[1] for kind in kinds)


def _build_mixed_basket(
    rng: random.Random,
    *,
    category_id: str,
    vendor: dict,
    count: int,
    coverage_target: Decimal,
    language: str = "uk",
    currency: str = "UAH",
) -> list[LineItem]:
    """A basket drawn from both the covered and the excluded bucket of a category.

    This is how `partially_covered` by `mixed_items` is realized. The planner has already
    chosen that verdict; the builder's obligation is to produce evidence consistent with
    it, never the other way round — so the non-covered lines are priced toward the
    requested coverage ratio rather than left wherever the price draw put them.

    `ambiguous_items` are deliberately not drawn from: policy.yaml states no coverage
    answer for that bucket, and a `covered` flag for one would be this generator's
    invention rather than the policy's rule.
    """
    spec = category(category_id)
    covered_catalogue = spec["covered_items"]
    excluded_catalogue = spec["excluded_items"]
    vat_payer = vendor_is_vat_payer(vendor)

    covered = _draw_distinct_items(
        rng,
        sellable_kinds(covered_catalogue, vendor),
        covered_catalogue,
        count,
        covered=True,
        vat_payer=vat_payer,
        language=language,
        currency=currency,
    )
    if not covered:
        raise ValueError(f"category {category_id!r} produced no covered line")

    excluded_kinds = sellable_kinds(excluded_catalogue, vendor)
    if not excluded_kinds:
        raise ValueError(
            f"vendor {vendor['name']!r} (profile {vendor['profile']!r}) sells nothing "
            f"category {category_id!r} excludes, so it cannot carry a mixed basket — ask "
            "`vendor_can_carry` before choosing the vendor"
        )

    # covered / (covered + excluded) = target  =>  excluded = covered × (1 − target) / target
    def budget(items: list[LineItem]) -> Decimal:
        return line_items_total(items) * (1 - coverage_target) / coverage_target

    ceiling = _excluded_ceiling(excluded_kinds, currency)
    wanted = max(rng.choice(excluded_line_counts()), math.ceil(budget(covered) / ceiling))
    excluded = _draw_distinct_items(
        rng, excluded_kinds, excluded_catalogue,
        min(wanted, MAX_LINE_ITEMS - len(covered)), covered=False, vat_payer=vat_payer,
        language=language, currency=currency,
    )
    if not excluded:
        raise ValueError(f"category {category_id!r} produced no non-covered line")

    # A low coverage target can ask for more non-covered money than the kinds this vendor
    # sells will plausibly carry: every non-covered line is clamped into its own price
    # range, so a pharmacy basket of vitamins cannot be balanced by one 4 UAH tube of
    # cream. Shrinking the covered side is the honest way to reach the ratio; the
    # alternative, one absurdly priced non-covered line, would be a visible artifact in
    # the image.
    while len(covered) > 1 and budget(covered) > ceiling * len(excluded):
        covered.pop()

    excluded = [
        _repriced(item, budget(covered) / len(excluded), currency) for item in excluded
    ]

    items = covered + excluded
    # Otherwise every non-covered line is the last one on every mixed receipt, which is a
    # position a consumer could learn instead of learning to read the line.
    rng.shuffle(items)
    return items


def _draw_basket(
    rng: random.Random,
    *,
    document: str,
    category_id: str,
    vendor: dict,
    vat_payer: bool,
    covered_only: bool,
    coverage_target: Decimal | None,
    item_count: int | None,
    language: str = "uk",
    currency: str = "UAH",
) -> list[LineItem]:
    """What a document lists, drawn from the category's own buckets.

    🔴 ONE DRAW FOR EVERY CLASS THAT CARRIES A BASKET, and the reason is a label rather than
    tidiness: coverage is a property of WHAT WAS BOUGHT and not of the document that lists it,
    so a receipt and an invoice listing the same purchase must produce the same covered
    fraction. Two builders drawing baskets two ways would make the verdict depend on which
    class a claim happened to be given. The three callers said so in three copies of this
    block before it was extracted; the third copy is what made the duplication worth removing.

    `covered_only` is the label-first knob: the planner has already chosen the verdict, and the
    builder realizes it. For `covered` the basket is drawn from the category's covered items
    alone; for `partially_covered` by `mixed_items` the caller clears the flag and states the
    `coverage_target` the basket should come to; for the zero-coverage route to `rejected` the
    caller states a target of exactly ZERO and every line comes from `excluded_items` — the
    mirror of `covered_only`, and the branch that used to be refused while nothing could plan it.

    `document` names the class in the length message and changes nothing else — "a receipt
    carries 1 to 20 lines" is what a caller of that builder needs to read, and the bound itself
    is `MAX_LINE_ITEMS` for every class.

    ⚠️ THE ORDER OF THE DRAWS IS PART OF THE SEED'S MEANING. The line count is taken from `rng`
    before anything else here, exactly as it was in each copy; moving it would change every
    document of every existing corpus for a refactor that is meant to change nothing.

    `language` and `currency` are the platform receipt's two axes and neither is a draw:
    they select the template list and the price block, and their defaults keep every
    Ukrainian caller — and every existing seed — exactly where it was.
    """
    count = item_count if item_count is not None else rng.randint(2, 4)
    if not 1 <= count <= MAX_LINE_ITEMS:
        raise ValueError(f"{document} carries 1 to {MAX_LINE_ITEMS} lines, not {count}")

    if covered_only:
        if coverage_target is not None:
            raise ValueError(
                "coverage_target describes a mixed basket; covered_only=True already "
                "means every line is covered"
            )
        catalogue = category(category_id)["covered_items"]
        kinds = sellable_kinds(catalogue, vendor)
        if not kinds:
            raise ValueError(
                f"vendor {vendor['name']!r} (profile {vendor['profile']!r}) sells nothing "
                f"category {category_id!r} covers"
            )
        return _draw_distinct_items(
            rng, kinds, catalogue, count, covered=True, vat_payer=vat_payer,
            language=language, currency=currency,
        )

    if coverage_target is None:
        raise ValueError(
            "a mixed basket needs the coverage_target the planner chose — the builder "
            "realizes a verdict, it does not decide one"
        )
    if coverage_target == Decimal(0):
        # 🔴 THE ZERO-COVERAGE ROUTE TO `rejected`: every line drawn from the category's
        # `excluded_items`, none covered, so the covered amount comes to zero and
        # `policy_engine.verdict_for` answers `rejected` with no cause. The mirror of the
        # `covered_only` branch above rather than a degenerate mixed basket — there is no ratio
        # to price toward and no covered side to shrink, so `_build_mixed_basket` has nothing to
        # do here and its "produced no covered line" guard keeps meaning what it says.
        catalogue = category(category_id)["excluded_items"]
        kinds = sellable_kinds(catalogue, vendor)
        if not kinds:
            raise ValueError(
                f"vendor {vendor['name']!r} (profile {vendor['profile']!r}) sells nothing "
                f"category {category_id!r} excludes, so it cannot carry a zero-coverage "
                "basket — ask `vendor_can_carry` before choosing the vendor"
            )
        return _draw_distinct_items(
            rng, kinds, catalogue, count, covered=False, vat_payer=vat_payer,
            language=language, currency=currency,
        )
    if not Decimal(0) < coverage_target < Decimal(1):
        raise ValueError(
            f"a coverage target lies between 0 inclusive — the zero-coverage route to "
            f"`rejected`, handled above — and 1 exclusive, got {coverage_target}"
        )
    return _build_mixed_basket(
        rng,
        category_id=category_id,
        vendor=vendor,
        count=count,
        coverage_target=coverage_target,
        language=language,
        currency=currency,
    )


def _build_tax_lines(items: list[LineItem], *, vat_payer: bool) -> list[TaxLine]:
    """One row per VAT letter present, in the order the jurisdiction declares them.

    Ukrainian receipts print VAT-inclusive prices, so the tax is extracted from the
    gross rather than added on top: vat = gross − gross / (1 + rate/100).

    EMPTY FOR A SELLER THAT IS NOT REGISTERED FOR ПДВ, which is a document with no VAT block
    at all rather than one with an empty block. That case used to raise; the raise has become
    two narrower guards, because both directions are now a builder bug:

    * a line with NO letter on a REGISTERED seller's receipt — turnover would be left out of
      the tax block silently, which is the case the original raise was written for;
    * a line WITH a letter on a non-payer's receipt — the seller has assigned no rate group to
      anything, so the letter contradicts the document's own seller block.

    A jurisdiction with ``line_item_letter_position: none`` (ES, EU) prints no per-line letter
    at all; nothing calls this function for one.
    """
    if not vat_payer:
        lettered = [item.name for item in items if item.vat_letter is not None]
        if lettered:
            raise ValueError(
                f"line items {lettered!r} carry a ПДВ-літера on a receipt whose seller is not "
                "registered for ПДВ"
            )
        return []

    gross_by_letter: defaultdict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for item in items:
        if item.vat_letter is None:
            raise ValueError(
                f"line item {item.name!r} carries no ПДВ-літера on a registered payer's receipt"
            )
        gross_by_letter[item.vat_letter] += (item.qty * item.price).quantize(KOPIYKA)

    lines = []
    for letter in jurisdiction("UA")["vat_letters"]:
        gross = gross_by_letter.get(letter)
        if gross is None:
            continue
        rate = Decimal(str(vat_rate_for_letter(letter, "UA")))
        vat = (gross - gross / (1 + rate / 100)).quantize(KOPIYKA)
        lines.append(TaxLine(letter=letter, rate=vat_rate_for_letter(letter, "UA"),
                             gross=gross, vat=vat))
    return lines


_DIGITS = "0123456789"
_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The identifier patterns of config/fiscal-rules.yaml are a sequence of character classes,
# each with a length or a length range: `[A-Za-z0-9]{11}`, `[0-9]{4,6}`, `[A-Z]{2}[0-9]{8}`.
# Parsed rather than restated, so that the alphabet and the length are written down once, in
# the file that states the format. An `11` in code beside a `{11}` in config is two sources of
# truth about one number, and the first edit to either makes them disagree silently.
_PATTERN_SEGMENT_RE = re.compile(r"\[(?P<cls>[^\]]+)\]\{(?P<low>\d+)(?:,(?P<high>\d+))?\}")

# Character class as written in the configuration -> the characters to draw from. Keyed by
# the class text verbatim rather than interpreted, because a class this drawer does not know
# has to fail loudly: silently drawing from a wrong alphabet would produce an identifier the
# configuration says is impossible, and the fullmatch below would be the only thing to notice.
_PATTERN_ALPHABETS = {
    "A-Za-z0-9": _ALNUM,
    "0-9": _DIGITS,
    "A-Z": _UPPERCASE,
}


def _draw_from_pattern(rng: random.Random, pattern: str) -> str:
    """A value matching a configured identifier pattern.

    A length RANGE is drawn from, because a range in the configuration means the length
    genuinely varies: a hardware РРО's receipt counter reads four digits early in the life of
    the register and six later, and a corpus that only ever printed one of those widths would
    teach a consumer that width.

    The result is matched against the whole pattern before it is returned. That check is what
    makes the pattern load-bearing rather than decorative — narrow a class in config without
    teaching this drawer the new one, and the run fails instead of printing a value the file
    forbids.
    """
    segments = list(_PATTERN_SEGMENT_RE.finditer(pattern))
    if not segments or "".join(m.group(0) for m in segments) != pattern:
        raise ValueError(
            f"pattern {pattern!r} in config/fiscal-rules.yaml is not a sequence of character "
            "classes with lengths, which is all this drawer handles"
        )

    value = ""
    for segment in segments:
        alphabet = _PATTERN_ALPHABETS.get(segment["cls"])
        if alphabet is None:
            raise ValueError(
                f"pattern {pattern!r} in config/fiscal-rules.yaml uses the character class "
                f"[{segment['cls']}], which content_builder._PATTERN_ALPHABETS does not know"
            )
        low = int(segment["low"])
        high = int(segment["high"]) if segment["high"] else low
        value += "".join(rng.choice(alphabet) for _ in range(rng.randint(low, high)))

    if not re.fullmatch(pattern, value):
        raise ValueError(
            f"drew {value!r} for pattern {pattern!r} in config/fiscal-rules.yaml — the "
            "alphabets this builder draws from no longer match the pattern's classes"
        )
    return value


def _tax_line_format(rules: dict, form: str | None) -> str:
    """The format string for a tax summary row, by jurisdiction and by chosen form.

    A jurisdiction declares EITHER a single `tax_line_label_format` — which is every jurisdiction
    but Ukraine, where the variation has been observed — OR a map of named forms under
    `tax_line_label_forms`, one of which the document chose. Two shapes rather than one because
    only one jurisdiction has evidence of variation, and giving the others a one-entry map would
    state a choice nobody has observed them making.
    """
    forms = rules.get("tax_line_label_forms")
    if forms is None:
        return rules["tax_line_label_format"]
    if form not in forms:
        raise ValueError(
            f"{form!r} is not a tax-line form this jurisdiction declares; it has {sorted(forms)}"
        )
    return forms[form]


def _draw_tax_line_form(rng: random.Random, rules: dict, medium: Medium) -> str | None:
    """Which of the 👁 observed VAT-row forms this document prints.

    🔴 THE MEDIUM DECIDES, AND ASYMMETRICALLY. Paper takes the equals form and nothing else — 👁 two
    independent installations, no counterexample. Electronic draws between both, because there is
    ONE electronic observation and one observation cannot support a rule; drawing is what "no
    evidence either way" looks like once it has to be written down.

    UNIFORM AMONG THE PERMITTED FORMS, and the uniformity is a consequence of the list rather than a
    share somebody chose: `rng.choice` over what the medium allows. A weighted draw would be a claim
    about how often each form occurs electronically, which one observation cannot support.

    `None` where the jurisdiction declares no forms — every one but Ukraine.
    """
    forms = rules.get("tax_line_label_forms")
    if forms is None:
        return None
    allowed = rules["tax_line_forms_by_medium"].get(medium.value)
    if not allowed:
        raise ValueError(
            f"config/fiscal-rules.yaml declares no tax-line forms for the {medium.value!r} "
            f"medium; it knows {sorted(rules['tax_line_forms_by_medium'])}"
        )
    return rng.choice(allowed)


def build_prro_receipt(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    address: str = "м. Київ",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
    registrar: str = "prro",
    capture: Capture,
) -> PrroReceipt:
    """Build one Ukrainian fiscal receipt — from a ПРРО or from a classic hardware РРО.

    ``covered_only`` is the label-first knob: the planner has already chosen the verdict,
    and the builder realizes it. For ``covered`` the basket is drawn from the category's
    covered items alone; for ``partially_covered`` by ``mixed_items`` the caller clears
    the flag and states the ``coverage_target`` the basket should come to.

    ``vendor`` is an entry of `config/vendors.json`: its ``profile`` decides which item kinds may
    appear on the receipt, its ``vat_payer`` decides whether the seller block prints a ПН line
    beside the ІД line it always prints and whether the document has a VAT block, and its
    ``legal_form`` decides how the name is printed and which register both identifiers come from.
    Ask `vendor_can_carry` before choosing one for a mixed basket.

    ``identity`` carries those two identifiers, drawn once for the claim. Required on this builder
    as on the other three even though a fiscal receipt is a whole claim by itself: an optional
    parameter here would leave one class deciding who a seller is by a route of its own, and the
    day a receipt is paired with anything it would be the class that got missed.

    ``registrar`` names a key of ``receipt.registrars`` in config/fiscal-rules.yaml — ``prro``
    for the software register, ``rro`` for the classic hardware one. IT DECIDES THE FISCAL
    IDENTITY AND NOTHING ELSE: which prefix the fiscal number carries, whether a «ЗН» factory
    serial is printed at all, whether the online/offline marker appears, which of the two
    receipt-number formats is used, and which population the maker's name is drawn from. The
    basket, the seller and the money are the same document either way, which is why one builder
    produces both rather than two builders sharing everything but twenty lines.

    THE PAPER WIDTH IS NOT HERE. It is the one difference that is purely visual, so it lives in
    the stylesheet of each template, and two templates may therefore share a registrar.

    ``capture`` IS NOT THE DEGRADER'S BUSINESS ALONE. It decides the document's MEDIUM, and 👁 the
    VAT summary row takes one form on paper and either of two electronically — so the channel a
    document will reach a verifier on has to be known while the document is BUILT, not only while
    it is degraded. The assembler decides it once and passes the same value to both.

    🔴 IT HAS NO DEFAULT, DELIBERATELY. It had one — `Capture.SCREENSHOT` — which equalled the only
    channel the assembler produces, so a caller that stopped passing it produced IDENTICAL output
    and no test could tell. That was found by a mutation surviving: removing the assembler's
    `capture=` changed nothing observable, because the default silently supplied the same value.
    A silent fallback that coincides with the live value is not a convenience, it is a wiring break
    waiting to be invisible — so the parameter is required and a caller that omits it fails loudly.
    """
    rules = jurisdiction("UA")
    receipt_rules = rules["receipt"]
    vat_payer = vendor_is_vat_payer(vendor)

    # -- what was bought
    items = _draw_basket(
        rng,
        document="a receipt",
        category_id=category_id,
        vendor=vendor,
        vat_payer=vat_payer,
        covered_only=covered_only,
        coverage_target=coverage_target,
        item_count=item_count,
    )
    total = line_items_total(items)

    # -- who sold it
    #
    # TWO possible identifier lines, and a registered payer prints one MORE than a non-payer
    # rather than a different one. The ІД is the seller's identification code and its register
    # follows the legal form — a ФОП has no ЄДРПОУ at all, so an eight-digit code there would
    # put an identifier on the document that no register could resolve to the seller named
    # beside it. The ПН exists only for a registered payer, and is derived from the ІД rather
    # than drawn independently, so the two lines cannot contradict each other. See `identifiers`
    # in config/fiscal-rules.yaml for the sources and for what is not settled.
    # Both identifiers come from `identity`, drawn once for the claim. A fiscal receipt is a whole
    # claim on its own today, so nothing of this one is compared against a second document — the
    # parameter is here so that ONE mechanism decides who a seller is, rather than three builders
    # deciding it three ways and the fourth being fixed later.
    is_sole_trader = vendor["legal_form"] == _SOLE_TRADER
    id_code_rules = rules["identifiers"]["rnokpp" if is_sole_trader else "edrpou"]
    vat_number_rules = rules["identifiers"]["vat_number"]

    seller = Seller(
        name=vendor["name"],
        legal_form=vendor["legal_form"],
        address=address,
        vat_payer=vat_payer,
        tax_code=identity.tax_code,
        tax_code_label=id_code_rules["label"],
        vat_number=identity.vat_number,
        vat_number_label=vat_number_rules["label"],
    )

    # -- how it was paid for
    acquiring_rules = rules["acquiring_block"]
    acquiring_examples = {f["key"]: f for f in acquiring_rules["fields"]}
    acquiring = Acquiring(
        # Drawn rather than fixed. The acquirer used to be the single `example` string of
        # fiscal-rules.yaml, which put the same bank name on every receipt in the dataset —
        # a printed field with one value teaches a consumer the value, not the field.
        acquirer=rng.choice(acquirers("UA")),
        terminal_id=f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(0, 10**7 - 1):07d}",
        # `[0]` and not a draw: every receipt in the corpus is a sale, so «ПОВЕРНЕННЯ» is
        # declared in config and printed on nothing. Stated at the config site too.
        operation=acquiring_examples["operation"]["values"][0],
        card_masked=f"{rng.randint(0, 9999):04d}XXXXXXXX{rng.randint(0, 9999):04d}",
        auth_code=f"{rng.randint(0, 999_999):06d}",
        rrn=f"{rng.randint(0, 10**12 - 1):012d}",
    )

    # -- the fiscal identity of the document
    #
    # Which lines appear here is the whole of what `registrar` decides. ⚠️ «ЗН» and «ФН» are
    # not a pair: a ПРРО prints the fiscal number alone, a hardware РРО prints both, and the
    # set is read from config rather than assembled here — see `registrars` in
    # config/fiscal-rules.yaml for the observations behind each line.
    try:
        registrar_rules = receipt_rules["registrars"][registrar]
    except KeyError:
        raise ValueError(
            f"config/fiscal-rules.yaml declares no registrar {registrar!r} for UA; it knows "
            f"{sorted(receipt_rules['registrars'])}"
        ) from None

    # The maker printed at the foot, drawn from the pool config/fiscal-rules.yaml names for this
    # kind of register — a publicly marketed ПРРО provider or a manufacturer from the published
    # state register of cash registers. Suffix and name arrive together so the two cannot name
    # different makers; see `config.fiscal_makers`.
    suffix, provider_name = rng.choice(fiscal_makers(registrar_rules["maker_pool"], "UA"))
    receipt_number = _draw_from_pattern(
        rng,
        receipt_rules["receipt_number"][registrar_rules["receipt_number_format"]]["pattern"],
    )
    fiscal_number_rules = rules["identifiers"]["fiscal_device_number"]
    fiscal_device_number = "".join(
        rng.choice(_DIGITS) for _ in range(fiscal_number_rules["length"])
    )
    device_serial_rules = rules["identifiers"]["device_serial"]
    device_serial = (
        _draw_from_pattern(rng, device_serial_rules["pattern"])
        if registrar_rules["prints_device_serial"]
        else None
    )

    return PrroReceipt(
        seller=seller,
        issued_at=issued_at,
        title=f"{receipt_rules['title']} {suffix}".strip(),
        mode_marker=(
            receipt_rules["mode_markers"][0]
            if registrar_rules["prints_mode_marker"]
            else None
        ),
        receipt_number=receipt_number,
        fiscal_device_number=fiscal_device_number,
        fiscal_number_label=registrar_rules["fiscal_number_label"],
        device_serial=device_serial,
        device_serial_label=device_serial_rules["label"],
        provider_name=provider_name,
        line_items=items,
        total=total,
        # Zero this version, and the reason is the policy's silence rather than the arithmetic
        # — see `PrroReceipt.amount_due`. The lines are printed all the same, because 👁 all
        # four appear on the observed receipt.
        discount=Decimal(0),
        rounding=Decimal(0),
        amount_in_words=amount_in_words_uk(total),
        tax_lines=_build_tax_lines(items, vat_payer=vat_payer),
        # `[0]` for the same reason as `operation` above: every receipt is card-paid, so
        # «ГОТІВКА» is unreachable. A cash receipt is not this label with the acquiring block
        # left in place — 👁 there is no acquiring block on one — so it is a template, not a
        # draw here.
        payment_method=acquiring_rules["payment_method_labels"][0],
        acquiring=acquiring,
        decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
        qr_payload=rules["qr"]["fiscal_payload"].format(
            receipt_number=receipt_number,
            date=issued_at,
            time=issued_at,
            fiscal_device_number=fiscal_device_number,
            total=total,
        ),
        footer=receipt_rules["footer"],
        # 👁 The VAT row's form follows the MEDIUM this document will reach a verifier on, and only
        # a registered payer has such a row at all. `None` for a non-payer is the same statement
        # its empty `tax_lines` makes: there is no tax block to take a form.
        vat_row_form=(
            _draw_tax_line_form(rng, rules, capture.medium) if vat_payer else None
        ),
    )


# =============================================================================
# Bank payment confirmation
# =============================================================================
#
# A DIFFERENT DOCUMENT CLASS AND NOT A VARIANT OF THE RECEIPT ABOVE: a bank issues it, it
# carries no fiscal identity, and 👁 8 of 8 observed confirmations list NO ITEMS at all. What it
# proves is therefore only that money moved — policy.yaml's `document_evidence` gives the type
# `proves_subject: false`, and 👁 0 of 7 payment purposes name what was bought, which is the
# strongest confirmation of that entry in this repository.
#
# ONE ARCHETYPE WITH A CONDITIONAL BLOCK, NOT TWO. An earlier reading split the class in two — a
# quittance carrying a purpose against a card slip carrying an authorization code — and 👁 3 of 8
# documents carry a masked card AND an authorization code AND a purpose at once, which refutes
# it. What varies is HOW THE PAYMENT WAS INITIATED; see `initiation` in
# config/fiscal-rules.yaml.


def iban_check_digits(country: str, bban: str) -> str:
    """The two check digits of an IBAN, per 📄 ISO 13616 / ISO 7064 mod 97-10.

    The published algorithm: move the country code and the two check positions to the end,
    replace each letter by its position in the alphabet plus 9 (A = 10 … Z = 35), read the result
    as one integer and take it mod 97; the check digits are 98 minus that.

    Computed rather than drawn, for the reason every identifier in this module is: an account
    number that fails its own published checksum is a broken invariant, and a broken invariant in
    this repository has to be a labelled choice of a fraud archetype rather than a side effect of
    a generator that did not bother.
    """
    rearranged = f"{bban}{country}00"
    digits = "".join(
        str(int(char, 36)) if char.isalpha() else char for char in rearranged.upper()
    )
    if not digits.isdigit():
        raise ValueError(f"an IBAN body is alphanumeric, got {bban!r}")
    return f"{98 - int(digits) % 97:02d}"


def is_valid_iban(value: str) -> bool:
    """Whether a string is an IBAN whose check digits agree with its body."""
    if len(value) < 5 or not value[:2].isalpha() or not value[2:4].isdigit():
        return False
    return iban_check_digits(value[:2], value[4:]) == value[2:4]


def generate_iban(rng: random.Random, bank_code: str, country: str = "UA") -> str:
    """A checksum-correct IBAN of this jurisdiction, on this bank's code.

    The length and the bank-code length come from `identifiers.iban_format` in
    config/fiscal-rules.yaml rather than from literals here: an IBAN's length is fixed per
    country by the published registry, and that is a fact about the jurisdiction — which is why
    the rule sits beside the МФО rather than under the first document class that printed one.

    ⚠️ `country` IS THE POOL THE ACCOUNT BELONGS TO AND NOT ALWAYS A COUNTRY. `EU` is a pool of
    cross-border sellers, and the structure it draws is stated where the pool is.
    """
    rules = jurisdiction(country)["identifiers"]["iban_format"]
    if len(bank_code) != rules["bank_code_length"]:
        raise ValueError(
            f"a {country} IBAN carries a {rules['bank_code_length']}-digit bank code, got "
            f"{bank_code!r}"
        )
    account_length = rules["length"] - 4 - len(bank_code)
    account = "".join(rng.choice(_DIGITS) for _ in range(account_length))
    bban = bank_code + account
    return f"{rules['country']}{iban_check_digits(rules['country'], bban)}{bban}"


def luhn_check_digit(body: str) -> int:
    """The final digit a card number would need for its 📄 published Luhn checksum to pass.

    Here to be AVOIDED rather than satisfied — see `unissuable_card_number`.
    """
    if not body.isdigit():
        raise ValueError(f"a card number body is digits, got {body!r}")
    total = 0
    for index, char in enumerate(reversed(body)):
        digit = int(char)
        if index % 2 == 0:  # the position the check digit will make even from the right
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10


def passes_luhn(number: str) -> bool:
    """Whether a number satisfies the Luhn checksum every payment card satisfies."""
    return number.isdigit() and luhn_check_digit(number[:-1]) == int(number[-1])


def unissuable_card_number(rng: random.Random, length: int = 16) -> str:
    """Digits that LOOK like a card number and cannot be one, by construction.

    🔴 The reason is publication, not realism. 👁 One observed confirmation prints a recipient's
    card with no masking at all, so this generator must be able to print sixteen visible digits —
    and sixteen digits drawn freely would satisfy the Luhn checksum one time in ten, at which
    point a published image would carry a string that could be somebody's actual card.
    Deliberately failing a published checksum makes "this is not a card number" a verifiable
    property of every such image rather than an assurance in a README.

    The masked schemes use it too: what stays visible is drawn from the same digits, so the head
    and tail of a masked number belong to a number that could not exist either.
    """
    body = "".join(rng.choice(_DIGITS) for _ in range(length - 1))
    correct = luhn_check_digit(body)
    return body + str((correct + rng.randint(1, 9)) % 10)


@dataclass(frozen=True)
class Party:
    """One side of a payment, as a bank confirmation prints it.

    Every field is nullable because 👁 real confirmations leave them out in TWO mechanically
    different ways, and the difference matters to whoever scores an extraction: a HYPHEN printed
    as the value (`name="-"`) is a value an extractor returns as the string "-", while a caption
    with nothing under it returns nothing at all. Collapsing both into "field absent" would make
    accuracy on empty fields unmeasurable.

    `code` is a РНОКПП (10 digits) or a ЄДРПОУ (8), and ⚠️ the length distinguishes the KIND OF
    CODE rather than the kind of party: a sole trader is a business with a ten-digit code, so
    "ten digits means a private individual" is false.
    """

    name: str
    # THE BARE TRADING NAME, which is what the LABEL carries — `name` above is what the document
    # PRINTS. config/labelling-schema.yaml makes the bare name authoritative under
    # `normalization.party_name` ("this file makes the bare name authoritative"), because a legal
    # form is a property of the seller's registration rather than of the merchant identity a claim
    # is about. For a natural person the two coincide: there is no legal form to strip.
    #
    # 🔴 IT EXISTS BECAUSE THE TWO HAD SILENTLY DIVERGED. This class labelled the PRINTED form —
    # «ТОВ «Ключ»» — while a receipt of the same seller labelled «Ключ», so two documents of one
    # claim carried two strings for one merchant. Nothing caught it: the contract's comparison
    # rule strips the legal form, so both compare equal to a consumer, and only a cross-document
    # test asking whether a claim agrees with itself could see it.
    trade_name: str
    code: str | None
    account: str | None
    bank: str | None


@dataclass(frozen=True)
class PaymentConfirmation:
    """One Ukrainian bank payment confirmation, complete but not yet rendered.

    THE THREE AMOUNTS ARE THE HEART OF THIS CLASS, and 👁 0 of 8 observed documents carry only
    one:

    * ``transfer`` — 📄 the amount of the payment OPERATION, which the National Bank's instruction
      on non-cash settlements makes a mandatory requisite. **This is the claim's amount.**
    * ``fee`` — the bank's charge. 📄 Absent from the requisite list entirely, and it pays for a
      banking service rather than for anything a benefit category covers, so it is never
      reimbursable. 👁 Non-zero on 3 of 8, and generated at about that rate: keeping it at zero
      would make the field discriminate nothing.
    * ``total_charged`` — their sum, DERIVED rather than stored so the three cannot drift.
      👁 Printed on 1 of 8, and where it is printed it is the LARGEST NUMBER ON THE PAGE.

    🔴 That last point is the divergence this archetype exists to make measurable. An extractor
    that takes the most salient figure returns ``total_charged`` while the oracle, following the
    norm, expects ``transfer`` — and the gap between them is exactly the fee. It is now visible
    whether a system resists the salient number instead of being invisible.

    ``initiation`` names a key of ``payment_confirmation.initiation`` in
    config/fiscal-rules.yaml and decides the CONDITIONAL BLOCK: whether a card and an
    authorization code are printed, whether there is a payment purpose at all, and whether the
    payer is identified. It is the axis of variation because 👁 the documents vary along it — not
    by family of document, which was the reading the observation refuted.
    """

    bank_name: str
    bank_code: str
    title: str
    document_code: str
    issued_at: datetime
    # The caption printed beside the date. 👁 Ten captions over six concepts appear on real
    # confirmations; only the three the labelling contract accepts as the payment date are ever
    # printed here — see `payment_date` in config/labelling-schema.yaml and `period.payment_date`
    # in config/policy.yaml for the concept itself.
    date_caption: str
    payer: Party
    payee: Party
    initiation: str
    transfer: Decimal
    fee: Decimal
    # 🔴 WHAT THE THREE AMOUNTS ARE IN, and it is a field rather than a constant because a
    # Ukrainian bank executes instructions in foreign currency too. The label reads it; the page
    # states it in the captions and in the words. ⚠️ THE FEE IS IN THE SAME CURRENCY as the
    # transfer — one page, one currency, exactly as one claim is — so nothing here has to say
    # which of the three a code applies to.
    currency: str
    # 👁 Four captions for the amount and three for the fee. Carried per document because a
    # consumer keying on one string reads a minority of real documents.
    amount_caption: str
    fee_caption: str
    prints_total: bool
    amount_in_words: str | None
    amount_in_words_caption: str
    purpose: str | None
    # The рахунок number `purpose` names, where it names one — the structured copy the label
    # ships as `cites_document_no`. Computed beside the fill (`_fill_reference_traced`), never
    # parsed back out of the text.
    cites_document_no: str | None
    auth_code: str | None
    card_masked: str | None
    terminal_label: str | None
    terminal_value: str | None
    signature_path: str | None
    signatory_post: str | None
    electronic_note: str | None
    verification_footer: str | None
    qr_payload: str | None
    decimal_separator: str

    @property
    def total_charged(self) -> Decimal:
        """Everything that left the payer's account: the transfer plus the bank's fee.

        Derived for the reason `PrroReceipt.amount_due` is derived and `_vat_number` is derived:
        two numbers that must agree should not be two numbers. Unlike ``amount_due`` this one
        genuinely DIFFERS from the amount it is derived from — 👁 on about a third of documents —
        so it discriminates, and a consumer may score it.
        """
        return (self.transfer + self.fee).quantize(KOPIYKA)

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted.

        Formatting lives here for the same reason it does on the receipt: a jurisdiction's number
        format is decided once rather than in every template that shows an amount.
        """
        rules = jurisdiction("UA")
        block = rules["payment_confirmation"]
        return {
            "bank_name": self.bank_name,
            "bank_code": self.bank_code,
            "bank_code_label": block["parties"]["bank_code_label"],
            "title": self.title,
            "document_code": self.document_code,
            "document_code_label": block["document_code"]["label"],
            "date_caption": self.date_caption,
            "date": self.issued_at.strftime(rules["date_format"]),
            # This class's own `time_format`, not the jurisdiction's: the shared one separates
            # with dashes, which is 👁 a till-printer quirk observed on receipts.
            "time": self.issued_at.strftime(block["time_format"]),
            "party_labels": block["parties"],
            "payer": self.payer,
            "payee": self.payee,
            "amount_caption": self.amount_caption,
            "amount": self._amount(self.transfer),
            "fee_caption": self.fee_caption,
            "fee": self._amount(self.fee),
            "total_caption": block["total_caption"],
            "total_charged": self._amount(self.total_charged) if self.prints_total else None,
            "amount_in_words": self.amount_in_words,
            "amount_in_words_caption": self.amount_in_words_caption,
            "purpose_label": block["purpose_label"],
            "purpose": self.purpose,
            "auth_code": self.auth_code,
            "auth_code_label": block["auth_code"]["label"],
            "card_masked": self.card_masked,
            "card_label": block["card"]["label"],
            "terminal_label": self.terminal_label,
            "terminal_value": self.terminal_value,
            "stamp_caption": block["signature"]["stamp_caption"],
            "signature_label": block["signature"]["signature_label"],
            "signature_path": self.signature_path,
            "signatory_post": self.signatory_post,
            "electronic_note": self.electronic_note,
            "verification_caption": block["verification_footer"]["present_caption"],
            "verification_footer": self.verification_footer,
            "qr_caption": block["qr"]["marketing_caption"],
            "qr_payload": self.qr_payload,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this confirmation.

        ``amount`` IS THE TRANSFER AND NOT THE TOTAL. Two supports, and they agree: 📄 the
        instruction on non-cash settlements calls the amount of the operation the requisite, and a
        fee buys a banking service rather than anything a benefit category covers, so the policy
        does not reach it. ``total_charged`` and ``fee`` are labelled beside it, which is what
        makes the disagreement with a salience-following extractor measurable.

        ``amount_due`` stays ``None``: it is a receipt requisite (ДО СПЛАТИ) and this document has
        no such line. Reusing it for «Загальна сума» would give one key two meanings, which is the
        failure config/labelling-schema.yaml exists to prevent.

        ``line_items`` is EMPTY rather than omitted — the document lists nothing, and that empty
        list is the statement.
        """
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.PAYMENT_CONFIRMATION,
            language="uk",
            currency=self.currency,
            amount=self.transfer,
            fee=self.fee,
            total_charged=self.total_charged,
            date=self.issued_at.date(),
            # The BARE trade name, not the printed one — see `Party.trade_name`.
            counterparty=self.payee.trade_name,
            payer=self.payer.name,
            payment_purpose=self.purpose,
            cites_document_no=self.cites_document_no,
            document_code=self.document_code,
            auth_code=self.auth_code,
            line_items=[],
            # 👁 2 of 8 carry a QR and BOTH are marketing — an application download. So a QR on
            # this class is positive evidence that a QR says nothing about fiscality, and
            # `qr_is_fiscal` is False whether or not one is printed.
            has_qr=self.qr_payload is not None,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            # From the RENDERER, like the boxes: neither is decided by the content class, and both
            # describe the page that was produced from it.
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def _grouped(value: str, size: int, separator: str) -> str:
    """A digit run broken into groups, as a document prints it."""
    if size <= 1 or not separator:
        return value
    return separator.join(value[i : i + size] for i in range(0, len(value), size))


def _draw_document_code(rng: random.Random, block: dict) -> str:
    """The bank's own number for the document — 👁 8/8, 📄 mandatory, and the deduplication key.

    Three formats are observed and all three are drawn from: a consumer validating the field by
    shape needs every one of them, and a corpus printing a single shape would teach the shape.
    """
    spec = rng.choice(block["document_code"]["formats"])
    groups = [_draw_from_pattern(rng, spec["pattern"]) for _ in range(spec["groups"])]
    return spec["separator"].join(groups)


def _draw_card(rng: random.Random, block: dict) -> str:
    """A card number under one of the 👁 six observed masking schemes.

    ⛔ NO NORM WAS FOUND for masking, in either the instruction on non-cash settlements or the
    regulation on payment instruments, and 🔴 the spread confirms the absence from the opposite
    direction: one observed document prints a recipient's card unmasked. So the scheme is drawn,
    the unmasked one included, and a consumer whose pattern is "six digits, asterisks, four" is
    matching a minority of the field.
    """
    card = block["card"]
    scheme = rng.choice(card["masking_schemes"])
    digits = unissuable_card_number(rng)

    head, tail = scheme["head"], scheme["tail"]
    hidden = len(digits) - head - tail
    if hidden < 0:
        raise ValueError(
            f"masking scheme {scheme!r} in config/fiscal-rules.yaml reveals more digits than a "
            f"card number has ({len(digits)})"
        )
    shown = digits[:head] + scheme["mask"] * hidden + (digits[-tail:] if tail else "")
    if scheme.get("grouped"):
        shown = _grouped(shown, 4, " ")
    if scheme.get("name_scheme"):
        shown = f"{shown} ({rng.choice(card['schemes'])})"
    return shown


def _draw_signature_path(rng: random.Random) -> str:
    """An SVG path for a handwritten signature — 👁 present beside the stamp on 7 of 8.

    Drawn per document rather than fixed, because a single path would be a constant on every
    image of the class and a constant is something a model learns instead of learning the field.
    Deterministic under the seed like everything else here: the control points come from the
    caller's generator.

    Geometry only. It spells no name and is not meant to: what the layout carries is that a
    signature is there, and a legible name in script would be a personal name printed on a
    published image for no gain.
    """
    points = [(0, 22)]
    x = 0
    for _ in range(4):
        x += rng.randint(16, 26)
        points.append((x, rng.randint(2, 30)))
    curves = " ".join(
        f"Q {px - 8} {rng.randint(0, 32)} {px} {py}" for px, py in points[1:]
    )
    return f"M 0 22 {curves}"


def _fill_reference(
    rng: random.Random,
    template: str,
    rules: dict,
    at: datetime,
    cites: DocumentReference | None,
) -> str:
    """A purpose line with the document it names filled in.

    `cites` is the claim's own subject document, and where it is given the line names THAT
    invoice — its number and the date it bears. Where it is not, the line names a document outside
    the claim, which is the honest case for every ordinary statement row and for a payment
    document built on its own.

    ⚠️ `{delivery_note_no}` IS NEVER FILLED FROM `cites`. A ВН is a delivery note and no claim
    holds one, so its number stays drawn however the caller was called — see the note beside the
    templates in config/generation.yaml for why the two placeholders are separate.

    🔴 THE SAME THREE VALUES ARE DRAWN WHETHER OR NOT `cites` IS GIVEN, and one of the three is
    then discarded. A branch that skipped the draw would make the LENGTH of the run's draw depend
    on whether a claim happened to have a subject document, so every later value in the whole run
    would shift with it — the same reason `build_bank_statement` nudges a colliding amount instead
    of redrawing it.
    """
    return _fill_reference_traced(rng, template, rules, at, cites)[0]


def _fill_reference_traced(
    rng: random.Random,
    template: str,
    rules: dict,
    at: datetime,
    cites: DocumentReference | None,
) -> tuple[str, str | None]:
    """`_fill_reference`, plus WHICH рахунок number the line ended up naming — or `None` where the
    template names none (a generic formula) or names a ВН, whose number belongs to no document of
    the claim.

    The second value is what the label field `cites_document_no` carries, so it is computed here,
    beside the fill, rather than re-parsed out of the finished text — an instrument reading its
    own output back is the failure mode `tools/cross_document_audit.py` exists to catch, not one
    to build in. Same draws, same order, so a caller switching to the traced form moves no seed.
    """
    drawn_number = f"{rng.randint(1, 9999)}"
    drawn_date = at - timedelta(days=rng.randint(0, 20))
    delivery_note_no = f"{rng.randint(1, 9999)}"
    number, issued_at = (
        (cites.number, cites.issued_at) if cites else (drawn_number, drawn_date)
    )
    text = template.format(
        invoice_no=number,
        invoice_date=issued_at.strftime(rules["date_format"]),
        delivery_note_no=delivery_note_no,
    )
    return text, (number if "{invoice_no}" in template else None)


def _money_caption(
    rng: random.Random, captions: list[str], block: dict, *, currency: str, domestic: bool
) -> str:
    """One caption for an amount, saying which currency the figure beside it is in.

    Domestic: the observed pool, drawn from as it always was. Foreign: the same pool less the
    captions that name hryvnias, with the ISO code in brackets — the form those captions
    themselves use. ONE DRAW EITHER WAY, so the currency of a document does not change how many
    values a run takes from the generator.
    """
    if domestic:
        return rng.choice(captions)
    neutral = [c for c in captions if c not in block["captions_naming_the_domestic_currency"]]
    return block["foreign_currency_caption_format"].format(
        caption=rng.choice(neutral), code=currency
    )


def build_payment_confirmation(
    rng: random.Random,
    *,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    payer_name: str,
    payer_tax_id: str,
    amount: Decimal | None = None,
    initiation: str | None = None,
    cites: DocumentReference | None = None,
    must_cite: bool = False,
    country: str = "UA",
    currency: str | None = None,
) -> PaymentConfirmation:
    """Build one Ukrainian bank payment confirmation.

    🔴 ``currency`` IS THE ONE AXIS THAT MOVES, and everything else about the page holds still.
    ``None`` means the jurisdiction's own — every claim of this corpus until a foreign one was
    planned — and naming another one states a transfer executed in it. 📄 The requisites do not
    change: the instruction on non-cash settlements names an amount of the operation, not an amount
    in hryvnias. What changes is what the page must SAY: the caption carries the code (see
    `captions_naming_the_domestic_currency` in config/fiscal-rules.yaml) and the words spell euros,
    so a label recording EUR is readable off the image twice over.

    ⛔ A CARD OPERATION IS REFUSED IN A FOREIGN CURRENCY, and it is refused rather than quietly
    turned into a transfer. 👁 The card modes print an authorization code and a masked card — a
    domestic acquiring operation — and what a Ukrainian bank executes against a foreign
    beneficiary's account is a transfer by account details. A caller asking for both has asked for
    a document this repository has no evidence for.

    ``must_cite`` is the label-first knob of the `subject` axis: the plan has decided this
    payment's purpose NAMES the рахунок in ``cites``, so the initiation mode is drawn among those
    that print a purpose at all and the formula among those that cite an invoice by number.
    Without it either draw may honestly produce a page that cites nothing — the ordinary case —
    and a claim built to disagree about its subject would disagree about nothing. It requires
    ``cites``: a forced citation of no document is not a page anything plans.

    ``vendor`` is an entry of config/vendors.json, resolved — it is the PAYEE, and the same
    instance is passed to every document of a claim so two documents cannot name two firms. Its
    ``profile`` is not consulted: this document lists nothing, so what the payee sells cannot be
    read off it, which is the whole reason the type proves no subject.

    ``identity`` is that payee's `PartyIdentity`, and it is REQUIRED for the reason ``capture`` is
    required on the receipt builder: a default would draw a valid-looking code and IBAN, the
    document would render, and the only symptom would be that it disagreed with the invoice beside
    it — which is exactly the defect this parameter exists to remove. It is not drawn here because
    it is a property of the CLAIM's payee and not of this page.

    ``cites`` is the claim's subject document, when it has one. Given, the purpose line names that
    invoice; omitted, it names a document outside the claim, which is what a confirmation built on
    its own honestly does.

    ``payer_name`` and ``payer_tax_id`` come from the persona. This is the first archetype that
    prints a persona's own name, which is why the surname pool was narrowed to a published
    high-frequency set before it was ever printed — see `personal_names` in
    config/generation.yaml.

    ``amount`` IS THE TRANSFER — the value the label's `amount` field takes, which on this class is
    📄 the amount of the operation and not the largest number printed. It is drawn when not given.
    THE PARAMETER IS SPELLED THE SAME ON EVERY PAYMENT BUILDER, while the dataclass field below
    keeps its own name `transfer`: the assembler hands one claim's amount to whichever
    payment-proving archetype the plan chose, and a keyword that differed per builder would be a
    second table saying how to call each one. Uniformity belongs at the call boundary; the
    distinction between a transfer and a total belongs in the model, where it means something.

    ``initiation`` is drawn when not given. Both are parameters so a test can pin the conditional
    block instead of hunting for a seed that produces it, and so that the three initiation modes can
    each be rendered and looked at.
    """
    rules = jurisdiction(country)
    block = rules["payment_confirmation"]
    currency = currency or rules["currency"]
    domestic_currency = currency == rules["currency"]

    if must_cite and cites is None:
        raise ValueError(
            "must_cite forces the purpose line to name the document in `cites`, and none was "
            "given — a forced citation of no document is not a page anything plans"
        )

    try:
        mode = block["initiation"][initiation] if initiation else None
    except KeyError:
        raise ValueError(
            f"config/fiscal-rules.yaml declares no initiation mode {initiation!r} for "
            f"{country}; it knows {sorted(block['initiation'])}"
        ) from None
    if mode is not None and must_cite and not mode["prints_purpose"]:
        raise ValueError(
            f"initiation mode {initiation!r} prints no purpose line, and must_cite asks this "
            "page to cite its subject there — the plan and the named mode have come apart"
        )
    if mode is not None and not domestic_currency and mode["prints_card"]:
        raise ValueError(
            f"initiation mode {initiation!r} prints a card and an authorization code — a domestic "
            f"acquiring operation — and this page states a transfer in {currency}. Nothing "
            "observed here says what such a document looks like."
        )
    if mode is None:
        shares = initiation_shares()
        if must_cite:
            # The same draw over the modes that can carry the citation, renormalized — the page
            # is planned to print one, so a mode with no purpose line is not drawn from.
            shares = {
                name: share for name, share in shares.items()
                if block["initiation"][name]["prints_purpose"]
            }
        initiation = rng.choices(list(shares), weights=list(shares.values()), k=1)[0]
        mode = block["initiation"][initiation]

    # -- the money. Drawn in ten-kopiyka steps, as prices are: neither a transfer nor a fee is
    # quoted to an arbitrary kopiyka, and whole steps keep the sum exact.
    if amount is None:
        low, high = payment_confirmation_money_range("transfer_amount")
        amount = Decimal(rng.randrange(_minor(low), _minor(high), 10)) / 100
    if amount <= 0:
        raise ValueError(f"a payment confirmation states a positive amount, got {amount}")

    fee = Decimal(0)
    if rng.random() < payment_confirmation_share("nonzero_fee"):
        fee_low, fee_high = payment_confirmation_money_range("fee")
        fee = Decimal(rng.randrange(_minor(fee_low), _minor(fee_high), 10)) / 100

    # -- the two parties. The PAYER's bank is this document's own — the confirmation is issued by
    # the bank that moved the money — while the PAYEE's comes from the claim's identity, because
    # the account printed for the payee is the claim's and an IBAN carries its bank's code.
    bank_name, bank_code = _draw_bank(rng, country)

    # 👁 The SECOND form of emptiness — a caption with nothing under it — observed on the
    # recipient's bank among three such fields on one document. ⛔ The same form was observed on
    # the recipient's NAME too and is deliberately not produced there: `counterparty` is a
    # required label field, and emitting an empty one would assert that the document names no
    # counterparty, a case whose comparison rule the labelling contract has not settled. Declared
    # as a narrowing in config/labelling-schema.yaml rather than left for a reader to notice.
    # 🔴 A BANK OUTSIDE THE NATIONAL REGISTER HAS NO МФО TO PRINT. The code beside the payee's
    # bank is 📄 assigned in the National Bank's register of participants (`identifiers.bank_code`),
    # so a beneficiary banked abroad has none — and the beneficiary's IBAN is what says so, since
    # its country is the country of the account. The NAME is printed either way; the code is a
    # domestic requisite and is omitted rather than invented.
    payee_bank_label = block["parties"]["bank_code_label"]
    payee_is_domestic = identity.account.startswith(
        rules["identifiers"]["iban_format"]["country"]
    )
    payee_bank = (
        None
        if rng.random() < payment_confirmation_share("empty_captioned_field")
        else f"{identity.bank_name}, {payee_bank_label} {identity.bank_code}"
        if payee_is_domestic
        else identity.bank_name
    )
    payee = Party(
        name=printed_legal_name(vendor["name"], vendor["legal_form"]),
        trade_name=vendor["name"],
        # 🔴 AND NO CODE FOR A BENEFICIARY OUTSIDE THE REGISTER, for the reason its bank carries
        # none. 👁 The «Код» line holds a ЄДРПОУ or a РНОКПП — both Ukrainian registers — so a
        # foreign firm has no value for it, and printing the code drawn for the claim would put
        # an eight-digit Ukrainian identifier under a foreign company's name. The caption
        # disappears with the value: this is the field being INAPPLICABLE, not empty, and the two
        # observed forms of emptiness are about a field the document does have.
        code=identity.tax_code if payee_is_domestic else None,
        account=identity.account,
        bank=payee_bank,
    )

    # 👁 The FIRST form of emptiness, and the mode it belongs to: on the internet-acquiring
    # document the payer is not identified at all and a HYPHEN is printed as the value. An
    # extractor reads that hyphen as a string, which is why it is a value here and not a `None`.
    if mode["identifies_payer"]:
        payer = Party(
            name=payer_name,
            trade_name=payer_name,
            code=payer_tax_id,
            account=generate_iban(rng, bank_code, country),
            bank=None,
        )
    else:
        payer = Party(
            name=block["parties"]["empty_value"],
            trade_name=block["parties"]["empty_value"],
            code=None,
            account=None,
            bank=None,
        )

    # -- the conditional block: what a card operation adds
    auth_code = (
        _draw_from_pattern(rng, block["auth_code"]["pattern"])
        if mode["prints_auth_code"]
        else None
    )
    card_masked = _draw_card(rng, block) if mode["prints_card"] else None

    purpose = None
    cites_document_no = None
    if mode["prints_purpose"]:
        pool = payment_purposes(rules["language"])
        if must_cite:
            # The formula draw over the templates that name a рахунок, for the reason the mode
            # draw above narrowed: a plan that aims a subject disagreement at this page needs the
            # citation ON the page, and «Оплата за товар» carries none.
            pool = tuple(t for t in pool if "{invoice_no}" in t)
        template = rng.choice(pool)
        # An invoice number and its date, filled here rather than from the placeholder
        # vocabulary: this is a reference to another document, not merchandise. 🔴 It still proves
        # nothing about the SUBJECT — it names a document, and a document number says nothing
        # about what was bought — but where `cites` is given it names the claim's OWN invoice, so
        # the two documents can be linked by somebody willing to parse both ends. Some templates
        # name no document at all; that absence is deliberate and is what stops a linker from
        # assuming the reference is always there.
        purpose, cites_document_no = _fill_reference_traced(
            rng, template, rules, issued_at, cites
        )

    terminal_label = terminal_value = None
    if rng.random() < payment_confirmation_share("terminal"):
        caption = rng.choice(block["terminal"]["captions"])
        terminal_label = caption["label"]
        terminal_value = (
            rng.choice(initiating_systems(rules["language"]))
            if caption["value_kind"] == "system_name"
            else _draw_from_pattern(rng, block["terminal"]["mnemonic"]["pattern"])
        )

    signature = block["signature"]
    return PaymentConfirmation(
        bank_name=bank_name,
        bank_code=bank_code,
        title=rng.choice(block["titles"]),
        document_code=_draw_document_code(rng, block),
        issued_at=issued_at,
        date_caption=rng.choice(block["date_captions"]),
        payer=payer,
        payee=payee,
        initiation=initiation,
        transfer=amount,
        fee=fee,
        currency=currency,
        amount_caption=_money_caption(
            rng, block["amount_captions"], block,
            currency=currency, domestic=domestic_currency,
        ),
        fee_caption=_money_caption(
            rng, block["fee_captions"], block,
            currency=currency, domestic=domestic_currency,
        ),
        prints_total=rng.random() < payment_confirmation_share("prints_total"),
        # 👁 The words spell the TRANSFER and not the total: 📄 the amount of the operation is the
        # requisite, and the words are the same requisite written twice.
        amount_in_words=(
            amount_in_words_uk(amount, currency)
            if rng.random() < payment_confirmation_share("amount_in_words")
            else None
        ),
        amount_in_words_caption=rng.choice(block["amount_in_words_captions"]),
        purpose=purpose,
        cites_document_no=cites_document_no,
        auth_code=auth_code,
        card_masked=card_masked,
        terminal_label=terminal_label,
        terminal_value=terminal_value,
        # 👁 The stamp is 8/8 and is therefore not a draw; the signature beside it is 7/8, and the
        # document without one printed an EMPTY SIGNATURE LINE, so the caption stays either way.
        signature_path=(
            _draw_signature_path(rng)
            if rng.random() < payment_confirmation_share("signature")
            else None
        ),
        signatory_post=(
            rng.choice(signature["post_labels"])
            if rng.random() < payment_confirmation_share("signatory_post")
            else None
        ),
        electronic_note=(
            rng.choice(signature["electronic_notes"])
            if rng.random() < payment_confirmation_share("electronic_note")
            else None
        ),
        # 👁 3/8, which refutes "every confirmation carries a verification footer".
        verification_footer=(
            block["verification_footer"]["steps"]
            if rng.random() < payment_confirmation_share("verification_footer")
            else None
        ),
        # 👁 2/8, both marketing. `None` means no QR block on the page at all.
        qr_payload=(
            rng.choice(block["qr"]["marketing_payloads"])
            if rng.random() < payment_confirmation_share("qr")
            else None
        ),
        decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
    )


@dataclass(frozen=True)
class StatementRow:
    """One operation as an account statement prints it.

    Seven printed values, 👁 all of them observed on the corporate statement this class is built
    from. Only some of them reach a label, and which ones is the point of the class: the row the
    claim rests on contributes `amount`, `date`, `counterparty` and `payment_purpose` to
    `DocGroundTruth`, every other row on the page contributes nothing at all, and the four
    counterparty fields collapse to the NAME alone even on the labelled row.

    `amount` IS UNSIGNED. `direction` says which of the two money columns it is printed in — see
    `schemas.Direction` for why that is a field rather than a sign.
    """

    number: str
    at: datetime
    amount: Decimal
    direction: Direction
    purpose: str
    counterparty_name: str
    counterparty_code: str
    counterparty_account: str
    counterparty_bank: str
    # The рахунок number `purpose` names, where it names one — set on the LABELLED row, whose
    # purpose is the one the label ships, and left `None` on ordinary rows: their citations point
    # outside the claim by construction and no label reads them.
    cites_document_no: str | None = None

    @property
    def is_debit(self) -> bool:
        return self.direction is Direction.DEBIT


@dataclass(frozen=True)
class BankStatement:
    """One Ukrainian bank account statement, complete but not yet rendered.

    🔴 THE LABEL CARRIES ONE TRANSACTION AND THE DOCUMENT'S OWN TOTALS CARRY NOTHING. That is the
    whole shape of this class, and it is derived rather than chosen: the consumer's required-field
    table names a statement's amount, date, payee and payment purpose in the SINGULAR and states
    that the type has no line items, so what it describes is a transaction. `relevant` is that
    transaction and `ground_truth` reads every scoreable field off it.

    🔴 THE FOUR TURNOVER TOTALS ARE PRINTED AND NEVER LABELLED — opening balance, closing balance,
    total credit, total debit. 👁 They are the most prominent numbers on the page, and omitting
    them would make "find the relevant transaction" artificially easy and inflate whatever is
    measured on the class. It is the same refusal as `PrroReceipt.amount_due` not being scoreable
    and as the confirmation's total being printed while the transfer is the answer: a number is
    printed because the document prints it, and labelled only where a label can mean something.
    All four are DERIVED from the rows, so the page cannot contradict itself.

    ⚠️ THE DIFFICULTY OF FINDING THE RELEVANT ROW IS UNDERSTATED IN THIS VERSION. The other rows
    are ordinary operations drawn from the same pools, with no deliberate resemblance to the
    labelled one — a row carrying the same counterparty or a nearby amount is a DECOY, that is a
    difficulty dial belonging to the trap design, and it is deferred. Anyone quoting a number
    measured on this class has to say so, which is why the contract states it too rather than
    leaving it here.

    THE LABELLED AMOUNT APPEARS ON NO OTHER ROW, and that is a separate decision from deferring
    decoys — it is what makes the label well-posed. A consumer identifies the row from the claim's
    OTHER document, an invoice naming a seller and a total, so two rows answering that description
    would leave the ground truth pointing at one of two indistinguishable answers. The claim's
    payee is additionally kept out of the pool the other rows draw from.

    The residual case, stated rather than implied: a sole trader's printed name is DRAWN, so an
    ordinary row could in principle draw the same personal name as the claim's payee. The amount is
    what makes the row unique in that case, which is why the amount is the invariant and the payee
    exclusion is not.
    """

    bank_name: str
    bank_code: str
    holder_name: str
    holder_code: str
    account: str
    period_start: date
    period_end: date
    # The date and time the statement itself was produced, which is NOT the date of the
    # transaction it is labelled for. 👁 The observed statement was issued the morning after its
    # period closed.
    issued_at: datetime
    opening_balance: Decimal
    rows: tuple[StatementRow, ...]
    # Which row the label is about, as an index into `rows`. An index rather than a copy of the
    # row so there is one row object and no way for the two to drift; the label exports the row's
    # printed NUMBER, which is what a reader of the image can point at.
    relevant_index: int
    # The BARE trading name of the labelled row's counterparty. The row PRINTS the name with its
    # legal form; the label carries the bare one, which config/labelling-schema.yaml makes
    # authoritative under `normalization.party_name`. Carried on the statement rather than on every
    # row because only one row is labelled — and see `Party.trade_name` for the defect that made
    # both classes need it.
    payee_trade_name: str
    decimal_separator: str

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a statement lists at least one operation")
        if not 0 <= self.relevant_index < len(self.rows):
            raise ValueError(
                f"relevant_index {self.relevant_index} is outside the {len(self.rows)} rows of "
                "this statement, so the label would point at no row at all"
            )
        # The pointer has to point at one row. 👁 A drawn number repeated on two rows of one render
        # before the numbering became a counter, and nothing but this would have caught it: the
        # label was still correct about the row's values, and only the POINTER was ambiguous.
        numbers = [row.number for row in self.rows]
        if len(set(numbers)) != len(numbers):
            raise ValueError(
                "two operations on this statement carry the same number, so "
                "`relevant_transaction` would point at both — see `_draw_operation_numbers`"
            )
        # 🔴 THE INVARIANT OF THIS CLASS, enforced at construction rather than trusted: only a
        # DEBIT can be proof of payment. A credit is money arriving — a refund, a reversal, a
        # transfer in — and it evidences no expense whatever its amount, so a statement whose
        # labelled row is a credit is a document that cannot support the claim it was built for.
        # `proves_payment_by_direction` states the same rule as a validator, and
        # `policy_engine.resolve_evidence` refuses such a claim rather than labelling it.
        if not self.relevant.is_debit:
            raise ValueError(
                f"row {self.relevant.number} is a {self.relevant.direction.value} and cannot be "
                "proof of payment: money arriving is a refund, not an expense. A statement whose "
                "labelled transaction is a credit is a trap archetype, which has to declare the "
                "broken invariant in the claim's `imperfection` rather than be built here"
            )

    @property
    def relevant(self) -> StatementRow:
        """The one transaction this document is labelled for."""
        return self.rows[self.relevant_index]

    # -- the sheets ------------------------------------------------------------

    @property
    def page_sheets(self) -> tuple[tuple[StatementRow, ...], ...]:
        """The operations grouped by the sheet each one is printed on, in reading order.

        🔴 DERIVED FROM THE ROW COUNT AND NOTHING ELSE, which is what makes it honest: a page holds
        what fits on it, so how many sheets this document has is a consequence of how many
        operations it lists, not a second decision that could disagree with the first. The
        capacities are `bank_statement.pagination` in config/fiscal-rules.yaml — layout, checked
        against a real render by a test.

        ⚠️ THE FIRST SHEET IS FILLED BEFORE THE SECOND IS STARTED, and it holds FEWER rows than a
        continuation sheet: it carries the bank header, the title and the turnover block, and a
        continuation sheet gets that height back. A split that balanced the sheets evenly would be
        a layout no printer produces.

        🔴 A SHEET THAT IS THE WHOLE STATEMENT HOLDS ONE ROW MORE THAN THE FIRST SHEET OF A
        PAGINATED ONE, because a paginated sheet carries the «Сторінка N з M» footer and that line
        costs a row. The two capacities are not circular: whether there is a footer at all is
        decided by whether the rows fit under the LARGER bound, which is asked first and answered
        without reference to the smaller.
        """
        single, first, per_sheet = _statement_capacities()
        if len(self.rows) <= single:
            return (self.rows,)
        sheets = [self.rows[:first]]
        rest = self.rows[first:]
        while rest:
            sheets.append(rest[:per_sheet])
            rest = rest[per_sheet:]
        return tuple(sheets)

    @property
    def page_count(self) -> int:
        """How many sheets this statement is printed on — `DocGroundTruth.page_count`."""
        return len(self.page_sheets)

    # -- the four turnover totals, all derived --------------------------------

    @property
    def total_debit(self) -> Decimal:
        return sum((row.amount for row in self.rows if row.is_debit), Decimal(0)).quantize(KOPIYKA)

    @property
    def total_credit(self) -> Decimal:
        return sum(
            (row.amount for row in self.rows if not row.is_debit), Decimal(0)
        ).quantize(KOPIYKA)

    @property
    def closing_balance(self) -> Decimal:
        """👁 The observed statement's four figures satisfy this exactly, so it is arithmetic
        rather than a convention: what was there, plus what arrived, less what left."""
        return (self.opening_balance + self.total_credit - self.total_debit).quantize(KOPIYKA)

    @property
    def debit_count(self) -> int:
        return sum(1 for row in self.rows if row.is_debit)

    @property
    def credit_count(self) -> int:
        return len(self.rows) - self.debit_count

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted.

        Rows arrive as a list of dicts rather than as `StatementRow` objects, because the template
        needs each amount already in its own column and formatted: a template deciding which of
        two columns a number goes in would put the direction rule in the markup, where no test
        looks for it.

        `relevant` is passed as a FLAG ON EACH ROW, and it is what marks the labelled cells for
        the bounding-box collector. Nothing about it is visible on the page: it adds no class, no
        emphasis and no ordering, so 👁 to the eye the row is one of twenty. That has to be true
        or the corpus measures a highlighted row.
        """
        rules = jurisdiction("UA")
        block = rules["bank_statement"]
        return {
            "bank_name": self.bank_name,
            "bank_code": self.bank_code,
            "header": block["header"],
            "title": block["header"]["title"].format(
                start=self.period_start.strftime(rules["date_format"]),
                end=self.period_end.strftime(rules["date_format"]),
            ),
            "holder_name": self.holder_name,
            "holder_code": self.holder_code,
            "account": self.account,
            "last_operation_on": self.rows[-1].at.strftime(rules["date_format"]),
            "issued_on": self.issued_at.strftime(rules["date_format"]),
            "issued_time": self.issued_at.strftime(block["time_format"]),
            "totals": block["totals"],
            "opening_balance": self._amount(self.opening_balance),
            "closing_balance": self._amount(self.closing_balance),
            "total_credit": self._amount(self.total_credit),
            "total_debit": self._amount(self.total_debit),
            "credit_count": self.credit_count,
            "debit_count": self.debit_count,
            "columns": block["columns"],
            "pages": self._pages(rules, block),
            # This class carries no QR — 👁 none was observed on a statement, and the renderer
            # requires the key on every context.
            "qr_payload": None,
        }

    def _pages(self, rules: dict, block: dict) -> list[dict]:
        """The sheets the template lays out, each with its own rows and its own footer.

        🔴 THE ROW'S INDEX IS THE ONE IT HAS IN THE DOCUMENT, not on its sheet. Every row carries an
        `operation_<i>` box so that a test can decide box containment per row, and a counter that
        restarted on each sheet would give two rows one box name — which the renderer refuses, and
        which would otherwise have made the second sheet's boxes overwrite the first's.

        `folio` IS `None` ON A ONE-SHEET STATEMENT, and that is the whole of the difference between
        today's page and this one: «Сторінка 1 з 1» printed on the majority of this class's images
        would be a visible change to a settled look, in exchange for a count a reader can see.
        """
        total = self.page_count
        folio = block["pagination"]["folio"]
        pages = []
        offset = 0
        for sheet in self.page_sheets:
            pages.append(
                {
                    "folio": (
                        folio.format(page=len(pages) + 1, total=total) if total > 1 else None
                    ),
                    "rows": [
                        {
                            "index": offset + index,
                            "number": row.number,
                            "date": row.at.strftime(rules["date_format"]),
                            "time": row.at.strftime(block["time_format"]),
                            "debit": self._amount(row.amount) if row.is_debit else None,
                            "credit": None if row.is_debit else self._amount(row.amount),
                            "purpose": row.purpose,
                            "counterparty_name": row.counterparty_name,
                            "counterparty_code": row.counterparty_code,
                            "counterparty_account": row.counterparty_account,
                            "counterparty_bank": row.counterparty_bank,
                            "relevant": offset + index == self.relevant_index,
                        }
                        for index, row in enumerate(sheet)
                    ],
                }
            )
            offset += len(sheet)
        return pages

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
        page_regions: list[tuple[float, float, float, float]] | None = None,
    ) -> DocGroundTruth:
        """The label record for this statement — one transaction, not one document.

        `amount`, `date`, `counterparty`, `payment_purpose` and `direction` are read off
        `relevant`, and `relevant_transaction` carries that row's printed number so the label says
        WHICH row the answers came from. `field_bboxes` then says where: the LABELLED field keys
        are the cells of that row, because only its cells carry a field marker. Every row also
        carries an indexed marker of its own — `operation_<i>` over the whole row rect — so the
        boxes on a statement are NOT the labelled row's alone, and the labelled ones are.

        NONE OF THE FOUR TURNOVER TOTALS IS LABELLED, and none of them is `total_charged` under
        another name either — that field is the confirmation's «Загальна сума», one payment plus
        its fee. Reusing it for a period's turnover would give one key two meanings, which is the
        failure config/labelling-schema.yaml exists to prevent.

        `counterparty` is the payee's NAME only, out of the 👁 four fields the row prints for it.
        ⚠️ A single counterparty field is ambiguous about ROLE on a statement — for a debit the
        counterparty received the money, for a credit it sent it — and `direction` is what removes
        the ambiguity. That is the second reason the field exists, beside proof of payment.

        🔴 `page_regions` COMES FROM THE RENDERER AND `page_count` FROM THIS OBJECT, and the two
        must be handed in together: the document knows how many sheets it has, only the render
        knows where they landed. ⛔ NO SILENT FALLBACK — a paginated statement whose regions were
        not supplied is refused rather than labelled as one page, because that label would be
        wrong about an image already written to disk and nothing downstream would say so.
        """
        if self.page_count > 1 and page_regions is None:
            raise ValueError(
                f"this statement is printed on {self.page_count} sheets, so its label needs "
                "`page_regions` — the renderer's `page_N` boxes, in reading order"
            )
        row = self.relevant
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.BANK_STATEMENT,
            language="uk",
            currency="UAH",
            amount=row.amount,
            direction=row.direction,
            date=row.at.date(),
            counterparty=self.payee_trade_name,
            payer=self.holder_name,
            payment_purpose=row.purpose,
            cites_document_no=row.cites_document_no,
            # 👁 The observed statement's header carries no number of its own — a client, an
            # account, a period and a production time. `document_code` is therefore `None`, and
            # `relevant_transaction` below is a pointer INTO the document rather than its identity.
            document_code=None,
            relevant_transaction=row.number,
            # EMPTY, and settled by the requirement rather than open: a statement has no line
            # items. It lists transactions, and a transaction is not a line of a basket — which is
            # exactly why 👁 a statement establishes nothing about what was bought.
            line_items=[],
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            # ⛔ `file_region` STAYS `None`: this file holds one document. Several documents in one
            # file is a separate relation and a separate step.
            page_count=self.page_count,
            page_regions=page_regions,
            # From the RENDERER, like the boxes: neither is decided by the content class, and both
            # describe the page that was produced from it.
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def _draw_operation_numbers(rng: random.Random, block: dict, count: int) -> list[str]:
    """`count` per-row operation numbers, in the 👁 two observed shapes and all distinct.

    🔴 DISTINCT BY CONSTRUCTION, because the number is the label's pointer at the labelled row: two
    rows carrying the same one would point at both. The trailing digits are ONE COUNTER across the
    page, which is 👁 what the short form is — a client's own sequential document numbering — so the
    realism and the uniqueness are the same fact rather than a constraint bolted onto a draw.

    Why a counter rather than rejection sampling: a loop that redrew on collision would take a
    number of values from `rng` that depends on which collisions occurred, and every later value in
    the whole run would shift with them.

    The counter is enough on its own. Two rows of one width differ because their ordinals do; two
    rows of different widths differ in length; and a prefixed number can never equal a bare one.
    """
    first = rng.randrange(10**6)
    numbers = []
    for ordinal in range(count):
        spec = rng.choice(block["operation_number"]["formats"])
        width = spec["serial_digits"]
        head = _draw_from_pattern(rng, spec["pattern"]) if spec["pattern"] else ""
        serial = (first + ordinal) % 10**width
        numbers.append(f"{spec['prefix']}{head}{serial:0{width}d}")
    return numbers


def _draw_row_amount(rng: random.Random, low: Decimal, high: Decimal) -> Decimal:
    """An amount in ten-kopiyka steps, as every other amount in this repository is drawn."""
    return Decimal(rng.randrange(_minor(low), _minor(high), 10)) / 100


def _statement_capacities(country: str = "UA") -> tuple[int, int, int]:
    """How many operations fit on a sheet that is the whole statement, on the first sheet of a
    paginated one, and on each sheet after that.

    ⛔ THE CONTINUATION SHEET IS NOT AN OBSERVED ANATOMY. Only the first page of the one statement
    was ever seen; the capacities are `bank_statement.pagination` in config/fiscal-rules.yaml, and
    what a continuation sheet carries there is general layout of a paginated table.
    """
    block = jurisdiction(country)["bank_statement"]["pagination"]
    return (
        int(block["rows_single_sheet"]),
        int(block["rows_first_page"]),
        int(block["rows_continuation_page"]),
    )


def draw_statement_pages(rng: random.Random, country: str = "UA") -> int:
    """How many sheets the next statement runs to — 1 or 2, at the declared share.

    🔴 A COMPOSITION KNOB AND NOT A LABEL, which is why it reads config/generation.yaml and why it
    is a function of its own rather than a draw buried in `build_bank_statement`. The assembler
    calls it and passes the answer in, exactly as it draws the capture channel and passes that in:
    how difficult a document is belongs to whoever is composing the run, and a builder that decided
    it would leave every caller — including a test — unable to ask for either case.

    ⛔ THREE SHEETS ARE NOT DRAWN, though `page_sheets` splits any number of rows. The second page
    is what makes «one page = one document» falsifiable; a third would add paper and test nothing
    the second does not.
    """
    return 2 if rng.random() < bank_statement_share("two_page") else 1


def build_bank_statement(
    rng: random.Random,
    *,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    payer_name: str,
    payer_tax_id: str,
    amount: Decimal | None = None,
    cites: DocumentReference | None = None,
    must_cite: bool = False,
    pages: int = 1,
    country: str = "UA",
) -> BankStatement:
    """Build one Ukrainian bank account statement.

    ``must_cite`` narrows the LABELLED row's purpose formula to those naming a рахунок by number
    — the `subject` axis's label-first knob, exactly as on the confirmation builder. The ordinary
    rows are untouched: their citations point outside the claim by construction. Requires
    ``cites`` for the same reason the confirmation does.

    ⚠️ `issued_at` IS THE MOMENT OF THE LABELLED TRANSACTION, not of the document. Every other
    archetype of this repository is a document about one payment, so the two coincide there and
    the parameter means the same thing to the assembler; here the statement covers a period, and
    its own production time is derived from the period's end. The claim's payment date is the
    row's, which is what `ground_truth` puts in `date`.

    `vendor` is the claim's payee, resolved, and it appears on the labelled row and on no other:
    no ordinary row repeats that counterparty, so the row the label points at is the only one
    matching the claim's other document. `identity` is that payee's, and it is what makes the
    match hold on more than the NAME: the labelled row prints the claim's tax code, account and
    bank, while every ordinary row draws its own. `payer_name` and `payer_tax_id` come from the
    persona and are the ACCOUNT HOLDER — a statement of anybody else's account would evidence
    nothing about this claimant's money.

    `cites` is the claim's subject document. ⛔ IT REACHES THE LABELLED ROW AND NO OTHER. Every
    ordinary row names a document outside the claim, which is the whole reason a statement
    establishes nothing about what was bought, and a page whose every row cited the same invoice
    would be a different document altogether.

    `amount` is the labelled transaction's amount, drawn when not given. It is a parameter for the
    reason the confirmation's `transfer` is one: a caller pairing this statement with an invoice
    has to be able to state the amount both documents describe.

    `pages` IS HOW MANY SHEETS THE DOCUMENT RUNS TO, and it is a parameter for a third reason: it
    is a composition decision, drawn by the assembler from `two_page_share` — see
    `draw_statement_pages`. ⚠️ It reaches this function as a ROW COUNT and nothing else: which
    range the operations are drawn from is the whole of what it changes, and how those rows then
    fall across sheets is `BankStatement.page_sheets`, derived from the count. A page holds what
    fits on it, so the two cannot disagree. DEFAULTING TO 1 is what keeps every existing caller —
    and every figure already measured on this class — on the one-page document it was built for.
    """
    rules = jurisdiction(country)
    block = rules["bank_statement"]
    if pages not in (1, 2):
        raise ValueError(
            f"a statement is built on one sheet or two, not {pages}: config/generation.yaml draws "
            "a row count for each, and a third sheet has no range to draw from"
        )

    if amount is None:
        amount = _draw_row_amount(rng, *bank_statement_money_range("transaction_amount"))
    if amount <= 0:
        raise ValueError(f"a statement row states a positive amount, got {amount}")

    # -- the account and its holder
    bank_name, bank_code = _draw_bank(rng, country)
    account = generate_iban(rng, bank_code, country)

    # -- the period. Derived from where the operations fall rather than declared: ⛔ nothing
    # observed says how long a personal statement's period is, while the observed document's own
    # period was exactly the span of its operations.
    lead = rng.randint(*bank_statement_count_range("period_lead_days"))
    trail = rng.randint(*bank_statement_count_range("period_trail_days"))
    period_start = issued_at.date() - timedelta(days=lead)
    period_end = issued_at.date() + timedelta(days=trail)

    # -- how many operations, and how many of them arrive rather than leave. The range is the one
    # tuned against the sheets this document runs to — see `two_page_row_count_range`.
    row_count = rng.randint(
        *bank_statement_count_range("row_count" if pages == 1 else "two_page_row_count")
    )
    # One row is the labelled transaction and one is the bank's own service charge; the rest are
    # ordinary payments. `max` keeps a short page from having no ordinary rows at all.
    ordinary = max(1, row_count - 2)
    # At least one credit whatever the share returns: `direction` is a label field, and a page
    # with no credit row carries no visible instance of the distinction it names.
    credits = max(1, round(bank_statement_share("credit") * ordinary))
    credits = min(credits, ordinary)

    # The claim's own payee is taken OUT of the pool the ordinary rows draw from, so that no other
    # row names it. See the uniqueness note on `BankStatement` for what that buys and for the one
    # residual case it does not cover.
    # Every row's number, allocated together so the counter that keeps them distinct is one
    # counter. `ordinary + 2` is the fee row and the labelled transaction beside the ordinary ones.
    numbers = _draw_operation_numbers(rng, block, ordinary + 2)

    pool = [
        dict(entry)
        for entry in every_vendor(country)
        if dict(entry).get("name") != vendor.get("name")
    ]
    purposes = {
        kind: statement_purposes(rules["language"], kind)
        for kind in ("debit", "credit", "credit_from_self", "service_fee")
    }

    def counterparty_of(entry: dict) -> tuple[str, str, str, str]:
        """The four printed fields of a counterparty: name, code, account, bank.

        A FRESH IDENTITY PER ROW, and that is correct here: these are other firms the holder paid,
        each appearing once on the page. The claim's own payee is the one counterparty whose
        identity is fixed for the whole claim, and it is not drawn through this function.
        """
        resolved = resolve_vendor(rng, entry, country)
        their = draw_party_identity(rng, resolved, country)
        return (
            printed_legal_name(resolved["name"], resolved["legal_form"]),
            their.tax_code,
            their.account,
            their.bank_name,
        )

    def purpose_of(kind: str, at: datetime, cites: DocumentReference | None = None) -> str:
        """A purpose line, with the document it refers to filled in.

        🔴 Without `cites` it refers to a document that is not in the claim, which is the whole
        reason a statement establishes nothing about what was bought. The LABELLED row passes the
        claim's invoice, so that one row can be linked to the invoice beside it — and a purpose
        naming a ВН still points outside the claim even there, because a delivery note is a
        different class of document.
        """
        return _fill_reference(rng, rng.choice(purposes[kind]), rules, at, cites)

    relevant_name = printed_legal_name(vendor["name"], vendor["legal_form"])
    rows: list[StatementRow] = []

    # -- the ordinary operations. Their amounts avoid the labelled one, so the labelled row is the
    # only row on the page carrying that amount.
    debit_range = bank_statement_money_range("debit_amount")
    credit_range = bank_statement_money_range("credit_amount")
    for index in range(ordinary):
        direction = Direction.CREDIT if index < credits else Direction.DEBIT
        entry = rng.choice(pool)
        name, code, iban, their_bank = counterparty_of(entry)
        # A credit may be a transfer from ANOTHER ACCOUNT OF THE HOLDER'S, in which case the
        # counterparty is the holder. Drawn after the vendor rather than instead of it, so that
        # adding the case shifted no later value in the run.
        kind = "credit" if direction is Direction.CREDIT else "debit"
        if direction is Direction.CREDIT and rng.random() < bank_statement_share(
            "credit_from_self"
        ):
            kind = "credit_from_self"
            name, code, iban = payer_name, payer_tax_id, generate_iban(rng, bank_code, country)
            their_bank = bank_name
        row_amount = _draw_row_amount(
            rng, *(credit_range if direction is Direction.CREDIT else debit_range)
        )
        if row_amount == amount:
            # Nudged by one draw step rather than redrawn: a rejection loop would make the number
            # of values taken from `rng` depend on what the earlier draws returned, and every later
            # value in the whole run would shift with it.
            row_amount += KOPIYKA * 10
        at = _draw_row_time(rng, period_start, period_end)
        rows.append(
            StatementRow(
                number=numbers[index],
                at=at,
                amount=row_amount,
                direction=direction,
                purpose=purpose_of(kind, at),
                counterparty_name=name,
                counterparty_code=code,
                counterparty_account=iban,
                counterparty_bank=their_bank,
            )
        )

    # -- the bank's own service charge: 👁 a debit whose counterparty is the issuer itself.
    fee_at = _draw_row_time(rng, period_start, period_end)
    fee_low, fee_high = bank_statement_money_range("service_fee")
    rows.append(
        StatementRow(
            number=numbers[ordinary],
            at=fee_at,
            amount=_draw_row_amount(rng, fee_low, fee_high),
            direction=Direction.DEBIT,
            purpose=purposes["service_fee"][0],
            # 👁 The counterparty of a service charge is the issuer itself, and the code beside it
            # is a МФО — six digits under the same «Код» caption that carries eight and ten
            # elsewhere on the page. ⚠️ That is the confirmation's finding about caption plus
            # length appearing again on a second class, and it is why a rule reading the KIND of
            # code off its caption alone is wrong.
            #
            # 🔴 THE ISSUER'S OWN CODE, NOT A FRESH DRAW: this row names the same bank the header
            # does, so it prints the header's `bank_code` rather than drawing one of its own — a
            # second draw here is exactly what let this row disagree with the header AND with the
            # МФО inside its own IBAN, on the same line, on a delivered document.
            counterparty_name=bank_name,
            counterparty_code=bank_code,
            counterparty_account=generate_iban(rng, bank_code, country),
            counterparty_bank=bank_name,
        )
    )

    # -- the labelled transaction. The ONE row that carries the claim's own payee, and therefore
    # the only row printing the claim's identity and citing the claim's invoice. Its purpose is
    # filled TRACED — the label ships the cited рахунок number structured — and, under
    # `must_cite`, from the formulas that name one.
    labelled_pool = purposes["debit"]
    if must_cite:
        if cites is None:
            raise ValueError(
                "must_cite forces the labelled row's purpose to name the document in `cites`, "
                "and none was given — a forced citation of no document is not a page anything "
                "plans"
            )
        labelled_pool = [t for t in labelled_pool if "{invoice_no}" in t]
    labelled_purpose, labelled_cites_no = _fill_reference_traced(
        rng, rng.choice(labelled_pool), rules, issued_at, cites
    )
    rows.append(
        StatementRow(
            number=numbers[ordinary + 1],
            at=issued_at,
            amount=amount,
            direction=Direction.DEBIT,
            purpose=labelled_purpose,
            counterparty_name=relevant_name,
            counterparty_code=identity.tax_code,
            counterparty_account=identity.account,
            counterparty_bank=identity.bank_name,
            cites_document_no=labelled_cites_no,
        )
    )

    # 👁 Operations are printed in the order they happened, so the labelled one lands wherever its
    # timestamp puts it — usually in the middle of the page. Sorted by the timestamp AND then by
    # the drawn order, so two operations in the same second keep a defined order under a seed.
    ordered = sorted(range(len(rows)), key=lambda index: (rows[index].at, index))
    rows = [rows[index] for index in ordered]
    relevant_index = ordered.index(len(ordered) - 1)

    return BankStatement(
        bank_name=bank_name,
        bank_code=bank_code,
        holder_name=payer_name,
        holder_code=payer_tax_id,
        account=account,
        period_start=period_start,
        period_end=period_end,
        # 👁 The observed statement was produced the morning after its period closed.
        issued_at=datetime(
            period_end.year, period_end.month, period_end.day, rng.randint(9, 11),
            rng.randint(0, 59)
        )
        + timedelta(days=1),
        # 🔴 DERIVED, NOT DRAWN: what the account started with is what makes the period's
        # arithmetic land on a plausible residue. Drawing it independently produced statements with
        # a NEGATIVE closing balance — an account with a credit line, which is a different document
        # and one nothing observed supports. The residue is drawn; `max` keeps the opening balance
        # non-negative in the other direction, where more money arrived than left.
        opening_balance=(
            _draw_row_amount(rng, *bank_statement_money_range("closing_balance"))
            + max(Decimal(0), _statement_net_outflow(rows))
        ),
        rows=tuple(rows),
        relevant_index=relevant_index,
        payee_trade_name=vendor["name"],
        decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
    )


def _statement_net_outflow(rows: list[StatementRow]) -> Decimal:
    """What the operations take out of the account, less what they put in."""
    return sum(
        (row.amount if row.is_debit else -row.amount for row in rows), Decimal(0)
    ).quantize(KOPIYKA)


def _draw_row_time(rng: random.Random, start: date, end: date) -> datetime:
    """A timestamp inside the statement's period, at an hour a payment is made at."""
    day = start + timedelta(days=rng.randint(0, (end - start).days))
    return datetime(day.year, day.month, day.day, rng.randint(8, 21), rng.randint(0, 59))


@dataclass(frozen=True)
class InvoiceParty:
    """One side of an invoice — the supplier, or the buyer it is addressed to.

    NOT `Seller`, which models the party block of a FISCAL RECEIPT: that class carries the two
    identifier lines «ІД» and «ПН» with their prefixes, which are requisites of the receipt form and
    appear nowhere on an invoice. Two classes rather than one with half its fields unused, because
    the overlap is a coincidence of both documents naming a firm.

    Every field but `name` and `code` is optional: 👁 an invoice names the supplier fully — address,
    telephone, account, bank — and the buyer by name and code alone.
    """

    name: str
    legal_form: str
    code: str
    code_label: str
    address: str | None = None
    phone: str | None = None
    account: str | None = None
    bank: str | None = None


@dataclass(frozen=True)
class Invoice:
    """One Ukrainian рахунок на оплату, complete but not yet rendered.

    🔴 AN OFFER TO PAY, AND EVERY DECISION HERE FOLLOWS FROM THAT. 📄 An invoice is not a primary
    accounting document: it proposes that the buyer pay, and the fact of payment is established by a
    payment document. Two consequences that a reader coming from the consumer's field list will
    look for and not find:

    * **NO PAYMENT STATUS.** 👁 0 of 2 open invoices print one, and the reason is structural rather
      than a small sample. The consumer's requirement asks this type for a payment status — the
      example it gives, «Zapłacono», is Polish — and that requirement is recorded as a DIVERGENCE in
      config/labelling-schema.yaml rather than satisfied. Nothing in this repository lets a verdict
      rest on such a line, which is the guard that matters: an oracle reading proof of payment off a
      printed word would be deriving the answer from the thing under test.

      🔴 `schedule` IS NOT THAT LINE AND MUST NOT BE READ AS ONE. It states a PAYMENT TERM — that
      the obligation is settled in equal parts, and what one part comes to — which is a condition
      of the offer, settled when the invoice is drawn up and before any money exists to record.
      The guard above survives intact: a verdict does rest on the term's presence, but the term
      says only how the seller proposes to be paid, and whether money actually moved is still
      decided from the PAYMENT document's type, exactly as it is on every other claim. Nothing on
      this page becomes proof of payment.
    * **NO `amount_due`.** 👁 1/1 has a single total block. «ДО СПЛАТИ» is 📄 line 24 of the fiscal
      receipt form, where it differs from «СУМА» by the discount and the rounding. An invoice has
      one total and nothing for a second field to differ from.

    THIS IS THE SUBJECT DOCUMENT OF THE DOMINANT PAIR. It states what was bought and does not prove
    payment; a confirmation or a statement proves the payment and states no subject. That is the
    exact inverse of the bank classes, and it is why registering this archetype is what makes a
    two-document claim buildable at all.

    ⛔ NO PER-LINE VAT LETTER. 👁 The observed table prices VAT-inclusive and states the tax once at
    the foot, so `line_items` carry `vat_letter=None` — a letter labelled and not printed would be a
    ground-truth value unreadable from the image, which is the rule that gave the bank statement its
    second money column. The tax total is computed from the letters BEFORE they are dropped, so the
    figure is the same one a receipt would print.
    """

    number: str
    issued_at: datetime
    supplier: InvoiceParty
    buyer: InvoiceParty
    vat_payer: bool
    line_items: list[LineItem]
    # 👁 The tax contained within the total, stated once. `Decimal(0)` for a seller that is not
    # registered, whose document carries no tax line at all.
    vat_total: Decimal
    unit: str
    # 👁 1 of 2 carries an agreement; 📄 both sources call it optional.
    agreement: str | None
    # 📄 The recommended alternative to a payment status — how long the offer stands.
    validity: str | None
    # WHICH INSTALMENT SCHEDULE THIS OBLIGATION IS SETTLED ON — a key of
    # `config.partial_payment_schedules`, or `None` for an invoice payable in one. The COUNT is not
    # stored beside it: two values that must agree should not be two values, and the count is a
    # lookup away. The page prints the schedule's Ukrainian adverb and the amount of one part; it
    # never prints the count, so no label carries it either — a labelled value unreadable from the
    # image is the thing this class refuses everywhere else.
    schedule: str | None
    signatory_name: str
    signatory_post: str | None
    bank_code: str
    decimal_separator: str

    @property
    def total(self) -> Decimal:
        """Σ over the line items. DERIVED rather than stored: an invoice states one total, and two
        numbers that must agree should not be two numbers."""
        return line_items_total(self.line_items)

    @property
    def instalment_amount(self) -> Decimal | None:
        """What ONE PART of this obligation comes to, or `None` for an invoice payable in one.

        DERIVED from the total and the schedule for the reason `total` itself is derived: an
        invoice states one obligation, and a part of it that could disagree with the whole would be
        a second number saying the same thing. Rounded to the kopiyka half-up, the rule every
        amount in this repository is normalized under.

        ⚠️ THE PARTS NEED NOT SUM BACK TO THE TOTAL, and the page never claims they do. A total of
        1000.00 in three parts prints 333.33, and three of those come to 999.99; the invoice states
        the amount of the NEXT payment, not a schedule of every one, so there is no printed
        arithmetic for the missing kopiyka to contradict. Deciding where a remainder is carried is
        a commercial term nothing here observes.
        """
        if self.schedule is None:
            return None
        return (self.total / partial_payment_schedules()[self.schedule]).quantize(
            KOPIYKA, rounding=ROUND_HALF_UP
        )

    @property
    def reference(self) -> DocumentReference:
        """How a payment document names this invoice.

        A property rather than a field the assembler assembles: the number and the date are already
        on this object, and a caller composing them itself would be a second place that decides what
        a reference to an invoice consists of.
        """
        return DocumentReference(number=self.number, issued_at=self.issued_at)

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def _long_date(self) -> str:
        """👁 The title's date in words — «24 травня 2025». The month is genitive, which is the case
        that follows a day number in Ukrainian; the names are in config/fiscal-rules.yaml."""
        months = jurisdiction("UA")["invoice"]["long_date_months"]
        return f"{self.issued_at.day} {months[self.issued_at.month - 1]} {self.issued_at.year}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted.

        👁 THE TITLE'S DATE IS IN WORDS AND THE REST OF THE PAGE'S DATES ARE IN DIGITS, so both forms
        appear on one document. That is a date-parsing case a corpus of receipts never presents, and
        it is reproduced because the observed invoice does it.
        """
        rules = jurisdiction("UA")
        block = rules["invoice"]
        columns = dict(block["columns"])
        if not self.vat_payer:
            # 📄 A seller that is not registered prices without ПДВ, so the two money columns lose
            # the suffix. The status is the vendor's, exactly as on a receipt.
            columns["price"], columns["sum"] = columns["price_no_vat"], columns["sum_no_vat"]
        # The instalment term, composed here rather than in Jinja: the template prints a caption
        # and a money value or nothing at all, and whether this invoice states one is a property
        # of the document. `instalment_amount` is not None exactly when `schedule` is not.
        part = self.instalment_amount
        instalment = None if part is None else {
            "caption": block["instalment_caption_format"].format(
                period=block["instalment_periods"][self.schedule]
            ),
            "amount": self._amount(part),
        }
        return {
            "attention_line": block["attention_line"],
            "title": block["title_format"].format(number=self.number, date=self._long_date()),
            "number": self.number,
            "date": self.issued_at.strftime(rules["date_format"]),
            "payment_order": block["payment_order_block"],
            "party_labels": block["parties"],
            "supplier": self.supplier,
            "supplier_display": printed_legal_name(
                self.supplier.name, self.supplier.legal_form
            ),
            "buyer": self.buyer,
            "buyer_display": self.buyer.name,
            "bank_code": self.bank_code,
            "agreement": self.agreement,
            "columns": columns,
            "items": [
                {
                    "name": item.name,
                    "qty": f"{item.qty:g}",
                    "unit": self.unit,
                    "price": self._amount(item.price),
                    "sum": self._amount((item.qty * item.price).quantize(KOPIYKA)),
                }
                for item in self.line_items
            ],
            "totals_labels": block["totals"],
            "total": self._amount(self.total),
            "vat_total": self._amount(self.vat_total) if self.vat_payer else None,
            "count_line": block["totals"]["count_format"].format(
                count=len(self.line_items), amount=self._amount(self.total)
            ),
            "amount_in_words": amount_in_words_uk(self.total),
            "vat_in_words": amount_in_words_uk(self.vat_total) if self.vat_payer else None,
            "validity": self.validity,
            "instalment": instalment,
            "signature_labels": block["signature"],
            "signatory_name": self.signatory_name,
            "signatory_post": self.signatory_post,
            # This class carries no QR — 👁 none was observed on an invoice — and the renderer
            # requires the key on every context.
            "qr_payload": None,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this invoice.

        `amount` is the total. The only other money field an invoice can carry is
        `instalment_amount`, and it is populated exactly when the page prints the instalment term:
        no `amount_due`, no `fee`, no `total_charged`. `counterparty` is the SUPPLIER — the party
        opposite the claimant, as on every class — and `payer` is the buyer, which is the claimant.

        NOTHING RECORDS WHETHER IT WAS PAID, and that is the point of the class rather than a gap.
        `instalment_amount` is not that record either — see the field's own entry in `schemas.py`
        and the `schedule` bullet above: it says what one part of the obligation is, not that any
        part of it has been settled.
        `has_fiscal_number` and `qr_is_fiscal` are `False` because an invoice is not a fiscal
        document at all; a consumer classifying on a fiscal marker must not find one here.
        """
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.INVOICE,
            language="uk",
            currency="UAH",
            amount=self.total,
            instalment_amount=self.instalment_amount,
            date=self.issued_at.date(),
            counterparty=self.supplier.name,
            payer=self.buyer.name,
            # The invoice's own printed № — the number a payment's purpose cites where it cites
            # this document, and the value the `subject` axis compares `cites_document_no`
            # against on the payment beside it.
            document_code=self.number,
            line_items=self.line_items,
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            # From the RENDERER, like the boxes: neither is decided by the content class, and both
            # describe the page that was produced from it.
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def _draw_invoice_number(rng: random.Random, block: dict) -> str:
    """👁 One of the two number shapes an open invoice was seen to carry."""
    spec = rng.choice(block["number"]["formats"])
    return _draw_from_pattern(rng, spec["pattern"])


def _draw_phone(rng: random.Random) -> str:
    """A mobile number on a published prefix and drawn digits.

    📄 The prefix is one the national numbering plan assigns; the seven digits after it are drawn,
    so the number designates nobody in particular. A number taken off a document designates
    whoever holds it, which is why none is.
    """
    return f"+380 ({rng.choice(phone_prefixes())[1:]}) {rng.randint(0, 9_999_999):07d}"


def build_invoice(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    buyer_name: str,
    buyer_tax_id: str,
    address: str = "м. Київ",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
    schedule: str | None = None,
    settled_at: datetime | None = None,
    country: str = "UA",
) -> Invoice:
    """Build one Ukrainian рахунок на оплату.

    THE BASKET IS DRAWN EXACTLY AS A RECEIPT'S IS — same knobs, same meaning, and since the third
    basket-carrying class landed the same FUNCTION: `_draw_basket`, whose docstring carries the
    reasoning. `covered_only` for a `covered` claim, `coverage_target` for a mixed one.

    `identity` is the SUPPLIER's `PartyIdentity` — its code, its account and the bank holding it —
    drawn once for the claim so that the payment document settling this invoice names the same
    party by the same numbers. Required rather than defaulted: a builder that quietly drew its own
    would produce a document that renders perfectly and agrees with nothing.

    `buyer_name` and `buyer_tax_id` are the CLAIMANT's — an invoice is addressed to somebody, and an
    invoice addressed to anybody else would evidence nothing about the persona filing the claim.
    This is where an invoice differs structurally from a receipt: a till receipt names no buyer
    because the payer is standing at the till, while an offer to pay has to say to whom it is made.

    🔴 `settled_at` IS THE DATE THE CLAIM'S MONEY MOVED, AND IT BOUNDS A PRINTED TERM. The validity
    line — 📄 «Рахунок дійсний до X р.» — is a CONDITION OF THE OFFER, and an offer settled after it
    lapsed is not the obligation the payment discharged: a seller reissues a lapsed invoice rather
    than banking against it. Measured over eight seeds before this parameter existed: of 490
    invoices, 182 printed the line and 100 of those were paid later than the date they printed — 27
    of them on claims the label calls `covered`. Nothing in policy.yaml reads the line, so no
    verdict moved and nothing noticed; a consumer that learned to read it would have rejected those
    claims and been right, which makes it a defect of this generator rather than noise.

    `None` is for a document with no settlement to respect — a builder called directly, and the
    mock-ups — and then the drawn window is the whole of the span.

    🔴 `schedule` IS NAMED BY THE PLAN AND NEVER DRAWN HERE, unlike every other optional requisite
    of this class. The others are variation — an agreement line, a telephone, a validity — and a
    builder may draw them because no label depends on which way they come out. This one decides
    whether the page carries the marker `policy_engine` reads to tell `partially_paid` from
    `amount_mismatch`, so drawing it would let the builder choose a claim's verdict. It is a
    keyword of `claim_planner.ClaimPlan`, and `None` is the ordinary invoice payable in one.
    """
    rules = jurisdiction(country)
    block = rules["invoice"]
    if schedule is not None and schedule not in partial_payment_schedules():
        raise ValueError(
            f"config/generation.yaml declares no payment schedule {schedule!r}; it has "
            f"{sorted(partial_payment_schedules())}"
        )
    vat_payer = vendor_is_vat_payer(vendor)

    # -- what was bought. The receipt's own draw, called with the receipt's own arguments.
    items = _draw_basket(
        rng,
        document="an invoice",
        category_id=category_id,
        vendor=vendor,
        vat_payer=vat_payer,
        covered_only=covered_only,
        coverage_target=coverage_target,
        item_count=item_count,
    )

    # -- the tax, computed from the letters and THEN the letters dropped. ⛔ The observed table has
    # no per-line letter column, so a label carrying one would be unreadable from the image; the
    # figure itself is the same one a receipt would print, which is why it is taken first.
    vat_total = sum(
        (line.vat for line in _build_tax_lines(items, vat_payer=vat_payer)), Decimal(0)
    ).quantize(KOPIYKA)
    # `model_copy` and not `dataclasses.replace`: `LineItem` is a pydantic model. The first
    # version of this line used `replace` and raised on the first render, which is the cheapest
    # way this could have failed.
    items = [item.model_copy(update={"vat_letter": None}) for item in items]

    # -- who is selling. 👁 The supplier block names the firm, its code, its address, sometimes a
    # telephone, and always an account with the bank holding it. Every one of those requisites but
    # the address and the telephone is the CLAIM's, not this page's: an invoice and the payment
    # settling it name one seller, and naming it by four independently drawn numbers is what made
    # the pair unlinkable. 👁 The bank is printed twice on this form — beside the account in the
    # supplier block and again in the payment-order sample at the head — and both read the same
    # value here, which is why the sample block takes `supplier.bank` rather than a draw.
    is_sole_trader = vendor["legal_form"] == _SOLE_TRADER
    supplier = InvoiceParty(
        name=vendor["name"],
        legal_form=vendor["legal_form"],
        code=identity.tax_code,
        code_label=block["parties"]["supplier_code_label"],
        address=address,
        phone=_draw_phone(rng) if rng.random() < invoice_share("phone") else None,
        account=identity.account,
        bank=identity.bank_name,
    )
    # ⚠️ The buyer is a natural person and carries a РНОКПП. 👁 The observed invoice was addressed to
    # a company; the narrowing is declared in config/labelling-schema.yaml.
    buyer = InvoiceParty(
        name=buyer_name,
        legal_form="PERSON",
        code=buyer_tax_id,
        code_label=block["parties"]["buyer_code_label"],
    )

    agreement = None
    if rng.random() < invoice_share("agreement"):
        # 👁 «№ Д-27/25 від 10.05.2025 надання послуг» — a number, a date and a subject in words.
        signed = issued_at - timedelta(days=rng.randint(10, 400))
        agreement = (
            f"№ Д-{rng.randint(1, 999)}/{signed.year % 100} "
            f"від {signed.strftime(rules['date_format'])} надання послуг"
        )

    validity = None
    if rng.random() < invoice_share("validity"):
        until = issued_at + timedelta(days=rng.randint(*invoice_count_range("validity_days")))
        # 🔴 THE OFFER STILL STANDS ON THE DAY IT IS SETTLED, and the drawn window is a FLOOR on the
        # span rather than the whole of it for exactly that reason. The draw happens first and
        # unconditionally, so a seed's stream is the same whether or not the claim's payment outruns
        # the window — the printed date moves, and nothing else about the run does. Compared as a
        # DATE: the line prints a day, and a payment on the last day of the offer is inside it.
        if settled_at is not None and settled_at.date() > until.date():
            until = settled_at
        validity = block["validity_format"].format(date=until.strftime(rules["date_format"]))

    # 📄 A sole trader signs in their own name and states no post; a company names the post of the
    # authorized person. The name is a surname with initials — ⛔ narrower than the observed full
    # name, and declared as such where the format lives.
    signatory_name = (
        vendor["name"] if is_sole_trader else personal_signatory(rng, rules["language"])
    )
    return Invoice(
        number=_draw_invoice_number(rng, block),
        issued_at=issued_at,
        supplier=supplier,
        buyer=buyer,
        vat_payer=vat_payer,
        line_items=items,
        vat_total=vat_total,
        # 👁 Services are counted in «посл.», goods in «шт.». Read from what the vendor sells rather
        # than drawn: a gym membership is not measured in pieces.
        unit=block["units"]["service" if _sells_services(vendor) else "goods"],
        agreement=agreement,
        validity=validity,
        schedule=schedule,
        signatory_name=signatory_name,
        signatory_post=(
            None if is_sole_trader else rng.choice(block["signature"]["posts"])
        ),
        bank_code=identity.bank_code,
        decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
    )


# Vendor profiles that sell a SERVICE rather than a thing, for the unit column. Read from the
# profile because that is where what a vendor sells already lives; a profile absent from this set
# sells goods, which is the safe default — «шт.» beside a service reads as a clerical slip, while
# «посл.» beside a bottle of vitamins reads as a different document.
_SERVICE_PROFILES = frozenset(
    {
        "insurer", "language_school", "private_tutor", "exam_centre", "training_centre",
        "online_learning_platform", "conference_organizer", "gym", "pool", "fitness_studio",
        "mental_health_clinic", "therapy_practice", "nutrition_practice", "art_studio",
        "music_school", "photo_school", "hobby_club",
    }
)


def _sells_services(vendor: dict) -> bool:
    return vendor["profile"] in _SERVICE_PROFILES


def personal_signatory(rng: random.Random, language: str) -> str:
    """The person who wrote the invoice out, as «Прізвище І. Б.».

    ⛔ NARROWER THAN OBSERVED: the real invoice prints a full given name and patronymic. This form
    carries the same information for extraction — a personal name in the signature block — while
    drawing only from the narrowed high-frequency surname pool, so a published image never names a
    person more specifically than that pool justifies. Same rule and same pool as a sole trader's
    printed name; see `personal_names` in config/generation.yaml.
    """
    return sole_trader_name(rng, "UA" if language == "uk" else language.upper())


# =============================================================================
# Non-fiscal sales slip — товарний чек
# =============================================================================
#
# 🔴 THE DOCUMENT THE FISCALITY RULE HAS NEVER HAD A TEST CASE FOR. A verifier is expected to
# treat a NEGATIVE fiscality signal as overriding every positive one, and until this class landed
# no document of the corpus carried anything negative to override with: every archetype either
# printed a full fiscal identity or belonged to a class nobody would look for one on. This one is
# the hard case — the basket, the arithmetic, the totals block and the column layout are a fiscal
# receipt's, and 📄 the difference is exactly the two requisites the tax service says such a
# document omits.
#
# 📄 THE FORM IS NOT DEFINED BY LAW and the sources are named where the strings live —
# `receipt.non_fiscal` in config/fiscal-rules.yaml. What matters here is what follows from them:
# the seller may not be registered for ПДВ (a payer is obliged to use a register), so no line
# carries a ПДВ letter and no tax block is printed; and 📄 ст. 9 of the accounting law obliges the
# document to name the person responsible and carry their signature, which no fiscal receipt does.
#
# ⛔ IT PROVES NO PAYMENT, which is policy.yaml's `document_evidence` and not this class's opinion
# of itself. A claim evidenced by one alone is `not_proof_of_payment` — see
# `claim_planner.EvidenceIntent.PAYMENT_GAP`, which is what plans such a claim.


@dataclass(frozen=True)
class NonFiscalReceipt:
    """One товарний чек, complete but not yet rendered.

    `Seller` IS REUSED AND `PrroReceipt` IS NOT, and the split is the sources' rather than a
    convenience: 📄 the tax service says this document's content is the FISCAL RECEIPT'S FORM less
    the fiscal number and the fiscal wording, so the party block is literally the receipt's — one
    identifier line, no «ПН», because its seller cannot be a registered payer. What differs is the
    fiscal identity, which this class does not have a field for at all. Modelling the difference
    as `None` on the receipt class would have made "no fiscal number" a value of a document that
    has one, and every consumer of `PrroReceipt` would then carry a branch for a class it never
    sees.

    ⛔ NO `vat_row_form`, NO `tax_lines`, NO `acquiring`, NO `qr_payload`, and none of them is an
    omission: a non-payer's receipt has no tax block to take a form, 📄 a card sale is a settlement
    operation that obliges the seller to use a register, and 📄 the QR is a requisite of the fiscal
    form. The absences ARE the archetype.
    """

    seller: Seller
    issued_at: datetime
    title: str
    receipt_number: str
    line_items: list[LineItem]
    total: Decimal
    # СУМА and ДО СПЛАТИ, exactly as on the fiscal form: 📄 lines 20 and 24, differing by the two
    # between them. Both adjustments are zero in this version for the reason stated at
    # `PrroReceipt.amount_due` — the policy says nothing about distributing a discount across
    # covered and non-covered lines — and the lines are printed all the same, because this
    # document's form IS that form.
    discount: Decimal
    rounding: Decimal
    amount_in_words: str
    payment_method: str
    # 📄 ст. 9 of the law on accounting № 996-XIV: the person responsible for the operation. A sole
    # trader is that person; a company names an authorized one, as on an invoice.
    issuer_name: str
    footer: str
    decimal_separator: str

    @property
    def amount_due(self) -> Decimal:
        """ДО СПЛАТИ — the basket less any discount, plus cash rounding.

        DERIVED for the same reason `PrroReceipt.amount_due` is: two amounts that must agree
        should not be two stored numbers. It equals `total` while both adjustments are zero, and
        the consequence for a consumer is the one recorded against that field — while the two
        coincide the field discriminates nothing and its accuracy is not a metric.
        """
        return (self.total - self.discount + self.rounding).quantize(KOPIYKA)

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted."""
        rules = jurisdiction("UA")
        block = rules["receipt"]["non_fiscal"]
        return {
            "title": self.title,
            "seller": self.seller,
            "seller_display": legal_name(self.seller),
            "date": self.issued_at.strftime(rules["date_format"]),
            "time": self.issued_at.strftime(rules["time_format"]),
            "receipt_number": self.receipt_number,
            "items": [
                {
                    "name": item.name,
                    "qty": f"{item.qty:g}",
                    "price": self._amount(item.price),
                    "sum": self._amount((item.qty * item.price).quantize(KOPIYKA)),
                }
                for item in self.line_items
            ],
            "total": self._amount(self.total),
            "discount": self._amount(self.discount),
            "rounding": self._amount(self.rounding),
            "amount_due": self._amount(self.amount_due),
            "totals_labels": rules["receipt"]["totals_labels"],
            "amount_in_words": self.amount_in_words,
            "payment_method": self.payment_method,
            "issuer_label": block["issuer_label"],
            "issuer_name": self.issuer_name,
            "signature_label": block["signature_label"],
            "footer": self.footer,
            # 📄 No QR: it is a requisite of the FISCAL form. The renderer requires the key on
            # every context, and `None` is how a template says the document carries none.
            "qr_payload": None,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this slip.

        🔴 THE THREE FISCALITY FLAGS ARE ALL FALSE, AND THEY ARE THE POINT OF THE RECORD. A
        consumer classifying on a fiscal marker must find none here, on a page that otherwise
        looks like a fiscal receipt line for line. `payer` is `None`: 📄 the form this document
        follows has no buyer field, and the person who paid was standing at the counter.

        ⛔ NOTHING RECORDS THE NEGATIVE MARKER ITSELF. There is no such field on `DocGroundTruth`,
        and adding one is RC-08 in config/labelling-schema.yaml — still an open decision, because
        what this document carries is a positive TITLE plus an ABSENCE of requisites, which a
        field shaped as "marker text and its position" cannot hold. The absence is real ground
        truth and it is expressed by the three flags below rather than invented as a string.
        """
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.NON_FISCAL_RECEIPT,
            language="uk",
            currency="UAH",
            amount=self.total,
            amount_due=self.amount_due,
            date=self.issued_at.date(),
            counterparty=self.seller.name,
            line_items=self.line_items,
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def build_non_fiscal_receipt(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    address: str = "м. Київ",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
) -> NonFiscalReceipt:
    """Build one товарний чек.

    THE BASKET IS THE RECEIPT'S AND THE INVOICE'S — `_draw_basket`, same knobs, same meaning —
    because coverage is a property of what was bought and not of the class that lists it.

    🔴 THE SELLER MUST NOT BE REGISTERED FOR ПДВ, and this refuses rather than printing one that
    is. 📄 A registered payer is obliged to use a cash register, so a seller who issues this
    document is a non-payer; a payer issuing one would be a page whose own requisites say it
    should not exist, and every VAT decision below — no «ПН» line, no letter on any line, no tax
    block — would then contradict the vendor record behind it. The caller chooses the vendor
    (`assembler._pick_vendor`), so the constraint belongs at that choice and this is the guard
    that keeps it from being silently skipped.

    `identity` carries the seller's identification code, drawn once for the claim, exactly as on
    the other classes. A claim evidenced by this document alone has no second page to agree with —
    the parameter is required all the same, so that ONE mechanism decides who a seller is.
    """
    if vendor_is_vat_payer(vendor):
        raise ValueError(
            f"vendor {vendor['name']!r} is registered for ПДВ, and 📄 a registered payer is "
            "obliged to use a cash register — so it cannot be the seller on a товарний чек. "
            "Choose a non-payer vendor for this archetype; see `assembler._pick_vendor`."
        )

    rules = jurisdiction("UA")
    receipt_rules = rules["receipt"]
    block = receipt_rules["non_fiscal"]

    items = _draw_basket(
        rng,
        document="a sales slip",
        category_id=category_id,
        vendor=vendor,
        # The seller is a non-payer by the guard above, so no line carries a ПДВ letter and the
        # page prints no tax block. One statement, not two: the flag is not read from the vendor
        # again here, because the guard has already settled what it can be.
        vat_payer=False,
        covered_only=covered_only,
        coverage_target=coverage_target,
        item_count=item_count,
    )
    total = line_items_total(items)

    is_sole_trader = vendor["legal_form"] == _SOLE_TRADER
    id_code_rules = rules["identifiers"]["rnokpp" if is_sole_trader else "edrpou"]
    seller = Seller(
        name=vendor["name"],
        legal_form=vendor["legal_form"],
        address=address,
        vat_payer=False,
        tax_code=identity.tax_code,
        tax_code_label=id_code_rules["label"],
        # 📄 A ПДВ payer is obliged to use a register, so there is no «ПН» line to print at all.
        # `None` here is the same statement the guard above makes, carried onto the page.
        vat_number=None,
        vat_number_label=rules["identifiers"]["vat_number"]["label"],
    )

    return NonFiscalReceipt(
        seller=seller,
        issued_at=issued_at,
        title=block["title"],
        receipt_number=_draw_from_pattern(rng, block["number"]["pattern"]),
        line_items=items,
        total=total,
        discount=Decimal(0),
        rounding=Decimal(0),
        amount_in_words=amount_in_words_uk(total),
        # 🔴 CASH, AND IT IS NOT A COSMETIC CHOICE. 📄 A card sale is a settlement operation that
        # obliges the seller to use a register, so a slip issued without one records cash. This is
        # the first archetype of the corpus to print «ГОТІВКА» — the second entry of
        # `payment_method_labels`, which was unreachable until this class landed — and the value
        # is read from config rather than written here.
        payment_method=rules["acquiring_block"]["payment_method_labels"][1],
        # 📄 ст. 9 № 996-XIV. A sole trader signs in their own name; a company names an authorized
        # person, drawn the same way an invoice's signatory is.
        issuer_name=(
            vendor["name"] if is_sole_trader else personal_signatory(rng, rules["language"])
        ),
        footer=receipt_rules["footer"],
        decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
    )


# =============================================================================
# The destination tax of the EU pages — one draw, two document classes
# =============================================================================


@dataclass(frozen=True)
class TaxTreatment:
    """What an EU page's totals block says about the tax — one of three drawn FORMS.

    A digital service is taxed where its consumer is, so which form a page takes is DERIVED from
    the buyer's country rather than being a new fact about the seller or the persona: a Ukrainian
    resident's page adds the tax on top at the UA rate parameter, a relocated buyer's at their
    own country's, and a business customer under reverse charge sees no charge at all. The three:

    * `tax_on_top` — a row and a positive figure between the subtotal and the total, so the
      printed total EXCEEDS the line items: `total = subtotal + tax`. 👁 The form the author's
      own cross-border platform receipts print, and the one the corpus lacked by construction
      while every EU page reproduced the out-of-scope convention only;
    * `out_of_scope` — no row (`label is None`): the old page exactly, whose licence to omit the
      block each EU class argues in its own terms;
    * `reverse_charge` — a row whose caption names the mechanism, a figure of 0.00, and the
      sentence `note()` returns in the foot.

    🔴 FORM, NOT VERDICT, which is what licenses a drawn share at all: the oracle's
    `policy_engine.covered_total` runs over the LINE ITEMS alone, and the payment document beside
    a subject page is told the PRINTED total whichever form came out
    (`assembler._amount_the_payment_states` reads the subject's `amount`) — so the draw moves
    pixels and the `tax` label field, never a verdict.

    ⛔ THE RATES BEHIND `amount` ARE PROJECT PARAMETERS, not statements about any jurisdiction's
    tax law — `tax_on_top` in config/fiscal-rules.yaml carries the boundary and the one cited
    exception (Ukraine's 20%).
    """

    form: str
    # `label` and `amount` are None together — a row is a caption AND a figure, and an absent row
    # is absent whole, exactly as its box is absent from the labelling rather than empty.
    label: str | None
    amount: Decimal | None

    def note(self, *, out_of_scope: str | None) -> str | None:
        """The one tax-status sentence of the page's foot, or None for a foot without one.

        The reverse-charge sentence is the treatment's own — one wording for every EU class,
        read from config. What the OUT-OF-SCOPE form says is the CLASS's business, which is why
        it arrives as an argument: the invoice names Articles 44 and 59 there, and the platform
        receipt says nothing — a receipt records a payment and argues no law.
        """
        if self.form == "reverse_charge":
            return str(tax_on_top_rules()["reverse_charge_note"])
        if self.form == "out_of_scope":
            return out_of_scope
        return None


def draw_tax_treatment(
    rng: random.Random, *, buyer_country: Country, subtotal: Decimal
) -> TaxTreatment:
    """Draw which of the three forms this page takes, and compute what its row prints.

    ONE WEIGHTED DRAW WHATEVER THE OUTCOME, so a seed's stream does not depend on which form
    came out — the same discipline every conditional requisite of this module follows. The
    weights are `eu_tax_treatment` in config/generation.yaml, in file order; the captions and
    the rate parameters are `tax_on_top` in config/fiscal-rules.yaml.

    The figure rounds half up at the kopiyka, as `EuInvoice.instalment_amount` does: a derived
    printed amount takes the module's one explicit rounding rather than the context default.
    """
    shares = eu_tax_treatment_shares()
    form = rng.choices(list(shares), weights=list(shares.values()), k=1)[0]
    rules = tax_on_top_rules()
    if form == "tax_on_top":
        by_country = rules["rate_by_buyer_country"]
        if buyer_country.value not in by_country:
            raise KeyError(
                f"config/fiscal-rules.yaml declares no `tax_on_top.rate_by_buyer_country` entry "
                f"for {buyer_country.value!r}; it has {sorted(by_country)}"
            )
        rate = Decimal(str(by_country[buyer_country.value]))
        return TaxTreatment(
            form=form,
            label=rules["label_format"].format(rate=f"{float(rate):g}"),
            amount=(subtotal * rate / 100).quantize(KOPIYKA, rounding=ROUND_HALF_UP),
        )
    if form == "reverse_charge":
        return TaxTreatment(
            form=form, label=str(rules["reverse_charge_label"]), amount=Decimal("0.00")
        )
    if form == "out_of_scope":
        return TaxTreatment(form=form, label=None, amount=None)
    # A form named in the shares that nothing here prints would silently become a no-row page —
    # a config edit choosing a form by omission, which is exactly what this refuses.
    raise KeyError(
        f"config/generation.yaml draws the tax-treatment form {form!r}, and "
        "`content_builder.draw_tax_treatment` prints no such form"
    )


@dataclass(frozen=True)
class PlatformReceipt:
    """One platform receipt — an online platform's own page for a paid order.

    THE FIRST CLASS IN TWO DIMENSIONS AT ONCE: the corpus's first English document and its
    first euro one, which is the reason the archetype exists — the contract's reference
    profile records `currency_UAH` and `language_uk` at 100% of documents and names the
    single-valued dimensions a finding.

    🔴 WHAT IT LACKS IS ITSELF EVIDENCED, by the method the non-fiscal slip used on the
    Ukrainian fiscal form. 📄 Article 226 of Directive 2006/112/EC lists the particulars a
    VAT invoice must carry — the supplier's address, the supplier's and the customer's VAT
    identification numbers — and this document declares itself not to be one («This is not
    a VAT invoice.», 👁 four independent commentators describe the wording on real
    platform receipts) and carries none of them. So the seller block is one line, the
    mark; there is no `Seller` here at all, because every other field of that class is a
    requisite this page is licensed to omit.

    ⚠️ A TAX ROW IS NOT ONE OF THOSE ABSENCES ANY MORE. The EU variant draws one of the
    three forms of `TaxTreatment` — 👁 the author's own cross-border receipts add the
    destination tax ON TOP of net prices and still say they are not a VAT invoice, the
    declaration being about the DOCUMENT and the row about the TAX. So the total may
    exceed the line items, `tax_amount` below says by how much, and the declaration
    stays printed whatever the form.

    ⛔ NO QR, NO FISCAL IDENTITY, NO AMOUNT IN WORDS — absences, not gaps. The class
    establishes both facts of `document_evidence` (the lines say what was bought, the
    paid caption and the card say money moved) while being fiscal like neither receipt
    class: that pairing is what the archetype puts into the data.
    """

    # The vendors.json jurisdiction block the page's captions, formats, language and
    # currency come from — "EU" for the English euro variant, "UA" for the Ukrainian one.
    # ONE CLASS, TWO JURISDICTIONS, THREE AXES: language and currency are data, and the
    # third axis is LAW — the UA seller is a domestic company whose requisites and tax row
    # are ordinary, the EU page disclaims them all. The optional fields below are that
    # third axis: all None on the EU variant, filled on the UA one.
    jurisdiction_code: str
    seller_name: str
    buyer_name: str
    buyer_country: str
    issued_at: datetime
    receipt_number: str
    # The last digits of the paying card behind bullets, as the class prints it. The
    # digits identify nothing: four digits are a tail every issued range shares.
    card_masked: str
    line_items: list[LineItem]
    # THE PRINTED GRAND TOTAL — the line items PLUS any destination tax the EU variant added on
    # top (`tax_amount` below), so `total > Σ lines` exactly where a tax row is printed. The
    # subtotal is not stored: it is Σ over `line_items` by construction, and `render_context`
    # derives it so the two figures cannot drift.
    total: Decimal
    # What the header prints where the EU page prints the bare mark alone: the seller's
    # legal name («ТОВ «Prometheus»»), address, identification code and ПН — ordinary
    # requisites of a domestic company, each with the caption fiscal-rules.yaml declares.
    seller_display: str | None = None
    seller_address: str | None = None
    seller_tax_code: str | None = None
    seller_tax_code_label: str | None = None
    seller_vat_number: str | None = None
    seller_vat_number_label: str | None = None
    # 📄 «У т.ч. ПДВ» — the tax CONTAINED in a gross price, not added to it, exactly as the
    # invoice states it: subtotal and total are one figure and the row sits between them
    # for information. None where the page asserts no tax treatment at all.
    vat_label: str | None = None
    vat_amount: Decimal | None = None
    # 👁 Real platform receipts in the English-speaking practice commonly say so in as many
    # words; no Ukrainian equivalent was found in any public source, so the UA variant
    # carries None rather than a translation nothing evidences.
    not_a_tax_invoice_note: str | None = None
    # THE DESTINATION TAX THE EU VARIANT MAY ADD ON TOP — `TaxTreatment`'s caption and figure,
    # None together where the drawn form prints no row, and None on the UA variant always.
    # ⛔ DISTINCT FROM `vat_label`/`vat_amount` ABOVE, deliberately: that pair states the tax
    # CONTAINED in a Ukrainian gross price and never moves the total, this pair is ADDED and
    # does — one field for both would give a key two meanings, and a consumer subtracting it
    # would corrupt exactly the documents where the subtraction is wrong.
    tax_label: str | None = None
    tax_amount: Decimal | None = None
    # The reverse-charge sentence of the foot (`TaxTreatment.note`), under the same box name the
    # invoice's tax-status sentence carries. ⛔ The out-of-scope form prints NOTHING here: a
    # receipt records a payment and argues no law, which is where it differs from the invoice.
    vat_note: str | None = None
    # UA draws its decimal separator per document, as every Ukrainian class does; the EU
    # block declares exactly one, so the draw collapses there.
    decimal_separator: str = "."

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction(self.jurisdiction_code)["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{self.decimal_separator}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted.

        The captions come from `platform_receipt.labels` in config/fiscal-rules.yaml —
        moved there from the mock-up script when the archetype was connected, the same
        migration the товарний чек's strings made. The body is
        templates/platform_receipt.jinja, shared with the Ukrainian variant of the class;
        what makes this document English and EUR is this context, not the markup.
        """
        rules = jurisdiction(self.jurisdiction_code)
        block = rules["platform_receipt"]
        return {
            "language": rules["language"],
            "labels": block["labels"],
            "currency": rules["currency"],
            # On the EU variant the seller block is the mark and nothing beside it — see
            # the class docstring; the UA variant fills the ordinary requisites. The keys
            # exist either way because the shared body asks conditionally.
            "seller": {
                "name": self.seller_display or self.seller_name,
                "address": self.seller_address,
                "tax_code": self.seller_tax_code,
                "tax_code_label": self.seller_tax_code_label,
                "vat_number": self.seller_vat_number,
                "vat_number_label": self.seller_vat_number_label,
            },
            "buyer": {"name": self.buyer_name, "country": self.buyer_country},
            "receipt_number": self.receipt_number,
            "date": self.issued_at.strftime(rules["date_format"]),
            "card_masked": self.card_masked,
            "lines": [
                {
                    "name": item.name,
                    # ⛔ No period sub-line: the drawn name carries everything the label
                    # knows, and a second printed string absent from the label would be
                    # text `reference_text` carries and nothing accounts for.
                    "period": None,
                    "qty": f"{item.qty:g}",
                    "amount": self._amount((item.qty * item.price).quantize(KOPIYKA)),
                }
                for item in self.line_items
            ],
            # Derived, not read from `total`: the subtotal is the LINE ITEMS' sum on both
            # variants, and the total below may exceed it by the EU variant's tax on top.
            "subtotal": self._amount(line_items_total(self.line_items)),
            # 📄 Present only where the supplier states the contained tax: the UA seller's gross
            # prices carry it, and the row never moves the total.
            "vat": (
                {"label": self.vat_label, "amount": self._amount(self.vat_amount)}
                if self.vat_amount is not None
                else None
            ),
            # The EU variant's drawn row — see `TaxTreatment`. Never present beside `vat` above:
            # one states a tax added, the other a tax contained, and a page asserting both would
            # contradict itself.
            "tax": (
                {"label": self.tax_label, "amount": self._amount(self.tax_amount)}
                if self.tax_amount is not None
                else None
            ),
            "total": self._amount(self.total),
            "not_a_tax_invoice_note": self.not_a_tax_invoice_note,
            "vat_note": self.vat_note,
            "support_note": None,
            "qr_payload": None,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this receipt.

        `currency` is the record's load-bearing field: every amount on it is in EUR, and
        the oracle converts at the vendored rate — recording what it applied in the
        claim's `fx_rates` — rather than refusing, which is the decision that let this
        archetype into the corpus at all. `payer` carries the buyer the page names; the
        three fiscality flags are false, as on the slip, and for the same reason: a
        consumer classifying on a fiscal marker must find none here. `amount` is the
        PRINTED total and `tax` the row the EU variant may add on top — labelled apart so
        `amount ≠ Σ line items` is measurable where it is true; ⛔ the UA variant's
        contained-VAT row is NOT `tax` (see the field in `schemas.DocGroundTruth`).
        """
        rules = jurisdiction(self.jurisdiction_code)
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.PLATFORM_RECEIPT,
            language=rules["language"],
            currency=rules["currency"],
            amount=self.total,
            tax=self.tax_amount,
            date=self.issued_at.date(),
            counterparty=self.seller_name,
            payer=self.buyer_name,
            line_items=self.line_items,
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def build_platform_receipt(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    buyer_name: str,
    buyer_tax_id: str,
    buyer_country: Country = Country.UA,
    address: str = "",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
    jurisdiction_code: str = "EU",
) -> PlatformReceipt:
    """Build one platform receipt — English and EUR by default, Ukrainian and UAH for
    the domestic variant (`jurisdiction_code="UA"`), one class either way.

    THE BASKET IS THE RECEIPT'S AND THE INVOICE'S — `_draw_basket`, same knobs, same
    meaning — with the two axes that make this class what it is: `language="en"` selects
    the English template list policy.yaml carries for every kind, and `currency="EUR"`
    selects `price_ranges_eur`, whose comment states the arithmetic tying it to the UAH
    ranges the planner sizes baskets by.

    WHAT THE EU VARIANT ACCEPTS AND DELIBERATELY DOES NOT PRINT, because the assembler
    hands these to every archetype of their kind and a page must not grow a requisite to
    use them up — while the UA variant prints the first and third as its ordinary
    requisites:

    * `identity` — the seller's drawn identity. 📄 A supplier's identification numbers are
      invoice particulars, and the EU page declares itself not an invoice; the UA seller
      is a domestic company and prints them.
    * `buyer_tax_id` — 📄 the customer's VAT identification number is on the same Article
      226 list, and 👁 a platform account has a name and a country, not a tax number.
      Printed by NEITHER variant, ⛔ the reverse-charge form included.
    * `address` — the seller's address slot. Blank on the EU page; the UA page prints it.

    `buyer_country` IS THE AXIS THE EU VARIANT'S TAX TREATMENT DERIVES FROM — the persona's own
    country, printed under the buyer's name on both variants and deciding which rate parameter a
    tax-on-top page charges (see `TaxTreatment`). Defaulted to UA because every persona of
    today's corpus is a Ukrainian resident. ⛔ The UA variant reads it for the printed name only:
    its tax statement is the contained-VAT row, a fact of the SELLER's registration, not of the
    buyer's country.
    """
    del buyer_tax_id  # accepted, never printed — see the docstring

    rules = jurisdiction(jurisdiction_code)
    block = rules["platform_receipt"]
    domestic = jurisdiction_code == "UA"

    items = _draw_basket(
        rng,
        document="a platform receipt",
        category_id=category_id,
        vendor=vendor,
        # ⛔ NO VAT LETTER ON ANY LINE, ON EITHER VARIANT, and for two different reasons
        # that end in one flag: the EU page prints no tax requisite of any kind (its own
        # footer is the licence), and the UA page prices VAT-inclusive and states the tax
        # once at the foot — the same convention the invoice follows, whose lines carry no
        # letter either. The letter is a requisite of the FISCAL receipt's line.
        vat_payer=False,
        covered_only=covered_only,
        coverage_target=coverage_target,
        item_count=item_count,
        language=rules["language"],
        currency=rules["currency"],
    )
    total = line_items_total(items)

    number_rules = block["receipt_number"]
    receipt_number = number_rules["separator"].join(
        _draw_from_pattern(rng, number_rules["group_pattern"])
        for _ in range(int(number_rules["groups"]))
    )

    # The third axis — law. The Ukrainian seller is a domestic company: its legal name,
    # address, identification code and (for a registered payer) the ПН and the contained
    # VAT are ordinary requisites, printed exactly as the invoice prints them. The EU page
    # disclaims all of them in one sentence, carried in `not_a_tax_invoice_note`.
    if domestic:
        vat_payer = vendor_is_vat_payer(vendor)
        vat_rate = Decimal(str(rules["vat_rates"]["standard"]))
        seller_extras = dict(
            seller_display=printed_legal_name(vendor["name"], vendor["legal_form"]),
            seller_address=address,
            seller_tax_code=identity.tax_code,
            seller_tax_code_label=rules["identifiers"][
                "rnokpp" if vendor["legal_form"] == _SOLE_TRADER else "edrpou"
            ]["label"],
            seller_vat_number=identity.vat_number if vat_payer else None,
            seller_vat_number_label=(
                rules["identifiers"]["vat_number"]["label"] if vat_payer else None
            ),
            vat_label=(
                block["vat_label_format"].format(rate=f"{float(vat_rate):g}")
                if vat_payer
                else None
            ),
            # 📄 The tax CONTAINED in a gross price: gross × rate / (100 + rate).
            vat_amount=(
                (total * vat_rate / (100 + vat_rate)).quantize(KOPIYKA)
                if vat_payer
                else None
            ),
            not_a_tax_invoice_note=None,
            decimal_separator=rng.choice(rules["number_format"]["decimal_separator_variants"]),
        )
    else:
        del identity, address  # not printed on the EU page — see the docstring
        # The drawn tax treatment — the same draw the cross-border invoice makes, at the same
        # point relative to the basket, so the two classes vary the same way. The tax-on-top
        # form lifts the PRINTED total above the line items; the not-a-tax-invoice declaration
        # stays whatever the form, because it is a statement about the DOCUMENT, not the tax.
        treatment = draw_tax_treatment(
            rng, buyer_country=buyer_country, subtotal=total
        )
        total = (total + (treatment.amount or Decimal(0))).quantize(KOPIYKA)
        seller_extras = dict(
            not_a_tax_invoice_note=block["not_a_tax_invoice_note"],
            tax_label=treatment.label,
            tax_amount=treatment.amount,
            # A receipt records a payment and argues no law, so the out-of-scope form prints no
            # sentence — only the reverse-charge form carries one here.
            vat_note=treatment.note(out_of_scope=None),
            decimal_separator=rules["number_format"]["decimal_separator_variants"][0],
        )

    return PlatformReceipt(
        jurisdiction_code=jurisdiction_code,
        # The bare mark, which is also what `counterparty` carries — the contract's
        # authoritative form on every class of this dataset.
        seller_name=vendor["name"],
        buyer_name=buyer_name,
        buyer_country=rules["country_names"][buyer_country.value],
        issued_at=issued_at,
        receipt_number=receipt_number,
        card_masked=block["card_mask_format"].format(tail=f"{rng.randint(0, 9999):04d}"),
        line_items=items,
        total=total,
        **seller_extras,
    )


# =============================================================================
# The cross-border invoice — an offer to pay, in euros
# =============================================================================


@dataclass(frozen=True)
class EuInvoice:
    """One invoice issued by a platform of the `EU` pool to a Ukrainian claimant.

    🔴 THE CLASS THAT MAKES A EURO CLAIM A PAIR. Every euro document before it was a
    `platform_receipt`, which proves BOTH facts and is therefore the whole of its claim: the
    oracle converted a page. This one proves the subject and no payment — 📄 an invoice is an
    offer to pay — so the claim it belongs to carries a payment document beside it, and the
    conversion runs through a cross-document check for the first time.

    📄 NOT A VAT INVOICE, AND FOR A DIFFERENT REASON FROM THE PLATFORM RECEIPT'S. That page
    declares itself not to be one; this one IS an invoice, and what its totals block says about
    the tax is DRAWN — one of the three forms `TaxTreatment` carries, derived from the buyer's
    country. The mass form adds the destination tax ON TOP, so `total = subtotal + tax` and the
    printed total exceeds the line items — honestly, which is what the corpus could not show
    before this form existed. The out-of-scope form is the old page exactly: no row, and the foot
    names Articles 44 and 59 of Directive 2006/112/EC, the provisions that place the supply
    outside the scope of EU VAT. The reverse-charge form prints a zero row and says who accounts
    for the tax. Three forms of one block, which is exactly the variation a consumer that
    classifies or checks arithmetic on a tax block has to survive.

    ⛔ NO PAYMENT STATUS, EVER. The class states an obligation and a date by which it should be
    settled; whether it was is what the document beside it establishes. A "Paid" mark here would
    make the class prove both facts and would contradict `document_evidence` in policy.yaml.
    """

    # THE BARE MARK, which is what `counterparty` carries on every class of this dataset. The
    # page prints `seller_display` — the same firm with the designation its own register gives it
    # — so the label and the image differ in exactly the way the contract's `normalization`
    # section says they may, and the payment document beside it names the party identically.
    seller_name: str
    seller_display: str
    buyer_name: str
    buyer_country: str
    issued_at: datetime
    # 🔴 A TERM OF THE OFFER, and it is bound by the day the claim's money moves: an offer whose
    # date has passed is not the obligation the payment discharged. `build_eu_invoice` pushes it
    # out to the settlement where a drawn window would fall short — the same rule the Ukrainian
    # invoice's validity line follows, and for the reason recorded there.
    due_at: datetime
    number: str
    line_items: list[LineItem]
    # The seller's account, which is what makes the invoice payable by the transfer beside it.
    # ⚠️ ORDINARY COMMERCIAL CONTENT AND NOT A TAX REQUISITE — Article 226 lists no bank details.
    # It is printed because a document nobody could pay is not an offer to pay.
    iban: str
    bank_name: str
    # THE ONE TAX-STATUS SENTENCE OF THE FOOT, or None for a foot the template omits whole. Which
    # sentence follows the drawn form (see `TaxTreatment.note`): the out-of-scope page names
    # Articles 44 and 59, the reverse-charge page says who accounts for the tax, and the
    # tax-on-top page prints NO sentence — its row IS the statement, and the out-of-scope wording
    # beside a charged rate would contradict the figure above it.
    vat_note: str | None
    # THE TAX ROW OF THE TOTALS BLOCK — `TaxTreatment`'s caption and figure, None together where
    # the drawn form prints no row. Kept as two plain fields rather than the treatment object so
    # the class stores exactly what the page prints, as every other class of this module does.
    tax_label: str | None = None
    tax_amount: Decimal | None = None
    # Present exactly when the plan asked for an obligation settled in parts — see
    # `claim_planner.STATES_AN_INSTALMENT_TERM`. Never drawn here: it decides a marker the oracle
    # reads, so a builder that drew it would be choosing a claim's verdict.
    schedule: str | None = None

    @property
    def subtotal(self) -> Decimal:
        """Σ over the line items — derived, for the reason every other total in this module is."""
        return line_items_total(self.line_items)

    @property
    def total(self) -> Decimal:
        """What the page asks to be paid: the line items plus any tax printed on top.

        🔴 NO LONGER Σ OVER THE LINE ITEMS ALONE, and that is the whole point of the tax-on-top
        form: a real cross-border page whose total exceeds its lines is honest, and a corpus in
        which `Σ lines = total` held on every document could not test a consumer against it.
        Derived rather than stored so the three figures cannot drift — the same reasoning as
        `PaymentConfirmation.total_charged`, whose `amount ≠ total_charged` this mirrors on the
        subject side.
        """
        return (self.subtotal + (self.tax_amount or Decimal(0))).quantize(KOPIYKA)

    @property
    def instalment_amount(self) -> Decimal | None:
        """What ONE PART of this obligation comes to, or `None` for an invoice payable in one.

        The Ukrainian invoice's rule, unchanged and deliberately so: the parts need not sum back
        to the total, because the page states the amount of the NEXT payment rather than a
        schedule of every one. See `Invoice.instalment_amount`, which carries the argument.
        """
        if self.schedule is None:
            return None
        return (self.total / partial_payment_schedules()[self.schedule]).quantize(
            KOPIYKA, rounding=ROUND_HALF_UP
        )

    @property
    def reference(self) -> DocumentReference:
        """How a payment document names this invoice — its number and the date it bears."""
        return DocumentReference(number=self.number, issued_at=self.issued_at)

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("EU")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        return f"{grouped}{rules['decimal_separator_variants'][0]}{fraction}"

    def render_context(self) -> dict:
        """Everything the template prints, already formatted."""
        rules = jurisdiction("EU")
        block = rules["invoice"]
        part = self.instalment_amount
        return {
            "language": rules["language"],
            "labels": block["labels"],
            "currency": rules["currency"],
            "title": block["title"],
            "seller_name": self.seller_display,
            "buyer": {"name": self.buyer_name, "country": self.buyer_country},
            "number": self.number,
            "date": self.issued_at.strftime(rules["date_format"]),
            "due_date": block["due_date_format"].format(
                date=self.due_at.strftime(rules["date_format"])
            ),
            "lines": [
                {
                    "name": item.name,
                    "qty": f"{item.qty:g}",
                    "price": self._amount(item.price),
                    "amount": self._amount((item.qty * item.price).quantize(KOPIYKA)),
                }
                for item in self.line_items
            ],
            "subtotal": self._amount(self.subtotal),
            # The row between them — see `TaxTreatment`. A dict or None, so the template's
            # conditional mirrors the class: an absent row is absent whole.
            "tax": (
                None
                if self.tax_amount is None
                else {"label": self.tax_label, "amount": self._amount(self.tax_amount)}
            ),
            "total": self._amount(self.total),
            # A caption and a money value, so the amount carries a box of its own rather than
            # being scored inside a sentence.
            "instalment": None if part is None else {
                "caption": block["instalment_caption_format"].format(
                    period=block["instalment_periods"][self.schedule]
                ),
                "amount": self._amount(part),
            },
            "iban": self.iban,
            "bank_name": self.bank_name,
            "reference": block["payment_reference_format"].format(number=self.number),
            "vat_note": self.vat_note,
            # This class carries no QR, and the renderer requires the key on every context.
            "qr_payload": None,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this invoice — the Ukrainian invoice's record in another currency.

        Same class, same fields, same meanings: `amount` is the PRINTED total — the line items
        plus any tax on top, because the total is what the page asks for and what the payment
        document beside it states — `document_code` is the printed number a payment's purpose
        cites, `instalment_amount` is populated exactly when the page prints the term. `tax` is
        the row between the subtotal and the total, labelled apart so `amount ≠ Σ line items` is
        measurable where it is true (see the field in `schemas.DocGroundTruth`). What differs
        from the рахунок is `currency` and `language`, which is the whole point of the archetype.
        """
        rules = jurisdiction("EU")
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.INVOICE,
            language=rules["language"],
            currency=rules["currency"],
            amount=self.total,
            tax=self.tax_amount,
            instalment_amount=self.instalment_amount,
            date=self.issued_at.date(),
            counterparty=self.seller_name,
            payer=self.buyer_name,
            document_code=self.number,
            line_items=self.line_items,
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def build_eu_invoice(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    buyer_name: str,
    buyer_tax_id: str,
    buyer_country: Country = Country.UA,
    address: str = "",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
    schedule: str | None = None,
    settled_at: datetime | None = None,
) -> EuInvoice:
    """Build one invoice from a platform of the `EU` pool — English, euros, one class with the
    Ukrainian рахунок.

    THE BASKET IS THE PLATFORM RECEIPT'S — `_draw_basket` with `language="en"` and
    `currency="EUR"`, which selects the English template lists policy.yaml carries per item kind
    and the euro price ranges of config/generation.yaml. Same knobs, same meaning, same function
    as every other basket-carrying class.

    WHAT IT ACCEPTS AND DELIBERATELY DOES NOT PRINT, because the assembler hands these to every
    subject document and a page must not grow a requisite to use one up:

    * `buyer_tax_id` — 📄 the customer's VAT identification number is an Article 226 particular of
      a VAT invoice, and this is not one; the customer is in any case a private individual.
      ⛔ The reverse-charge form does NOT change this: the drawn note states the mechanism and
      the page still prints no number for it, because the buyer whose number it would be is a
      private person whose РНОКПП belongs on no cross-border invoice.
    * `address` — the assembler's address slot holds the CLAIMANT's city, which is the seller's
      own on a domestic claim and nobody's here. ⛔ Inventing a seat for a named real platform is
      a checkable claim about a real firm, which this repository does not make.

    `buyer_country` IS THE AXIS THE TAX TREATMENT DERIVES FROM — the persona's own country,
    printed under the buyer's name AND deciding which rate parameter a tax-on-top page charges
    (see `TaxTreatment`). Defaulted to UA because every persona of today's corpus is a Ukrainian
    resident; the assembler passes the persona's country either way, so a relocated persona
    changes this page without this builder being touched.

    `identity` IS PRINTED, and it is the one identity field this class carries: the seller's IBAN
    and the bank holding it, drawn once for the claim so the payment document settling this
    invoice names the same account. Its `tax_code` is a Ukrainian register's and is not printed —
    see the block comment in config/fiscal-rules.yaml.

    🔴 `settled_at` BOUNDS THE DUE DATE, exactly as it bounds the Ukrainian invoice's validity
    line and for the same reason: an offer whose term lapsed before the payment is not the
    obligation that payment discharged. The window is drawn first and unconditionally, so a seed's
    stream does not depend on whether the claim's payment outran it.
    """
    del buyer_tax_id, address  # accepted, never printed — see the docstring

    rules = jurisdiction("EU")
    block = rules["invoice"]

    items = _draw_basket(
        rng,
        document="an invoice",
        category_id=category_id,
        vendor=vendor,
        # ⛔ NO LINE CARRIES A VAT LETTER, whatever the drawn tax treatment: the letter is a
        # requisite of the Ukrainian fiscal receipt's line, and the tax this page may charge
        # stands in ONE row of the totals block (see `TaxTreatment`), never per line.
        vat_payer=False,
        covered_only=covered_only,
        coverage_target=coverage_target,
        item_count=item_count,
        language=rules["language"],
        currency=rules["currency"],
    )

    # The tax treatment, drawn AFTER the basket because its figure is a rate over the lines and
    # BEFORE the due date so the two draws keep their positions whatever either returns.
    treatment = draw_tax_treatment(
        rng, buyer_country=buyer_country, subtotal=line_items_total(items)
    )

    due_at = issued_at + timedelta(days=rng.randint(*invoice_count_range("due_days")))
    if settled_at is not None and settled_at.date() > due_at.date():
        due_at = settled_at

    return EuInvoice(
        seller_name=vendor["name"],
        seller_display=printed_legal_name(vendor["name"], vendor["legal_form"]),
        buyer_name=buyer_name,
        buyer_country=rules["country_names"][buyer_country.value],
        issued_at=issued_at,
        due_at=due_at,
        number=block["number"]["prefix"] + _draw_from_pattern(rng, block["number"]["pattern"]),
        line_items=items,
        iban=identity.account,
        bank_name=identity.bank_name,
        # The out-of-scope sentence is this class's own — an invoice argues its tax treatment,
        # and 📄 Articles 44 and 59 are the provisions the `vat_note` in fiscal-rules names.
        vat_note=treatment.note(out_of_scope=block["vat_note"]),
        tax_label=treatment.label,
        tax_amount=treatment.amount,
        schedule=schedule,
    )


# =============================================================================
# The banking application — two carriers, one bank, one phone
# =============================================================================

# 👁 «2 лютого 2026, 13:46» — how an application screen states a date, unlike every
# printed document of this corpus, which writes 02.02.2026. Ukrainian linguistic data
# lives in code exactly as `amount_in_words_uk`'s number words do.
_MONTHS_GENITIVE = (
    "січня", "лютого", "березня", "квітня", "травня", "червня",
    "липня", "серпня", "вересня", "жовтня", "листопада", "грудня",
)


def date_in_words_uk(moment: datetime) -> str:
    """The date as an app screen states it — day, month in the genitive, year, time."""
    return (
        f"{moment.day} {_MONTHS_GENITIVE[moment.month - 1]} {moment.year}, "
        f"{moment:%H:%M}"
    )


def descriptor_prefixes() -> tuple[str, ...]:
    """The processor prefixes a card-network descriptor can carry.

    Read from config/vendors.json rather than written here: `payment_providers` and
    `aggregators` are the public Ukrainian processors this repository already names, and
    each is used for exactly the trade it is publicly in. A prefix invented to look
    plausible would be a mark invented to look plausible, which config/generation.yaml
    forbids at the top of the file.
    """
    vendors = load_vendors()
    providers = [entry["display_name"] for entry in vendors["payment_providers"]["UA"]]
    aggregators = [entry["name"] for entry in vendors["aggregators"]["UA"]]
    return tuple(sorted(name.upper() for name in providers + aggregators))


def merchant_descriptor(rng: random.Random, merchant_name: str) -> str:
    """🔴 `LIQPAY*КОВАЛЬЧУК О.С.` — the counterparty as the card network carries it.

    NOT a legal name, and that is the whole phenomenon: the tail is the merchant's name
    after the descriptor field has had it — uppercased, punctuation squeezed — and the
    prefix is the processor's, so the one party line the screen prints matches no party
    block of any other document of the claim. The LABEL still carries the bare trade name
    in `counterparty`, exactly as on every class: what breaks is the PRINTED cross-check,
    which is the thing this archetype exists to put in the data. The normalization rule
    for descriptor prefixes is deliberately a downstream question, not this generator's.
    """
    tail = merchant_name.upper().replace(". ", ".")
    return f"{rng.choice(descriptor_prefixes())}*{tail}"


@dataclass(frozen=True)
class AppTransaction:
    """One operation as a banking application shows it — the transaction screen.

    🔴 THE STRONGEST NEGATIVE EXAMPLE FOR THE PAYMENT CLASS BY LACK OF REQUISITES. The
    screen looks like proof of payment and carries none of the requisites by which proof
    of payment is recognized: ⛔ no document number, no authorization code, no RRN, no
    stamp, no signature, no amount in words, no payment purpose. What it does carry —
    👁 from public examples — is a status bar, a back arrow, a processor descriptor for
    the counterparty, a category chip, a date in words with a time, a signed amount in
    the largest type, a balance after the operation, the paying card, and a tap that
    opens the framed receipt the sibling archetype renders.

    ⚠️ THE CATEGORY CHIP IS THE BANK'S, NOT THE POLICY'S, and it is drawn without
    reference to the claim — it may plainly disagree with what was bought, which is a
    property of the domain rather than a defect.
    """

    bank_name: str
    card_tail: str
    merchant_descriptor_text: str
    category_chip: str
    issued_at: datetime
    amount: Decimal
    balance_after: Decimal
    # The bare trade name the LABEL carries as `counterparty`, while the page prints the
    # descriptor above — the split is the archetype's point.
    counterparty: str
    battery_fill_px: int

    # -- rendering ------------------------------------------------------------

    def _amount(self, value: Decimal) -> str:
        rules = jurisdiction("UA")["number_format"]
        whole, _, fraction = f"{value:.2f}".partition(".")
        grouped = f"{int(whole):,}".replace(",", rules["thousands_separator"])
        # 👁 An application prints the sign; the dot is the app convention observed, so
        # there is nothing to draw.
        return f"{grouped}.{fraction}"

    def render_context(self) -> dict:
        rules = jurisdiction("UA")
        block = rules["bank_app"]
        strings = block["transaction"]
        return {
            "status_time": f"{self.issued_at:%H:%M}",
            "battery_fill_px": self.battery_fill_px,
            "merchant_descriptor": self.merchant_descriptor_text,
            "category": self.category_chip,
            "datetime_words": date_in_words_uk(self.issued_at),
            # 👁 Signed, with the true minus sign rather than a hyphen — money leaving
            # the account. The direction is the one fact this screen states that the A4
            # confirmation never prints.
            "amount_signed": f"−{self._amount(self.amount)} ₴",
            "description_label": strings["description_label"],
            "description_placeholder": strings["description_placeholder"],
            "balance_label": strings["balance_label"],
            "balance": f"{self._amount(self.balance_after)} ₴",
            "payment_method_label": strings["payment_method_label"],
            "payment_method": f"{self.bank_name} ••{self.card_tail}",
            "actions": list(strings["actions"]),
            "tabs": list(block["tabs"]),
            "qr_payload": None,
        }

    # -- labels ---------------------------------------------------------------

    def ground_truth(
        self,
        *,
        doc_id: str,
        source_file: str,
        capture: Capture,
        field_bboxes: dict[str, tuple[float, float, float, float]],
        reference_text: str = "",
        content_bbox: tuple[float, float, float, float] | None = None,
        content_lost_edges: tuple[str, ...] = (),
    ) -> DocGroundTruth:
        """The label record for this screen.

        `payment_purpose`, `auth_code` and `fee` are `None`, `payer` is `None` — 👁 the
        screen is the claimant's own application, so it names nobody — and `direction` is
        DEBIT, which the printed minus states. `counterparty` carries the bare trade
        name, as on every class; the descriptor is what the PAGE prints, and the gap
        between the two is measured on pixels, not smuggled into the label.
        """
        return DocGroundTruth(
            doc_id=doc_id,
            source_file=source_file,
            doc_type=DocType.PAYMENT_CONFIRMATION,
            language="uk",
            currency="UAH",
            amount=self.amount,
            date=self.issued_at.date(),
            counterparty=self.counterparty,
            direction=Direction.DEBIT,
            line_items=[],
            has_qr=False,
            qr_is_fiscal=False,
            has_fiscal_number=False,
            capture=capture,
            field_bboxes=field_bboxes,
            reference_text=reference_text,
            content_bbox=content_bbox,
            content_lost_edges=list(content_lost_edges),
        )


def build_app_transaction(
    rng: random.Random,
    *,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    payer_name: str,
    payer_tax_id: str,
    amount: Decimal | None = None,
    cites: DocumentReference | None = None,
    must_cite: bool = False,
) -> AppTransaction:
    """Build one app transaction screen.

    ``must_cite`` is REFUSED where true: the screen has no purpose line, so a plan forcing a
    citation onto it has aimed the `subject` axis at a class that cannot carry the negative —
    `claim_planner` keeps such plans off this archetype, and reaching the refusal means the two
    have come apart.

    THE TRANSFER KEYWORDS OF EVERY PAYMENT BUILDER, four of them unprinted here:
    `identity`, `payer_name` and `payer_tax_id` are accepted so that one mechanism
    serves every payment archetype — ⛔ nothing on this screen names a party by
    requisites: the seller appears only as the descriptor, and the payer is holding the
    phone. `cites` is accepted and NOT printed, and that is the honest outcome rather
    than an error: the screen has no purpose line, so a claim settled by it carries its
    citation nowhere — the same case as the internet-acquiring confirmation that prints
    no purpose, which the contract's KL-16 already tells a consumer to stratify by. The
    label's `payment_purpose` is `None`, which is what says so.

    `amount` is the claim's wherever the claim has a subject document — a payment that
    drew its own would disagree with the subject beside it on every claim — and is drawn
    here only for the claim that deliberately has none: the evidence gap, exactly as on
    the A4 confirmation, from the same range.
    """
    if must_cite:
        raise ValueError(
            "an app transaction screen prints no purpose line, so it cannot be forced to cite "
            "its subject — see `claim_planner._CITES_THE_SETTLED_DOCUMENT`, which is what keeps "
            "a subject-mismatch plan off this archetype"
        )
    del identity, payer_name, payer_tax_id, cites  # accepted, never printed — see above
    if amount is None:
        # An EVIDENCE-GAP claim carries this payment document ALONE — there is no subject
        # whose amount the assembler could hand over — so the transfer is drawn, exactly
        # as the A4 confirmation draws its own in the same case: the same declared range,
        # the same ten-kopiyka steps. 🔴 Found by the cross-seed verification sweep, not
        # by the suite: a refusal here crashed the one seed whose gap claim drew this
        # rendering, a stage away from the plan that legitimately asked for it.
        low, high = payment_confirmation_money_range("transfer_amount")
        amount = Decimal(rng.randrange(_minor(low), _minor(high), 10)) / 100
    if amount <= 0:
        raise ValueError(f"an app transaction states a positive amount, got {amount}")

    strings = jurisdiction("UA")["bank_app"]["transaction"]
    # The NAME is drawn and the pool is the shared one; the МФО table stays honest
    # because no code is printed beside the name here.
    bank_name, _ = _draw_bank(rng, "UA")
    return AppTransaction(
        bank_name=bank_name,
        card_tail=f"{rng.randint(1000, 9999)}",
        merchant_descriptor_text=merchant_descriptor(rng, vendor["name"]),
        category_chip=rng.choice(list(strings["categories"])),
        issued_at=issued_at,
        amount=amount,
        # 👁 A balance strictly above the amount: the operation went through.
        balance_after=amount + Decimal(rng.randrange(50_000, 900_000)) / 100,
        counterparty=vendor["name"],
        battery_fill_px=46 - rng.randrange(0, 20),
    )


@dataclass(frozen=True)
class BankReceiptInApp:
    """The A4 payment confirmation, captured inside the banking application.

    🔴 THE SAME FORMAL DOCUMENT, NOT A SIBLING OF IT: the inner document is a
    `PaymentConfirmation` built by the same builder, rendered through the same shared
    body (`templates/ua_bank_payment_confirmation.jinja`), and labelled by ITS OWN
    `ground_truth` — this class delegates rather than copies, so the two carriers cannot
    drift. One document, two carriers, and the labels must agree: that is the archetype's
    whole argument, and delegation is what makes it a property of the construction
    rather than a promise.

    What the frame adds is chrome an extractor has to see past — a dark surround, a
    status bar, a back arrow, a share button, a tab bar, and a screen heading that
    repeats the document's own number in DIFFERENT WORDS than the document's own
    heading. 👁 The two headings disagreeing over one number is observed, not staged.
    """

    inner: PaymentConfirmation
    battery_fill_px: int

    def render_context(self) -> dict:
        rules = jurisdiction("UA")
        block = rules["bank_app"]
        return {
            **self.inner.render_context(),
            "status_time": f"{self.inner.issued_at:%H:%M}",
            "battery_fill_px": self.battery_fill_px,
            "screen_title": (
                f"{block['receipt_screen_title']} № {self.inner.document_code}"
            ),
            "share_label": block["share_label"],
            "tabs": list(block["tabs"]),
        }

    def ground_truth(self, **kwargs) -> DocGroundTruth:
        """The inner document's label, verbatim — the medium does not change the ground
        truth. The carrier contributes only what the render measured: the boxes (which
        include the chrome's own `screen_title`) and the reference text (which includes
        the chrome's strings), both of which describe the IMAGE and are exactly what the
        parameters of this call carry."""
        return self.inner.ground_truth(**kwargs)


def build_bank_receipt_in_app(
    rng: random.Random,
    *,
    issued_at: datetime,
    vendor: dict,
    identity: PartyIdentity,
    payer_name: str,
    payer_tax_id: str,
    amount: Decimal | None = None,
    cites: DocumentReference | None = None,
    must_cite: bool = False,
) -> BankReceiptInApp:
    """Build the confirmation and the frame around it.

    The inner document is built FIRST and with the same keywords the A4 archetype gets —
    `must_cite` passes through untouched, so a subject-mismatch claim carried by this archetype
    is document-for-document what it would have been on paper; the chrome draws come after, and
    the order is part of the seed's meaning.
    """
    inner = build_payment_confirmation(
        rng,
        issued_at=issued_at,
        vendor=vendor,
        identity=identity,
        payer_name=payer_name,
        payer_tax_id=payer_tax_id,
        amount=amount,
        cites=cites,
        must_cite=must_cite,
    )
    return BankReceiptInApp(inner=inner, battery_fill_px=46 - rng.randrange(0, 20))


# =============================================================================
# Invariant validators
# =============================================================================


def line_items_total(items: list[LineItem]) -> Decimal:
    """Σ(qty × price) over the line items, to the kopiyka."""
    return sum((item.qty * item.price for item in items), Decimal(0)).quantize(KOPIYKA)


def validate_line_item_sum(items: list[LineItem], total: Decimal) -> bool:
    """Whether the line items account for exactly the stated total.

    Exact, not approximate. Amounts are Decimal so that no tolerance is needed, and a
    tolerance would hide the drift this exists to catch. A receipt with no lines fails:
    it states an amount nothing accounts for.
    """
    if not items:
        return False
    return line_items_total(items) == Decimal(total).quantize(KOPIYKA)


def validate_amount_in_words(text: str, amount: Decimal) -> bool:
    """Whether the text states this amount.

    Answers the question asked rather than raising: a trap archetype may print anything
    at all where the amount in words belongs, and "unreadable" answers "no".
    """
    try:
        return words_to_amount_uk(text) == Decimal(amount).quantize(KOPIYKA)
    except (ValueError, ArithmeticError):
        return False


def proves_payment_by_direction(direction: Direction | None) -> bool:
    """Whether a transaction in this direction can be proof that an expense was paid.

    🔴 ONLY A DEBIT CAN. A credit is money arriving — a refund, a reversal, a transfer from
    another account of the holder's — and it evidences no expense whatever its amount: for a
    reimbursement claim the difference between a debit and a credit is the difference between
    proof of an expense and proof that the money came back.

    `None` PASSES, and that is a statement rather than a lenience: a document class that prints no
    direction is a document about ONE movement of money, made because a payment was made, so
    there is nothing for the rule to be about. A receipt and a confirmation are that; a statement,
    which lists movements both ways, is what the rule exists for.

    Stated as a question with a boolean answer, like every validator here. Two things enforce it,
    and only one of them goes through this function: `policy_engine.resolve_evidence` CALLS it, and
    refuses a claim that rests on a credit rather than assigning it a verdict policy.yaml does not
    support, while `BankStatement.__post_init__` refuses independently — it tests the row's own
    direction, so the builder's guard survives a mistake in here and vice versa. A trap
    archetype that prints a refund as though it were a payment is where the invariant is broken
    deliberately, and it will have to name the break in the claim's `imperfection`.
    """
    return direction is not Direction.CREDIT


def validate_vat_letter(item_kind: str, letter: str | None, country: str) -> bool:
    """Whether this item kind may carry this VAT letter in this jurisdiction."""
    if letter is None or letter not in jurisdiction(country)["vat_letters"]:
        return False
    return letter in allowed_vat_letters(item_kind, country)


def printed_legal_name(name: str, legal_form: str) -> str:
    """A party's name as printed, with its legal form.

    A sole trader is printed without quotes — ``ФОП Ковальчук О. С.`` — because the name is
    a person's, not a firm's. Every other Ukrainian form takes the Ukrainian quotation marks.

    A FOREIGN FIRM TAKES NEITHER, and carries its own designation after the name instead:
    ``Coursera Inc.`` The quotation marks are a rule about a Ukrainian firm's name, and the first
    document to print a cross-border seller's legal name is what made the difference visible —
    every earlier one either named a domestic firm or, like the platform receipt, printed the
    bare trade name.

    Takes the two strings rather than a ``Seller`` because the same rule prints the RECIPIENT of
    a bank payment confirmation, which is the same firm named on a different document class and
    is not a seller of anything on that page.
    """
    if legal_form in _LEGAL_FORM_SUFFIX:
        suffix = _LEGAL_FORM_SUFFIX[legal_form]
        return f"{name} {suffix}" if suffix else name
    prefix = _LEGAL_FORM_PREFIX.get(legal_form, legal_form)
    if legal_form == _SOLE_TRADER:
        return f"{prefix} {name}"
    return f"{prefix} «{name}»"


def legal_name(seller: Seller) -> str:
    """The seller's name as printed on a receipt. See `printed_legal_name` for the rule."""
    return printed_legal_name(seller.name, seller.legal_form)
