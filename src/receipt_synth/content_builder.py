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
    category,
    excluded_line_counts,
    high_frequency_surnames,
    jurisdiction,
    placeholder_values,
    price_range,
    quantity_choices,
    unprintable_item_kinds,
    vendor_profile,
)
from receipt_synth.schemas import Capture, DocGroundTruth, DocType, LineItem

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


def rnokpp_check_digit(body: str) -> int:
    """The tenth digit of a РНОКПП, given its first nine."""
    if not _is_digits(body, 9):
        raise ValueError(f"a РНОКПП body is nine digits, got {body!r}")
    return sum(w * int(d) for w, d in zip(_RNOKPP_WEIGHTS, body, strict=True)) % 11 % 10


def is_valid_rnokpp(value: str) -> bool:
    """Whether a string is a well-formed, checksum-correct РНОКПП."""
    return _is_digits(value, 10) and rnokpp_check_digit(value[:9]) == int(value[9])


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


def amount_in_words_uk(amount: Decimal) -> str:
    """An amount as printed in words on a Ukrainian document.

    Hryvnias in words, kopiykas in digits, both units declined to agree with their own
    count — the convention Ukrainian receipts and invoices follow:

        Decimal("2500.00") -> "дві тисячі п'ятсот гривень 00 копійок"
    """
    if amount < 0:
        raise ValueError(f"a printed amount is not negative: {amount}")
    if amount != amount.quantize(KOPIYKA):
        raise ValueError(f"an amount has at most two decimal places: {amount}")

    units, kopiykas = divmod(int(amount.scaleb(2)), 100)
    if units >= _MAX_UNITS:
        raise ValueError(f"amount too large to spell out: {amount}")

    words = _int_to_words_uk(units, feminine=True)
    words.append(_plural_uk(units, *_HRYVNIA_FORMS))
    return f"{' '.join(words)} {kopiykas:02d} {_plural_uk(kopiykas, *_KOPIYKA_FORMS)}"


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

_AMOUNT_IN_WORDS_RE = re.compile(
    rf"^(?P<units>.+?)\s+(?P<hryvnia>{'|'.join(_HRYVNIA_FORMS)})"
    rf"\s+(?P<kopiykas>\d{{2}})\s+(?P<kopiyka>{'|'.join(_KOPIYKA_FORMS)})$"
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
    """The amount a Ukrainian amount-in-words states. Raises ``ValueError`` if the text
    is not one."""
    match = _AMOUNT_IN_WORDS_RE.match(text.strip())
    if match is None:
        raise ValueError(f"not an amount in words: {text!r}")
    units = _words_to_int_uk(match["units"])
    return (Decimal(units) + Decimal(match["kopiykas"]) / 100).quantize(KOPIYKA)


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
    """One Ukrainian ПРРО fiscal receipt, complete but not yet rendered."""

    seller: Seller
    issued_at: datetime
    title: str
    mode_marker: str  # Онлайн / Офлайн
    receipt_number: str
    fiscal_device_number: str  # ФН ПРРО
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
                    "label": rules["tax_line_label_format"].format(
                        name=rules["vat_letters"][line.letter]["name"],
                        letter=line.letter,
                        rate=line.rate,
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
    item_kind: str, templates: list[str], rng: random.Random, *, covered: bool, vat_payer: bool
) -> LineItem:
    names = [_fill_placeholders(template, item_kind, rng) for template in templates]

    low, high = price_range(item_kind)
    return LineItem(
        name=rng.choice(names),
        item_kind=item_kind,
        qty=Decimal(rng.choice(quantity_choices())),
        # Drawn in whole ten-kopiyka steps: retail prices do not end in arbitrary
        # kopiykas, and an exact integer keeps the sum exact.
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
) -> list[LineItem]:
    """Line items with distinct printed names.

    A receipt lists a product once and says how many; the same article appearing twice on
    one receipt at two different prices is not something a cash register produces. Kinds
    are still drawn with replacement — a pharmacy basket really can hold two different
    vitamins — it is the printed name that has to be unique.
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
            kind, catalogue[kind]["uk"], rng, covered=covered, vat_payer=vat_payer
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


def _repriced(item: LineItem, line_total: Decimal) -> LineItem:
    """The same line, priced so it comes to about ``line_total``.

    Clamped into the item kind's own range and rounded to ten kopiykas, so that hitting a
    coverage target cannot print a 4 UAH blood-pressure monitor. The clamp is why the
    realized coverage only approaches the target — which is enough, because the target
    only has to land the claim on the right side of `full_threshold`.
    """
    low, high = price_range(item.item_kind)
    kopiykas = int((line_total / item.qty * 100).to_integral_value(rounding=ROUND_HALF_UP))
    kopiykas = min(max(kopiykas - kopiykas % 10, _minor(low)), _minor(high))
    return item.model_copy(update={"price": Decimal(kopiykas) / 100})


def _excluded_ceiling(kinds: list[str]) -> Decimal:
    """The most one non-covered line may cost, over the kinds available.

    The sizing bound for the loop below: at qty 1 no non-covered line can be repriced
    above this without leaving the range its item kind is plausible in.
    """
    return max(price_range(kind)[1] for kind in kinds)


def _build_mixed_basket(
    rng: random.Random, *, category_id: str, vendor: dict, count: int, coverage_target: Decimal
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

    ceiling = _excluded_ceiling(excluded_kinds)
    wanted = max(rng.choice(excluded_line_counts()), math.ceil(budget(covered) / ceiling))
    excluded = _draw_distinct_items(
        rng, excluded_kinds, excluded_catalogue,
        min(wanted, MAX_LINE_ITEMS - len(covered)), covered=False, vat_payer=vat_payer,
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

    excluded = [_repriced(item, budget(covered) / len(excluded)) for item in excluded]

    items = covered + excluded
    # Otherwise every non-covered line is the last one on every mixed receipt, which is a
    # position a consumer could learn instead of learning to read the line.
    rng.shuffle(items)
    return items


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


def build_prro_receipt(
    rng: random.Random,
    *,
    category_id: str,
    issued_at: datetime,
    vendor: dict,
    address: str = "м. Київ",
    covered_only: bool = True,
    coverage_target: Decimal | None = None,
    item_count: int | None = None,
) -> PrroReceipt:
    """Build one Ukrainian ПРРО fiscal receipt.

    ``covered_only`` is the label-first knob: the planner has already chosen the verdict,
    and the builder realizes it. For ``covered`` the basket is drawn from the category's
    covered items alone; for ``partially_covered`` by ``mixed_items`` the caller clears
    the flag and states the ``coverage_target`` the basket should come to.

    ``vendor`` is an entry of `config/vendors.json`: its ``profile`` decides which item kinds may
    appear on the receipt, its ``vat_payer`` decides whether the seller block prints a ПН line
    beside the ІД line it always prints and whether the document has a VAT block, and its
    ``legal_form`` decides how the name is printed and which register both identifiers come from.
    Ask `vendor_can_carry` before choosing one for a mixed basket.
    """
    rules = jurisdiction("UA")
    receipt_rules = rules["receipt"]
    vat_payer = vendor_is_vat_payer(vendor)

    # -- what was bought
    count = item_count if item_count is not None else rng.randint(2, 4)
    if not 1 <= count <= MAX_LINE_ITEMS:
        raise ValueError(f"a receipt carries 1 to {MAX_LINE_ITEMS} lines, not {count}")

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
        items = _draw_distinct_items(
            rng, kinds, catalogue, count, covered=True, vat_payer=vat_payer
        )
    else:
        if coverage_target is None:
            raise ValueError(
                "a mixed basket needs the coverage_target the planner chose — the builder "
                "realizes a verdict, it does not decide one"
            )
        if not Decimal(0) < coverage_target < Decimal(1):
            raise ValueError(
                f"a coverage target lies strictly between 0 and 1, got {coverage_target}"
            )
        items = _build_mixed_basket(
            rng,
            category_id=category_id,
            vendor=vendor,
            count=count,
            coverage_target=coverage_target,
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
    is_sole_trader = vendor["legal_form"] == _SOLE_TRADER
    id_code_rules = rules["identifiers"]["rnokpp" if is_sole_trader else "edrpou"]
    vat_number_rules = rules["identifiers"]["vat_number"]

    tax_code = generate_rnokpp(rng) if is_sole_trader else generate_edrpou(rng)
    seller = Seller(
        name=vendor["name"],
        legal_form=vendor["legal_form"],
        address=address,
        vat_payer=vat_payer,
        tax_code=tax_code,
        tax_code_label=id_code_rules["label"],
        vat_number=(
            _vat_number(rng, tax_code, is_sole_trader=is_sole_trader) if vat_payer else None
        ),
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
        operation=acquiring_examples["operation"]["values"][0],
        card_masked=f"{rng.randint(0, 9999):04d}XXXXXXXX{rng.randint(0, 9999):04d}",
        auth_code=f"{rng.randint(0, 999_999):06d}",
        rrn=f"{rng.randint(0, 10**12 - 1):012d}",
    )

    # -- the fiscal identity of the document
    suffix = rng.choice(receipt_rules["title_suffixes"])
    receipt_number = "".join(rng.choice(_ALNUM) for _ in range(11))
    fiscal_device_number = f"{rng.randint(0, 10**10 - 1):010d}"

    return PrroReceipt(
        seller=seller,
        issued_at=issued_at,
        title=f"{receipt_rules['title']} {suffix}".strip(),
        mode_marker=receipt_rules["mode_markers"][0],
        receipt_number=receipt_number,
        fiscal_device_number=fiscal_device_number,
        line_items=items,
        total=total,
        # Zero this version, and the reason is the policy's silence rather than the arithmetic
        # — see `PrroReceipt.amount_due`. The lines are printed all the same, because 👁 all
        # four appear on the observed receipt.
        discount=Decimal(0),
        rounding=Decimal(0),
        amount_in_words=amount_in_words_uk(total),
        tax_lines=_build_tax_lines(items, vat_payer=vat_payer),
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

    )


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


def validate_vat_letter(item_kind: str, letter: str | None, country: str) -> bool:
    """Whether this item kind may carry this VAT letter in this jurisdiction."""
    if letter is None or letter not in jurisdiction(country)["vat_letters"]:
        return False
    return letter in allowed_vat_letters(item_kind, country)


def legal_name(seller: Seller) -> str:
    """The seller's name as printed, with its legal form.

    A sole trader is printed without quotes — ``ФОП Ковальчук О. С.`` — because the name is
    a person's, not a firm's. Every other form takes the Ukrainian quotation marks.
    """
    prefix = _LEGAL_FORM_PREFIX.get(seller.legal_form, seller.legal_form)
    if seller.legal_form == _SOLE_TRADER:
        return f"{prefix} {seller.name}"
    return f"{prefix} «{seller.name}»"
