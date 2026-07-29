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

from receipt_synth.config import category, jurisdiction
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

    ``tax_code`` is not named after either identifier because it is whichever one
    applies: a ТОВ prints its ЄДРПОУ labelled "ІД", a ФОП prints its РНОКПП labelled
    "ІПН".
    """

    name: str
    legal_form: str
    address: str
    tax_code: str
    tax_code_label: str
    vat_number: str  # ПН — the taxpayer number of a registered VAT payer


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
    amount_in_words: str
    tax_lines: list[TaxLine]
    payment_method: str
    acquiring: Acquiring | None
    decimal_separator: str
    qr_payload: str
    footer: str

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
            "tax_lines": [
                rules["tax_line_format"].format(
                    name=rules["vat_letters"][line.letter]["name"],
                    letter=line.letter,
                    rate=line.rate,
                    amount=self._amount(line.vat),
                )
                for line in self.tax_lines
            ],
            "total": self._amount(self.total),
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

# Values for the placeholders in the line-item name templates of policy.yaml. Plausible
# over-the-counter strengths and pack sizes — general product knowledge, not taken from
# any particular vendor's catalogue.
#
# `{brand}` is deliberately absent: brand vocabulary belongs in vendors.json, which is
# still a stub. Templates carrying a placeholder that cannot be filled are skipped
# rather than rendered with a literal brace.
_PLACEHOLDER_VALUES: dict[str, tuple[str, ...]] = {
    "dose": ("400", "500", "1000", "2000", "5000"),
    "n": ("20", "30", "60", "90", "120"),
    "w": ("250", "500", "900"),
}

# Retail price ranges in kopiykas, per item kind. Plausible Ukrainian pharmacy prices;
# like the placeholder values, these are a property of the merchandise rather than of
# any one seller, and they move to vendors.json when it carries real catalogues.
_PRICE_RANGE_KOPIYKAS: dict[str, tuple[int, int]] = {
    "vitamin_complex": (18_000, 95_000),
    "mineral_supplement": (9_000, 42_000),
    "nutritionist_visit": (60_000, 150_000),
    "default": (10_000, 50_000),
}

_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

_LEGAL_FORM_PREFIX = {"TOV": "ТОВ", "FOP": "ФОП", "PRAT": "ПрАТ"}

# How many of an article a receipt lists. An assumption, not a measurement: nothing here
# was fitted to observed baskets, and the weighting toward one is ordinary shopping
# knowledge — a shopper buying two identical packs is less common than buying one. It
# affects how a document looks, never what a label says. Kept as a constant because
# `estimated_line_value` sizes a basket from its mean, and the two must not drift apart.
_QTY_CHOICES = (1, 1, 1, 2)

# The longest receipt this builder will print. A cap rather than a preference: the planner
# sizes a basket upward when it needs one large enough to overrun an annual limit, and
# without a bound a large enough remaining balance would ask for a receipt no shop issues.
MAX_LINE_ITEMS = 20


def _placeholders(template: str) -> list[str]:
    # Sorted, because iterating a set would order the draws by a hash that varies
    # between interpreter runs — and determinism under --seed would quietly stop holding.
    return sorted({name for _, name, _, _ in Formatter().parse(template) if name})


def _is_renderable(template: str) -> bool:
    return set(_placeholders(template)) <= _PLACEHOLDER_VALUES.keys()


def _fill_placeholders(template: str, rng: random.Random) -> str | None:
    """A line-item name with its placeholders resolved, or ``None`` if some placeholder
    has no vocabulary yet."""
    if not _is_renderable(template):
        return None
    names = _placeholders(template)
    return template.format(**{name: rng.choice(_PLACEHOLDER_VALUES[name]) for name in names})


def renderable_kinds(catalogue: dict, language: str = "uk") -> list[str]:
    """The item kinds of a bucket that can currently be printed at all.

    A kind whose every name template carries a placeholder with no vocabulary yet — the
    `{brand}` and `{drug}` families, which wait on `config/vendors.json` — is not a bug and
    must not raise: it simply cannot appear on a document until the vocabulary lands. This
    is where that is decided, once, instead of at each draw.
    """
    return sorted(
        kind
        for kind, names in catalogue.items()
        if any(_is_renderable(template) for template in names[language])
    )


def _build_line_item(
    item_kind: str, templates: list[str], rng: random.Random, *, covered: bool
) -> LineItem:
    usable = [name for name in (_fill_placeholders(t, rng) for t in templates) if name]
    if not usable:
        raise ValueError(f"no renderable name template for item kind {item_kind!r}")

    low, high = _PRICE_RANGE_KOPIYKAS.get(item_kind, _PRICE_RANGE_KOPIYKAS["default"])
    return LineItem(
        name=rng.choice(usable),
        item_kind=item_kind,
        qty=Decimal(rng.choice(_QTY_CHOICES)),
        # Drawn in whole ten-kopiyka steps: retail prices do not end in arbitrary
        # kopiykas, and an exact integer keeps the sum exact.
        price=Decimal(rng.randrange(low, high, 10)) / 100,
        covered=covered,
        vat_letter=vat_letter_for_kind(item_kind, "UA", rng),
    )


# How many draws to allow before accepting a shorter receipt. Reached only when a
# category has fewer distinct renderable names than the requested line count.
_DISTINCT_DRAW_LIMIT = 40


def _draw_distinct_items(
    rng: random.Random, kinds: list[str], catalogue: dict, count: int, *, covered: bool
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
        item = _build_line_item(kind, catalogue[kind]["uk"], rng, covered=covered)
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
    """
    kinds = renderable_kinds(category(category_id)["covered_items"])
    if not kinds:
        raise ValueError(f"category {category_id!r} has no printable covered item")

    ranges = [_PRICE_RANGE_KOPIYKAS.get(kind, _PRICE_RANGE_KOPIYKAS["default"]) for kind in kinds]
    mean_price = Decimal(sum((low + high) for low, high in ranges)) / (2 * len(ranges) * 100)
    mean_qty = Decimal(sum(_QTY_CHOICES)) / len(_QTY_CHOICES)
    return (mean_price * mean_qty).quantize(KOPIYKA)


# Non-covered lines per mixed basket. Usually one — the carrier bag, the tube of cream —
# because that is what the imperfection catalogue describes: a non-qualifying item inside
# an otherwise qualifying purchase.
_EXCLUDED_LINE_COUNTS = (1, 1, 2)


def _repriced(item: LineItem, line_total: Decimal) -> LineItem:
    """The same line, priced so it comes to about ``line_total``.

    Clamped into the item kind's own range and rounded to ten kopiykas, so that hitting a
    coverage target cannot print a 4 UAH blood-pressure monitor. The clamp is why the
    realized coverage only approaches the target — which is enough, because the target
    only has to land the claim on the right side of `full_threshold`.
    """
    low, high = _PRICE_RANGE_KOPIYKAS.get(item.item_kind, _PRICE_RANGE_KOPIYKAS["default"])
    kopiykas = int((line_total / item.qty * 100).to_integral_value(rounding=ROUND_HALF_UP))
    kopiykas = min(max(kopiykas - kopiykas % 10, low), high)
    return item.model_copy(update={"price": Decimal(kopiykas) / 100})


def _excluded_ceiling(kinds: list[str]) -> Decimal:
    """The most one non-covered line may cost, over the kinds available.

    The sizing bound for the loop below: at qty 1 no non-covered line can be repriced
    above this without leaving the range its item kind is plausible in.
    """
    return Decimal(
        max(_PRICE_RANGE_KOPIYKAS.get(kind, _PRICE_RANGE_KOPIYKAS["default"])[1] for kind in kinds)
    ) / 100


def _build_mixed_basket(
    rng: random.Random, *, category_id: str, count: int, coverage_target: Decimal
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

    covered = _draw_distinct_items(
        rng, renderable_kinds(covered_catalogue), covered_catalogue, count, covered=True
    )
    if not covered:
        raise ValueError(f"category {category_id!r} produced no covered line")

    excluded_kinds = renderable_kinds(excluded_catalogue)
    if not excluded_kinds:
        raise ValueError(
            f"category {category_id!r} has no non-covered item this generator can print "
            "yet — every excluded name template still needs a placeholder vocabulary"
        )

    # covered / (covered + excluded) = target  =>  excluded = covered × (1 − target) / target
    def budget(items: list[LineItem]) -> Decimal:
        return line_items_total(items) * (1 - coverage_target) / coverage_target

    ceiling = _excluded_ceiling(excluded_kinds)
    wanted = max(rng.choice(_EXCLUDED_LINE_COUNTS), math.ceil(budget(covered) / ceiling))
    excluded = _draw_distinct_items(
        rng, excluded_kinds, excluded_catalogue,
        min(wanted, MAX_LINE_ITEMS - len(covered)), covered=False,
    )
    if not excluded:
        raise ValueError(f"category {category_id!r} produced no non-covered line")

    # A low coverage target asks for more non-covered money than the category's excluded
    # bucket can plausibly carry — today most of those name templates cannot be printed at
    # all, because `config/vendors.json` is still a stub and `{brand}` and `{drug}` have no
    # vocabulary. Shrinking the covered side is the honest way to reach the ratio; the
    # alternative, one absurdly priced non-covered line, would be a visible artifact in
    # the image. This loop stops firing as the vocabulary lands.
    while len(covered) > 1 and budget(covered) > ceiling * len(excluded):
        covered.pop()

    excluded = [_repriced(item, budget(covered) / len(excluded)) for item in excluded]

    items = covered + excluded
    # Otherwise every non-covered line is the last one on every mixed receipt, which is a
    # position a consumer could learn instead of learning to read the line.
    rng.shuffle(items)
    return items


def _build_tax_lines(items: list[LineItem]) -> list[TaxLine]:
    """One row per VAT letter present, in the order the jurisdiction declares them.

    Ukrainian receipts print VAT-inclusive prices, so the tax is extracted from the
    gross rather than added on top: vat = gross − gross / (1 + rate/100).
    """
    gross_by_letter: defaultdict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for item in items:
        if item.vat_letter is None:
            # Only jurisdictions with `line_item_letter_position: none` (ES, EU) print
            # lines without a letter. On a UA receipt this is a builder bug, and a
            # silently untaxed line would corrupt the tax block rather than fail.
            raise ValueError(f"line item {item.name!r} carries no ПДВ-літера")
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
    """
    rules = jurisdiction("UA")
    receipt_rules = rules["receipt"]

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
        items = _draw_distinct_items(
            rng, renderable_kinds(catalogue), catalogue, count, covered=True
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
            rng, category_id=category_id, count=count, coverage_target=coverage_target
        )
    total = line_items_total(items)

    # -- who sold it
    seller = Seller(
        name=vendor["name"],
        legal_form=vendor["legal_form"],
        address=address,
        tax_code=generate_edrpou(rng),
        tax_code_label=rules["identifiers"]["edrpou"]["label"],
        vat_number=f"{rng.randint(0, 10**12 - 1):012d}",
    )

    # -- how it was paid for
    acquiring_rules = rules["acquiring_block"]
    acquiring_examples = {f["key"]: f for f in acquiring_rules["fields"]}
    acquiring = Acquiring(
        acquirer=acquiring_examples["acquirer"]["example"],
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
        amount_in_words=amount_in_words_uk(total),
        tax_lines=_build_tax_lines(items),
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
    """The seller's name as printed, with its legal form."""
    prefix = _LEGAL_FORM_PREFIX.get(seller.legal_form, seller.legal_form)
    return f"{prefix} «{seller.name}»"
