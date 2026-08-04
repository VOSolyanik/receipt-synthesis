#!/usr/bin/env python3
"""Render the mock-up archetypes to PNGs, so they can be looked at.

    uv run python tools/render_mockups.py [--out DIR] [--seed N]

WHY THIS IS A SCRIPT AND NOT PART OF THE PIPELINE. Every template it renders is a MOCK-UP:
none is registered in `claim_planner.ARCHETYPES`, none has a builder in
`assembler._BUILDERS`, and none reaches a dataset. So there is no place in the pipeline for them,
and inventing one would be a connection nobody has decided to make. See templates/README.md.

⚠️ TWO MOCK-UPS HAVE LEFT THIS SCRIPT BY BEING CONNECTED, and the rule is the same both times: a
second, script-only way of building a shipped page drifts from the pipeline the moment either
changes. `ua_non_fiscal_receipt` is registered, built by `content_builder.build_non_fiscal_receipt`
and rendered by runs; its Ukrainian strings went to `receipt.non_fiscal` in
config/fiscal-rules.yaml, which is what the block below always said would happen to them.
`ua_claim_bundle` is now the composition template `assembler._compose_bundle` renders, at the
share `file_composition.bundle_share` declares in config/generation.yaml — so the file it shows is
one a run produces, and this script would only be photographing a second copy of it.

THE OUTPUT GOES OUTSIDE THE REPOSITORY by default, for the same reason `out/` is gitignored:
rendered images are not what this repository ships. `--out` overrides it, and a path inside
the repository is refused rather than quietly written.

NOTHING HERE PRODUCES A LABEL. No `data-field` attribute exists in any of the templates,
so `RenderedDocument.field_bboxes` comes back empty by construction, and this script writes no
JSON beside the images. That is the boundary the branch was given: labels and boxes arrive with
the connection, not with the layout.

DETERMINISM. One seed, one set of images. Every draw goes through the `random.Random` created
here, and the document built by a shipped builder (`build_payment_confirmation`) takes it as its
generator, exactly as the assembler does.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from receipt_synth.config import (
    jurisdiction,
    load_fx_rates,
    load_vendors,
)
from receipt_synth.content_builder import (
    build_payment_confirmation,
    draw_party_identity,
    generate_edrpou,
    printed_legal_name,
    resolve_vendor,
)
from receipt_synth.renderer import REPO_ROOT, Renderer

DEFAULT_OUT = Path(tempfile.gettempdir()) / "receipt-synth-mockups"

# The instant every mock-up is dated from. Fixed rather than "now": an image that changes with
# the clock is one no second run reproduces, which is the property this repository is built on.
ISSUED_AT = datetime(2026, 3, 17, 13, 52, 41)

# ---------------------------------------------------------------------------
# Ukrainian strings that have no home in config/ yet.
#
# ⛔ CONFIG IS NOT TOUCHED ON THIS BRANCH — not `fiscal-rules.yaml`, not `generation.yaml`,
# not `vendors.json`. Every string below is printed by a template that no run renders, so
# putting it in config would declare a vocabulary for documents the generator does not
# produce. When these archetypes are connected, THIS BLOCK IS WHAT MOVES INTO CONFIG: the
# titles and captions are jurisdiction wording (fiscal-rules.yaml), the category and action
# vocabularies are draw inputs (generation.yaml).
# ---------------------------------------------------------------------------

# ⚠️ THE NON-FISCAL SLIP'S STRINGS ARE GONE FROM HERE, and their absence is the block's own rule
# working. `ua_non_fiscal_receipt` is a SHIPPED archetype now — registered, built and rendered by
# runs — so its title and its two accounting-law captions are jurisdiction wording like any other
# and live in `receipt.non_fiscal` in config/fiscal-rules.yaml. This block holds the strings of
# templates NO RUN RENDERS; a string here for a shipped page would be a second source of truth.

# Ukrainian month names in the genitive, which is the case a date written out in words takes:
# «2 лютого 2024». Jurisdiction wording, and it belongs in config/fiscal-rules.yaml the moment
# any RENDERED archetype needs it — no shipped template writes a date this way today.
MONTHS_GENITIVE = (
    "січня", "лютого", "березня", "квітня", "травня", "червня",
    "липня", "серпня", "вересня", "жовтня", "листопада", "грудня",
)

# The hryvnia sign, which config/fiscal-rules.yaml does not carry: it states the currency CODE
# («UAH»), and no shipped template prints a symbol — a receipt writes bare figures and the A4
# confirmation writes the code. An application prints the sign, so the sign has to come from
# somewhere, and this is the honest somewhere until the archetype is connected.
CURRENCY_SIGN = "₴"

# The interface strings of a banking application. Not marks and not law — an app's own wording.
APP_STRINGS = {
    "description_label": "Опис і теги",
    "description_placeholder": "Додати",
    "balance_label": "Залишок після операції",
    "payment_method_label": "Спосіб оплати",
    "share_label": "Поділитися",
    "receipt_screen_title": "Квитанція",
}

# The first entry is the one that matters: it is the tap that opens `ua_bank_receipt_in_app`.
APP_ACTIONS = ("Переглянути квитанцію", "Створити шаблон", "Повідомити про проблему")
APP_TABS = ("Головна", "Платежі", "Історія", "Ще")

# A bank's own spending categories, which are NOT the benefit categories of config/policy.yaml
# and must not be mistaken for them — that mismatch is a property of the domain, not a defect.
APP_CATEGORIES = ("Комуналка та інтернет", "Спорт", "Здоровʼя", "Розваги", "Різне")

# The screen geometry, from templates/ua_phone_chrome.css. Stated here because the script has to
# compute the scale the document inside the card is drawn at, and a number guessed against a
# stylesheet is a number that goes stale the first time the padding changes.
SCREEN_WIDTH_PX = 1170
SHEET_PADDING_PX = 48
CARD_WIDTH_PX = SCREEN_WIDTH_PX - 2 * SHEET_PADDING_PX

# ---------------------------------------------------------------------------
# The platform-receipt class — a second currency and a second language.
# ---------------------------------------------------------------------------

# 🔴 A PUBLIC MARK, AND IT BELONGS IN config/vendors.json. It is here for the same reason every
# other string in this block is: the archetype is not connected, and declaring a vendor for a
# document no run produces would put data in the live file for a document that does not exist.
# It satisfies the rule at the head of config/generation.yaml — a publicly known mark, used only
# for the trade it is publicly in, which for this one is selling online courses to individuals.
# NOTHING IS PRINTED BESIDE IT THAT ASSERTS ANYTHING ABOUT THE COMPANY: no address, no tax
# number, no registration. Those are 📄 invoice particulars under Article 226 of Directive
# 2006/112/EC, and this document declares itself not to be an invoice, so their absence is
# evidenced rather than convenient. See templates/README.md.
FOREIGN_PLATFORM = "Coursera"

# 🔴 THE NEGATIVE MARKER OF THIS CLASS, AND IT IS NOT THE STRING CONFIG HOLDS EITHER.
# `receipt.non_fiscal_marker` in the EU block of config/fiscal-rules.yaml is "NOT A FISCAL
# DOCUMENT" — a statement about FISCALITY, which is a cash-register concept. 👁 What four
# independent commentators describe as printed on real platform receipts is a statement about
# being a TAX INVOICE, which is a different claim: a receipt can be perfectly fiscal and still
# not be the document that lets a buyer deduct the tax. The two are not synonyms and the mock-up
# does not treat them as such; config is left untouched, and which wording that field should
# carry is the author's decision.
NOT_A_TAX_INVOICE = "This is not a VAT invoice."

# Interface wording of a platform receipt. Not law and not marks — the captions the field set
# arrives under. Both dictionaries carry the SAME KEYS, which is what makes the template's
# claim — that language is data — checkable rather than asserted.
EN_LABELS = {
    "title": "Receipt",
    "paid_caption": "Amount paid",
    "receipt_number": "Receipt number",
    "date_paid": "Date paid",
    "payment_method": "Payment method",
    "billed_to": "Billed to",
    "description": "Description",
    "quantity": "Qty",
    "amount": "Amount",
    "subtotal": "Subtotal",
    "total": "Total",
}
UA_LABELS = {
    "title": "Квитанція",
    "paid_caption": "Сплачено",
    "receipt_number": "Номер квитанції",
    "date_paid": "Дата оплати",
    "payment_method": "Спосіб оплати",
    "billed_to": "Платник",
    "description": "Опис",
    "quantity": "К-сть",
    "amount": "Сума",
    "subtotal": "Разом",
    "total": "До сплати",
}

# ⚠️ INVENTED, AND SAFE BECAUSE OF WHAT THEY ARE. A course is a WORK, not a mark under which a
# firm trades — config/generation.yaml admits invented works in as many words while admitting no
# invented marks. Naming a real course would instead assert that a named platform sells it.
EN_COURSES = (
    ("Foundations of Applied Data Analysis", "12 weeks · self-paced"),
    ("Product Analytics for Engineering Teams", "8 weeks · self-paced"),
)
UA_COURSES = (
    ("Основи аналізу даних на практиці", "12 тижнів · у власному темпі"),
    ("Продуктова аналітика для інженерів", "8 тижнів · у власному темпі"),
)

EN_BUYER = {"name": "Olena Kovalchuk", "country": "Ukraine"}
UA_BUYER = {"name": "Ковальчук Олена Петрівна", "country": "Україна"}

# ---------------------------------------------------------------------------
# The multi-page contract — the control for «one page = one document».
#
# 📄 Every caption below is an item of Article 89(2) of the law on insurance (№ 1909-IX), which
# lists NINETEEN particulars an insurance contract must contain. The clause headings follow the
# same list. That is what makes three pages evidenced rather than chosen.
# ---------------------------------------------------------------------------

# 📄 Article 979 of the Civil Code makes the CONTRACT the thing and lets it be issued as a
# поліс or сертифікат — the поліс is a form of the contract, not a class of its own. So the
# document is titled as a contract and the word «поліс» appears nowhere on it: printing both
# would suggest the corpus holds two classes where the law holds one.
INSURANCE_TITLE = "Договір добровільного медичного страхування"
INSURANCE_PLACE = "м. Київ"
INSURANCE_FOLIO = "Сторінка {page} з {of}"

INSURANCE_LABELS = {
    "number": "№",
    "insurer": "Страховик",
    "policyholder": "Страхувальник",
    "tax_code": "ЄДРПОУ",
    "tax_id": "РНОКПП",
    "born": "Дата народження:",
    "subject": "Предмет страхування",
    "object": "Об'єкт страхування",
    "sum_insured": "Страхова сума",
    "tariff": "Страховий тариф",
    "premium": "Страховий платіж (премія)",
    "term": "Строк дії договору",
    "territory": "Територія дії договору",
    "signature_caption": "підпис",
    # 👁 Such a document names the register its licence sits in. No licence number is printed:
    # a number invented beside a named insurer would assert a registration this branch has not
    # checked, exactly as a foreign platform's Ukrainian tax number would.
    "licence_note": "Ліцензія на здійснення страхової діяльності — за даними державного реєстру",
}

INSURANCE_SUBJECT = "Майнові інтереси, пов'язані зі здоров'ям Застрахованої особи"
INSURANCE_OBJECT = "Здоров'я Застрахованої особи"
INSURANCE_TERRITORY = "Україна, крім тимчасово окупованих територій"

# ⚠️ THE CLAUSE PROSE IS INVENTED AND SHORT, and both halves of that are deliberate. Invented is
# safe — a contract clause is a text, not a mark under which a firm trades, and no real document
# was read to produce any of it. Short is a declared limit: what this archetype models is the
# PAGE STRUCTURE and where the required particulars fall, which is what a segmenter and a
# classifier read. Padding it to a realistic length would mean inventing legal text at length to
# no measurable end.
#
# 🔴 WHICH SECTION LANDS ON WHICH PAGE IS DECIDED HERE, not by the stylesheet: a sheet has a
# fixed height and clips rather than reflowing, because a page is a rectangle. So this tuple IS
# the pagination, and a section that grows has to be looked at in the render.
INSURANCE_PAGES = (
    (
        {
            "heading": "1. Загальні положення",
            "clauses": (
                "1.1. Цей Договір укладено відповідно до Закону України «Про страхування» "
                "та Правил добровільного медичного страхування Страховика.",
                "1.2. Договір набирає чинності з дати, зазначеної як початок строку його дії, "
                "за умови сплати страхового платежу в повному обсязі.",
                "1.3. Застрахованою особою за цим Договором є Страхувальник.",
                "1.4. Терміни, вжиті в цьому Договорі, застосовуються у значенні, наведеному "
                "в Законі України «Про страхування» та Правилах страхування.",
            ),
        },
        {
            "heading": "2. Предмет Договору",
            "clauses": (
                "2.1. Страховик зобов'язується у разі настання страхового випадку організувати "
                "та оплатити медичну допомогу Застрахованій особі в межах страхової суми.",
                "2.2. Страхувальник зобов'язується сплатити страховий платіж у розмірі та "
                "строки, визначені цим Договором.",
                "2.3. Обсяг медичної допомоги визначається програмою страхування, що є "
                "невід'ємною частиною цього Договору.",
                "2.4. Медична допомога надається закладами охорони здоров'я, з якими Страховик "
                "має чинні договори про співпрацю на дату звернення.",
            ),
        },
    ),
    (
        {
            "heading": "3. Перелік страхових ризиків",
            "clauses": (
                "3.1. Звернення Застрахованої особи по амбулаторно-поліклінічну допомогу.",
                "3.2. Госпіталізація за медичними показаннями, у тому числі екстрена.",
                "3.3. Виклик швидкої медичної допомоги.",
                "3.4. Придбання лікарських засобів за призначенням лікаря.",
                "3.5. Стоматологічна допомога в обсязі, визначеному програмою страхування.",
                "3.6. Лабораторна та інструментальна діагностика за призначенням лікаря.",
            ),
        },
        {
            "heading": "4. Винятки із страхових випадків та обмеження страхування",
            "clauses": (
                "4.1. Не є страховими випадками звернення з приводу захворювань, діагностованих "
                "до початку строку дії цього Договору.",
                "4.2. Не покриваються витрати на косметологічні та естетичні процедури.",
                "4.3. Не покриваються витрати, понесені поза територією дії Договору.",
                "4.4. Страхова виплата не здійснюється, якщо звернення сталося внаслідок дій "
                "Застрахованої особи у стані алкогольного або наркотичного сп'яніння.",
                "4.5. Не покриваються витрати на санаторно-курортне лікування та реабілітацію, "
                "якщо інше не передбачено програмою страхування.",
            ),
        },
        {
            "heading": "5. Порядок і строки здійснення страхової виплати",
            "clauses": (
                "5.1. Страхова виплата здійснюється шляхом оплати рахунків закладу охорони "
                "здоров'я або відшкодування документально підтверджених витрат.",
                "5.2. Про настання страхового випадку Застрахована особа повідомляє Страховика "
                "за цілодобовим номером контакт-центру до звернення по медичну допомогу, крім "
                "випадків, коли стан здоров'я не дозволяє цього зробити.",
                "5.3. Рішення про виплату приймається протягом 10 робочих днів з дня отримання "
                "повного пакета документів.",
                "5.4. Виплата здійснюється протягом 10 банківських днів з дня прийняття рішення.",
                "5.5. Перелік документів, що подаються для отримання страхової виплати, "
                "визначено програмою страхування.",
            ),
        },
        {
            "heading": "6. Причини відмови у страховій виплаті",
            "clauses": (
                "6.1. Подання Страхувальником свідомо недостовірних відомостей про предмет "
                "Договору або про обставини настання страхового випадку.",
                "6.2. Несвоєчасне повідомлення Страховика про настання страхового випадку без "
                "поважних причин.",
                "6.3. Настання події, що не є страховим випадком за умовами цього Договору.",
                "6.4. Відмова Застрахованої особи від проведення медичного обстеження, "
                "призначеного Страховиком для встановлення обставин страхового випадку.",
            ),
        },
    ),
    (
        {
            "heading": "7. Права та обов'язки Сторін",
            "clauses": (
                "7.1. Страховик зобов'язаний ознайомити Страхувальника з умовами страхування та "
                "нерозголошувати відомості про Страхувальника і його майновий стан.",
                "7.2. Страхувальник зобов'язаний своєчасно сплачувати страхові платежі та "
                "повідомляти Страховика про зміну обставин, що мають істотне значення.",
                "7.3. Страхувальник має право отримати дублікат цього Договору в разі його втрати.",
                "7.4. Страховик має право перевіряти достовірність наданих Страхувальником "
                "відомостей та вимагати документи, необхідні для встановлення обставин "
                "страхового випадку.",
                "7.5. Сторони несуть відповідальність за невиконання або неналежне виконання "
                "умов цього Договору згідно із законодавством України.",
            ),
        },
        {
            "heading": "8. Порядок внесення змін і припинення дії Договору",
            "clauses": (
                "8.1. Зміни до цього Договору вносяться за письмовою згодою Сторін шляхом "
                "укладення додаткової угоди.",
                "8.2. Дія Договору припиняється у випадках, передбачених Законом України "
                "«Про страхування», а також за згодою Сторін.",
            ),
        },
        {
            "heading": "9. Порядок вирішення спорів",
            "clauses": (
                "9.1. Спори за цим Договором вирішуються шляхом переговорів, а в разі "
                "недосягнення згоди — у судовому порядку.",
                "9.2. Договір складено у двох примірниках, що мають однакову юридичну силу, "
                "по одному для кожної із Сторін.",
            ),
        },
    ),
)

# The battery glyph in ua_phone_chrome.jinja is 56 px of shell with a 5 px inset, so a full
# charge is 46 px of fill. Invented, like the clock beside it.
BATTERY_FULL_PX = 46


def _amount(value: Decimal, country: str = "UA") -> str:
    """An amount printed the way a jurisdiction writes one.

    Both separators come from `number_format` in config/fiscal-rules.yaml rather than from
    literals here — UA groups thousands with a NO-BREAK SPACE and the EU block with a comma, and
    the decimal separator goes the other way round. Two currencies in one corpus is exactly the
    condition under which a hard-coded separator starts printing one jurisdiction's number in
    another's dress, with nothing downstream to report it.

    UA declares TWO decimal separators and the last is taken. The contract says real ПРРО
    vendors print both and that the choice is drawn per document; drawing it here would make a
    mock-up's pixels depend on where in the run it was built, and these documents exist to be
    compared with each other.
    """
    number_format = jurisdiction(country)["number_format"]
    decimal_separator = number_format["decimal_separator_variants"][-1]
    thousands_separator = number_format["thousands_separator"]
    whole, _, fraction = f"{value:.2f}".partition(".")
    grouped = f"{int(whole):,}".replace(",", thousands_separator)
    return f"{grouped}{decimal_separator}{fraction}"


def _date_in_words(moment: datetime) -> str:
    """👁 «2 лютого 2024, 13:46» — how the screen states a date, and unlike every document in
    this repository, which prints 02.02.2024. A consumer needs a second reader for it."""
    return (
        f"{moment.day} {MONTHS_GENITIVE[moment.month - 1]} {moment.year}, "
        f"{moment:%H:%M}"
    )


def _descriptor_prefixes() -> tuple[str, ...]:
    """The processor prefixes a card-network descriptor can carry.

    Read from config/vendors.json rather than written here: `payment_providers` and
    `aggregators` are the public Ukrainian processors this repository already names, and each is
    used for exactly the trade it is publicly in. A prefix invented to look plausible would be a
    mark invented to look plausible, which config/generation.yaml forbids at the top of the file.
    """
    vendors = load_vendors()
    providers = [entry["display_name"] for entry in vendors["payment_providers"]["UA"]]
    aggregators = [entry["name"] for entry in vendors["aggregators"]["UA"]]
    return tuple(sorted(name.upper() for name in providers + aggregators))


def _descriptor(rng: random.Random, merchant_name: str) -> str:
    """🔴 `LIQPAY*КОВАЛЬЧУК О.С.` — the counterparty as the card network carries it.

    NOT a legal name, and that is the whole phenomenon: the tail is the merchant's name after
    the descriptor field has had it — uppercased, punctuation squeezed out, the legal form gone.
    An invoice for the same purchase names the merchant properly, so a cross-check comparing the
    two strings fails on a pair of entirely genuine documents.

    ⛔ THE TAIL IS A SOLE TRADER'S NAME, composed by `content_builder.sole_trader_name` from a
    published high-frequency surname, and never an invented firm. A plausible-sounding firm name
    is plausible precisely because it is drawn from the space of real firms — the reasoning
    stated at the head of config/generation.yaml, which cost this project two live companies.
    """
    tail = merchant_name.upper().replace(". ", ".")
    return f"{rng.choice(_descriptor_prefixes())}*{tail}"


# ---------------------------------------------------------------------------
# The contexts
# ---------------------------------------------------------------------------


def app_transaction_context(rng: random.Random) -> dict:
    """One operation as the banking application shows it.

    The amount and the merchant are drawn here rather than taken from a builder, because no
    builder produces this class: it carries no line items, no parties and no requisites, so
    there is nothing for `content_builder` to have built.
    """
    merchant = resolve_vendor(
        rng,
        {"legal_form": "FOP", "profile": "fitness_studio", "vat_payer": False},
        "UA",
    )
    amount = Decimal(rng.randrange(15_000, 240_000)) / 100
    balance = amount + Decimal(rng.randrange(50_000, 900_000)) / 100
    bank = rng.choice(load_vendors()["banks"]["UA"])["printed_name"]

    return {
        "status_time": f"{ISSUED_AT:%H:%M}",
        "battery_fill_px": BATTERY_FULL_PX - rng.randrange(0, 20),
        "merchant_descriptor": _descriptor(rng, merchant["name"]),
        "category": rng.choice(APP_CATEGORIES),
        "datetime_words": _date_in_words(ISSUED_AT),
        # 👁 Signed, with the true minus sign rather than a hyphen — money leaving the account.
        # The direction is the one fact this screen states that the A4 confirmation never prints.
        "amount_signed": f"−{_amount(amount)} {CURRENCY_SIGN}",
        "description_label": APP_STRINGS["description_label"],
        "description_placeholder": APP_STRINGS["description_placeholder"],
        "balance_label": APP_STRINGS["balance_label"],
        "balance": f"{_amount(balance)} {CURRENCY_SIGN}",
        "payment_method_label": APP_STRINGS["payment_method_label"],
        "payment_method": f"{bank} ••{rng.randrange(1000, 10000)}",
        "actions": APP_ACTIONS,
        "tabs": APP_TABS,
        "qr_payload": None,
    }


def receipt_in_app_context(
    rng: random.Random, renderer: Renderer, out_dir: Path
) -> dict:
    """The frame around a bank payment confirmation, and the confirmation itself.

    🔴 THE INNER DOCUMENT IS THE SHIPPED ARCHETYPE, not a copy of it: the same builder, the same
    template, the same stylesheet, rendered by the same renderer. It is written to disk beside
    the images and the frame points an `<iframe>` at it, which is the only way to reuse the
    template without either copying its markup or editing a file three shipped archetypes render
    from. `ua_bank_receipt_in_app.html` states the cost of that choice.

    The confirmation is ALSO rendered on its own, as `ua_bank_payment_confirmation.png`. That is
    not a spare image: the pair is the archetype's whole argument — one document, two carriers,
    and whoever looks at them has to see the same document twice.
    """
    payee = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy",
             "vat_payer": True}
    confirmation = build_payment_confirmation(
        rng,
        issued_at=ISSUED_AT,
        vendor=payee,
        identity=draw_party_identity(rng, payee, "UA"),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
    )
    inner_context = confirmation.render_context()

    # Written from `build_html`, which is what `Renderer.render` writes to its own temporary
    # directory — so the file the frame loads and the page rendered as the A4 twin are one page.
    document_path = out_dir / "ua_bank_receipt_in_app.document.html"
    document_path.write_text(
        renderer.build_html("ua_bank_payment_confirmation", inner_context), encoding="utf-8"
    )

    # THE HEIGHT IS MEASURED, NOT ASSUMED. The A4 stylesheet sets a `min-height`, so a long
    # payment purpose makes the sheet taller; a frame sized from the minimum would clip the foot
    # of the document off the bottom of the card, and the two carriers would then show DIFFERENT
    # documents — the one failure this archetype exists to rule out.
    paper = renderer.render(
        "ua_bank_payment_confirmation",
        inner_context,
        out_dir / "ua_bank_payment_confirmation.png",
    )
    scale = CARD_WIDTH_PX / paper.width

    return {
        "status_time": f"{ISSUED_AT:%H:%M}",
        "battery_fill_px": BATTERY_FULL_PX - rng.randrange(0, 20),
        # The app's wording for the document, which is 👁 not always the document's own.
        "screen_title": (
            f"{APP_STRINGS['receipt_screen_title']} № {confirmation.document_code}"
        ),
        "document_url": document_path.as_uri(),
        "document_width": paper.width,
        "document_height": paper.height,
        "document_scale": f"{scale:.6f}",
        "frame_height": round(paper.height * scale),
        "share_label": APP_STRINGS["share_label"],
        "tabs": APP_TABS,
        "qr_payload": None,
    }


def insurance_contract_context(rng: random.Random) -> dict:
    """A three-page voluntary health insurance contract — the control for «page = document».

    🔴 THE PAGE COUNT IS EVIDENCED, NOT CHOSEN. 📄 Article 89(2) of the law on insurance
    (№ 1909-IX) lists NINETEEN particulars such a contract must contain, including the list of
    risks, the list of exclusions, the payout procedure, the grounds for refusal, the parties'
    rights and obligations, the amendment procedure and the dispute-resolution procedure. Three
    sheets is what those come to. A one-page insurance contract would be the invention.

    ⛔ NOT EVIDENCE OF ANYTHING. `medical_insurance` in config/policy.yaml says a claim of this
    category is proven by a bank payment confirmation naming the policy, and that «the policy
    document itself is out of scope and is never parsed». This archetype exists to be segmented
    and classified, never read for a verdict.

    THE SPLIT ACROSS SHEETS IS DECIDED HERE, not by the stylesheet: a sheet has a fixed height and
    clips, because a page is a rectangle. Which section lands on which page is therefore a
    property of this function, and the render has to be looked at when a section changes length.
    """
    insurer = rng.choice(
        [
            entry
            for entry in load_vendors()["vendors"]["UA"]["medical_insurance"]
            if entry.get("profile") == "insurer"
        ]
    )
    rules = jurisdiction("UA")
    number = f"{rng.randrange(100, 1000)}/{rng.randrange(10000, 100000)}-МС"

    # 📄 Item 7 against item 13: the sum insured and the premium. An order of magnitude apart, and
    # the larger one is the number nobody paid.
    sum_insured = Decimal(rng.randrange(100, 401)) * 1000
    premium = Decimal(rng.randrange(600_000, 1_500_000)) / 100
    starts = ISSUED_AT.date()
    ends = starts.replace(year=starts.year + 1)

    return {
        "title": INSURANCE_TITLE,
        "number_label": INSURANCE_LABELS["number"],
        "number": number,
        "place": INSURANCE_PLACE,
        "date": ISSUED_AT.strftime(rules["date_format"]),
        "running_head": f"{INSURANCE_TITLE} {INSURANCE_LABELS['number']} {number}",
        "labels": INSURANCE_LABELS,
        "insurer": {
            "name": printed_legal_name(insurer["name"], insurer["legal_form"]),
            "address": "м. Київ, вул. Хрещатик, 22",
            "tax_code": generate_edrpou(rng),
            # 👁 Such a document names the register its licence sits in. The register is public
            # and the entry is not quoted — no licence number is printed, because a number
            # invented beside a named insurer would assert a registration this branch has not
            # checked, the way a foreign platform's Ukrainian tax number would.
            "licence": INSURANCE_LABELS["licence_note"],
        },
        "policyholder": {
            "name": UA_BUYER["name"],
            "born": "14.06.1991",
            "address": "м. Київ, вул. Січових Стрільців, 17, кв. 4",
            "tax_id": "2345678901",
        },
        # `mono` marks a value as a FIGURE or a DATE. Three of the required particulars are
        # sentences, and a sentence set in a monospaced face flush right reads as data.
        "facts": [
            {"label": INSURANCE_LABELS["subject"], "value": INSURANCE_SUBJECT,
             "emphasis": False, "mono": False},
            {"label": INSURANCE_LABELS["object"], "value": INSURANCE_OBJECT,
             "emphasis": False, "mono": False},
            # 🔴 The biggest figure on the document, and not an amount anyone paid.
            {
                "label": INSURANCE_LABELS["sum_insured"],
                "value": f"{_amount(sum_insured)} {rules['currency']}",
                "emphasis": True,
                "mono": True,
            },
            {"label": INSURANCE_LABELS["tariff"], "value": "2,8 %",
             "emphasis": False, "mono": True},
            {
                "label": INSURANCE_LABELS["premium"],
                "value": f"{_amount(premium)} {rules['currency']}",
                "emphasis": False,
                "mono": True,
            },
            {
                "label": INSURANCE_LABELS["term"],
                "value": (
                    f"{starts.strftime(rules['date_format'])} — "
                    f"{ends.strftime(rules['date_format'])}"
                ),
                "emphasis": False,
                "mono": True,
            },
            {"label": INSURANCE_LABELS["territory"], "value": INSURANCE_TERRITORY,
             "emphasis": False, "mono": False},
        ],
        "pages": [
            {"folio": INSURANCE_FOLIO.format(page=index + 1, of=len(INSURANCE_PAGES)),
             "sections": sections}
            for index, sections in enumerate(INSURANCE_PAGES)
        ],
        "qr_payload": None,
    }


def _platform_lines(
    rng: random.Random, courses: tuple, price_range: tuple[int, int], country: str
) -> tuple[list[dict], Decimal]:
    """The table of what was bought, and what it comes to — one row per course.

    A PRICE PER LINE, not one price repeated. Two courses at the same figure is what the first
    render showed, and it reads as a template filling itself rather than as a purchase: real
    courses are priced independently, and a corpus in which every line of a document carries the
    same number teaches an extractor that it only has to read one of them.

    Quantity is always one — a course is bought once. The column is on the page because 📄 the
    extent and nature of the service is an invoice particular and 👁 a receipt prints the column
    regardless, not because anything here varies it.
    """
    lines, total = [], Decimal(0)
    for name, period in courses:
        price = Decimal(rng.randrange(*price_range)) / 100
        total += price
        lines.append(
            {"name": name, "period": period, "qty": "1", "amount": _amount(price, country)}
        )
    return lines, total


def eu_platform_receipt_context(rng: random.Random) -> dict:
    """A platform receipt in English and EUR — the second language and the second currency.

    🔴 EVERY REQUISITE OF A VAT INVOICE IS ABSENT, and each absence is 📄 an item of the
    exhaustive list in Article 226 of Directive 2006/112/EC: no supplier address, no supplier VAT
    identification number, no customer VAT identification number, no tax rate and no tax amount.
    The document says as much in the footer, which is the whole reason it may lack them. The
    method is the one `ua_non_fiscal_receipt` used on the Ukrainian fiscal form — read the list of
    required particulars backwards and print the document that carries none of them.

    ⛔ NO CONVERSION IS PRINTED. No public source found shows such a receipt stating a rate or an
    equivalent in the buyer's home currency, so none is stated, and the claim this document
    evidences therefore carries its conversion nowhere on paper.
    """
    rules = jurisdiction("EU")
    lines, total = _platform_lines(rng, EN_COURSES, (4_900, 24_900), "EU")

    return {
        "language": rules["language"],
        "labels": EN_LABELS,
        "currency": rules["currency"],
        "seller": {
            "name": FOREIGN_PLATFORM,
            "address": None,
            "tax_code": None,
            "tax_code_label": None,
            "vat_number": None,
            "vat_number_label": None,
        },
        "buyer": EN_BUYER,
        "receipt_number": f"{rng.randrange(1000, 10000)}-{rng.randrange(1000, 10000)}",
        "date": ISSUED_AT.strftime(rules["date_format"]),
        "card_masked": f"•••• {rng.randrange(1000, 10000)}",
        "lines": lines,
        "subtotal": _amount(total, "EU"),
        # No tax block at all. A zero row would ASSERT a tax treatment, and the absence of the
        # block is what this document's own footer is about.
        "vat": None,
        "total": _amount(total, "EU"),
        "not_a_tax_invoice_note": NOT_A_TAX_INVOICE,
        "support_note": None,
        "qr_payload": None,
        # Not printed anywhere on the page — carried out of here so the run can report it.
        "_amount_decimal": total,
    }


def ua_platform_receipt_context(rng: random.Random) -> dict:
    """The same class in Ukrainian and UAH — the domestic control.

    The seller is read from config/vendors.json: an `online_learning_platform` of the
    `professional_development` category, a public Ukrainian mark already in the pool for exactly
    that trade. Its identifiers are generated with the shipped checksum builders, the way every
    other Ukrainian seller in this repository gets them.

    📄 THE TAX LINE IS «У т.ч. ПДВ», WHICH IS NOT AN ADDITION. The Ukrainian convention prints a
    VAT-inclusive price and states the tax contained in it, so the subtotal and the total are the
    same figure and the tax row sits between them for information. That is what `ua_invoice`
    already does, and its labels are read from config/fiscal-rules.yaml rather than restated.
    """
    platform = next(
        entry
        for entry in load_vendors()["vendors"]["UA"]["professional_development"]
        if entry.get("profile") == "online_learning_platform"
    )
    rules = jurisdiction("UA")
    totals_labels = rules["invoice"]["totals"]
    vat_rate = rules["vat_rates"]["standard"]

    lines, gross = _platform_lines(rng, UA_COURSES, (90_000, 400_000), "UA")
    # The tax CONTAINED in a gross price, not added to it: gross × rate / (100 + rate).
    vat_amount = (gross * Decimal(str(vat_rate)) / (100 + Decimal(str(vat_rate)))).quantize(
        Decimal("0.01")
    )

    edrpou = generate_edrpou(rng)
    labels = dict(UA_LABELS)
    labels["subtotal"] = totals_labels["total_label"]
    labels["total"] = totals_labels["single_label"]

    return {
        "language": rules["language"],
        "labels": labels,
        "currency": rules["currency"],
        "seller": {
            # A ТОВ prints its mark in quotes, the way `legal_name` renders one.
            "name": f"ТОВ «{platform['name']}»",
            "address": "м. Київ, вул. Хрещатик, 22",
            "tax_code": edrpou,
            "tax_code_label": rules["identifiers"]["edrpou"]["label"],
            "vat_number": f"{edrpou}{rng.randrange(1000, 10000)}",
            "vat_number_label": rules["identifiers"]["vat_number"]["label"],
        },
        "buyer": UA_BUYER,
        "receipt_number": f"{rng.randrange(1000, 10000)}-{rng.randrange(1000, 10000)}",
        "date": ISSUED_AT.strftime(rules["date_format"]),
        "card_masked": f"•••• {rng.randrange(1000, 10000)}",
        "lines": lines,
        "subtotal": _amount(gross, "UA"),
        "vat": {
            "label": f"{totals_labels['vat_label']} {vat_rate:g}%",
            "amount": _amount(vat_amount, "UA"),
        },
        "total": _amount(gross, "UA"),
        # 👁 No such note. This seller is registered for ПДВ and prints the tax, so it makes no
        # statement about not being a tax document — and no Ukrainian equivalent of the English
        # wording was found in any public source. The asymmetry is declared in templates/README.md
        # rather than smoothed over with a translation nothing evidences.
        "not_a_tax_invoice_note": None,
        "support_note": None,
        "qr_payload": None,
        "_amount_decimal": gross,
    }


def _report_conversion(eur_total: Decimal) -> None:
    """State the conversion the page cannot show, and say where the rate came from.

    This used to be the only call of `config.load_fx_rates` in the repository, while
    `policy_engine` refused a foreign-currency document outright. The refusal fell with
    the decision to convert in the oracle: `policy_engine.fx_rate` now reads the same
    table, converts at the point fx-rates.yaml declares, and records the applied rate in
    the claim's label. What is still true, and still the reason this prints to the
    console rather than onto the page: 👁 no public source shows such a receipt STATING a
    rate or an equivalent, so the conversion appears on no document — it appears in the
    label, which is where the proof lives.
    """
    fx = load_fx_rates()
    rate = Decimal(str(fx["rates"]["EUR"]))
    base = fx["base"]
    converted = (eur_total * rate).quantize(Decimal("0.01"))
    print(
        f"  note: the EUR receipt totals {eur_total} EUR = {converted} {base} at "
        f"{rate} {base}/EUR (config/fx-rates.yaml). NOT PRINTED ON ANY DOCUMENT — no public "
        "source shows such a receipt stating a rate; the applied rate lives in the label."
    )


# ---------------------------------------------------------------------------


def render_all(out_dir: Path, seed: int) -> list[Path]:
    """Render every mock-up, and the extra page one of them is made of. Returns what was
    written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Renderer() as renderer:
        # One generator for the whole run, so the images are a function of the seed alone.
        rng = random.Random(seed)

        # The contexts are built lazily, one at a time, because the third one RENDERS while it
        # is built — it measures the document that goes inside its card — and a tuple of
        # already-built contexts would put that render before the first line of output.
        for slug, build_context in (
            ("ua_bank_app_transaction", lambda: app_transaction_context(rng)),
            ("ua_bank_receipt_in_app", lambda: receipt_in_app_context(rng, renderer, out_dir)),
            ("eu_platform_receipt", lambda: eu_platform_receipt_context(rng)),
            ("ua_platform_receipt", lambda: ua_platform_receipt_context(rng)),
            ("ua_insurance_contract", lambda: insurance_contract_context(rng)),
        ):
            context = build_context()
            # Keys the run needs and no template prints. Popped rather than left for Jinja to
            # ignore: a context is what the page says, and a value in it that reaches no page is
            # the sort of thing a later reader wires into markup by mistake.
            amount = context.pop("_amount_decimal", None)

            result = renderer.render(slug, context, out_dir / f"{slug}.png")
            written.append(result.image_path)
            # Reported so the boundary is visible in the run rather than only in a README: a
            # template with no `data-field` attribute yields no boxes, and no labels are written.
            print(
                f"{slug}: {result.width}×{result.height} px, "
                f"{len(result.field_bboxes)} bounding boxes, no labels"
            )
            if slug == "eu_platform_receipt":
                _report_conversion(amount)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render-mockups",
        description="Render the mock-up archetypes to PNGs outside the repository.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--seed", type=int, default=20260731, help="the run's only seed")
    args = parser.parse_args(argv)

    out_dir = args.out.resolve()
    if out_dir.is_relative_to(REPO_ROOT):
        # Refused rather than gitignored: rendered images are not what this repository ships,
        # and a path that merely happens to be ignored today is one commit from being tracked.
        print(
            f"refusing to write inside the repository: {out_dir}\n"
            "  the mock-up renders belong outside it — pass --out with a path elsewhere",
            file=sys.stderr,
        )
        return 2

    for path in render_all(out_dir, args.seed):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
