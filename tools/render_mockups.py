#!/usr/bin/env python3
"""Render the mock-up archetypes to PNGs, so they can be looked at.

    uv run python tools/render_mockups.py [--out dir] [--seed N]

why this is a script and not part of the pipeline. Every template it renders is a mock-up:
none is registered in `claim_planner.ARCHETYPES`, none has a builder in
`assembler._BUILDERS`, and none reaches a dataset. So there is no place in the pipeline for them,
and inventing one would be a connection nobody has decided to make. See templates/README.md.

⚠️ six mock-ups have left this script by being connected, and the rule was the same every time:
a second, script-only way of building a shipped page drifts from the pipeline the moment either
changes. `ua_non_fiscal_receipt`, the two platform receipts and the two phone carriers are
registered, built by their builders and rendered by runs — their strings live in
config/fiscal-rules.yaml (`receipt.non_fiscal`, `platform_receipt`, `bank_app`), which is what
this file's own comments always said would happen to them. `ua_claim_bundle` is the composition
template `assembler._compose_bundle` renders at `file_composition.bundle_share`. What remains
below is one mock-up: the insurance contract, blocked on RC-14, the author's open decision.

The output goes outside the repository by default, for the same reason `out/` is gitignored:
rendered images are not what this repository ships. `--out` overrides it, and a path inside
the repository is refused rather than quietly written.

Nothing here produces a label. No `data-field` attribute exists in any of the templates,
so `RenderedDocument.field_bboxes` comes back empty by construction, and this script writes no
JSON beside the images. That is the boundary the branch was given: labels and boxes arrive with
the connection, not with the layout.

Determinism. One seed, one set of images. Every draw goes through the `random.Random` created
here.
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
    load_vendors,
)
from receipt_synth.content_builder import (
    generate_edrpou,
    printed_legal_name,
)
from receipt_synth.renderer import REPO_ROOT, Renderer

DEFAULT_OUT = Path(tempfile.gettempdir()) / "receipt-synth-mockups"

# The instant every mock-up is dated from. Fixed rather than "now": an image that changes with
# the clock is one no second run reproduces, which is the property this repository is built on.
ISSUED_AT = datetime(2026, 3, 17, 13, 52, 41)

# ---------------------------------------------------------------------------
# The multi-page contract — the control for «one page = one document».
#
# 📄 Every caption below is an item of Article 89(2) of the law on insurance (№ 1909-IX), which
# lists nineteen particulars an insurance contract must contain. The clause headings follow the
# same list. That is what makes three pages evidenced rather than chosen.
# ---------------------------------------------------------------------------

# 📄 Article 979 of the Civil Code makes the contract the thing and lets it be issued as a
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

# ⚠️ The clause prose is invented and short, and both halves of that are deliberate. Invented is
# safe — a contract clause is a text, not a mark under which a firm trades, and no real document
# was read to produce any of it. Short is a declared limit: what this archetype models is the
# page structure and where the required particulars fall, which is what a segmenter and a
# classifier read. Padding it to a realistic length would mean inventing legal text at length to
# no measurable end.
#
# 🔴 Which section lands on which page is decided here, not by the stylesheet: a sheet has a
# fixed height and clips rather than reflowing, because a page is a rectangle. So this tuple is
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
    literals here — UA groups thousands with a no-break space and the EU block with a comma, and
    the decimal separator goes the other way round. Two currencies in one corpus is exactly the
    condition under which a hard-coded separator starts printing one jurisdiction's number in
    another's dress, with nothing downstream to report it.

    UA declares two decimal separators and the last is taken. The contract says real ПРРО
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


# ---------------------------------------------------------------------------
# The contexts
# ---------------------------------------------------------------------------


def insurance_contract_context(rng: random.Random) -> dict:
    """A three-page voluntary health insurance contract — the control for «page = document».

    🔴 the page count is evidenced, not chosen. 📄 Article 89(2) of the law on insurance
    (№ 1909-IX) lists nineteen particulars such a contract must contain, including the list of
    risks, the list of exclusions, the payout procedure, the grounds for refusal, the parties'
    rights and obligations, the amendment procedure and the dispute-resolution procedure. Three
    sheets is what those come to. A one-page insurance contract would be the invention.

    ⛔ not evidence of anything. `medical_insurance` in config/policy.yaml says a claim of this
    category is proven by a bank payment confirmation naming the policy, and that «the policy
    document itself is out of scope and is never parsed». This archetype exists to be segmented
    and classified, never read for a verdict.

    The split across sheets is decided here, not by the stylesheet: a sheet has a fixed height and
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
            # The corpus's stock buyer name — a drawn-style full name no register holds.
            "name": "Ковальчук Олена Петрівна",
            "born": "14.06.1991",
            "address": "м. Київ, вул. Січових Стрільців, 17, кв. 4",
            "tax_id": "2345678901",
        },
        # `mono` marks a value as a figure or a date. Three of the required particulars are
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


# ---------------------------------------------------------------------------


def render_all(out_dir: Path, seed: int) -> list[Path]:
    """Render every mock-up, and the extra page one of them is made of. Returns what was
    written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Renderer() as renderer:
        # One generator for the whole run, so the images are a function of the seed alone.
        rng = random.Random(seed)

        # One mock-up left. This script once drew seven; six have been connected and each
        # took its strings into config and its rendering into the pipeline as it went —
        # the same migration every time, recorded per archetype in templates/README.md.
        for slug, build_context in (
            ("ua_insurance_contract", lambda: insurance_contract_context(rng)),
        ):
            context = build_context()
            # Keys the run needs and no template prints. Popped rather than left for Jinja to
            # ignore: a context is what the page says, and a value in it that reaches no page is
            # the sort of thing a later reader wires into markup by mistake.
            context.pop("_amount_decimal", None)

            result = renderer.render(slug, context, out_dir / f"{slug}.png")
            written.append(result.image_path)
            # Reported so the boundary is visible in the run rather than only in a README: a
            # template with no `data-field` attribute yields no boxes, and no labels are written.
            print(
                f"{slug}: {result.width}×{result.height} px, "
                f"{len(result.field_bboxes)} bounding boxes, no labels"
            )

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
