#!/usr/bin/env python3
"""Render the three mock-up archetypes to PNGs, so they can be looked at.

    uv run python tools/render_mockups.py [--out DIR] [--seed N]

WHY THIS IS A SCRIPT AND NOT PART OF THE PIPELINE. The three templates it renders are
MOCK-UPS: none is registered in `claim_planner.ARCHETYPES`, none has a builder in
`assembler._BUILDERS`, and none may reach a dataset yet — the production run measures four
document classes, and adding layout variety before that measurement would change what the
measurement means. So there is no place in the pipeline for them, and inventing one would be
the connection this branch deliberately does not make. See templates/README.md.

THE OUTPUT GOES OUTSIDE THE REPOSITORY by default, for the same reason `out/` is gitignored:
rendered images are not what this repository ships. `--out` overrides it, and a path inside
the repository is refused rather than quietly written.

NOTHING HERE PRODUCES A LABEL. No `data-field` attribute exists in any of the three templates,
so `RenderedDocument.field_bboxes` comes back empty by construction, and this script writes no
JSON beside the images. That is the boundary the branch was given: labels and boxes arrive with
the connection, not with the layout.

DETERMINISM. One seed, one set of images. Every draw goes through the `random.Random` created
here, and the two documents built by the shipped builders (`build_prro_receipt`,
`build_payment_confirmation`) take it as their generator, exactly as the assembler does.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from receipt_synth.config import jurisdiction, load_vendors
from receipt_synth.content_builder import (
    build_payment_confirmation,
    build_prro_receipt,
    resolve_vendor,
)
from receipt_synth.renderer import REPO_ROOT, Renderer
from receipt_synth.schemas import Capture

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

# 🔴 THE TITLE OF THE NON-FISCAL SLIP, AND IT IS NOT THE STRING CONFIG HOLDS.
# `receipt.non_fiscal_marker` in config/fiscal-rules.yaml is «НЕ ФІСКАЛЬНИЙ ЧЕК», and no public
# source found says that wording is printed on a Ukrainian SALES document. What the tax service
# states (БЗ 109.10) is that such a document carries the wording «Товарний чек» and omits the
# fiscal number and the wording «Фіскальний чек». So the mock-up prints what is evidenced, and
# the configured constant is left exactly as it is: changing it is the author's decision after
# reading the report, not a side effect of a mock-up. See templates/README.md.
NON_FISCAL_TITLE = "ТОВАРНИЙ ЧЕК"

# 📄 ст. 9 of the accounting law № 996-XIV: a primary document names the person responsible for
# the operation and carries their signature. A fiscal receipt carries neither.
ISSUER_LABEL = "Видав:"
SIGNATURE_LABEL = "Підпис"

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

# The battery glyph in ua_phone_chrome.jinja is 56 px of shell with a 5 px inset, so a full
# charge is 46 px of fill. Invented, like the clock beside it.
BATTERY_FULL_PX = 46


def _amount_uk(value: Decimal) -> str:
    """An amount the way a Ukrainian interface writes it: comma decimals, spaced thousands."""
    whole, _, fraction = f"{value:.2f}".partition(".")
    grouped = f"{int(whole):,}".replace(",", " ")
    return f"{grouped},{fraction}"


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
# The three contexts
# ---------------------------------------------------------------------------


def non_fiscal_context(rng: random.Random) -> dict:
    """A товарний чек, built from the fiscal receipt's own builder.

    🔴 THE BASKET, THE ARITHMETIC AND THE AMOUNT IN WORDS COME FROM `build_prro_receipt`, and
    that is the point rather than a shortcut: the trap only works if this document is what a
    fiscal receipt would have been for the same purchase. The builder also computes a fiscal
    identity — a fiscal number, a QR payload, a register's maker — and the template prints none
    of it. The difference between the two documents IS the set of requisites left out.

    The seller is a sole trader who is not registered for ПДВ. 📄 A ПДВ payer is obliged to use
    a register, so the issuer of such a slip is a non-payer; the receipt therefore carries an
    «ІД» line and no «ПН» line, and no tax rows under the totals.
    """
    vendor = resolve_vendor(
        rng,
        {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False},
        "UA",
    )
    receipt = build_prro_receipt(
        rng,
        category_id="vitamins_nutrition",
        issued_at=ISSUED_AT,
        vendor=vendor,
        capture=Capture.SCAN,
    )
    context = receipt.render_context()
    rules = jurisdiction("UA")

    # A hand-kept book of товарні чеки is numbered sequentially. The builder's own number is a
    # ПРРО's eleven-character identifier, which a document with no ПРРО cannot have carried.
    context["receipt_number"] = f"{rng.randint(1, 9999):04d}"
    context["title"] = NON_FISCAL_TITLE
    context["issuer_label"] = ISSUER_LABEL
    context["issuer_name"] = vendor["name"]
    context["signature_label"] = SIGNATURE_LABEL
    context["footer"] = rules["receipt"]["footer"]

    # 🔴 CASH, AND IT IS NOT A COSMETIC CHOICE. The builder drew «БЕЗГОТІВКОВА», which is the
    # only value any shipped document prints — `payment_method_labels` in config/fiscal-rules.yaml
    # notes that its second entry is unreachable. On THIS document the first entry is the wrong
    # one: 📄 a card sale is a settlement operation that obliges the seller to use a register, so
    # a slip issued without one records cash. The mock-up is therefore the first document in this
    # repository to print «ГОТІВКА», and the value is read from config rather than written here.
    context["payment_method"] = rules["acquiring_block"]["payment_method_labels"][1]
    # No QR: 📄 it is a requisite of the FISCAL form. The renderer requires the key, and `None`
    # is how a template says the document has none.
    context["qr_payload"] = None
    return context


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
        "amount_signed": f"−{_amount_uk(amount)} {CURRENCY_SIGN}",
        "description_label": APP_STRINGS["description_label"],
        "description_placeholder": APP_STRINGS["description_placeholder"],
        "balance_label": APP_STRINGS["balance_label"],
        "balance": f"{_amount_uk(balance)} {CURRENCY_SIGN}",
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
    confirmation = build_payment_confirmation(
        rng,
        issued_at=ISSUED_AT,
        vendor={"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy",
                "vat_payer": True},
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


# ---------------------------------------------------------------------------


def render_all(out_dir: Path, seed: int) -> list[Path]:
    """Render the three mock-ups, and the A4 twin of the third. Returns what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Renderer() as renderer:
        # One generator for the whole run, so the images are a function of the seed alone.
        rng = random.Random(seed)

        # The contexts are built lazily, one at a time, because the third one RENDERS while it
        # is built — it measures the document that goes inside its card — and a tuple of
        # already-built contexts would put that render before the first line of output.
        for slug, build_context in (
            ("ua_non_fiscal_receipt", lambda: non_fiscal_context(rng)),
            ("ua_bank_app_transaction", lambda: app_transaction_context(rng)),
            ("ua_bank_receipt_in_app", lambda: receipt_in_app_context(rng, renderer, out_dir)),
        ):
            result = renderer.render(slug, build_context(), out_dir / f"{slug}.png")
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
