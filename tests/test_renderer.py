"""Rendering: the image, and the bounding boxes that must agree with it.

A bounding box is only worth anything if it indexes the image it claims to. These tests
check that agreement — bounds, dimensions, one box per marked field — and that the same
input renders to the same bytes, which is what `--seed` promises.

One browser session for the whole module: launching Chromium costs far more than
rendering a page.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from jinja2 import Environment, FileSystemLoader, UndefinedError, meta
from PIL import Image

from receipt_synth.assembler import _BUILDERS
from receipt_synth.claim_planner import ARCHETYPES
from receipt_synth.config import jurisdiction
from receipt_synth.content_builder import (
    build_payment_confirmation,
    build_prro_receipt,
    resolve_vendor,
)
from receipt_synth.renderer import FONT_FILES, FONTS_DIR, TEMPLATES_DIR, Renderer, qr_svg
from receipt_synth.schemas import DocType

TEMPLATE = "ua_prro_receipt"
# Millimetres of paper per rendered pixel is fixed across the UA fiscal receipts: 640 px is
# the 80 mm roll, so eight pixels are one millimetre. Stated here because it is what ties a
# stylesheet's `width` to `receipt.widths_mm` in config/fiscal-rules.yaml — without it a width
# in pixels and a width in millimetres are two unrelated numbers.
PIXELS_PER_MM = 8

# Every archetype the planner may draw, and therefore every template a dataset can contain.
# Read from the registry rather than listed, so a template registered without being rendered
# here is impossible.
REGISTERED_SLUGS = sorted(ARCHETYPES)

# AND THE SAME SET SPLIT BY DOCUMENT CLASS, because the registry no longer holds one class. A
# receipt is printed on a till roll whose width is one the jurisdiction's suppliers sell and
# carries a fiscal foot; a bank payment confirmation is an A4 page with neither. Every test below
# that asserts a receipt fact reads THIS list, so registering a third class cannot make a
# receipt-shaped assertion quietly apply to it — and the two lists are checked against the
# registry, so a class nobody assigned cannot slip through either.
FISCAL_SLUGS = sorted(
    slug for slug, a in ARCHETYPES.items() if a.doc_type is DocType.FISCAL_RECEIPT
)
CONFIRMATION_SLUGS = sorted(
    slug for slug, a in ARCHETYPES.items() if a.doc_type is DocType.PAYMENT_CONFIRMATION
)


def template_source(template_name: str) -> str:
    """A template's own source plus the source of every template it includes.

    FOLLOWING INCLUDES IS WHAT KEEPS THE SOURCE-LEVEL GUARDS BELOW HONEST, and it is not a
    convenience. Three archetypes share one body — `templates/ua_fiscal_receipt.jinja` — so
    each `<slug>.html` is a comment and one `{% include %}`. A guard that read `<slug>.html`
    alone would inspect a file with no markup in it and pass on anything whatsoever, while
    still reporting green: the most expensive kind of failure, because it disarms the reader.

    Jinja comments are stripped, so a rule quoted in a comment is not mistaken for markup.
    The stylesheet include is skipped: every guard built on this asks a question about markup,
    and a Ukrainian term inside a CSS comment is exactly the false positive `printed_text`
    exists to avoid.
    """
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    pending = [template_name]
    seen: set[str] = set()
    sources: list[str] = []
    while pending:
        name = pending.pop()
        if name in seen or name.endswith(".css"):
            continue
        seen.add(name)
        source = env.loader.get_source(env, name)[0]
        sources.append(source)
        referenced = list(meta.find_referenced_templates(env.parse(source)))
        # `None` is what a dynamically named include resolves to. There are none today, and one
        # added later must fail here rather than drop silently out of the sweep, which would
        # narrow every guard below without any of them going red.
        assert None not in referenced, f"{name} includes a dynamically named template"
        pending += referenced
    return re.sub(r"\{#.*?#\}", "", "\n".join(sources), flags=re.DOTALL)

# A registered ПДВ payer and an unregistered seller. Both are ordinary — a pharmacy chain is
# registered, a sole trader on the simplified system is not — and the two print DIFFERENT
# REQUISITES, which is what the pair is here to exercise.
#
# The unregistered one is a SOLE TRADER with a DRAWN name, matching the only unregistered
# variety config/vendors.json configures. Nothing rendered from it names a firm, which matters
# because an «ІД» line asserts that its seller is not VAT-registered.
PAYER = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
NON_PAYER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False},
    "UA",
)
# A REGISTERED sole trader — on the general system rather than the simplified one. Its ПН is its
# РНОКПП, so both identifier lines carry the same ten digits.
PAYER_SOLE_TRADER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": True},
    "UA",
)


def make_receipt(seed: int = 20260803, vendor: dict = PAYER, registrar: str = "prro"):
    return build_prro_receipt(
        random.Random(seed),
        category_id="vitamins_nutrition",
        issued_at=datetime(2026, 8, 3, 14, 22, 51),
        vendor=vendor,
        address="м. Київ, вул. Хрещатик, 22",
        registrar=registrar,
    )


def make_confirmation(seed: int = 20260417, vendor: dict = PAYER, initiation: str | None = None):
    """One bank payment confirmation.

    The payer's name and tax number are stated here rather than drawn from a persona: this test
    module renders documents, and pulling in `persona_generator` would make every rendering test
    depend on the draw order of a stage that has nothing to do with the page.
    """
    return build_payment_confirmation(
        random.Random(seed),
        issued_at=datetime(2026, 4, 17, 11, 3, 9),
        vendor=vendor,
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        initiation=initiation,
    )


def context_for(slug: str) -> dict:
    """A render context for any registered archetype, built by its document class.

    THE ONE PLACE THAT KNOWS WHICH BUILDER FEEDS WHICH TEMPLATE, so the whole-registry tests
    below stay whole-registry as classes are added. Before the second class landed they all built
    a fiscal receipt, which was correct only because every archetype was one — and a payment
    confirmation rendered from a receipt's context would raise on the first missing key rather
    than assert anything about the page.
    """
    doc_type = ARCHETYPES[slug].doc_type
    if doc_type is DocType.FISCAL_RECEIPT:
        return make_receipt(registrar=registrar_of(slug)).render_context()
    if doc_type is DocType.PAYMENT_CONFIRMATION:
        return make_confirmation().render_context()
    raise AssertionError(
        f"{slug} is a {doc_type.value}, and this module has no context for that class — a "
        "registered archetype nothing here can render is one no test below covers"
    )


# Which kind of register each archetype's documents come from — the pairing the assembler
# makes. Taken from `assembler._BUILDERS` rather than restated, because a slug rendered here
# under the wrong registrar would produce a page no run can produce.
def registrar_of(slug: str) -> str:
    return getattr(_BUILDERS[slug], "keywords", {}).get("registrar", "prro")


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    context = make_receipt().render_context()
    output = tmp_path_factory.mktemp("render") / "receipt.png"
    return renderer.render(TEMPLATE, context, output)


# ---------------------------------------------------------------- the image --


def test_image_is_written_with_the_reported_dimensions(rendered):
    with Image.open(rendered.image_path) as image:
        assert image.size == (rendered.width, rendered.height)


@pytest.mark.parametrize(
    "template_path", sorted(TEMPLATES_DIR.glob("*.html")), ids=lambda path: path.stem
)
def test_every_template_declares_exactly_one_document_root(template_path):
    """The renderer falls back to the whole document when no `[data-document]` element
    is found, and that fallback is silent: a template missing the marker would still
    render, just framed as the browser window rather than as the document. Nothing
    downstream would report it, so it is asserted here, for every template — including
    the ones still to be written.

    Counted over the template AND its includes, because three archetypes carry their root
    through a shared body. EXACTLY one either way: a root in the body and another in an
    including template would frame the image on whichever the browser met first.
    """
    assert template_source(template_path.name).count("data-document") == 1, (
        "expected exactly one document root, counting includes"
    )


# The class each document class names its page root. `.receipt` for a till roll and `.page` for
# an A4 sheet — a distinction worth keeping in the markup, since the two are not the same object
# and a rule written for one must not reach the other through a shared name.
_ROOT_SELECTORS = ("receipt", "page")


def declared_width_px(slug: str) -> int:
    """The width `<slug>.css` gives the document root.

    Read out of the archetype's OWN stylesheet, which is where the paper width lives: the
    shared stylesheet the three fiscal archetypes layer it over sets no width at all, so a width
    that migrated there would leave one archetype silently taking another's paper.

    EXACTLY ONE root rule may declare a width, over both spellings. A file declaring neither
    would leave the paper to the browser; one declaring both would make this function report
    whichever came first, and the test built on it would pass while measuring the wrong rule.
    """
    stylesheet = (TEMPLATES_DIR / f"{slug}.css").read_text(encoding="utf-8")
    declared = [
        int(match.group(1))
        for selector in _ROOT_SELECTORS
        for match in re.finditer(
            rf"\.{selector}\s*\{{[^}}]*?\bwidth:\s*(\d+)px", stylesheet, re.DOTALL
        )
    ]
    assert len(declared) == 1, (
        f"{slug}.css should give its document root exactly one explicit width; found {declared}"
    )
    return declared[0]


def test_the_image_width_is_the_declared_document_width(rendered):
    """Positive counterpart to the guard below.

    Together with the marker test above this closes the chain: the template declares
    exactly one document root, the stylesheet gives that root a width, and the image is
    that wide — so the frame came from the document rather than from the browser.
    """
    assert rendered.width == declared_width_px(TEMPLATE)


@pytest.mark.parametrize("slug", FISCAL_SLUGS)
def test_every_fiscal_archetype_is_rendered_on_a_paper_width_the_jurisdiction_sells(slug):
    """A receipt is printed on a roll, and rolls come in fixed widths — `receipt.widths_mm` in
    config/fiscal-rules.yaml, 58 mm and 80 mm for Ukraine. Until this test the list was read by
    nothing, so a stylesheet at 700 px would have rendered a paper width no supplier stocks and
    every image of that archetype would carry it.

    The scale is the one 640 px = 80 mm fixes, and it is asserted for the whole set rather than
    per template: if the two widths mapped through different scales the ratio between the images
    would mean nothing.

    SCOPED TO THE FISCAL CLASS, and the scope is the point rather than a caveat. `widths_mm` is
    the widths a thermal ROLL is sold in; a bank payment confirmation is an A4 page, and holding
    it to this list would fail on a document that is correct — or, worse, pass if somebody
    "fixed" it by adding 210 mm to a list of till-roll widths.
    """
    widths_mm = jurisdiction("UA")["receipt"]["widths_mm"]
    width_px = declared_width_px(slug)
    assert width_px % PIXELS_PER_MM == 0, f"{width_px} px is not a whole number of millimetres"
    assert width_px // PIXELS_PER_MM in widths_mm, (
        f"{slug} is {width_px // PIXELS_PER_MM} mm wide, and config/fiscal-rules.yaml lists "
        f"{widths_mm}"
    )


def test_both_configured_paper_widths_are_actually_rendered():
    """The other direction. Every archetype sitting on one width would satisfy the test above
    while the narrow roll reached no image at all — and the whole reason the 58 mm variant is
    its own archetype is that a corpus rendered only on the wide roll teaches a consumer where
    the amount column sits."""
    rendered_mm = {declared_width_px(slug) // PIXELS_PER_MM for slug in FISCAL_SLUGS}
    assert set(jurisdiction("UA")["receipt"]["widths_mm"]) == rendered_mm


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_every_registered_archetype_has_a_template_a_stylesheet_and_a_builder(slug):
    """An archetype the planner can draw and nothing can produce fails mid-run, on a machine
    nobody is watching, after some of the dataset has been written. All three halves have to
    exist: `_build_document` names a missing builder, but a missing `<slug>.css` surfaces as a
    `FileNotFoundError` out of the renderer and a missing `<slug>.html` as a Jinja
    `TemplateNotFound`, neither of which says that the registry is what disagrees."""
    assert (TEMPLATES_DIR / f"{slug}.html").is_file(), "no template for this archetype"
    assert (TEMPLATES_DIR / f"{slug}.css").is_file(), "no stylesheet for this archetype"
    assert slug in _BUILDERS, "no builder produces this archetype"


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_every_registered_archetype_renders_with_every_box_on_the_paper(renderer, tmp_path, slug):
    """Every archetype rendered, not just the one the fixtures use — and the narrow roll is why.

    A template that is correct and ugly is still a defect on every image of its class: a line
    that runs past the paper edge, or a column that collapses onto its label, is invisible to a
    test that only ever renders the wide roll. Three things are asserted per archetype, and
    together they are what "it fits on the paper" means: every marked field has a box, every box
    lies inside the image, and no box has zero area.

    NOT A SUBSTITUTE FOR LOOKING AT THE IMAGE. Text that overflows its own element still reports
    a box inside the paper, so this catches the collapse and the overrun and cannot catch
    ugliness. The renders were also inspected by eye when this archetype landed.
    """
    context = context_for(slug)
    result = renderer.render(slug, context, tmp_path / f"{slug}.png")
    marked = marked_fields(renderer.build_html(slug, context))

    assert set(result.field_bboxes) == marked, "a marked field lost its box, or gained one"
    for name, (x, y, width, height) in result.field_bboxes.items():
        assert width > 0 and height > 0, f"{slug}: {name} has no area"
        assert x >= 0 and y >= 0, f"{slug}: {name} starts off the paper"
        assert x + width <= result.width, f"{slug}: {name} runs past the paper edge"
        assert y + height <= result.height, f"{slug}: {name} runs past the bottom"


def test_image_is_the_document_not_the_window(rendered):
    """The viewport defaults to 1280×720. If either dimension came back as a browser
    default, the image would be the document plus a margin of blank paper — and every
    bounding box would still be correct, so nothing else here would notice."""
    assert rendered.width == 640, "the receipt is 640 px wide in the stylesheet"
    assert rendered.height not in (720, 1024), "height looks like a viewport, not content"


def test_rendering_is_byte_identical_for_the_same_input(renderer, tmp_path):
    """What `--seed` promises, at the last stage that could break it. Fonts are vendored
    precisely so this holds on a machine other than the one that wrote it."""
    context = make_receipt().render_context()
    first = renderer.render(TEMPLATE, context, tmp_path / "a.png")
    second = renderer.render(TEMPLATE, context, tmp_path / "b.png")

    assert first.image_path.read_bytes() == second.image_path.read_bytes()
    assert first.field_bboxes == second.field_bboxes


# --------------------------------------------------------------- the boxes --


def test_every_marked_field_has_a_box(renderer, rendered):
    """The template is the source of truth for what is extractable. A field marked in
    the markup but missing from the labels would be a silently unannotated field."""
    html = renderer.build_html(TEMPLATE, make_receipt().render_context())
    marked = set(re.findall(r'data-field="([^"]+)"', html))

    assert marked, "the template should mark extractable fields"
    assert set(rendered.field_bboxes) == marked


def test_boxes_lie_inside_the_image(rendered):
    for name, (x, y, width, height) in rendered.field_bboxes.items():
        assert x >= 0 and y >= 0, f"{name} starts outside the image"
        assert x + width <= rendered.width, f"{name} runs past the right edge"
        assert y + height <= rendered.height, f"{name} runs past the bottom edge"


def test_boxes_are_not_degenerate(rendered):
    """A zero-area box points at nothing. It is what an element hidden by CSS, or one
    that failed to receive its content, leaves behind."""
    for name, (_, _, width, height) in rendered.field_bboxes.items():
        assert width > 0 and height > 0, f"{name} has no area"


def test_boxes_are_whole_pixels(rendered):
    """A box is an index into an image; a fractional index means nothing to a consumer."""
    for box in rendered.field_bboxes.values():
        assert all(float(value).is_integer() for value in box)


def test_indexed_fields_follow_the_line_items(renderer, tmp_path):
    """One set of boxes per line item, ordered down the page — so a consumer can pair
    box i with line item i without matching text."""
    receipt = make_receipt()
    result = renderer.render(TEMPLATE, receipt.render_context(), tmp_path / "items.png")

    tops = []
    for index in range(len(receipt.line_items)):
        assert f"item_{index}_name" in result.field_bboxes
        tops.append(result.field_bboxes[f"item_{index}_name"][1])
    assert tops == sorted(tops), "line items are not in document order"


def test_the_total_box_sits_below_the_line_items(rendered):
    """LINE-ITEM names only, matched on the indexed key. `endswith("_name")` also caught
    `seller_name` — harmlessly, being above the total — and then `provider_name`, which sits
    in the foot BELOW it, so the loose filter turned a true property into a failing test. A
    filter that happens to work is a filter that stops working when a field is added."""
    item_names = [name for name in rendered.field_bboxes if re.fullmatch(r"item_\d+_name", name)]
    assert item_names, "no line-item name boxes — this test would assert nothing"

    last_item = max(rendered.field_bboxes[name][1] for name in item_names)
    assert rendered.field_bboxes["total"][1] > last_item


def test_the_four_total_lines_are_printed_in_the_order_of_the_form(rendered):
    """`СУМА`, `ЗНИЖКА`, `ЗАОКРУГЛЕННЯ`, `ДО СПЛАТИ` are four lines of the published form,
    in that order. They were one merged line until this version, which is neither of the two
    the form defines."""
    order = ("total", "discount", "rounding", "amount_due")
    tops = [rendered.field_bboxes[name][1] for name in order]
    assert tops == sorted(tops), "the totals are not in the order the form gives them"
    assert len(set(tops)) == 4, "four separate lines, not one row of four values"


# --------------------------------------------------- VAT-payer status on the page --


@pytest.fixture(scope="module")
def rendered_non_payer(renderer, tmp_path_factory):
    context = make_receipt(vendor=NON_PAYER).render_context()
    output = tmp_path_factory.mktemp("render_non_payer") / "receipt.png"
    return renderer.render(TEMPLATE, context, output)


def marked_fields(html: str) -> set[str]:
    """The `data-field` names present in a RENDERED page — which is what exists on this
    document, as opposed to what the template can print for some other seller."""
    return set(re.findall(r'data-field="([^"]+)"', html))


def printed_text(html: str) -> str:
    """The markup after the inlined stylesheet, so that a Ukrainian term in a CSS comment
    cannot be mistaken for one printed on the paper."""
    head, _, body = html.rpartition("</style>")
    assert head, "the page should carry an inlined stylesheet"
    return body


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_identifier_prefix_is_not_written_into_the_template(slug):
    """Every prefix comes from `identifiers` in config/fiscal-rules.yaml or from
    `receipt.registrars`, chosen by the seller's VAT status and the kind of register. A prefix
    in the markup is the whole cause of the defect the ПН/ІД pair once had: the template cannot
    know the status, so one of the two lines it printed was always a requisite the seller
    cannot hold.

    «ФН» and «ЗН» are in the sweep for the same reason and it was earned the same way. «ФН
    ПРРО» WAS a literal in this markup, so the hardware archetype would have printed a ПРРО's
    prefix on a document from a machine that has no ПРРО — the identical defect one requisite
    over.
    """
    body = template_source(f"{slug}.html")

    assert "ПН" not in body, "the VAT-payer prefix is hardcoded in the markup"
    assert "ІД" not in body, "the tax-number prefix is hardcoded in the markup"
    assert "ФН" not in body, "the fiscal-number prefix is hardcoded in the markup"
    assert "ЗН" not in body, "the factory-serial prefix is hardcoded in the markup"


def test_the_totals_labels_come_from_the_configuration():
    """The four amount lines under the items are receipt layout constants, so they live in
    `receipt.totals_labels` in config/fiscal-rules.yaml beside the acquiring-block labels and
    the tax-row label — not as literals in the markup.

    It is not tidiness: ONE OF THE FOUR VARIES BETWEEN SOURCES. `ЗАОКРУГЛЕННЯ` is what the
    observed receipt prints and `ОКРУГЛЕННЯ` is what the published table of the form writes, and
    a config value can carry that alternative with the evidence for each side while a literal
    cannot. Asserted from the config rather than restated here, so the two cannot drift.
    """
    labels = jurisdiction("UA")["receipt"]["totals_labels"]
    assert set(labels) == {"total", "discount", "rounding", "amount_due"}

    body = template_source(f"{TEMPLATE}.html")
    for key, label in labels.items():
        assert label not in body, f"{key} label {label!r} is hardcoded in the markup"


def test_the_configured_totals_labels_are_what_gets_printed(renderer):
    """Positive counterpart: the labels are absent from the template because they arrive from
    config, not because they stopped being printed."""
    html = printed_text(renderer.build_html(TEMPLATE, make_receipt().render_context()))
    for label in jurisdiction("UA")["receipt"]["totals_labels"].values():
        assert label in html


def test_a_tax_row_puts_its_amount_in_the_receipts_amount_column(renderer, rendered):
    """The VAT amounts belong in the same right-hand column as every other amount. While the
    whole row was one string it sat hard against the label, so `111,32` floated mid-line with
    `1 545,70` flush right two lines above — correct and ugly, which is still a defect on every
    image of a registered seller.

    TWO CHECKS, because neither alone is enough. The geometric one says the row's box spans to
    the amount column — but a full-width box would satisfy that however its text were laid out,
    so it cannot see the alignment on its own. The structural one says why the amount lands
    there: the row is a `row` like the totals beside it, and the amount is its own trailing
    element rather than the tail of one string, which is what the flex rule right-aligns.
    """
    total_right = sum(rendered.field_bboxes["total"][i] for i in (0, 2))
    tax_rows = [name for name in rendered.field_bboxes if name.startswith("tax_line_")]
    assert tax_rows, "a registered payer's receipt should carry at least one tax row"

    for name in tax_rows:
        x, _, width, _ = rendered.field_bboxes[name]
        assert x + width == pytest.approx(total_right, abs=1), (
            f"{name} does not reach the amount column"
        )

    html = renderer.build_html(TEMPLATE, make_receipt().render_context())
    row = re.search(
        r'<div class="(?P<cls>[^"]*)" data-field="tax_line_0">(?P<body>.*?)</div>',
        html,
        re.DOTALL,
    )
    assert row, "the tax row should carry the box on the row element"
    assert "row" in row["cls"].split(), "the tax row is not laid out as a row"
    assert len(re.findall(r"<span>", row["body"])) == 2, (
        "label and amount must be separate children, or the flex rule has nothing to push apart"
    )


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        (PAYER, {"seller_vat_number", "seller_tax_code"}),
        (NON_PAYER, {"seller_tax_code"}),
    ],
    ids=["payer", "non_payer"],
)
def test_a_registered_seller_renders_both_identifier_lines(renderer, vendor, expected):
    """REWRITTEN, AND THE OLD ASSERTION WAS FALSE. This test used to require EXACTLY ONE
    identifier line, on a published table of the form that lists the VAT-payer number and the
    identification code as rows 4 and 5 with alternative examples. 👁 Real ПРРО output prints
    both on a registered company's receipt, so the old assertion forbade the very document the
    generator must produce.

    A payer carries one line MORE than a non-payer; a non-payer still carries its ІД.
    """
    context = make_receipt(vendor=vendor).render_context()
    fields = marked_fields(renderer.build_html(TEMPLATE, context))

    assert fields & {"seller_vat_number", "seller_tax_code"} == expected


@pytest.mark.parametrize(
    ("vendor", "prefix", "field"),
    [
        (PAYER, "ПН", "seller_vat_number"),
        (PAYER, "ІД", "seller_tax_code"),
        (NON_PAYER, "ІД", "seller_tax_code"),
    ],
    ids=["payer_pn", "payer_id", "non_payer_id"],
)
def test_the_prefix_printed_beside_each_identifier_is_the_one_the_form_prescribes(
    renderer, vendor, prefix, field
):
    html = renderer.build_html(TEMPLATE, make_receipt(vendor=vendor).render_context())
    assert f'{prefix} <span data-field="{field}"' in html


def test_a_registered_sole_trader_prints_the_same_number_under_both_prefixes(renderer):
    """👁 A sole trader's ПН is its РНОКПП, so the two lines carry identical digits. Worth its
    own rendering test: the page must show the value twice under different prefixes, and code
    that deduplicated identical identifier values would silently drop a requisite."""
    receipt = make_receipt(vendor=PAYER_SOLE_TRADER)
    html = renderer.build_html(TEMPLATE, receipt.render_context())
    number = receipt.seller.tax_code

    assert receipt.seller.vat_number == number
    assert f'ПН <span data-field="seller_vat_number">{number}</span>' in html
    assert f'ІД <span data-field="seller_tax_code">{number}</span>' in html


def test_a_non_payers_page_carries_no_vat_block_at_all(renderer):
    """👁 The observed line ends with the amount: no letter after it, no `ПДВ …%` summary
    row, and no "Без ПДВ" either. The last is worth asserting separately — it is permitted
    in writing and appears on no open sample, so printing it would be a plausible-looking
    invention."""
    html = renderer.build_html(TEMPLATE, make_receipt(vendor=NON_PAYER).render_context())
    fields = marked_fields(html)

    assert not [name for name in fields if name.endswith("_vat_letter")]
    assert not [name for name in fields if name.startswith("tax_line_")]
    assert "ПДВ" not in printed_text(html)
    assert "Без ПДВ" not in printed_text(html)


def test_a_payers_page_does_carry_one(renderer):
    """The positive half. Without it the test above passes on a template that prints no VAT
    block for anyone."""
    html = renderer.build_html(TEMPLATE, make_receipt(vendor=PAYER).render_context())
    fields = marked_fields(html)

    assert "item_0_vat_letter" in fields
    assert "tax_line_0" in fields
    assert "ПДВ" in printed_text(html)


def test_a_non_payers_boxes_lack_exactly_what_is_not_on_the_document(
    renderer, rendered, rendered_non_payer
):
    """The bbox invariant, and the distinction it turns on: a key is absent because the
    ELEMENT DOES NOT EXIST on this document, never because a box was lost.

    Those two are told apart by comparing each document's boxes against ITS OWN rendered
    markup — the template is the authority on what that document contains. The delta between
    the two documents is then checked against the model rather than against whatever was
    produced: a non-payer loses the ПН line, every per-line letter and every tax row, and gains
    NOTHING, because its ІД line is on the payer's document too.

    That last clause is the part this round corrected. While the two identifier lines were
    modelled as alternatives the non-payer was expected to gain `seller_tax_code`, and asserting
    that gain would now pass only on a document no real seller issues.
    """
    payer_fields = marked_fields(renderer.build_html(TEMPLATE, make_receipt().render_context()))
    non_payer_fields = marked_fields(
        renderer.build_html(TEMPLATE, make_receipt(vendor=NON_PAYER).render_context())
    )

    # Nothing lost on either document: every marked field of the page has a box, and no box
    # exists for a field the page does not carry.
    assert set(rendered.field_bboxes) == payer_fields
    assert set(rendered_non_payer.field_bboxes) == non_payer_fields

    vat_fields = {
        name for name in payer_fields
        if name.endswith("_vat_letter") or name.startswith("tax_line_")
    }
    assert vat_fields, "the payer's page carries no VAT fields, so the delta proves nothing"

    assert payer_fields - non_payer_fields == {"seller_vat_number"} | vat_fields
    assert non_payer_fields - payer_fields == set(), (
        "a non-payer's page carries no field the payer's does not — its ІД line is on both"
    )
    assert "seller_tax_code" in payer_fields & non_payer_fields


# ------------------------------------------- the fiscal foot of the receipt --


@pytest.mark.parametrize("slug", FISCAL_SLUGS)
def test_the_maker_name_is_printed_immediately_after_the_fiscal_title(renderer, tmp_path, slug):
    """📄 Line 35 of the published form is ONE requisite — the wording «ФІСКАЛЬНИЙ ЧЕК»
    together with the name or logo of the maker — and 👁 11 of 11 open receipts print such a
    name directly after the wording, which makes it the best-evidenced layout fact available.
    The title was printed alone until this version.

    IMMEDIATELY, asserted as "nothing between them": the maker's box starts below the title's
    and no other field of the document begins in the gap. Checking only that it comes after
    would pass with the QR, the mode marker and the thanks line wedged in between.
    """
    receipt = make_receipt(registrar=registrar_of(slug))
    result = renderer.render(slug, receipt.render_context(), tmp_path / f"{slug}.png")

    title_top = result.field_bboxes["title"][1]
    maker_top = result.field_bboxes["provider_name"][1]
    assert maker_top > title_top, "the maker's name is not below the fiscal wording"

    between = [
        name
        for name, (_, top, _, _) in result.field_bboxes.items()
        if name not in ("title", "provider_name") and title_top < top < maker_top
    ]
    assert not between, f"{between} sit between the fiscal wording and the maker's name"


def test_only_the_hardware_archetype_puts_a_factory_serial_on_the_page(renderer):
    """The rendered half of ⚠️ «ЗН» and «ФН» are not a pair, asserted on the fields the page
    actually carries rather than on the builder's own attributes.

    The delta is checked in BOTH directions and against the model rather than against whatever
    came out: the hardware page gains «ЗН» and loses the online marker, and nothing else about
    the two documents differs. A page that simply printed every line for everyone would satisfy
    a one-directional check.
    """
    prro = marked_fields(
        renderer.build_html("ua_prro_receipt", make_receipt().render_context())
    )
    rro = marked_fields(
        renderer.build_html("ua_rro_receipt", make_receipt(registrar="rro").render_context())
    )

    assert rro - prro == {"device_serial"}
    assert prro - rro == {"mode_marker"}


def test_the_two_prro_archetypes_carry_the_same_requisites(renderer):
    """The 58 mm variant differs in the PAPER and in nothing else. Worth pinning, because the
    two share a body: a conditional added for one width would silently drop a requisite from
    that archetype's every image while the other stayed correct."""
    context = make_receipt().render_context()
    wide = marked_fields(renderer.build_html("ua_prro_receipt", context))
    narrow = marked_fields(renderer.build_html("ua_prro_receipt_58mm", context))

    assert wide == narrow
    assert declared_width_px("ua_prro_receipt_58mm") < declared_width_px("ua_prro_receipt"), (
        "the narrow archetype is not narrower, so it is not the archetype it claims to be"
    )


# ---------------------------------------------------- fonts and determinism --


def test_every_font_is_loaded_from_the_repository(renderer):
    """A font resolved from the host system renders different pixels on a different
    machine, which would quietly break reproducibility while everything still looked
    right locally."""
    html = renderer.build_html(TEMPLATE, make_receipt().render_context())
    faces = re.findall(r'font-family:\s*"([^"]+)";\s*src:\s*url\("([^"]+)"\)', html)

    assert {family for family, _ in faces} == set(FONT_FILES)
    for family, url in faces:
        assert url.startswith("file://"), f"{family} is not loaded from a file"
        path = Path(url2pathname(urlparse(url).path))
        assert path.is_file(), f"{family} points at {path}, which does not exist"
        assert FONTS_DIR in path.parents, f"{family} is loaded from outside fonts/"


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_stylesheet_falls_back_to_noto(renderer, slug):
    """The house rule from fonts/README.md: Roboto has no ₴ glyph, so a Noto face must
    always be reachable for Chromium to substitute from.

    Asserted on the BUILT PAGE and for every archetype, rather than on one `<slug>.css`. The
    font stack now lives in a shared stylesheet that each archetype's own file is layered over,
    so reading one file could report the rule kept while an archetype that overrode the stack
    shipped without a fallback. What has to hold is that the fallback reaches the page.

    IT ASSERTS ON THE `font-family` DECLARATIONS, not on the page text, and two false passes are
    the reason. Searching the raw page for the name is green however the stack is written: each
    vendored face is DECLARED with `font-family: "Noto Sans"` in an `@font-face` block, and the
    shared stylesheet's own comment explains the rule in the same words. Both survived stripping
    the other. Declaring a face makes it available; only the stack makes it reachable, so the
    stack is what has to be read — a stack being the declarations that list more than one family.

    EVERY stack, not merely one of them, and that too was a false pass. The shared stylesheet
    always supplies a correct stack, so "some stack on the page names Noto" stayed green when an
    archetype's own `<slug>.css` overrode `.receipt`'s font with a Noto-less one — the very case
    the per-archetype parametrization is here for, since the override wins the cascade. Any stack
    a document renders through has to be able to substitute ₴.
    """
    html = renderer.build_html(slug, context_for(slug))
    body = re.sub(r"/\*.*?\*/", "", html, flags=re.DOTALL)
    body = re.sub(r"@font-face\s*\{[^}]*\}", "", body, flags=re.DOTALL)
    stacks = [
        value for value in re.findall(r"font-family:\s*([^;{}]+);", body) if "," in value
    ]
    assert stacks, "the page declares no font stack at all"
    without = [value.strip() for value in stacks if '"Noto Sans"' not in value]
    assert not without, f"these font stacks cannot substitute ₴: {without}"


def test_qr_encoding_is_deterministic():
    """segno chooses its mask by evaluating the symbol rather than at random."""
    payload = "https://cabinet.tax.gov.ua/cashregs/check?id=Kht4lxDyk0r"
    assert qr_svg(payload) == qr_svg(payload)
    assert qr_svg(payload) != qr_svg(payload + "x")


def test_the_rendered_qr_carries_the_fiscal_payload(renderer):
    receipt = make_receipt()
    html = renderer.build_html(TEMPLATE, receipt.render_context())
    assert qr_svg(receipt.qr_payload) in html


# ------------------------------------------------------------------ safety --


def test_a_missing_context_value_is_an_error_not_a_blank(renderer, tmp_path):
    """StrictUndefined. A field rendered as an empty string would produce a document
    missing a requisite, labelled as though it were there."""
    context = make_receipt().render_context()
    del context["total"]

    with pytest.raises(UndefinedError):
        renderer.render(TEMPLATE, context, tmp_path / "broken.png")


def test_rendering_outside_a_context_manager_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        Renderer().render(TEMPLATE, make_receipt().render_context(), tmp_path / "x.png")


def test_html_escapes_content(renderer):
    """Line-item names come from configuration, and a stray angle bracket in a name must
    print as text rather than become markup."""
    receipt = make_receipt()
    context = receipt.render_context()
    context["items"][0]["name"] = "<script>alert(1)</script>"

    html = renderer.build_html(TEMPLATE, context)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_amounts_render_with_the_receipt_decimal_separator(renderer):
    receipt = make_receipt()
    html = renderer.build_html(TEMPLATE, receipt.render_context())
    printed = f"{receipt.total:.2f}".replace(".", receipt.decimal_separator)
    # The thousands separator is a no-break space, so compare only the tail.
    assert printed[-6:] in html


def test_ground_truth_carries_the_rendered_boxes(rendered):
    """The join between the two halves of this step: what the renderer measured is what
    the label file states."""
    truth = make_receipt().ground_truth(
        doc_id="p001_c1_d1",
        source_file=rendered.image_path.name,
        capture="screenshot",
        field_bboxes=rendered.field_bboxes,
    )
    assert truth.field_bboxes["total"] == rendered.field_bboxes["total"]
    assert truth.amount == Decimal(str(make_receipt().total))
