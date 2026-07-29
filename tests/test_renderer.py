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
from jinja2 import UndefinedError
from PIL import Image

from receipt_synth.content_builder import build_prro_receipt
from receipt_synth.renderer import FONT_FILES, FONTS_DIR, TEMPLATES_DIR, Renderer, qr_svg

TEMPLATE = "ua_prro_receipt"


def make_receipt(seed: int = 20260803):
    return build_prro_receipt(
        random.Random(seed),
        category_id="vitamins_nutrition",
        issued_at=datetime(2026, 8, 3, 14, 22, 51),
        vendor={"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy"},
        address="м. Київ, вул. Хрещатик, 22",
    )


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
    the twenty-three still to be written."""
    source = re.sub(r"\{#.*?#\}", "", template_path.read_text(encoding="utf-8"), flags=re.DOTALL)
    assert source.count("data-document") == 1, "expected exactly one document root"


def test_the_image_width_is_the_declared_document_width(rendered):
    """Positive counterpart to the guard below.

    Together with the marker test above this closes the chain: the template declares
    exactly one document root, the stylesheet gives that root a width, and the image is
    that wide — so the frame came from the document rather than from the browser.
    """
    stylesheet = (TEMPLATES_DIR / f"{TEMPLATE}.css").read_text(encoding="utf-8")
    declared = re.search(r"\.receipt\s*\{[^}]*?\bwidth:\s*(\d+)px", stylesheet, re.DOTALL)

    assert declared, "the stylesheet should give the document root an explicit width"
    assert rendered.width == int(declared.group(1))


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
    last_item = max(
        box[1] for name, box in rendered.field_bboxes.items() if name.endswith("_name")
    )
    assert rendered.field_bboxes["total"][1] > last_item


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


def test_the_stylesheet_falls_back_to_noto(renderer):
    """The house rule from fonts/README.md: Roboto has no ₴ glyph, so a Noto face must
    always be reachable for Chromium to substitute from."""
    stylesheet = (TEMPLATES_DIR / f"{TEMPLATE}.css").read_text(encoding="utf-8")
    assert '"Noto Sans"' in stylesheet


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
