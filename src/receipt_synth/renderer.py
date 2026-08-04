"""Jinja2 → HTML → Playwright screenshot, with a bounding box per extractable field.

The boxes are read with ``getBoundingClientRect()`` in the same pass as the screenshot,
from the browser's own layout engine. They are not OCR output and carry no detection
error: an element's box is where the renderer put it.

Two details keep them honest:

* the viewport is resized to the full document height before anything is measured, so
  ``getBoundingClientRect()`` (viewport coordinates) and the screenshot (page
  coordinates) are the same coordinate system;
* the device scale factor is 1, so one CSS pixel is one image pixel.

Fonts are loaded from ``fonts/`` by absolute path rather than by family name. A font
resolved from the host system would render different pixels on a different machine, and
the same seed has to produce the same image everywhere.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import segno
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from playwright.sync_api import Page, sync_playwright

from receipt_synth.schemas import BBox

# Intermediate viewport height used only while measuring. Any value works; the final
# viewport is set to the document's own height immediately afterwards.
_PROBE_HEIGHT = 1024

# The page root every template marks with `data-document`, falling back to the whole
# document for a template that has not declared one.
_DOCUMENT_RECT = (
    "(document.querySelector('[data-document]') ?? document.documentElement)"
    ".getBoundingClientRect()"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
FONTS_DIR = REPO_ROOT / "fonts"

# Families the templates may name, and the vendored file each resolves to. Declared here
# rather than in CSS so a template cannot silently fall through to a host font.
FONT_FILES = {
    "Noto Sans Mono": "NotoSansMono[wdth,wght].ttf",
    "Noto Sans": "NotoSans[wdth,wght].ttf",
    "Roboto": "Roboto[wdth,wght].ttf",
    "Roboto Mono": "RobotoMono[wght].ttf",
}


@dataclass(frozen=True)
class RenderedDocument:
    """One rendered page: the image on disk, where every field ended up on it, and what it says.

    `reference_text` and `field_bboxes` COVER DIFFERENT THINGS, and the difference is the whole
    reason both exist. The boxes cover the LABELLED FIELDS; the text covers ALL PRINTED TEXT. A
    document whose every labelled field survived a crop while the footer carrying the fiscal
    wording was lost would look complete measured on the boxes alone.
    """

    image_path: Path
    width: int
    height: int
    field_bboxes: dict[str, BBox]
    # One box per `[data-region]` element, keyed by the attribute's value — EMPTY on every
    # template today, none of which mark one. Collected in the SAME pass as `field_bboxes` (see
    # `_COLLECT_BBOXES`) rather than a second page evaluation, and with the same rounding
    # convention. Nothing draws a multi-document file yet; this is the geometry a later stage
    # needs once a template starts marking more than one document's rectangle on a page.
    region_bboxes: dict[str, BBox]
    # The page's text in reading order, taken from the layout engine BEFORE rasterization — so it
    # is ground truth by construction rather than by annotation. See `_COLLECT_TEXT`.
    reference_text: str
    # Where that text is, as one box. NOT the union of the field boxes and not the page: the extent
    # of the rendered TEXT, which is what a later measurement compares a degraded capture against.
    content_bbox: BBox


def _font_faces() -> list[dict[str, str]]:
    faces = []
    for family, filename in FONT_FILES.items():
        path = FONTS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"vendored font missing: {path}")
        faces.append({"family": family, "url": path.as_uri()})
    return faces


def qr_svg(payload: str, *, error: str = "m") -> str:
    """An inline SVG QR code for a payload.

    Deterministic: segno picks its mask by evaluating the symbol, not at random, so the
    same payload always yields the same modules.
    """
    return segno.make(payload, error=error).svg_inline(border=0, scale=3)


# THE PAGE'S TEXT, IN READING ORDER, AND THE EXTENT OF IT.
#
# `innerText` rather than `textContent`, and the difference is not cosmetic: `textContent` returns
# the source order of every node including ones CSS never paints, while `innerText` is what the
# layout engine decided a reader sees — hidden elements excluded, line boxes reflected as newlines.
# Since this dataset's whole premise is that the label describes the IMAGE, the text has to come
# from the same authority that produced the image.
#
# THE EXTENT IS COMPUTED FROM TEXT RANGES, not from element boxes. An element's box includes its
# padding and can be far larger than the ink inside it — a table cell, a full-width footer div —
# and a measurement of whether the CONTENT survived a crop wants where the characters are. A
# `Range` over a text node reports exactly the rectangles the glyphs occupy.
#
# ⚠️ ITS SCOPE IS TEXT AND ONLY TEXT. A QR code, a stamp and a signature are ink that this box does
# not cover, deliberately: it is the counterpart of `reference_text`, which is also text only, and
# a later measurement must not read it as "everything printed".
_COLLECT_TEXT = """
() => {
  const root = document.querySelector('[data-document]') ?? document.documentElement;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  let node;
  while ((node = walker.nextNode())) {
    if (!node.textContent.trim()) continue;
    const range = document.createRange();
    range.selectNodeContents(node);
    for (const rect of range.getClientRects()) {
      if (rect.width === 0 || rect.height === 0) continue;
      left = Math.min(left, rect.left);
      top = Math.min(top, rect.top);
      right = Math.max(right, rect.right);
      bottom = Math.max(bottom, rect.bottom);
    }
  }
  const empty = !Number.isFinite(left);
  return {
    text: root.innerText,
    bbox: empty ? [0, 0, 0, 0] : [
      Math.floor(left), Math.floor(top),
      Math.ceil(right - left), Math.ceil(bottom - top),
    ],
  };
}
"""

# Reading a rect straight off the layout engine. `x`/`y` are viewport coordinates, which
# equal page coordinates because the renderer never scrolls and sizes the viewport to the
# whole document first. Rounded to whole pixels: a box is an index into an image, and a
# fractional pixel index means nothing to a consumer.
#
# ONE PASS COLLECTS BOTH `[data-field]` AND `[data-region]` BOXES, rather than a second page
# evaluation for the region markers — the same layout, read once. Regions come back as a LIST OF
# PAIRS rather than an object: `Object.fromEntries` on a duplicate key keeps only the last write
# silently, and a duplicate `data-region` value has to fail loudly instead (`Renderer.render`
# checks the list for repeats before turning it into the dict `region_bboxes` holds).
_COLLECT_BBOXES = """
() => {
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];
  };
  return {
    fields: Object.fromEntries(
      [...document.querySelectorAll('[data-field]')].map(el => [el.dataset.field, box(el)])
    ),
    regions: [...document.querySelectorAll('[data-region]')].map(
      el => [el.dataset.region, box(el)]
    ),
  };
}
"""


class Renderer:
    """Renders documents in one browser session.

    A context manager because starting Chromium costs far more than rendering a page;
    a dataset of any size should pay it once.

        with Renderer() as renderer:
            result = renderer.render("ua_prro_receipt", context, out_path)
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR) -> None:
        self._templates_dir = templates_dir
        self._env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
            # Undefined variables raise instead of rendering as an empty string. A field
            # silently missing from a document would be a label pointing at nothing.
            undefined=StrictUndefined,
        )
        self._playwright = None
        self._browser = None

    def __enter__(self) -> Renderer:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._playwright = None

    def build_html(self, template_name: str, context: dict) -> str:
        """The full HTML page for a document, stylesheet and fonts inlined.

        `qr_svg` IS `None` WHERE THE DOCUMENT CARRIES NO QR, and that case is real rather than
        defensive: 👁 only 2 of 8 bank payment confirmations print one. `context["qr_payload"]` is
        still required — a template that shows a QR has to say what is in it, and a missing key
        here would be a document silently losing a requisite — but a payload of `None` means the
        page has no QR block, and the template decides that with its own conditional.
        """
        stylesheet = (self._templates_dir / f"{template_name}.css").read_text(encoding="utf-8")
        template = self._env.get_template(f"{template_name}.html")
        payload = context["qr_payload"]
        return template.render(
            **context,
            stylesheet=stylesheet,
            font_faces=_font_faces(),
            qr_svg=qr_svg(payload) if payload else None,
        )

    def render(self, template_name: str, context: dict, output_path: Path) -> RenderedDocument:
        """Render one document to a PNG and return the bounding box of every field."""
        if self._browser is None:
            raise RuntimeError("Renderer must be used as a context manager")

        html = self.build_html(template_name, context)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page = self._browser.new_page(device_scale_factor=1)
        try:
            # Written to a file rather than injected: the @font-face rules point at
            # file:// URLs, which a page served from about:blank is not allowed to fetch.
            with tempfile.TemporaryDirectory() as tmp:
                page_path = Path(tmp) / f"{template_name}.html"
                page_path.write_text(html, encoding="utf-8")
                page.goto(page_path.as_uri())
                width, height = _fit_viewport_to_content(page)
                collected = page.evaluate(_COLLECT_BBOXES)
                bboxes = {name: tuple(box) for name, box in collected["fields"].items()}
                region_bboxes = _region_bboxes(collected["regions"])
                # Read BEFORE the screenshot, from the same page state. The order matters only in
                # that nothing may change between them; there is no scrolling or animation here, so
                # both describe one layout.
                content = page.evaluate(_COLLECT_TEXT)
                page.screenshot(path=output_path, full_page=True)
        finally:
            page.close()

        return RenderedDocument(
            image_path=output_path,
            width=width,
            height=height,
            field_bboxes=bboxes,
            region_bboxes=region_bboxes,
            reference_text=content["text"],
            content_bbox=tuple(content["bbox"]),
        )


def _region_bboxes(pairs: list[list]) -> dict[str, BBox]:
    """`[data-region]` boxes, keyed by attribute value — refusing a duplicate key rather than
    letting one silently overwrite another.

    Takes the pairs as JavaScript returned them (a list, not an object) precisely so a repeat
    survives to be checked here: an `Object.fromEntries` on the JavaScript side would already
    have collapsed it to whichever element came last, with nothing left to detect.
    """
    names = [name for name, _ in pairs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate data-region value(s): {duplicates} — every data-region must be unique "
            "within one render"
        )
    return {name: tuple(box) for name, box in pairs}


def _fit_viewport_to_content(page: Page) -> tuple[int, int]:
    """Size the viewport to the document element, so that viewport coordinates and image
    coordinates coincide and the image is the document rather than the window around it.

    Width first, then height: narrowing the viewport can reflow the content, so the
    height has to be measured at the final width or a tall document would be cut off.
    """
    page.wait_for_load_state("networkidle")

    # `right` and `bottom` rather than `width` and `height`, so that any offset of the
    # document element from the page origin is inside the frame.
    #
    # Not `scrollHeight`: it never reports less than the viewport, so a document shorter
    # than the probe would come back as the probe height and the image would be padded
    # with blank paper.
    width = page.evaluate(f"() => Math.ceil({_DOCUMENT_RECT}.right)")
    page.set_viewport_size({"width": width, "height": _PROBE_HEIGHT})

    height = page.evaluate(f"() => Math.ceil({_DOCUMENT_RECT}.bottom)")
    page.set_viewport_size({"width": width, "height": height})
    return width, height
