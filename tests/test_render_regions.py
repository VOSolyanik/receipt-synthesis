"""`[data-region]` collection: the geometry a later multi-document file needs.

A SEPARATE MODULE FROM `test_renderer.py`, and deliberately so: this file needs its own
`Renderer`, pointed at a fixture template directory rather than `templates/`, and
Playwright's sync API refuses to run two instances in one process at once — `renderer` in
`test_renderer.py` is a module-scoped fixture that stays alive for that module's whole run,
so a second one has to live in a module of its own (see the note on this in
test_reference_text.py, next to its own nested-`Renderer()` warning).

The fixture templates below are not under `templates/` — that directory is another task's,
and this one only proves the renderer's collection pass.
"""

from __future__ import annotations

import pytest

from receipt_synth.renderer import Renderer

# A page carrying TWO documents, each with fields nested inside its `[data-region]` box — the
# shape a multi-document file will have once a later task starts building one. `data-document`
# still marks the page root, one level above both regions.
_REGIONS_HTML = """
<div data-document>
  <div data-region="doc_a" style="padding: 12px; margin-bottom: 20px;">
    <div data-field="a_name">{{ name_a }}</div>
    <div data-field="a_amount">{{ amount_a }}</div>
  </div>
  <div data-region="doc_b" style="padding: 12px;">
    <div data-field="b_name">{{ name_b }}</div>
  </div>
</div>
"""

_REGIONS_CSS = """
body { margin: 0; font-family: sans-serif; font-size: 14px; }
[data-region] { display: block; width: 200px; }
"""

# The same shape, except both documents claim the same region key — the case `_region_bboxes`
# must refuse rather than resolve by keeping whichever element it saw last.
_DUPLICATE_REGION_HTML = """
<div data-document>
  <div data-region="dup"><div data-field="x">1</div></div>
  <div data-region="dup"><div data-field="y">2</div></div>
</div>
"""


@pytest.fixture(scope="module")
def fixture_templates_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("region_fixture_templates")
    (directory / "fixture_regions.html").write_text(_REGIONS_HTML, encoding="utf-8")
    (directory / "fixture_regions.css").write_text(_REGIONS_CSS, encoding="utf-8")
    (directory / "fixture_duplicate_region.html").write_text(
        _DUPLICATE_REGION_HTML, encoding="utf-8"
    )
    (directory / "fixture_duplicate_region.css").write_text(_REGIONS_CSS, encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def renderer(fixture_templates_dir):
    with Renderer(templates_dir=fixture_templates_dir) as instance:
        yield instance


def region_context(**overrides):
    context = {"name_a": "Alpha", "amount_a": "10.00", "name_b": "Beta", "qr_payload": None}
    return context | overrides


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    output = tmp_path_factory.mktemp("regions") / "regions.png"
    return renderer.render("fixture_regions", region_context(), output)


def test_every_data_region_marker_gets_a_box(rendered):
    assert set(rendered.region_bboxes) == {"doc_a", "doc_b"}


def test_region_boxes_are_whole_pixels_and_lie_inside_the_image(rendered):
    for name, (x, y, width, height) in rendered.region_bboxes.items():
        assert all(float(value).is_integer() for value in (x, y, width, height)), name
        assert width > 0 and height > 0, f"{name} has no area"
        assert x >= 0 and y >= 0, f"{name} starts outside the image"
        assert x + width <= rendered.width, f"{name} runs past the right edge"
        assert y + height <= rendered.height, f"{name} runs past the bottom edge"


def test_each_field_lies_inside_its_own_region(rendered):
    ax, ay, aw, ah = rendered.region_bboxes["doc_a"]
    for name in ("a_name", "a_amount"):
        x, y, width, height = rendered.field_bboxes[name]
        assert x >= ax and y >= ay, f"{name} starts outside doc_a"
        assert x + width <= ax + aw and y + height <= ay + ah, f"{name} ends outside doc_a"

    bx, by, bw, bh = rendered.region_bboxes["doc_b"]
    x, y, width, height = rendered.field_bboxes["b_name"]
    assert x >= bx and y >= by, "b_name starts outside doc_b"
    assert x + width <= bx + bw and y + height <= by + bh, "b_name ends outside doc_b"


def test_a_duplicate_data_region_value_fails_loudly_rather_than_last_write_wins(
    renderer, tmp_path
):
    with pytest.raises(ValueError, match="dup"):
        renderer.render("fixture_duplicate_region", region_context(), tmp_path / "dup.png")
