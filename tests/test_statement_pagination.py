"""A bank statement that runs onto a second sheet — and the per-page ground truth of it.

🔴 why this file exists. Until this archetype paginated, one page was one document throughout the
corpus, by construction: every template rendered a single sheet, so a step that splits a submitted
file into logical documents scored a perfect result on data whose ideal it could not fail. A
measurement whose answer is guaranteed by the construction of the dataset measures the
construction. A statement of forty operations is the counter-example, and this file is what says
the counter-example is well-formed.

What is asserted, and each of the four is a different way the pagination could be right on the page
and wrong in the label:

  * the rows are split — every operation appears exactly once, across the sheets, in order;
  * the printed «Сторінка N з M» agrees with `page_count`. It is the human-visible half of the
    segmentation signal, and a corpus whose footer said "з 2" on a one-page label would teach a
    reader the opposite of the truth;
  * the regions frame the sheets — inside the image, disjoint, in reading order;
  * the regions survive degradation by the same transform as the fields, which is the failure that
    would be invisible everywhere else. See the known-answer test at the foot of this file.

⛔ nothing here is an observed-anatomy claim about a second page. Only the first page of a real
statement was ever seen. What a continuation sheet carries here — the repeated column headings, the
page footer — is general layout of a paginated table and is marked as such in the template, in
config/fiscal-rules.yaml and in config/generation.yaml.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("albumentations")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402

from receipt_synth.assembler import (  # noqa: E402
    _CONTENT_BBOX_KEY,
    take_page_regions,
    tracked_boxes,
)
from receipt_synth.config import (  # noqa: E402
    bank_statement_count_range,
    jurisdiction,
)
from receipt_synth.content_builder import (  # noqa: E402
    build_bank_statement,
    draw_party_identity,
    draw_statement_pages,
    generate_rnokpp,
    resolve_vendor,
)
from receipt_synth.degrader import carry_boxes  # noqa: E402
from receipt_synth.renderer import Renderer  # noqa: E402
from receipt_synth.schemas import Capture  # noqa: E402

SLUG = "ua_bank_statement"
BLOCK = jurisdiction("UA")["bank_statement"]
PAGINATION = BLOCK["pagination"]

VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
HOLDER_NAME = "Ковальчук Олена Петрівна"
HOLDER_CODE = generate_rnokpp(random.Random(3))
WHEN = datetime(2026, 5, 12, 14, 33)

# The seed the two-page assertions below are taken at. One seed rather than a sweep wherever the
# claim is about geometry: a rendered sheet costs a browser page, and the properties asserted on it
# are structural rather than statistical.
TWO_PAGE_SEED = 7


def make_statement(seed: int = TWO_PAGE_SEED, *, pages: int = 2):
    rng = random.Random(seed)
    resolved = resolve_vendor(rng, VENDOR, "UA")
    return build_bank_statement(
        rng,
        issued_at=WHEN,
        vendor=resolved,
        identity=draw_party_identity(rng, resolved, "UA"),
        payer_name=HOLDER_NAME,
        payer_tax_id=HOLDER_CODE,
        pages=pages,
    )


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    """One two-page statement, rendered once, with the object it was built from beside it."""
    statement = make_statement()
    output = tmp_path_factory.mktemp("paginated") / "statement.png"
    return statement, renderer.render(SLUG, statement.render_context(), output)


def contains(outer, inner) -> bool:
    """Whether `inner` lies wholly inside `outer`, both in the COCO convention."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ix + iw <= ox + ow and iy + ih <= oy + oh


# ------------------------------------------------------- the split, in the builder --


def test_a_paginated_statement_puts_every_operation_on_exactly_one_sheet():
    """The rows are partitioned, not copied and not dropped. A sheet that repeated the last row of
    the previous one would print a document whose turnover totals — derived from `rows` — no longer
    reconcile with what a reader can count on the page."""
    statement = make_statement()

    assert statement.page_count == 2
    assert [row for sheet in statement.page_sheets for row in sheet] == list(statement.rows)
    assert all(sheet for sheet in statement.page_sheets)


def test_the_first_sheet_is_filled_before_the_second_is_started():
    """A page holds what fits on it, so the split is not free: the first sheet carries its declared
    capacity and the remainder falls to the next. A split that balanced the sheets evenly would be
    a layout no printer produces."""
    statement = make_statement()

    assert len(statement.page_sheets[0]) == PAGINATION["rows_first_page"]
    assert len(statement.page_sheets[1]) == len(statement.rows) - PAGINATION["rows_first_page"]
    assert len(statement.page_sheets[1]) <= PAGINATION["rows_continuation_page"]


def test_a_single_page_statement_is_one_sheet_and_reports_one_page():
    """The overwhelming majority path, unchanged. `page_count == 1` is what keeps `page_regions`
    absent from the label — see `DocGroundTruth._page_count_and_regions_agree`."""
    statement = make_statement(pages=1)

    assert statement.page_count == 1
    assert len(statement.page_sheets) == 1
    assert len(statement.rows) <= bank_statement_count_range("row_count")[1]


def test_the_row_count_of_a_two_page_statement_is_what_the_configuration_declares():
    """Its own range, and not `row_count_range` widened: the single-page bound is tuned against the
    sheet and a range that served both would either underfill the second sheet or overflow the
    first."""
    low, high = bank_statement_count_range("two_page_row_count")

    for seed in range(12):
        statement = make_statement(seed)
        assert low <= len(statement.rows) <= high


# ------------------------------------------------------------- the share parameter --


def test_the_share_of_nought_paginates_nothing(monkeypatch):
    """The knob has to be able to turn the case off completely — a run comparing itself against an
    earlier one needs a way to reproduce the earlier composition exactly."""
    monkeypatch.setattr(
        "receipt_synth.content_builder.bank_statement_share", lambda name: 0.0
    )
    assert {draw_statement_pages(random.Random(seed)) for seed in range(40)} == {1}


def test_the_share_of_one_paginates_everything(monkeypatch):
    """And the other end, which is what makes the parameter load-bearing rather than a constant
    that happens to be small."""
    monkeypatch.setattr(
        "receipt_synth.content_builder.bank_statement_share", lambda name: 1.0
    )
    assert {draw_statement_pages(random.Random(seed)) for seed in range(40)} == {2}


def test_the_declared_share_leaves_the_single_page_statement_the_strong_majority():
    """The composition decision itself, asserted rather than left to the file. Every figure already
    measured on this class was measured on one-page statements, and a share that made the paginated
    form common would change what those numbers describe without anybody editing them."""
    drawn = [draw_statement_pages(random.Random(seed)) for seed in range(400)]

    assert drawn.count(1) / len(drawn) > 0.75


# ------------------------------------------------------ the sheets, on the rendered page --


def test_each_sheet_is_marked_as_its_own_region_in_reading_order(rendered):
    """One `data-region` per sheet, named `page_N`. The renderer collects them in the same pass as
    the fields, so the two are in one coordinate system by construction."""
    statement, result = rendered

    assert sorted(result.region_bboxes) == [f"page_{n}" for n in range(1, statement.page_count + 1)]


def test_the_page_regions_frame_the_sheets_and_do_not_overlap(rendered):
    """Inside the image, disjoint, and each one strictly below the last. ⚠️ "Strictly below" is the
    assertion that would fail first if a sheet ever stopped being a full-width band — it is what
    makes the regions a page split rather than an arbitrary partition of the picture."""
    statement, result = rendered
    regions = [result.region_bboxes[f"page_{n}"] for n in range(1, statement.page_count + 1)]

    for region in regions:
        assert contains((0, 0, result.width, result.height), region)
    for above, below in zip(regions, regions[1:], strict=False):
        assert above[1] + above[3] <= below[1], "a sheet begins before the one above it ends"


def test_every_sheet_of_a_paginated_statement_is_the_same_A4_paper(renderer, tmp_path):
    """🔴 two sheets of one document are one paper. `bank_statement.page` declares A4 landscape, and
    a first sheet that ran a few pixels taller than its continuation would be a document printed on
    two different sizes — which is what happened at a capacity of 25: the page footer's line costs
    a row, and `pagination.rows_first_page` is one fewer than `rows_single_sheet` for that reason
    alone. Swept over the row counts that actually occur, because the tallest one is where a sheet
    overflows and a single render would miss it."""
    height = round(Decimal(BLOCK["page"]["height_mm"]) * 96 / Decimal("25.4"))
    widest = 0
    # ⚠️ A list of seeds is a measurement of one draw stream and goes stale whenever the stream
    # moves — the same warning `test_every_row_of_every_statement_fits_the_declared_sheet` carries.
    # The assertion under the loop is what says so rather than the test quietly exercising 43.
    for seed in (0, 7, 12, 16, 19):
        statement = make_statement(seed)
        result = renderer.render(SLUG, statement.render_context(), tmp_path / f"{seed}.png")
        widest = max(widest, len(statement.rows))
        for page in range(1, statement.page_count + 1):
            region = result.region_bboxes[f"page_{page}"]
            assert region[3] == height, f"seed {seed}, sheet {page}, {len(statement.rows)} rows"
    assert widest == bank_statement_count_range("two_page_row_count")[1], (
        "no seed in this set produced the maximum row count, so the bound was not exercised"
    )


def test_the_regions_cover_every_row_of_the_document(rendered):
    """The regions are a partition of the content and not two boxes that merely happen to be in the
    right places: every row's own `operation_<i>` box falls inside exactly one of them."""
    statement, result = rendered
    regions = [result.region_bboxes[f"page_{n}"] for n in range(1, statement.page_count + 1)]

    for index in range(len(statement.rows)):
        row_box = result.field_bboxes[f"operation_{index}"]
        assert sum(contains(region, row_box) for region in regions) == 1


def test_the_printed_page_footer_agrees_with_the_page_count(rendered):
    """🔴 the human-visible half of the segmentation signal, and the one that can drift silently.
    «Сторінка N з M» is read back out of the page's own text — the renderer's `reference_text`,
    taken from the layout engine — so this compares what a reader sees against what the label
    says, not one part of the builder against another."""
    statement, result = rendered
    folios = re.findall(r"Сторінка (\d+) з (\d+)", result.reference_text)

    assert [page for page, _ in folios] == [str(n) for n in range(1, statement.page_count + 1)]
    assert {total for _, total in folios} == {str(statement.page_count)}


def test_a_one_page_statement_prints_no_page_footer(renderer, tmp_path):
    """⚠️ and the single-page look is settled. A footer reading «Сторінка 1 з 1» on the majority of
    this class's images would be a visible change to every figure already measured on it, in
    exchange for a count a reader can already see is one."""
    statement = make_statement(pages=1)

    result = renderer.render(SLUG, statement.render_context(), tmp_path / "one.png")

    assert "Сторінка" not in result.reference_text
    assert result.region_bboxes == {"page_1": pytest.approx(result.region_bboxes["page_1"])}


def test_the_labelled_row_is_not_pinned_to_the_first_sheet():
    """🔴 the «PAGE = DOCUMENT» trap, one level down. A paginated corpus whose answer was always on
    sheet one would replace the guarantee this archetype exists to break with a smaller one of the
    same kind: a reader could skip every continuation page and lose nothing. 👁 Operations print in
    the order they happened, so the labelled row lands where its timestamp puts it — this asserts
    that the ordering actually reaches the second sheet."""
    landed = set()
    for seed in range(24):
        statement = make_statement(seed)
        first = len(statement.page_sheets[0])
        landed.add(0 if statement.relevant_index < first else 1)

    assert landed == {0, 1}


def test_a_row_on_the_second_sheet_carries_its_boxes_on_the_second_sheet(renderer, tmp_path):
    """The geometric half of the claim above, and the one that matters to a consumer: when the
    answer is on sheet two, the labelled boxes are inside sheet two's region. A label that pointed
    at the right values through boxes on the wrong page would read as correct everywhere except in
    the picture."""
    statement = next(
        candidate
        for seed in range(24)
        if (candidate := make_statement(seed)).relevant_index >= len(candidate.page_sheets[0])
    )

    result = renderer.render(SLUG, statement.render_context(), tmp_path / "second.png")

    second = result.region_bboxes["page_2"]
    for field in ("amount", "date", "counterparty", "payment_purpose", "relevant_transaction"):
        assert contains(second, result.field_bboxes[field]), field


# ------------------------------------------------------------------------ the label --


def test_the_label_reports_the_pages_and_where_they_are(rendered):
    """`page_count` from the document, `page_regions` from the renderer, in reading order — and
    `file_region` still `None`, because this file holds one document. Task 3 is what puts two."""
    statement, result = rendered
    regions = [result.region_bboxes[f"page_{n}"] for n in range(1, statement.page_count + 1)]

    label = statement.ground_truth(
        doc_id="c1_d1",
        source_file="c1_d1.png",
        capture=Capture.SCREENSHOT,
        field_bboxes=result.field_bboxes,
        page_regions=regions,
    )

    assert label.page_count == statement.page_count
    assert label.page_regions == [tuple(region) for region in regions]
    assert label.file_region is None


def test_a_one_page_statement_reports_no_regions_at_all():
    """The other side of `DocGroundTruth._page_count_and_regions_agree`: a list of one region would
    say what `file_region` already says, so the single-page label carries none."""
    label = make_statement(pages=1).ground_truth(
        doc_id="c1_d1", source_file="c1_d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )

    assert (label.page_count, label.page_regions) == (1, None)


# --------------------------------------------- the regions through the degrader --

# The known-answer fixture of tests/test_bbox_gate.py, reused deliberately: this file's claim is
# that a region goes through the same arithmetic as a field, so it has to be measured with the same
# rectangle and against the same hand-computed answer.
WIDTH, HEIGHT = 200, 100
MARKER = (20.0, 30.0, 40.0, 10.0)


def fixture_image() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x, y, w, h = (int(value) for value in MARKER)
    image[y : y + h, x : x + w] = 255
    return image


class _Rendered:
    """The three geometry channels of a render, without a browser.

    A stub rather than a real page because what is under test is the assembler's merge and split,
    and a rendered statement would put a hundred boxes between the input and the assertion.
    """

    def __init__(self, fields, content_bbox, region_bboxes):
        self.field_bboxes = fields
        self.content_bbox = content_bbox
        self.region_bboxes = region_bboxes


def unpack(moved: dict, *, page_count: int):
    """The assembler's own two steps on the far side of the degrader, in its own order: the content
    extent is popped out, then the page regions, and what is left is the label's `field_bboxes`."""
    boxes = dict(moved)
    content_bbox = boxes.pop(_CONTENT_BBOX_KEY)
    return boxes, content_bbox, take_page_regions(boxes, page_count=page_count)


def test_a_page_region_moves_exactly_as_a_field_box_of_the_same_shape_does():
    """🔴 the failure this test exists for is invisible everywhere else. A region transformed by a
    second, independent draw would land somewhere plausible and be wrong, and no metric in the
    dataset would point at it — a desynchronized region reads downstream as a bad segmenter.

    Hand-computed, from tests/test_bbox_gate.py's own geometry: a horizontal flip of a 200-wide
    image maps column c to 199 - c, so the marker's columns 20 … 59 become 140 … 179 —

        (20, 30, 40, 10)  ->  (140, 30, 40, 10)

    The field, the content extent and the page region all start at the marker, so all three must
    end there. Equality between them is the real assertion; the hand-computed value is what stops
    all three from being equally wrong.
    """
    clean = _Rendered(
        fields={"amount": MARKER},
        content_bbox=MARKER,
        region_bboxes={"page_1": MARKER, "page_2": MARKER},
    )

    moved_image, moved = carry_boxes(
        [A.HorizontalFlip(p=1.0)], fixture_image(), tracked_boxes(clean), seed=0
    )
    fields, content_bbox, page_regions = unpack(moved, page_count=2)

    assert fields == {"amount": (140.0, 30.0, 40.0, 10.0)}
    assert content_bbox == (140.0, 30.0, 40.0, 10.0)
    assert page_regions == [(140.0, 30.0, 40.0, 10.0)] * 2
    assert moved_image.shape[:2] == (HEIGHT, WIDTH)


def test_the_regions_ride_the_same_translation_as_the_fields():
    """A second transform of a different kind, for the reason the gate gives: a mirror is its own
    inverse, so a carrier that applied the transform twice would still pass the test above. A
    translation of +12 columns and -8 rows applied twice lands at 44, not 32."""
    shift = A.Affine(
        translate_px={"x": 12, "y": -8},
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=1.0,
    )
    clean = _Rendered(
        fields={"amount": MARKER},
        content_bbox=MARKER,
        region_bboxes={"page_1": MARKER, "page_2": MARKER},
    )

    _, moved = carry_boxes([shift], fixture_image(), tracked_boxes(clean), seed=0)
    fields, _, page_regions = unpack(moved, page_count=2)

    assert page_regions == [(32.0, 22.0, 40.0, 10.0)] * 2
    assert page_regions[0] == fields["amount"]


def test_the_reserved_keys_never_reach_a_label(rendered):
    """⚠️ A one-sheet document still marks its sheet, so the carried boxes always hold a region the
    single-page label has no room for. A reserved key left behind would reach a consumer as a
    `data-field` named `__page_region_1__` — a field no template prints and no requirement names."""
    statement, result = rendered
    one_sheet = _Rendered({"amount": MARKER}, MARKER, {"page_1": MARKER})

    paginated, _, _ = unpack(tracked_boxes(result), page_count=statement.page_count)
    single, _, regions = unpack(tracked_boxes(one_sheet), page_count=1)

    assert not [name for name in paginated if name.startswith("__")]
    assert single == {"amount": MARKER} and regions is None


def test_a_template_may_not_claim_a_field_name_the_carrier_reserves():
    """The reserved keys are checked rather than assumed unique. `data-field` names are printed
    fields and none begins with two underscores — but "none does today" is what a refusal is for."""
    clean = _Rendered(
        fields={"__page_region_1__": MARKER}, content_bbox=MARKER, region_bboxes={"page_1": MARKER}
    )

    with pytest.raises(ValueError, match="reserve"):
        tracked_boxes(clean)


def test_a_statement_whose_regions_are_missing_is_refused_rather_than_labelled():
    """⛔ no silent fallback to one page. A render that lost a `page_N` marker would otherwise
    produce a two-sheet image labelled as one page, which is worse than a crash: the image is
    wrong in the dataset and nothing says so."""
    statement = make_statement()

    with pytest.raises(ValueError, match="page_regions"):
        statement.ground_truth(
            doc_id="c1_d1",
            source_file="c1_d1.png",
            capture=Capture.SCREENSHOT,
            field_bboxes={},
        )


def test_the_carrier_refuses_a_region_count_it_was_not_told_about():
    """The split is told how many pages to expect, so a mismatch between the render and the label
    fails here instead of producing a short `page_regions` list the schema would then reject with a
    message about the label rather than about its cause."""
    moved = {_CONTENT_BBOX_KEY: MARKER, "__page_region_1__": MARKER}

    with pytest.raises(ValueError, match="__page_region_2__"):
        unpack(moved, page_count=2)


def test_the_amounts_of_a_paginated_statement_still_reconcile():
    """The turnover block is derived from `rows`, and pagination must not touch it: a reader adding
    up two sheets has to arrive at the block printed on the first."""
    statement = make_statement()

    printed = sum(
        (row.amount if row.is_debit else -row.amount for row in statement.rows), Decimal(0)
    )
    assert statement.closing_balance - statement.opening_balance == -printed.quantize(
        Decimal("0.01")
    )
