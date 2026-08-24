"""🔴 the known-answer gate: a box must land where the pixels it describes landed.

Every expectation in this file is computed by hand from the geometry of the transform, and
none of it was read off a run. That is the whole value of the file: an expectation copied
from the code's own output cannot detect the defect it is here for.

**Why this is a gate and not a test at the end of the step.** A bounding box that does not
follow its pixels corrupts the ground truth of every image in the dataset and is invisible
in every metric — downstream it reads as poor extraction accuracy rather than as a
coordinate defect, so nothing in a scorecard points back here. With three capture channels
in place a single desync produces three different wrong answers, and separating "the
transform is wrong" from "this channel is wrong" then costs a day. So: one transform, one
rectangle, one answer worked out on paper, before any channel fanned out.

It exercises `degrader.carry_boxes` — the production carrier — with transforms of its own
choosing. A test that assembled its own `A.Compose` would be measuring whether
Albumentations is correct, which is not what is at risk; what is at risk is whether this
repository's box parameters, rounding and rebuild-by-label keep a box on top of its ink.

The fixture. A 200 × 100 black image with one white rectangle:

    columns 20 … 59  (40 wide)      rows 30 … 39  (10 tall)

so its box in the COCO convention the renderer emits — [x, y, width, height] — is

    (20, 30, 40, 10)

Both the box and the ink are checked after every transform, and that pairing is the test.
Checking the box alone would pass for a pipeline that transformed coordinates and left the
image behind; checking the ink alone would pass for one that did the reverse.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("albumentations")

import albumentations as A  # noqa: E402

from receipt_synth.degrader import carry_boxes, clipped_edges  # noqa: E402

WIDTH, HEIGHT = 200, 100
MARKER = (20.0, 30.0, 40.0, 10.0)


def fixture_image() -> np.ndarray:
    """Black, with the marker rectangle painted white at exactly `MARKER`."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x, y, w, h = (int(value) for value in MARKER)
    image[y : y + h, x : x + w] = 255
    return image


def ink_extent(image: np.ndarray) -> tuple[int, int, int, int]:
    """Where the white ink actually is, as (x, y, width, height) — the counterpart of the box.

    Read from the pixels rather than from anything the pipeline reported, which is the point:
    it is the independent measurement the returned box is compared against.
    """
    rows, cols = np.where(image.max(axis=2) > 127)
    assert rows.size, "the marker vanished entirely — nothing to compare a box against"
    return (
        int(cols.min()),
        int(rows.min()),
        int(cols.max() - cols.min()) + 1,
        int(rows.max() - rows.min()) + 1,
    )


# --------------------------------------------------------------------- the gate --


def test_a_mirrored_document_carries_its_box_to_the_mirrored_place():
    """hand-computed. A horizontal flip maps column c to W - 1 - c, so the marker's columns
    20 … 59 become 200 - 1 - 59 = 140 … 200 - 1 - 20 = 179. That is 40 pixels beginning at
    140, and the rows are untouched:

        (20, 30, 40, 10)  ->  (140, 30, 40, 10)

    A flip is used because it is lossless — no resampling, so the ink after the transform is
    exactly as crisp as before it and the pixel measurement has no interpolation slack to
    hide a one-pixel disagreement in.
    """
    image, boxes = fixture_image(), {"marker": MARKER}

    moved_image, moved = carry_boxes([A.HorizontalFlip(p=1.0)], image, boxes, seed=0)

    assert moved["marker"] == (140.0, 30.0, 40.0, 10.0)
    assert ink_extent(moved_image) == (140, 30, 40, 10)


def test_a_translated_document_carries_its_box_by_the_same_offset():
    """hand-computed. A translation of +12 columns and -8 rows moves the marker's columns
    20 … 59 to 32 … 71 and its rows 30 … 39 to 22 … 31:

        (20, 30, 40, 10)  ->  (32, 22, 40, 10)

    A second transform of a different kind, because a mirror is its own inverse and a
    pipeline that applied the box transform twice would still pass the first test. A
    translation applied twice lands at 44, not 32.

    Nearest-neighbour resampling on an exact integer offset is an exact copy, so this
    transform is lossless too and the ink assertion stays exact.
    """
    image, boxes = fixture_image(), {"marker": MARKER}
    shift = A.Affine(
        translate_px={"x": 12, "y": -8},
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=1.0,
    )

    moved_image, moved = carry_boxes([shift], image, boxes, seed=0)

    assert moved["marker"] == (32.0, 22.0, 40.0, 10.0)
    assert ink_extent(moved_image) == (32, 22, 40, 10)


def test_a_box_pushed_off_the_edge_is_reported_off_the_edge_and_not_trimmed():
    """hand-computed, and the case `content_complete` rests on. A translation of -30 columns
    moves the marker's columns 20 … 59 to -10 … 29, so ten of its forty columns are outside
    the image and thirty remain:

        the box, untrimmed:  (-10, 30, 40, 10)
        the ink that is left:  columns 0 … 29, rows 30 … 39  ->  (0, 30, 30, 10)

    the two disagree on purpose. A pipeline that clipped the box to the frame would return
    (0, 30, 30, 10) — indistinguishable from a document that never lost anything, which is
    precisely how a crop would come to be reported as complete. `carry_boxes` sets
    `clip=False` for this reason, and this is the test that says so.
    """
    image, boxes = fixture_image(), {"marker": MARKER}
    shift = A.Affine(
        translate_px={"x": -30, "y": 0},
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=1.0,
    )

    moved_image, moved = carry_boxes([shift], image, boxes, seed=0)

    assert moved["marker"] == (-10.0, 30.0, 40.0, 10.0)
    assert ink_extent(moved_image) == (0, 30, 30, 10)
    assert clipped_edges(moved["marker"], WIDTH, HEIGHT) == ("left",)


def test_a_box_wholly_inside_the_frame_crosses_no_edge():
    """The negative half of `clipped_edges`, hand-computed from the frame: the marker spans
    columns 20 … 59 of 200 and rows 30 … 39 of 100, so it touches nothing."""
    assert clipped_edges(MARKER, WIDTH, HEIGHT) == ()


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        # Hand-computed against a 200 × 100 frame. Each case puts the box one pixel over one
        # edge and nowhere else, so a mixed-up comparison shows as the wrong edge name rather
        # than as no finding at all.
        ((-1.0, 30.0, 40.0, 10.0), ("left",)),
        ((20.0, -1.0, 40.0, 10.0), ("top",)),
        ((161.0, 30.0, 40.0, 10.0), ("right",)),  # 161 + 40 = 201 > 200
        ((20.0, 91.0, 40.0, 10.0), ("bottom",)),  # 91 + 10 = 101 > 100
        # Exactly flush with the far corner: 160 + 40 = 200 and 90 + 10 = 100, both equal to
        # the frame and therefore inside it. The off-by-one that would call this clipped is
        # the likeliest defect in the whole function.
        ((160.0, 90.0, 40.0, 10.0), ()),
        ((-5.0, -5.0, 400.0, 200.0), ("left", "top", "right", "bottom")),
    ],
)
def test_each_edge_is_named_by_itself(box, expected):
    assert clipped_edges(box, WIDTH, HEIGHT) == expected
