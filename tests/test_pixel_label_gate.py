"""The gate that reads pixels, and the teeth that keep it able to bite.

🔴 why this file is not like the other gate tests. Every measuring instrument in this repository
has been wrong at least once, and an image-side instrument is the easiest kind to be wrong quietly:
a threshold that never fires reads exactly like a corpus with nothing wrong in it. So the file is in
two halves. The first drives the measures with hand-made pixels whose answer is known. The second
runs the gate over a real generated corpus twice — once as the pipeline ships, where it must find
nothing, and once with every channel's JPEG quality forced to 3–5, where it must go red. The second
half is the whole point: it is the mutation that passed all 1524 tests before this gate existed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest

from receipt_synth import assembler, degrader
from receipt_synth.schemas import Capture

pytest.importorskip("albumentations")

from pixel_label_gate import (  # noqa: E402  — `tools/` reaches the path through conftest
    LEGIBILITY_FLOORS,
    Coverage,
    _ink_template,
    _labelled,
    _peak_correlation,
    edit_fields,
    edited_text,
    scan_run,
    survival_findings,
)


def a_page_of_text(width: int = 420, height: int = 120) -> np.ndarray:
    """Marks on paper, drawn rather than rendered: thin strokes and small digits, which is what a
    low JPEG quality destroys first. A uniform rectangle would survive any compression and would
    make the mutation half of this file pass for the wrong reason."""
    image = np.full((height, width, 3), 255, np.uint8)
    for index, line in enumerate(("2 301,90  383,65", "1 x 866,50  866,50", "UA1639 0035 7542")):
        cv2.putText(
            image, line, (12, 30 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1
        )
    return image


# The three lines above, as boxes. Deliberately tight around the strokes rather than generous: a
# labelled box in this corpus is an element's own rectangle.
BOXES = {
    "line_0": (12.0, 14.0, 210.0, 22.0),
    "line_1": (12.0, 48.0, 230.0, 22.0),
    "line_2": (12.0, 82.0, 230.0, 22.0),
}


# --------------------------------------------------------------------- the measures alone --


def test_the_ink_template_is_the_marks_and_not_the_box():
    """A right-aligned invoice cell is mostly blank paper, and correlating over the blank part is
    what put three legible digits below the floor before this existed."""
    crop = np.full((20, 60), 255, np.uint8)
    crop[8:12, 40:46] = 0
    template = _ink_template(crop)
    assert template is not None
    assert template.shape == (4, 6)


def test_a_blank_crop_has_no_ink_template():
    """`None` rather than an empty array, because a blank box is a question for the evidence
    statement rather than a failure of the capture."""
    assert _ink_template(np.full((20, 60), 255, np.uint8)) is None


def test_a_shifted_pattern_still_matches_itself():
    """The correlation slides on purpose: the box a rotation leaves behind is the axis-aligned hull
    of a rotated rectangle, so the marks sit somewhere inside the crop rather than at its corner."""
    page = a_page_of_text()
    grey = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    template = _ink_template(grey[14:36, 12:222])
    padded = cv2.copyMakeBorder(grey[10:40, 8:230], 4, 4, 6, 6, cv2.BORDER_CONSTANT, value=255)
    assert _peak_correlation(padded, template) > 0.99


def test_the_edit_keeps_the_glyphs_and_changes_the_order():
    assert edited_text("2 301,90") == "09,103 2"


def test_a_palindrome_falls_back_to_a_substitution():
    """A one-character value reads the same backwards, and a box whose value cannot be changed at
    all cannot be measured — so those take the confusable-digit route instead."""
    assert edited_text("1") == "7"
    assert edited_text("8") == "0"


def test_only_the_named_leaf_fields_are_edited():
    html = (
        '<div data-field="total">2 301,90</div>'
        '<div data-field="vat_total">383,65</div>'
        '<div data-field="row"><span data-field="inner">7</span></div>'
    )
    edited_html, edited = edit_fields(html, {"total", "row"})
    assert edited == {"total"}, "a wrapper holding another field's element is not editable text"
    assert "09,103 2" in edited_html
    assert "383,65" in edited_html, "a field outside the group is left alone"
    assert '<span data-field="inner">7</span>' in edited_html


# ------------------------------------------------------- the survival statement, by itself --


def _screenshot_pair(quality: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """A page and what the screenshot channel does to it, optionally at a forced quality.

    The screenshot recipe is `A.NoOp` plus one JPEG step, so a compression applied here is that
    channel — which is what lets this half of the file run without a browser.
    """
    page = a_page_of_text()
    if quality is None:
        return page, degrader.degrade(page, BOXES, seed=7, capture=Capture.SCREENSHOT).image
    compressed = A.ImageCompression(compression_type="jpeg", quality_range=quality, p=1.0)
    return page, compressed(image=page)["image"]


def test_the_shipped_screenshot_quality_leaves_every_box_readable():
    clean, shipped = _screenshot_pair(None)
    assert (
        survival_findings(
            "hand-made",
            clean=clean,
            boxes=BOXES,
            shipped=shipped,
            shipped_boxes=BOXES,
            capture=Capture.SCREENSHOT,
            seed=7,
        )
        == []
    )


def test_quality_three_to_five_is_reported_on_hand_made_pixels():
    """The mutation, at the smallest scale it can be stated at."""
    clean, shipped = _screenshot_pair((3, 5))
    findings = survival_findings(
        "hand-made",
        clean=clean,
        boxes=BOXES,
        shipped=shipped,
        shipped_boxes=BOXES,
        capture=Capture.SCREENSHOT,
        seed=7,
    )
    assert [f.kind for f in findings], "every box was destroyed and the gate said nothing"
    assert {f.kind for f in findings} == {"illegible_after_capture"}


def test_a_box_that_moved_is_reported_as_the_instrument_failing_and_not_the_corpus():
    """The reference has to describe the same geometry, or the comparison answers the wrong
    question. A gate that reported this as illegible text would send a reader hunting a defect in
    the degrader's compression that is really in this file's own assumption."""
    clean, shipped = _screenshot_pair(None)
    moved = {name: (x + 3, y, w, h) for name, (x, y, w, h) in BOXES.items()}
    findings = survival_findings(
        "hand-made",
        clean=clean,
        boxes=BOXES,
        shipped=shipped,
        shipped_boxes=moved,
        capture=Capture.SCREENSHOT,
        seed=7,
    )
    assert [f.kind for f in findings] == ["instrument_geometry_drift"]


def test_every_channel_has_a_floor():
    """A channel with no floor would be measured against `None` and pass by accident. The recipes
    live in `degrader`; the floors are calibrated per recipe, so a new channel needs one."""
    assert set(LEGIBILITY_FLOORS) == set(Capture)


# ------------------------------------------------------------- the gate over a real corpus --


@pytest.fixture(scope="module")
def real_run() -> tuple[list, Coverage]:
    """A small generated corpus, held to both statements. Two personas rather than one: the run has
    to contain more than a single capture channel for the survival statement to mean anything."""
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        yield scan_run(
            seed=20260803,
            personas=2,
            claims_per_persona=2,
            out_dir=Path(tmp) / "corpus",
            staging=staging,
        )


def test_the_shipped_corpus_has_no_findings(real_run):
    findings, _ = real_run
    assert [str(f) for f in findings] == []


def test_a_bundled_document_s_field_is_a_labelled_box_and_its_frame_is_not():
    """🔴 the distinction that cost this gate a fifth of the production corpus.

    Both names begin `__`, and only one of them is geometry. `__doc_1__amount` is document 1's
    amount, offset into the composed page — a rectangle that reaches a consumer's label.
    `__file_region_1__` is where that document sits on the sheet, which `assembler` strips before
    writing. Filtering on `__doc` took the first with the second, and the first production sweep
    measured 30 605 boxes of a corpus carrying 37 796 while reporting no shortfall.
    """
    kept = _labelled(
        {
            "amount": (0, 0, 10, 10),
            "__doc_1__amount": (0, 20, 10, 10),
            "__doc_11__amount": (0, 40, 10, 10),
            "__file_region_1__": (0, 0, 100, 100),
            "__page_region_1__": (0, 0, 100, 50),
            "__content_extent__": (0, 0, 100, 100),
        }
    )

    assert sorted(kept) == ["__doc_11__amount", "__doc_1__amount", "amount"]


def test_the_gate_sees_every_box_the_run_wrote_including_the_bundled_ones():
    """The denominator, checked against the corpus rather than against itself.

    The bundle share is forced to 1 so that every eligible claim files as one composed page: on a
    run of this size the ordinary 0.25 draws too few bundles for their absence to be visible, which
    is how the shortfall survived four seeds of sweeping. `boxes_seen` must then equal the
    `field_bboxes` count of the written labels exactly — not "at least", because the gate exceeding
    the corpus would mean it is measuring something a consumer never receives.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "file_composition_share", lambda name: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            staging.mkdir()
            _, coverage = scan_run(
                seed=20260803,
                personas=2,
                claims_per_persona=2,
                out_dir=Path(tmp) / "corpus",
                staging=staging,
                evidence=False,
            )

    written = coverage.unmeasured["run: labelled boxes in the manifest"]
    assert written > 0, "nothing was written, so this assertion would compare two zeros"
    assert coverage.boxes_seen == written


def test_the_gate_measured_something(real_run):
    """The empty-denominator guard, and it is not ceremony: the gate reaches a run by wrapping two
    functions, so a rename would leave it observing nothing and reporting a clean corpus."""
    _, coverage = real_run
    assert coverage.documents >= 4
    assert coverage.survival_measured >= 50
    assert coverage.evidence_measured >= 50


def test_the_gate_goes_red_when_the_channels_compress_at_three_to_five(monkeypatch):
    """🔴 the mutation this gate exists for, run through the whole pipeline rather than on hand-made
    pixels: every channel's JPEG quality forced to 3–5, which is what passed the entire suite before
    this file was written.

    ⚠️ the sample has to contain a lossy channel, and it used to hold one by luck. `digital_pdf`
    applies no compression at all — the mutation has nothing to bite on — so a run whose every
    document happens to be drawn on that channel makes this test measure nothing while looking
    like it passed. That is exactly what a redrawn seed stream produced: one persona and two
    claims came back as two digital PDFs. Three claims is the smallest sample that still draws a
    photo or a scan at this seed. The assertion below is what says so out loud rather than
    reporting a clean corpus — it is the failure this note is written from."""
    real_geometry = degrader._geometry

    def compressing_at_three(capture: Capture):
        return [
            A.ImageCompression(compression_type="jpeg", quality_range=(3, 5), p=1.0)
            if isinstance(transform, A.ImageCompression)
            else transform
            for transform in real_geometry(capture)
        ]

    monkeypatch.setattr(degrader, "_geometry", compressing_at_three)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        findings, coverage = scan_run(
            seed=20260803,
            personas=1,
            claims_per_persona=3,
            out_dir=Path(tmp) / "corpus",
            staging=staging,
            evidence=False,
        )
    assert coverage.survival_measured >= 20, "the run has to have been measured to be red"
    illegible = [f for f in findings if f.kind == "illegible_after_capture"]
    assert illegible, (
        "the corpus was compressed into mush and the gate reported nothing — if every document "
        "of this run was drawn on `digital_pdf`, which applies no compression, the sample "
        "measured nothing and has to grow rather than the gate being blamed"
    )
