"""The gate that reads PIXELS — the one side of this generator nothing else measures.

    uv run python tools/pixel_label_gate.py --seed 20260803 --personas 8 --claims-per-persona 3

🔴 WHY IT EXISTS. Generation is label-first, so the label is the CAUSE of the image and any
disagreement between what is printed and what is recorded is a defect of this generator rather
than annotation noise. Every gate in `tests/` nevertheless compares code against code: the
reference-text gate reads `innerText` against the markup (DOM against DOM), the bbox gate moves a
synthetic white rectangle, the render-region gate does box-in-box arithmetic. None of them opens
the PNG a consumer receives. Measured rather than argued: setting every channel's JPEG quality to
3–5 — a mutation that visibly wrecks decimal separators — passed the whole suite.

WHAT A `field_bboxes` ENTRY PROMISES, because that is what this gate holds the corpus to. It is
part of the ground truth, and it says: *this rectangle of this image is where that field is
printed*. Two things have to be true of it, and each is a separate statement here.

  1. EVIDENCE — the pixels inside the box are that field's own ink. Tested by EDITING the field's
     printed characters, in place, to characters of the same width, and re-rendering: the box's
     pixels must change, and the pixels of boxes that share no area with it must not. A box whose
     pixels do not move when its own value changes is pointing at nothing, or at another field's
     value; a box that moves when a stranger's value changes is showing that stranger.
     ⚠️ This is the statement that is INDEPENDENT of the renderer's own account of itself. It does
     not ask the DOM where a value is; it asks the raster which pixels the value is responsible for.

  2. SURVIVAL — the capture channel did not destroy the marks. Tested against THE SAME CHANNEL
     WITH ITS JPEG STEP LEFT OUT (`degrader.degrade(..., compress=False)`): same paper phase, same
     padding, perspective, rotation and blur, same seed, so the geometry cancels and the peak
     normalized cross-correlation over the box answers one question — what the compression cost.

⛔ IT IS NOT AN OCR TEST AND CARRIES NO RECOGNIZER. Two were tried and rejected before this shape
was chosen: Tesseract 5 (eng and snum) misreads the mono digits of these renders on CLEAN pixels —
«2 301,90» came back as «2 361,99» — so it would have manufactured findings, and a template
rasterized from the vendored TTF separated the true string from a one-digit decoy by 0.02 NCC,
which is not an instrument either. Both failures are of the same kind: an instrument less reliable
than the thing it measures. What is asked here instead is answerable exactly — whether the pixels
of a labelled box depend on that field's value, and whether the shipped file still carries them.

THE CALIBRATION, and it is a measurement rather than a preference. Peak NCC over the MARKS of each
labelled box (see `_ink_template`), four seeds × 6 personas × 3 claims — 2 461 boxes — shipped
recipe against the same recipe with every channel's JPEG quality forced to 3–5:

                    shipped: worst box / median     quality 3–5: median / p05
    digital_pdf         1.000 / 1.000                 1.000 / 1.000   (compresses nothing)
    screenshot          0.986 / 0.997                 0.802–0.865 / 0.66–0.72
    scan                0.904 / 0.987                 0.638–0.713 / 0.33–0.45
    photo               0.765 / 0.977                 0.639–0.746 / 0.23–0.31

🔴 A FLOOR PER CHANNEL AND NOT ONE FLOOR, because the four are not comparable: a photograph is
padded, warped, rotated, blurred and compressed at 45–75, a screenshot only compressed at 70–88,
and `digital_pdf` is the identity. One number would have to clear the photograph's worst box, which
would put it beneath the mutation's median on the other three — a floor that could not go red where
the damage is easiest to see. `LEGIBILITY_FLOORS` therefore sits below each channel's own worst
shipped box by 3.6–6.5 points, and every one of them is far above that channel's mutated median.

⚠️ A CHANNEL WHOSE RECIPE GAINS A FURTHER LOSSY STEP HAS TO REDO THIS MEASUREMENT. The floors are a
property of the recipes in `degrader`, not of the JPEG format, and a new blur or a wider quality
range moves them.

WHAT IT DOES NOT SEE, stated so the report cannot be read as wider than it is:
  ⛔ it does not judge whether the printed value equals the LABEL value — `reference_text` and the
     builders' own invariant tests answer that, and the chain is: label = printed text (those
     tests) ∧ printed text ⇒ these pixels (this gate);
  ⛔ it does not read a caption, so a value printed under the wrong heading in the right place is
     invisible to it;
  ⛔ a box with no editable text of its own (a wrapper element whose children carry the values, an
     empty box, a QR) is REPORTED AS UNMEASURED rather than passed. The counts are printed, because
     a gate that silently narrows its own denominator reads as coverage it does not have.

HOW IT REACHES A REAL RUN. It wraps `assembler.degrade` and `Renderer.render` for the duration of
one `generate_dataset` call, so the documents it measures are the documents that run produced —
every archetype, every channel, at their real draw rates — and it adds NOTHING to the shipped
pipeline: no byte of a corpus changes because this file exists. The wrapping is the fragile part,
so the gate asserts its own coverage: a run whose observations are fewer than the manifest's
documents is reported as a broken instrument, not as a clean corpus.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from receipt_synth import assembler
from receipt_synth.assembler import generate_dataset
from receipt_synth.degrader import degrade
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import BBox, Capture, Country

# The floors derived in the header. Peak NCC over a box's marks, shipped pixels against the same
# channel without its JPEG step. `digital_pdf` compresses nothing, so its floor is the identity
# stated as a number rather than a tolerance: anything but a perfect match is a channel that stopped
# being the identity, which `tests/test_capture_mix.py` also asserts from the other side.
LEGIBILITY_FLOORS = {
    Capture.DIGITAL_PDF: 0.999,
    Capture.SCREENSHOT: 0.95,
    Capture.SCAN: 0.85,
    Capture.PHOTO: 0.70,
}

# How many groups the editable fields of a page are split into, so that a box showing a NEIGHBOUR's
# value is caught. Three, because the boxes of one receipt line sort as `item_0_name`,
# `item_0_price`, `item_0_qty`, `item_0_sum` — with two groups the price and the sum would be edited
# together and a swap between them would look like each box answering to itself.
EDIT_GROUPS = 3

# When a box that was NOT edited still differs, this is how close its pattern has to stay to its own
# former self to count as the same value merely re-positioned. Antialiasing under a sub-pixel shift
# scores 0.999 and above; a box that acquired a different value scores far below.
_SAME_PATTERN = 0.99

# 🔴 THE EDIT IS A REORDERING AND NOT A SUBSTITUTION, and the reason is the whole reason this
# statement can be measured at all. An edited element must occupy exactly the width it occupied
# before, or the page reflows and a pixel comparison answers about the reflow instead. A reversal
# keeps the GLYPH MULTISET, so the advance width is the same in a proportional face as well as in a
# mono one — measured: substituting characters left 94 of 137 boxes unmeasurable on a two-persona
# run, reversing them leaves the layout alone.
#
# The fallback matters for the same reason the reversal does: a one-character value and a
# palindrome read the same backwards, so those fall back to a substitution — a digit for the digit
# it is most easily mistaken for in this mono face, which makes the edit hard to see rather than
# easy. A box whose pixels barely respond to a hard edit is a box worth reporting.
_SUBSTITUTIONS = {
    "0": "8", "8": "0", "1": "7", "7": "1", "3": "9", "9": "3", "5": "6", "6": "5",
    "2": "4", "4": "2",
    "а": "о", "о": "а", "е": "с", "с": "е", "и": "н", "н": "и", "р": "в", "в": "р",
    "a": "o", "o": "a", "b": "d", "d": "b", "m": "n", "n": "m",
    "А": "О", "О": "А", "Е": "С", "С": "Е", "И": "Н", "Н": "И",
}


def edited_text(body: str) -> str:
    """The same characters in another order — or, where that reads the same, other characters."""
    reversed_body = body[::-1]
    if reversed_body != body:
        return reversed_body
    return "".join(_SUBSTITUTIONS.get(ch, ch) for ch in body)

# An element that carries a `data-field` AND whose whole content is text. A wrapper whose children
# hold the values is deliberately not matched: editing it would edit its children too, and the
# question this gate asks is per box.
_LEAF_FIELD = re.compile(
    r'<(?P<tag>[a-z]+)(?P<attrs>[^>]*\sdata-field="(?P<name>[^"]+)"[^>]*)>(?P<body>[^<>]*)</\1>'
)


@dataclass(frozen=True)
class Finding:
    """One defect, in the corpus or in this instrument.

    `field` is empty on a document-level finding.
    """

    kind: str
    doc_id: str
    field: str
    detail: str

    def __str__(self) -> str:
        where = f"{self.doc_id}:{self.field}" if self.field else self.doc_id
        return f"{self.kind:26} {where:34} {self.detail}"


@dataclass
class Coverage:
    """The denominators. Printed whether or not anything was found — see the header."""

    documents: int = 0
    files: int = 0
    boxes_seen: int = 0
    evidence_measured: int = 0
    survival_measured: int = 0
    unmeasured: Counter[str] = field(default_factory=Counter)

    def report(self) -> str:
        lines = [
            f"  documents observed        {self.documents}",
            f"  files observed            {self.files}",
            f"  labelled boxes seen       {self.boxes_seen}",
            f"  boxes measured: evidence  {self.evidence_measured}",
            f"  boxes measured: survival  {self.survival_measured}",
        ]
        if self.unmeasured:
            lines.append("  unmeasured boxes, by reason:")
            lines += [f"    {reason:34} {n}" for reason, n in sorted(self.unmeasured.items())]
        return "\n".join(lines)


# --------------------------------------------------------------------- measuring one box --


def _crop(image: np.ndarray, box: BBox) -> np.ndarray:
    """The visible part of a box. A box may lie partly outside the frame by design (see
    `degrader.carry_boxes`), and the pixels that exist are the ones a consumer can read."""
    x, y, w, h = (int(round(v)) for v in box)
    return image[max(y, 0) : y + h, max(x, 0) : x + w]


def _peak_correlation(target: np.ndarray, template: np.ndarray) -> float:
    """Peak normalized cross-correlation of the template inside the target.

    A peak over a sliding window rather than a pixel-for-pixel comparison, because the box a
    rotation leaves behind is the axis-aligned hull of a rotated rectangle — a pixel further left
    than where the ink now starts — so the ink is somewhere inside the crop rather than at its
    corner.
    """
    if template.size == 0 or target.size == 0:
        return float("nan")
    th, tw = template.shape[:2]
    hh, ww = target.shape[:2]
    if th > hh or tw > ww:
        target = cv2.copyMakeBorder(
            target, 0, max(th - hh, 0), 0, max(tw - ww, 0), cv2.BORDER_CONSTANT, value=255
        )
    return float(cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED).max())


def _grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


# How far from the paper a pixel has to be to count as a mark. Read against the crop's OWN median
# rather than against white, because a photographed page carries a lighting gradient and a shadow:
# the paper inside one box is grey, and a fixed threshold would call a whole box ink.
#
# ⚠️ IT IS A FIXED DISTANCE ALL THE SAME, AND THAT IS THE INSTRUMENT'S KNOWN LIMIT. On a photograph
# whose shadow falls across a requisite the WHOLE crop can span 37 grey levels — the digits stay
# legible to a reader, and their distance from the local paper is under this threshold, so the box
# goes unmeasured. Measured: 2 boxes of 1580 on one seed, both of them a tax code under a cast
# shadow. They are counted under `survival: no marks in the reference` and printed, so the gap is a
# number in the report rather than a silence; a threshold derived from each crop's own dynamic range
# would close it and would re-open the calibration above, which is why it is recorded rather than
# tuned here.
#
# 🔴 AND IN EITHER DIRECTION, WHICH THE FIRST VERSION GOT WRONG AND THE COVERAGE REPORT CAUGHT. Two
# archetypes of this corpus are banking-app screens in a DARK theme — light text on a dark panel —
# so "darker than the paper" found no marks in any of their boxes and nine of them went unmeasured
# on a single seed. They were counted and printed rather than passed over, which is the only reason
# the gap was visible at all; a gate that had reported them as clean would have been silent about
# two whole archetypes.
_INK_CONTRAST = 25


def _ink_template(reference: np.ndarray) -> np.ndarray | None:
    """The part of a reference crop that has marks in it, or `None` if it has none.

    🔴 THE CORRELATION IS TAKEN OVER THE GLYPHS AND NOT OVER THE BOX, and the difference is the
    difference between an instrument and a rumour. A right-aligned invoice cell is 66 px wide and
    holds a single «1»: over the whole box the comparison is dominated by how faithfully the
    compression reproduced blank paper, and on a shadowed photograph that put three such cells
    below the floor with their digits plainly still there. Cropping to the ink first asks about the
    mark.
    """
    paper = float(np.median(reference))
    marks = np.argwhere(np.abs(reference.astype(np.int16) - paper) > _INK_CONTRAST)
    if marks.size == 0:
        return None
    (top, left), (bottom, right) = marks.min(axis=0), marks.max(axis=0)
    return reference[top : bottom + 1, left : right + 1]


def _shares_a_line(one: BBox, other: BBox, *, slack: int = 1) -> bool:
    """Whether two boxes sit on the same line of text.

    🔴 THE EXEMPTION THAT SEPARATES THE INSTRUMENT'S OWN NOISE FROM A FINDING, and it took three
    false ones to state. A line of text is laid out as ONE FLOW: editing anything in it re-shapes
    what follows, and Cyrillic reversed inside «Олекса Семенюк, РНОКПП 2674077210» moves the code
    after it by a FRACTION of a pixel. The box rounds to the same rectangle and its antialiasing
    does not, so every invoice reported its `buyer_code` and its `signatory_post` as showing
    somebody else's value at pattern correlations of 0.68–0.98 — real pixel differences, and not one
    of them a wrong value.
    ⛔ Nothing is lost by exempting them: a box that shows a NEIGHBOUR's value instead of its own
    fails the first half of the statement — its own edit moves nothing — and that half is exempted
    from nothing.
    """
    _, ay, _, ah = one
    _, by, _, bh = other
    return ay - slack < by + bh + slack and by - slack < ay + ah + slack


# ------------------------------------------------------------------ statement 2: survival --


def survival_findings(
    doc_id: str,
    *,
    clean: np.ndarray,
    boxes: dict[str, BBox],
    shipped: np.ndarray,
    shipped_boxes: dict[str, BBox],
    capture: Capture,
    seed: int,
    floor: float | None = None,
    coverage: Coverage | None = None,
) -> list[Finding]:
    """Did the capture channel leave every labelled box readable.

    The reference is the same channel without its JPEG step, so what is measured is the
    compression alone. `shipped_boxes` is compared against the reference's own boxes first: if the
    two disagree the reference is not describing the same geometry, which is a fault of this
    instrument and is reported as one.
    """
    if floor is None:
        floor = LEGIBILITY_FLOORS[capture]
    reference = degrade(clean, boxes, seed=seed, capture=capture, compress=False)
    findings: list[Finding] = []
    if reference.field_bboxes != shipped_boxes:
        moved = sorted(
            name
            for name, box in shipped_boxes.items()
            if reference.field_bboxes.get(name) != box
        )
        return [
            Finding(
                "instrument_geometry_drift",
                doc_id,
                "",
                f"the reference channel put {len(moved)} box(es) elsewhere ({moved[:4]}); the "
                "comparison would measure geometry rather than compression",
            )
        ]
    target, model = _grey(shipped), _grey(reference.image)
    for name, box in sorted(shipped_boxes.items()):
        crop_target, crop_model = _crop(target, box), _crop(model, box)
        if min(crop_model.shape[:2] or (0,)) < 3 or crop_model.size == 0:
            if coverage is not None:
                coverage.unmeasured["survival: no pixels on the page"] += 1
            continue
        template = _ink_template(crop_model)
        if template is None:
            # A box the render left blank. Not a pass and not a failure of the capture: whether a
            # labelled box may be empty at all is the evidence statement's question.
            if coverage is not None:
                coverage.unmeasured["survival: no marks in the reference"] += 1
            continue
        score = _peak_correlation(crop_target, template)
        if np.isnan(score):
            if coverage is not None:
                coverage.unmeasured["survival: correlation undefined"] += 1
            continue
        if coverage is not None:
            coverage.survival_measured += 1
        if score < floor:
            findings.append(
                Finding(
                    "illegible_after_capture",
                    doc_id,
                    name,
                    f"{capture.value}: peak correlation {score:.3f} against the same channel "
                    f"without its JPEG step, floor {floor:.3f}",
                )
            )
    return findings


# ------------------------------------------------------------------ statement 1: evidence --


def edit_fields(html: str, names: set[str]) -> tuple[str, set[str]]:
    """Substitute same-width characters inside the named leaf fields; return the edited page and
    the names actually edited.

    A name that is not returned had nothing this gate can edit — no text of its own, or text made
    entirely of characters `_SUBSTITUTIONS` does not cover — and its caller counts it as unmeasured
    rather than as passed.
    """
    edited: set[str] = set()

    def substitute(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in names:
            return match.group(0)
        body = match.group("body")
        swapped = edited_text(body)
        if swapped == body:
            return match.group(0)
        edited.add(name)
        return f"<{match.group('tag')}{match.group('attrs')}>{swapped}</{match.group('tag')}>"

    return _LEAF_FIELD.sub(substitute, html), edited


def evidence_findings(
    doc_id: str,
    *,
    renderer: Renderer,
    html: str,
    clean: np.ndarray,
    boxes: dict[str, BBox],
    staging: Path,
    coverage: Coverage | None = None,
) -> list[Finding]:
    """Do the pixels of each labelled box answer to that field's own printed characters.

    One re-render per edit group. Every group is checked for both halves of the statement — its own
    boxes must change, and boxes outside it that share no area with an edited box must not.
    """
    findings: list[Finding] = []
    names = sorted(boxes)
    groups = [set(names[index::EDIT_GROUPS]) for index in range(EDIT_GROUPS)]
    for index, group in enumerate(groups):
        page, edited = edit_fields(html, group)
        if not edited:
            if coverage is not None:
                coverage.unmeasured["evidence: no editable text in the element"] += len(group)
            continue
        if coverage is not None:
            coverage.unmeasured["evidence: no editable text in the element"] += len(
                group - edited
            )
        variant = renderer.render_html(page, staging / f"{doc_id}_edit{index}.png", name=doc_id)
        after = cv2.imread(str(variant.image_path))
        if after is None or after.shape != clean.shape:
            if coverage is not None:
                coverage.unmeasured["evidence: the edit resized the page"] += len(edited)
            continue
        difference = cv2.absdiff(_grey(clean), _grey(after))
        # A box that MOVED cannot be measured: the comparison would answer about the move. Judged
        # per box rather than per page, because an edit that reflows one table cell leaves the rest
        # of the page where it was, and dropping the whole page for it would throw away most of the
        # corpus.
        settled = {name for name, box in variant.field_bboxes.items() if boxes.get(name) == box}
        for name in sorted(edited):
            if name not in settled:
                if coverage is not None:
                    coverage.unmeasured["evidence: the edited box moved"] += 1
                continue
            if coverage is not None:
                coverage.evidence_measured += 1
            if int(_crop(difference, boxes[name]).sum()) == 0:
                findings.append(
                    Finding(
                        "not_evidenced",
                        doc_id,
                        name,
                        "the box is unchanged when this field's own printed characters change, "
                        "so its pixels are not this field's",
                    )
                )
        # The other half of the statement, and it is only answerable where NOTHING moved: on a page
        # that reflowed, a box may differ because the text beside it shifted.
        if set(variant.field_bboxes) != settled:
            if coverage is not None:
                coverage.unmeasured["strangers: the page reflowed"] += 1
            continue
        for name in sorted(set(names) - edited):
            if int(_crop(difference, boxes[name]).sum()) == 0:
                continue
            if any(_shares_a_line(boxes[name], boxes[n]) for n in edited):
                continue
            # ⚠️ A DIFFERENT PIXEL IS NOT A DIFFERENT VALUE, and this is the distinction that
            # separated the instrument's own noise from a finding. Reversing a proportional name
            # changes its kerning, so the text after it on the same line moves by a FRACTION of a
            # pixel: the box rounds to the same rectangle, the antialiasing does not, and every
            # invoice reported its `buyer_code` as showing somebody else's value. The peak
            # correlation slides, so a pattern that merely moved still matches itself; only a box
            # whose pattern is no longer there is reported.
            moved_only = _peak_correlation(
                _crop(_grey(after), boxes[name]), _crop(_grey(clean), boxes[name])
            )
            if not np.isnan(moved_only) and moved_only >= _SAME_PATTERN:
                continue
            findings.append(
                Finding(
                    "shows_another_field",
                    doc_id,
                    name,
                    f"the box changed while {sorted(edited)[:4]} were edited and it shares no "
                    f"area with any of them (pattern correlation {moved_only:.3f})",
                )
            )
    return findings


# ------------------------------------------------------------------------ driving a run --


@dataclass
class _Observed:
    """One document as the pipeline had it at the seam: the page, the clean raster, the boxes."""

    doc_id: str
    html: str
    clean_path: Path
    boxes: dict[str, BBox]


@contextmanager
def _observing(staging: Path) -> Iterator[tuple[list[_Observed], list[dict]]]:
    """Record what `Renderer.render` and `assembler.degrade` were given, for one run.

    Wrapping rather than a parameter threaded through the assembler: the gate then adds nothing to
    the shipped pipeline and no corpus can differ for its existence. The cost is that a rename
    silently unhooks it, which is what the coverage assertion in `main` exists to catch.
    """
    documents: list[_Observed] = []
    files: list[dict] = []
    real_render = Renderer.render
    real_degrade = assembler.degrade

    def render(self: Renderer, template_name: str, context: dict, output_path: Path):
        html = self.build_html(template_name, context)
        rendered = self.render_html(html, output_path, name=template_name)
        keep = staging / f"clean_{len(documents):04d}.png"
        keep.write_bytes(Path(rendered.image_path).read_bytes())
        documents.append(
            _Observed(
                doc_id=output_path.stem,
                html=html,
                clean_path=keep,
                boxes=dict(rendered.field_bboxes),
            )
        )
        return rendered

    def wrapped_degrade(image, field_bboxes, *, seed, capture=Capture.SCREENSHOT, compress=True):
        result = real_degrade(
            image, field_bboxes, seed=seed, capture=capture, compress=compress
        )
        files.append(
            {
                "clean": image.copy(),
                "boxes": dict(field_bboxes),
                "shipped": result.image.copy(),
                "shipped_boxes": dict(result.field_bboxes),
                "capture": capture,
                "seed": seed,
            }
        )
        return result

    Renderer.render = render
    assembler.degrade = wrapped_degrade
    try:
        yield documents, files
    finally:
        Renderer.render = real_render
        assembler.degrade = real_degrade


# Box families the degrader carries alongside the labelled fields — the content extent, the page
# regions, the bundle's per-document frames. They are geometry rather than printed values, and
# `assembler` strips them before a label is written.
_RESERVED = ("__content_extent__", "__page_region", "__file_region", "__doc")


def _labelled(boxes: dict[str, BBox]) -> dict[str, BBox]:
    return {
        name: box
        for name, box in boxes.items()
        if not any(reserved in name for reserved in _RESERVED)
    }


def scan_run(
    *,
    seed: int,
    personas: int,
    claims_per_persona: int,
    out_dir: Path,
    staging: Path,
    floor: float | None = None,
    evidence: bool = True,
) -> tuple[list[Finding], Coverage]:
    """Generate a corpus, then hold every labelled box of it to both statements."""
    findings: list[Finding] = []
    coverage = Coverage()
    with _observing(staging) as (documents, files):
        dataset = generate_dataset(
            seed=seed,
            out_dir=out_dir,
            personas=personas,
            claims_per_persona=claims_per_persona,
            country=Country.UA,
            train_fraction=0.5,
        )
    coverage.documents = len(documents)
    coverage.files = len(files)

    for observed_file in files:
        boxes = _labelled(observed_file["shipped_boxes"])
        coverage.boxes_seen += len(boxes)
        findings += survival_findings(
            _file_id(observed_file, documents),
            clean=observed_file["clean"],
            boxes=_labelled(observed_file["boxes"]),
            shipped=observed_file["shipped"],
            shipped_boxes=boxes,
            capture=observed_file["capture"],
            seed=observed_file["seed"],
            floor=floor,
            coverage=coverage,
        )

    if evidence:
        with Renderer() as renderer:
            for observed in documents:
                clean = cv2.imread(str(observed.clean_path))
                findings += evidence_findings(
                    observed.doc_id,
                    renderer=renderer,
                    html=observed.html,
                    clean=clean,
                    boxes=_labelled(observed.boxes),
                    staging=staging,
                    coverage=coverage,
                )
    return findings, _with_manifest(coverage, dataset)


def _file_id(observed_file: dict, documents: list[_Observed]) -> str:
    """Which document (or documents) the file under measurement carries.

    Matched on the PIXELS the degrader was handed rather than on call order: a bundled claim renders
    two documents and then a composition, so the order of the two streams differs from the order of
    the files. A composed file matches no single render and is named after the documents whose boxes
    it carries.
    """
    for observed in documents:
        clean = cv2.imread(str(observed.clean_path))
        if clean is not None and np.array_equal(clean, observed_file["clean"]):
            return observed.doc_id
    carried = sorted(
        {name.split("__")[1] for name in observed_file["boxes"] if name.startswith("__doc_")}
    )
    return "+".join(f"doc_{index}" for index in carried) if carried else "unmatched-file"


def _with_manifest(coverage: Coverage, dataset) -> Coverage:
    """Attach nothing; check the denominator. A run that observed fewer documents than it wrote is
    an unhooked instrument, and the caller turns that into a finding."""
    coverage.unmeasured["run: documents in the manifest"] = len(dataset.documents)
    return coverage


def floors_named(override: float | None) -> str:
    if override is not None:
        return f"{override:.3f} on every channel (override)"
    return ", ".join(
        f"{capture.value} {floor:.3f}" for capture, floor in LEGIBILITY_FLOORS.items()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pixel-label-gate",
        description=(
            "Hold every labelled bounding box of a generated corpus to two statements: its "
            "pixels are that field's own, and the capture channel left them readable."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--personas", type=int, default=4)
    parser.add_argument("--claims-per-persona", type=int, default=2)
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="one peak-correlation floor for every channel, for exploring a distribution; the "
        "shipped statement is the per-channel table in LEGIBILITY_FLOORS",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="run the survival statement only; the evidence statement costs three renders a page",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the corpus goes; a temporary directory by default, because the gate reads it "
        "rather than ships it",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        findings, coverage = scan_run(
            seed=args.seed,
            personas=args.personas,
            claims_per_persona=args.claims_per_persona,
            out_dir=args.out if args.out is not None else Path(tmp) / "corpus",
            staging=staging,
            floor=args.floor,
            evidence=not args.no_evidence,
        )

    written = coverage.unmeasured.pop("run: documents in the manifest", 0)
    print(f"pixel↔label gate — seed {args.seed}")
    print(coverage.report())
    print(f"  documents in the manifest {written}")
    if coverage.documents < written:
        findings.append(
            Finding(
                "instrument_unhooked",
                "run",
                "",
                f"{coverage.documents} document(s) observed against {written} written; the gate "
                "did not see the whole run and its silence means nothing",
            )
        )
    if not findings:
        print(f"\nno findings. floors: {floors_named(args.floor)}")
        return 0
    print(f"\n{len(findings)} finding(s):")
    for finding in findings:
        print(f"  {finding}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
