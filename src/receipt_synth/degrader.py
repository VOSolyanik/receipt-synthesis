"""Degradation: what happens to a document between being issued and being submitted.

Real evidence arrives as a screenshot, a photo taken at an angle, or a flatbed scan. The
label is invariant under all of it — nothing about *what a document says* changes when it
is photographed badly — so one rendered document yields several training examples.

Two libraries, with a strict division of labour:

* **Augraphy** applies paper, ink and sensor effects. Only non-geometric ones are used
  here, so it never has to move a bounding box. ⚠️ And only deterministic ones: an Augraphy
  effect is free to ignore `random_seed`, and one that fits this dataset well does. See
  `_paper_pipeline`.
* **Albumentations** owns every geometric operation, and the coordinates travel through its
  own keypoint pipeline — not its bbox pipeline, for the reason `carry_boxes` gives.

The split is the point. Perspective, rotation and padding all invalidate coordinates, and
having exactly one library responsible for transforming them is what keeps annotations
aligned. `carry_boxes` is that one route, and it is the function the known-answer gate in
`tests/test_bbox_gate.py` exercises — deliberately, so the gate measures the production
carrier rather than a pipeline assembled inside a test.

🔴 why a bbox/geometry desync gets its own gate rather than a test at the end. A box that
does not follow its pixels corrupts the ground truth of every image in the dataset and is
invisible in every metric: downstream it reads as poor extraction accuracy, not as a
coordinate defect. And with three channels in place one desync produces three different
wrong answers, so telling "the transform is wrong" from "this channel is wrong" stops being
cheap. Hence: one channel, one transform, one hand-computed answer, before anything fanned
out.

Four channels, and one of them applies nothing. `Capture` is read as what its docstring says —
ways a document reached the verifier — and `digital_pdf` is the way that damages nothing: the
original file, submitted as generated. This module's contract for it is the identity, asserted
by test rather than implied: same pixels, same boxes. (An earlier version of this header argued
the opposite — that the enum holds only kinds of damage and the undamaged case is the absence of
a channel. That reading kept the consumer's fourth `medium` value unrepresentable and was
reversed at contract version 35; RC-11's naming-and-home question is what remains open.)

which documents take which channel is not this module's question: the mix per document class is
`capture_mix` in config/policy.yaml — a label-sizing decision — and the heavy artefacts' per-
document probabilities are `degradation` in config/generation.yaml, a pixels-only one. This
module owns the recipes; both dials live beside their own kind.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
from augraphy import (
    AugraphyPipeline,
    BrightnessTexturize,
    LightingGradient,
    NoiseTexturize,
    NoisyLines,
    ShadowCast,
    SubtleNoise,
)

from receipt_synth.config import degradation_p
from receipt_synth.schemas import BBox, Capture

# Augraphy seeds OpenCV through `cv2.setRNGSeed`, which takes a signed C int. Callers
# derive seeds from a 64-bit generator, so the seed is folded into range here rather than
# at every call site — a limit of one library is not something the pipeline should have
# to know about.
_C_INT_MAX = 2**31 - 1

# The surface a photographed document lies on: mid grey rather than black, so the paper
# has an edge a detector could find. A constant rather than a draw — the colour of somebody's
# desk is not a parameter this dataset makes a claim about.
_SURFACE = 128


@dataclass(frozen=True)
class DegradedDocument:
    image: np.ndarray
    field_bboxes: dict[str, BBox]


def _paper_pipeline(capture: Capture, seed: int) -> AugraphyPipeline | None:
    """Paper, ink and sensor character for one channel — `None` where a channel has none.

    Each channel gets the artefacts of the device that produced it, and only those:

    * `digital_pdf` — the original file. Nothing: no device produced it, so there is no
      device character to apply, and `None` is that stated rather than an empty pipeline
      run for show.
    * `screenshot` — a screen capture of an electronic document. `None` too, and that is a
      correction rather than a variant: the pixels were never light on paper and never
      crossed a sensor, so the paper grain and the sensor noise this channel used to carry
      were physically impossible artefacts — a screenshot showing paper texture is a
      composite no capture produces. What a screenshot does lose is compression, which is
      geometry-phase (`_geometry`) and stays.
    * `photo` — a hand-held camera over paper on a desk. Paper texture and sensor noise
      always (they are the physics of a camera over paper); uneven illumination and a cast
      shadow at the rates config/generation.yaml declares (`degradation.photo`) — common,
      and no longer certain, because a defect carried by 100% of a channel is a property of
      the channel constant rather than evidence about documents.
    * `scan` — a flatbed. Evenly lit by construction, so no lighting gradient; the faint
      transport streaking fires at `degradation.scan.streak_p` — a clean flatbed produces
      none — over the paper texture and mild noise that always ride a scan.

    Which heavy effects fire is drawn from `seed` in a fixed order, through this function's
    own `random.Random` — one decision stream per document, separate from the pixel noise
    Augraphy derives from the same integer — so a run stays byte-reproducible under
    `--seed` and a test can pin the inclusion rate against the config value.

    ⚠️ the intensity settings are plausible rather than measured. No corpus of real captures
    was available to fit them against, so they are a designer's choice of what each device
    does; what the config calibrates is how often each heavy artefact occurs, not how hard
    it hits. A consumer must not read the difficulty of a channel here as an estimate of the
    difficulty of that channel in the field.

    🔴 `DirtyRollers` is deliberately absent from the scan recipe, and it is the effect that
    names what a scanner does. It ignores `random_seed`: two pipelines built with the same
    seed produce different pixels, measured directly rather than suspected. Determinism under
    `--seed` is an invariant of this generator, so the effect cannot be used however well it
    fits, and `NoisyLines` carries the streaking instead. Every other effect named in this
    function was checked the same way and is reproducible.
    """
    if capture in (Capture.SCREENSHOT, Capture.DIGITAL_PDF):
        return None
    include = random.Random(seed)
    if capture is Capture.PHOTO:
        paper = [
            NoiseTexturize(sigma_range=(3, 8), turbulence_range=(3, 9), p=1.0),
            BrightnessTexturize(texturize_range=(0.85, 0.98), deviation=0.06, p=1.0),
        ]
        post: list[object] = []
        # Fixed order — lighting, then shadow — so the seed means the same thing on every
        # build of the same document.
        if include.random() < degradation_p("photo", "lighting"):
            post.append(
                LightingGradient(
                    light_position=None, direction=90, max_brightness=250, min_brightness=0,
                    mode="gaussian", transparency=0.5, p=1.0,
                )
            )
        if include.random() < degradation_p("photo", "shadow"):
            post.append(
                ShadowCast(
                    shadow_side="random", shadow_vertices_range=(2, 3),
                    shadow_width_range=(0.5, 0.8), shadow_height_range=(0.5, 0.8),
                    shadow_color=(0, 0, 0), shadow_opacity_range=(0.3, 0.4),
                    shadow_iterations_range=(1, 2), shadow_blur_kernel_range=(101, 301),
                    p=1.0,
                )
            )
        post.append(SubtleNoise(subtle_range=14, p=1.0))
    elif capture is Capture.SCAN:
        paper = [
            NoiseTexturize(sigma_range=(2, 6), turbulence_range=(3, 7), p=1.0),
            BrightnessTexturize(texturize_range=(0.92, 1.0), deviation=0.04, p=1.0),
        ]
        post = []
        if include.random() < degradation_p("scan", "streak"):
            post.append(
                NoisyLines(
                    noisy_lines_direction=0,  # horizontal: the direction a sheet travels
                    noisy_lines_location="random",
                    noisy_lines_number_range=(2, 5),
                    noisy_lines_thickness_range=(1, 1),
                    noisy_lines_random_noise_intensity_range=(0.01, 0.04),
                    p=1.0,
                )
            )
        post.append(SubtleNoise(subtle_range=6, p=1.0))
    else:
        raise NotImplementedError(_unknown(capture))

    return AugraphyPipeline(
        ink_phase=[], paper_phase=paper, post_phase=post, random_seed=seed
    )


def _geometry(capture: Capture) -> list[A.BasicTransform]:
    """The geometry of one channel — and the reason the three differ is the device, not taste.

    * `digital_pdf` — none at all, not even compression: the original file, as generated. Its
      boxes still take this route, so the channel that changes nothing is measured by the same
      gate as the three that do.
    * `screenshot` — no geometry. A screen capture is axis-aligned by construction; there is no
      hand holding it and no sheet to lie crooked. What it does carry is the compression of
      whatever produced and re-sent it, which is the one loss this channel has.
    * `photo` — the paper is first padded onto a surface, because a photograph frames a
      document with the desk around it and a full-bleed render has no room to rotate into.
      Then a small perspective (the camera is not parallel to the page), a small rotation
      (nobody holds it square), the compression of a phone — always, being the physics of the
      device — and motion blur at `degradation.photo.motion_blur_p`: most phone shots of a
      still page are sharp, so blur is the minority case rather than the rule. Its inclusion is
      drawn by `A.Compose`'s own seeded stream, which the same `seed` drives.
    * `scan` — a rotation of a degree or so onto a narrow white margin, and nothing else. A
      flatbed is parallel to the page by construction; what it gets wrong is only how straight
      the sheet was laid.

    🔴 the padding is part of the geometry, not a cosmetic frame, and its size is the one
    setting here that was tuned rather than chosen. A render is full-bleed: the viewport is
    fitted to the document, so the paper's edge is the image's edge and there is nothing for a
    rotation or a perspective to turn into. Too little padding and every photographed document
    loses content — measured at 60 of 60 with a 48-pixel margin, which would make
    `content_complete` a constant `false` and the flag worthless. Too much and none ever does,
    which makes it a constant `true` and worthless the other way.

    The measurement that settled it — 60 seeds on each of four page shapes, from a 58 mm roll
    to an A4 statement, against a text extent inset 3% and 2% from the paper's edges:

        absolute 48 px    29 / 25 / 37 / 15 of 60
        6–10% per side    12 / 11 / 11 / 12 of 60
        10–16% per side    4 /  3 /  3 /  4 of 60

    The middle row is chosen, and the top row is the reason the margin is a proportion: at a
    fixed 48 pixels the crop rate runs from a quarter to nearly two thirds depending on how big
    the paper is, so the same setting would mean a different photographer for each archetype.
    A photograph that cuts an edge is then an outcome the label reports, at about one document
    in five, instead of a property of the render that nobody chose.

    `scan` is padded too, but only slightly and to white: a flatbed lays the whole sheet on the
    platen, so it is not expected to crop at all, and it does not.
    """
    if capture is Capture.DIGITAL_PDF:
        return [
            # The identity, written down — see the screenshot branch for why `A.NoOp` rather
            # than an empty list. No compression either: the consumer receives the file the
            # generator wrote.
            A.NoOp(p=1.0),
        ]
    if capture is Capture.SCREENSHOT:
        return [
            # The identity, written down. "No geometry" is a decision about this channel, and
            # `A.NoOp` is how the pipeline states it rather than leaves it implicit. It is not
            # decoration either: Albumentations warns, correctly, when a compose is given
            # coordinates and holds no transform that touches them, and a warning once per
            # document is how a real one stops being read.
            A.NoOp(p=1.0),
            A.ImageCompression(compression_type="jpeg", quality_range=(70, 88), p=1.0),
        ]
    if capture is Capture.PHOTO:
        return [
            # A proportion and not a pixel count, sampled per side. A fixed margin is a
            # different amount of framing on a 58 mm receipt and on an A4 invoice, so a fixed
            # margin would make the crop rate a function of the archetype's paper size — a
            # dependency nothing in the design intends and nobody would look for.
            A.CropAndPad(
                percent=(0.06, 0.10), keep_size=False, sample_independently=True,
                border_mode=cv2.BORDER_CONSTANT, fill=_SURFACE, p=1.0,
            ),
            A.Perspective(
                scale=(0.01, 0.035), keep_size=True, fit_output=False,
                border_mode=cv2.BORDER_CONSTANT, fill=_SURFACE, p=1.0,
            ),
            A.Affine(
                rotate=(-3.0, 3.0), border_mode=cv2.BORDER_CONSTANT, fill=_SURFACE, p=1.0
            ),
            A.MotionBlur(blur_limit=(3, 5), p=degradation_p("photo", "motion_blur")),
            A.ImageCompression(compression_type="jpeg", quality_range=(45, 75), p=1.0),
        ]
    if capture is Capture.SCAN:
        return [
            A.CropAndPad(
                percent=(0.02, 0.04), keep_size=False, sample_independently=True,
                border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0,
            ),
            A.Affine(rotate=(-1.5, 1.5), border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
            A.ImageCompression(compression_type="jpeg", quality_range=(60, 85), p=1.0),
        ]
    raise NotImplementedError(_unknown(capture))


def _unknown(capture: object) -> str:
    """The refusal both recipe tables share.

    Typed as `object` rather than `Capture` on purpose: the case it exists for is a channel
    that is not a member — a new enum value with no branch, or a caller passing a string — and
    a signature that promised `Capture` would make the raise look unreachable to a reader.
    """
    return (
        f"capture channel {getattr(capture, 'value', capture)!r} has no recipe; add one "
        "deliberately rather than letting a channel fall through to another's artefacts"
    )


def corners_of(box: BBox) -> tuple[tuple[float, float], ...]:
    """The four corner pixels of a box, clockwise from the top left.

    🔴 the corner pixels, not the corner edges, and the difference is a whole pixel on every
    box in the dataset. A COCO box's far edge is exclusive — (20, 30, 40, 10) covers columns
    20 … 59, and 60 is the boundary after the last one — while a keypoint names a pixel.
    Handing 60 to the keypoint pipeline asks it where a boundary went, and it answers about
    pixel 60, which belongs to whatever is next to the box.

    Measured on the gate rather than reasoned about: under a horizontal flip of a 200-wide
    image the library maps a keypoint x to 199 - x, so the edge 60 came back as 139 and the
    box was rebuilt one column left of its ink. The inclusive corner 59 gives 140, which is
    where the pixels are.
    """
    x, y, w, h = box
    far_x, far_y = x + max(w - 1, 0), y + max(h - 1, 0)
    return ((x, y), (far_x, y), (far_x, far_y), (x, far_y))


def hull_of(points: list[tuple[float, float]]) -> BBox:
    """The smallest axis-aligned box containing some points, in the COCO convention.

    The `+ 1` is the inverse of `corners_of`: the points are the first and last pixel, and a
    box covering columns 140 … 179 is 40 wide. Rounded first and the size derived from the
    rounded corners, so `x + width` is exactly the far edge a frame comparison uses.
    """
    left, right = round(min(p[0] for p in points)), round(max(p[0] for p in points))
    top, bottom = round(min(p[1] for p in points)), round(max(p[1] for p in points))
    return (float(left), float(top), float(right - left + 1), float(bottom - top + 1))


def carry_boxes(
    transforms: list[A.BasicTransform],
    image: np.ndarray,
    field_bboxes: dict[str, BBox],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, BBox]]:
    """🔴 the one route coordinates take through geometry. Every box in the dataset comes
    out of this function, and nothing else in the generator moves a coordinate.

    Public rather than private because the known-answer gate drives it directly with a
    transform whose answer is computed by hand. A gate that built its own `A.Compose` would
    prove that Albumentations is correct — which is not the thing at risk.

    🔴 the boxes travel as four corner keypoints, not as boxes, and that is not a stylistic
    choice. Albumentations' bbox pipeline clips every box to the image in `postprocess`, on
    every path — `filter_bboxes` returns `clipped_bboxes[mask]`, and neither `clip=False` nor
    `filter_invalid_bboxes=False` nor `check_each_transform=False` reaches it. Measured, not
    assumed: a box at (20, 30, 40, 10) translated 30 pixels left came back as
    (0, 30, 30, 10) under all three of those settings.

    That value is a lie of exactly the kind this module must not tell. It is what a document
    that lost a tenth of its content looks like and what a document that lost nothing looks
    like, so a completeness measurement built on it would return `True` for every image in
    the corpus and nothing would ever go red. The keypoint pipeline with
    `remove_invisible=False` preserves the true coordinate, negative or past the edge, and
    the box is rebuilt from the four corners here.

    A box in the result may lie partly outside the image. That is information, not damage: a
    consumer that wants an index into the image clips it, and the clipped form can be derived
    from this one while this one cannot be derived from the clipped form.

    Under a rotation or a perspective the rebuilt box is the axis-aligned hull of the four
    transformed corners, so it is a little larger than the ink it covers. That is inherent to
    an axis-aligned annotation of a rotated rectangle, and is what Albumentations' own bbox
    handling does too; the alternative is a quadrilateral, which the label schema does not
    carry.
    """
    names = list(field_bboxes)
    corners: list[list[float]] = []
    owners: list[str] = []
    for name in names:
        corners += [list(point) for point in corners_of(field_bboxes[name])]
        owners += [name] * 4

    pipeline = A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["field_names"],
            # The whole reason this pipeline is keypoints rather than boxes — see above.
            remove_invisible=False,
        ),
        seed=seed,
    )
    result = pipeline(image=image, keypoints=corners, field_names=owners)

    # Regrouped by the returned labels rather than by position: a transform is allowed to
    # reorder its targets, and pairing by index would silently mix one field's corners with
    # another's — which would look like a box in a plausible-but-wrong place rather than
    # like an error.
    grouped: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    for name, point in zip(result["field_names"], result["keypoints"], strict=True):
        grouped[str(name)].append((float(point[0]), float(point[1])))

    moved: dict[str, BBox] = {}
    for name, points in grouped.items():
        if len(points) != 4:
            raise ValueError(
                f"field {name!r} came back with {len(points)} corners instead of 4; the "
                "keypoint pipeline dropped or duplicated one and the box cannot be rebuilt"
            )
        moved[name] = hull_of(points)
    return result["image"], moved


def degrade(
    image: np.ndarray,
    field_bboxes: dict[str, BBox],
    *,
    seed: int,
    capture: Capture = Capture.SCREENSHOT,
    compress: bool = True,
) -> DegradedDocument:
    """Apply one capture channel to a rendered document and its bounding boxes.

    🔴 `compress=False` is the legibility gate's reference and nothing in the pipeline passes it.
    It runs the channel the caller asked for with its JPEG step left out — same paper phase, same
    padding, same perspective, same rotation, same blur, same seed — so that a comparison between
    the two answers one question: what the compression cost the marks inside a labelled box. A
    reference rendered without the geometry would answer a different question (what the geometry
    cost), and a reference rendered by another route would measure this module against a second
    implementation of it. See `tools/pixel_label_gate.py`, which is the only caller.

    ⚠️ the assumption that makes the omission safe is that `A.ImageCompression` is last in every
    channel's recipe, so dropping it removes no draw any earlier transform depends on and the boxes
    come out where the shipped run put them. The gate does not take that on trust: it compares the
    two box sets and reports a mismatch as a defect of its own instrument rather than of the
    corpus.
    """
    seed %= _C_INT_MAX
    paper = _paper_pipeline(capture, seed)
    degraded = paper(image) if paper is not None else image
    transforms = _geometry(capture)
    if not compress:
        transforms = [t for t in transforms if not isinstance(t, A.ImageCompression)]
    moved_image, moved_boxes = carry_boxes(transforms, degraded, field_bboxes, seed=seed)
    return DegradedDocument(image=moved_image, field_bboxes=moved_boxes)


def clipped_edges(box: BBox, width: int, height: int) -> tuple[str, ...]:
    """Which edges of the image a box crosses — empty when it is wholly on the page.

    The arithmetic is deliberately trivial and deliberately here rather than in the caller:
    it is the definition of what "the content survived" means, and a definition that lives
    beside the transform that can break it is one a reader can check in one place.

    Named edges rather than a bare boolean because a consumer filtering a corpus wants to know
    what left: a document missing its bottom is a different training example from one missing
    its left margin, and reconstructing that from a box and an image size is work every
    consumer would otherwise repeat.

    A box flush with the far edge is inside. Its last pixel is then column `width - 1`, which
    exists; `x + width_of_box` equalling the frame width is the boundary after it. That
    off-by-one is the likeliest defect in the whole function, so the gate pins the flush case
    explicitly rather than only the crossing ones.
    """
    x, y, w, h = box
    edges = []
    if x < 0:
        edges.append("left")
    if y < 0:
        edges.append("top")
    if x + w > width:
        edges.append("right")
    if y + h > height:
        edges.append("bottom")
    return tuple(edges)
