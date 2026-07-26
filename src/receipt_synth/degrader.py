"""Degradation: what happens to a document between being issued and being submitted.

Real evidence arrives as a screenshot, a photo taken at an angle, or a flatbed scan. The
label is invariant under all of it — nothing about *what a document says* changes when it
is photographed badly — so one rendered document yields several training examples.

Two libraries, with a strict division of labour:

* **Augraphy** applies paper, ink and sensor effects. Only non-geometric ones are used
  here, so it never has to move a bounding box.
* **Albumentations** owns every geometric operation, and the boxes travel through its
  own bbox pipeline.

The split is the point. Perspective, rotation and crop all invalidate coordinates, and
having exactly one library responsible for transforming them is what keeps annotations
aligned once those steps arrive. This skeleton applies no geometry at all — the boxes
come back unchanged — but they are routed through the mechanism that will move them.
"""

from __future__ import annotations

from dataclasses import dataclass

import albumentations as A
import numpy as np
from augraphy import AugraphyPipeline, BrightnessTexturize, NoiseTexturize, SubtleNoise

from receipt_synth.schemas import BBox, Capture

# Augraphy seeds OpenCV through `cv2.setRNGSeed`, which takes a signed C int. Callers
# derive seeds from a 64-bit generator, so the seed is folded into range here rather than
# at every call site — a limit of one library is not something the pipeline should have
# to know about.
_C_INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class DegradedDocument:
    image: np.ndarray
    field_bboxes: dict[str, BBox]


def _paper_pipeline(seed: int) -> AugraphyPipeline:
    """Paper and sensor character. Procedural, and deliberately free of geometry.

    Every effect here is mild: the receipt is meant to look like a real capture of a
    thermal print, not like a damaged artifact. Heavier settings belong to the `photo`
    and `scan` modes.
    """
    return AugraphyPipeline(
        ink_phase=[],
        paper_phase=[
            NoiseTexturize(sigma_range=(2, 5), turbulence_range=(3, 7), p=1.0),
            BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.03, p=1.0),
        ],
        post_phase=[SubtleNoise(subtle_range=8, p=1.0)],
        random_seed=seed,
    )


def _geometry_pipeline(seed: int) -> A.Compose:
    """Capture artifacts, and the carrier for bounding boxes.

    Only JPEG compression for now. It is listed as a geometric-pipeline step even though
    it moves nothing, so that the boxes are already flowing through the transform that
    will later also rotate and warp them.
    """
    return A.Compose(
        [A.ImageCompression(compression_type="jpeg", quality_range=(70, 88), p=1.0)],
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height] in pixels — what the renderer emits
            label_fields=["field_names"],
            # A field box is kept however small or however far off the edge a transform
            # pushes it. Dropping one would leave a document whose labels claim a field
            # the annotation no longer locates.
            min_area=0.0,
            min_visibility=0.0,
        ),
        seed=seed,
    )


def degrade(
    image: np.ndarray,
    field_bboxes: dict[str, BBox],
    *,
    seed: int,
    capture: Capture = Capture.SCREENSHOT,
) -> DegradedDocument:
    """Apply one capture mode to a rendered document and its bounding boxes."""
    if capture is not Capture.SCREENSHOT:
        raise NotImplementedError(
            f"capture mode {capture.value!r} needs its own calibrated recipe; "
            "this skeleton ships only 'screenshot'"
        )

    seed %= _C_INT_MAX
    degraded = _paper_pipeline(seed)(image)

    names = list(field_bboxes)
    result = _geometry_pipeline(seed)(
        image=degraded,
        bboxes=[list(field_bboxes[name]) for name in names],
        field_names=names,
    )

    # Rebuilt from the returned labels rather than by zipping with `names`: a transform
    # is allowed to drop or reorder boxes, and pairing by position would silently
    # mislabel every field after the first one lost.
    #
    # Rounded, because Albumentations normalizes coordinates to [0, 1] and back, which
    # leaves 26 as 25.999999217689037. The renderer emits whole pixels for a reason — a
    # box indexes an image — and passing through the degrader must not quietly undo that.
    moved = {
        str(name): (
            float(round(box[0])),
            float(round(box[1])),
            float(round(box[2])),
            float(round(box[3])),
        )
        for name, box in zip(result["field_names"], result["bboxes"], strict=True)
    }
    return DegradedDocument(image=result["image"], field_bboxes=moved)
