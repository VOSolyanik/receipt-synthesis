"""The capture channel follows the class's declared mix, and the degrader honours the channel.

Two families of guard, and every statistical one is sized from the config value it guards rather
than from "the event occurs at least once" — the S11 lesson: a test must be sized so the mutation
it exists for would be seen, and re-sized automatically when the config moves.

* the draw: a zero-weight channel never comes out, every non-zero one does, the realized shares
  match the declared weights, and a screen-born archetype is a screenshot without consuming the
  stream;
* the recipes: `digital_pdf` is the identity, a screenshot has no paper or sensor phase, and the
  heavy photo/scan artefacts fire at the rate config/generation.yaml declares.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from receipt_synth.assembler import draw_capture
from receipt_synth.claim_planner import ARCHETYPES
from receipt_synth.config import capture_mix, degradation_p
from receipt_synth.degrader import _geometry, _paper_pipeline, degrade
from receipt_synth.schemas import Capture

MIXED_CLASSES = sorted(
    {a.doc_type for a in ARCHETYPES.values() if not a.screen_native},
    key=lambda t: t.value,
)

SCREEN_NATIVE = sorted(a.slug for a in ARCHETYPES.values() if a.screen_native)


def _free_archetype(doc_type):
    return next(
        a for a in ARCHETYPES.values() if a.doc_type is doc_type and not a.screen_native
    )


def test_every_registered_class_declares_a_mix():
    """A class an archetype can produce must have a row in policy.yaml's `capture_mix` — a
    missing row fails at the first draw, and this test moves that failure to the suite."""
    for doc_type in MIXED_CLASSES:
        mix = capture_mix(doc_type.value)
        assert set(mix) == {member.value for member in Capture}


@pytest.mark.parametrize("doc_type", MIXED_CLASSES, ids=lambda t: t.value)
def test_a_zero_weight_channel_is_never_drawn(doc_type):
    """⛔ weight zero means never, not rarely: `digital_pdf` on a till roll would be an undamaged
    electronic original of a document that has none.

    Sized against the mutation, which is the draw losing its conditioning and falling back to a
    uniform over the enum. Under it a forbidden channel comes out at 1/len(Capture) per draw, so
    `draws` is chosen to see that at least once with probability 1 - 1e-6 — and it re-derives
    itself when the enum grows.
    """
    mix = capture_mix(doc_type.value)
    forbidden = {channel for channel, weight in mix.items() if weight == 0.0}
    if not forbidden:
        pytest.skip(f"{doc_type.value} declares no zero-weight channel")
    mutation_rate = 1.0 / len(Capture)
    draws = math.ceil(math.log(1e-6) / math.log(1.0 - mutation_rate))
    archetype = _free_archetype(doc_type)
    rng = random.Random(20260805)
    drawn = {draw_capture(rng, archetype).value for _ in range(draws)}
    assert not (drawn & forbidden), (
        f"{doc_type.value} drew {sorted(drawn & forbidden)}, which its mix gives weight 0"
    )


@pytest.mark.parametrize("doc_type", MIXED_CLASSES, ids=lambda t: t.value)
def test_every_nonzero_channel_is_reachable(doc_type):
    """The dual of the zero-weight guard: a declared channel that never occurs is a label value
    the contract promises and no corpus contains. `draws` is sized from the thinnest non-zero
    weight in the config, so re-balancing the mix re-sizes the test."""
    mix = capture_mix(doc_type.value)
    expected = {channel for channel, weight in mix.items() if weight > 0.0}
    w_min = min(mix[channel] for channel in expected)
    draws = math.ceil(math.log(1e-6) / math.log(1.0 - w_min))
    archetype = _free_archetype(doc_type)
    rng = random.Random(20260805)
    drawn = {draw_capture(rng, archetype).value for _ in range(draws)}
    assert drawn == expected


@pytest.mark.parametrize("doc_type", MIXED_CLASSES, ids=lambda t: t.value)
def test_realized_shares_match_the_declared_mix(doc_type):
    """The walk over cumulative weights realizes the declared distribution — a guard against an
    off-by-one in the cumulation, which would shift every share while every channel still
    appears. Three-sigma binomial bounds per channel at n = 4000."""
    mix = capture_mix(doc_type.value)
    archetype = _free_archetype(doc_type)
    rng = random.Random(20260805)
    n = 4000
    counts = {channel.value: 0 for channel in Capture}
    for _ in range(n):
        counts[draw_capture(rng, archetype).value] += 1
    for channel, weight in mix.items():
        sigma = math.sqrt(weight * (1.0 - weight) / n)
        assert abs(counts[channel] / n - weight) <= max(3 * sigma, 1e-9), (
            f"{doc_type.value}.{channel}: {counts[channel] / n:.4f} against declared {weight}"
        )


def test_a_screen_native_archetype_is_a_screenshot_and_consumes_no_randomness():
    """🔴 the channel is a fact, not a draw, for a page that exists only on a screen — and the
    stream must not move: a constant that burned a uniform would shift every draw after it, which
    is exactly the kind of silent reordering seeded generation exists to forbid."""
    assert SCREEN_NATIVE, "the registry lost its screen-native archetypes; update this test"
    for slug in SCREEN_NATIVE:
        rng = random.Random(7)
        untouched = random.Random(7)
        assert draw_capture(rng, ARCHETYPES[slug]) is Capture.SCREENSHOT
        assert rng.random() == untouched.random(), "the constant consumed the stream"


def _page(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((240, 180, 3), 255, dtype=np.uint8)
    image[40:200, 20:160] = rng.integers(0, 255, (160, 140, 3), dtype=np.uint8)
    return image


def test_digital_pdf_is_the_identity():
    """🔴 the channel's whole meaning: the consumer receives the file the generator wrote. Same
    pixels, same boxes — not "close", equal."""
    image = _page()
    boxes = {"total": (30.0, 50.0, 60.0, 20.0), "date": (10.0, 10.0, 40.0, 12.0)}
    result = degrade(image, boxes, seed=99, capture=Capture.DIGITAL_PDF)
    assert np.array_equal(result.image, image)
    assert result.field_bboxes == boxes


def test_a_screenshot_has_no_paper_and_no_sensor_phase():
    """⛔ A screenshot's pixels were never light on paper and never crossed a sensor, so its
    recipe holds no Augraphy phase at all — compression is its one loss, and it rides the
    geometry pipeline."""
    assert _paper_pipeline(Capture.SCREENSHOT, 42) is None
    assert _paper_pipeline(Capture.DIGITAL_PDF, 42) is None
    names = [type(t).__name__ for t in _geometry(Capture.SCREENSHOT)]
    assert names == ["NoOp", "ImageCompression"]


@pytest.mark.parametrize(
    ("channel", "effect", "augmentation"),
    [
        (Capture.PHOTO, "lighting", "LightingGradient"),
        (Capture.PHOTO, "shadow", "ShadowCast"),
        (Capture.SCAN, "streak", "NoisyLines"),
    ],
)
def test_heavy_artefacts_fire_at_the_declared_rate(channel, effect, augmentation):
    """The per-effect probability is the config's, not 1.0 and not 0.0.

    Sized from the config value: the two mutations this guards are "p ignored, always on" and
    "always off", and under either the realized count leaves the three-sigma band around p long
    before n = 600 — while at the declared p the band holds. Inclusion is read off the built
    pipeline's own phase list, which is what the degrader actually runs.
    """
    p = degradation_p(channel.value, effect)
    assert 0.0 < p < 1.0, (
        f"degradation.{channel.value}.{effect}_p is {p}; at 0 or 1 the artefact stops being "
        "drawn and this test must be rethought, not skipped"
    )
    n = 600
    fired = 0
    for seed in range(n):
        pipeline = _paper_pipeline(channel, seed)
        assert pipeline is not None
        names = [type(a).__name__ for a in pipeline.post_phase.augmentations]
        fired += augmentation in names
    sigma = math.sqrt(p * (1.0 - p) / n)
    assert abs(fired / n - p) <= 3 * sigma, (
        f"{channel.value}.{effect}: fired {fired}/{n} against declared p={p}"
    )


def test_the_inclusion_draw_is_deterministic_per_seed():
    """Same seed, same recipe — the inclusion stream is part of the seed's meaning."""
    for seed in (0, 7, 12345):
        first = _paper_pipeline(Capture.PHOTO, seed)
        second = _paper_pipeline(Capture.PHOTO, seed)
        assert [type(a).__name__ for a in first.post_phase.augmentations] == [
            type(a).__name__ for a in second.post_phase.augmentations
        ]


def test_an_app_screen_document_in_a_run_is_always_a_screenshot(multi_claim_dataset_capture):
    """The integration half of the screen-native guard: through the whole assembler path, a
    document whose box vocabulary is the app's (`merchant_descriptor` / `screen_title`) carries
    `capture: screenshot`. A unit test on `draw_capture` cannot see an assembler that stopped
    calling it.

    ⚠️ unbundled documents only (`file_region is None`), by the v29 rule this test must not
    fight: a file is captured once, so an app screen bundled under an invoice inherits the
    file's channel — a screenshot pasted into the claimant's merged submission, which is the
    composite's carriage and not the screen's own.
    """
    result, _ = multi_claim_dataset_capture
    app_documents = [
        document
        for document in result.documents
        if {"merchant_descriptor", "screen_title"} & set(document.field_bboxes)
        and document.file_region is None
    ]
    assert app_documents, "the run drew no unbundled app rendering; enlarge the fixture"
    assert {d.capture for d in app_documents} == {Capture.SCREENSHOT}


def test_paper_classes_in_a_run_never_arrive_as_digital_pdf(multi_claim_dataset_capture):
    """The integration half of the zero-weight guard, through the whole assembler path."""
    result, _ = multi_claim_dataset_capture
    paper_only = {"fiscal_receipt", "non_fiscal_receipt"}
    offenders = [
        document.doc_id
        for document in result.documents
        if document.doc_type.value in paper_only
        and document.capture is Capture.DIGITAL_PDF
    ]
    assert not offenders


@pytest.fixture(scope="module")
def multi_claim_dataset_capture(tmp_path_factory):
    """A run of its own rather than test_pipeline's fixture: module-scoped reuse keeps the two
    integration guards to one render, and the seed is fixed apart from seed so a change there
    does not silently re-size these tests."""
    from receipt_synth.assembler import generate_dataset

    out = tmp_path_factory.mktemp("capture-mix")
    return (
        generate_dataset(
            seed=20260805, out_dir=out, train_fraction=0.5, personas=4, claims_per_persona=8
        ),
        out,
    )
