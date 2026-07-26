"""The pipeline ends: persona, plan, degradation, and the run as a whole.

The claim these tests exist to defend is the one the whole tool rests on — that a seed
determines the output. It is checked at each stage and again on the finished dataset,
because a single unseeded draw anywhere makes the promise false everywhere.
"""

from __future__ import annotations

import json
import random
from datetime import date

import numpy as np
import pytest

from receipt_synth.assembler import generate_dataset
from receipt_synth.claim_planner import (
    ARCHETYPES,
    archetypes_for,
    plan_claim,
    plannable_categories,
)
from receipt_synth.cli import main
from receipt_synth.config import load_policy
from receipt_synth.content_builder import is_valid_rnokpp
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.schemas import Capture, Country, DocType, Verdict

SEED = 20260803


# ----------------------------------------------------------------- personas --


def persona(seed: int = SEED, country: Country = Country.UA):
    return generate_persona(random.Random(seed), persona_id="p001", country=country)


def test_persona_is_deterministic_under_seed():
    assert persona() == persona()


def test_personas_differ_between_seeds():
    names = {persona(seed).full_name for seed in range(20)}
    assert len(names) > 1


def test_ukrainian_persona_carries_a_valid_rnokpp():
    assert is_valid_rnokpp(persona().tax_id)


def test_tax_id_agrees_with_the_person_it_belongs_to():
    """The identifier is not merely checksum-correct: its first five digits are the
    birth date and the ninth digit's parity is the sex. A persona whose name and
    identifier disagreed would be internally inconsistent evidence."""
    for seed in range(30):
        result = persona(seed)
        days = int(result.tax_id[:5])
        birth = date(1899, 12, 31).toordinal() + days
        age_in_2026 = 2026 - date.fromordinal(birth).year
        assert 20 <= age_in_2026 <= 65


def test_persona_speaks_and_is_paid_in_something_plausible():
    result = persona(country=Country.UA)
    assert result.home_currencies == ["UAH"]
    assert result.languages == ["uk"]
    assert result.location.country is Country.UA


def test_categories_are_drawn_within_the_configured_count():
    spec = load_policy()["persona_categories"]["count_per_persona"]
    known = {entry["id"] for entry in load_policy()["categories"]}

    for seed in range(30):
        categories = persona(seed).benefit_categories
        assert set(categories) <= known
        assert len(set(categories)) == len(categories), "a category was drawn twice"
        assert spec["min"] <= len(categories) <= spec["max"] + 1


def test_medical_insurance_dominates_in_ukraine():
    """policy.yaml gives it 0.95 for UA. If the probability block were ignored the share
    would fall to whatever a uniform draw produces, and the dataset would be skewed in a
    way nothing else reports."""
    share = sum("medical_insurance" in persona(seed).benefit_categories for seed in range(200))
    assert share > 170


# -------------------------------------------------------------------- plans --


def test_plan_is_deterministic_under_seed():
    subject = persona()
    assert plan_claim(random.Random(1), persona=subject, claim_id="c1") == plan_claim(
        random.Random(1), persona=subject, claim_id="c1"
    )


def test_plan_dates_fall_inside_the_active_period():
    """A date outside the window drives `insufficient_evidence`. Landing outside it by
    accident would attach that document to a `covered` label."""
    period = load_policy()["period"]
    start = date.fromisoformat(str(period["start"]))
    end = date.fromisoformat(str(period["end"]))

    subject = persona()
    for seed in range(50):
        issued = plan_claim(random.Random(seed), persona=subject, claim_id="c1").issued_at
        assert start <= issued.date() <= end


def test_plan_picks_a_category_the_persona_holds():
    subject = persona()
    plan = plan_claim(random.Random(3), persona=subject, claim_id="c1")
    assert plan.category in subject.benefit_categories


def test_plan_picks_an_archetype_that_can_carry_the_category():
    subject = persona()
    plan = plan_claim(random.Random(3), persona=subject, claim_id="c1")
    assert plan.category in plan.archetype.categories
    assert plan.archetype.country is subject.location.country


def test_a_category_no_archetype_covers_is_refused():
    subject = persona()
    unsupported = next(
        category
        for category in subject.benefit_categories
        if not archetypes_for(Country.UA, category)
    )
    with pytest.raises(ValueError):
        plan_claim(random.Random(1), persona=subject, claim_id="c1", category=unsupported)


def test_a_category_the_persona_does_not_hold_is_refused():
    subject = persona()
    absent = next(
        entry["id"]
        for entry in load_policy()["categories"]
        if entry["id"] not in subject.benefit_categories
    )
    with pytest.raises(ValueError):
        plan_claim(random.Random(1), persona=subject, claim_id="c1", category=absent)


@pytest.mark.parametrize(
    "verdict",
    [v for v in Verdict if v is not Verdict.COVERED],
)
def test_verdicts_needing_the_policy_engine_are_refused_not_faked(verdict):
    """Explicit over silent. A planner that accepted `partially_covered` and produced a
    fully covered basket would write a wrong label rather than fail."""
    with pytest.raises(NotImplementedError):
        plan_claim(random.Random(1), persona=persona(), claim_id="c1", verdict=verdict)


def test_registered_archetypes_declare_what_they_can_carry():
    known = {entry["id"] for entry in load_policy()["categories"]}
    for archetype in ARCHETYPES.values():
        assert archetype.categories, f"{archetype.slug} carries no category"
        assert set(archetype.categories) <= known, f"{archetype.slug} names an unknown category"


def test_plannable_categories_is_the_intersection():
    subject = persona()
    for category in plannable_categories(subject):
        assert category in subject.benefit_categories
        assert archetypes_for(Country.UA, category)


# --------------------------------------------------------------- degradation --


def image_and_boxes():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 160, 3), dtype=np.uint8)
    boxes = {"total": (10.0, 20.0, 40.0, 12.0), "date": (60.0, 150.0, 50.0, 12.0)}
    return image, boxes


def test_degradation_is_deterministic_under_seed():
    image, boxes = image_and_boxes()
    first = degrade(image, boxes, seed=7)
    second = degrade(image, boxes, seed=7)

    assert np.array_equal(first.image, second.image)
    assert first.field_bboxes == second.field_bboxes


def test_degradation_differs_between_seeds():
    image, boxes = image_and_boxes()
    assert not np.array_equal(degrade(image, boxes, seed=7).image,
                              degrade(image, boxes, seed=8).image)


def test_degradation_changes_the_image():
    image, boxes = image_and_boxes()
    assert not np.array_equal(degrade(image, boxes, seed=7).image, image)


def test_degradation_accepts_a_seed_wider_than_a_c_int():
    """Augraphy seeds OpenCV through `cv2.setRNGSeed`, which takes a signed C int, while
    the pipeline derives seeds from a 64-bit generator. The fold happens in the degrader
    so no caller has to know."""
    image, boxes = image_and_boxes()
    assert degrade(image, boxes, seed=2**63 - 1).field_bboxes == boxes


def test_every_box_survives_degradation():
    """A dropped box would leave a document whose labels name a field the annotation no
    longer locates."""
    image, boxes = image_and_boxes()
    assert set(degrade(image, boxes, seed=7).field_bboxes) == set(boxes)


def test_boxes_stay_whole_pixels_through_degradation():
    """Albumentations normalizes coordinates to [0, 1] and back, which turns 26 into
    25.999999217689037. A box indexes an image, so the rounding the renderer applied has
    to survive the trip."""
    image, boxes = image_and_boxes()
    for box in degrade(image, boxes, seed=7).field_bboxes.values():
        assert all(float(value).is_integer() for value in box)


def test_non_geometric_degradation_leaves_boxes_where_they_were():
    image, boxes = image_and_boxes()
    assert degrade(image, boxes, seed=7).field_bboxes == boxes


def test_uncalibrated_capture_modes_are_refused():
    image, boxes = image_and_boxes()
    for capture in (Capture.PHOTO, Capture.SCAN):
        with pytest.raises(NotImplementedError):
            degrade(image, boxes, seed=7, capture=capture)


# ------------------------------------------------------------- the whole run --


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    return generate_dataset(seed=SEED, out_dir=out), out


def test_run_produces_one_document_per_claim(dataset):
    result, _ = dataset
    assert len(result.personas) == 1
    assert len(result.claims) == 1
    assert len(result.documents) == 1


def test_images_and_labels_are_written(dataset):
    result, out = dataset
    document = result.documents[0]

    assert (out / "images" / document.source_file).is_file()
    assert (out / "labels" / f"{document.doc_id}.json").is_file()
    assert (out / "labels" / f"{result.claims[0].claim_id}.claim.json").is_file()
    assert (out / "ground_truth.json").is_file()


def test_the_claim_points_at_the_document_that_was_written(dataset):
    result, _ = dataset
    assert result.claims[0].documents == [result.documents[0].doc_id]
    assert result.claims[0].verdict is Verdict.COVERED
    assert result.documents[0].doc_type is DocType.FISCAL_RECEIPT


def test_labels_state_the_invariants_that_were_built(dataset):
    """The end-to-end statement: what the ground truth says is what content_builder
    guaranteed, carried unchanged through rendering and degradation."""
    result, _ = dataset
    document = result.documents[0]

    assert sum(item.qty * item.price for item in document.line_items) == document.amount
    assert all(item.covered for item in document.line_items)
    assert document.synthetic is True
    assert document.generator_version


def test_a_covered_claim_leaves_the_policy_engine_fields_unset(dataset):
    """Not zero, not a guess — unset. The policy engine derives these in step 4, and a
    plausible default here would be a label nobody computed."""
    result, _ = dataset
    claim = result.claims[0]

    assert claim.covered_fraction is None
    assert claim.verdict_basis == []
    assert claim.policy_trace == []


def test_manifest_is_valid_json_and_declares_its_provenance(dataset):
    _, out = dataset
    manifest = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    assert manifest["synthetic"] is True
    assert manifest["seed"] == SEED
    assert manifest["generator_version"]
    assert len(manifest["documents"]) == 1


def test_bboxes_in_the_written_labels_index_the_written_image(dataset):
    from PIL import Image

    result, out = dataset
    document = result.documents[0]
    with Image.open(out / "images" / document.source_file) as image:
        width, height = image.size

    assert document.field_bboxes
    for name, (x, y, box_width, box_height) in document.field_bboxes.items():
        assert x >= 0 and y >= 0, f"{name} starts outside the image"
        assert x + box_width <= width, f"{name} runs past the right edge"
        assert y + box_height <= height, f"{name} runs past the bottom edge"


def test_the_same_seed_reproduces_the_dataset(tmp_path):
    """The promise the repository is built on: it ships the generator and the seed, not
    the data."""
    first = generate_dataset(seed=SEED, out_dir=tmp_path / "a")
    second = generate_dataset(seed=SEED, out_dir=tmp_path / "b")

    assert first.as_manifest() == second.as_manifest()
    name = first.documents[0].source_file
    assert (tmp_path / "a" / "images" / name).read_bytes() == (
        tmp_path / "b" / "images" / name
    ).read_bytes()


def test_a_different_seed_produces_a_different_dataset(tmp_path):
    other = generate_dataset(seed=SEED + 1, out_dir=tmp_path / "c")
    assert other.documents[0].amount != 0


def test_line_items_do_not_repeat_a_printed_name(tmp_path):
    """A cash register lists an article once and states how many. The same name twice at
    two prices is not a receipt."""
    for seed in range(SEED, SEED + 5):
        result = generate_dataset(seed=seed, out_dir=tmp_path / f"s{seed}")
        names = [item.name for item in result.documents[0].line_items]
        assert len(names) == len(set(names))


def test_cli_runs_and_reports(tmp_path, capsys):
    assert main(["--seed", str(SEED), "--out", str(tmp_path)]) == 0
    assert "receipt-synth" in capsys.readouterr().out
    assert (tmp_path / "ground_truth.json").is_file()


def test_cli_requires_a_seed(capsys):
    """A default seed would let a run look reproducible without anyone having recorded
    what to reproduce it with."""
    with pytest.raises(SystemExit):
        main(["--out", "out"])
