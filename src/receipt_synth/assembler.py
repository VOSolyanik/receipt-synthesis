"""Runs the pipeline end to end and writes the dataset.

    persona_generator → claim_planner → content_builder → renderer → degrader → here

What lands on disk is the images plus their labels. The dataset itself is not committed:
the repository ships the generator, its configuration and the seed, so that anyone can
reproduce the same output rather than download it.
"""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from receipt_synth import __version__
from receipt_synth.claim_planner import ClaimPlan, plan_claim, plannable_categories
from receipt_synth.config import load_vendors
from receipt_synth.content_builder import build_prro_receipt
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, ClaimGroundTruth, Country, DocGroundTruth, Persona

# How many personas to draw before giving up on finding one a registered archetype can
# document. Only reachable while the template set is incomplete: with one archetype
# covering one category, most personas hold nothing it can carry. The bound exists so a
# misconfiguration fails loudly instead of looping.
_PERSONA_DRAW_LIMIT = 200


@dataclass(frozen=True)
class Dataset:
    """Everything one run produced."""

    seed: int
    personas: list[Persona]
    claims: list[ClaimGroundTruth]
    documents: list[DocGroundTruth]

    def as_manifest(self) -> dict:
        return {
            "generator_version": __version__,
            "seed": self.seed,
            "synthetic": True,
            "personas": [persona.model_dump(mode="json") for persona in self.personas],
            "claims": [claim.model_dump(mode="json") for claim in self.claims],
            "documents": [document.model_dump(mode="json") for document in self.documents],
        }


def _draw_documentable_persona(
    rng: random.Random, persona_id: str, country: Country
) -> Persona:
    """Draw personas until one holds a category some archetype can document."""
    for _ in range(_PERSONA_DRAW_LIMIT):
        persona = generate_persona(rng, persona_id=persona_id, country=country)
        if plannable_categories(persona):
            return persona
    raise RuntimeError(
        f"no persona in {_PERSONA_DRAW_LIMIT} draws held a category any registered "
        "archetype can document — check the archetype registry"
    )


def _pick_vendor(rng: random.Random, country: Country, category: str) -> dict:
    vendors = load_vendors()["vendors"][country.value].get(category, [])
    if not vendors:
        raise ValueError(f"config/vendors.json lists no vendor for {category!r} in {country.value}")
    return rng.choice(vendors)


def _build_document(
    rng: random.Random,
    *,
    persona: Persona,
    plan: ClaimPlan,
    doc_id: str,
    renderer: Renderer,
    out_dir: Path,
) -> DocGroundTruth:
    vendor = _pick_vendor(rng, persona.location.country, plan.category)
    receipt = build_prro_receipt(
        rng,
        category_id=plan.category,
        issued_at=plan.issued_at,
        vendor=vendor,
        # No "м." prefix: Faker's uk_UA city names already carry their settlement type
        # ("хутір Великі Мости"), and prefixing produced "м. хутір Великі Мости".
        address=persona.location.city,
    )

    image_path = out_dir / "images" / f"{doc_id}.png"
    with tempfile.TemporaryDirectory() as staging:
        # The clean render is an intermediate, not an artifact: the dataset ships the
        # document as it would have been captured.
        clean = renderer.render(
            plan.archetype.slug, receipt.render_context(), Path(staging) / f"{doc_id}.png"
        )
        degraded = degrade(
            cv2.imread(str(clean.image_path)),
            clean.field_bboxes,
            seed=rng.getrandbits(32),
            capture=Capture.SCREENSHOT,
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), degraded.image)

    return receipt.ground_truth(
        doc_id=doc_id,
        source_file=image_path.name,
        capture=Capture.SCREENSHOT,
        field_bboxes=degraded.field_bboxes,
    )


def generate_dataset(
    *,
    seed: int,
    out_dir: Path,
    personas: int = 1,
    country: Country = Country.UA,
) -> Dataset:
    """Generate the dataset for a seed, writing images and labels under `out_dir`.

    Fully determined by `seed`. Each persona gets its own generator, derived from the
    root one, so that adding a persona does not shift the content of the personas before
    it.
    """
    root = random.Random(seed)
    all_personas: list[Persona] = []
    all_claims: list[ClaimGroundTruth] = []
    all_documents: list[DocGroundTruth] = []

    with Renderer() as renderer:
        for index in range(personas):
            rng = random.Random(root.getrandbits(64))
            persona_id = f"p{index + 1:03d}"

            persona = _draw_documentable_persona(rng, persona_id, country)
            plan = plan_claim(rng, persona=persona, claim_id=f"{persona_id}_c1")
            document = _build_document(
                rng,
                persona=persona,
                plan=plan,
                doc_id=f"{plan.claim_id}_d1",
                renderer=renderer,
                out_dir=out_dir,
            )

            all_personas.append(persona)
            all_claims.append(plan.ground_truth([document.doc_id]))
            all_documents.append(document)

    dataset = Dataset(
        seed=seed, personas=all_personas, claims=all_claims, documents=all_documents
    )
    _write_labels(dataset, out_dir)
    return dataset


def _write_labels(dataset: Dataset, out_dir: Path) -> None:
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for document in dataset.documents:
        _write_json(labels_dir / f"{document.doc_id}.json", document.model_dump(mode="json"))
    for claim in dataset.claims:
        _write_json(labels_dir / f"{claim.claim_id}.claim.json", claim.model_dump(mode="json"))

    _write_json(out_dir / "ground_truth.json", dataset.as_manifest())


def _write_json(path: Path, payload: dict) -> None:
    # ensure_ascii=False so Ukrainian line items stay readable in the label files, which
    # are meant to be opened and checked by a human as well as parsed.
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
