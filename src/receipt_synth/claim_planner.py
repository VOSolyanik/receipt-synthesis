"""The label-first core: choose the verdict, then choose documents that realize it.

The direction matters and is easy to get backwards. This module never looks at a built
document — it decides what the answer will be, and `content_builder` is then obliged to
produce evidence consistent with that answer. Anything that read a rendered document to
work out its label would reintroduce exactly the uncertainty the design removes.

Scope here is one claim with the verdict handed in directly. The machinery that *draws*
a verdict from `verdict_mix`, tracks cumulative limits, computes `covered_fraction` and
emits `policy_trace` is the policy engine, and it is not in this file. What is here is
the archetype registry and the plan those parts will later fill in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from receipt_synth.config import load_policy
from receipt_synth.schemas import (
    ClaimGroundTruth,
    Country,
    DocType,
    Persona,
    Verdict,
)


@dataclass(frozen=True)
class Archetype:
    """One document template, with the facts it can establish.

    `proves_subject` and `proves_payment` are what let the planner assemble a claim from
    several documents: it keeps adding until both are satisfied. They default from
    `document_evidence` in policy.yaml and may be overridden per archetype — a bank
    confirmation whose payment purpose spells out what was bought does prove the subject,
    unlike a bare transfer.
    """

    slug: str
    doc_type: DocType
    country: Country
    language: str
    # Benefit categories this template can actually carry. A pharmacy receipt cannot
    # print a gym membership. As templates land this widens until every category has at
    # least one archetype in every jurisdiction.
    categories: tuple[str, ...]


# The registry the planner selects from. One entry today; the remaining twenty-three
# archetypes register here as their templates land.
ARCHETYPES: dict[str, Archetype] = {
    "ua_prro_receipt": Archetype(
        slug="ua_prro_receipt",
        doc_type=DocType.FISCAL_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=("vitamins_nutrition",),
    ),
}


@dataclass(frozen=True)
class ClaimPlan:
    """What to build, and what the answer will be once it is built."""

    claim_id: str
    persona_id: str
    category: str
    verdict: Verdict
    archetype: Archetype
    issued_at: datetime

    def ground_truth(self, document_ids: list[str]) -> ClaimGroundTruth:
        """The claim-level label record.

        `covered_fraction`, `verdict_basis` and `policy_trace` are left unset: they are
        derived by the policy engine, and filling them in with plausible values here
        would put labels in the dataset that nothing actually computed.
        """
        return ClaimGroundTruth(
            claim_id=self.claim_id,
            persona_id=self.persona_id,
            category=self.category,
            documents=document_ids,
            verdict=self.verdict,
            linked=len(document_ids) > 1,
            imperfection=[],
        )


def archetypes_for(
    country: Country,
    category: str | None = None,
    doc_type: DocType | None = None,
) -> list[Archetype]:
    """Registered archetypes for a jurisdiction, narrowed by category or document class."""
    return [
        archetype
        for archetype in ARCHETYPES.values()
        if archetype.country is country
        and (category is None or category in archetype.categories)
        and (doc_type is None or archetype.doc_type is doc_type)
    ]


def plannable_categories(persona: Persona) -> list[str]:
    """The persona's benefit categories some registered archetype can actually document.

    Empty is a normal answer while the template set is incomplete, not an error: with one
    archetype registered, most personas hold nothing it can carry.
    """
    return [
        category
        for category in persona.benefit_categories
        if archetypes_for(persona.location.country, category)
    ]


def _draw_date_in_period(rng: random.Random) -> datetime:
    """A timestamp inside the active benefit period.

    Inside, because this skeleton plans `covered`. A date outside the window is what
    drives the `insufficient_evidence` branch, and choosing it is a decision of the
    planner rather than an accident of the calendar — which is why the period is read
    from policy.yaml and never from today's date.
    """
    period = load_policy()["period"]
    start, end = date.fromisoformat(str(period["start"])), date.fromisoformat(str(period["end"]))

    day = start + timedelta(days=rng.randint(0, (end - start).days))
    # Trading hours, so a receipt is not timestamped at four in the morning.
    return datetime(day.year, day.month, day.day, rng.randint(9, 20), rng.randint(0, 59),
                    rng.randint(0, 59))


def plan_claim(
    rng: random.Random,
    *,
    persona: Persona,
    claim_id: str,
    verdict: Verdict = Verdict.COVERED,
    category: str | None = None,
) -> ClaimPlan:
    """Plan one claim for a persona.

    The verdict is passed in rather than drawn: drawing it from `verdict_mix` belongs to
    the policy engine. What this does decide is the category, the archetype that can
    carry it, and when it happened.
    """
    if verdict is not Verdict.COVERED:
        raise NotImplementedError(
            f"verdict {verdict.value!r} needs the imperfection catalogue and the policy "
            "engine; see docs/architecture.md#imperfection-catalogue"
        )

    if category is None:
        options = plannable_categories(persona)
        if not options:
            raise ValueError(
                f"persona {persona.persona_id} holds no category any registered archetype "
                f"can document: {persona.benefit_categories}"
            )
        category = rng.choice(options)
    elif category not in persona.benefit_categories:
        raise ValueError(f"persona {persona.persona_id} does not hold category {category!r}")

    candidates = archetypes_for(persona.location.country, category)
    if not candidates:
        raise ValueError(
            f"no archetype registered for {category!r} in {persona.location.country.value}"
        )

    return ClaimPlan(
        claim_id=claim_id,
        persona_id=persona.persona_id,
        category=category,
        verdict=verdict,
        archetype=rng.choice(candidates),
        issued_at=_draw_date_in_period(rng),
    )
