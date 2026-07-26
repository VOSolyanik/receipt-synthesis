"""Synthetic people.

A persona fixes the context every document of theirs inherits: which country's fiscal
rules apply, which currency and language are natural, and which benefit categories they
hold. Nothing here describes a real person — the names come from Faker, the identifier
is generated to satisfy its checksum, and the two are made consistent with each other
rather than with anybody.

The benefit categories are drawn according to `persona_categories` in policy.yaml rather
than uniformly, so the dataset is balanced along that axis on purpose instead of by
accident.
"""

from __future__ import annotations

import random
from datetime import date

from faker import Faker

from receipt_synth.config import load_policy
from receipt_synth.content_builder import generate_rnokpp
from receipt_synth.schemas import Country, Location, Persona

# Faker locale, home currency and language per jurisdiction. A persona generated for a
# country must speak and be paid in something plausible there.
_LOCALES: dict[Country, tuple[str, str, str]] = {
    Country.UA: ("uk_UA", "UAH", "uk"),
    Country.PL: ("pl_PL", "PLN", "pl"),
    Country.DE: ("de_DE", "EUR", "de"),
    Country.ES: ("es_ES", "EUR", "es"),
}


def _draw_categories(rng: random.Random, country: Country) -> list[str]:
    """Benefit categories for one persona, per `persona_categories` in policy.yaml.

    Some categories are near-universal in a given market and are drawn against their own
    probability; the remaining slots are filled uniformly from the rest.
    """
    policy = load_policy()
    spec = policy["persona_categories"]
    all_ids = [entry["id"] for entry in policy["categories"]]

    chosen: list[str] = []
    for category_id, by_country in spec.get("probability", {}).items():
        probability = by_country.get(country.value)
        if probability is not None and rng.random() < probability:
            chosen.append(category_id)

    total = rng.randint(spec["count_per_persona"]["min"], spec["count_per_persona"]["max"])
    remaining = [entry for entry in all_ids if entry not in chosen]
    chosen += rng.sample(remaining, max(0, min(total - len(chosen), len(remaining))))

    # Sorted by the policy's own order, so a persona's categories read the same way
    # whichever order they happened to be drawn in.
    return [entry for entry in all_ids if entry in set(chosen)]


def generate_persona(
    rng: random.Random,
    *,
    persona_id: str,
    country: Country = Country.UA,
) -> Persona:
    """One synthetic person, fully determined by `rng`."""
    locale, currency, language = _LOCALES[country]

    # Faker carries its own generator, so it is seeded from ours rather than left to
    # start from a clock. Everything below is then a function of the incoming seed.
    fake = Faker(locale)
    fake.seed_instance(rng.getrandbits(64))

    female = rng.random() < 0.5
    full_name = fake.name_female() if female else fake.name_male()
    birth_date: date = fake.date_of_birth(minimum_age=22, maximum_age=60)

    tax_id = (
        generate_rnokpp(rng, birth_date=birth_date, female=female)
        if country is Country.UA
        # Other jurisdictions have their own formats and checksums in fiscal-rules.yaml;
        # they arrive with the templates that print them.
        else ""
    )

    return Persona(
        persona_id=persona_id,
        full_name=full_name,
        location=Location(country=country, city=fake.city()),
        home_currencies=[currency],
        languages=[language],
        tax_id=tax_id,
        family=[],
        benefit_categories=_draw_categories(rng, country),
    )
