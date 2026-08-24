"""Synthetic people.

A persona fixes the context every document of theirs inherits: which country's fiscal
rules apply, which currency and language are natural, and which benefit categories they
hold. Nothing here describes a real person — the given name comes from Faker and the
surname from the narrowed pool `content_builder.personal_surname` draws, the identifier is
generated to satisfy its checksum, and the two are made consistent with each other rather
than with anybody.

The benefit categories are drawn according to `persona_categories` in policy.yaml rather
than uniformly, so the dataset is balanced along that axis on purpose instead of by
accident.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from faker import Faker

from receipt_synth.config import load_policy
from receipt_synth.content_builder import generate_rnokpp, personal_surname
from receipt_synth.policy_engine import active_period
from receipt_synth.schemas import Country, Location, Persona

# Faker locale, home currency and language per jurisdiction. A persona generated for a
# country must speak and be paid in something plausible there.
_LOCALES: dict[Country, tuple[str, str, str]] = {
    Country.UA: ("uk_UA", "UAH", "uk"),
    Country.PL: ("pl_PL", "PLN", "pl"),
    Country.DE: ("de_DE", "EUR", "de"),
    Country.ES: ("es_ES", "EUR", "es"),
}

# The age band a claimant of a workplace benefit is drawn from. Nothing prints an age; the band
# exists so that the birth date the РНОКПП encodes belongs to a working-age adult rather than to
# an arbitrary point in the century the identifier can express.
_MIN_AGE_YEARS, _MAX_AGE_YEARS = 22, 60

# Days per year, averaged over the Gregorian cycle. An age band is not a date arithmetic problem
# and does not need one: the band is a decision about who the personas are, and a leap day either
# way moves nobody across it.
_DAYS_PER_YEAR = 365.2425


def _draw_birth_date(rng: random.Random, reference: date) -> date:
    """A birth date in the age band, drawn from the seeded generator and anchored on `reference`.

    🔴 anchored on the benefit period, never on the clock, and that is the whole point of this
    function. It replaces `Faker.date_of_birth(minimum_age=…, maximum_age=…)`, which draws its
    offset from the seeded instance and takes its anchor from `datetime.now()` — so the same
    command at the same seed produced a different corpus on a different calendar day. Measured
    rather than argued: two production runs of the identical command, two hours apart across
    midnight, moved 36 of 96 personas' `tax_id` and 352 of 1141 images. The corpus was a function
    of the seed and of the day, and `corpus_identity` in the labelling contract — three sha256
    digests a consumer checks its directory against — cannot survive that.

    ⚠️ the anchor is `policy.yaml`'s benefit period, which is the frame every other date in this
    dataset already lives in: a claim is in or out of the period, a payment precedes or follows an
    invoice. Anchoring the one remaining date on the same declared window is what makes the whole
    corpus a function of the configuration plus the seed, with nothing left over.
    """
    oldest = reference - timedelta(days=round(_MAX_AGE_YEARS * _DAYS_PER_YEAR))
    youngest = reference - timedelta(days=round(_MIN_AGE_YEARS * _DAYS_PER_YEAR))
    return oldest + timedelta(days=rng.randint(0, (youngest - oldest).days))


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
    # start from a clock.
    #
    # ⚠️ Seeding the instance is not the whole of determinism. A seeded Faker provider still
    # reads the clock for anything defined relative to now, so a seed fixes the offset and not
    # the date. See `_draw_birth_date`.
    fake = Faker(locale)
    fake.seed_instance(rng.getrandbits(64))

    female = rng.random() < 0.5

    # Composed here rather than taken whole from `fake.name_*()`, because the surname comes
    # from `content_builder.personal_surname` — the same narrowed pool a printed sole trader
    # is drawn from. The persona's own name is printed — as the buyer on an invoice, as the
    # payer on a payment confirmation, as the account holder on a bank statement — so it
    # carries the same exposure as a sole trader's name and gets the same mechanism rather
    # than a second one.
    #
    # Composing costs the locale's own name format, which for some locales is more than a
    # given name and one surname — a Spanish full name carries two. Whoever adds the first
    # non-UA template decides what that jurisdiction prints; nothing renders a persona name
    # yet, so nothing is misprinted in the meantime.
    first = fake.first_name_female() if female else fake.first_name_male()
    full_name = f"{first} {personal_surname(rng, fake, country.value, female=female)}"
    birth_date: date = _draw_birth_date(rng, active_period()[1])

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
