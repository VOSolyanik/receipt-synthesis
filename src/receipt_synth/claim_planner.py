"""The label-first core: choose the verdict, then choose documents that realize it.

The direction matters and is easy to get backwards. This module never looks at a built
document — it decides what the answer will be, and `content_builder` is then obliged to
produce evidence consistent with that answer. Anything that read a rendered document to
work out its label would reintroduce exactly the uncertainty the design removes.

What the planner decides: which categories a persona claims, when, with which archetype,
which verdict is being aimed at, and — for `partially_covered` — by which of the two
causes. What it does *not* decide is the answer itself: the verdict a claim ends up
labelled with is computed by `policy_engine` from the documents that were actually built
and from the persona's ledger. The two agree on almost every claim; where they do not,
the engine is right and the difference is reported, because a planner that overruled the
oracle would be writing labels nothing derived.

Four of the six verdicts are still refused rather than faked — see
`_UNREALIZABLE_REASONS`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from receipt_synth.config import load_policy
from receipt_synth.content_builder import MAX_LINE_ITEMS, estimated_line_value
from receipt_synth.policy_engine import (
    ClaimEvaluation,
    Ledger,
    partially_covered_causes,
    verdict_mix,
)
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


# --- what can be realized -----------------------------------------------------

# The verdicts this planner can build documents for. The others are refused rather than
# approximated: a planner that accepted one and produced an ordinary basket would write a
# wrong label instead of failing.
REALIZABLE_VERDICTS: tuple[Verdict, ...] = (Verdict.COVERED, Verdict.PARTIALLY_COVERED)

_UNREALIZABLE_REASONS: dict[Verdict, str] = {
    Verdict.NOT_PROOF_OF_PAYMENT: (
        "needs a document type that establishes no payment — an invoice, an act, a sales "
        "slip — via the `proves_payment: false` entries of `document_evidence` in "
        "policy.yaml. No template in ARCHETYPES carries one yet. This verdict is reached "
        "ONLY that way: a basket bought in the wrong category is `rejected`, which is a "
        "separate member of the enum, so the two mechanisms no longer compete for one name"
    ),
    Verdict.INSUFFICIENT_EVIDENCE: (
        "needs a document dated outside the active period of policy.yaml, or a claim "
        "missing one of the two facts a reimbursement rests on. Content, not coverage "
        "arithmetic — `policy_engine` labels an out-of-period claim correctly today"
    ),
    Verdict.PARTIALLY_PAID: (
        "needs document types that do not exist yet: an invoice or a statement that "
        "shows part of the amount settled. No template in ARCHETYPES can carry it"
    ),
    Verdict.REJECTED: (
        "needs a basket drawn wholly from the `excluded_items` of the claimed category, so "
        "that the covered amount comes to zero. `policy_engine.verdict_for` already labels "
        "such a claim, but nothing builds one: `content_builder` always draws at least one "
        "covered line, and `verdict_mix` in policy.yaml sets no share for this verdict yet. "
        "Both arrive together — a share invented before the mechanism would size a bucket "
        "nothing can fill"
    ),
}

# Reason of last resort, so that a verdict added to the enum without a plan gets a clean
# NotImplementedError naming itself rather than a KeyError from the table above.
_NO_REASON_RECORDED = (
    "is in the Verdict enum but has no entry in claim_planner._UNREALIZABLE_REASONS and "
    "no mechanism registered — whoever added it owes both"
)

# What fraction of a mixed basket is meant to be covered. Bounded away from both ends: at
# 1.0 there would be no non-covered line and the claim would not be partially covered at
# all, and at 0 there would be no covered line — which `policy_engine` labels `rejected`, a
# different verdict from the one being planned here.
#
# An ASPIRATION, not a dial. `content_builder._repriced` clamps every non-covered line
# into its item kind's own price range, so a target the range cannot reach is not reached:
# with the excluded vocabulary `config/vendors.json` currently offers, a 0.90 target on a
# small covered side realizes nearer 0.64, because the non-covered line cannot be priced
# below its floor. What the target *does* guarantee is the verdict — any non-covered line
# at all makes the claim partially covered — and that is the only thing a label depends
# on. Read the realized coverage off `covered_fraction`, never off this tuple.
_MIXED_COVERAGE_TARGETS = tuple(Decimal(f"0.{percent}") for percent in range(30, 95, 5))


def unrealizable_verdicts() -> list[Verdict]:
    """Every verdict this planner cannot yet build.

    Taken from the `Verdict` enum and not from `verdict_mix`, so that the two lists always
    partition the enum. Iterating the mix instead left a hole: a verdict added to the enum
    but absent from the mix would be claimed by neither list, tested by nobody, and would
    reach `_UNREALIZABLE_REASONS` as a `KeyError` rather than a `NotImplementedError`
    saying what it needs.

    The balance report needs these by name: a realized distribution drawn from a subset of
    the target mix is conditional on that subset, and saying which part is missing is the
    difference between a report and a reassuring table.
    """
    return [verdict for verdict in Verdict if verdict not in REALIZABLE_VERDICTS]


def draw_verdict(rng: random.Random) -> Verdict:
    """A target verdict from `verdict_mix`, restricted to what can be built.

    The shares are renormalized over the realizable subset — `random.choices` does that
    from the raw weights — so the drawn distribution is the target mix *conditioned* on
    that subset, not the target mix. Whoever reads the realized distribution has to be
    told which verdicts were excluded, or the conditioning is invisible.
    """
    mix = verdict_mix()
    weights: list[float] = []
    for verdict in REALIZABLE_VERDICTS:
        share = mix[verdict]
        if share is None:
            # policy.yaml may declare a verdict with no share yet — see `verdict_mix`. That
            # is only coherent while nothing draws it, so the combination is named here
            # rather than left to surface as `None` inside `random.choices`.
            raise ValueError(
                f"verdict_mix declares {verdict.value!r} with no share, and this planner "
                "lists it as realizable. A verdict that can be built needs a share to be "
                "built at: either give it one in policy.yaml or take it out of "
                "REALIZABLE_VERDICTS."
            )
        weights.append(share)
    return rng.choices(REALIZABLE_VERDICTS, weights=weights, k=1)[0]


def draw_partially_covered_cause(rng: random.Random) -> str:
    """Why a `partially_covered` claim is partially covered, per `partially_covered_causes`.

    Drawn in the order policy.yaml declares the causes in, which is a file order rather
    than a set order, so the draw stays reproducible.
    """
    causes = partially_covered_causes()
    return rng.choices(list(causes), weights=list(causes.values()), k=1)[0]


@dataclass(frozen=True)
class ClaimPlan:
    """What to build, and what answer it is meant to produce once built.

    `verdict` and `cause` are the *target*. The label that reaches the dataset comes from
    `policy_engine`, which may disagree — a basket meant to overrun an annual limit that
    turned out too small comes back `covered`. `ground_truth` takes the engine's answer
    and never the target, so the disagreement is reported rather than papered over.
    """

    claim_id: str
    persona_id: str
    category: str
    verdict: Verdict
    archetype: Archetype
    issued_at: datetime
    cause: str | None = None
    # Passed straight to `content_builder`. `None` means "the builder's own default".
    coverage_target: Decimal | None = None
    item_count: int | None = None

    def ground_truth(
        self, document_ids: list[str], evaluation: ClaimEvaluation
    ) -> ClaimGroundTruth:
        """The claim-level label record, with the policy engine's answer in it."""
        return ClaimGroundTruth(
            claim_id=self.claim_id,
            persona_id=self.persona_id,
            category=self.category,
            documents=document_ids,
            verdict=evaluation.verdict,
            covered_fraction=evaluation.fraction_as_label(),
            reimbursable_amount=evaluation.reimbursable,
            linked=len(document_ids) > 1,
            imperfection=list(evaluation.imperfection),
            verdict_basis=list(evaluation.verdict_basis),
            policy_trace=list(evaluation.policy_trace),
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


def documentable_categories(persona: Persona) -> list[str]:
    """The persona's benefit categories some registered archetype can carry.

    Ignores the ledger entirely — this is a question about templates, and it is the one
    `assembler._draw_documentable_persona` asks before any claim exists. Empty is a normal
    answer while the template set is incomplete, not an error: with one archetype
    registered, most personas hold nothing it can carry.
    """
    return [
        category
        for category in persona.benefit_categories
        if archetypes_for(persona.location.country, category)
    ]


def plannable_categories(persona: Persona, ledger: Ledger) -> list[str]:
    """The persona's documentable categories that still have an annual balance.

    The ledger is required rather than optional. Defaulting it to "no history" made the
    exhausted-balance guard hold on some call paths and not on others, and a caller that
    omitted it would plan a claim `policy_engine` then refuses to label — the failure
    landing one stage away from its cause. A caller with genuinely no history passes a
    fresh `Ledger()` and says so at the call site.
    """
    return [
        category
        for category in documentable_categories(persona)
        if ledger.remaining(persona.persona_id, category) > 0
    ]


def _draw_date_in_period(rng: random.Random) -> datetime:
    """A timestamp inside the active benefit period.

    Inside, because the planner realizes `covered` and `partially_covered`. A date outside
    the window is what drives the `insufficient_evidence` branch, and choosing it is a
    decision of the planner rather than an accident of the calendar — which is why the
    period is read from policy.yaml and never from today's date.
    """
    period = load_policy()["period"]
    start, end = date.fromisoformat(str(period["start"])), date.fromisoformat(str(period["end"]))

    day = start + timedelta(days=rng.randint(0, (end - start).days))
    # Trading hours, so a receipt is not timestamped at four in the morning.
    return datetime(day.year, day.month, day.day, rng.randint(9, 20), rng.randint(0, 59),
                    rng.randint(0, 59))


def draw_claim_dates(rng: random.Random, count: int) -> list[datetime]:
    """`count` timestamps inside the active period, ascending.

    Drawn together and sorted, because cumulative limits bind in date order: a persona
    whose claims arrived in a random order would exhaust its balance on whichever claim
    happened to be planned last rather than on the one that happened last.
    """
    return sorted(_draw_date_in_period(rng) for _ in range(count))


def _overrun_item_count(remaining: Decimal, category: str) -> int:
    """How many lines a basket needs before it plausibly exceeds `remaining`.

    An estimate over a draw the planner has not made yet, so it is deliberately generous
    and still capped: past `MAX_LINE_ITEMS` the receipt stops being one a shop would
    print, and a balance too large to overrun within that bound simply is not overrun.
    """
    per_line = estimated_line_value(category)
    return max(2, min(MAX_LINE_ITEMS, math.ceil(remaining / per_line) + 1))


def plan_claim(
    rng: random.Random,
    *,
    persona: Persona,
    claim_id: str,
    ledger: Ledger,
    verdict: Verdict | None = None,
    cause: str | None = None,
    category: str | None = None,
    issued_at: datetime | None = None,
) -> ClaimPlan:
    """Plan one claim for a persona.

    `verdict` and `cause` are drawn from policy.yaml when not given. `ledger` is required:
    it carries what the persona has already been reimbursed, it keeps the planner out of
    categories with no balance left, and it is what a `limit_exhausted` claim is sized
    against. An optional ledger meant the balance guard held on the drawn path and not on
    the named one, and that a caller who simply omitted it planned a claim the engine then
    refused — the failure landing a stage away from its cause. A caller with no history
    passes `Ledger()` and says so at the call site.
    """
    if verdict is None:
        verdict = draw_verdict(rng)
    if verdict not in REALIZABLE_VERDICTS:
        raise NotImplementedError(
            f"verdict {verdict.value!r} "
            f"{_UNREALIZABLE_REASONS.get(verdict, _NO_REASON_RECORDED)}"
        )

    if category is None:
        options = plannable_categories(persona, ledger)
        if not options:
            raise ValueError(
                f"persona {persona.persona_id} holds no category any registered archetype "
                f"can document with balance left: {persona.benefit_categories}"
            )
        category = rng.choice(options)
    elif category not in persona.benefit_categories:
        raise ValueError(f"persona {persona.persona_id} does not hold category {category!r}")

    # One guard, every path. `plannable_categories` filters exhausted categories out of the
    # drawn one, but a caller naming a category explicitly went around that filter, and the
    # engine would then refuse to label the claim this planner had just produced. A rule
    # enforced on one branch is a rule the other branch is free to break.
    if ledger.remaining(persona.persona_id, category) <= 0:
        raise ValueError(
            f"persona {persona.persona_id} has no {category!r} balance left; policy.yaml "
            "assigns no verdict to a claim that covers something and is still paid nothing "
            "(see `policy_engine._reimbursable`), so there is nothing to plan here"
        )

    candidates = archetypes_for(persona.location.country, category)
    if not candidates:
        raise ValueError(
            f"no archetype registered for {category!r} in {persona.location.country.value}"
        )

    coverage_target: Decimal | None = None
    item_count: int | None = None
    if verdict is Verdict.PARTIALLY_COVERED:
        cause = cause or draw_partially_covered_cause(rng)
        if cause == "mixed_items":
            coverage_target = rng.choice(_MIXED_COVERAGE_TARGETS)
        elif cause == "limit_exhausted":
            # Realized by a basket the remaining balance cannot absorb. The engine decides
            # whether it actually did; nothing here assumes it.
            remaining = ledger.remaining(persona.persona_id, category)
            item_count = _overrun_item_count(remaining, category)
        else:
            raise ValueError(f"policy.yaml declares no such partially_covered cause: {cause!r}")
    elif cause is not None:
        raise ValueError(f"a cause belongs to a partially_covered claim, not to {verdict.value!r}")

    return ClaimPlan(
        claim_id=claim_id,
        persona_id=persona.persona_id,
        category=category,
        verdict=verdict,
        cause=cause,
        archetype=rng.choice(candidates),
        issued_at=issued_at if issued_at is not None else _draw_date_in_period(rng),
        coverage_target=coverage_target,
        item_count=item_count,
    )


def plan_claims(
    rng: random.Random,
    *,
    persona: Persona,
    count: int,
    ledger: Ledger,
    claim_id_prefix: str | None = None,
) -> Iterator[ClaimPlan]:
    """Plan up to `count` claims for one persona, in date order, against one ledger.

    A generator rather than a list, and that is the whole point: cumulative limits mean
    claim *k* depends on what claims 1…k−1 were actually reimbursed, which is not known
    until their documents exist. The caller therefore builds and evaluates each claim and
    records the reimbursement in `ledger` before asking for the next plan. Planning them
    all up front would size every `limit_exhausted` basket against a balance that no
    longer exists by the time it is built.

    Fewer than `count` plans come out when the persona runs out of categories with a
    balance — `count` is a ceiling, not a promise. Ask `why_no_claim` afterwards for the
    reason, and report it: a run that ordered 40 claims and built 31 has to say so, or a
    dataset size gets quoted from a number that was never the number asked for.
    """
    prefix = claim_id_prefix or persona.persona_id
    for index, issued_at in enumerate(draw_claim_dates(rng, count), start=1):
        if why_no_claim(persona, ledger) is not None:
            return
        yield plan_claim(
            rng,
            persona=persona,
            claim_id=f"{prefix}_c{index}",
            issued_at=issued_at,
            ledger=ledger,
        )


# Reasons `plan_claims` stops short of the count it was asked for. Strings rather than an
# enum because they are printed verbatim in the balance report, and a reason nobody can
# read is a reason nobody checks.
NO_BALANCE_LEFT = "every documentable category of the persona has its annual limit spent"
NO_ARCHETYPE = "no registered archetype can document any category the persona holds"


def why_no_claim(persona: Persona, ledger: Ledger) -> str | None:
    """Why no further claim can be planned for this persona, or `None` if one can.

    Exists so that the claims a run *ordered* but did not build can be accounted for by
    reason rather than silently dropped. Returning `None` while `plan_claims` has stopped
    would mean it stopped for a reason nothing here explains — the assembler reports that
    as unattributed rather than folding it into a bucket that sounds accounted for.
    """
    if plannable_categories(persona, ledger):
        return None
    return NO_BALANCE_LEFT if documentable_categories(persona) else NO_ARCHETYPE
