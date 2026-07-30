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
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from pathlib import Path

import cv2

from receipt_synth import __version__
from receipt_synth.claim_planner import (
    REALIZABLE_VERDICTS,
    ClaimPlan,
    DocumentPlan,
    documentable_categories,
    evidence_of,
    plan_claims,
    unrealizable_verdicts,
    why_no_claim,
)
from receipt_synth.config import load_vendors, mismatch_delta_range
from receipt_synth.content_builder import (
    KOPIYKA,
    build_bank_statement,
    build_invoice,
    build_payment_confirmation,
    build_prro_receipt,
    resolve_vendor,
    vendor_can_carry,
)
from receipt_synth.degrader import degrade
from receipt_synth.persona_generator import generate_persona
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    Ledger,
    evaluate_claim,
    insufficient_evidence_causes,
    partially_covered_causes,
    verdict_mix,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import (
    Capture,
    ClaimGroundTruth,
    Country,
    DocGroundTruth,
    Persona,
    Verdict,
)

# How many personas to draw before giving up on finding one a registered archetype can
# document. Only reachable while the template set is incomplete: the registered archetypes
# cover one category between them, so most personas hold nothing any of them can carry. The
# bound exists so a misconfiguration fails loudly instead of looping.
_PERSONA_DRAW_LIMIT = 200

# The bucket for ordered claims the planner stopped short of without a reason it could
# name. Spelled out rather than merged into the nearest plausible reason: a count under a
# label that sounds accounted for is worse than a count that admits it is not.
UNATTRIBUTED = "not attributed — planning stopped for a reason claim_planner cannot name"


@dataclass(frozen=True)
class Dataset:
    """Everything one run produced.

    `plans` is run metadata rather than ground truth: it holds the verdict each claim was
    *aimed at*, which the balance report needs in order to say where the realized labels
    differ from the drawn ones. It is deliberately not part of the manifest — a target is
    not a label, and putting the two in one file invites reading one as the other.

    `claims_ordered` and `claims_skipped` are the other half of that honesty. A run of
    five personas × eight claims orders forty and may build thirty-one, and a report that
    opens with thirty-one presents the shortfall as if it had never been asked for. The
    difference between "generated 150 claims" and "ordered 150, built 116" is the
    difference between a true and a false sentence in somebody's write-up, and nobody
    re-derives it later.
    """

    seed: int
    personas: list[Persona]
    claims: list[ClaimGroundTruth]
    documents: list[DocGroundTruth]
    plans: list[ClaimPlan] = field(default_factory=list)
    claims_ordered: int = 0
    # Reason -> how many ordered claims were not built for it. `claim_planner` supplies the
    # reasons; anything it cannot attribute is counted under `UNATTRIBUTED` rather than
    # being folded into a bucket that sounds accounted for.
    claims_skipped: dict[str, int] = field(default_factory=dict)

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
        if documentable_categories(persona):
            return persona
    raise RuntimeError(
        f"no persona in {_PERSONA_DRAW_LIMIT} draws held a category any registered "
        "archetype can document — check the archetype registry"
    )


def _pick_vendor(
    rng: random.Random, country: Country, category: str, *, mixed: bool
) -> dict:
    """A vendor that can issue the receipt this plan needs.

    Filtered by what the vendor sells, not only by its category. A mixed basket needs a
    non-covered line, and honest vendors exist that sell nothing the plan excludes — a
    nutrition practice sells consultations and lab tests and nothing else. Choosing one of
    those for a mixed plan would fail inside the builder, one stage away from the choice
    that caused it. The filter preserves file order, so the draw stays reproducible.

    Returns a RESOLVED vendor: a sole trader's name is drawn here, once, and the same
    instance is then carried to every document of the claim. Called from the claim loop and
    never from `_build_document`, because a per-document call would redraw the name and put
    two different sellers on two documents of one purchase.
    """
    vendors = load_vendors()["vendors"][country.value].get(category, [])
    if not vendors:
        raise ValueError(f"config/vendors.json lists no vendor for {category!r} in {country.value}")

    candidates = [v for v in vendors if vendor_can_carry(v, category, mixed=mixed)]
    if not candidates:
        raise ValueError(
            f"no vendor for {category!r} in {country.value} sells what a "
            f"{'mixed' if mixed else 'fully covered'} basket needs — check the profiles in "
            "config/vendors.json against the item buckets of config/policy.yaml"
        )
    return resolve_vendor(rng, rng.choice(candidates), country.value)


# Which builder produces which archetype. A table rather than a call, because a claim is
# now a list of documents and the loop below cannot assume they are all fiscal receipts —
# an unregistered slug has to fail by name instead of being silently handed to the one
# builder that exists. One entry per slug in `claim_planner.ARCHETYPES`.
#
# THE KIND OF CASH REGISTER IS BOUND HERE, and this is the place for it: an archetype is a
# template, `registrar` says which fiscal identity that template's document carries, and the
# pairing of the two is exactly what this table is for. It is deliberately NOT a field on
# `Archetype` — the planner decides labels, and which prefix a fiscal number takes is not one.
# The paper width is bound nowhere in Python at all: it lives in `<slug>.css`, which the
# renderer picks up from the slug.
#
# FOUR DOCUMENT CLASSES NOW, AND THE DISPATCH IS THREE-WAY — keyed on `document_evidence` and not
# on the class, so what a document must be told follows from what it proves:
#
#   proves both      a fiscal receipt. Takes a basket. Names NO buyer: the payer is standing at the
#                    till, so a receipt has no buyer field at all.
#   subject only     an invoice. Takes a basket AND the claimant, because an OFFER TO PAY has to say
#                    to whom it is made.
#   payment only     a confirmation or a statement. Takes the claimant as the payer, no basket, and
#                    THE CLAIM'S AMOUNT — see `_build_document`.
#
# The middle case is what the invoice added. It is a real relation rather than a convenient one: a
# document that proves the payment IS the payment, so its payer is present by construction; a
# document that does not prove payment is addressed to somebody and must name them.
_BUILDERS = {
    "ua_prro_receipt": build_prro_receipt,
    "ua_prro_receipt_58mm": build_prro_receipt,
    "ua_rro_receipt": partial(build_prro_receipt, registrar="rro"),
    "ua_bank_payment_confirmation": build_payment_confirmation,
    "ua_bank_statement": build_bank_statement,
    "ua_invoice": build_invoice,
}


def _build_document(
    rng: random.Random,
    *,
    persona: Persona,
    plan: ClaimPlan,
    document_plan: DocumentPlan,
    vendor: dict,
    doc_id: str,
    renderer: Renderer,
    out_dir: Path,
    settles: Decimal | None = None,
) -> DocGroundTruth:
    """One document of a claim.

    `settles` is THE AMOUNT THIS DOCUMENT'S CLAIM IS ABOUT, and it is required for a document that
    proves the payment and ignored by one that states the subject. The subject document decides the
    amount — its basket is drawn first and summed — and the payment document is then told what it
    settles. The order is not an accident of the loop: a claim's money is a property of what was
    bought, so the document that lists the purchase is the one that fixes it.

    `vendor` is passed in rather than chosen here. It is the claim's vendor instance, and
    every document of the claim has to name the same seller — while a sole trader's name
    was a stored constant that held by the nature of the type, and a drawn name can differ,
    so it is now a constraint somebody has to keep.

    The basket goes to the claim's SUBJECT document and to no other. Sizing is a claim-level
    decision (`ClaimPlan.coverage_target`, `item_count`), and giving the same basket to a
    second document would double the money a claim aimed at a limit was sized to spend.

    DISPATCHED ON WHAT THE DOCUMENT HAS TO STATE, read from `document_evidence` in policy.yaml
    rather than from a table of slugs. The two classes take different parameters because they
    state different things: a receipt needs a category to draw a basket from, and a payment
    confirmation needs the two parties and takes no basket at all. Keying on the evidence means a
    further archetype of either class arrives without this branch being touched.
    """
    archetype = document_plan.archetype
    slug = archetype.slug
    if slug not in _BUILDERS:
        raise NotImplementedError(
            f"archetype {slug!r} is registered in claim_planner.ARCHETYPES but no builder "
            f"produces it; assembler._BUILDERS knows {sorted(_BUILDERS)}"
        )

    evidence = evidence_of(archetype)
    if evidence.proves_subject:
        if document_plan is not plan.subject_document:
            # `ClaimPlan.subject_document` establishes that the claim has exactly ONE document
            # stating what was bought; this says that the one being built is that document. The
            # basket is sized once per claim, so a second carrier would spend the claim's money
            # twice.
            raise ValueError(
                f"{slug!r} states what was bought but is not this claim's subject document, "
                "and a basket is sized once per claim — see `ClaimPlan.coverage_target`"
            )
        basket = {
            "category_id": plan.category,
            "issued_at": document_plan.issued_at,
            "vendor": vendor,
            # No "м." prefix: Faker's uk_UA city names already carry their settlement type
            # ("хутір Великі Мости"), and prefixing produced "м. хутір Великі Мости".
            "address": persona.location.city,
            "covered_only": plan.coverage_target is None,
            "coverage_target": plan.coverage_target,
            "item_count": plan.item_count,
        }
        # A SUBJECT DOCUMENT THAT DOES NOT PROVE PAYMENT IS ADDRESSED TO SOMEBODY, and has to name
        # them: an invoice is an offer to pay. A receipt proves its own payment, so the payer is
        # present at the till and no buyer is named — 👁 a fiscal receipt has no buyer field.
        if not evidence.proves_payment:
            basket |= {"buyer_name": persona.full_name, "buyer_tax_id": persona.tax_id}
        document = _BUILDERS[slug](rng, **basket)
    else:
        # A document that proves the payment and states no subject. It takes NO basket and no
        # coverage target — there is nothing on it for a coverage rule to read, which is exactly
        # why its type proves no subject — and it takes the persona, because this class prints a
        # persona's own name, as the payer.
        #
        # 🔴 AND IT TAKES THE CLAIM'S AMOUNT, which is the whole of what makes a split pair
        # coherent. `policy_engine._cross_checks` compares a transaction's subject and payment
        # amounts EXACTLY — there is no tolerance anywhere in this repository — so a payment that
        # drew its own amount would disagree with the invoice beside it on every claim, and every
        # such claim would be labelled `insufficient_evidence` with the cause `amount_mismatch`.
        # Measured before this was written: an invoice of 1200.00 beside an independently drawn
        # payment came out exactly that, while the same pair agreeing came out `covered`.
        document = _BUILDERS[slug](
            rng,
            issued_at=document_plan.issued_at,
            vendor=vendor,
            payer_name=persona.full_name,
            payer_tax_id=persona.tax_id,
            amount=settles,
        )

    image_path = out_dir / "images" / f"{doc_id}.png"
    with tempfile.TemporaryDirectory() as staging:
        # The clean render is an intermediate, not an artifact: the dataset ships the
        # document as it would have been captured.
        clean = renderer.render(slug, document.render_context(), Path(staging) / f"{doc_id}.png")
        degraded = degrade(
            cv2.imread(str(clean.image_path)),
            clean.field_bboxes,
            seed=rng.getrandbits(32),
            capture=Capture.SCREENSHOT,
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), degraded.image)

    return document.ground_truth(
        doc_id=doc_id,
        source_file=image_path.name,
        capture=Capture.SCREENSHOT,
        field_bboxes=degraded.field_bboxes,
    )


def _amount_the_payment_states(
    rng: random.Random, plan: ClaimPlan, subject_amount: Decimal
) -> Decimal:
    """What the payment document of this claim should state, given the subject's amount.

    THE SAME AMOUNT ON AN ORDINARY CLAIM, which is what makes a pair one transaction:
    `policy_engine._cross_checks` compares the two EXACTLY, and there is no tolerance anywhere in
    this repository.

    🔴 A DIFFERENT AMOUNT WHEN THE PLAN ASKED FOR ONE. A claim planned as `insufficient_evidence`
    with the cause `amount_mismatch` is realized here and nowhere else: the label was chosen first
    and the documents are built to make it true, which is the whole direction of this generator.
    The delta is drawn from config/generation.yaml — the magnitude is a difficulty knob and changes
    no label — and it is applied in whichever direction keeps the payment positive, because a
    document stating a negative amount is a different defect from a document stating the wrong one.

    The engine still decides. If a delta ever came out zero the claim would come back `covered`,
    and `assembler._drift_lines` would report the target and the label disagreeing rather than
    anybody assuming they agree — which is why the floor is in the configured range rather than in
    an assertion here.
    """
    if plan.cause != AMOUNT_MISMATCH:
        return subject_amount

    low, high = mismatch_delta_range()
    delta = Decimal(rng.randrange(int(low * 100), int(high * 100), 10)) / 100
    if delta >= subject_amount:
        # A payment of zero or less is not a mismatched payment, it is a broken document.
        return (subject_amount + delta).quantize(KOPIYKA)
    return (subject_amount + (delta if rng.random() < 0.5 else -delta)).quantize(KOPIYKA)


def generate_dataset(
    *,
    seed: int,
    out_dir: Path,
    personas: int = 1,
    claims_per_persona: int = 1,
    country: Country = Country.UA,
) -> Dataset:
    """Generate the dataset for a seed, writing images and labels under `out_dir`.

    Fully determined by `seed`. Each persona gets its own generator, derived from the
    root one, so that adding a persona does not shift the content of the personas before
    it.

    `claims_per_persona` is a ceiling. Each persona carries its own ledger, and the
    planner stops early once no category of theirs has an annual balance left — which is
    also the only way the cumulative-limit mechanism can be exercised at all.
    """
    if claims_per_persona < 1:
        raise ValueError(f"a persona files at least one claim, not {claims_per_persona}")

    root = random.Random(seed)
    all_personas: list[Persona] = []
    all_claims: list[ClaimGroundTruth] = []
    all_documents: list[DocGroundTruth] = []
    all_plans: list[ClaimPlan] = []
    skipped: Counter[str] = Counter()

    with Renderer() as renderer:
        for index in range(personas):
            rng = random.Random(root.getrandbits(64))
            persona_id = f"p{index + 1:03d}"

            persona = _draw_documentable_persona(rng, persona_id, country)
            all_personas.append(persona)

            # One ledger per persona: limits are cumulative per persona per category, and
            # a shared ledger would let one person's spending exhaust another's benefit.
            ledger = Ledger()
            built = 0
            for plan in plan_claims(
                rng, persona=persona, count=claims_per_persona, ledger=ledger
            ):
                # The claim's vendor instance, chosen ONCE here and carried into every
                # document of the claim. Outside the loop below on purpose: a sole
                # trader's name is drawn rather than stored, so calling `_pick_vendor` per
                # document would put two different sellers on two documents of one
                # purchase. That constraint used to hold by the nature of the type.
                vendor = _pick_vendor(
                    rng,
                    persona.location.country,
                    plan.category,
                    mixed=plan.coverage_target is not None,
                )
                # A claim is a list of documents, and since the invoice archetype landed it may
                # genuinely hold two. A LOOP RATHER THAN A COMPREHENSION, because the documents are
                # no longer independent: the subject document fixes the claim's amount and the
                # payment document has to be told it, or the two disagree and the engine labels
                # every such claim `insufficient_evidence`.
                #
                # `plan.documents` is ordered subject-first — `_select_documents` builds it that
                # way — but nothing here relies on the order: `settles` is read off whichever
                # document proved the subject, and stays `None` for a self-contained claim, whose
                # single document is its own subject and its own payment.
                documents: list[DocGroundTruth] = []
                settles: Decimal | None = None
                for index, document_plan in enumerate(plan.documents, start=1):
                    document = _build_document(
                        rng,
                        persona=persona,
                        plan=plan,
                        document_plan=document_plan,
                        vendor=vendor,
                        doc_id=f"{plan.claim_id}_d{index}",
                        renderer=renderer,
                        out_dir=out_dir,
                        settles=settles,
                    )
                    if evidence_of(document_plan.archetype).proves_subject:
                        settles = _amount_the_payment_states(rng, plan, document.amount)
                    documents.append(document)
                # The oracle, not the plan, decides the label. The plan's verdict was the
                # target; where the two differ the balance report says so.
                evaluation = evaluate_claim(
                    persona_id=persona.persona_id,
                    category=plan.category,
                    documents=documents,
                    ledger=ledger,
                )
                ledger.record(persona.persona_id, plan.category, evaluation.reimbursable)

                all_plans.append(plan)
                all_claims.append(
                    plan.ground_truth([d.doc_id for d in documents], evaluation)
                )
                all_documents.extend(documents)
                built += 1

            if built < claims_per_persona:
                # Attributed, or explicitly not. `why_no_claim` returning None here would
                # mean planning stopped for a reason nothing in the planner explains, and
                # the report has to say that rather than assume the usual one.
                reason = why_no_claim(persona, ledger) or UNATTRIBUTED
                skipped[reason] += claims_per_persona - built

    dataset = Dataset(
        seed=seed,
        personas=all_personas,
        claims=all_claims,
        documents=all_documents,
        claims_ordered=personas * claims_per_persona,
        claims_skipped=dict(skipped),
        plans=all_plans,
    )
    _write_labels(dataset, out_dir)
    return dataset


def balance_report(dataset: Dataset) -> str:
    """The realized verdict distribution against `verdict_mix` — and what is missing from it.

    Deliberately not a tidy table. Only two of the six verdicts in `verdict_mix` can be
    built yet, so the draw is renormalized over those two and the realized shares are
    conditional on that subset. A report that renormalized silently would print a
    balanced-looking dataset while a third of the target mix was absent, which is worse
    than printing nothing: it answers the question nobody would then think to ask.

    One of those six carries no share yet — `verdict_mix` may declare a member as `null`,
    and `rejected` is one today. Such a member is named without a percentage and excluded
    from every sum, and the absent fraction is then reported as a lower bound: it is what
    the share-carrying verdicts account for, not the whole of what is missing.

    The full report — document classes, currencies, languages, the train/validation split
    — is a later step. This is the verdict axis, the claim count and the exclusion note,
    nothing else.
    """
    mix = verdict_mix()
    missing = unrealizable_verdicts()
    # A member declared with no share is not a zero-weight member: its share is undecided,
    # so it can be neither summed nor printed as a target. Kept out of the arithmetic and
    # named separately, because a `None` folded in as 0 would leave the shares below
    # looking like a complete account of the mix.
    undeclared = [verdict for verdict, share in mix.items() if share is None]
    realizable_share = sum(
        share
        for verdict, share in mix.items()
        if verdict not in missing and share is not None
    )

    realized = Counter(claim.verdict for claim in dataset.claims)
    total = sum(realized.values())

    lines = _count_lines(dataset, total)
    lines.append(f"Verdict balance — {total} claim(s) built")

    # Every realizable verdict gets a row whether or not it occurred, plus any verdict that
    # occurred without being realizable. Iterating the target mix instead skipped the
    # unrealizable ones, so a claim the engine did emit as one would vanish from the table
    # while still counting in `total`, and the rows would quietly stop summing.
    rows = list(REALIZABLE_VERDICTS) + [
        verdict for verdict in missing if realized[verdict]
    ]
    for verdict in rows:
        count = realized[verdict]
        if verdict in missing:
            lines.append(
                f"  {verdict.value:<22} {count:>4}  {_share(count, total):>6}"
                "   !! realized but not realizable — the planner cannot draw this verdict,"
                " so the engine emitted it on its own"
            )
            continue
        lines.append(
            f"  {verdict.value:<22} {count:>4}  {_share(count, total):>6}"
            f"   target {mix[verdict] / realizable_share:.1%} of the realizable subset"
            f" ({mix[verdict]:.1%} of the full mix)"
        )

    accounted = sum(realized[verdict] for verdict in rows)
    if accounted != total:
        lines.append(
            f"  !! the rows above account for {accounted} of {total} claim(s). "
            "This cannot happen unless a claim carries a verdict outside the enum; "
            "the table is wrong, not the dataset."
        )

    lines += _cause_lines(dataset)
    lines += _insufficient_evidence_cause_lines(dataset)

    if missing:
        named = ", ".join(_target_share(verdict, mix) for verdict in missing)
        lines += [
            f"NOT GENERATED IN THIS RUN: {named}",
            f"  {1 - realizable_share:.1%} of the target mix is absent, so the shares above are",
            "  CONDITIONAL on the realizable subset and are not this dataset's balance",
            "  against verdict_mix. Do not read them as one.",
        ]
        if undeclared:
            lines += [
                "  verdict_mix declares no share for "
                + ", ".join(verdict.value for verdict in undeclared)
                + ", so the figure above is a LOWER BOUND",
                "  on what is absent rather than the whole of it.",
            ]

    lines += _drift_lines(dataset)
    return "\n".join(lines)


def _count_lines(dataset: Dataset, built: int) -> list[str]:
    """How many claims the run ordered, how many it built, and why the rest were not.

    The first thing in the report, because it is the number that gets quoted. A run of
    `--personas 5 --claims-per-persona 8` orders forty; if planning stops early for three
    personas the report must not open with thirty-one as though thirty-one had been the
    request. Skipped claims are broken down by the reason `claim_planner` gives, and any
    it cannot attribute is said to be unattributed rather than assigned to the likeliest
    bucket.
    """
    ordered = dataset.claims_ordered
    if not ordered:
        return ["Claims — count not recorded for this dataset"]

    skipped = sum(dataset.claims_skipped.values())
    lines = [f"Claims — {ordered} ordered, {built} built, {skipped} not built"]
    if skipped != ordered - built:
        lines.append(
            f"  !! {ordered - built} claim(s) are missing but {skipped} are accounted for; "
            "the difference is unexplained"
        )
    # Sorted by reason so the report is byte-identical between runs of the same seed.
    for reason, count in sorted(dataset.claims_skipped.items()):
        lines.append(f"  {count:>4}  {reason}")
    return lines


def _cause_lines(dataset: Dataset) -> list[str]:
    """`partially_covered` by cause, counted the way policy.yaml defines it.

    `partially_covered_causes` splits the bucket **per claim** and sums to 1.0. Counting
    occurrences instead would put a claim that carries both causes in two rows, and the
    printed shares could then never converge on the target however large the run — a
    number that cannot reach its target is worse than no number, because it reads as a
    miss rather than as a category error.

    A claim with both causes therefore gets its own row and is attributed to neither
    target. Which of the two "really" caused it is not something policy.yaml answers, and
    picking one would be an invented tie-break sitting inside a report about balance.
    """
    causes = list(partially_covered_causes())
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(cause for cause in claim.imperfection if cause in causes)
        for claim in dataset.claims
    )
    del counts[()]  # claims with no cause are not in the partially_covered bucket
    total = sum(counts.values())

    lines = [
        f"partially_covered by cause — {total} claim(s); "
        "policy.yaml splits this bucket per claim, not per occurrence"
    ]
    for cause, share in partially_covered_causes().items():
        count = counts[(cause,)]
        lines.append(
            f"  {cause + ' only':<26} {count:>4}  {_share(count, total):>6}   target {share:.1%}"
        )
    both = total - sum(counts[(cause,)] for cause in causes)
    lines.append(
        f"  {'both causes on one claim':<26} {both:>4}  {_share(both, total):>6}"
        "   policy.yaml declares no share for this"
    )
    return lines


def _insufficient_evidence_cause_lines(dataset: Dataset) -> list[str]:
    """`insufficient_evidence` by cause, and WHAT THE CORPUS DOES NOT CONTAIN.

    A second cause block rather than a generalization of the one above, because the two verdicts
    differ in the thing that matters here: `partially_covered`'s causes can occur together on one
    claim and are counted per claim for that reason, while these are mutually exclusive by
    construction — a pair either disagrees about the amount or is dated backwards, and the planner
    draws one.

    🔴 THE THIRD CAUSE IS NAMED THOUGH IT NEVER OCCURS. `subject_not_evidenced` reaches this verdict
    too and carries no share in policy.yaml, because nothing can plan a deliberately incomplete
    claim. A report listing only what happened would let a reader take two causes for the whole
    vocabulary, which is the reading `known_limitations` KL-07 exists to prevent.
    """
    shares = insufficient_evidence_causes()
    counts = Counter(
        cause
        for claim in dataset.claims
        if claim.verdict is Verdict.INSUFFICIENT_EVIDENCE
        for cause in claim.imperfection
    )
    total = sum(counts.values())

    lines = [
        f"insufficient_evidence by cause — {total} claim(s); the two cross-check causes are "
        "mutually exclusive"
    ]
    for cause, share in shares.items():
        count = counts[cause]
        lines.append(
            f"  {cause:<26} {count:>4}  {_share(count, total):>6}   target {share:.1%}"
        )
    for cause in sorted(set(counts) - set(shares)):
        lines.append(
            f"  {cause:<26} {counts[cause]:>4}  {_share(counts[cause], total):>6}"
            "   !! realized with no share declared for it in policy.yaml"
        )
    lines.append(
        f"  {'subject_not_evidenced':<26} {counts['subject_not_evidenced']:>4}"
        "         NOT DRAWN — policy.yaml declares no share, because nothing can plan a"
    )
    lines.append(
        "                                          deliberately incomplete claim. See KL-07."
    )
    return lines


def _drift_lines(dataset: Dataset) -> list[str]:
    """Where the label differs from the verdict that was drawn, split by what it means.

    The two directions are not the same event and must not share a counter:

    * a `covered` target labelled `partially_covered` is the ledger binding — the oracle
      overruling the plan, which is the cumulative-limit mechanism working as designed;
    * a `partially_covered` target labelled `covered` is a builder shortfall — the basket
      `claim_planner._overrun_item_count` sized from an *estimate* of a line's value did
      not exceed the remaining balance. Nothing is mislabelled, but the run contains fewer
      of that mechanism than was asked for, and that is a defect to fix rather than a
      policy event to note.

    `Dataset.plans` may be empty for a dataset assembled without them, in which case there
    is nothing to compare and the report simply says so instead of raising.
    """
    if not dataset.plans:
        return ["(no plans recorded for this dataset — target-vs-realized comparison skipped)"]

    pairs = list(zip(dataset.plans, dataset.claims, strict=True))
    overruled = [c for p, c in pairs if p.verdict is Verdict.COVERED and p.verdict is not c.verdict]
    shortfall = [
        (p, c) for p, c in pairs
        if p.verdict is Verdict.PARTIALLY_COVERED and c.verdict is Verdict.COVERED
    ]

    lines: list[str] = []
    if overruled:
        lines.append(
            f"  {len(overruled)} claim(s) drawn as `covered` were labelled "
            f"{overruled[0].verdict.value} by a binding annual limit — the oracle "
            "overruling\n  the target, which is the limit mechanism doing its job."
        )
    if shortfall:
        named = ", ".join(f"{c.claim_id} ({p.cause})" for p, c in shortfall[:5])
        more = "" if len(shortfall) <= 5 else f" and {len(shortfall) - 5} more"
        lines.append(
            f"  {len(shortfall)} claim(s) drawn as `partially_covered` came out `covered`: "
            f"{named}{more}.\n  The basket sized from an estimated line value did not "
            "overrun the balance — a builder\n  shortfall, not a policy event."
        )
    return lines


def _share(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "—"


def _target_share(verdict: Verdict, mix: dict[Verdict, float | None]) -> str:
    """A verdict and its target share, for a verdict this run did not generate.

    Three cases, kept apart because they mean different things to whoever reads the report:
    a declared share, a member policy.yaml lists with the share still undecided, and a
    member of the enum that `verdict_mix` does not mention at all. Printing `0.0%` for
    either of the last two would state a decision nobody made.
    """
    if verdict not in mix:
        return f"{verdict.value} (not in verdict_mix)"
    share = mix[verdict]
    if share is None:
        return f"{verdict.value} (no share declared yet)"
    return f"{verdict.value} ({share:.1%})"


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
