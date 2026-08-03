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
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from receipt_synth import __version__
from receipt_synth.claim_planner import (
    ARCHETYPES,
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
    DocumentReference,
    PartyIdentity,
    build_bank_statement,
    build_invoice,
    build_payment_confirmation,
    build_prro_receipt,
    draw_party_identity,
    resolve_vendor,
    vendor_can_carry,
)
from receipt_synth.degrader import clipped_edges, degrade
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
    DocType,
    Persona,
    Split,
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

# The key the content extent rides under while it goes through the degrader's geometry, and which
# is removed again before the label is written. Two leading underscores so it cannot collide with a
# `data-field` name — those are printed-field names, and none begins that way — and the collision is
# checked rather than assumed.
_CONTENT_BBOX_KEY = "__content_extent__"

# Stamped into a `tEXt` chunk of every shipped PNG (see `_write_png`), so that a viewer who
# encounters one outside this repository — cropped into a slide, forwarded in a chat — can tell
# by inspecting the file that it is not a real document. Metadata only: it never touches a pixel.
SYNTHETIC_DATA_MARKER = (
    "SYNTHETIC TEST DATA — NOT VALID PROOF OF PAYMENT — github.com/VOSolyanik/receipt-synthesis"
)

# HOW A DOCUMENT REACHED THE VERIFIER — drawn per document, UNIFORMLY over the three channels.
#
# 🔴 UNIFORM IS A PLACEHOLDER, NOT A MEASUREMENT, and saying so is the whole of the comment. No
# survey of how real reimbursement evidence arrives was available to this repository, so any other
# split would be a made-up frequency wearing the authority of a config key. An equal draw claims
# nothing about the world and makes every channel large enough to measure, which is what a corpus
# built for evaluation needs from it.
#
# 🔴 WHY IT IS IN CODE AND NOT IN A CONFIG FILE, which is a question this repository normally
# answers the other way. A share that decides what fraction of the corpus carries a given LABEL
# VALUE sizes a labelled bucket, and config/generation.yaml says in its own words that such a share
# belongs beside `verdict_mix` in config/policy.yaml — `mismatch.delta_range` is there precisely
# because it changes no label. `capture` IS a label. So the honest home for a capture mix is
# policy.yaml, that file is the author's to change, and inventing a home for it in the nearest
# file that would accept it is how a config split stops meaning anything. Recorded as an open
# question rather than settled by whoever was passing.
#
# The tuple order is part of the seed's meaning: reordering it changes which document gets which
# channel for a given seed, exactly as reordering any drawn-from list in this repository does.
CAPTURE_CHANNELS = (Capture.SCREENSHOT, Capture.PHOTO, Capture.SCAN)

# 🔴 THE MINIMUM NUMBER OF DOCUMENTS A PER-CLASS FIGURE MAY BE QUOTED ON. Below it a per-class
# accuracy is not a measurement: at p ≈ 0.9 and n = 30 the 95% Wilson interval is about ±0.10, so
# "0.91" and "0.85" are the same reading. THE THRESHOLD IS REPORTED AND NEVER ENFORCED — nothing
# here resizes a run or reweights a draw to reach it, because a corpus tuned until its report looks
# healthy is a corpus whose report says nothing. The shortfall is made visible and left.
MIN_DOCUMENTS_PER_TARGET_CLASS = 30


def assign_splits(
    persona_ids: Sequence[str], *, seed: int, train_fraction: float
) -> dict[str, Split]:
    """Which side of the partition each persona is on. See `schemas.Split` for the unit.

    🔴 `train_fraction` HAS NO DEFAULT, HERE OR ANYWHERE ABOVE. It is a decision about the
    MEASUREMENT, and a default would let a run be performed without that decision ever having been
    declared — the argument `--seed` already wins in this repository, applied to a partition nobody
    declared. A caller has to pass one; what to pass, and why a half is the answer for a consumer
    that trains nothing, is in README.md's flag table and in `docs/architecture.md` under the
    assembler. The binding constraint is `MIN_DOCUMENTS_PER_TARGET_CLASS` above: a per-class figure
    needs that many documents ON THE SIDE IT IS MEASURED ON, so the thinnest class sets how small
    the validation side may ever be.

    🔴 SEEDED INDEPENDENTLY OF THE GENERATOR'S OWN DRAW, from `f"split:{seed}"` rather than from
    the root generator. That is what makes the partition a LABELLING OF AN EXISTING CORPUS instead
    of a change to it: the documents a seed produces are byte-identical with and without this
    step, so a corpus generated before the partition existed can be partitioned without being
    regenerated. Drawing from `root` would have shifted every document downstream of the first
    call — the ordinary cost of touching a seeded stream, paid here for nothing.

    `random.Random` seeds from a string through SHA-512 of its bytes, so the assignment does not
    depend on `PYTHONHASHSEED` and is reproducible across machines and interpreter runs.

    ⚠️ ADDING A PERSONA RESHUFFLES THE WHOLE PARTITION, and that is inherent rather than a defect
    of this implementation: a partition is a property of the SET, and any rule that kept earlier
    personas in place would have to place later ones by a fixed criterion, which stops honouring
    the requested fraction. What the generator does promise is reproducibility for a given seed and
    size, and that holds exactly.

    NO SIDE IS TOPPED UP TO BE NON-EMPTY. A run of one persona has an empty validation side, and
    the report says so — see `_split_lines`. Forcing a document into an empty side would satisfy
    the shape of a split while producing a validation set of one persona, which is worse than
    nothing precisely because it looks like something.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction is a proportion strictly between 0 and 1, not {train_fraction!r}; "
            "a run with everything on one side is not a partition"
        )
    shuffled = list(persona_ids)
    random.Random(f"split:{seed}").shuffle(shuffled)
    train_size = round(len(shuffled) * train_fraction)
    return {
        persona_id: Split.TRAIN if index < train_size else Split.VALIDATION
        for index, persona_id in enumerate(shuffled)
    }


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
    # Persona id -> which side of the train / validation partition. Empty for a dataset assembled
    # without a partition, which is a different state from a partition that put everything on one
    # side. See `assign_splits`.
    split: dict[str, Split] = field(default_factory=dict)
    # What was ASKED FOR, kept beside what was realized. The two differ by rounding on any run
    # whose persona count does not divide evenly, and a report that printed only the realized share
    # would leave a reader unable to tell a rounding from a mistake.
    train_fraction: float | None = None

    def as_manifest(self) -> dict:
        """The full index: every persona, claim and document, plus the partition.

        THE PARTITION IS DESCRIBED HERE AND APPLIED ON THE RECORDS. This block carries the
        DECISION — the unit, the fraction asked for, the realized counts — and each claim and
        document carries its own `split`. It deliberately does NOT repeat the ids: a list here
        beside a field there would be two statements of one fact, and the first edit to either
        makes them disagree with nothing to notice it.
        """
        manifest = {
            "generator_version": __version__,
            "seed": self.seed,
            "synthetic": True,
            "split": self._split_manifest(),
            "personas": [persona.model_dump(mode="json") for persona in self.personas],
            "claims": [claim.model_dump(mode="json") for claim in self.claims],
            "documents": [document.model_dump(mode="json") for document in self.documents],
        }
        return manifest

    def _split_manifest(self) -> dict | None:
        """The partition as a decision plus its realized sizes, or `None` if none was computed.

        `None` rather than an empty structure: no partition and a partition with an empty side are
        different states, and a consumer must be able to tell them apart without counting.
        """
        if not self.split:
            return None
        counts = {
            side.value: {
                "personas": sum(1 for value in self.split.values() if value is side),
                "claims": sum(1 for claim in self.claims if claim.split is side),
                "documents": sum(1 for document in self.documents if document.split is side),
            }
            for side in Split
        }
        return {
            "unit": "persona",
            "why_this_unit": (
                "annual limits are cumulative per persona, so a claim's verdict can depend on "
                "that persona's earlier claims; a per-claim partition would make a validation "
                "label a function of training data. See schemas.Split."
            ),
            "stratified": False,
            "train_fraction_requested": self.train_fraction,
            "realized": counts,
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


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a BGR image (OpenCV's convention) to `path`, stamped with `SYNTHETIC_DATA_MARKER`.

    THE LAST SAVE POINT FOR A SHIPPED IMAGE, and the only one: `renderer.render` also writes a
    PNG, but only to a temporary staging file that `_build_document` deletes before returning,
    so nothing downstream ever sees it; `degrader.degrade` never touches disk — it hands back a
    numpy array. This function is therefore the single place a PNG that lands in `out_dir/images`
    is written, which is what makes stamping it here sufficient for 100% of the corpus.

    PIL rather than `cv2.imwrite`: OpenCV's PNG writer has no `tEXt`-chunk support. The chunk is
    metadata appended to the file; it does not touch a pixel, so the decoded image is unchanged.
    """
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", SYNTHETIC_DATA_MARKER)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, pnginfo=info)


def _build_document(
    rng: random.Random,
    *,
    persona: Persona,
    plan: ClaimPlan,
    document_plan: DocumentPlan,
    vendor: dict,
    identity: PartyIdentity,
    doc_id: str,
    renderer: Renderer,
    out_dir: Path,
    settles: Decimal | None = None,
    cites: DocumentReference | None = None,
) -> tuple[DocGroundTruth, DocumentReference | None]:
    """One document of a claim, and the reference by which another document of it can cite this one.

    The second half of the return value is `None` for every class but the invoice: a payment
    document is not cited by anything in its own claim, and a fiscal receipt is the whole claim.

    `settles` is THE AMOUNT THIS DOCUMENT'S CLAIM IS ABOUT, and it is required for a document that
    proves the payment and ignored by one that states the subject. The subject document decides the
    amount — its basket is drawn first and summed — and the payment document is then told what it
    settles. The order is not an accident of the loop: a claim's money is a property of what was
    bought, so the document that lists the purchase is the one that fixes it.

    `cites` travels the same route and for the same reason: the subject document is built first, and
    what a payment's purpose line names is a property of the claim rather than of the page. Passed
    to the payment class and ignored by the subject class, which cites nothing — an invoice is
    issued before there is a payment to point at.

    `vendor` is passed in rather than chosen here. It is the claim's vendor instance, and
    every document of the claim has to name the same seller — while a sole trader's name
    was a stored constant that held by the nature of the type, and a drawn name can differ,
    so it is now a constraint somebody has to keep.

    `identity` is that seller's code, account and bank, drawn once for the claim beside the vendor.
    The `vendor` constraint was solved for the NAME alone, and every other identifier of one seller
    went on being drawn per document — 587 pairs of the delivered corpus, 587 disagreements. See
    `content_builder.PartyIdentity` and docs/cross-document-fields.md.

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

    # ONE CAPTURE CHANNEL PER DOCUMENT, DECIDED ONCE. It reaches three places — the builder, which
    # prints a requisite that depends on the medium; the degrader, which applies that channel's
    # artefacts; and the label. Two literals for one fact is how they come to disagree.
    capture = rng.choice(CAPTURE_CHANNELS)

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
            "identity": identity,
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
        else:
            # 🔴 THE CAPTURE CHANNEL REACHES THE BUILDER, not only the degrader. 👁 The VAT summary
            # row of a fiscal receipt takes one form on paper and either of two electronically, so
            # the MEDIUM a document will be captured on decides a requisite that is printed while
            # the document is built. Passed to the class that has the observed variation and to no
            # other: nothing analogous has been observed on the invoice.
            basket |= {"capture": capture}
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
            identity=identity,
            payer_name=persona.full_name,
            payer_tax_id=persona.tax_id,
            amount=settles,
            cites=cites,
        )

    image_path = out_dir / "images" / f"{doc_id}.png"
    with tempfile.TemporaryDirectory() as staging:
        # The clean render is an intermediate, not an artifact: the dataset ships the
        # document as it would have been captured.
        clean = renderer.render(slug, document.render_context(), Path(staging) / f"{doc_id}.png")
        # 🔴 THE CONTENT EXTENT TRAVELS WITH THE FIELD BOXES, THROUGH THE SAME TRANSFORM. Geometry
        # is Albumentations' alone (see `degrader`), and a box moved by a second route would drift
        # from the fields the moment a real geometric step arrives — which is exactly when a
        # measurement built on it would start being quietly wrong.
        if _CONTENT_BBOX_KEY in clean.field_bboxes:
            raise ValueError(
                f"a template marks a field named {_CONTENT_BBOX_KEY!r}, which this module reserves "
                "for the content extent; rename the `data-field`"
            )
        moved = degrade(
            cv2.imread(str(clean.image_path)),
            {**clean.field_bboxes, _CONTENT_BBOX_KEY: clean.content_bbox},
            seed=rng.getrandbits(32),
            capture=capture,
        )
        boxes = dict(moved.field_bboxes)
        content_bbox = boxes.pop(_CONTENT_BBOX_KEY)
        # 🔴 GATE 2 OF THE FIDELITY MATRIX: DID THE CONTENT SURVIVE THE CAPTURE. Measured against
        # the DEGRADED image, because that is the file a consumer receives, and from the CONTENT
        # extent rather than from the field boxes — a document whose every labelled field came
        # through while a crop took the footer with the fiscal wording would report as complete
        # measured on the fields, and would then hand a system a falsely high character error rate
        # for text that is not in the picture.
        height, width = moved.image.shape[:2]
        lost = clipped_edges(content_bbox, width, height)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _write_png(image_path, moved.image)

    return (
        document.ground_truth(
            doc_id=doc_id,
            source_file=image_path.name,
            capture=capture,
            field_bboxes=boxes,
            reference_text=clean.reference_text,
            content_bbox=content_bbox,
            content_lost_edges=lost,
        ),
        # Asked of the document rather than composed here: what a reference to an invoice consists
        # of is the invoice's business. A class that nothing cites has no such property and returns
        # nothing, so registering one does not bring this branch a case to handle.
        getattr(document, "reference", None),
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
    train_fraction: float,
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

    `train_fraction` IS REQUIRED AND SITS WITH `seed` FOR THE SAME REASON — a run performed
    without the partition ever having been declared documents a measurement nobody chose. It has
    no default at this entry point either: a default here would be the same defect one layer below
    the command line, where it would be harder to see. See `assign_splits`.
    """
    if claims_per_persona < 1:
        raise ValueError(f"a persona files at least one claim, not {claims_per_persona}")
    # Validated BEFORE the run rather than at the partition step at the end. A bad fraction would
    # otherwise be reported after every image had been rendered, which on a production run is an
    # hour and a half spent to learn that an argument was mistyped.
    assign_splits([], seed=seed, train_fraction=train_fraction)

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
                # And WHO THAT VENDOR IS ON PAPER, drawn here for the same reason and in the same
                # place. The name was fixed per claim and the code, the account and the bank were
                # not, so two documents of one purchase named one seller by four different numbers
                # — on every pair of the delivered corpus. The constraint was known; it had been
                # applied to one field.
                identity = draw_party_identity(rng, vendor, persona.location.country.value)
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
                cites: DocumentReference | None = None
                for index, document_plan in enumerate(plan.documents, start=1):
                    document, reference = _build_document(
                        rng,
                        persona=persona,
                        plan=plan,
                        document_plan=document_plan,
                        vendor=vendor,
                        identity=identity,
                        doc_id=f"{plan.claim_id}_d{index}",
                        renderer=renderer,
                        out_dir=out_dir,
                        settles=settles,
                        cites=cites,
                    )
                    if evidence_of(document_plan.archetype).proves_subject:
                        settles = _amount_the_payment_states(rng, plan, document.amount)
                        # What the payment document will cite. It travels beside `settles` because
                        # it is the same kind of fact — a property of the claim that the subject
                        # document decides and the payment document has to be told.
                        cites = reference
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

    # THE PARTITION IS APPLIED LAST, ON A FINISHED CORPUS, because its unit is the persona and no
    # persona is complete until its own planning has stopped. It changes no pixel and no drawn
    # value — see `assign_splits` on why it has its own generator — so this is a labelling pass over
    # records that already exist.
    splits = assign_splits(
        [persona.persona_id for persona in all_personas],
        seed=seed,
        train_fraction=train_fraction,
    )
    # A document record carries no `persona_id`, so its side is joined through the claim that owns
    # it — the same join a consumer makes, and the only one available. Built once rather than
    # searched per document.
    owner = {
        doc_id: claim.persona_id for claim in all_claims for doc_id in claim.documents
    }
    all_claims = [
        claim.model_copy(update={"split": splits[claim.persona_id]}) for claim in all_claims
    ]
    all_documents = [
        document.model_copy(update={"split": splits[owner[document.doc_id]]})
        for document in all_documents
    ]

    dataset = Dataset(
        seed=seed,
        personas=all_personas,
        claims=all_claims,
        documents=all_documents,
        claims_ordered=personas * claims_per_persona,
        claims_skipped=dict(skipped),
        plans=all_plans,
        split=splits,
        train_fraction=train_fraction,
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

    🔴 TWO MARKER VOCABULARIES, AND THEY MUST NOT MERGE. The report says two different kinds of
    thing and a reader has to be able to tell them apart at a glance:

      `!!`   THE REPORT CONTRADICTS ITSELF, or the run reached a state nothing should produce —
             rows that do not sum to their own header, a verdict realized that the planner cannot
             draw, claims missing that nothing accounts for. It should never fire, and if it does
             the TABLE is wrong rather than the dataset.
      words  A FINDING ABOUT THE CORPUS, in capitals and in plain English: ABSENT, EMPTY,
             BELOW <n>, ONE VALUE ACROSS THE WHOLE CORPUS, THIS SIDE CONTAINS NO <x>. These are
             expected to fire — a corpus that never trips one is a corpus nobody stressed — and
             they are not defects in the report.

    The distinction is load-bearing rather than cosmetic: a test asserts that `!!` is ABSENT from a
    healthy run's report, which is only meaningful while `!!` means the first thing. Marking a
    below-threshold class with `!!` broke that test the moment the class block landed, and the
    lesson is that the sigil is a reserved word, not emphasis.

    The report covers the verdict axis, the claim count, the exclusion note, imperfection causes,
    document classes, currency, language, capture channels and the train / validation partition.
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

    lines += _document_class_lines(dataset)
    lines += _value_dimension_lines(dataset, "Currency", lambda d: d.currency)
    lines += _value_dimension_lines(dataset, "Language", lambda d: d.language)
    lines += _capture_lines(dataset)
    lines += _split_lines(dataset)
    lines += _drift_lines(dataset)
    return "\n".join(lines)


def _document_class_lines(dataset: Dataset) -> list[str]:
    """The distribution over document classes, against the per-class minimum — REPORTED, NOT MET.

    🔴 THERE IS NO TARGET SHARE TO COMPARE AGAINST, AND THE ABSENCE IS DELIBERATE.
    config/policy.yaml declares `verdict_mix` and explicitly REFUSES a `document_mix`, on the
    ground that a share of receipts against invoices would read as an observation about which
    documents claimants actually submit — which nothing here has measured. So this block reports
    what a run produced and measures it against one thing only: a MINIMUM below which a per-class
    figure should not be quoted at all.

    🔴 AND NOTHING TUNES ANYTHING TO REACH IT. The threshold applies to the delivered corpus and is
    checked after it is generated; this block makes a shortfall visible and leaves it. A generator
    that resized a run until its own report looked healthy would produce a report that could not
    fail, which is a report nobody reads.

    A CLASS WITH NO REGISTERED ARCHETYPE IS ABSENT, NOT ZERO. `DocType` names seven classes and the
    registry can build four; the other three cannot appear in any run at any size, so a count of 0
    beside a threshold would invite somebody to fix a shortfall that no run can close. The minimum
    is stated only for the classes it can be asked of.
    """
    buildable = {archetype.doc_type for archetype in ARCHETYPES.values()}
    counts: Counter[DocType] = Counter(document.doc_type for document in dataset.documents)
    total = sum(counts.values())

    lines = [
        f"Document classes — {total} document(s); target ≥ {MIN_DOCUMENTS_PER_TARGET_CLASS} "
        "per buildable class,",
        "  REPORTED AND NEVER TUNED TO — policy.yaml declares no document mix, on purpose",
    ]
    for doc_type in DocType:
        count = counts[doc_type]
        if doc_type not in buildable:
            lines.append(
                f"  {doc_type.value:<22}    0         ABSENT — no archetype builds this class, so"
                " no run of any size"
            )
            lines.append(
                "                                  contains one; the minimum does not apply to it"
            )
            continue
        mark = (
            "ok" if count >= MIN_DOCUMENTS_PER_TARGET_CLASS
            else f"BELOW {MIN_DOCUMENTS_PER_TARGET_CLASS} — do not quote a per-class figure here"
        )
        lines.append(
            f"  {doc_type.value:<22} {count:>4}  {_share(count, total):>6}   {mark}"
        )
    return lines


def _value_dimension_lines(dataset: Dataset, title: str, of) -> list[str]:
    """One flat dimension of the corpus — currency, language — with its denominator.

    🔴 A DIMENSION WITH ONE VALUE IS SAID TO HAVE ONE VALUE. Printing `UAH 100.0%` and stopping
    reads as a balanced distribution that happens to have one member, which is exactly the
    reassuring shape this report is not allowed to take: a corpus of a single currency does not
    EXERCISE the currency dimension at all, and a consumer reporting per-currency accuracy on it
    would be reporting the corpus average under another name.

    An empty corpus is reported as having no documents rather than as a distribution over nothing.
    """
    counts: Counter[str] = Counter(of(document) for document in dataset.documents)
    total = sum(counts.values())

    lines = [f"{title} — {total} document(s)"]
    if not total:
        lines.append(f"  ABSENT — this run produced no documents, so there is no {title.lower()}")
        return lines
    for value, count in sorted(counts.items()):
        lines.append(f"  {value:<22} {count:>4}  {_share(count, total):>6}")
    if len(counts) == 1:
        lines.append(
            f"  ONE VALUE ACROSS THE WHOLE CORPUS: this run does not exercise the "
            f"{title.lower()} dimension."
        )
        lines.append(
            "     A per-value figure computed on it is the corpus average under another name."
        )
    return lines


def _split_lines(dataset: Dataset) -> list[str]:
    """The train / validation partition, and WHAT EACH SIDE IS MISSING.

    The sizes alone would be the reassuring half. The half that matters is the last one: any
    verdict or document class the corpus contains and a side does not. The partition is by persona
    and is NOT stratified — see `schemas.Split` — so on a small run a whole verdict can land on one
    side, and nothing else in this report would say so.

    ⚠️ AN EMPTY SIDE IS DECLARED AS EMPTY, not printed as 0.0%. A run of one persona has no
    validation set at all; that is a property of the run's size, and a percentage would present it
    as a partition that happens to be lopsided.
    """
    if not dataset.split:
        return [
            "Train / validation split — NOT COMPUTED for this dataset, which is not the same as "
            "an empty split"
        ]

    requested = dataset.train_fraction
    lines = [
        f"Train / validation split — by PERSONA, {requested:.0%} of personas requested for train;"
        " not stratified",
    ]
    corpus_verdicts = {claim.verdict for claim in dataset.claims}
    corpus_classes = {document.doc_type for document in dataset.documents}

    for side in Split:
        claims = [claim for claim in dataset.claims if claim.split is side]
        documents = [d for d in dataset.documents if d.split is side]
        personas = sum(1 for value in dataset.split.values() if value is side)
        if not personas:
            lines.append(
                f"  {side.value:<12}    0 persona(s)   EMPTY — the run is too small to partition at"
                f" {requested:.0%}."
            )
            lines.append(
                "                                 Not a share of zero: there is no such side here."
            )
            continue
        lines.append(
            f"  {side.value:<12} {personas:>4} persona(s) {len(claims):>5} claim(s) "
            f"{len(documents):>5} document(s)   "
            f"{_share(len(claims), len(dataset.claims))} of claims"
        )
        missing_verdicts = corpus_verdicts - {claim.verdict for claim in claims}
        missing_classes = corpus_classes - {document.doc_type for document in documents}
        for label, missing in (("verdict", missing_verdicts), ("document class", missing_classes)):
            if missing:
                named = ", ".join(sorted(item.value for item in missing))
                lines.append(
                    f"    THIS SIDE CONTAINS NO {named} — a {label} the corpus has and this "
                    "side does not."
                )
                lines.append(
                    "       Nothing measured on it can report that "
                    f"{label}; the partition is not stratified."
                )
    return lines


def _capture_lines(dataset: Dataset) -> list[str]:
    """How the corpus is split across the capture channels, AND WITH WHAT DENOMINATOR EACH
    CHANNEL COULD CARRY A CHARACTER ERROR RATE.

    🔴 A CHANNEL THAT PRODUCED NO DOCUMENT IS REPORTED ABSENT, NEVER AS ZERO. The two are
    different statements and only one of them can be true at a time: `0.0%` complete says a
    measurement was taken and came out at nothing, while absence says no measurement exists.
    Folding the first into the second is how an empty cell becomes a data point in somebody's
    table, and nobody re-derives it afterwards.

    🔴 AND THE SUBSET IS PART OF THE RESULT. `reference_text` is the reference side of a
    character error rate, and that rate is only defined where the text it references is
    actually in the picture. "CER on photo = X" is not a result; "CER on the N of M photos
    whose content survived" is. So this block prints the denominator beside the share, because
    a consumer that reads only the metric will never learn it anywhere else.

    ⚠️ Complete here means THE TEXT survived — see `DocGroundTruth.content_complete`. A capture
    that cut off a QR code while keeping every character is counted complete, and truthfully.
    """
    by_channel: Counter[Capture] = Counter(doc.capture for doc in dataset.documents)
    complete: Counter[Capture] = Counter(
        doc.capture for doc in dataset.documents if doc.content_complete
    )
    unmeasured: Counter[Capture] = Counter(
        doc.capture for doc in dataset.documents if doc.content_complete is None
    )
    total = sum(by_channel.values())

    lines = [
        f"Capture channels — {total} document(s); `complete` is the subset on which a "
        "character error",
        "  rate is defined at all, and it is about TEXT: a lost QR or stamp is not counted here",
    ]
    for channel in Capture:
        count = by_channel[channel]
        if not count:
            # ABSENT, not zero — see the docstring. There is no denominator here, so there is
            # no share and no completeness figure to print.
            lines.append(
                f"  {channel.value:<12}    0         ABSENT — no document of this run took this "
                "channel, which is"
            )
            lines.append(
                "                                  not the same statement as a completeness of 0%"
            )
            continue
        line = (
            f"  {channel.value:<12} {count:>4}  {_share(count, total):>6}"
            f"   complete {complete[channel]}/{count}"
        )
        if unmeasured[channel]:
            line += f", {unmeasured[channel]} unmeasured (no content extent)"
        lines.append(line)
    return lines


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
