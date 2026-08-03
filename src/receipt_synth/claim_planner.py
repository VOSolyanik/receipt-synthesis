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

from receipt_synth.config import coverage_targets, load_policy
from receipt_synth.content_builder import MAX_LINE_ITEMS, estimated_line_value
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    PAYMENT_PRECEDES_SUBJECT,
    ClaimEvaluation,
    Evidence,
    Ledger,
    document_evidence,
    insufficient_evidence_causes,
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

    What a template proves is read from `document_evidence` in policy.yaml, keyed by
    `doc_type` — see `evidence_of`. That is what lets the planner assemble a claim from
    several documents: it takes archetypes until both facts are satisfied.

    NO PER-ARCHETYPE OVERRIDE, and its absence is a decision rather than an omission.
    policy.yaml notes that a specific archetype may differ from its type's default — a bank
    confirmation whose payment purpose spells out what was bought does prove the subject,
    unlike a bare transfer — and an override declared here could not be honoured today: the
    verdict is derived by `policy_engine` from `DocGroundTruth`, which records a document's
    TYPE and not the archetype that produced it, so the engine would go on applying the
    default and label the claim by a rule the plan had overridden. Carrying the role per
    document is a change to the label shape, and it is listed as an open decision in
    config/labelling-schema.yaml. Until it is taken, an archetype whose evidence differs
    from its type's default must not be registered.
    """

    slug: str
    doc_type: DocType
    country: Country
    language: str
    # Benefit categories this template can actually carry. A pharmacy receipt cannot
    # print a gym membership. As templates land this widens until every category has at
    # least one archetype in every jurisdiction.
    categories: tuple[str, ...]


def evidence_of(archetype: Archetype) -> Evidence:
    """What a document built from this archetype establishes, per policy.yaml."""
    return document_evidence(archetype.doc_type)


# The registry the planner selects from. Three entries today, all of them Ukrainian fiscal
# receipts; the rest register here as their templates land.
#
# THREE ENTRIES OF ONE DOCUMENT CLASS ARE STILL THREE ARCHETYPES. They carry the same
# `doc_type`, so they establish the same facts and the planner treats them as interchangeable —
# what differs is the paper width and the kind of cash register, which the class of the document
# does not depend on. That is the point of the registry being keyed by slug rather than by type:
# `_select_documents` picks among them, so a claim's receipt is now drawn from three renderings
# rather than always being the same one.
ARCHETYPES: dict[str, Archetype] = {
    # 80 mm ПРРО — the software register.
    "ua_prro_receipt": Archetype(
        slug="ua_prro_receipt",
        doc_type=DocType.FISCAL_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=("vitamins_nutrition",),
    ),
    # The same register on the narrow 58 mm roll, where long names wrap and the amount column
    # moves. A width, not a document class.
    "ua_prro_receipt_58mm": Archetype(
        slug="ua_prro_receipt_58mm",
        doc_type=DocType.FISCAL_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=("vitamins_nutrition",),
    ),
    # The classic hardware РРО: «ЗН» beside «ФН», a sequential receipt number, no online marker.
    "ua_rro_receipt": Archetype(
        slug="ua_rro_receipt",
        doc_type=DocType.FISCAL_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=("vitamins_nutrition",),
    ),
    # THE SECOND DOCUMENT CLASS, and the first that establishes only one of the two facts a
    # reimbursement needs. A bank payment confirmation proves that money moved and says nothing
    # about what was bought — policy.yaml's `document_evidence`, confirmed by 👁 0 of 7 observed
    # payment purposes naming the subject of the expense.
    #
    # IT CARRIES EVERY CATEGORY, and that is a consequence of the class rather than a generous
    # guess. The other archetypes are limited by what a shop can sell — a pharmacy receipt cannot
    # print a gym membership — and this document PRINTS NO ITEMS AT ALL, so there is nothing on it
    # that any category could contradict. Every id below is a category of config/policy.yaml and
    # the two lists are checked against each other by a test, so a category added there cannot
    # silently drop out of this tuple.
    #
    # ⚠️ NO CLAIM CAN BE ASSEMBLED FROM IT YET, and the reason is structural rather than a
    # shortcoming of this entry: a claim needs both facts, this archetype supplies one, and the
    # archetype that supplies the other — an invoice — is not written. `documentable_categories`
    # therefore reports none of these categories as documentable, which is why registering it
    # changes no dataset. It is registered all the same: the template, the builder and the label
    # fields are what a later step pairs with an invoice, and an unregistered archetype is one
    # nothing renders and no test can reach.
    "ua_bank_payment_confirmation": Archetype(
        slug="ua_bank_payment_confirmation",
        doc_type=DocType.PAYMENT_CONFIRMATION,
        country=Country.UA,
        language="uk",
        categories=(
            "medical_insurance",
            "language_courses",
            "professional_development",
            "sport",
            "mental_health",
            "vitamins_nutrition",
            "hobby",
        ),
    ),
    # THE THIRD DOCUMENT CLASS, and the SECOND that proves the payment without stating what was
    # bought. 👁 A statement's payment purpose names an invoice or a delivery note, and at best a
    # generic category of goods — never the expense — which CONFIRMS the `proves_subject: false`
    # policy.yaml already gives this type rather than changing it.
    #
    # IT CARRIES EVERY CATEGORY for the reason the confirmation does: the page lists transactions
    # and no items, so there is nothing on it a category could contradict. The tuple is checked
    # against policy.yaml's categories by a test, so a category added there cannot silently drop
    # out of it.
    #
    # ⚠️ STILL NO CLAIM, and the reason is unchanged and structural: a claim needs both facts, this
    # supplies one, and the archetype that supplies the other — an invoice — is not written. Two
    # payment-proving archetypes are not better than one for that purpose. What it does change is
    # that `_select_documents` now has a CHOICE of payment archetype for the step that pairs them.
    "ua_bank_statement": Archetype(
        slug="ua_bank_statement",
        doc_type=DocType.BANK_STATEMENT,
        country=Country.UA,
        language="uk",
        categories=(
            "medical_insurance",
            "language_courses",
            "professional_development",
            "sport",
            "mental_health",
            "vitamins_nutrition",
            "hobby",
        ),
    ),
    # 🔴 THE FOURTH DOCUMENT CLASS, AND THE ONE THAT ACTIVATES THE DOMINANT PAIR. It is the exact
    # inverse of the two bank classes: an invoice states WHAT WAS BOUGHT and proves no payment —
    # 📄 a рахунок на оплату is an offer to pay, not a primary accounting document. So
    # `_select_documents`
    # can now assemble both facts from two documents for the first time, and a claim whose evidence
    # is split is buildable rather than merely modelled.
    #
    # IT CARRIES EVERY CATEGORY, and here that is a positive claim rather than the bank classes'
    # "nothing on the page can contradict one": an invoice LISTS ITEMS, so it carries a category
    # exactly when a basket can be drawn for it — and every category of policy.yaml has priced item
    # kinds and vendors that can carry both a covered and a mixed basket. Checked by a test rather
    # than asserted here.
    #
    # ⚠️ THE PAIR IS EXERCISED IN SIX CATEGORIES, NOT SEVEN, and the reason is worth knowing before
    # reading a dataset: `_select_documents` PREFERS a single document proving both facts, and the
    # three fiscal receipts carry `vitamins_nutrition`. So that category still gets a receipt and
    # the other six get an invoice plus a payment document. Registering the invoice for
    # `vitamins_nutrition` changes nothing about it today and is not a special case waiting to
    # happen: it is what a fourth receipt-less jurisdiction or a withdrawn receipt archetype would
    # need.
    #
    # 🔴 AND IT LEAVES THE REGISTRY WITH NO PAYMENT-ONLY CATEGORY AT ALL. That case — a category
    # covered by payment-proving archetypes alone, which `documentable_categories` must refuse —
    # was asserted from the registry until this entry landed, and is now asserted on a hand-built
    # archetype list in `tests/test_pipeline.py`, which is stronger: the mechanism stops depending
    # on a registry that changes every time a template lands.
    "ua_invoice": Archetype(
        slug="ua_invoice",
        doc_type=DocType.INVOICE,
        country=Country.UA,
        language="uk",
        categories=(
            "medical_insurance",
            "language_courses",
            "professional_development",
            "sport",
            "mental_health",
            "vitamins_nutrition",
            "hobby",
        ),
    ),
}


# --- what can be realized -----------------------------------------------------

# The verdicts this planner can build documents for. The others are refused rather than
# approximated: a planner that accepted one and produced an ordinary basket would write a
# wrong label instead of failing.
REALIZABLE_VERDICTS: tuple[Verdict, ...] = (
    Verdict.COVERED,
    Verdict.PARTIALLY_COVERED,
    # 🔴 THE THIRD, AND IT ARRIVED WITH A MECHANISM RATHER THAN WITH A TEMPLATE. A claim can now be
    # planned whose subject document and whose payment document DISAGREE — about the amount, or
    # about which came first — which is what `policy_engine._cross_checks` has always labelled and
    # what nothing could build until an archetype proving the subject alone existed. Only the two
    # CROSS-CHECK causes are drawn; see `_UNREALIZABLE_CAUSES` for the third and why it is not.
    Verdict.INSUFFICIENT_EVIDENCE,
)

# WHY A CAUSE OF A REALIZABLE VERDICT NEEDS ITS OWN TABLE. `_UNREALIZABLE_REASONS` below is keyed by
# VERDICT, and `insufficient_evidence` is realizable now — so the reason its third cause still
# cannot be built had nowhere to live, and would have vanished with the entry that moved. A verdict
# being realizable and every route to it being realizable are different statements, and the second
# is the one a reader of a corpus needs.
_UNREALIZABLE_CAUSES: dict[str, str] = {
    "subject_not_evidenced": (
        "is `insufficient_evidence` reached by a claim that establishes a movement of money and "
        "never what it bought — a bare payment confirmation or a bare statement. "
        "`policy_engine.resolve_evidence` labels such a claim today and NOTHING CAN PLAN ONE: "
        "`_select_documents` assembles both facts or refuses, `documentable_categories` reports a "
        "category it cannot complete as not documentable, and `ClaimPlan.subject_document` raises "
        "unless exactly one document states what was bought. Building it means planning "
        "DELIBERATELY INCOMPLETE evidence, which is a change to the claim model rather than to a "
        "template or a share — so config/policy.yaml declares no share for it, on the same "
        "reasoning that leaves `rejected` without one in `verdict_mix`"
    ),
}

_UNREALIZABLE_REASONS: dict[Verdict, str] = {
    Verdict.NOT_PROOF_OF_PAYMENT: (
        "is the MONEY-MOVED slot of `document_evidence` left unestablished, and needs a claim "
        "every document of which is a `proves_payment: false` type in policy.yaml — an invoice, "
        "an act, a sales slip. `policy_engine.resolve_evidence` labels such a claim today, and "
        "WHAT IS MISSING CHANGED WHEN THE INVOICE ARCHETYPE LANDED: an archetype of such a type "
        "now exists, so the gap is no longer the template. What does not exist is a planner that "
        "deliberately plans an INCOMPLETE claim — `_select_documents` assembles both facts or "
        "refuses, because the two verdicts this planner draws both need complete evidence, and "
        "`documentable_categories` reports a category it cannot complete as not documentable "
        "rather than letting such a plan be made. The document type is still the whole mechanism: "
        "amounts, baskets and dates are not consulted"
    ),
    Verdict.PARTIALLY_PAID: (
        "needs a document showing PART of an amount settled, and the statement archetype now "
        "registered does not narrow this: a statement row states the amount that moved, and a "
        "row worth less than the invoice beside it is a claim whose two documents disagree, "
        "which `policy_engine` already labels `insufficient_evidence` with the cause "
        "`amount_mismatch`. What is missing is what has always been missing — a document that "
        "states an amount OUTSTANDING against a total, which no archetype prints. `amount_due` is "
        "not that field: it is a receipt's ДО СПЛАТИ, the basket total less a discount plus cash "
        "rounding, and it says nothing about how much of an obligation is still open"
    ),
    Verdict.REJECTED: (
        "is the policy not covering a claim whose evidence is complete, on either of two axes, "
        "and the two are unreachable for DIFFERENT reasons — stated separately because one of "
        "them is weaker than it looks. By WHAT was bought: a basket drawn wholly from the "
        "`excluded_items` of the claimed category, so that the covered amount comes to zero. "
        "`policy_engine.verdict_for` already labels such a claim and NOTHING CAN BUILD ONE — "
        "`content_builder` always draws at least one covered line. By WHEN it was paid: a "
        "payment dated outside the active period of policy.yaml, which `evaluate_claim` labels "
        "with the cause `outside_period`. That one is merely NOT DRAWN, which is a weaker claim "
        "than not buildable: `_draw_date_in_period` draws inside the window, but `issued_at` is "
        "a public parameter with no guard on the period, so a caller naming an out-of-window "
        "date gets a plan the engine duly labels `rejected`. Either way `verdict_mix` in "
        "policy.yaml sets no share for this verdict yet, and a share invented before a "
        "mechanism exists would size a bucket nothing can fill"
    ),
}

# Reason of last resort, so that a verdict added to the enum without a plan gets a clean
# NotImplementedError naming itself rather than a KeyError from the table above.
_NO_REASON_RECORDED = (
    "is in the Verdict enum but has no entry in claim_planner._UNREALIZABLE_REASONS and "
    "no mechanism registered — whoever added it owes both"
)

# What fraction of a mixed basket is meant to be covered comes from
# `config/generation.yaml`, because it is a decision about the shape of the dataset's
# `covered_fraction` and not a mechanism this module implements. The reasoning for putting
# it in that file rather than beside `verdict_mix` in policy.yaml is stated at its head.
#
# An ASPIRATION, not a dial. `content_builder._repriced` clamps every non-covered line
# into its item kind's own price range, so a target the range cannot reach is not reached:
# a high target on a small covered side realizes lower, because the non-covered line cannot
# be priced below its floor. What the target *does* guarantee is the verdict — any
# non-covered line at all makes the claim partially covered — and that is the only thing a
# label depends on. Read the realized coverage off `covered_fraction`, never off the list.


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


def draw_verdict(
    rng: random.Random, realizable: tuple[Verdict, ...] = REALIZABLE_VERDICTS
) -> Verdict:
    """A target verdict from `verdict_mix`, restricted to what can be built.

    The shares are renormalized over the realizable subset — `random.choices` does that
    from the raw weights — so the drawn distribution is the target mix *conditioned* on
    that subset, not the target mix. Whoever reads the realized distribution has to be
    told which verdicts were excluded, or the conditioning is invisible.

    ⚠️ `realizable` NARROWS THAT SUBSET PER PERSONA, and the conditioning is one layer deeper than
    it was. `insufficient_evidence` needs a category documented by a PAIR — a claim cannot
    contradict itself with one document — so a persona holding only a category that has a fiscal
    receipt cannot realize it, and drawing it for them would plan a claim that has to be refused.
    The alternative was to skip such a claim and account for it, which spends a whole claim to
    preserve a share; renormalizing spends none and is visible here. Declared in the balance report
    and in config/labelling-schema.yaml, because a share conditioned on the persona is not the
    share the file declares.
    """
    mix = verdict_mix()
    weights: list[float] = []
    for verdict in realizable:
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
    return rng.choices(realizable, weights=weights, k=1)[0]


def draw_insufficient_evidence_cause(rng: random.Random) -> str:
    """Which cross-check a claim's two documents fail, per `insufficient_evidence_causes`.

    Drawn in the order policy.yaml declares the causes in, which is a file order rather than a set
    order, so the draw stays reproducible — the same rule as the `partially_covered` causes below.

    🔴 THE MAP IS NOT THE CAUSE VOCABULARY. `subject_not_evidenced` also leads to this verdict and
    carries no share, because nothing can plan a deliberately incomplete claim — see
    `_UNREALIZABLE_CAUSES`. A consumer reading a corpus must not infer from the two causes present
    that the third does not exist; `policy_engine` returns it and would label it correctly.
    """
    causes = insufficient_evidence_causes()
    return rng.choices(list(causes), weights=list(causes.values()), k=1)[0]


def draw_partially_covered_cause(rng: random.Random) -> str:
    """Why a `partially_covered` claim is partially covered, per `partially_covered_causes`.

    Drawn in the order policy.yaml declares the causes in, which is a file order rather
    than a set order, so the draw stays reproducible.
    """
    causes = partially_covered_causes()
    return rng.choices(list(causes), weights=list(causes.values()), k=1)[0]


# How far before the payment a subject document may be dated, in days. A plan's own
# choice rather than a policy value: policy.yaml has nothing to say about the gap between
# an invoice and its settlement, and the only rule the engine enforces is that the payment
# does not come first. Zero is included, because an invoice paid on the day it is issued is
# ordinary. The upper end deliberately reaches past a month, so that a subject document
# dated in the previous benefit period is reachable — that combination is an ordinary claim
# under the period rule and it has to occur in the data, not merely be permitted by it.
_SUBJECT_LEAD_DAYS = 45


@dataclass(frozen=True)
class DocumentPlan:
    """One document of a claim: which template, and dated when.

    Its own date, because the documents of a claim are not simultaneous: an invoice is
    issued and then settled. What it proves is not a field — that is `evidence_of` on the
    archetype, read from policy.yaml.
    """

    archetype: Archetype
    issued_at: datetime


@dataclass(frozen=True)
class ClaimPlan:
    """What to build, and what answer it is meant to produce once built.

    `documents` is a LIST, and everything downstream has to treat it as one. A claim's
    evidence may be split — an invoice proving what was bought plus a payment confirmation
    proving it was paid — and while every registered archetype proves both facts that list has
    exactly one entry. Nothing may depend on that: the guarantees that used to hold because a
    claim had one document (one vendor could not differ between documents, a claim could not
    disagree with itself) are now guarantees somebody has to keep.

    `issued_at` is the CLAIM's date — the date its money moved, which is the date of its
    proof of payment and the date the ledger orders it by. Each document carries its own.

    `verdict` and `cause` are the *target*. The label that reaches the dataset comes from
    `policy_engine`, which may disagree — a basket meant to overrun an annual limit that
    turned out too small comes back `covered`. `ground_truth` takes the engine's answer
    and never the target, so the disagreement is reported rather than papered over.
    """

    claim_id: str
    persona_id: str
    category: str
    verdict: Verdict
    documents: tuple[DocumentPlan, ...]
    issued_at: datetime
    cause: str | None = None
    # Passed straight to `content_builder`. `None` means "the builder's own default".
    # CLAIM-LEVEL, not per document: a basket sized to overrun an annual balance is sized
    # against the claim, and `subject_document` below is what stops a second document from
    # doubling it.
    coverage_target: Decimal | None = None
    item_count: int | None = None

    @property
    def subject_document(self) -> DocumentPlan:
        """The one document of this claim that states what was bought, and so carries the
        basket.

        Exactly one, and the plans this planner builds satisfy that by construction —
        `_select_documents` returns either a single document proving both facts or one
        subject plus one payment. It is asserted here rather than assumed because the
        basket is sized once for the whole claim: two documents carrying line items would
        make a claim aimed at overrunning a 12000 balance overrun it twice, and the label
        would be right about a dataset that no longer contains the mechanism it was
        counted under.
        """
        carriers = [d for d in self.documents if evidence_of(d.archetype).proves_subject]
        if len(carriers) != 1:
            raise ValueError(
                f"claim {self.claim_id} plans {len(carriers)} documents that state what "
                "was bought; the basket is sized once for the claim, so exactly one has "
                "to carry it. Splitting a basket across documents is a decision nobody "
                "has taken — see `coverage_target` above."
            )
        return carriers[0]

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


def can_assemble_evidence(candidates: list[Archetype]) -> bool:
    """Whether these archetypes can establish BOTH facts a reimbursement rests on.

    The precondition of `_select_documents`, asked separately so that a caller can find out
    before planning instead of by catching the refusal. The two shapes are the two that function
    builds: one archetype proving both facts, or a subject archetype together with a payment one.

    ONE ARCHETYPE THAT PROVES ONE FACT IS NOT ENOUGH, and until the payment-confirmation
    archetype was registered nothing in the registry could make that distinction visible: every
    entry proved both facts, so "some archetype exists" and "a claim can be built" were the same
    question. They are now different questions, and answering the first while meaning the second
    would plan a claim `_select_documents` then refuses — a failure landing a stage away from its
    cause.
    """
    both = [a for a in candidates if all(evidence_of(a))]
    subjects = [a for a in candidates if evidence_of(a) == Evidence(True, False)]
    payments = [a for a in candidates if evidence_of(a) == Evidence(False, True)]
    return bool(both) or bool(subjects and payments)


def documentable_categories(persona: Persona) -> list[str]:
    """The persona's benefit categories a COMPLETE claim can be documented in.

    Ignores the ledger entirely — this is a question about templates, and it is the one
    `assembler._draw_documentable_persona` asks before any claim exists. Empty is a normal answer
    while the template set is incomplete, not an error — and it stopped being the common answer when
    the invoice archetype landed: every Ukrainian category can now be completed, six of them by an
    invoice plus a payment document and `vitamins_nutrition` by a fiscal receipt. Both halves of the
    sentence matter, because a persona that holds no completable category is now the rare case
    rather than the usual one, and `_draw_documentable_persona`'s retry loop is correspondingly
    less exercised.

    COMPLETE, not merely covered by some archetype. A category whose only registered archetype is
    a payment confirmation has a document and no claim: the payment is proven and what was bought
    is not, and the planner draws neither of the verdicts that describes. Reporting such a category
    as documentable would hand `_select_documents` a plan it has to refuse.
    """
    return [
        category
        for category in persona.benefit_categories
        if can_assemble_evidence(archetypes_for(persona.location.country, category))
    ]


def plannable_categories(
    persona: Persona, ledger: Ledger, verdict: Verdict | None = None
) -> list[str]:
    """The persona's documentable categories that still have an annual balance.

    The ledger is required rather than optional. Defaulting it to "no history" made the
    exhausted-balance guard hold on some call paths and not on others, and a caller that
    omitted it would plan a claim `policy_engine` then refuses to label — the failure
    landing one stage away from its cause. A caller with genuinely no history passes a
    fresh `Ledger()` and says so at the call site.

    🔴 `verdict` NARROWS FURTHER, AND THE NARROWING IS PER VERDICT RATHER THAN GLOBAL. That is the
    subtle part of this function and the reason it takes the argument at all.

    `insufficient_evidence` is realized by a claim whose SUBJECT document and PAYMENT document
    disagree, so it needs a category documented by a PAIR. A category that has a fiscal receipt
    gets one self-sufficient document — `_select_documents` prefers that shape — and one document
    cannot contradict itself, so such a category can never realize this verdict.

    WHY NOT NARROW GLOBALLY. Because `covered` and `partially_covered` are realizable in BOTH
    shapes: a receipt category is perfectly plannable for them, and dropping it from every claim
    would remove the only `fiscal_receipt` documents the corpus has, to satisfy a constraint that
    belongs to one verdict out of three. The narrowing follows the verdict because the CONSTRAINT
    follows the verdict — a global filter would encode one verdict's requirement as a property of
    the persona.
    """
    open_balance = [
        category
        for category in documentable_categories(persona)
        if ledger.remaining(persona.persona_id, category) > 0
    ]
    if verdict is not Verdict.INSUFFICIENT_EVIDENCE:
        return open_balance
    return [
        category
        for category in open_balance
        if not any(
            all(evidence_of(archetype))
            for archetype in archetypes_for(persona.location.country, category)
        )
    ]


def realizable_verdicts_for(persona: Persona, ledger: Ledger) -> tuple[Verdict, ...]:
    """The verdicts this persona can realize right now, given the categories it can still spend in.

    Every verdict this planner draws needs a category; `insufficient_evidence` needs one of a
    particular SHAPE, so a persona whose only remaining category carries a fiscal receipt cannot
    realize it. Drawing it for them would produce a plan `_select_documents` has to refuse — the
    failure landing a stage away from its cause, which is the thing this module keeps not doing.
    """
    return tuple(
        verdict
        for verdict in REALIZABLE_VERDICTS
        if plannable_categories(persona, ledger, verdict)
    )


def _draw_date_in_period(rng: random.Random) -> datetime:
    """A timestamp inside the active benefit period.

    Inside, because the planner realizes `covered` and `partially_covered`. A payment dated
    outside the window is what drives the `rejected` branch on the period, and choosing it is
    a decision of the planner rather than an accident of the calendar — which is why the
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


def _select_documents(
    rng: random.Random,
    candidates: list[Archetype],
    issued_at: datetime,
    *,
    payment_precedes_subject: bool = False,
) -> tuple[DocumentPlan, ...]:
    """The documents a claim needs to establish both facts a reimbursement rests on.

    A reimbursement needs to know what was bought and that it was paid for
    (`document_evidence` in policy.yaml), and the two need not come from the same
    document. Two shapes are built, in this order:

    1. **One document that proves both** — a fiscal receipt. Preferred wherever one is
       registered, because it is the shape a real claim usually takes and because a claim
       whose evidence is split has more ways to be wrong. Where several such archetypes are
       registered the choice between them is drawn, which is how the three UA fiscal receipts
       all reach a dataset.
    2. **A subject document plus a payment document.** The subject is dated on or before
       the payment: an invoice is issued and then settled, and a payment that came first
       is a defect the engine names. The lead is drawn from the seeded generator, so a
       subject dated in the previous benefit period occurs — that is an ordinary claim
       under the period rule, and the rule is only exercised if the data contains one.

    🔴 `payment_precedes_subject` INVERTS THAT ORDER DELIBERATELY, and it is the whole of what the
    cause of the same name needs: the subject is dated AFTER the payment, so the claim states that
    money moved before there was anything to pay for. One sign, because the defect is the ORDER and
    nothing else — the lead is drawn identically either way, so a claim built for this cause is
    distinguishable from an ordinary one by nothing except the thing being labelled. That is what
    makes it a usable negative: had the flag also changed the gap, a consumer could learn the gap.

    Where neither shape can be assembled the planner refuses. It does not build a claim
    out of whichever archetypes exist and leave the engine to discover that half the
    evidence is missing: the verdict is chosen first here, and the two verdicts this
    planner draws both need complete evidence.
    """
    both = [a for a in candidates if all(evidence_of(a))]
    if both:
        return (DocumentPlan(archetype=rng.choice(both), issued_at=issued_at),)

    subjects = [a for a in candidates if evidence_of(a) == Evidence(True, False)]
    payments = [a for a in candidates if evidence_of(a) == Evidence(False, True)]
    if subjects and payments:
        # Drawn before the archetypes so that adding a template does not shift the dates
        # of a run: the lead is a property of the claim, the templates are a property of
        # the registry, and the two should not be entangled in the seed stream.
        lead = timedelta(days=rng.randint(0, _SUBJECT_LEAD_DAYS))
        if payment_precedes_subject:
            # ⚠️ A ZERO LEAD WOULD NOT BREAK ANYTHING. `_cross_checks` compares dates STRICTLY —
            # paying an invoice on the day it is issued is ordinary — so the subject has to land at
            # least one day after the payment for the cause to be real. `max` is what guarantees it
            # rather than a redraw, which would make the number of values taken from `rng` depend on
            # what the first one returned.
            lead = max(lead, timedelta(days=1))
        subject_at = issued_at + lead if payment_precedes_subject else issued_at - lead
        return (
            DocumentPlan(archetype=rng.choice(subjects), issued_at=subject_at),
            DocumentPlan(archetype=rng.choice(payments), issued_at=issued_at),
        )

    raise ValueError(
        "no registered archetype, and no pair of them, establishes both what was bought "
        "and that it was paid for: "
        f"{sorted(a.slug for a in candidates)}. policy.yaml's `document_evidence` says "
        "what each document type proves, and a claim needs both facts."
    )


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
        realizable = realizable_verdicts_for(persona, ledger)
        if not realizable:
            # RAISE HERE, NOT A STAGE LATER. The `or REALIZABLE_VERDICTS` this replaced drew
            # from the full list when nothing was realizable, and the resulting `ValueError`
            # then named whichever verdict the draw happened to land on rather than the real
            # reason — exactly the failure-lands-a-stage-from-its-cause anti-pattern this
            # module's own docstrings call out (`plannable_categories`, `_select_documents`,
            # `realizable_verdicts_for`). `why_no_claim` already knows the reason: no
            # documentable category, or every one exhausted.
            raise ValueError(
                f"persona {persona.persona_id} can realize no verdict: "
                f"{why_no_claim(persona, ledger)}"
            )
        verdict = draw_verdict(rng, realizable)
    if verdict not in REALIZABLE_VERDICTS:
        raise NotImplementedError(
            f"verdict {verdict.value!r} "
            f"{_UNREALIZABLE_REASONS.get(verdict, _NO_REASON_RECORDED)}"
        )

    if category is None:
        options = plannable_categories(persona, ledger, verdict)
        if not options:
            raise ValueError(
                f"persona {persona.persona_id} holds no category any registered archetype can "
                f"document with balance left, for verdict {verdict.value!r}: "
                f"{persona.benefit_categories}"
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
    if issued_at is None:
        issued_at = _draw_date_in_period(rng)

    coverage_target: Decimal | None = None
    item_count: int | None = None
    if verdict is Verdict.PARTIALLY_COVERED:
        cause = cause or draw_partially_covered_cause(rng)
        if cause == "mixed_items":
            coverage_target = rng.choice(coverage_targets())
        elif cause == "limit_exhausted":
            # Realized by a basket the remaining balance cannot absorb. The engine decides
            # whether it actually did; nothing here assumes it.
            remaining = ledger.remaining(persona.persona_id, category)
            item_count = _overrun_item_count(remaining, category)
        else:
            raise ValueError(f"policy.yaml declares no such partially_covered cause: {cause!r}")
    elif verdict is Verdict.INSUFFICIENT_EVIDENCE:
        # TWO VERDICTS CARRY A CAUSE NOW, which is why the guard below widened from "a cause
        # belongs to a partially_covered claim". The cause decides which cross-check the claim's
        # two documents fail; the engine decides whether they actually did, and nothing here
        # assumes it.
        cause = cause or draw_insufficient_evidence_cause(rng)
        if cause not in (AMOUNT_MISMATCH, PAYMENT_PRECEDES_SUBJECT):
            raise ValueError(
                f"policy.yaml declares no such buildable insufficient_evidence cause: {cause!r}. "
                f"{cause!r} may still be a cause the ENGINE returns — see "
                "`claim_planner._UNREALIZABLE_CAUSES` for the one that is and cannot be planned."
            )
    elif cause is not None:
        raise ValueError(
            f"a cause belongs to a partially_covered or an insufficient_evidence claim, not to "
            f"{verdict.value!r}"
        )

    return ClaimPlan(
        claim_id=claim_id,
        persona_id=persona.persona_id,
        category=category,
        verdict=verdict,
        cause=cause,
        documents=_select_documents(
            rng, candidates, issued_at,
            payment_precedes_subject=cause == PAYMENT_PRECEDES_SUBJECT,
        ),
        issued_at=issued_at,
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
