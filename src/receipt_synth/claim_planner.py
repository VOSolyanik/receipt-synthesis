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

It also decides whether a claim's evidence is COMPLETE — see `EvidenceIntent`. Building a
claim that establishes only one of the two facts is a thing this generator has to do, and
it is done by naming the intent, never by letting a document quietly fail to turn up.

All six verdicts are realizable, so `_UNREALIZABLE_REASONS` is empty and kept only for the
next member of the enum. What is still refused rather than faked is one ROUTE to a
realizable verdict — `rejected` reached by an uncovered basket rather than by a payment
outside the period — which is a different statement and has its own table,
`_UNREALIZABLE_ROUTES`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from receipt_synth.config import (
    archetype_draw_weights,
    coverage_targets,
    partial_payment_schedules,
)
from receipt_synth.content_builder import MAX_LINE_ITEMS, estimated_line_value
from receipt_synth.policy_engine import (
    AMOUNT_MISMATCH,
    COUNTERPARTY_MISMATCH,
    OUTSIDE_PERIOD,
    PAYMENT_PRECEDES_SUBJECT,
    SUBJECT_MISMATCH,
    SUBJECT_NOT_EVIDENCED,
    ClaimEvaluation,
    Evidence,
    Ledger,
    active_period,
    document_evidence,
    insufficient_evidence_causes,
    partially_covered_causes,
    rejected_routes,
    verdict_mix,
)
from receipt_synth.schemas import (
    ClaimGroundTruth,
    Country,
    DocType,
    FxRateApplied,
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
    # 🔴 THE CURRENCY THE DOCUMENT IS DRAWN UP IN, AND IT DECIDES WHICH PAYMENT MAY SETTLE WHICH
    # SUBJECT. `policy_engine._one_claim_one_currency` REFUSES a claim whose documents are stated
    # in two: coverage pools line items across them and every cross-document axis compares amounts
    # between them, and policy.yaml states no rule for doing either across a rate. So a pair drawn
    # from two currencies is not a harder claim — it is one the oracle cannot label at all.
    # `_settleable_subjects` is where that refusal becomes a selection rule, before a plan exists
    # to be refused.
    #
    # DECLARED HERE AND PRINTED BY THE BUILDER, exactly as `language` is, so the two can drift.
    # `tests/test_archetype_currency.py` builds one document per archetype and compares the
    # label's `currency` against this field, which is what stops them.
    currency: str = "UAH"
    # 🔴 `country` IS THE CLAIMANT'S JURISDICTION — the key `archetypes_for` selects by,
    # which is the persona's — and the SELLER need not share it: a Ukrainian employee
    # buys a course from a foreign platform and submits its receipt. `vendor_pool` names
    # the config/vendors.json block that seller draws from where the two differ; `None`
    # means the claimant's own, which is every domestic archetype.
    vendor_pool: str | None = None
    # 🔴 THE PAGE EXISTS ONLY ON A SCREEN — a banking-app rendering, not a sheet anything could
    # print — so its capture channel is `screenshot` BY CONSTRUCTION and the class's
    # `capture_mix` in policy.yaml is never consulted for it. A physical fact of the archetype,
    # declared where the archetype is, exactly as its paper size is; the tunable shares stay in
    # policy.yaml, where a label-sizing weight belongs.
    screen_native: bool = False


def evidence_of(archetype: Archetype) -> Evidence:
    """What a document built from this archetype establishes, per policy.yaml."""
    return document_evidence(archetype.doc_type)


# The registry the planner selects from — eleven archetypes over six document classes today;
# further ones register here as their templates land.
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
    # It is the payment half of the dominant pair: the invoice below supplies the subject fact,
    # and every category this tuple names is documentable through the two together. (Until the
    # invoice landed, no claim could be assembled from this archetype at all — a claim needs both
    # facts — and registering it changed no dataset; that early state is recorded in the
    # contract's version history rather than restated here as if it were current.)
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
    # The second payment-proving class of the pair era: `_select_documents` draws the payment
    # half of a split claim from this archetype and the confirmations, so a statement reaches a
    # dataset on every run. (It, too, could reach none before the invoice existed — the planner
    # had a payment fact twice over and no subject fact to pair it with.)
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
    # 🔴 THE FIFTH DOCUMENT CLASS, AND THE ONE THAT LOOKS LIKE THE FIRST. A товарний чек prints a
    # basket, the same four totals, the same columns and the same arithmetic as a fiscal receipt,
    # and 📄 differs from it by exactly the two requisites the tax service says such a document
    # omits — the fiscal number of the register and the wording «ФІСКАЛЬНИЙ ЧЕК». So it is the
    # first archetype whose classification cannot be reached from the layout at all.
    #
    # 🔴 IT IS A SUBJECT-CLASS DOCUMENT, registered as what it IS. policy.yaml's
    # `document_evidence` gives `non_fiscal_receipt` `proves_subject: true, proves_payment: false`,
    # and this entry says nothing else — an archetype registered as a payment class because an
    # employee SUBMITS it in place of a payment proof would contradict the evidence table and
    # relabel every claim carrying one. What the employee believes is not a property of the
    # document; it is the SHAPE OF THE CLAIM, and it lives in the planner as
    # `EvidenceIntent.PAYMENT_GAP`.
    #
    # SIX CATEGORIES OF SEVEN, AND THE MISSING ONE IS A CONSEQUENCE OF WHO MAY ISSUE THE DOCUMENT.
    # 📄 A ПДВ payer is obliged to use a cash register, so the seller of a slip is a non-payer —
    # `content_builder.build_non_fiscal_receipt` refuses any other — and `medical_insurance` is
    # served in config/vendors.json by insurers alone, every one of them registered. The other six
    # each have a non-payer vendor. Checked against vendors.json by a test rather than trusted
    # here, so a vendor list edited on one side cannot leave this tuple asserting the other.
    "ua_non_fiscal_receipt": Archetype(
        slug="ua_non_fiscal_receipt",
        doc_type=DocType.NON_FISCAL_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=(
            "language_courses",
            "professional_development",
            "sport",
            "mental_health",
            "vitamins_nutrition",
            "hobby",
        ),
    ),
    # 🔴 THE SIXTH DOCUMENT CLASS, AND THE SECOND THAT PROVES BOTH FACTS — the first that
    # does so without being fiscal, and the corpus's first document in a second language
    # and a second currency. `country=UA` because that field is the CLAIMANT'S
    # jurisdiction: a Ukrainian employee buys a course from a foreign platform, pays by
    # card in euros and submits the platform's receipt; the seller draws from the `EU`
    # vendor pool instead (`vendor_pool`), which is what the field exists for.
    #
    # ONE CATEGORY, AND THE CHOICE IS THE BLAST RADIUS. `_select_documents` PREFERS a
    # single document proving both facts wherever one is registered — the rule that gives
    # `vitamins_nutrition` its receipts — so every category listed here loses its
    # invoice-plus-payment pair for single-document verdicts and drops out of the
    # pair-realized ones (`plannable_categories`). `professional_development` is the
    # class's natural home — the `online_learning_platform` vendor profile already lives
    # there — and confining the archetype to it keeps the split pair exercised in five
    # categories rather than four. Wider registration is a distribution decision, not a
    # template property, and it is not taken here.
    #
    # ⚠️ THE ORACLE CONVERTS THIS ARCHETYPE'S CLAIMS. Every amount on the document is in
    # EUR; limits are in UAH; `policy_engine` converts at the vendored rate of
    # config/fx-rates.yaml and records it in the claim's `fx_rates` — the decision that
    # unblocked this registration, and the reason it could not land before it.
    "eu_platform_receipt": Archetype(
        slug="eu_platform_receipt",
        doc_type=DocType.PLATFORM_RECEIPT,
        country=Country.UA,
        language="en",
        currency="EUR",
        categories=("professional_development",),
        vendor_pool="EU",
    ),
    # THE DOMESTIC VARIANT OF THE SAME CLASS — Ukrainian, UAH, the seller a domestic
    # company whose requisites and contained-VAT row are ordinary where the EU page
    # disclaims them. One shared body renders both, so the pair is a controlled
    # comparison of the language and currency axes with the layout held fixed; the split
    # between the two within a claim is `archetype_shares` in config/generation.yaml.
    # Same one-category confinement as its twin, and for the same blast-radius reason.
    "ua_platform_receipt": Archetype(
        slug="ua_platform_receipt",
        doc_type=DocType.PLATFORM_RECEIPT,
        country=Country.UA,
        language="uk",
        categories=("professional_development",),
    ),
    # 🔴 THE THIRD AND FOURTH RENDERINGS OF THE `payment_confirmation` CLASS — the phone.
    # Neither is a new document type: the transaction screen and the framed receipt are
    # what the SAME payment looks like inside the banking application, and registering
    # them as payment_confirmation is what keeps the four classifier target classes at
    # four while the corpus gains the domain's dominant carrier.
    #
    # `ua_bank_app_transaction` is the strongest negative example for the class by LACK
    # OF REQUISITES: it looks like proof of payment and carries no document number, no
    # authorization code, no RRN, no stamp, no signature, no purpose — and its
    # counterparty line is a processor descriptor (`LIQPAY*…`) that matches no party
    # block of any document beside it. The LABEL still carries the bare trade name; what
    # breaks is the printed cross-check, deliberately.
    #
    # ⚠️ REGISTERING IT AS ITS TYPE MEANS THE ORACLE TREATS IT AS PROVING PAYMENT — the
    # evidence model is by type, per-archetype overrides are impossible (`Evidence`), and
    # whether a screen with no requisites SHOULD prove payment to a benefit plan is a
    # policy question policy.yaml does not ask. What the corpus records is that the
    # transaction happened; what a consumer's plan does about weak carriers is measured
    # against these pages, not answered by them.
    #
    # `ua_bank_receipt_in_app` is the A4 confirmation ITSELF inside the app's frame —
    # built by the same builder, labelled by the same `ground_truth`, so "the medium does
    # not change the ground truth" holds by construction and the archetype is the
    # cheapest carrier-invariance test the corpus can hold.
    #
    # Both carry every category, for the reason the A4 confirmation does: the page lists
    # no items, so there is nothing a category could contradict. Their shares of the
    # payment draw are `archetype_shares` in config/generation.yaml.
    "ua_bank_app_transaction": Archetype(
        slug="ua_bank_app_transaction",
        doc_type=DocType.PAYMENT_CONFIRMATION,
        country=Country.UA,
        language="uk",
        screen_native=True,
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
    "ua_bank_receipt_in_app": Archetype(
        slug="ua_bank_receipt_in_app",
        doc_type=DocType.PAYMENT_CONFIRMATION,
        country=Country.UA,
        language="uk",
        screen_native=True,
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


# 🔴 WHICH SUBJECT CLASSES A PAYMENT DOCUMENT CAN SETTLE, and therefore which ones may be the
# subject half of a SPLIT PAIR. It exists because the registry now holds two subject-only classes
# that are not interchangeable, and `_select_documents` would otherwise draw between them.
#
# 📄 An invoice is an OFFER TO PAY: it is issued, it names an obligation, and a payment settles it
# later. 👁 That is also what a payment purpose names — an invoice or a delivery note — and it is
# how the pair is linked on the page: `content_builder.Invoice.reference` is the only
# `DocumentReference` any class produces, and the confirmation's purpose cites it.
#
# ⛔ A товарний чек SETTLES NOTHING AND IS CITED BY NOTHING. It is handed over when the goods are,
# it states no obligation, and no observation in this project shows a payment purpose naming one.
# A pair built from it would print a purpose citing an empty reference and would assert a
# settlement relation nothing evidences. It reaches a claim as the WHOLE of that claim's evidence
# instead — see `EvidenceIntent.PAYMENT_GAP`.
#
# ⚠️ `act` AND `order_screenshot` ARE ABSENT because they have no archetype. An act of services
# rendered is settled by a transfer and would belong here the day one is written; listing a class
# nothing can build would be a decision nothing exercises.
_SETTLED_BY_A_PAYMENT: frozenset[DocType] = frozenset({DocType.INVOICE})

# 🔴 WHICH PAYMENT ARCHETYPES CAN PRINT THE РАХУНОК THEY SETTLE — and therefore which ones can be
# the payment half of a `subject_mismatch` claim, whose whole defect is a citation naming the
# wrong one. BY SLUG AND NOT BY TYPE, because the app-transaction screen shares
# `DocType.PAYMENT_CONFIRMATION` with two archetypes that do print a purpose line: what decides
# membership is the PAGE, not the label class.
#
# ⛔ `ua_bank_app_transaction` IS ABSENT because 👁 the observed screen prints no purpose line at
# all — its builder refuses `must_cite` for the same reason, and the two refusals meeting would
# mean this set and the builder have come apart. The A4 confirmation and the in-app receipt print
# one on the purpose-printing initiation modes, which `must_cite` selects among; the statement's
# labelled row always carries one.
_CITES_THE_SETTLED_DOCUMENT: frozenset[str] = frozenset({
    "ua_bank_payment_confirmation",
    "ua_bank_receipt_in_app",
    "ua_bank_statement",
})

# 🔴 WHICH SUBJECT CLASSES CAN STATE THAT THEIR OBLIGATION IS SETTLED IN PARTS, and therefore
# which ones can be the subject half of a `partially_paid` claim. PUBLIC, unlike the set above,
# because `assembler` reads it too: it is what decides whether a plan's schedule may be handed to
# a builder, and a second copy of the answer there is how the two would come to differ.
#
# 📄 An instalment term is a condition of an OFFER — it says how the seller proposes to be paid —
# so it belongs to the class that makes an offer. ⛔ A товарний чек makes none: it is handed over
# with the goods, it states no obligation, and there is nothing left of it to settle in parts.
# That is the same reasoning that keeps it out of `_SETTLED_BY_A_PAYMENT`, arrived at from the
# other end.
STATES_AN_INSTALMENT_TERM: frozenset[DocType] = frozenset({DocType.INVOICE})


def _pairable_subjects(candidates: list[Archetype]) -> list[Archetype]:
    """The subject-only archetypes that may be paired with a payment document.

    ONE PREDICATE, TWO CALLERS, and that is the whole reason it is a function:
    `can_assemble_evidence` answers whether a claim can be built and `_select_documents` builds
    it, so a subject class admitted by the first and refused by the second would plan a claim the
    builder then rejects — the failure landing a stage away from its cause, which is the thing
    this module keeps not doing.
    """
    return [
        archetype
        for archetype in candidates
        if evidence_of(archetype) == Evidence(True, False)
        and archetype.doc_type in _SETTLED_BY_A_PAYMENT
    ]


def _payment_archetypes(candidates: list[Archetype]) -> list[Archetype]:
    """The archetypes proving the payment and no subject — the payment half of every pair.

    A one-line comprehension, extracted on its third call site (`can_assemble_evidence`,
    `plannable_categories`, `_select_documents`) because all three now feed it to
    `_settleable_subjects`, and a pool spelled out three ways is a pool that will one day be
    spelled out three DIFFERENT ways.
    """
    return [a for a in candidates if evidence_of(a) == Evidence(False, True)]


def _settleable_subjects(
    subjects: list[Archetype], payments: list[Archetype]
) -> list[Archetype]:
    """The subjects some payment archetype of this pool could actually settle — the currency rule.

    🔴 ONE CLAIM, ONE CURRENCY, APPLIED WHERE THE PAIR IS CHOSEN. `policy_engine` refuses to label
    a claim whose two documents are stated in different currencies (`_one_claim_one_currency`), and
    that refusal cannot be met by converting: policy.yaml states no rule for pooling line items or
    comparing amounts across a rate, and the oracle raises rather than guessing one. An invoice in
    euros beside a confirmation in hryvnias is therefore not a defect a label could describe — it
    is a claim nothing can label — so it must not be planned in the first place.

    ⛔ AND IT IS NOT A FILTER ON THE PAYMENT ALONE. Filtering the payment after the subject is drawn
    would leave the subject pool free to draw an archetype no payment can settle, and the refusal
    would land inside the draw. Narrowing the SUBJECTS first is what makes every draw below
    reachable — and where a subject is dropped, it is dropped because this registry holds no
    payment document in its currency, which is a statement about the registry rather than about
    the claim.

    ONE PREDICATE, THREE CALLERS, for the reason `_pairable_subjects` is one function:
    `can_assemble_evidence` asks whether a pair exists, `plannable_categories` asks it per verdict,
    and `_select_documents` builds the pair. A subject admitted by one and refused by another would
    fail a stage away from its cause.
    """
    currencies = {payment.currency for payment in payments}
    return [subject for subject in subjects if subject.currency in currencies]


def _instalment_subjects(candidates: list[Archetype]) -> list[Archetype]:
    """The pairable subjects that can also STATE an instalment term — the subject half of a
    `partially_paid` claim.

    Narrower than `_pairable_subjects` and for a different reason, so it is a second predicate
    rather than a widened first one: that one asks whether a payment can settle this class at all,
    this one asks whether the class can print the marker the engine reads. A class could satisfy
    either without the other.

    ONE PREDICATE, TWO CALLERS, for the reason `_pairable_subjects` is one: `plannable_categories`
    answers whether such a claim can be planned and `_select_documents` selects the documents for
    it, and a category admitted by the first and refused by the second would fail a stage away
    from its cause.
    """
    return [
        archetype
        for archetype in _pairable_subjects(candidates)
        if archetype.doc_type in STATES_AN_INSTALMENT_TERM
    ]


def _settleable_instalment_subjects(candidates: list[Archetype]) -> list[Archetype]:
    """The subject half a `partially_paid` claim needs: states an instalment term AND is settleable
    by a payment document of this registry.

    The two narrowings composed, at the one place a caller wants them composed — the whole of what
    `plannable_categories` asks for that verdict, and the same pair `_select_documents` then builds.
    """
    return _settleable_subjects(
        _instalment_subjects(candidates), _payment_archetypes(candidates)
    )


class EvidenceIntent(Enum):
    """Whether a claim's evidence is meant to establish both facts, or deliberately not.

    🔴 THE DIFFERENCE BETWEEN "COULD NOT ASSEMBLE THE SUBJECT" AND "CHOSE NOT TO", MADE A VALUE SO
    THAT NOBODY HAS TO INFER IT FROM A COUNT OF DOCUMENTS. `_select_documents` assembles both facts
    or refuses, and that refusal is what stops a missing template from turning into a mislabelled
    claim. An incomplete claim is nevertheless something this generator has to produce — a bare
    payment confirmation is the case the corpus exists to show a system failing to notice — so the
    refusal is lifted by NAMING the intent at the call site rather than by weakening the check.

    Two documents shaped the same way can therefore mean different things, and a reader of a plan
    can tell which: a claim whose evidence is short by accident is a bug this module still raises
    on, and a claim whose evidence is short on purpose says so in a field.

    🔴 THERE ARE TWO GAPS AND THEY ARE OPPOSITE, one per fact of `document_evidence`. Neither is a
    weaker form of the other and neither may absorb the other: `EVIDENCE_GAP` is a payment with no
    subject beside it (`insufficient_evidence` / `subject_not_evidenced`), `PAYMENT_GAP` is a
    subject with no payment beside it (`not_proof_of_payment`). One employee attached the transfer
    and forgot to say what it bought; the other attached the thing they bought and never showed
    that money moved.

    ⚠️ THE NAMES ARE NOT SYMMETRIC AND THE MEMBERS ARE. `EVIDENCE_GAP` is the SUBJECT gap and was
    named before a second gap existed; renaming it now would reach config/policy.yaml,
    config/labelling-schema.yaml, docs/architecture.md and this module's callers, for a rename
    rather than a change of behaviour. So the asymmetry is recorded here instead, where a reader
    who has only the names to go on will look.
    """

    #: Both facts. Every claim was this until the gap below was named.
    COMPLETE = "complete"
    #: The WHAT-WAS-BOUGHT slot of `document_evidence` left open on purpose: a payment document
    #: and nothing beside it. The planner does not assert the resulting label — `policy_engine`
    #: derives `insufficient_evidence` with the cause `subject_not_evidenced` from the documents
    #: that were built, exactly as it does for every other claim.
    EVIDENCE_GAP = "evidence_gap"
    #: The MONEY-MOVED slot left open on purpose: a document that states what was bought and
    #: nothing that attests a payment — a bare invoice, a товарний чек. The planner asserts no
    #: label here either; `policy_engine` reads the document types and answers
    #: `not_proof_of_payment`, with no cause, there being one slot and one way to fail it.
    PAYMENT_GAP = "payment_gap"


# --- what can be realized -----------------------------------------------------

# The verdicts this planner can build documents for. The others are refused rather than
# approximated: a planner that accepted one and produced an ordinary basket would write a
# wrong label instead of failing.
# The route to `rejected` that carries NO cause in the label — the verdict says the whole of it
# (`policy_engine.verdict_for`). A PLANNER NAME, NOT AN ENGINE CAUSE: the engine never emits this
# string, so it lives here rather than beside `OUTSIDE_PERIOD` in policy_engine, and `plan.cause`
# holding it means "aim at a wholly non-covered basket" — realized as `coverage_target` ZERO,
# which is the label-first knob `content_builder._draw_basket` reads. The name is the
# `rejected_routes` key in policy.yaml and the route suffix `_UNREALIZABLE_ROUTES` used to file
# it under while nothing could build one.
ZERO_COVERAGE = "zero_coverage"

REALIZABLE_VERDICTS: tuple[Verdict, ...] = (
    Verdict.COVERED,
    Verdict.PARTIALLY_COVERED,
    # 🔴 THE THIRD, AND IT ARRIVED WITH A MECHANISM RATHER THAN WITH A TEMPLATE. A claim can now be
    # planned whose subject document and whose payment document DISAGREE — about the amount, about
    # which came first, or about WHO the other party is — which is what `policy_engine`'s
    # cross-check branch has always labelled and what nothing could build until an archetype
    # proving the subject alone existed. ALL FOUR of its causes are drawn: the three axes
    # `cross_document_agreement` declares, and the missing subject, which `EvidenceIntent` made
    # plannable.
    Verdict.INSUFFICIENT_EVIDENCE,
    # 🔴 THE FOURTH, AND THE FIRST WHOSE MECHANISM IS A DATE RATHER THAN A DOCUMENT. Everything
    # above is realized by WHAT the claim carries; this one is realized by WHEN its money moved —
    # `_payment_outside_period` puts the payment outside the window policy.yaml declares, and the
    # engine answers `rejected` with the cause `outside_period` off the payment date alone. No
    # archetype, no basket and no template had to change for it, which is why it was reachable long
    # before it was drawn: `issued_at` has always been a public parameter with no guard on the
    # period.
    #
    # 🔴 AND THE SECOND ROUTE IS DRAWN NOW TOO — a basket holding no covered line, which
    # `content_builder._draw_basket` builds when `coverage_target` is ZERO and the engine labels
    # `rejected` off the line items alone, with no cause (`verdict_for`). `rejected_routes` in
    # policy.yaml splits the bucket between the two; what the split buys is that neither the
    # payment date nor the basket alone predicts this verdict, which is what
    # `_UNREALIZABLE_ROUTES` used to warn consumers it did.
    Verdict.REJECTED,
    # 🔴 THE FIFTH, AND IT NEEDED NO NEW MECHANISM — only the OTHER HALF of one that existed.
    # `EvidenceIntent` already let a claim be planned short of a fact on purpose; this verdict is
    # the same idea with the slots exchanged (`PAYMENT_GAP`), and it became realizable the moment
    # a claim could carry a subject document with nothing beside it. TWO ARCHETYPES REALIZE IT —
    # a bare invoice and a товарний чек — and the second is why the corpus finally contains a
    # document that looks fiscal and is not.
    #
    # ⛔ AMOUNTS, BASKETS AND DATES ARE NOT CONSULTED, here or in the engine. The verdict is a
    # property of the document TYPES a claim carries, so a claim planned for it is an ORDINARY
    # claim minus its payment document: an ordinary covered basket, an ordinary date inside the
    # period, and `covered_fraction` typically 1.0 beside a reimbursable amount of zero.
    Verdict.NOT_PROOF_OF_PAYMENT,
    # 🔴 THE SIXTH, WHICH COMPLETES THE ENUM, AND THE ONLY ONE THAT NEEDED THE CORPUS'S BYTES TO
    # CHANGE. Every verdict above was realized by a SHAPE — which documents a claim carries, which
    # dates they bear — and this one needed a document to PRINT something no archetype printed: an
    # invoice that states its obligation is settled in equal parts and what one part comes to
    # (`content_builder.Invoice.schedule`). The planner then dates and sizes the claim exactly as
    # an ordinary one and lets the payment settle one part.
    #
    # 🔴 THE MARKER IS THE WHOLE MECHANISM, AND THE ARITHMETIC IS NOT. A payment smaller than its
    # invoice is ALSO how `insufficient_evidence` / `amount_mismatch` is built, so what separates
    # the two is the printed term and nothing else — see `partial_payment` in policy.yaml. A plan
    # aimed here that produced no term would be an `amount_mismatch` claim wearing this label.
    Verdict.PARTIALLY_PAID,
)

# WHY A ROUTE TO A REALIZABLE VERDICT NEEDS ITS OWN TABLE. `_UNREALIZABLE_REASONS` below is keyed
# by VERDICT, so the reason a single ROUTE to a realizable verdict could not be built has nowhere
# to live there — a verdict being realizable and every route to it being realizable are different
# statements, and the second is the one a reader of a corpus needs.
#
# ⚠️ IT WAS `_UNREALIZABLE_CAUSES` AND IS KEYED BY ROUTE NOW, because its first non-empty entry is a
# route with NO CAUSE. A `rejected` claim carries the cause `outside_period` or nothing at all —
# policy.yaml gives the zero-coverage route no cause on purpose, the verdict being the whole of what
# it says — so a table keyed by cause could not hold the one thing it exists to record. Where a
# route does have a cause the cause IS its name, unprefixed, exactly as the old table keyed it —
# which is how `plan_claim` can go on pointing a caller here when it refuses a cause it cannot aim
# at.
_UNREALIZABLE_ROUTES: dict[str, str] = {
    # 🔴 EMPTY, AND KEPT, for the reason `_UNREALIZABLE_REASONS` below is: every route to every
    # realizable verdict is now buildable, and the next route that cannot be built lands here
    # before anything promises it.
    #
    # ⚠️ `rejected/zero_coverage` WAS THE FIRST AND ONLY ENTRY AND ITS REASON IS WORTH ONE LINE,
    # because it is the sentence a reader of the git history will find: the route needed
    # `content_builder` to draw no covered line, and it always drew at least one. What opened it
    # is `_draw_basket` accepting a `coverage_target` of ZERO — the same label-first knob that
    # realizes `mixed_items`, at the value that used to be refused — and `rejected_routes` in
    # policy.yaml giving the draw a share. Until then every `rejected` claim in a corpus was an
    # out-of-period one, and config/labelling-schema.yaml told consumers so as a known
    # limitation; the corpus stopped being that the day this entry left the table.
}

_UNREALIZABLE_REASONS: dict[Verdict, str] = {
    # 🔴 EMPTY, AND KEPT. Every member of `Verdict` is realizable, so there is no verdict left for
    # this table to explain — which is a statement about today's registry and not about the design.
    # `unrealizable_verdicts()` accordingly returns nothing, and the balance report's "NOT
    # GENERATED IN THIS RUN" block is silent for the first time; both are exercised against a
    # patched registry in the tests, because a capability with no live case is one that rots.
    # The next verdict added to the enum lands here before it lands in `REALIZABLE_VERDICTS`.
    #
    # ⚠️ `PARTIALLY_PAID` WAS THE LAST ENTRY AND ITS REASON IS WORTH ONE LINE, because it is the
    # sentence a reader of the git history will find: it needed "a document that states an amount
    # OUTSTANDING against a total, which no archetype prints". What supplied it is not that field
    # — an invoice now prints the INSTALMENT its obligation is settled in, which is the same
    # discriminating fact stated as a term of the offer rather than as a record of what has been
    # paid. `amount_due` never was that field and still is not: it is a receipt's ДО СПЛАТИ, the
    # basket total less a discount plus cash rounding.
    #
    # ⚠️ `NOT_PROOF_OF_PAYMENT` LEFT THIS TABLE WHOLE, ROUTES AND ALL, which is why nothing of its
    # entry moved to `_UNREALIZABLE_ROUTES` the way `rejected`'s did. It has ONE route — a claim
    # every document of which is a `proves_payment: false` type — and that route is now planned by
    # `EvidenceIntent.PAYMENT_GAP` and built from two archetypes. What its old entry said is
    # nonetheless worth keeping in one line, because it is the sentence a reader will look for:
    # the mechanism was never the template, it was a planner that would deliberately plan an
    # INCOMPLETE claim, and naming the intent is what supplied it.
    #
    # ⚠️ `REJECTED` LEFT THIS TABLE AND ONE OF ITS TWO ROUTES DID NOT. The verdict is realizable:
    # `_payment_outside_period` dates a claim's payment outside policy.yaml's window and the engine
    # labels it `rejected`, cause `outside_period`. The route by WHAT WAS BOUGHT — a wholly
    # non-covered basket — is still unbuildable and is recorded in `_UNREALIZABLE_ROUTES` above,
    # which is where a route to a REALIZABLE verdict explains itself. Neither statement implies the
    # other, so the entry moved rather than being deleted with the verdict.
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
    """Which fact a claim aimed at this verdict leaves unestablished, per
    `insufficient_evidence_causes`.

    Drawn in the order policy.yaml declares the causes in, which is a file order rather than a set
    order, so the draw stays reproducible — the same rule as the `partially_covered` causes below.

    🔴 THE FIVE ARE NOT THE SAME KIND OF DEFECT, AND THE PLANNER REALIZES THEM DIFFERENTLY. The
    four cross-check causes need a claim whose two documents disagree — about the amount, about
    the order, about the party, or about which рахунок the payment settles; `subject_not_evidenced`
    needs a claim with no subject document at all, which is `EvidenceIntent.EVIDENCE_GAP`. One draw
    decides which, and `plan_claim` turns the answer into a shape.

    ⚠️ AND WHICH CAUSES EXIST AT ALL IS `cross_document_agreement`'s, NOT THIS BLOCK'S. Each
    cross-check cause is the `cause` of one declared axis, so withdrawing an axis there withdraws a
    cause the engine can return — while a share here would go on sizing a bucket nothing could
    fill. The two blocks are edited together; policy.yaml says so at both of them.

    ⚠️ THE MAP IS STILL THE DRAW AND NOT THE VOCABULARY, though the two coincide today: policy.yaml
    declares a share per cause the generator BUILDS, and `policy_engine` may return a cause nothing
    draws. A consumer must read the vocabulary from config/labelling-schema.yaml, which is the
    contract, and never from the realized shares of one corpus.
    """
    causes = insufficient_evidence_causes()
    return rng.choices(list(causes), weights=list(causes.values()), k=1)[0]


def draw_rejected_route(rng: random.Random) -> str:
    """Which of the two routes a claim aimed at `rejected` takes, per `rejected_routes`.

    Drawn in the order policy.yaml declares the routes in, which is a file order rather than a
    set order, so the draw stays reproducible — the same rule as every cause draw above.

    🔴 A ROUTE AND NOT A CAUSE, and the difference is what the label carries: `outside_period`
    is both a route name and the cause such a claim's label carries, while `ZERO_COVERAGE` names
    a route whose label carries NO cause at all — the verdict says the whole of it. `plan.cause`
    records the route either way, so a reader of a PLAN tells the two apart by value where a
    reader of the LABEL tells them apart by presence (config/labelling-schema.yaml,
    `imperfection.cardinality`).
    """
    routes = rejected_routes()
    return rng.choices(list(routes), weights=list(routes.values()), k=1)[0]


def draw_payment_schedule(rng: random.Random) -> str:
    """Into how many parts a `partially_paid` claim's obligation is divided — by name, per
    `config/generation.yaml`.

    UNIFORM, and drawn rather than fixed. A quarterly arrangement is surely more common in the
    world than a half-yearly one, but nothing in this repository has measured that, and a weight
    here would smuggle a claim about the world into a draw input — the same reasoning that keeps
    `insufficient_evidence_causes` even. What the draw buys is that the distance between the
    payment and the invoice varies, so a consumer cannot learn one ratio.

    ⚠️ NOT A LABEL, UNLIKE A CAUSE. The schedule decides how much the payment states; the verdict
    turns on the presence of the printed term and on the payment matching it, never on which
    schedule it was. That is why it lives in generation.yaml with the mismatch delta rather than
    in policy.yaml with the shares.
    """
    return rng.choice(list(partial_payment_schedules()))


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

    `intent` says whether `documents` is meant to establish both facts. It is not derivable from
    the list: a claim carrying one payment document and nothing else looks identical whether the
    subject was omitted on purpose or was never found, and the two are different events — see
    `EvidenceIntent`.
    """

    claim_id: str
    persona_id: str
    category: str
    verdict: Verdict
    documents: tuple[DocumentPlan, ...]
    issued_at: datetime
    cause: str | None = None
    intent: EvidenceIntent = EvidenceIntent.COMPLETE
    # Passed straight to `content_builder`. `None` means "the builder's own default".
    # CLAIM-LEVEL, not per document: a basket sized to overrun an annual balance is sized
    # against the claim, and `subject_document` below is what stops a second document from
    # doubling it.
    coverage_target: Decimal | None = None
    item_count: int | None = None
    # INTO HOW MANY PARTS THE OBLIGATION IS SETTLED, by the name `config/generation.yaml` gives the
    # arrangement — `None` for every claim paid in one, which is every claim but a `partially_paid`
    # one. CLAIM-LEVEL like the two above, and for the same reason: it decides what the subject
    # document PRINTS and what the payment document then STATES, so it belongs to neither document
    # alone. `assembler` hands it to the subject builder and reads the resulting instalment back
    # off the built document to size the payment.
    schedule: str | None = None

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

        🔴 A CLAIM PLANNED AS AN EVIDENCE GAP HAS NONE, AND IS REFUSED BY ITS INTENT RATHER THAN BY
        ITS COUNT. Both refusals raise; only one of them says the plan is inconsistent. Keeping them
        apart is the whole point of `EvidenceIntent`: a caller that reaches for the subject of a gap
        claim has assumed every claim carries one, and telling it that zero documents were found
        would send it looking for the missing document instead of at its own assumption.
        """
        if self.intent is EvidenceIntent.EVIDENCE_GAP:
            raise ValueError(
                f"claim {self.claim_id} was planned as {EvidenceIntent.EVIDENCE_GAP.name}: it "
                "carries a payment document and nothing that states what was bought, ON PURPOSE — "
                "that is the claim `policy_engine` labels `insufficient_evidence` with the cause "
                "`subject_not_evidenced`. Ask `intent` before asking for a subject document; "
                "nothing is missing here."
            )
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
            fx_rates=[
                FxRateApplied(
                    doc_id=applied.doc_id,
                    currency=applied.currency,
                    rate=applied.rate,
                    on=applied.on,
                )
                for applied in evaluation.fx
            ],
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

    ⚠️ AND NOT EVERY SUBJECT ARCHETYPE COUNTS TOWARDS THE PAIR — see `_pairable_subjects`, which
    is the same predicate `_select_documents` builds from. A category served by a товарний чек and
    a payment document would satisfy "a subject exists and a payment exists" and still not be
    completable, because nothing settles a sales slip.

    ⚠️ NOR IS EVERY PAIRABLE SUBJECT PAYABLE BY EVERY PAYMENT — see `_settleable_subjects`. Since
    the registry holds documents in a second currency, "a subject exists and a payment exists" can
    be true of two documents the oracle would refuse to label together.
    """
    both = [a for a in candidates if all(evidence_of(a))]
    payments = _payment_archetypes(candidates)
    subjects = _settleable_subjects(_pairable_subjects(candidates), payments)
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

    ⚠️ THE NARROWING IS BOUND BY THE STRICTEST CAUSE, NOT BY THE ONE THIS CLAIM WILL DRAW. One
    cause, `subject_not_evidenced`, needs no pair at all — a payment archetype alone realizes
    it, which every category here has — so a receipt category could carry that one. It is filtered
    out regardless because `plan_claim` chooses the CATEGORY BEFORE THE CAUSE, and a category
    admitted for the cause that happens to be drawn would be a category the other three causes
    cannot use. Widening this means drawing the cause first, which is a reordering of the seed
    stream and a decision nobody has needed to take.

    ⛔ AND ONE CONSTRAINT IS NOT ENFORCED HERE AT ALL: `counterparty_mismatch` needs a category with
    at least TWO sellers, since the payment has to name a party the invoice does not. This module
    does not model vendors — config/vendors.json is `assembler`'s — and importing it to filter here
    would put a fact about the vendor file into the planner, where nothing else about a vendor
    lives. `assembler._payee_the_payment_names` refuses instead, naming the cause and the category,
    which is a refusal a reader can act on. Every Ukrainian category carries five sellers or more,
    so the refusal is unreachable today.

    `partially_paid` NARROWS THE SAME WAY AND THEN ONCE MORE. It also needs a pair — a payment
    settles PART of an obligation some other document states — and it additionally needs that
    other document to be a class that can STATE the arrangement, which is what
    `_instalment_subjects` answers. The second step excludes nothing today and is written for the
    day a second pairable subject class is registered.

    `not_proof_of_payment` NARROWS THE OTHER WAY, and the two narrowings are not versions of one
    rule. It is realized by a claim carrying a SUBJECT DOCUMENT ALONE, so it needs a category some
    subject-only archetype covers — the opposite requirement to the pair rule above, which is
    about two documents being able to disagree. A category documented by a fiscal receipt and
    nothing else is complete, plannable for three verdicts, and cannot realize this one: the
    receipt proves its own payment, so there is no gap to leave.

    ⚠️ THE SECOND NARROWING EXCLUDES NOTHING TODAY, and it is written anyway. Every Ukrainian
    category carries the invoice archetype, so every documentable one already has a subject-only
    document; the filter earns its place the day a category is documented by a receipt alone,
    which the registry has held before and will again as jurisdictions land. Without it that
    category would be drawn here and refused inside `_select_documents` — the failure landing a
    stage from its cause, which is what this whole function exists to prevent. It is exercised by
    a test against a hand-built registry rather than by the live one.

    WHY NOT NARROW GLOBALLY. Because `covered` and `partially_covered` are realizable in BOTH
    shapes: a receipt category is perfectly plannable for them, and dropping it from every claim
    would remove the only `fiscal_receipt` documents the corpus has, to satisfy a constraint that
    belongs to one verdict out of four. The narrowing follows the verdict because the CONSTRAINT
    follows the verdict — a global filter would encode one verdict's requirement as a property of
    the persona.
    """
    open_balance = [
        category
        for category in documentable_categories(persona)
        if ledger.remaining(persona.persona_id, category) > 0
    ]
    if verdict in (Verdict.INSUFFICIENT_EVIDENCE, Verdict.PARTIALLY_PAID):
        # BOTH NEED A PAIR, and a category served by a self-contained document does not get one:
        # `_select_documents` prefers that shape, and one document can neither contradict itself
        # nor settle part of itself.
        paired = [
            category
            for category in open_balance
            if not any(
                all(evidence_of(archetype))
                for archetype in archetypes_for(persona.location.country, category)
            )
        ]
        if verdict is Verdict.INSUFFICIENT_EVIDENCE:
            return paired
        # 🔴 AND `partially_paid` NEEDS THE SUBJECT OF THAT PAIR TO BE ABLE TO STATE THE TERM. A
        # pair can disagree about an amount whatever its subject class is; only a class that makes
        # an OFFER can say the offer is settled in parts, which is a narrower requirement and not a
        # stricter version of the same one. It excludes nothing today — the invoice is the only
        # pairable subject there is — and it is written because the day a second one is registered
        # is the day a category would be drawn here and refused inside `_select_documents`.
        return [
            category
            for category in paired
            if _settleable_instalment_subjects(
                archetypes_for(persona.location.country, category)
            )
        ]
    if verdict is Verdict.NOT_PROOF_OF_PAYMENT:
        return [
            category
            for category in open_balance
            if any(
                evidence_of(archetype) == Evidence(True, False)
                for archetype in archetypes_for(persona.location.country, category)
            )
        ]
    return open_balance


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

    Inside, because that is what every verdict except `rejected` needs. A payment dated
    outside the window is what drives the `rejected` branch on the period, and choosing it is
    a decision of the planner rather than an accident of the calendar — which is why the
    period is read from policy.yaml and never from today's date. `_payment_outside_period`
    below is the other half of that decision, and it reads the window through the same
    `active_period` — one function deciding what the window IS, two deciding what to do about it.
    """
    start, end = active_period()

    day = start + timedelta(days=rng.randint(0, (end - start).days))
    # Trading hours, so a receipt is not timestamped at four in the morning.
    return datetime(day.year, day.month, day.day, rng.randint(9, 20), rng.randint(0, 59),
                    rng.randint(0, 59))


def _payment_outside_period(rng: random.Random, issued_at: datetime) -> datetime:
    """The same moment ONE BENEFIT PERIOD earlier or later, which is outside the active window.

    The whole of what `rejected` needs: `policy_engine.evaluate_claim` checks the period against
    the claim's payment date and against nothing else, so a claim is uncovered by WHEN it was paid
    as soon as this date leaves the window. Nothing about the documents changes — the evidence of
    such a claim is complete and flawless, which is exactly why the verdict is `rejected` and not
    `insufficient_evidence`.

    🔴 DISPLACED BY A WHOLE PERIOD RATHER THAN REDRAWN, and the arithmetic is what guarantees the
    result instead of a range somebody has to check: for any date d in [start, end], d - span lands
    at or before start - 1 and d + span at or after end + 1, where `span` is the window's length in
    days. So the displacement puts the payment outside the window for ANY period this file
    declares, and it lands in the ADJACENT benefit year — the same calendar position, one year
    early or late, which is what an expense filed against the wrong period actually looks like. A
    fresh draw would have needed a range chosen by hand and a test to keep it honest.

    🔴 THE SIDE IS DRAWN, and that is not decoration. Displacing only forwards would make every
    `rejected` claim in the corpus later than every other claim, and a consumer could then key the
    verdict on "after the end of the window" rather than on "outside it" — a shortcut the dataset
    would have taught. Both signs occur at the same rate for want of any observation that would
    justify preferring one.

    ⚠️ IT HONOURS A DATE THAT IS ALREADY OUTSIDE. A caller may name `issued_at` itself, and
    displacing an out-of-window date by a period would move it back IN — realizing the opposite of
    what was asked for. The check is on the window rather than on how the date arrived, because
    that is the property the verdict rests on.
    """
    start, end = active_period()
    if not start <= issued_at.date() <= end:
        return issued_at
    span = timedelta(days=(end - start).days + 1)
    return issued_at + rng.choice((-1, 1)) * span


def draw_claim_dates(rng: random.Random, count: int) -> list[datetime]:
    """`count` timestamps inside the active period, ascending.

    Drawn together and sorted, because cumulative limits bind in date order: a persona
    whose claims arrived in a random order would exhaust its balance on whichever claim
    happened to be planned last rather than on the one that happened last.
    """
    return sorted(_draw_date_in_period(rng) for _ in range(count))


def _draw_archetype(rng: random.Random, pool: list[Archetype]) -> Archetype:
    """One archetype from a pool, weighted where config/generation.yaml says so.

    🔴 THE ALL-OR-NONE RULE, enforced here because this is the one place the table is
    read: a pool consults `archetype_shares` only when EVERY member is listed; a pool
    with no member listed draws uniformly, exactly as every pool did before the table
    existed; and a pool with SOME members listed is refused — a half-declared pool is a
    distribution nobody chose, and defaulting the missing weight would choose it
    silently. The weights move no label: they decide which RENDERING a claim's document
    takes after the claim's shape is already fixed, which is why they live in
    generation.yaml rather than beside `verdict_mix`.
    """
    weights = archetype_draw_weights()
    listed = [archetype for archetype in pool if archetype.slug in weights]
    if not listed:
        return rng.choice(pool)
    if len(listed) != len(pool):
        missing = sorted(a.slug for a in pool if a.slug not in weights)
        raise ValueError(
            f"`archetype_shares` in config/generation.yaml lists {len(listed)} of "
            f"{len(pool)} archetypes of this pool and not {missing} — a half-declared "
            "pool is a distribution nobody chose. List every member or none."
        )
    return rng.choices(pool, weights=[weights[a.slug] for a in pool], k=1)[0]


def _select_documents(
    rng: random.Random,
    candidates: list[Archetype],
    issued_at: datetime,
    *,
    intent: EvidenceIntent = EvidenceIntent.COMPLETE,
    payment_precedes_subject: bool = False,
    subject_states_instalments: bool = False,
    payment_must_cite: bool = False,
) -> tuple[DocumentPlan, ...]:
    """The documents a claim carries: both facts a reimbursement rests on, or the one
    `intent` asks for.

    A reimbursement needs to know what was bought and that it was paid for
    (`document_evidence` in policy.yaml), and the two need not come from the same
    document. FOUR SHAPES ARE BUILT. Two of them are decided by what the registry offers,
    in this order:

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

    ⚠️ THE SUBJECT OF A PAIR IS NARROWER THAN "AN ARCHETYPE THAT PROVES THE SUBJECT" — see
    `_pairable_subjects`. A payment settles an obligation, and a class that states none is not
    half of a pair however well it states what was bought.

    ⚠️ AND NARROWER AGAIN BY CURRENCY — see `_settleable_subjects`. The two documents of a pair are
    stated in ONE currency, because the oracle refuses to label a claim whose documents are stated
    in two. The subject is drawn from what some payment here can settle, and the payment from the
    subject's own currency.

    🔴 `subject_states_instalments` NARROWS IT AGAIN, and it is what `partially_paid` needs: the
    subject has to be a class that can PRINT the term saying its obligation is settled in parts
    (`STATES_AN_INSTALMENT_TERM`), because the verdict rests on that printed marker and on nothing
    else. It changes no date and no amount — the whole difference between such a claim and an
    ordinary one is a line on the invoice and the size of the payment beside it, and the payment's
    size is `assembler`'s to apply.

    🔴 `payment_must_cite` NARROWS THE PAYMENT SIDE THE SAME WAY, and it is what `subject_mismatch`
    needs: the payment has to be an archetype whose page can PRINT the рахунок it settles
    (`_CITES_THE_SETTLED_DOCUMENT`), because the defect is the citation and a class with no
    purpose line has nowhere to carry one. It changes no date and no amount either — the wrong
    reference is `assembler`'s to draw.

    🔴 `intent` IS THE THIRD AND FOURTH SHAPES, AND THEY ARE THE ONES THAT ARE NOT ABOUT WHICH
    ARCHETYPES EXIST.
    `EvidenceIntent.EVIDENCE_GAP` asks for a PAYMENT DOCUMENT ALONE — the claim then states that
    money moved and never what it bought, which `policy_engine` labels `insufficient_evidence` with
    the cause `subject_not_evidenced`. It is a separate branch rather than a fallback for the same
    reason it is a named value: the refusal below must go on meaning "this registry cannot document
    this claim", and a gap that arrived by falling through it would be indistinguishable from that.
    A payment-only archetype is required even so — the gap is in the SUBJECT, and a claim proving
    neither fact is a different label nobody asked for.

    `EvidenceIntent.PAYMENT_GAP` asks for the OPPOSITE ONE — a SUBJECT DOCUMENT ALONE, so that the
    claim states what was bought and never that money moved, which `policy_engine` labels
    `not_proof_of_payment` with no cause at all. Same construction, other slot.

    ⛔ NEITHER DRAWS A SELF-CONTAINED DOCUMENT. A fiscal receipt proves both facts, so a claim
    carrying one has no gap of either kind to label: only the `proves_subject: false` payment
    classes can realize the first, and only the `proves_payment: false` subject classes the
    second.

    Where neither shape can be assembled the planner refuses. It does not build a claim
    out of whichever archetypes exist and leave the engine to discover that half the
    evidence is missing: the verdict is chosen first here, and a claim short of a fact is
    short of it because this function was asked for that, never because a template was absent.
    """
    payments = _payment_archetypes(candidates)
    if payment_must_cite:
        payments = [a for a in payments if a.slug in _CITES_THE_SETTLED_DOCUMENT]
        if not payments:
            raise ValueError(
                "a subject-mismatch claim is a payment citing the wrong рахунок on its own "
                "purpose line, and no registered payment archetype here can print one: "
                f"{sorted(a.slug for a in candidates)}. See "
                "`claim_planner._CITES_THE_SETTLED_DOCUMENT`."
            )
    if intent is EvidenceIntent.EVIDENCE_GAP:
        if not payments:
            raise ValueError(
                "an evidence gap is a claim proving the payment and not the subject, and no "
                "registered archetype here proves the payment alone: "
                f"{sorted(a.slug for a in candidates)}. A claim that establishes neither fact is "
                "not this cause and is not what was asked for."
            )
        return (DocumentPlan(archetype=_draw_archetype(rng, payments), issued_at=issued_at),)

    if intent is EvidenceIntent.PAYMENT_GAP:
        # 🔴 EVERY SUBJECT-ONLY ARCHETYPE IS ELIGIBLE HERE, INCLUDING THE ONES NO PAYMENT CAN
        # SETTLE, and that is the difference from the pair branch below rather than an oversight.
        # The claim has no payment document to relate this one to, so `_SETTLED_BY_A_PAYMENT` —
        # which is about a RELATION between two documents — has nothing to say about it. It is
        # what lets a товарний чек reach the corpus at all.
        unsettled_subjects = [a for a in candidates if evidence_of(a) == Evidence(True, False)]
        if not unsettled_subjects:
            raise ValueError(
                "a payment gap is a claim stating what was bought and never that money moved, "
                "and no registered archetype here states the subject alone: "
                f"{sorted(a.slug for a in candidates)}. A self-contained receipt proves its own "
                "payment, so a claim carrying one has no such gap to label."
            )
        return (
            DocumentPlan(
                archetype=_draw_archetype(rng, unsettled_subjects), issued_at=issued_at
            ),
        )

    both = [a for a in candidates if all(evidence_of(a))]
    if both:
        return (DocumentPlan(archetype=_draw_archetype(rng, both), issued_at=issued_at),)

    subjects = _settleable_subjects(
        _instalment_subjects(candidates)
        if subject_states_instalments
        else _pairable_subjects(candidates),
        payments,
    )
    if subject_states_instalments and not subjects:
        raise ValueError(
            "a partially paid claim is a payment settling one part of an obligation some other "
            "document STATES is settled in parts, and no registered archetype here can print such "
            f"a term: {sorted(a.slug for a in candidates)}. Without it the pair is a payment for "
            "the wrong amount, which is a different verdict."
        )
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
        subject = _draw_archetype(rng, subjects)
        # 🔴 THE PAYMENT IS DRAWN FROM THE SUBJECT'S OWN CURRENCY, and the pool is never empty:
        # `_settleable_subjects` admitted this subject precisely because some payment here is
        # stated in it. Two draws either way, in the order they have always happened, so a
        # single-currency registry takes the same values from the generator as before this
        # narrowing existed.
        return (
            DocumentPlan(archetype=subject, issued_at=subject_at),
            DocumentPlan(
                archetype=_draw_archetype(
                    rng, [p for p in payments if p.currency == subject.currency]
                ),
                issued_at=issued_at,
            ),
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
    schedule: str | None = None
    intent = EvidenceIntent.COMPLETE
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
        # belongs to a partially_covered claim". The cause decides WHICH FACT the claim fails to
        # establish; the engine decides whether it actually failed, and nothing here assumes it.
        cause = cause or draw_insufficient_evidence_cause(rng)
        # 🔴 FIVE BUILDABLE CAUSES, REALIZED IN THREE DIFFERENT PLACES, and the list is here because
        # THIS is where a cause is aimed at. `SUBJECT_NOT_EVIDENCED` is a SHAPE and is realized
        # below by naming an intent; `PAYMENT_PRECEDES_SUBJECT` is an ORDER and is realized in
        # `_select_documents`; `AMOUNT_MISMATCH`, `COUNTERPARTY_MISMATCH` and `SUBJECT_MISMATCH`
        # are CONTENT — what the payment document states, whom it names and which рахунок it
        # cites — and are realized by `assembler`, which reads `plan.cause` when it sizes the
        # payment (`_amount_the_payment_states`), when it draws the party the payment names
        # (`_payee_the_payment_names`) and when it draws the reference the payment cites
        # (`_reference_the_payment_cites`). `SUBJECT_MISMATCH` additionally narrows the payment
        # archetype in `_select_documents` — the page has to be one that can PRINT a citation. A
        # plan is the whole of the intent in every case; nothing downstream infers a defect from
        # a document.
        if cause not in (
            SUBJECT_NOT_EVIDENCED,
            AMOUNT_MISMATCH,
            PAYMENT_PRECEDES_SUBJECT,
            COUNTERPARTY_MISMATCH,
            SUBJECT_MISMATCH,
        ):
            raise ValueError(
                f"policy.yaml declares no such buildable insufficient_evidence cause: {cause!r}. "
                f"{cause!r} may still be a cause the ENGINE returns — see "
                "`claim_planner._UNREALIZABLE_ROUTES`, which records the routes to a realizable "
                "verdict that this planner cannot aim at."
            )
        if cause == SUBJECT_NOT_EVIDENCED:
            # The one cause realized by the SHAPE of the evidence rather than by its content: no
            # subject document is planned at all. Named rather than achieved by omission — see
            # `EvidenceIntent`.
            intent = EvidenceIntent.EVIDENCE_GAP
    elif verdict is Verdict.REJECTED:
        # 🔴 BOTH ROUTES ARE DRAWN NOW, per `rejected_routes` in policy.yaml, and `plan.cause`
        # records which one — `OUTSIDE_PERIOD`, which is also the cause the label will carry, or
        # `ZERO_COVERAGE`, a planner name for a route whose label carries no cause at all. The
        # engine tells a consumer the two apart by the PRESENCE of the cause; a reader of a plan
        # tells them apart by value.
        cause = cause or draw_rejected_route(rng)
        if cause == OUTSIDE_PERIOD:
            # 🔴 THE DATE IS DISPLACED HERE, AFTER IT WAS DRAWN OR NAMED, and that ordering is
            # what makes the branch cheap: every other verdict wants a payment inside the window,
            # this one wants the same claim with its money moved outside it, and nothing else
            # about the plan differs. The subject document keeps its lead from the payment, so a
            # claim whose invoice falls INSIDE the period while its payment does not is an
            # ordinary outcome here — that is the case a consumer checking the wrong document's
            # date gets wrong, and the corpus has to contain it.
            #
            # ⚠️ IT BREAKS THE ASCENDING ORDER `plan_claims` DRAWS ITS DATES IN, and the ledger
            # does not care: a `rejected` claim reimburses nothing (`policy_engine`'s `refused`),
            # so it consumes no balance and cannot change what a later claim of the same persona
            # has left. The order matters because cumulative limits bind in it; a claim outside
            # the period binds nothing.
            issued_at = _payment_outside_period(rng, issued_at)
        elif cause == ZERO_COVERAGE:
            # 🔴 THE OTHER ROUTE IS A BASKET, NOT A DATE: an ordinary claim, inside the period,
            # whose every line the category excludes. `coverage_target` at ZERO is how the
            # builder is asked for that — the same label-first knob `mixed_items` uses, at the
            # value that describes "no covered money at all" — and the engine answers `rejected`
            # off the line items alone (`verdict_for`), with an empty `imperfection`. The date
            # stays where it was drawn, which is the point: on this route the payment date is
            # ordinary, so a consumer cannot read the verdict off the calendar.
            coverage_target = Decimal(0)
        else:
            raise ValueError(
                f"policy.yaml declares no such rejected route: {cause!r}. The two it declares "
                f"are {OUTSIDE_PERIOD!r} and {ZERO_COVERAGE!r} — see `rejected_routes`."
            )
    elif verdict is Verdict.NOT_PROOF_OF_PAYMENT:
        # 🔴 THE WHOLE OF THE MECHANISM IS THE SHAPE OF THE EVIDENCE, and it is named rather than
        # arrived at: the claim carries a document that states what was bought and NOTHING that
        # attests a payment. Everything else about the plan is ordinary — the basket is drawn
        # covered, the date sits inside the period — because the engine decides this verdict from
        # the document types before it reads either.
        if cause is not None:
            raise ValueError(
                f"`not_proof_of_payment` carries no cause and {cause!r} was named. There is one "
                "slot of `document_evidence` behind this verdict — that money moved — and one way "
                "to fail it, so there is nothing for a cause to distinguish; see "
                "`policy_engine.evaluate_claim`, which returns it with an empty `imperfection`."
            )
        intent = EvidenceIntent.PAYMENT_GAP
    elif verdict is Verdict.PARTIALLY_PAID:
        # 🔴 THE ONE VERDICT REALIZED BY WHAT A DOCUMENT PRINTS. Everything else about the plan is
        # ordinary — a complete pair, a covered basket, a payment inside the period — and the
        # difference is the schedule drawn here: it makes the invoice state that its obligation is
        # settled in parts, and `assembler` then sizes the payment to one of them.
        #
        # ⛔ NO CAUSE, like `not_proof_of_payment`. policy.yaml gives this verdict one mechanism,
        # the engine returns it with an empty `imperfection`, and a plan naming a cause would be
        # aiming at a distinction the label cannot carry.
        if cause is not None:
            raise ValueError(
                f"`partially_paid` carries no cause and {cause!r} was named. There is one way to "
                "reach it — a payment equal to an instalment its subject document prints — so "
                "there is nothing for a cause to distinguish; see "
                "`policy_engine._settles_one_instalment`."
            )
        schedule = draw_payment_schedule(rng)
    elif cause is not None:
        raise ValueError(
            f"a cause belongs to a partially_covered, an insufficient_evidence or a rejected "
            f"claim, not to {verdict.value!r}"
        )

    return ClaimPlan(
        claim_id=claim_id,
        persona_id=persona.persona_id,
        category=category,
        verdict=verdict,
        cause=cause,
        documents=_select_documents(
            rng, candidates, issued_at,
            intent=intent,
            payment_precedes_subject=cause == PAYMENT_PRECEDES_SUBJECT,
            subject_states_instalments=schedule is not None,
            payment_must_cite=cause == SUBJECT_MISMATCH,
        ),
        issued_at=issued_at,
        intent=intent,
        coverage_target=coverage_target,
        item_count=item_count,
        schedule=schedule,
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

    ⚠️ "IN DATE ORDER" IS ABOUT THE CLAIMS THAT SPEND, and one kind does not. A plan aimed at
    `rejected` has its payment displaced out of the benefit period by `plan_claim`, so the dates
    coming out of this generator are no longer ascending. What the order is FOR survives intact:
    such a claim reimburses nothing and consumes no balance, so it cannot change what a later
    claim of the same persona has left.

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
