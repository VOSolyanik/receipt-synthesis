"""Ground-truth models.

Because generation is label-first, these models are not a description of a rendered
document — they are its specification. Every field is decided before a pixel exists,
which is what makes the dataset annotation-free. See docs/architecture.md#ground-truth.

Money is carried as ``Decimal`` so that arithmetic on amounts stays exact while a
document is being built. It is serialized to a JSON number, following the sample in the
architecture reference — the artifact therefore holds a float, and the exactness lives
in Python only. Nothing here enforces that the line items sum to the total; that
invariant is the content builder's, and is checked there.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from receipt_synth import __version__

# Exact in Python, a plain JSON number on the way out.
_as_number = PlainSerializer(float, return_type=float, when_used="json")

Money = Annotated[Decimal, _as_number]
Quantity = Annotated[Decimal, _as_number]

# Bounding box in image pixels, ``[x, y, width, height]`` — the COCO convention and the
# shape ``getBoundingClientRect()`` returns, so the renderer copies it across unchanged
# and the degrader can hand it to Albumentations without a format conversion.
BBox = tuple[float, float, float, float]


class Country(StrEnum):
    UA = "UA"
    PL = "PL"
    DE = "DE"
    ES = "ES"


class DocType(StrEnum):
    """Document classes. Values match the keys of ``document_evidence`` in policy.yaml."""

    FISCAL_RECEIPT = "fiscal_receipt"
    PAYMENT_CONFIRMATION = "payment_confirmation"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    ACT = "act"
    ORDER_SCREENSHOT = "order_screenshot"
    NON_FISCAL_RECEIPT = "non_fiscal_receipt"


class Capture(StrEnum):
    """How the document reached the verifier — see docs/architecture.md#degradation."""

    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    SCAN = "scan"


class Verdict(StrEnum):
    """The answer a claim gets. Values match the keys of ``verdict_mix`` in policy.yaml.

    Each member below states WHAT IT ITSELF IS ABOUT, and never by contrast with another
    verdict. Three of them are easy to collapse into one another, and the construction is what
    keeps them apart: while
    one is written as "the case that is not the other", editing either silently moves the other.
    See ``verdict_notes.definitions_name_their_own_slot`` in config/labelling-schema.yaml.

    ``document_evidence`` carries two facts plus the linkage between them — three slots — and
    the two evidence verdicts divide them:

    * ``NOT_PROOF_OF_PAYMENT`` — the MONEY-MOVED slot is unestablished: every document of the
      claim is of a type whose ``proves_payment`` is ``false`` in policy.yaml's
      ``document_evidence``. A bare invoice, an act, an order screenshot. Decided from the
      type; amounts, baskets and dates are not consulted. Carries no cause — one slot, one way
      to fail it.
    * ``INSUFFICIENT_EVIDENCE`` — the other two slots. WHAT WAS BOUGHT is unestablished when no
      document of the claim is of a type that states it (cause ``subject_not_evidenced``); ONE
      TRANSACTION is unestablished when a subject document and its payment both exist and fail
      a cross-check (``amount_mismatch``, ``payment_precedes_subject``).

    ``REJECTED`` is not about the evidence at all — the documents establish every slot, and the
    policy still does not cover the claim, on either of two axes:

    * by WHAT was bought — nothing on the documents is covered by the claimed category, a
      property of the LINE ITEMS resolved against ``covered_items`` / ``excluded_items``, which
      is where policy.yaml's ``coverage`` block sends a covered fraction of zero. No cause.
    * by WHEN it was paid — the payment falls outside the active ``period``. Cause
      ``outside_period``, which is what tells the two axes apart.

    HISTORY, not part of the definition above: the out-of-window case was labelled
    ``INSUFFICIENT_EVIDENCE`` until the revision that added ``outside_period`` here. It moved
    because that verdict's slots are all established for such a claim. See
    ``verdicts.rejected.by_when_it_was_paid`` in config/labelling-schema.yaml.

    Appended to rather than reordered: the member order is the row order of the assembler's
    balance report, so reordering would change output that a seed is supposed to determine.
    """

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_PROOF_OF_PAYMENT = "not_proof_of_payment"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIALLY_PAID = "partially_paid"
    REJECTED = "rejected"


class VerdictBasis(StrEnum):
    """What a verdict depends on.

    ``DOCUMENTS`` alone means the verdict is derivable from the images; adding
    ``ACCOUNT_STATE`` means it also needs the persona's spending history, which no
    consumer can read off a receipt. Document-understanding metrics belong on the first
    subset only.
    """

    DOCUMENTS = "documents"
    ACCOUNT_STATE = "account_state"


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: Country
    city: str


class FamilyMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: str  # spouse | child
    name: str
    birth_date: date


class Persona(BaseModel):
    """A synthetic person. Fixes the context every document of theirs inherits:
    jurisdiction, currency, language, and which benefit categories they hold."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    full_name: str
    location: Location
    home_currencies: list[str]
    languages: list[str]
    tax_id: str  # РНОКПП in UA: individual taxpayer number, checksum-correct
    family: list[FamilyMember] = Field(default_factory=list)
    benefit_categories: list[str]


class LineItem(BaseModel):
    """One printed line of a receipt, plus the labels behind it.

    ``price`` is the unit price; the line total is ``qty * price`` and the sum of those
    is the document amount.

    ``item_kind`` is the join key of the vocabulary in policy.yaml — it is never printed
    on a document. It is kept in the ground truth because it *is* the label space a
    consumer maps a printed line onto (docs/architecture.md#the-item-kind-vocabulary);
    ``covered`` is the answer for this category, ``item_kind`` is what the answer is
    about.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    item_kind: str
    qty: Quantity
    price: Money
    covered: bool
    vat_letter: str | None = None  # ПДВ-літера: VAT rate code printed per line (UA: А=20%, В=7%)


class DocGroundTruth(BaseModel):
    """The label record of a single rendered document.

    ONE MODEL FOR EVERY DOCUMENT CLASS, so a field a class does not print is ``None`` on its
    records rather than absent. ``amount`` is the exception that is never optional, and its
    meaning is the same question on every class — *the amount this document is about* — answered
    by that class's own requisite: the basket total on a fiscal receipt, and on a payment
    confirmation the TRANSFER amount, which is 📄 what the National Bank's instruction calls the
    amount of the operation and is NOT the largest number printed on the page. See
    ``content_builder.PaymentConfirmation``.

    ``amount_due`` is ``None`` for a document type that prints no such line, and is EQUAL TO
    ``amount`` wherever it is populated today, because the discount and the rounding that make
    the two differ are zero in this version. A consumer must therefore not report accuracy on
    it: a system that echoes ``amount`` satisfies it perfectly while having read nothing. The
    reason the divergence is deferred is the policy's silence about distributing a
    basket-level discount over per-line coverage, not the difficulty of printing it — see
    ``content_builder.PrroReceipt.amount_due`` and the field's entry in
    config/labelling-schema.yaml.

    ``fee`` and ``total_charged`` ARE NOT THAT SAME DEFERRAL WEARING A SECOND NAME, and the
    contrast is worth having in one place. A discount is a property of a basket, so making it
    non-zero needs a rule for splitting it across covered and non-covered lines that policy.yaml
    does not have. A bank's fee is charged on the payment, never enters a basket, and is not
    reimbursable, so nothing about it is undecided: it is 👁 non-zero on about a third of real
    confirmations and is generated that way, and ``total_charged`` genuinely differs from
    ``amount`` whenever it is.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_file: str
    doc_type: DocType
    language: str
    currency: str
    amount: Money
    # ДО СПЛАТИ on a Ukrainian receipt: the total less any discount, plus cash rounding.
    amount_due: Money | None = None
    # The bank's own charge for executing the payment — комісія. `None` on a class that has no
    # such requisite, and NEVER reimbursable: it pays for a banking service rather than for
    # anything a benefit category covers, so no policy limit applies to it.
    fee: Money | None = None
    # `amount` + `fee`: everything that left the payer's account. Where a confirmation prints it
    # («Загальна сума») it is the largest number on the page, which is exactly why it is labelled
    # apart from `amount` — an extractor that reaches for the most salient figure is measurably
    # wrong rather than invisibly wrong.
    total_charged: Money | None = None
    date: date
    counterparty: str
    # The party the document names OPPOSITE `counterparty`. `counterparty` is the other side of
    # the transaction from the claimant — the seller on a receipt, the payee on a confirmation —
    # and `payer` is the claimant's own side, which only a document naming both parties carries.
    # `None` where the class names one party; the empty string is NOT that case, and the
    # difference is deliberate: 👁 a confirmation may print a payer field whose value is a
    # hyphen, and a real extractor reads that hyphen as a value.
    payer: str | None = None
    # Призначення платежу — free text written by the payer. 🔴 It never names what was bought
    # (👁 0 of 7 observed), which is the observation behind `proves_subject: false` for the
    # payment-confirmation type in policy.yaml.
    payment_purpose: str | None = None
    # The bank's own number for the document, 👁 present on 8 of 8 and 📄 mandatory. It is the
    # deduplication key of this class — the authorization code is not, being six digits and
    # unique only within an issuer and a window.
    document_code: str | None = None
    # Код авторизації — six digits, and only where a card operation was authorized (👁 4 of 8).
    auth_code: str | None = None
    # EMPTY on a class that lists nothing — a payment confirmation proves one movement of money
    # and 👁 8 of 8 carry no table of items at all. An empty list is the statement "this document
    # lists nothing", which is why the field stays required rather than becoming nullable.
    line_items: list[LineItem]

    has_qr: bool
    qr_is_fiscal: bool
    has_fiscal_number: bool

    capture: Capture
    field_bboxes: dict[str, BBox] = Field(default_factory=dict)

    # Provenance. Not decoration: this is what keeps the origin of an individual file
    # unambiguous once it leaves this repository.
    synthetic: bool = True
    generator_version: str = __version__


class ClaimGroundTruth(BaseModel):
    """The label record of a claim, which may span several documents.

    ``documents`` is a LIST and is the join from a claim to its evidence — a document
    belongs to exactly one claim. Which of them proved what is NOT recorded: the roles are
    derived at evaluation time from a document's ``doc_type`` and ``document_evidence`` in
    policy.yaml (`policy_engine.resolve_evidence`), and a consumer holding only the labels
    re-derives them the same way. Nor is the claim's amount the sum of its documents': an
    invoice and the payment that settles it describe one movement of money, and adding them
    would count it twice.

    ``covered_fraction``, ``verdict_basis``, ``imperfection`` and ``policy_trace`` are
    populated by `policy_engine`, which computes them from the built documents and the
    persona's ledger. They are optional here rather than required because they are
    derived: a record that carries them because something derived them is a label, and one
    that carries them by default would be a guess. The defaults are what an undecided
    claim looks like, not what a claim should look like.

    ``covered_fraction`` is covered amount over total amount, taken from the line items of
    the claim's subject documents and nothing else. A claim held back only by an exhausted
    annual limit therefore still reads 1.0; the limit shows up in ``imperfection``, in
    ``verdict_basis`` and in ``reimbursable_amount``. It is ``None`` where there is no line
    item to compute it from, which is a different statement from ``0.0``.

    ``imperfection`` names why the verdict is what it is where the verdict alone does not
    say. Two causes for ``partially_covered``, declared in policy.yaml; three for
    ``insufficient_evidence`` and one for ``rejected``, named in `policy_engine` because
    policy.yaml declares a cause vocabulary only where it is declaring shares. The three
    sets are disjoint, so a cause determines its verdict; the converse does not hold for
    ``rejected``, whose zero-coverage mechanism carries no cause at all. The whole vocabulary
    is contracted in config/labelling-schema.yaml.

    The two money-ish fields answer different questions and neither substitutes for the
    other: ``covered_fraction`` is *how much of this document belongs to the category*,
    ``reimbursable_amount`` is *how much the plan actually pays out*. They differ exactly
    when an annual limit binds.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    persona_id: str
    category: str
    documents: list[str]
    verdict: Verdict
    covered_fraction: float | None = None
    # What the plan pays out for this claim, in the policy's reporting currency: the
    # covered amount, capped by whatever is left of the annual limit. It is the only
    # number that distinguishes a limit-bound claim, and without it a consumer would have
    # to parse ``policy_trace`` — a field that is prose by design.
    reimbursable_amount: Money | None = None
    linked: bool = False
    imperfection: list[str] = Field(default_factory=list)
    verdict_basis: list[VerdictBasis] = Field(default_factory=list)
    policy_trace: list[str] = Field(default_factory=list)
