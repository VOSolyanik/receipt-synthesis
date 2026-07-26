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
    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_PROOF_OF_PAYMENT = "not_proof_of_payment"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIALLY_PAID = "partially_paid"


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
    """The label record of a single rendered document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_file: str
    doc_type: DocType
    language: str
    currency: str
    amount: Money
    date: date
    counterparty: str
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

    ``covered_fraction``, ``verdict_basis`` and ``policy_trace`` are populated by the
    policy engine. Until it exists they stay empty rather than being guessed: an unset
    field is honest, a defaulted one would be a label nobody derived.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    persona_id: str
    category: str
    documents: list[str]
    verdict: Verdict
    covered_fraction: float | None = None
    linked: bool = False
    imperfection: list[str] = Field(default_factory=list)
    verdict_basis: list[VerdictBasis] = Field(default_factory=list)
    policy_trace: list[str] = Field(default_factory=list)
