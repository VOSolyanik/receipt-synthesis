"""The oracle: what a claim's answer is, and why.

`claim_planner` decides *what to build*; this module decides *what the answer is*. They
are kept apart so that the answer stays a pure function of (claims, policy), with no
rendering anywhere near it — and it has to stay one, because every quality number this
project reports is measured against the labels computed here. A wrong rule in this file
does not surface as a failing test. It surfaces as a silently wrong ground truth that
whatever is being evaluated gets blamed for disagreeing with, and nothing downstream can
catch it.

Every rule below comes out of `config/policy.yaml` — including the one that matters most,
which is whether a given item kind is covered in a given category. The engine resolves
that itself, against `covered_items` and `excluded_items`, and treats the `covered` flag
`content_builder` set on each line as an *assertion to be checked* rather than as an
input to be trusted. An oracle that could not disagree with the thing it is labelling
would be a pass-through with a docstring.

Six rules, in the order they are applied:

1. **Currency.** Limits are expressed in `reporting_currency`, and every archetype this
   generator has emits documents in it. A document in any other currency is therefore a
   contradiction upstream, not a case to handle: the engine raises. It does not convert,
   at any rate, from any source — a converted amount would land in the ground truth as a
   number nothing in the dataset can prove.
2. **Evidence.** A claim's documents are resolved into transactions against
   `document_evidence` in policy.yaml — see `resolve_evidence`. That table carries TWO
   facts a reimbursement rests on, plus the LINKAGE between them: three slots, and each
   slot has its own name. Rules 2 and 3 answer them in order.

   The **money-moved** slot: every document is of a type whose `proves_payment` is false,
   so nothing the claim carries attests a movement of money → `not_proof_of_payment`, with
   no cause, because there is one slot and one way to fail it.

   The **what-was-bought** slot: no document is of a type that states it →
   `insufficient_evidence`, cause `subject_not_evidenced`.
3. **The linkage slot**: a subject document and the payment that settles it both exist and
   have to describe the same purchase — the same amount, and the payment not before the
   subject. Failing either → `insufficient_evidence`, with the cause named in
   `imperfection`; the claim does not establish that *this* payment paid for *this*
   subject.

   Each slot is decided from the claim's own documents and never by comparison with
   another verdict. That construction is deliberate: while two of these were written as
   "the one that is not the other", editing either silently moved the other, and it went
   wrong twice. The rule and its history live in
   `config/labelling-schema.yaml`, `verdict_notes.definitions_name_their_own_slot`.
4. **Period**, checked on the PAYMENT date and on no other, and failing it is `rejected`
   with the cause `outside_period`. A limit is consumed when money moves, so a December
   invoice paid in January is an ordinary January expense and not a period failure. The
   subject document's date is checked for order instead (rule 3), which is a different
   defect with a different name — and now a different verdict.

   HISTORY of this branch, not part of the rule above. It answered
   `insufficient_evidence` until the revision that moved it here. It moved because an
   out-of-window claim leaves every slot of rule 2 and rule 3 ESTABLISHED — the purchase
   happened, the proof is flawless, the documents agree — while the policy still does not
   cover it, by *when* rather than by *what*, symmetrically to the wrong-subject case of
   rule 5. Nothing was collapsed: a CASE moved to the verdict whose own subject matter
   covers it, and both verdicts kept every mechanism that is theirs. The alternative — a
   seventh enum member meaning "outside period" — was rejected because it would pull a
   share into `verdict_mix` and cost the downstream contract another revision, for no gain
   in precision over a named cause on a verdict that already fits.
5. **The verdict.** STRICT, as the prose of the `coverage` block states it: *any*
   non-covered line makes the claim `partially_covered`, however small — the fraction is
   reported, never used as a tolerance. Everything covered is `covered`, and
   `full_threshold` is kept alive as a **self-check** on that case (see `verdict_for`),
   which is the role its own comment gives it. A claim whose covered amount is 0 is
   `rejected` — the third branch that block declares, decided from the line items and from
   nothing else. This is the SECOND route to `rejected`, rule 4 being the first; they are
   told apart by `imperfection`, which the coverage route leaves empty. Note that this rule
   reads amounts, and rule 2 reads document types: neither can answer the other's question,
   which is why they are separate rules rather than one with a branch.
6. **The cumulative annual limit.** Claims are processed per persona per category in date
   order, carrying the balance. When what remains is less than what the document covers,
   the claim is `partially_covered`, and the verdict then also depends on the persona's
   history — which is what `verdict_basis` records.

`covered_fraction` is a property of the line items in every case, including those where
it does not decide the verdict: it is covered amount over total amount and nothing else,
so a claim held back only by an exhausted limit still reports 1.0, and an out-of-period
claim still reports what its lines cover. It is `None` only where there are no lines to
compute it from — a claim evidenced by a payment and nothing else. The money actually
payable is a separate field, `reimbursable_amount`; `policy_trace` carries only what
justified the verdict.

**A claim's amount is not the sum of its documents.** The dominant pair is an invoice plus
the payment that settles it, and those are one movement of money described twice; adding
them would count it twice, and the doubled figure would travel silently into
`covered_fraction`, `reimbursable_amount` and the cumulative limit while the verdict still
came out plausible. So the line items of a claim are the line items of its SUBJECT
documents, counted once per transaction, and a payment document contributes the amount its
subject is cross-checked against rather than money of its own. Several fiscal receipts do
add, because each is its own transaction — that is the same rule, not an exception to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import NamedTuple

from receipt_synth.config import category, load_policy
from receipt_synth.content_builder import KOPIYKA, line_items_total
from receipt_synth.schemas import DocGroundTruth, DocType, LineItem, Verdict, VerdictBasis

# Causes of a `partially_covered` verdict. The strings are the keys of
# `partially_covered_causes` in policy.yaml and land verbatim in `imperfection`.
MIXED_ITEMS = "mixed_items"
LIMIT_EXHAUSTED = "limit_exhausted"

# The four causes below are recorded in `imperfection` beside the two above, under the
# verdicts that are NOT decided by coverage arithmetic — three under `insufficient_evidence`
# and one under `rejected`. One verdict, several named causes — the shape
# `partially_covered` already has, and for the same reason: a consumer can know different
# things about each. A document that is simply absent and two documents that contradict
# each other are not the same problem.
#
# POLICY.YAML NAMES NONE OF THE FOUR. Its only cause vocabulary is
# `partially_covered_causes`, which exists to declare SHARES of the partially_covered
# bucket; none of these four has a share, because no share of a dataset is being sized by
# them. So the names are this module's, they are exported for anyone comparing labels against
# a vocabulary, and they are recorded in config/labelling-schema.yaml, which is the contract
# a consumer reads. If policy.yaml ever grows a cause vocabulary of its own, these move
# there and the constants read it — the same way MIXED_ITEMS reads its own key today.

# Causes of `insufficient_evidence`, which owns two of the three slots of `document_evidence`
# (see the module docstring, rules 2 and 3): the WHAT-WAS-BOUGHT slot, unestablished when no
# document is of a type that states it, and the LINKAGE slot, unestablished when a subject
# document and its payment both exist and fail a cross-check. Each is read off the claim's own
# documents.
SUBJECT_NOT_EVIDENCED = "subject_not_evidenced"
AMOUNT_MISMATCH = "amount_mismatch"
PAYMENT_PRECEDES_SUBJECT = "payment_precedes_subject"

# The one cause of `rejected`, and the only one it needs. `rejected` has two mechanisms —
# a basket the category covers none of, and a payment outside the benefit period — and only
# the second is worth naming: the first is the whole of what the verdict already says,
# while the second says the claim is uncovered by WHEN rather than by WHAT. So a `rejected`
# claim carries either this cause or none, which is how the two mechanisms are told apart
# without parsing `policy_trace`. See the module docstring, rules 4 and 5, for why this case
# is not `insufficient_evidence`.
OUTSIDE_PERIOD = "outside_period"


class PolicyGapError(Exception):
    """Raised where policy.yaml specifies no answer and guessing one would corrupt the
    ground truth.

    Five cases reach it, all of them narrow and all of them deliberate:
    `coverage_of_kind` for an `ambiguous_items` kind and for a kind foreign to the claimed
    category, `_reimbursable` for a claim wholly beyond an exhausted annual limit, and
    `resolve_evidence` for a claim carrying a bank statement and for a set of documents
    that does not pair into transactions.
    """


# =============================================================================
# Policy accessors
# =============================================================================


def full_threshold() -> Decimal:
    """The coverage fraction a fully covered claim must reach.

    Read as a string so the value is the decimal written in the file rather than the
    nearest binary float to it. It is not a tolerance — see `verdict_for` for the one
    thing it is used for.
    """
    return Decimal(str(load_policy()["coverage"]["full_threshold"]))


def annual_limit(category_id: str) -> Decimal:
    """The cumulative reimbursement limit of a category, in the reporting currency."""
    return Decimal(str(category(category_id)["annual_limit"]))


def reporting_currency() -> str:
    return str(load_policy()["reporting_currency"])


def active_period() -> tuple[date, date]:
    """The benefit period, inclusive at both ends."""
    period = load_policy()["period"]
    return date.fromisoformat(str(period["start"])), date.fromisoformat(str(period["end"]))


def verdict_mix() -> dict[Verdict, float | None]:
    """The target distribution over verdicts, in the order policy.yaml declares them.

    A member may be declared with **no share** — `null` in the file — and it comes back as
    `None` rather than as a weight. policy.yaml uses that for a verdict whose place in the
    vocabulary is settled while its share of the dataset is not, and the distinction is
    load-bearing: a `0` share is a decision that behaves like any other weight, so it would
    sum, renormalize and print as a target, and nothing downstream could tell it apart from
    a number somebody chose. `None` cannot be summed or drawn with, so every caller has to
    say what it does with an undecided share instead of quietly assuming one.
    """
    return {
        Verdict(name): None if share is None else float(share)
        for name, share in load_policy()["verdict_mix"].items()
    }


def partially_covered_causes() -> dict[str, float]:
    """How the `partially_covered` bucket splits by cause, in declaration order."""
    return {str(name): float(share) for name, share in
            load_policy()["partially_covered_causes"].items()}


# =============================================================================
# Evidence: what a claim's documents establish between them
# =============================================================================


class Evidence(NamedTuple):
    """What one document TYPE establishes, per `document_evidence` in policy.yaml.

    A tuple rather than a dataclass so that a test can pin a row against the file as a
    pair, which is how the policy writes it.

    TYPE-LEVEL, and only type-level. policy.yaml notes that a specific archetype may
    override its type's defaults — a bank confirmation whose payment purpose spells out
    what was bought does prove the subject, unlike a bare transfer — and no such override
    exists here, because it could not be honoured: the engine sees `DocGroundTruth`, which
    records the document's TYPE and not the archetype that produced it, so an override
    declared on an archetype would be invisible at the moment the verdict is derived.
    Recording the role per document is a change to the label shape, listed in
    config/labelling-schema.yaml as an open decision. Until it is taken, an archetype whose
    evidence differs from its type's default cannot be labelled correctly and must not be
    registered.
    """

    proves_subject: bool
    proves_payment: bool


def document_evidence(doc_type: DocType) -> Evidence:
    """What this document type proves, read from policy.yaml.

    `DocType` and the keys of `document_evidence` are the same vocabulary — schemas.py
    says so — so a missing row is a policy error rather than a case to default, and the
    `KeyError` says which type has none.
    """
    entry = load_policy()["document_evidence"][doc_type.value]
    return Evidence(bool(entry["proves_subject"]), bool(entry["proves_payment"]))


@dataclass(frozen=True)
class Transaction:
    """One movement of money, and the document that says what it bought.

    `subject` and `payment` are the same object for a self-contained document — a fiscal
    receipt is both — and different objects for a split pair.
    """

    subject: DocGroundTruth
    payment: DocGroundTruth

    @property
    def is_split(self) -> bool:
        """Whether the two facts come from two documents, which is when they can disagree."""
        return self.subject is not self.payment


@dataclass(frozen=True)
class EvidenceShape:
    """How a claim's documents resolve into transactions, or which fact they leave open."""

    transactions: tuple[Transaction, ...]
    subject_documents: tuple[DocGroundTruth, ...]
    payment_documents: tuple[DocGroundTruth, ...]

    @property
    def proves_subject(self) -> bool:
        return bool(self.subject_documents)

    @property
    def proves_payment(self) -> bool:
        return bool(self.payment_documents)

    @property
    def complete(self) -> bool:
        return bool(self.transactions)


def _types(documents: Iterable[DocGroundTruth]) -> str:
    """The document types of a set, for a trace line. Sorted and de-duplicated, so the
    sentence does not depend on the order documents were handed over in."""
    return ", ".join(sorted({document.doc_type.value for document in documents}))


def resolve_evidence(documents: Sequence[DocGroundTruth]) -> EvidenceShape:
    """Group a claim's documents into the transactions they evidence.

    This is where "a claim is a list of documents" stops being a container and becomes a
    rule. Everything is read off `document_evidence` in policy.yaml and nothing off the
    documents' own opinion of themselves.

    TWO SHAPES ARE DERIVABLE, and they are the two the evidence table can settle:

    * **self-contained** — every payment document proves its own subject and no other
      document is present. Each is its own transaction, and their money adds: several
      fiscal receipts are several purchases.
    * **one split pair** — exactly one document proves the subject, exactly one proves the
      payment, and they are different documents. One transaction, described twice, whose
      money is counted once.

    Anything else is refused rather than guessed, because the guess would decide the
    claim's amount. A fiscal receipt beside an invoice is either the same purchase twice
    or two purchases one of which was never paid; two payments beside one invoice leave
    open which payment settles it. `document_evidence` answers neither, and picking would
    either double a claim or silently drop a document.

    A BANK STATEMENT IS REFUSED OUTRIGHT, and it is the one worth stating separately. It
    genuinely adds money — it lists several transactions — but a document label carries a
    single `amount` for the whole statement and nothing points at the row this claim is
    about. Whether a statement produces one label or one label per transaction is open on
    both sides of the labelling contract, so identifying "the transaction in the statement
    that corresponds to this claim" is not something the ground truth can do yet.
    """
    for document in documents:
        if document.doc_type is DocType.BANK_STATEMENT:
            raise PolicyGapError(
                f"document {document.doc_id} is a bank_statement, and a claim's amount "
                "cannot be resolved from one: the statement lists several transactions "
                "while its label carries a single amount for the whole document, so "
                "nothing identifies the transaction this claim is about. Whether a "
                "statement is labelled once or once per transaction is undecided on both "
                "sides of config/labelling-schema.yaml; the engine will not pick."
            )
        evidence = document_evidence(document.doc_type)
        if not (evidence.proves_subject or evidence.proves_payment):
            raise PolicyGapError(
                f"document {document.doc_id} is a {document.doc_type.value}, which "
                "policy.yaml says establishes neither what was bought nor that it was "
                "paid for. Such a document is in the claim without contributing to it, "
                "and dropping it silently would hide whatever put it there."
            )

    # Partitioned by identity rather than by equality: two documents of one claim may
    # legitimately carry the same values — a near-duplicate counter-example is in the
    # imperfection catalogue — and `==` on a pydantic model compares fields, so an
    # equality-based partition would silently merge them.
    subjects = tuple(d for d in documents if document_evidence(d.doc_type).proves_subject)
    payments = tuple(d for d in documents if document_evidence(d.doc_type).proves_payment)
    self_contained = tuple(d for d in payments if document_evidence(d.doc_type).proves_subject)
    subject_only = tuple(d for d in subjects if not document_evidence(d.doc_type).proves_payment)
    unsettled = tuple(d for d in payments if not document_evidence(d.doc_type).proves_subject)

    def shape(transactions: tuple[Transaction, ...]) -> EvidenceShape:
        return EvidenceShape(transactions, subjects, payments)

    if not payments or not subjects:
        # One fact is missing outright. Which one, and what that is called, is the
        # caller's to say — both are verdicts rather than shapes.
        return shape(())

    if len(self_contained) == len(payments) and not subject_only:
        return shape(tuple(Transaction(subject=d, payment=d) for d in payments))

    if len(subject_only) == 1 and len(unsettled) == 1 and not self_contained:
        return shape((Transaction(subject=subject_only[0], payment=unsettled[0]),))

    raise PolicyGapError(
        "the documents of this claim do not pair into transactions: "
        f"{len(self_contained)} prove both facts, {len(subject_only)} prove only what was "
        f"bought, {len(unsettled)} prove only the payment. policy.yaml's "
        "`document_evidence` says what each type establishes and nothing about which "
        "payment settles which subject, so pairing them would be the engine deciding the "
        "claim's amount — either counting one purchase twice or dropping a document."
    )


# =============================================================================
# Coverage: resolved from the policy, not read off the line
# =============================================================================


def coverage_of_kind(category_id: str, item_kind: str) -> bool:
    """Whether this item kind is reimbursable in this category, per policy.yaml.

    The join the whole item-kind vocabulary exists for
    (docs/architecture.md#the-item-kind-vocabulary). Coverage is scoped to the category
    and never global to the kind: `sports_nutrition` is excluded under `sport` and
    ambiguous under `vitamins_nutrition`, and policy.yaml says so in a comment because it
    is the kind of thing an implementation gets wrong.

    Two cases raise rather than answer:

    * a kind in `ambiguous_items` — the bucket policy.yaml deliberately gives no coverage
      answer for. Returning either answer would be this generator inventing policy, and
      the whole point of resolving here is that the refusal is enforced by the oracle
      instead of being a convention the builder happens to follow.
    * a kind that belongs to no bucket of this category — a line from some other
      category's vocabulary. The policy simply does not say what such a line is worth
      here, so there is no fraction to compute over it. Distinct from a basket of lines the
      category *does* name and does not cover, which is `rejected`: there the policy has an
      answer for every line and the answer is no, whereas here it has no answer at all.
    """
    spec = category(category_id)
    if item_kind in spec["covered_items"]:
        return True
    if item_kind in spec["excluded_items"]:
        return False
    if item_kind in spec.get("ambiguous_items", {}):
        raise PolicyGapError(
            f"item kind {item_kind!r} is in the `ambiguous_items` bucket of "
            f"{category_id!r}, and policy.yaml states no coverage answer for that bucket. "
            "Resolving it is a policy decision, not an implementation one."
        )
    raise PolicyGapError(
        f"item kind {item_kind!r} appears in no bucket of category {category_id!r}, so "
        "the policy gives no coverage answer for it. A claim whose lines belong to "
        "another category cannot be scored against this one."
    )


def resolved_coverage(category_id: str, items: Sequence[LineItem]) -> list[bool]:
    """The policy's coverage answer for each line, with the line's own flag checked against it.

    `LineItem.covered` is a label somebody else wrote. Here it is an assertion: when it
    disagrees with policy.yaml the engine refuses both readings, because exactly one of
    them is wrong and nothing in the ground truth can say which. Silently preferring the
    policy would hide a builder bug; silently preferring the flag would make this module a
    pass-through.
    """
    answers = []
    for item in items:
        answer = coverage_of_kind(category_id, item.item_kind)
        if item.covered != answer:
            bucket = "covered_items" if answer else "excluded_items"
            raise ValueError(
                f"line {item.name!r} of kind {item.item_kind!r} is labelled "
                f"covered={item.covered}, but policy.yaml puts that kind in the "
                f"{bucket} of category {category_id!r}. One of the two is wrong and the "
                "engine will not choose between them."
            )
        answers.append(answer)
    return answers


def covered_total(category_id: str, items: Sequence[LineItem]) -> Decimal:
    """Σ(qty × price) over the lines this category reimburses, to the kopiyka."""
    answers = resolved_coverage(category_id, items)
    return sum(
        (item.qty * item.price for item, covered in zip(items, answers, strict=True) if covered),
        Decimal(0),
    ).quantize(KOPIYKA)


def covered_fraction(category_id: str, items: Sequence[LineItem]) -> Decimal:
    """Covered amount / total amount.

    Raises when the total is zero. 0/0 is not 0: a claim whose documents carry no line, or
    whose lines are all free, states an amount nothing accounts for, and answering `0`
    would hand `verdict_for` a case it cannot name — turning a claim that should have
    failed loudly here into one that fails obscurely two calls later.
    """
    total = line_items_total(list(items))
    if total <= 0:
        raise ValueError(f"a claim with a total of {total} has no coverage fraction")
    return covered_total(category_id, items) / total


def verdict_for(covered: Decimal, total: Decimal, *, every_line_covered: bool) -> Verdict:
    """The `coverage` rule of policy.yaml, applied to one claim.

    STRICT, as the prose of that block states it: any non-covered line makes the claim
    partially covered, however small, so the branch turns on whether such a line exists
    and not on where the fraction falls. This is deliberately *not* the numeric reading of
    `full_threshold`. A claim may span any number of documents, so a single real
    non-covered line can be diluted below 0.01% of the total by adding documents, and a
    fraction branch would then read `covered` on a claim with a visibly non-reimbursable
    line on it — precisely the outcome the comment above `full_threshold` exists to forbid.

    `full_threshold` keeps the job its own comment gives it: absorbing arithmetic error.
    It is applied here as a self-check on the one case where it can say anything — a claim
    with no non-covered line must come to the full amount — and a failure raises rather
    than being rounded away.

    A covered amount of zero is `rejected` — the third branch the `coverage` block
    declares. It is checked before the strict rule because it is the stronger statement:
    every line of such a claim is non-covered, so `partially_covered` would be true of the
    lines and false about the claim, which qualifies for nothing.

    This is one of the TWO routes to `rejected`, and the only one this function can see. The
    other is a payment outside the benefit period, decided in `evaluate_claim` from dates
    and never from amounts; it carries the cause `outside_period` in `imperfection`, while
    the zero-coverage route here carries none.

    A WARNING FOLLOWS, and it is not part of the rule above. Everything this function needs to
    decide a verdict is stated already: amounts in, `rejected` / `partially_covered` / `covered`
    out. So the paragraph below is optional — a caller that skips it still gets the right answer
    from this docstring — which is the one condition under which a contrast with another verdict
    is allowed at all. See `verdict_notes.definitions_name_their_own_slot` in
    config/labelling-schema.yaml: text a reader must consult in order to classify has to be
    positive; text nobody has to consult may contrast. Being optional is the licence, not being
    positioned last.

    `rejected` is easy to merge with `not_proof_of_payment` by accident, and the two are
    decided from different inputs entirely. This branch sees amounts and nothing else, and
    amounts cannot say whether the evidence proves a payment: that is a property of the document
    type, declared by `proves_payment: false` in `document_evidence`, and it is decided nowhere
    near here. A pharmacy receipt listing nothing but medicines proves its payment perfectly
    well and is still about the wrong subject.
    """
    if total <= 0:
        raise ValueError(f"a claim with a total of {total} has no verdict")
    if covered <= 0:
        return Verdict.REJECTED
    if not every_line_covered:
        return Verdict.PARTIALLY_COVERED

    threshold = full_threshold()
    if covered < threshold * total:
        raise ArithmeticError(
            f"every line of this claim is covered, yet the covered amount {covered} falls "
            f"below full_threshold × total ({threshold} × {total}). policy.yaml keeps that "
            "threshold to absorb arithmetic error; reaching it means the amounts stopped "
            "adding up, which is a bug rather than a verdict."
        )
    return Verdict.COVERED


# =============================================================================
# The ledger
# =============================================================================


class Ledger:
    """Cumulative reimbursed spend per persona per category, in the reporting currency.

    Only what a claim actually reimburses is recorded: the limit caps a payout, so a
    non-covered line on a receipt cannot eat into it. Nothing here is ever iterated —
    it is looked up by key — so it cannot introduce an order-dependent result.
    """

    def __init__(self) -> None:
        self._spent: dict[tuple[str, str], Decimal] = {}

    def spent(self, persona_id: str, category_id: str) -> Decimal:
        return self._spent.get((persona_id, category_id), Decimal("0.00"))

    def remaining(self, persona_id: str, category_id: str) -> Decimal:
        """What the persona may still be reimbursed in this category. Never negative."""
        return max(annual_limit(category_id) - self.spent(persona_id, category_id), Decimal(0))

    def record(self, persona_id: str, category_id: str, amount: Decimal) -> None:
        if amount < 0:
            raise ValueError(f"a reimbursement is not negative: {amount}")
        key = (persona_id, category_id)
        self._spent[key] = self.spent(*key) + Decimal(amount)


# =============================================================================
# Evaluation
# =============================================================================


@dataclass(frozen=True)
class ClaimEvaluation:
    """The claim-level labels, and the money behind them."""

    verdict: Verdict
    # `None` where no line item exists to compute a fraction from — a claim evidenced by a
    # payment and nothing saying what it bought. Distinct from `0`, which says the lines
    # were read and none of them is covered.
    covered_fraction: Decimal | None
    reimbursable: Decimal
    verdict_basis: tuple[VerdictBasis, ...]
    imperfection: tuple[str, ...]
    policy_trace: tuple[str, ...]

    def fraction_as_label(self) -> float | None:
        """`covered_fraction` as it is written to the label file.

        Quantized before it becomes a float, so the JSON carries a readable number rather
        than the nearest binary approximation of a long division. Six places is far below
        anything a verdict turns on — the verdict does not read this number at all.
        """
        if self.covered_fraction is None:
            return None
        return float(self.covered_fraction.quantize(Decimal("0.000001")))


@dataclass(frozen=True)
class ClaimInput:
    """One claim as the engine sees it: who, which category, and which documents."""

    claim_id: str
    persona_id: str
    category: str
    documents: tuple[DocGroundTruth, ...]

    @property
    def dated(self) -> date:
        """The date the claim takes its place in the ledger by: its PROOF OF PAYMENT.

        A limit is consumed when money moves, so the claim is dated by the document that
        says it moved. The earliest document is the invoice — the date of the obligation,
        not of the expense — and dating a claim by it would put a December invoice paid in
        January into the wrong year's balance and order it against other claims by when
        the paperwork started rather than by when the plan paid.

        The earliest payment where a claim carries several, which is the same `min` as
        before over a narrower set of documents. A claim with no proof of payment at all
        reimburses nothing and cannot move the balance whatever its position, so it falls
        back to its earliest document rather than refusing to have a place in the order.
        """
        if not self.documents:
            raise ValueError(f"claim {self.claim_id} has no documents")
        payments = [
            document.date
            for document in self.documents
            if document_evidence(document.doc_type).proves_payment
        ]
        return min(payments) if payments else min(d.date for d in self.documents)


def _percent(fraction: Decimal) -> str:
    """A fraction as a percentage, with just enough places to stay honest.

    One decimal place normally, but never at the cost of printing "100%" for a claim that
    is not fully covered or "0%" for one that covers something — the trace is read as the
    justification of the verdict, so a rounding that contradicts it is worse than a long
    number.
    """
    percentage = fraction * 100
    for places in (1, 2, 4, 6, 9):
        text = f"{percentage:.{places}f}"
        rounded = Decimal(text)
        if (rounded != 100 or percentage == 100) and (rounded != 0 or percentage == 0):
            break
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}%"


def _money(amount: Decimal) -> str:
    return f"{amount:.2f}"


def _coverage_trace(answers: Sequence[bool], fraction: Decimal) -> str:
    not_covered = sum(1 for covered in answers if not covered)
    if not_covered == 0:
        return f"coverage {_percent(fraction)} (all line items covered)"
    return (
        f"coverage {_percent(fraction)} "
        f"({not_covered} of {len(answers)} line items not covered)"
    )


def _evidence_trace(shape: EvidenceShape) -> str:
    """What the claim's documents were resolved into.

    In the trace of every claim that gets past the evidence check, not only of the ones it
    catches. A verdict on a claim spanning several documents rests on which document was
    read as proving what, and a consumer cannot recover that pairing from the label —
    `documents` is a flat list. A self-contained claim says so in one clause rather than
    naming the same document twice.
    """
    parts = []
    for transaction in shape.transactions:
        if transaction.is_split:
            parts.append(
                f"subject {transaction.subject.doc_id} "
                f"({transaction.subject.doc_type.value}), "
                f"payment {transaction.payment.doc_id} "
                f"({transaction.payment.doc_type.value})"
            )
        else:
            parts.append(
                f"{transaction.payment.doc_id} ({transaction.payment.doc_type.value}) "
                "proves both"
            )
    count = len(shape.transactions)
    return f"evidence: {count} transaction{'' if count == 1 else 's'} — {'; '.join(parts)}"


def _disagreements(shape: EvidenceShape) -> list[tuple[str, str]]:
    """Where the two documents of one transaction fail to describe one transaction.

    Returns (cause, trace line) pairs, in a fixed order so that a claim failing both is
    reported the same way every run.

    Only split pairs are checked, and that is not a simplification: a self-contained
    document cannot disagree with itself about which payment settled it. Whether such a
    document's stated amount matches its own line items is a DIFFERENT invariant, owned by
    `content_builder.validate_line_item_sum`, deliberately without a caller on the honest
    path, and belonging to the fraud archetypes that break it on purpose. Checking it here
    would pre-empt the decision about how those archetypes are labelled.
    """
    found: list[tuple[str, str]] = []
    for transaction in shape.transactions:
        if not transaction.is_split:
            continue
        subject, payment = transaction.subject, transaction.payment

        if subject.amount.quantize(KOPIYKA) != payment.amount.quantize(KOPIYKA):
            found.append((
                AMOUNT_MISMATCH,
                f"documents disagree: {subject.doc_id} ({subject.doc_type.value}) states "
                f"{_money(subject.amount)} {reporting_currency()}, payment "
                f"{payment.doc_id} states {_money(payment.amount)} {reporting_currency()}",
            ))
        # Strictly before: paying an invoice on the day it is issued is ordinary. This is
        # deliberately NOT folded into the period check — a payment that precedes what it
        # settles is an impossible order, not a date outside a window, and the two have
        # different repairs.
        if payment.date < subject.date:
            found.append((
                PAYMENT_PRECEDES_SUBJECT,
                f"documents disagree: payment {payment.doc_id} dated {payment.date} "
                f"precedes {subject.doc_id} dated {subject.date}",
            ))
    return found


def _check_currency(documents: Iterable[DocGroundTruth]) -> None:
    """Refuse a document whose amounts are not in the currency the limits are stated in.

    Not a limitation and not a stub. Every archetype in this generator issues documents in
    `reporting_currency`, so a document in anything else means something upstream produced
    a claim that cannot be scored — a jurisdiction wired to the wrong currency, or a
    document attached to the wrong persona. The engine raises and says so.

    It deliberately offers no conversion. A rate applied here would put a number in the
    ground truth that nothing in the dataset can prove, and it would look like a tidy
    piece of engineering while doing it. Whatever currency a claim's amounts are compared
    against a limit in, the dataset has to be able to show where that comparison came
    from.
    """
    currency = reporting_currency()
    for document in documents:
        if document.currency != currency:
            raise ValueError(
                f"document {document.doc_id} is denominated in {document.currency}, "
                f"while annual limits are expressed in {currency} (policy.yaml "
                "`reporting_currency`). Every archetype this generator has emits "
                f"{currency}, so this document should not exist; the engine will not "
                "score it, and will not convert it."
            )


def _reimbursable(
    *, persona_id: str, category_id: str, covered: Decimal, ledger: Ledger
) -> Decimal:
    remaining = ledger.remaining(persona_id, category_id)
    if covered > 0 and remaining <= 0:
        raise PolicyGapError(
            f"persona {persona_id} has no {category_id} balance left "
            f"(limit {annual_limit(category_id)} {reporting_currency()} fully reimbursed), "
            "and policy.yaml assigns no verdict to a claim that covers something and is "
            "still paid nothing. None of the candidates fits: `rejected` is the policy "
            "plainly not covering the claim — by what was bought or by when it was paid — "
            "and here it covers part or all of it, inside the period; "
            "`not_proof_of_payment` is for a claim whose document types all declare "
            "`proves_payment: false`, and a receipt's does not; and `partially_covered` says "
            "some of the amount is payable, which none of it is. Deciding this means changing the "
            "policy, so the planner does not plan such a claim and the engine will not "
            "invent a label for one."
        )
    return min(covered, remaining)


def evaluate_claim(
    *,
    persona_id: str,
    category: str,
    documents: Sequence[DocGroundTruth],
    ledger: Ledger | None = None,
) -> ClaimEvaluation:
    """The verdict, the covered fraction and the justification for one claim.

    `ledger` carries the persona's history; omitting it means no history, i.e. the full
    annual limit is available. The ledger is read, never written — the caller records the
    reimbursement, so that evaluating a claim twice cannot double-count it.
    """
    if not documents:
        raise ValueError("a claim with no documents has nothing to evaluate")

    _check_currency(documents)
    ledger = ledger or Ledger()
    limit = annual_limit(category)  # raises KeyError for a category the policy lacks
    trace = [f"category={category} ok"]

    shape = resolve_evidence(documents)

    # Coverage is resolved from the SUBJECT documents, once per transaction, and never
    # from every document of the claim: an invoice and the payment that settles it
    # describe one purchase, and pooling both would count the money twice. It is computed
    # before any verdict branch, so that `covered_fraction` is the same property of the
    # line items wherever it is reported — including the branches where it decides
    # nothing.
    items = [item for document in shape.subject_documents for item in document.line_items]
    answers = resolved_coverage(category, items)
    total = line_items_total(items)
    covered = sum(
        (item.qty * item.price for item, is_covered in zip(items, answers, strict=True)
         if is_covered),
        Decimal(0),
    ).quantize(KOPIYKA)
    # `None`, not 0: a claim with no lines has no fraction, and 0 would say its lines were
    # read and covered nothing.
    fraction = covered / total if total > 0 else None

    def refused(verdict: Verdict, causes: tuple[str, ...]) -> ClaimEvaluation:
        """A claim that reimburses nothing, with the cause named rather than described.

        Every branch that reaches this has established the verdict from the CONTENT of the
        documents — which types they are, whether they agree, when the payment happened —
        rather than from coverage arithmetic, so the trace carries no coverage line: it
        states what justified the verdict, and coverage did not. The fraction is still
        reported where there were lines to compute one from, as a fact about those lines.

        Three verdicts reach it, not two: `not_proof_of_payment`, `insufficient_evidence`
        and — for a payment outside the benefit period — `rejected`. `verdict_basis` is
        `DOCUMENTS` for all three, because each of the facts above is printed on the image.
        """
        return ClaimEvaluation(
            verdict=verdict,
            covered_fraction=fraction,
            reimbursable=Decimal("0.00"),
            verdict_basis=(VerdictBasis.DOCUMENTS,),
            imperfection=causes,
            policy_trace=tuple(trace),
        )

    # -- both facts, or neither verdict. See `resolve_evidence`.
    if not shape.proves_payment:
        trace.append(f"evidence: no document proves payment — {_types(documents)}")
        return refused(Verdict.NOT_PROOF_OF_PAYMENT, ())
    if not shape.proves_subject:
        trace.append(f"evidence: no document states what was bought — {_types(documents)}")
        return refused(Verdict.INSUFFICIENT_EVIDENCE, (SUBJECT_NOT_EVIDENCED,))
    trace.append(_evidence_trace(shape))

    # -- the documents of one transaction have to describe one transaction
    disagreements = _disagreements(shape)
    if disagreements:
        trace.extend(line for _, line in disagreements)
        return refused(
            Verdict.INSUFFICIENT_EVIDENCE, tuple(cause for cause, _ in disagreements)
        )

    # -- the period, on the payment date and on no other. `rejected`, not
    # `insufficient_evidence`: nothing here is unestablished, the policy simply does not
    # cover a payment made outside its window. See the module docstring, rule 4.
    start, end = active_period()
    late = [t.payment for t in shape.transactions if not start <= t.payment.date <= end]
    if late:
        first = min(late, key=lambda document: document.date)
        trace.append(
            f"period: payment {first.doc_id} dated {first.date} falls outside "
            f"{start}..{end}"
        )
        return refused(Verdict.REJECTED, (OUTSIDE_PERIOD,))
    trace.append("period ok")

    if total <= 0:
        raise ValueError(f"a claim with a total of {total} has no verdict")

    # Before the trace line, so that a claim with no verdict does not get a justification
    # for one.
    verdict = verdict_for(covered, total, every_line_covered=all(answers))
    trace.append(_coverage_trace(answers, covered / total))
    reimbursable = _reimbursable(
        persona_id=persona_id, category_id=category, covered=covered, ledger=ledger
    )

    basis = [VerdictBasis.DOCUMENTS]
    imperfection: list[str] = []
    if verdict is Verdict.PARTIALLY_COVERED:
        imperfection.append(MIXED_ITEMS)

    if reimbursable < covered:
        verdict = Verdict.PARTIALLY_COVERED
        imperfection.append(LIMIT_EXHAUSTED)
        basis.append(VerdictBasis.ACCOUNT_STATE)
        spent = ledger.spent(persona_id, category)
        trace.append(
            f"annual limit {category} {_money(limit)} {reporting_currency()}: "
            f"{_money(spent)} already reimbursed, {_money(limit - spent)} remaining"
        )
        trace.append(
            f"reimbursable {_money(reimbursable)} {reporting_currency()} of "
            f"{_money(covered)} covered — annual limit exhausted"
        )

    return ClaimEvaluation(
        verdict=verdict,
        covered_fraction=fraction,
        reimbursable=reimbursable,
        verdict_basis=tuple(basis),
        imperfection=tuple(imperfection),
        policy_trace=tuple(trace),
    )


def evaluate_claims(
    claims: Sequence[ClaimInput], ledger: Ledger | None = None
) -> list[ClaimEvaluation]:
    """Evaluate a set of claims against one ledger, in date order.

    **This is not the path `assembler` takes, and that is not an oversight.** Cumulative
    limits are what makes it so: claim *k*'s documents are sized against the balance left
    by claims 1…k−1, so the assembler has to interleave planning, building and evaluating
    one claim at a time, and it cannot hand a finished list to a batch function that does
    not exist yet at that point. This entry point serves the other direction — a caller
    that already holds every document and wants the labels re-derived: a consumer checking
    the dataset it was given, or the test that re-derives a whole finished run and
    requires it to agree with what the assembler wrote
    (`test_the_dataset_labels_are_reproducible_from_the_documents_alone`). That test is
    what keeps the two orderings from drifting apart.

    Results come back aligned with the input, so a caller keeps its own ordering; only the
    arithmetic is reordered. Ties are left to the stable sort, which means claims sharing a
    date are resolved in the order they were presented — for a dataset re-derived from
    disk, the order the assembler wrote them in. policy.yaml says nothing about ties, so
    this is the engine's own choice; the alternative, ordering by claim id, was rejected
    because it makes the balance depend on how ids happen to be formatted.
    """
    ledger = ledger or Ledger()
    results: dict[int, ClaimEvaluation] = {}

    for index, claim in sorted(enumerate(claims), key=lambda pair: pair[1].dated):
        evaluation = evaluate_claim(
            persona_id=claim.persona_id,
            category=claim.category,
            documents=claim.documents,
            ledger=ledger,
        )
        ledger.record(claim.persona_id, claim.category, evaluation.reimbursable)
        results[index] = evaluation

    return [results[index] for index in range(len(claims))]
