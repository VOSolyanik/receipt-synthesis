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

Four rules, in the order they are applied:

1. **Currency.** Limits are expressed in `reporting_currency`, and every archetype this
   generator has emits documents in it. A document in any other currency is therefore a
   contradiction upstream, not a case to handle: the engine raises. It does not convert,
   at any rate, from any source — a converted amount would land in the ground truth as a
   number nothing in the dataset can prove.
2. **Coverage of each line**, resolved from the item-kind vocabulary of the claimed
   category. A kind in `ambiguous_items` — for which policy.yaml deliberately states no
   coverage answer — or a kind that belongs to no bucket of this category at all, is
   refused rather than guessed.
3. **Period.** A document dated outside the active window makes the claim
   `insufficient_evidence`, and the claim consumes no balance. policy.yaml does not say
   which check wins when a claim fails both this one and coverage; the period is checked
   first, because a document from outside the window is not evidence about the period at
   all.
4. **The verdict.** STRICT, as the prose of the `coverage` block states it: *any*
   non-covered line makes the claim `partially_covered`, however small — the fraction is
   reported, never used as a tolerance. Everything covered is `covered`, and
   `full_threshold` is kept alive as a **self-check** on that case (see `verdict_for`),
   which is the role its own comment gives it. A claim whose covered amount is 0 is
   `rejected`, which is the third branch that block declares and is about coverage alone —
   not `not_proof_of_payment`, which is a property of the document type and lives in
   `document_evidence`.
5. **The cumulative annual limit.** Claims are processed per persona per category in date
   order, carrying the balance. When what remains is less than what the document covers,
   the claim is `partially_covered`, and the verdict then also depends on the persona's
   history — which is what `verdict_basis` records.

`covered_fraction` is a property of the line items in every case, including the two where
it does not decide the verdict: it is covered amount over total amount and nothing else,
so a claim held back only by an exhausted limit still reports 1.0, and an out-of-period
claim still reports what its lines cover. The money actually payable is a separate field,
`reimbursable_amount`; `policy_trace` carries only what justified the verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from receipt_synth.config import category, load_policy
from receipt_synth.content_builder import KOPIYKA, line_items_total
from receipt_synth.schemas import DocGroundTruth, LineItem, Verdict, VerdictBasis

# Causes of a `partially_covered` verdict. The strings are the keys of
# `partially_covered_causes` in policy.yaml and land verbatim in `imperfection`.
MIXED_ITEMS = "mixed_items"
LIMIT_EXHAUSTED = "limit_exhausted"


class PolicyGapError(Exception):
    """Raised where policy.yaml specifies no answer and guessing one would corrupt the
    ground truth.

    Three cases reach it, all of them narrow and all of them deliberate:
    `coverage_of_kind` for an `ambiguous_items` kind and for a kind foreign to the claimed
    category, and `_reimbursable` for a claim wholly beyond an exhausted annual limit.
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

    `rejected` is emphatically NOT `not_proof_of_payment`, and the two are easy to merge by
    accident. This branch sees amounts and nothing else, and amounts cannot say whether the
    evidence proves a payment: that is a property of the document type, declared by
    `proves_payment: false` in `document_evidence`, and it is decided nowhere near here. A
    pharmacy receipt listing nothing but medicines proves its payment perfectly well and is
    still about the wrong subject.
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
    covered_fraction: Decimal
    reimbursable: Decimal
    verdict_basis: tuple[VerdictBasis, ...]
    imperfection: tuple[str, ...]
    policy_trace: tuple[str, ...]

    def fraction_as_label(self) -> float:
        """`covered_fraction` as it is written to the label file.

        Quantized before it becomes a float, so the JSON carries a readable number rather
        than the nearest binary approximation of a long division. Six places is far below
        anything a verdict turns on — the verdict does not read this number at all.
        """
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
        """The date the claim takes its place in the ledger by.

        The earliest of its documents: a claim may span several, and the balance moves
        when the money did, not when the last piece of paper arrived.
        """
        if not self.documents:
            raise ValueError(f"claim {self.claim_id} has no documents")
        return min(document.date for document in self.documents)


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
            "still paid nothing. None of the candidates fits: `rejected` is the policy not "
            "covering the purchase, and here it covers part or all of it; "
            "`not_proof_of_payment` says the evidence does not establish that money "
            "changed hands, which a receipt does; and `partially_covered` says some of the "
            "amount is payable, which none of it is. Deciding this means changing the "
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

    # Resolved before the period is looked at, so that `covered_fraction` is the same
    # property of the line items on every branch — including the one where it does not
    # decide anything.
    items = [item for document in documents for item in document.line_items]
    answers = resolved_coverage(category, items)
    total = line_items_total(items)
    if total <= 0:
        raise ValueError(f"a claim with a total of {total} has no verdict")
    covered = sum(
        (item.qty * item.price for item, is_covered in zip(items, answers, strict=True)
         if is_covered),
        Decimal(0),
    ).quantize(KOPIYKA)
    fraction = covered / total

    start, end = active_period()
    outside = [document for document in documents if not start <= document.date <= end]
    if outside:
        first = min(outside, key=lambda document: document.date)
        trace.append(
            f"period: document {first.doc_id} dated {first.date} falls outside "
            f"{start}..{end}"
        )
        # No coverage line: the trace states what justified the verdict, and coverage did
        # not. The fraction is still reported, as a fact about the lines.
        return ClaimEvaluation(
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            covered_fraction=fraction,
            reimbursable=Decimal("0.00"),
            verdict_basis=(VerdictBasis.DOCUMENTS,),
            imperfection=(),
            policy_trace=tuple(trace),
        )
    trace.append("period ok")

    # Before the trace line, so that a claim with no verdict does not get a justification
    # for one.
    verdict = verdict_for(covered, total, every_line_covered=all(answers))
    trace.append(_coverage_trace(answers, fraction))
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
