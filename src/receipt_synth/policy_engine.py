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

Six rules, in the order they are applied — seven branches, rule 4a having been inserted
BETWEEN two of them rather than appended, because where it sits is what it says. The numbering
is not renumbered for it: a dozen cross-references in this file and in
config/labelling-schema.yaml point at these numbers, and shifting them all to place one branch
would make every one of those references silently wrong in the git history.

1. **Currency.** Limits are expressed in `reporting_currency`. A document in another
   currency is CONVERTED — at the rate config/fx-rates.yaml states for the document's
   date, quantized at the point that file declares, with the applied rate recorded in
   the claim's label beside the original amount, currency and date. An earlier revision
   refused instead, on the ground that "a converted amount would land in the ground
   truth as a number nothing in the dataset can prove" — which is true of a rate taken
   from nowhere and stops being true here: the rate is a vendored, versioned constant
   both sides load, and a label that carries it proves the conversion completely. What
   the engine still refuses is a currency the table has no rate for, and a table whose
   seeded jitter is enabled — the engine draws no randomness, so a jittered rate would
   be one it cannot reproduce. See `fx_rate` and `_applied_conversions`.

   Two constraints hold the conversion honest, and both live in the config rather than
   here. The QUANTIZATION POINT is declared in fx-rates.yaml (`conversion`) and this
   engine checks the declaration against its own arithmetic on every conversion — a
   consumer implements the same declared point independently, so a symmetry check
   between the two engines measures the policy and not a rounding convention. And the
   documents of ONE CLAIM must share one currency: coverage pools line items across a
   claim's subject documents and the cross-document axes compare amounts between its
   documents, and policy.yaml states no rule for doing either across two currencies —
   such a claim is refused, not guessed at.
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
   have to describe the same purchase. ON WHICH AXES THEY ARE COMPARED IS READ FROM
   policy.yaml — `cross_document_agreement` declares them, and today they are the same amount,
   the payment not before the subject, and the same counterparty. Failing one →
   `insufficient_evidence`, with the cause named in `imperfection`; the claim does not
   establish that *this* payment paid for *this* subject. The VERDICT is declared per axis
   there as well, so the outcome of a disagreement is a policy parameter rather than a
   constant of this module — see `cross_document_agreement` and `_AXIS_DISAGREEMENTS`, which
   is the whole of the split: the file decides which fields are compared and what a
   disagreement costs, this module decides what comparing them means.

   🔴 THE SAME AMOUNT, OR ONE PART OF IT WHERE THE SUBJECT SAYS SO — and this is an EXEMPTION
   from the amount check rather than a second failure of the slot. A subject document may
   state that its obligation is settled in equal parts and what one part comes to — an annual
   subscription billed for the year and paid quarterly — and a payment equal to that part
   then describes the same transaction as the document beside it, so the linkage holds and
   this rule finds nothing. The marker has to be PRINTED on the subject document
   (`instalment_amount`); the arithmetic alone cannot tell the two apart, because "the
   payment is smaller" is true of a mismatch as well, and a rule reading only the amounts
   would swallow that cause whole. See `_settles_one_instalment` and `partial_payment` in
   policy.yaml, which states the rule for a consumer building its own engine from that file.

   WHAT SUCH A CLAIM IS LABELLED is decided further down, at rule 4a, and NOT here: the
   exemption and the label are separate steps on purpose, so that the period can come between
   them.

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
4a. **A transaction exempted by rule 3** — a payment equal to an instalment its subject
   document prints — is `partially_paid`, with no cause: one mechanism, one way to reach it.

   NUMBERED 4a BECAUSE THE POSITION IS THE RULE. It sits AFTER the period deliberately, so a
   partial settlement paid outside the window is `rejected` and not this. That precedence is
   policy.yaml's — `partial_payment.outside_the_period` — and it is stated there rather than
   left to the order these branches happen to be written in, because a consumer builds its own
   engine from that file and two faithful engines disagreeing about a LABEL is the most
   expensive defect this repository has. The reasoning, in one line: the period asks whether
   the plan covers this expense at all, and `partially_paid` is a statement about a claim the
   plan does cover.

   ⚠️ AND IT COULD NOT HAVE BEEN FOLDED INTO RULE 3. The exemption there decides whether the
   documents AGREE; this decides what an agreeing pair is called. Keeping them one step would
   have put the label before the period with no way to say why.
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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

from receipt_synth.config import category, load_fx_rates, load_policy
from receipt_synth.content_builder import (
    KOPIYKA,
    line_items_total,
    proves_payment_by_direction,
)
from receipt_synth.schemas import DocGroundTruth, DocType, LineItem, Verdict, VerdictBasis

# Causes of a `partially_covered` verdict. The strings are the keys of
# `partially_covered_causes` in policy.yaml and land verbatim in `imperfection`.
MIXED_ITEMS = "mixed_items"
LIMIT_EXHAUSTED = "limit_exhausted"

# The five causes below are recorded in `imperfection` beside the two above, under the
# verdicts that are NOT decided by coverage arithmetic — four under `insufficient_evidence`
# and one under `rejected`. One verdict, several named causes — the shape
# `partially_covered` already has, and for the same reason: a consumer can know different
# things about each. A document that is simply absent and two documents that contradict
# each other are not the same problem.
#
# POLICY.YAML NAMES FOUR OF THE FIVE AND SIZES NOTHING BY THE FIFTH. Its cause vocabularies
# exist to declare SHARES — `partially_covered_causes`, and `insufficient_evidence_causes` since
# the planner learned to build all four of those — so a cause appears there exactly when a
# bucket of the dataset is sized by it. `OUTSIDE_PERIOD` is not: `rejected` has two routes, one
# of them buildable, and a share over a single route would be the number 1.0 written down.
# THE NAMES ARE STILL THIS MODULE'S EITHER WAY, and a share is not a vocabulary: this engine
# derives a cause from a claim's documents without consulting one, so a cause whose share was
# removed would still be returned. They are exported for anyone comparing labels against a
# vocabulary and recorded in config/labelling-schema.yaml, which is the contract a consumer
# reads.

# Causes of `insufficient_evidence`, which owns two of the three slots of `document_evidence`
# (see the module docstring, rules 2 and 3): the WHAT-WAS-BOUGHT slot, unestablished when no
# document is of a type that states it, and the LINKAGE slot, unestablished when a subject
# document and its payment both exist and fail a cross-check. Each is read off the claim's own
# documents.
#
# ⚠️ THE LINKAGE CAUSES ARE NAMED HERE AND CHOSEN IN policy.yaml. Each is the `cause` of one axis
# of `cross_document_agreement`, so the file decides which of them an engine can return at all —
# an axis withdrawn there is a cause no claim receives. The names stay this module's constants
# because this module is what writes them into a label.
SUBJECT_NOT_EVIDENCED = "subject_not_evidenced"
AMOUNT_MISMATCH = "amount_mismatch"
PAYMENT_PRECEDES_SUBJECT = "payment_precedes_subject"
# 🔴 THE THIRD CAUSE OF THE LINKAGE SLOT, and the first that is not about a number: the invoice was
# issued by one party and the payment went to another. Nothing is missing from such a claim and
# nothing about it is arithmetically wrong — which is why it belongs to the LINKAGE slot and not to
# either of the others. See `cross_document_agreement` in policy.yaml, which declares the axis, and
# the structural table in config/labelling-schema.yaml, which is where the vocabulary is contracted.
COUNTERPARTY_MISMATCH = "counterparty_mismatch"
# 🔴 THE FOURTH CAUSE OF THE LINKAGE SLOT, and the axis that closes the transaction's last
# dimension: the amount says HOW MUCH, the counterparty says TO WHOM, the order says WHEN — and
# this one says FOR WHAT. A payment document never lists what was bought (👁 0 of 7 observed
# purposes name it, which is `proves_subject: false`); its one statement about the subject is the
# рахунок its purpose cites by number, so the axis compares that citation against the subject
# document's own № — `cites_document_no` against `document_code` — and only where a citation is
# printed at all. A payment quoting the right amount to the right party in the right order, for a
# DIFFERENT invoice, is what this catches; no other axis can see it.
SUBJECT_MISMATCH = "subject_mismatch"

# The one cause of `rejected`, and the only one it needs. `rejected` has two mechanisms —
# a basket the category covers none of, and a payment outside the benefit period — and only
# the second is worth naming: the first is the whole of what the verdict already says,
# while the second says the claim is uncovered by WHEN rather than by WHAT. So a `rejected`
# claim carries either this cause or none, which is how the two mechanisms are told apart
# without parsing `policy_trace`. See the module docstring, rules 4 and 5, for why this case
# is not `insufficient_evidence`.
OUTSIDE_PERIOD = "outside_period"

# 🔴 THE DOCUMENT FIELDS WHOSE VALUE MAKES A SMALLER PAYMENT A PART RATHER THAN A DISAGREEMENT —
# the marker of `partially_paid`, declared in policy.yaml under `partial_payment.marker_fields`
# and repeated here because THIS module reads the attribute by name. Two places holding one fact
# is how they come apart, so `_settles_one_instalment` compares them on every call rather than
# trusting that they still agree: a consumer builds its own engine from the file, and an engine
# reading a field the file does not name would be labelling by a rule nobody can reproduce.
PARTIAL_PAYMENT_MARKER_FIELDS: tuple[str, ...] = ("instalment_amount",)


class PolicyGapError(Exception):
    """Raised where policy.yaml specifies no answer and guessing one would corrupt the
    ground truth.

    Five cases reach it, all of them narrow and all of them deliberate:
    `coverage_of_kind` for an `ambiguous_items` kind and for a kind foreign to the claimed
    category, `_reimbursable` for a claim wholly beyond an exhausted annual limit, and
    `resolve_evidence` for a claim whose evidence is a CREDIT — money arriving, which proves no
    expense — and for a set of documents that does not pair into transactions.

    The credit case replaced a blanket refusal of any claim carrying a bank statement, which was
    right while a statement's label described the whole document and could not point at the row a
    claim was about. It now points at one row, so the class needs no refusal of its own; what
    survives is the narrower rule that money must have left the account. Recorded because a reader
    of the git history will find the old refusal and should not have to guess why it went.
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


# What this engine's conversion arithmetic does, mirrored against the declaration in
# config/fx-rates.yaml on every conversion — the `PARTIAL_PAYMENT_MARKER_FIELDS` pattern.
# Two places holding one fact is how they come apart, so `_fx_quantization_declared` compares
# them at the point of use rather than trusting that they still agree: a consumer implements
# the declared point independently, and an engine quantizing at an undeclared one would be
# converting by a rule nobody can reproduce.
_FX_STEP = Decimal("0.01")
_FX_ROUNDING = "half-up"
_FX_ORDER = "sum-then-convert-then-quantize"


def _fx_quantization_declared() -> None:
    """Refuse to convert if fx-rates.yaml declares an arithmetic this engine does not do."""
    declared = load_fx_rates()["conversion"]
    stated = (str(declared["step"]), str(declared["rounding"]), str(declared["order"]))
    performed = (str(_FX_STEP), _FX_ROUNDING, _FX_ORDER)
    if stated != performed:
        raise PolicyGapError(
            f"config/fx-rates.yaml declares the conversion as {stated} while this engine "
            f"performs {performed}. A consumer implements the declared point independently, "
            "so the two would produce converted amounts a rounding unit apart while each was "
            "faithful to what it read. Move both or neither."
        )


def fx_rate(currency: str, *, on: date) -> Decimal:
    """The reporting-currency price of one unit of `currency` on the given date.

    Read from config/fx-rates.yaml — a vendored, versioned CONSTANT, not a live source,
    and that is the whole reason a conversion built on it is provable: the label carries
    the applied rate, the file carries where it came from, and a consumer loading the same
    file reproduces the same number exactly.

    🔴 THE DATE IS PART OF THE CONTRACT, NOT OF TODAY'S TABLE. The rule is "the rate on the
    transaction date"; the vendored table happens to be date-invariant, so every date maps
    to the same figure today. The parameter is required anyway, because the signature is
    what a consumer implements against a table that DOES vary — a live product converts at
    the transaction date, and an engine written against a dateless lookup would silently
    take today's rate instead. The gap between this static table and a live source is a
    declared external-validity limit of the corpus, recorded in
    config/labelling-schema.yaml, not a detail to discover.

    Two refusals rather than answers:

    * a currency the table has no rate for — converting at a guessed rate would put a
      number in the ground truth nothing can prove, which is the exact defect the table
      exists to rule out;
    * a table whose seeded `jitter` is enabled — a jittered rate is a function of the run
      seed, and this engine receives no seed: its answer is a pure function of
      (documents, policy). See the ⛔ note in fx-rates.yaml.
    """
    del on  # date-invariant today — see the docstring; the parameter is the contract
    fx = load_fx_rates()
    if str(fx["base"]) != reporting_currency():
        raise PolicyGapError(
            f"config/fx-rates.yaml expresses rates in {fx['base']!r} while policy.yaml "
            f"expresses limits in {reporting_currency()!r}. The two files disagree about "
            "the reporting currency, so no conversion built on them can be right."
        )
    if fx["jitter"]["enabled"]:
        raise PolicyGapError(
            "config/fx-rates.yaml has `jitter.enabled: true`, and this engine draws no "
            "randomness — a jittered rate would be one the label cannot prove. Disable the "
            "jitter, or first teach the pipeline to carry the drawn rate into the label."
        )
    rates = fx["rates"]
    if currency not in rates:
        raise PolicyGapError(
            f"config/fx-rates.yaml states no rate for {currency!r}, so an amount in it "
            "cannot be expressed in the reporting currency. Converting at a guessed rate "
            "would put a number in the ground truth that nothing in the dataset can prove "
            "— add the rate to the vendored table instead."
        )
    return Decimal(str(rates[currency]))


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


def insufficient_evidence_causes() -> dict[str, float]:
    """How the `insufficient_evidence` bucket splits by cause, in declaration order.

    🔴 OVER THE CAUSES THE GENERATOR CAN BUILD, WHICH IS NOT THE SAME QUESTION AS THE VOCABULARY
    EVEN WHERE THE TWO COINCIDE. Five causes lead to this verdict — `SUBJECT_NOT_EVIDENCED`,
    `AMOUNT_MISMATCH`, `PAYMENT_PRECEDES_SUBJECT`, `COUNTERPARTY_MISMATCH`, `SUBJECT_MISMATCH` —
    and policy.yaml now declares a share for each: the first gained a mechanism
    (`claim_planner.EvidenceIntent.EVIDENCE_GAP`), the fourth a second party on the payment
    (`assembler._payee_the_payment_names`), and the fifth a wrong reference on its purpose line
    (`assembler._reference_the_payment_cites`), rather than an exemption.

    A reader of this map must still not take it for the cause vocabulary. This engine derives a
    cause from a claim's documents and does not consult these shares at all, so it would go on
    returning a cause whose share was removed, and a cause it can return is a label a consumer can
    receive. The vocabulary is contracted in config/labelling-schema.yaml; what the map describes
    is the DRAW — the same distinction `verdict_mix` and `REALIZABLE_VERDICTS` have always had
    between them.
    """
    return {str(name): float(share) for name, share in
            load_policy()["insufficient_evidence_causes"].items()}


def rejected_routes() -> dict[str, float]:
    """How the `rejected` bucket splits by route, in declaration order.

    A ROUTE MAP AND NOT A CAUSE MAP, which is why its keys are not all causes this engine can
    emit: `outside_period` is the cause of one route, while `zero_coverage` names a route whose
    claims carry NO cause at all — the verdict says the whole of it (`verdict_for`). This engine
    derives a `rejected` verdict from dates and line items and never consults these shares; what
    the map describes is the DRAW, exactly as `insufficient_evidence_causes` above.
    """
    return {str(name): float(share) for name, share in
            load_policy()["rejected_routes"].items()}


def partial_payment_outside_the_period() -> Verdict:
    """Which verdict a partial settlement paid outside the benefit period gets.

    A PRECEDENCE, not a preference, and it is read from policy.yaml for the same reason the
    marker is: a downstream consumer builds its own engine from that file, and the order of two
    branches is exactly the kind of thing two faithful implementations decide differently. The
    file's own reasoning is beside the key.

    Returned as a `Verdict` so a value the enum does not name fails here — where the file is
    being read — rather than three branches later as a label nothing recognizes.
    """
    return Verdict(load_policy()["partial_payment"]["outside_the_period"])


def partial_payment_marker_fields() -> tuple[str, ...]:
    """Which document field(s) policy.yaml says carry the partial-payment marker.

    Read rather than assumed for the reason every other rule in this module is read from that
    file: a downstream consumer builds its own engine from it, and the two engines have to be
    labelling on the same marker or they will disagree about a VERDICT while both are correct
    about what they read.
    """
    return tuple(str(name) for name in load_policy()["partial_payment"]["marker_fields"])


def insufficient_evidence_causes_min_run_size() -> int:
    """The run size at which `insufficient_evidence_causes` binds "every cause non-zero".

    Read from policy.yaml rather than hardcoded so the derivation comment beside the number
    stays the single place it is justified — the balance report cites this value, it does
    not compute or restate it.
    """
    return int(load_policy()["insufficient_evidence_causes_min_run_size"])


class AgreementAxis(NamedTuple):
    """One field the documents of a split pair must agree on, and what it costs them not to.

    `axis` names a COMPARISON this module implements — `_AXIS_DISAGREEMENTS` is the table of
    them — while `verdict` and `cause` are the OUTCOME policy.yaml assigns to failing it. The
    split is the whole point: the comparison is code, because comparing two dates is not a policy
    question; which label the failure earns is policy, because two engines that compared the same
    fields and answered different verdicts would disagree about a LABEL while both were right
    about what they read.
    """

    axis: str
    verdict: Verdict
    cause: str


def cross_document_agreement() -> tuple[AgreementAxis, ...]:
    """The axes a claim's documents must agree on, in the order policy.yaml declares them.

    🔴 THE LIST IS THE FILE'S AND THE COMPARISONS ARE THIS MODULE'S, which is what makes an axis a
    policy parameter rather than an `if` somebody wrote. Adding a field two documents of one claim
    must agree on is an edit to `cross_document_agreement` in policy.yaml; removing one there stops
    this engine from checking it, with no code change on either side. That is the property the
    block exists for, and it is asserted in both directions in tests/test_claim_evidence.py.

    TWO GUARDS, AND THEY ARE NOT SYMMETRIC — deliberately, because the two asymmetries mean
    different things:

    * an axis DECLARED here that this engine cannot compare is a `PolicyGapError`. The file would
      be promising a check nothing performs, and the claims it should have caught would come back
      labelled as if their documents agreed — a silently wrong ground truth, which is the one
      failure mode this module exists to prevent;
    * an axis this engine COULD compare and the file does not declare is simply not checked. That
      is not drift: a consumer building its own engine from the same file also does not check it,
      so the two agree about every label. What the file omits, nobody compares.

    A DUPLICATE AXIS IS ALSO A GAP. Declaring one twice would put its cause into `imperfection`
    twice, or — worse, if the two entries named different verdicts — make the label depend on
    which of the two the engine happened to read first.

    Returned with `verdict` as a `Verdict`, so a label the enum does not name fails here, where
    the file is being read, rather than several branches later as a value nothing recognizes.
    """
    declared: list[AgreementAxis] = []
    for entry in load_policy()["cross_document_agreement"]:
        axis = str(entry["axis"])
        if axis not in _AXIS_DISAGREEMENTS:
            raise PolicyGapError(
                f"policy.yaml declares that the documents of a claim must agree on {axis!r} "
                f"(`cross_document_agreement`), and this engine compares "
                f"{sorted(_AXIS_DISAGREEMENTS)}. A declared axis nothing performs would let every "
                "claim that fails it be labelled as though its documents agreed — implement the "
                "comparison or withdraw the axis."
            )
        if any(axis == already.axis for already in declared):
            raise PolicyGapError(
                f"policy.yaml declares the axis {axis!r} twice in `cross_document_agreement`. "
                "A claim failing it would carry its cause twice, and two entries naming different "
                "verdicts would make the label depend on the order they are read in."
            )
        declared.append(
            AgreementAxis(axis, Verdict(entry["verdict"]), str(entry["cause"]))
        )
    return tuple(declared)


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

    A BANK STATEMENT USED TO BE REFUSED OUTRIGHT HERE, on the ground that it lists several
    transactions while its label carried a single amount for the whole document, so nothing
    identified the row a claim was about. THAT REASON IS GONE: a statement's label now describes
    ONE TRANSACTION — the row's amount, date, counterparty, purpose and direction, with
    `relevant_transaction` naming the row and `field_bboxes` pointing at its cells — so the claim's
    money is exactly as well identified as it is on a confirmation. A statement is now an ordinary
    payment-proving document to this function.

    WHAT REPLACED IT IS NARROWER AND IS A DIFFERENT QUESTION. Only a DEBIT can be proof of
    payment; a credit is money arriving — a refund, a reversal — and evidences no expense. Such a
    claim is refused rather than labelled, because policy.yaml assigns no verdict to a claim whose
    proof of payment is a refund, and inventing one here would be the engine deciding policy. The
    generator cannot build one: `content_builder.BankStatement` refuses to label a credit row. The
    guard is here for the case that does not go through that builder — a trap archetype, or a
    hand-built record — and it is what makes the invariant enforced rather than described.
    """
    for document in documents:
        if not proves_payment_by_direction(document.direction):
            raise PolicyGapError(
                f"document {document.doc_id} states a credit transaction and is being offered "
                "as evidence. Only a debit can be proof of "
                "payment: money arriving is a refund or a reversal and evidences no expense, so "
                "the claim's proof of payment would be proof that the money came back. "
                "policy.yaml assigns no verdict to such a claim and the engine will not invent "
                "one — see `content_builder.proves_payment_by_direction`."
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
    # One entry per document not stated in the reporting currency — the rate the oracle
    # read from config/fx-rates.yaml at that document's date, whatever branch the verdict
    # took. Empty for an all-reporting-currency claim: nothing was converted, and the
    # absence says so. See `_applied_conversions`.
    fx: tuple[AppliedFxRate, ...] = ()

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
    else:
        # The loop above is bounded, not unbounded: it never resolves a fraction whose
        # distance from 0% or 100% needs more than 9 decimal places to show. No basket this
        # generator draws carries that many line items, so `fraction` never has that much
        # precision — but an oracle that silently printed "100%" for a claim it knows is not
        # fully covered would be exactly the bug this function exists to avoid. Fail loudly
        # instead of falling through with the last (wrong) `text` from the loop above.
        raise AssertionError(
            f"coverage fraction {fraction} needs more than 9 decimal places to be told "
            "apart from 0% or 100% — outside what this generator's baskets can produce"
        )
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


def _settles_one_instalment(transaction: Transaction) -> bool:
    """Whether this payment settles ONE PART of an obligation its subject document says is
    settled in parts.

    🔴 THE DISCRIMINATOR BETWEEN `partially_paid` AND `amount_mismatch`, and the whole of it. Both
    are a payment stating less than the subject beside it; what tells them apart is whether the
    SUBJECT DOCUMENT SAYS SO — see `partial_payment` in policy.yaml, which states the rule for a
    consumer building its own engine.

    THREE CONDITIONS, AND EACH ONE IS LOAD-BEARING:

    * the subject document PRINTS an instalment amount. Without it there is no intent on the page
      and a smaller payment is a smaller payment;
    * the payment EQUALS that instalment. A document stating its parts beside a payment matching
      none of them is a payment for some third amount, which is the mismatch case again — the
      marker has to agree with the payment, not merely be present;
    * the instalment is SMALLER than the whole. A "part" equal to the total settles the whole
      obligation and is an ordinary claim; the builder cannot produce one (every schedule divides
      into at least two parts) and the engine does not rely on it not doing so.

    ⛔ NOT CHECKED ON A SELF-CONTAINED DOCUMENT. A receipt is its own payment, so there is no
    second document for it to settle part of, and its `instalment_amount` is `None` on every class
    but the invoice anyway. The same shape as `_disagreements`, which is deliberate: both ask what
    the two documents of one transaction say about each other.
    """
    if not transaction.is_split:
        return False
    declared = partial_payment_marker_fields()
    if declared != PARTIAL_PAYMENT_MARKER_FIELDS:
        raise PolicyGapError(
            f"policy.yaml declares the partial-payment marker as {list(declared)} while this "
            f"engine reads {list(PARTIAL_PAYMENT_MARKER_FIELDS)}. A consumer builds its own "
            "engine from that file, so the two would label the same claim differently while each "
            "was correct about the field it read. Move both or neither."
        )
    instalment = transaction.subject.instalment_amount
    if instalment is None:
        return False
    instalment = instalment.quantize(KOPIYKA)
    return (
        transaction.payment.amount.quantize(KOPIYKA) == instalment
        and instalment < transaction.subject.amount.quantize(KOPIYKA)
    )


def _amount_disagreement(transaction: Transaction) -> str | None:
    """The `amount` axis: the two documents state the same money, to the kopiyka.

    🔴 A PAIR THAT SETTLES ONE INSTALMENT IS EXEMPT, and the exemption is HERE rather than a
    finding discarded afterwards — see `_settles_one_instalment`, which is the whole of the
    discrimination between `partially_paid` and this cause. Nothing else about such a pair is
    exempt: a payment dated before the invoice it settles is an impossible order whether it pays a
    part or the whole, so the `date_order` axis still runs and still wins.
    """
    if _settles_one_instalment(transaction):
        return None
    subject, payment = transaction.subject, transaction.payment
    if subject.amount.quantize(KOPIYKA) == payment.amount.quantize(KOPIYKA):
        return None
    # Each amount is printed with ITS OWN document's currency. The two are the same
    # currency — `_one_claim_one_currency` refused the claim otherwise, which is what
    # licenses the raw comparison above — but the trace line must not assert the
    # reporting currency beside a number that is stated in another one.
    return (
        f"documents disagree: {subject.doc_id} ({subject.doc_type.value}) states "
        f"{_money(subject.amount)} {subject.currency}, payment "
        f"{payment.doc_id} states {_money(payment.amount)} {payment.currency}"
    )


def _date_order_disagreement(transaction: Transaction) -> str | None:
    """The `date_order` axis: the payment is not dated before what it settles.

    Strictly before: paying an invoice on the day it is issued is ordinary. This is deliberately
    NOT folded into the period check — a payment that precedes what it settles is an impossible
    order, not a date outside a window, and the two have different repairs and different verdicts.
    """
    subject, payment = transaction.subject, transaction.payment
    if payment.date >= subject.date:
        return None
    return (
        f"documents disagree: payment {payment.doc_id} dated {payment.date} "
        f"precedes {subject.doc_id} dated {subject.date}"
    )


def _counterparty_disagreement(transaction: Transaction) -> str | None:
    """The `counterparty` axis: the money went to the party that issued the obligation.

    🔴 THE AXIS THAT NEEDS NEITHER DOCUMENT TO BE WRONG. Both pages may be flawless, the amounts
    may agree to the kopiyka and the dates may be in order — and if the invoice was issued by one
    party and the payment made to another, nothing establishes that this money settled that
    obligation. It is the LINKAGE slot exactly as the amount is, on the other of the two things a
    transaction is: who, rather than how much.

    COMPARED RAW, AND THAT IS NOT A SHORTCUT. `counterparty` carries the BARE trading name on every
    class of this dataset — config/labelling-schema.yaml makes that form authoritative under
    `normalization.party_name` — so both sides are already in the one form the contract compares in,
    and a normalization applied here would be a second implementation of that rule, drifting from it
    the first time either changed. ⚠️ The consequence is that two spellings of one merchant would
    read as two merchants; nothing in this generator produces such a pair, because a claim's parties
    come from config/vendors.json by name.
    """
    subject, payment = transaction.subject, transaction.payment
    if subject.counterparty == payment.counterparty:
        return None
    return (
        f"documents disagree: {subject.doc_id} ({subject.doc_type.value}) names "
        f"{subject.counterparty!r}, payment {payment.doc_id} names {payment.counterparty!r}"
    )


def _subject_disagreement(transaction: Transaction) -> str | None:
    """The `subject` axis: where the payment names the document it settles, it names this one.

    🔴 THE AXIS THAT RUNS ONLY WHERE THERE IS SOMETHING TO READ. A payment document states no
    basket (`proves_subject: false`), so the comparison is NOT between two subjects — it is
    between the subject document's own printed № (`document_code`) and the рахунок the payment's
    purpose cites by number (`cites_document_no`). Both sides `None`-guard: a purpose citing a ВН,
    a generic formula, an unprinted purpose line, and a subject class with no printed № all leave
    nothing to compare, and NOTHING here treats absence as disagreement — an uncited payment
    establishes no subject, which `subject_not_evidenced` and this axis divide between them
    exactly as policy.yaml's preamble states.

    What survives every guard is the one defect no other axis can see: amounts equal to the
    kopiyka, dates in order, one party on both pages — and the payment declares, on its own face,
    that it settles a different purchase.
    """
    subject, payment = transaction.subject, transaction.payment
    cited = payment.cites_document_no
    if cited is None or subject.document_code is None:
        return None
    if cited == subject.document_code:
        return None
    return (
        f"documents disagree: payment {payment.doc_id} settles document no. {cited}, "
        f"{subject.doc_id} ({subject.doc_type.value}) is no. {subject.document_code}"
    )


# WHAT THIS ENGINE CAN COMPARE, keyed by the axis name policy.yaml declares. The FILE decides which
# of these are applied and in which order (`cross_document_agreement`); this table decides only what
# each one means. A key here that the file does not name is simply not checked — see
# `cross_document_agreement`, which is where the asymmetry between the two directions is reasoned
# out — and a name the file declares that is missing here is a `PolicyGapError` raised at the read.
_AXIS_DISAGREEMENTS: dict[str, Callable[[Transaction], str | None]] = {
    "amount": _amount_disagreement,
    "date_order": _date_order_disagreement,
    "counterparty": _counterparty_disagreement,
    "subject": _subject_disagreement,
}


def _disagreements(shape: EvidenceShape) -> list[tuple[Verdict, str, str]]:
    """Where the two documents of one transaction fail to describe one transaction.

    Returns (verdict, cause, trace line) triples, one per axis a transaction fails, in the order
    policy.yaml declares the axes in — so a claim failing two of them is reported the same way
    every run, and the report follows the file rather than the order the branches happen to be
    written in.

    🔴 THE AXES ARE READ, NOT LISTED. Which fields must agree, and what it costs them not to, is
    `cross_document_agreement` in policy.yaml; this function applies what it finds there. A
    consumer builds its own engine from that same block, which is the point of the block.

    Only split pairs are checked, and that is not a simplification: a self-contained
    document cannot disagree with itself about which payment settled it. Whether such a
    document's stated amount matches its own line items is a DIFFERENT invariant, owned by
    `content_builder.validate_line_item_sum`, deliberately without a caller on the honest
    path, and belonging to the fraud archetypes that break it on purpose. Checking it here
    would pre-empt the decision about how those archetypes are labelled.
    """
    found: list[tuple[Verdict, str, str]] = []
    for declared in cross_document_agreement():
        disagrees = _AXIS_DISAGREEMENTS[declared.axis]
        for transaction in shape.transactions:
            if not transaction.is_split:
                continue
            line = disagrees(transaction)
            if line is not None:
                found.append((declared.verdict, declared.cause, line))
    return found


class AppliedFxRate(NamedTuple):
    """One conversion the oracle applied, as it lands in the claim's label.

    The rate itself, not only its result — beside the original amount, currency and date
    the document's own record already carries, this is what turns a converted figure from
    "asserted" into "derived from a stated input": amount × rate, quantized at the point
    fx-rates.yaml declares, is reproducible by anyone holding the same vendored table.
    `on` repeats the document's date so the record stands alone — it is the date the rate
    was taken at, and the lookup key a consumer uses against a table that varies by date.
    """

    doc_id: str
    currency: str
    rate: Decimal
    on: date


def _applied_conversions(
    documents: Iterable[DocGroundTruth],
) -> tuple[AppliedFxRate, ...]:
    """The rate for every document not stated in the reporting currency.

    This replaces a refusal. The engine used to raise here, on the ground that every
    archetype emitted `reporting_currency` and a converted amount would be a number
    nothing in the dataset can prove. Both halves of that ground are gone: an archetype
    now legitimately emits EUR, and the applied rate is recorded in the label, so the
    conversion is proven by the vendored table rather than taken on trust.

    One rate per foreign document, at the DOCUMENT'S OWN date, whatever branch the claim
    later takes — a rejected EUR claim still records the rate, because a consumer
    re-deriving any of its amounts needs the same constant the oracle read. Documents
    already in the reporting currency get no entry: nothing was converted, and an identity
    rate on every UAH document would be noise dressed as information.
    """
    applied = []
    for document in documents:
        if document.currency != reporting_currency():
            applied.append(
                AppliedFxRate(
                    doc_id=document.doc_id,
                    currency=document.currency,
                    rate=fx_rate(document.currency, on=document.date),
                    on=document.date,
                )
            )
    if applied:
        _fx_quantization_declared()
    return tuple(applied)


def _in_reporting(amount: Decimal, *, rate: Decimal) -> Decimal:
    """One amount expressed in the reporting currency, at the declared quantization point.

    The whole of the arithmetic, so that it exists once: multiply by the rate once,
    quantize once — to the step, with the rounding, in the order fx-rates.yaml declares
    and `_fx_quantization_declared` verifies. The exact document-currency sum goes in;
    nothing downstream re-quantizes what comes out.
    """
    return (amount * rate).quantize(_FX_STEP, rounding=ROUND_HALF_UP)


def _one_claim_one_currency(documents: Sequence[DocGroundTruth]) -> str:
    """The single currency a claim's documents are stated in, or a refusal.

    Coverage pools line items across the claim's subject documents, and every
    cross-document axis compares amounts between its documents. Both assume one currency,
    and policy.yaml states no rule for either across two — which order to convert and
    compare in, at whose date — so a mixed-currency claim is refused rather than guessed
    at. The planner never builds one; the guard is for the path that does not go through
    the planner.
    """
    currencies = sorted({document.currency for document in documents})
    if len(currencies) > 1:
        raise PolicyGapError(
            f"the documents of this claim are stated in {len(currencies)} currencies "
            f"({', '.join(currencies)}), and policy.yaml states no rule for pooling or "
            "comparing amounts across two — converting either side at either document's "
            "date is a choice the file does not make. One claim, one currency."
        )
    return currencies[0]


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

    claim_currency = _one_claim_one_currency(documents)
    fx = _applied_conversions(documents)
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

        Four verdicts reach it: `not_proof_of_payment`, `insufficient_evidence`, `rejected` for
        a payment outside the benefit period, and `partially_paid`. `verdict_basis` is
        `DOCUMENTS` for all four, because each of the facts above is printed on the image.

        ⚠️ THE FOURTH IS NOT A CLAIM THAT FAILED ANYTHING, and it reaches this helper for what the
        helper does rather than for what its name suggests. A `partially_paid` claim's documents
        agree; what it reimburses is nothing, because policy.yaml has not decided how much of a
        partly settled obligation is payable — see `partial_payment` there. The helper's job is
        "verdict established from the CONTENT of the documents, paying out nothing", and that is
        exactly this case.
        """
        return ClaimEvaluation(
            verdict=verdict,
            covered_fraction=fraction,
            reimbursable=Decimal("0.00"),
            verdict_basis=(VerdictBasis.DOCUMENTS,),
            imperfection=causes,
            policy_trace=tuple(trace),
            fx=fx,
        )

    # -- both facts, or neither verdict. See `resolve_evidence`.
    if not shape.proves_payment:
        trace.append(f"evidence: no document proves payment — {_types(documents)}")
        return refused(Verdict.NOT_PROOF_OF_PAYMENT, ())
    if not shape.proves_subject:
        trace.append(f"evidence: no document states what was bought — {_types(documents)}")
        return refused(Verdict.INSUFFICIENT_EVIDENCE, (SUBJECT_NOT_EVIDENCED,))
    trace.append(_evidence_trace(shape))

    # -- the documents of one transaction have to describe one transaction, on every axis
    # policy.yaml declares. The VERDICT comes from the file too — see `cross_document_agreement`.
    disagreements = _disagreements(shape)
    if disagreements:
        verdicts = {verdict for verdict, _, _ in disagreements}
        if len(verdicts) > 1:
            # 🔴 POLICY.YAML MAY DECLARE A DIFFERENT OUTCOME PER AXIS, AND SAYS NOTHING ABOUT A
            # CLAIM THAT FAILS TWO AXES DECLARING DIFFERENT ONES. Picking either would be this
            # engine deciding a precedence the file does not state, and a consumer's engine
            # picking the other would label the same claim differently — see
            # `partial_payment.outside_the_period`, which is what a stated precedence looks like.
            # Unreachable while every declared axis names one verdict, which is today's file.
            raise PolicyGapError(
                "the documents of this claim disagree on axes policy.yaml gives different "
                f"verdicts: {sorted((cause, verdict.value) for verdict, cause, _ in disagreements)}"
                ". `cross_document_agreement` states no precedence between them, so there is no "
                "answer to derive — declare one verdict for both axes, or state which wins."
            )
        trace.extend(line for _, _, line in disagreements)
        return refused(
            disagreements[0][0], tuple(cause for _, cause, _ in disagreements)
        )

    # -- the period, on the payment date and on no other. `rejected`, not
    # `insufficient_evidence`: nothing here is unestablished, the policy simply does not
    # cover a payment made outside its window. See the module docstring, rule 4.
    start, end = active_period()
    late = [t.payment for t in shape.transactions if not start <= t.payment.date <= end]
    if late:
        # 🔴 THE ONE CASE WHERE THIS BRANCH AND THE ONE BELOW BOTH APPLY, and policy.yaml decides
        # which wins. The order these two are written in IS the answer this engine gives, so the
        # order is checked against the file rather than assumed to still agree with it: a consumer
        # building its own engine from that file must not be able to reach a different LABEL while
        # reading the same rules. Flipping the declared precedence is a real edit somebody may
        # make; it has to move this code too, and this is what says so instead of a comment.
        if any(_settles_one_instalment(t) for t in shape.transactions):
            declared = partial_payment_outside_the_period()
            if declared is not Verdict.REJECTED:
                raise PolicyGapError(
                    f"policy.yaml says a partial settlement paid outside the benefit period is "
                    f"{declared.value!r} (`partial_payment.outside_the_period`), and this engine "
                    "checks the period first, which answers 'rejected'. The two would label the "
                    "same claim differently. Move the partial-settlement branch above the period "
                    "check, or restore the declared precedence."
                )
        first = min(late, key=lambda document: document.date)
        trace.append(
            f"period: payment {first.doc_id} dated {first.date} falls outside "
            f"{start}..{end}"
        )
        return refused(Verdict.REJECTED, (OUTSIDE_PERIOD,))
    trace.append("period ok")

    # -- one transaction, settled in parts. AFTER THE PERIOD AND BEFORE COVERAGE, which is a
    # precedence policy.yaml decides and this code follows — `partial_payment.outside_the_period`
    # there, with the reasoning in the paragraphs above it. In one line: the evidence verdicts come
    # first because a claim that has established nothing cannot be assessed at all, and among the
    # POLICY verdicts the period is prior, because it answers whether the plan covers this expense
    # while `partially_paid` is a statement about a claim the plan does cover.
    #
    # 🔴 THE DISCRIMINATION FROM `amount_mismatch` DOES NOT LIVE HERE and is unaffected by this
    # ordering: it is the exemption inside `_disagreements`, which runs before either branch. What
    # this branch decides is only what an exempted pair is LABELLED, so moving it past the period
    # check relabels the out-of-window case and nothing else. See `_settles_one_instalment`.
    instalments = [t for t in shape.transactions if _settles_one_instalment(t)]
    if instalments:
        settlement = instalments[0]
        subject, payment = settlement.subject, settlement.payment
        part = subject.instalment_amount
        if part is None:  # pragma: no cover - `_settles_one_instalment` returns False for one
            raise AssertionError(
                f"{subject.doc_id} was read as settling one instalment and carries none; the "
                "predicate and this branch have come apart"
            )
        # The subject document's own currency, not the reporting one: the amounts on this
        # line are read off that document, and the claim's single currency is guaranteed
        # by `_one_claim_one_currency` rather than assumed to be the reporting one.
        trace.append(
            f"partial settlement: {subject.doc_id} states {_money(subject.amount)} "
            f"{subject.currency} settled in parts of {_money(part)}, and payment "
            f"{payment.doc_id} states {_money(payment.amount)}"
        )
        # NO CAUSE, for the reason `not_proof_of_payment` carries none: there is one mechanism
        # behind this verdict and one way to reach it, so there is nothing for a cause to
        # distinguish.
        return refused(Verdict.PARTIALLY_PAID, ())

    if total <= 0:
        raise ValueError(f"a claim with a total of {total} has no verdict")

    # Before the trace line, so that a claim with no verdict does not get a justification
    # for one.
    #
    # The verdict and the fraction are decided in the CLAIM'S OWN currency: both are
    # ratios of the same line items, so the rate cancels, and converting first would only
    # add a quantization the ratio does not need.
    verdict = verdict_for(covered, total, every_line_covered=all(answers))
    trace.append(_coverage_trace(answers, covered / total))

    # 🔴 THE CURRENCY BOUNDARY. Everything above this line is stated in the claim's own
    # currency; everything below — the ledger, the annual limit, `reimbursable` — is
    # stated in the reporting currency. The covered amount crosses here: converted per
    # subject document, at that document's date, at the point fx-rates.yaml declares
    # (sum-then-convert-then-quantize — the document's exact covered sum, times the rate,
    # quantized once). The trace shows the arithmetic so the label proves the crossing.
    covered_reporting = covered
    if claim_currency != reporting_currency():
        rate_of = {applied.doc_id: applied.rate for applied in fx}
        cursor = 0
        pieces = []
        for document in shape.subject_documents:
            lines = document.line_items
            doc_covered = sum(
                (
                    item.qty * item.price
                    for item, is_covered in zip(
                        lines, answers[cursor : cursor + len(lines)], strict=True
                    )
                    if is_covered
                ),
                Decimal(0),
            )
            cursor += len(lines)
            piece = _in_reporting(doc_covered, rate=rate_of[document.doc_id])
            trace.append(
                f"currency: {document.doc_id} covers {_money(doc_covered)} "
                f"{claim_currency} × {rate_of[document.doc_id]} (fx-rates.yaml, "
                f"{document.date}) = {_money(piece)} {reporting_currency()}"
            )
            pieces.append(piece)
        covered_reporting = sum(pieces, Decimal(0))

    reimbursable = _reimbursable(
        persona_id=persona_id, category_id=category, covered=covered_reporting,
        ledger=ledger,
    )

    basis = [VerdictBasis.DOCUMENTS]
    imperfection: list[str] = []
    if verdict is Verdict.PARTIALLY_COVERED:
        imperfection.append(MIXED_ITEMS)

    if reimbursable < covered_reporting:
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
            f"{_money(covered_reporting)} covered — annual limit exhausted"
        )

    return ClaimEvaluation(
        verdict=verdict,
        covered_fraction=fraction,
        reimbursable=reimbursable,
        verdict_basis=tuple(basis),
        imperfection=tuple(imperfection),
        policy_trace=tuple(trace),
        fx=fx,
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
