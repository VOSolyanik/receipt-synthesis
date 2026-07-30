# Configuration

Six files, split by **nature** rather than by convenience. The split is deliberate and worth preserving:
each file changes for a different reason, and two of them are meaningful outside this repository.

| File | Nature | Changes when | Meaningful to consumers |
|---|---|---|---|
| `policy.yaml` | **Policy** — what a benefit plan reimburses | The plan being modelled changes | **Yes** |
| `labelling-schema.yaml` | **Contract** — what is labelled, and how a value is compared | A field, a type or a comparison rule changes | **Yes** |
| `fiscal-rules.yaml` | **Law** — VAT, identifiers, receipt layout | Legislation changes | No |
| `generation.yaml` | **Generation input** — the vocabulary and distributions the draws use | You want more variety, different prices or a different basket shape | No |
| `fx-rates.yaml` | **Reference data** — exchange rates | Rarely; static on purpose | No |
| `vendors.json` | **Data** — merchant names, and what each sort of outlet sells | You want different merchants | No |

Five of the six are read by the generator. `labelling-schema.yaml` is not: it describes the generator's
output to whoever consumes it, and there is deliberately no loader for it in `src/`.

---

## `policy.yaml` — the labelling policy

Because generation is label-first, this file *is* the ground truth. A claim's verdict and covered fraction are
derived from these rules, and the documents are then built to satisfy them — the rules are never applied to a
finished document after the fact.

It is declarative, and separate from the code, so that a downstream document-verification pipeline can load
the same file and stay consistent with the labels of the dataset it was trained on. Nothing in the pipeline
hardcodes category names, limits or thresholds: replacing this file retargets the generator at a different
plan.

### Anatomy of a category

```yaml
- id: sport
  display: { uk: "Спортивна програма", en: "Sports program" }

  # The prose rule — what a real benefit policy document would say.
  intent: >
    Membership fees and paid classes at sports facilities. Goods, apparel,
    supplements and food bought at the same venue are not reimbursed.

  # Binds on CUMULATIVE spend over the period, not on a single claim.
  annual_limit: 12000

  # The machine-readable specification of `intent`:
  # item kind -> line-item name templates, per document language.
  covered_items:
    gym_membership:
      uk: ["Абонемент {n} занять", "Клубна карта, {period}"]
      en: ["Membership, {n} sessions", "Club card, {period}"]
  excluded_items:
    sports_nutrition:
      uk: ["Протеїн {brand} {w} г"]
      en: ["Protein {brand} {w} g"]
  ambiguous_items:
    personal_training:
      uk: ["Персональне тренування, {n} шт"]
      en: ["Personal training, {n} sessions"]
```

**Item kinds are the join key** between three layers: content generation (which name to print), this policy
(covered or not), and `fiscal-rules.yaml` (which VAT letter applies).

```
                    item_kind: "vitamin_complex"
                   ┌────────┬───────────┬────────────┐
                   │        │           │            │
           content_builder  policy.yaml  fiscal-rules.yaml
           which name to    covered?     which VAT letter
           print                         (7% vs 20%)
```

No such kind is printed on a real receipt — the receipt says "Вітамін D3 2000 МО табл. №60" and something has
to decide whether that is reimbursable. The vocabulary is a **ground-truth label space**, not a runtime
mechanism. It serves consumers that resolve coverage with explicit rules (it is the target of the mapping
step) and consumers that use a language model (it is the reference to score against) equally.

`ambiguous_items` holds kinds a careful human would have to think about. These produce the most valuable
examples in the dataset — keep the bucket populated when adding a category.

### Verdict derivation

```yaml
coverage:
  full_threshold: 0.9999       # absorbs floating-point error only, never a real item

limits:
  enforce_cumulative: true     # claims processed in date order, carrying the remaining balance

verdict_mix:                   # target distribution; the balance report checks the realized one
  covered: 0.50
  partially_covered: 0.20
  not_proof_of_payment: 0.10
  insufficient_evidence: 0.10
  partially_paid: 0.10
  rejected: null               # share decided together with the mechanism that builds one

partially_covered_causes:      # splits the bucket above by cause; must sum to 1.0
  mixed_items: 0.65
  limit_exhausted: 0.35

document_evidence:
  fiscal_receipt:       { proves_subject: true,  proves_payment: true  }
  payment_confirmation: { proves_subject: false, proves_payment: true  }
  bank_statement:       { proves_subject: false, proves_payment: true  }
  invoice:              { proves_subject: true,  proves_payment: false }
  act:                  { proves_subject: true,  proves_payment: false }
  order_screenshot:     { proves_subject: true,  proves_payment: false }
  non_fiscal_receipt:   { proves_subject: true,  proves_payment: false }
```

**Coverage is strict.** Any non-covered line item makes a claim `partially_covered`, however small — a 3 UAH
carrier bag on a 1000 UAH pharmacy receipt is not reimbursed, and the label says so. The threshold sits at
0.9999 because it exists to absorb rounding when summing amounts, not to absorb real items. A looser
threshold would put images containing a visibly non-covered line into the dataset labelled as fully covered,
teaching consumers to ignore small exclusions.

**Two causes of partial coverage, with different consequences.** A non-covered line item is visible in the
image; an exhausted annual limit is not. The split is stated explicitly rather than as a single rate so the
resulting share of the whole dataset is predictable — `0.20 × 0.35 = 7%` of claims exercise the limit
mechanism. Limit-driven claims are recorded with `verdict_basis: ["documents", "account_state"]` and should
be excluded from document-understanding metrics. See
[docs/architecture.md](../docs/architecture.md#ground-truth).

**Evidence is two facts, not one.** *What was bought* and *that it was paid for* are established
independently, possibly by different documents. `document_evidence` is the matrix the planner uses to
assemble a claim: it keeps adding documents until both columns are satisfied. That single table is where
linked claims come from (an order screenshot plus an account statement) and where the traps come from (an
invoice on its own proves nothing was paid). A fiscal receipt is the only common document establishing both
at once.

These are type-level defaults. A specific archetype may override them — a bank payment confirmation whose
payment purpose spells out what was bought does prove the subject, unlike a bare transfer. That override
belongs to the archetype registration, not to this file.

**`verdict_mix` is a dataset-balance decision, not an estimate of anything.** In production the `covered`
class would dominate heavily. Over-representing the harder verdicts is what makes per-class F1 meaningful,
but it carries an obligation for anyone measuring against this dataset:

> Report every metric twice — unweighted on the balanced dataset, which is what makes per-class numbers
> meaningful, and reweighted by the class prior you expect in your own deployment, which is what an
> operational estimate needs.

**That prior is deliberately not defined here.** It is an assumption about a particular deployment, it is
consumed by whoever computes metrics, and no stage of this generator reads it — so it belongs to the
consumer's evaluation configuration, not to the specification of the dataset. Nothing in this repository
measures the natural distribution of verdicts, and a held-out set of genuine receipts would not measure it
either: real receipts establish how well documents are read, not how often claims come out one way or another.

Worth being precise about what gets calibrated to this distribution: the evaluator, and whoever tunes prompts
and thresholds against the dataset — not a set of model weights. The effect is real, but it runs through the
human in the loop, so it applies just as much to a pipeline that trains nothing at all.

---

## `labelling-schema.yaml` — the contract

`policy.yaml` says what the plan reimburses. This file settles the **names, types and comparison rules** of
everything the generator labels, so that a consumer's extraction schema and its scorecard are derived from
one agreement instead of two. Without it a system that reads every field correctly can score zero because it
returns `merchant` where the label says `counterparty`.

It is the only file here that carries **both sides**: what the generator emits today, and what the consumer's
requirements demand. Where the two disagree it names both and picks neither — a contract that resolved a
divergence quietly would hide it rather than fix it. Every `src/` change those divergences imply is collected
in a `required_changes` block at the foot of the file; none has been made.

Read the file itself for the field lists and rules. Four things about it belong here:

- **One model serves all seven document types.** Nothing in `schemas.py` varies the label shape by
  `doc_type`, while the consumer's requirements are stated per type. That gap is recorded in the file rather
  than closed, because a per-type label shape would make the label depend on the classification answer — one
  of the things being evaluated.
- **Every section describing a label is marked `emitted`, `forward_contract`, `divergent` or `undecided`.**
  One archetype ships so far, so much of the file is contract rather than observation, and the two are never
  mixed in one section. Each `emitted` statement was verified field by field against a freshly generated
  label file.
- **`known_limitations` names every place a label is recoverable without reading the document properly** —
  which predictor leaks, on which subset, what share of a stated reference run, and how a consumer should
  stratify to avoid being flattered by it. It is here rather than in a code comment because a consumer reads
  the contract and not `content_builder.py`, and it distinguishes the two kinds of consumer: one that trains
  on the dataset absorbs the shortcut, one that prompts a language model does not — but its scorecard, and
  the human tuning against it, still do.
- **Nothing in `src/` reads it.** It describes this generator's output to its consumers; adding a loader
  would be inventing a caller.

---

## `fiscal-rules.yaml` — law

One block per jurisdiction (UA, PL, DE, ES, EU): VAT rates, VAT letter codes, identifier formats with the
name of their checksum algorithm, number and date formatting, receipt layout constants, the acquiring block,
and the fiscal QR payload.

Two things worth knowing before editing:

- **VAT letter mappings are not uniform across jurisdictions.** In Poland the PTU letters are fixed by law
  (A = 23%, B = 8%, C = 5%…). In Ukraine the seller assigns the letters and prints a legend on the receipt,
  so the mapping here is a convention, not a statute.
- **The same item kind takes different rates in different countries.** Medicines are 7% in Ukraine and 5% in
  Poland, but carry the standard 19% rate in Germany. `item_vat_letter` is per jurisdiction for this reason.

The UA block carries a `verified_against_own_receipts` list: layout facts confirmed against real receipts, so
the provenance of each non-obvious detail is auditable. Only layout facts are ever carried over from a real
document — never the concrete values printed on it.

---

## `generation.yaml` — generation input

The vocabulary and the distributions the draws use: what fills a `{brand}` or a `{dose}` in the name
templates of `policy.yaml`, what an article of each item kind plausibly costs, how many of it a basket
holds, and what coverage ratio a deliberately mixed basket aims at.

**Why it is not part of `policy.yaml`.** A benefit plan document states what is reimbursed; it says nothing
about what a vitamin pack costs or which brands a pharmacy stocks. Those are properties of the merchandise.
The item kinds and the name templates stay in `policy.yaml` because they *are* the label space — this file
only fills the holes those templates leave.

**Why the coverage targets are here and not beside `verdict_mix`.** They shape a label distribution, which
makes them a relative of `verdict_mix` rather than of a price range. But they are an *aspiration*: the
builder clamps every non-covered line into its item kind's own price range, so the realized ratio only
approaches the target, and `covered_fraction` in the ground truth is measured from the document that was
actually built. A consumer that loaded the target would learn nothing it could use and might mistake an
aspiration for a label — so it belongs with the generator's own draw inputs.

**Placeholder vocabularies are keyed by item kind**, then by language, with a language-neutral `shared`
scope and a kind-independent `default` scope. Keyed by kind because one placeholder means different things
in different kinds: `{brand}` on `hardware` wants a laptop maker and on `cosmetics` wants a skincare house,
and `{dose}` is 400–5000 IU for a vitamin but 14–45 mg for an iron tablet.

**A placeholder with no vocabulary raises.** It used to make the builder skip the template silently, which
left the printed vocabulary of the dataset a subset of the one `policy.yaml` declares, with nothing saying
which subset. The test suite sweeps every template of every category in both languages, so a new template
naming a new placeholder fails there rather than at generation time.

**Prices are stated on the same scale as `policy.yaml`** — hryvnias, as decimal text, matching
`annual_limit`. One scale across both files, because they are read together and a second scale carried only
by a suffix in a field name is a two-order-of-magnitude error that breaks nothing: it prints an implausible
receipt and waits. Strings rather than YAML numbers so the value reaches `Decimal` as decimal text and stays
exact to the kopiyka; converting to minor units for the draw is the code's job. A range stated to more than
two decimal places is refused rather than rounded.

UAH-only, honestly so: every archetype this generator has is Ukrainian. Per-jurisdiction ranges arrive with
the first non-UA archetype.

**Personal names are composed, and the surname pool is narrowed on purpose.** `personal_names.surnames`
holds, per document language, the surnames a sole trader's printed name and a persona's own name are built
from; the given name and the patronymic still come from `Faker`. The narrowing is not "avoid real surnames"
— every Ukrainian surname belongs to real people, and a sole trader has to look real. It is that the name
must not point at a *particular* person, and the property that decides is rarity together with
recognisability: a rare, loaded surname on a published receipt points, and one carried by a hundred thousand
people is noise. The list is therefore a published high-frequency set with a stated bearer threshold, not a
denylist of names to avoid — a denylist would never be finished, and every surname missing from it would
become a silent claim that it had been checked. The block in `generation.yaml` cites its source and
threshold, and says why nothing is excluded for being iconic. A language with no entry falls back to
`Faker`'s pool for that locale.

---

## `fx-rates.yaml` — reference data

Static rates, per unit of foreign currency, in the reporting currency of `policy.yaml`.

Static **by design**: a live rates API would make the same seed produce different labels on different days,
which would break the reproducibility the whole tool is built around. A seeded jitter is available and
disabled by default, for datasets that need rate variation as a signal.

---

## `vendors.json` — data

Merchant, bank, payment-provider and marketplace names, per category and jurisdiction. Category keys must
match the category ids in `policy.yaml`.

Two blocks besides the names themselves:

- **`vendor_profiles`** — vendor ↔ item-kind affinity. A pharmacy does not sell a personalised nutrition
  plan, and until this block existed nothing stopped it from printing one. Each profile lists every item
  kind an outlet of that sort can put on a line; a vendor carries a profile slug rather than its own list,
  because outlets of the same sort sell the same things — the affinity is a property of the trade. A profile
  that intersects a category's `covered_items` can issue a receipt for it at all; one that also intersects
  its `excluded_items` can carry a mixed basket. Honest profiles exist that cannot, so the assembler picks a
  vendor that can carry the plan instead of letting the builder fail on one that never could.
- **`acquirers`** — the bank name printed in the card-acquiring block of someone else's receipt. Kept apart
  from `banks`, whose entries own a template and therefore decide a layout.

The `aggregators` block holds payment intermediaries — payees that break the visible link between a payment
and the merchant, which drives the "paid through an aggregator" imperfection.

`legal_form` decides how the name is printed: a ТОВ prints `ТОВ «Name»`, a ФОП prints `ФОП Surname I. B.`
without quotes, because a sole trader's name is a person's. The names cover naming patterns on purpose —
legal entity vs sole trader, Cyrillic vs Latin vs mixed, chain vs single outlet — because that is what
counterparty extraction is trained on.

`vat_payer` decides **whether the seller block prints a VAT-payer number, and whether the receipt has a VAT
block at all**. Every seller prints an identification code under `ІД` — an ЄДРПОУ (8 digits) for a company, a
РНОКПП (10 digits) for a ФОП, so the length follows the type of person. A **registered** seller prints, *in
addition*, its VAT-payer number under `ПН`: 12 digits for a company, and for a ФОП the same 10-digit РНОКПП
its `ІД` line carries. A seller that is not registered prints no `ПН`, no per-line VAT letter and no tax
summary. **The two identifier lines are not alternatives** — a payer carries one line more, not a different
one. `vat_payer` is a separate field from `legal_form` because the two do not coincide: a ФОП on the general
system is registered, a small company on the simplified system is not. The assigned values are the author's assumption rather than a
statistic, stated as such in `$note_vat_payer`, and an entry that omits the field is refused rather than
defaulted.

---

## Adding to the configuration

**A benefit category** — add an entry to `policy.yaml` with `intent`, an annual limit and the three item
buckets; a price range and any placeholder vocabulary its templates need in `generation.yaml`; then vendors
for it under each jurisdiction in `vendors.json`, each with a profile that sells at least one of its covered
kinds.

**A jurisdiction** — add a block to `fiscal-rules.yaml` (VAT rates and letters, identifier formats and
checksums, number and date formats, layout constants, fiscal QR payload), then add matching templates and a
font covering the script.

**A currency** — add it to `fx-rates.yaml`.
