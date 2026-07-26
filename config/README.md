# Configuration

Four files, split by **nature** rather than by convenience. The split is deliberate and worth preserving:
each file changes for a different reason, and two of them are meaningful outside this repository.

| File | Nature | Changes when | Meaningful to consumers |
|---|---|---|---|
| `policy.yaml` | **Policy** — what a benefit plan reimburses | The plan being modelled changes | **Yes** |
| `fiscal-rules.yaml` | **Law** — VAT, identifiers, receipt layout | Legislation changes | No |
| `fx-rates.yaml` | **Reference data** — exchange rates | Rarely; static on purpose | No |
| `vendors.json` | **Data** — merchant names | You want different merchants | No |

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

## `fx-rates.yaml` — reference data

Static rates, per unit of foreign currency, in the reporting currency of `policy.yaml`.

Static **by design**: a live rates API would make the same seed produce different labels on different days,
which would break the reproducibility the whole tool is built around. A seeded jitter is available and
disabled by default, for datasets that need rate variation as a signal.

---

## `vendors.json` — data

Merchant, bank, payment-provider and marketplace names, per category and jurisdiction. Category keys must
match the category ids in `policy.yaml`.

The `aggregators` block holds payment intermediaries — payees that break the visible link between a payment
and the merchant, which drives the "paid through an aggregator" imperfection.

---

## Adding to the configuration

**A benefit category** — add an entry to `policy.yaml` with `intent`, an annual limit and the three item
buckets, then add vendors for it under each jurisdiction in `vendors.json`.

**A jurisdiction** — add a block to `fiscal-rules.yaml` (VAT rates and letters, identifier formats and
checksums, number and date formats, layout constants, fiscal QR payload), then add matching templates and a
font covering the script.

**A currency** — add it to `fx-rates.yaml`.
