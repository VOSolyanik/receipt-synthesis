# Architecture

How `receipt-synth` produces document images whose ground truth is known by construction.

- [Label-first generation](#label-first-generation)
- [Pipeline](#pipeline)
- [Configuration model](#configuration-model)
- [Ground truth](#ground-truth)
- [Document archetypes](#document-archetypes)
- [Imperfection catalogue](#imperfection-catalogue)
- [Degradation](#degradation)
- [Determinism](#determinism)
- [Extending the generator](#extending-the-generator)
- [Provenance](#provenance)

---

## Label-first generation

The usual route to a labelled dataset is *collect, then annotate*. This generator inverts it.

A claim's **target verdict is chosen first** — say `partially_covered` at 72% coverage, caused by a
non-reimbursable line item — and the documents are then constructed to realize exactly that. The label is a
premise of generation, not a conclusion drawn from the output.

Two consequences follow, and both are load-bearing:

1. **There is no annotation step, and no annotation error.** Field values, bounding boxes, per-line coverage
   flags and the final verdict are all known before a single pixel is rendered.
2. **The generator has to know the coverage rules.** A verdict and a covered fraction cannot be computed
   without them. That is why a labelling policy is a first-class, declarative input rather than logic buried
   in code — see [Configuration model](#configuration-model).

Nothing in the pipeline may infer a label from a rendered image. Such a step would reintroduce exactly the
uncertainty the design exists to remove.

---

## Pipeline

```
persona_generator → claim_planner → content_builder → renderer → degrader → assembler
```

Each stage is a module with an explicit input and output contract.

### 1. `persona_generator`

Produces synthetic people. A persona fixes the context every downstream document inherits: which country's
fiscal rules apply, which currencies and languages are natural, which benefit categories the person holds.

```yaml
persona_id: p001
full_name: <Faker, locale-appropriate>
location: { country: UA|PL|DE|ES, city: <Faker> }
home_currencies: [UAH, ...]
languages: [uk, ...]
tax_id: <valid, checksum-correct>
family: [{ relation: spouse|child, name, birth_date }]
benefit_categories: [<drawn per config/policy.yaml>]
```

The distribution of countries, currencies and languages is controlled rather than uniform, so the resulting
dataset is balanced along those axes instead of accidentally skewed.

### 2. `claim_planner`

The label-first core. For each persona × category it draws a target verdict and an imperfection mechanism,
then selects the document archetypes that realize them — including claims whose evidence is deliberately
split across several documents.

It processes a persona's claims in date order and carries the remaining category balance, so that a claim can
also become partially covered by exhausting an annual limit rather than by containing a non-covered item.

The verdict itself is not computed here. The planner chooses what to build and what answer it is aiming at; a
separate policy engine reads `config/policy.yaml` and derives the verdict, the covered fraction, the
reimbursable amount and the justification from the documents that were actually built. Keeping the two apart
is what lets the answer be checked as a pure function of (claims, policy), with no rendering anywhere near it
— and the engine is free to disagree with the plan. It does, whenever a limit binds.

### 3. `content_builder`

Fills the plan with concrete, valid data: checksum-correct tax identifiers, the VAT letter that matches the
item kind, an amount in words that matches the amount in digits, line items that sum to the stated total,
dates consistent with the claimed period.

For fraud and trap archetypes it breaks these invariants **on purpose**. A broken invariant is always an
explicit, labelled choice — never an accident. The same invariant validators are exercised by the test suite.

The validators are a deliberate second implementation, not a redundant one. An honest document satisfies its
invariants *by construction* — the total is computed as the sum of the line items, so it cannot disagree with
them — and on that path the validators are exercised only by the tests. They earn their place on the other
path: a fraud archetype states an amount that does **not** follow from its lines, and something has to
establish which invariant was broken and by how much, so that the imperfection can be named in the label
rather than merely rendered. Read `validate_line_item_sum` and its siblings as the specification the
generator is checked against, and as the mechanism the deliberately-broken archetypes will be built on — not
as guards on the honest path.

### 4. `renderer`

Jinja2 → HTML → Playwright screenshot.

In the same pass it reads `getBoundingClientRect()` for every element carrying a `data-field` attribute, so
each extracted field arrives with pixel coordinates. The boxes come from the browser's own layout engine, not
from OCR — they are exact by construction.

Output: a clean image plus its ground-truth record.

### 5. `degrader`

Real documents reach a verification system as a screenshot, a photo taken at an angle, or a flatbed scan.
Augraphy supplies paper, ink and shadow effects; Albumentations supplies perspective, blur and JPEG
artifacts.

Labels are unchanged by degradation, so **one rendered document yields several training examples**.
Bounding boxes are transformed together with the image under geometric operations.

### 6. `assembler`

Writes the dataset, splits it into train and validation, and emits a balance report over document classes,
verdicts, currencies and languages so that skew is visible rather than discovered later.

---

## Configuration model

Six files, kept apart because they have different natures and different rates of change.
`config/README.md` is the authoritative description of the model; this table is a summary of it.

| File | Nature | Changes when |
|---|---|---|
| `config/policy.yaml` | **Policy.** What a benefit plan reimburses, up to what limit, in what period | The plan being modelled changes |
| `config/labelling-schema.yaml` | **Contract.** What is labelled, how a value is compared, and the dataset's known limitations | A field, a type or a comparison rule changes |
| `config/fiscal-rules.yaml` | **Law.** VAT rates and letter codes, identifier formats and checksums, receipt layout constants, fiscal QR payloads | Legislation changes |
| `config/generation.yaml` | **Generation input.** The vocabulary that fills the placeholders of a name template, retail price ranges, basket shape, mixed-basket coverage targets | You want more variety, different prices or a different basket shape |
| `config/fx-rates.yaml` | **Reference data.** Static exchange rates | Rarely; static on purpose, see [Determinism](#determinism) |
| `config/vendors.json` | **Data.** Vendor, bank and payment-provider names per category and jurisdiction, and which item kinds each sort of outlet sells | You want different merchants |

`policy.yaml` is declarative rather than embedded in code for a specific reason: because generation is
label-first, that file *is* the ground truth. Any downstream document-verification pipeline can load the same
file and stay consistent with the labels of the dataset it was trained on.

Five of the six are read by the generator. `labelling-schema.yaml` is not: it describes the output to whoever
consumes it — field names, comparison rules, and the dataset's `known_limitations` — and adding a loader for
it in `src/` would be inventing a caller.

Nothing in `src/` carries merchandise data of its own. A name a document prints, a price it charges and a
merchant that issued it are all read from configuration — the one thing deliberately left in code is
`MAX_LINE_ITEMS`, which is a bound the planner needs a name for rather than a knob anyone tunes.

### The item-kind vocabulary

Each category lists item kinds — `gym_membership`, `vitamin_complex`, `medicine` — with the line-item name
templates used when rendering. The kind is the join key between three layers:

```
                    item_kind: "vitamin_complex"
                   ┌────────┬───────────┬────────────┐
                   │        │           │            │
           content_builder  policy.yaml  fiscal-rules.yaml
           which name to    covered?     which VAT letter
           print                         (7% vs 20%)
```

No such kind is printed on a real receipt — the receipt says "Вітамін D3 2000 МО табл. №60" and something has
to decide whether that is reimbursable. The vocabulary is therefore a **ground-truth label space**, not a
runtime mechanism. It is equally useful whether a consumer resolves coverage with explicit rules or with a
language model: in the first case it is the target of the mapping step, in the second it is the reference the
model is scored against.

Each category also carries an `ambiguous_items` bucket — kinds a careful human would have to think about.
These are the most valuable examples in the dataset.

---

## Ground truth

### Document level

```json
{
  "doc_id": "p001_c2_d1",
  "source_file": "p001_c2.png",
  "doc_type": "fiscal_receipt",
  "language": "uk",
  "currency": "UAH",
  "amount": 1250.00,
  "date": "2026-08-03",
  "counterparty": "<vendor>",
  "line_items": [
    { "name": "...", "qty": 1, "price": 1000.00, "covered": true, "vat_letter": "А" }
  ],
  "has_qr": true,
  "qr_is_fiscal": false,
  "has_fiscal_number": true,
  "capture": "screenshot",
  "field_bboxes": { "amount": [0, 0, 0, 0], "date": [0, 0, 0, 0] },
  "synthetic": true,
  "generator_version": "0.1.0"
}
```

Every record carries `synthetic: true` and the generator version. This is not decoration — it is what makes
the provenance of any individual file unambiguous once it leaves this repository.

### Claim level

A claim may span several documents.

```json
{
  "claim_id": "p001_c2",
  "persona_id": "p001",
  "category": "sport",
  "documents": ["p001_c2_d1", "p001_c2_d2"],
  "verdict": "partially_covered",
  "covered_fraction": 0.72,
  "reimbursable_amount": 900.00,
  "linked": true,
  "imperfection": ["mixed_items"],
  "verdict_basis": ["documents"],
  "policy_trace": [
    "category=sport ok",
    "period ok",
    "coverage 72% (1 of 4 line items not covered)"
  ]
}
```

Five fields deserve attention.

**`covered_fraction`** is the covered amount over the total, taken from the line items and nothing else. It
reports what the *document* covers, which is not always what the plan *pays*: a claim whose every line is
covered but whose annual limit has run out reads `covered_fraction: 1.0` and
`verdict: "partially_covered"`. That pairing is not a contradiction, it is the two facts kept apart.

Note also that the fraction never decides the verdict. The coverage rule is strict — *any* non-covered line
makes a claim partially covered, however small — so a claim spanning enough documents cannot dilute a real
non-reimbursable article into a rounding error.

**`reimbursable_amount`** is what the plan actually pays out for the claim, in the policy's reporting
currency: the covered amount, capped by whatever is left of the annual limit. It is the only field that
distinguishes a limit-bound claim numerically, and it is what makes the pairing above readable.

**`imperfection`** names why a `partially_covered` claim is partial, using the causes declared in
`policy.yaml`: `mixed_items` (a non-covered line is on the document) and `limit_exhausted` (the annual
balance ran out). A claim can carry both.

**`verdict_basis`** records what the verdict actually depends on:

- `["documents"]` — derivable from the images alone.
- `["documents", "account_state"]` — also requires the persona's spending history, for example when an annual
  limit is already exhausted. No model can read a remaining balance off a receipt.

This is exactly why the two causes are distinguished: `mixed_items` is visible in the image, `limit_exhausted`
is not. Document-understanding metrics should be computed on the first subset only; end-to-end system metrics
on both. Without this distinction a model is penalized for information it was never given.

**`policy_trace`** is the human-readable justification of the verdict. It costs nothing to emit — the
generator necessarily knows why the verdict is what it is — and it gives consumers a reference for evaluating
explanation quality, not just decision accuracy. It carries only what the verdict rested on: a limit that did
not bind is not mentioned, and a claim rejected for its date says so without arguing about coverage.

### Verdicts

| Verdict | Meaning |
|---|---|
| `covered` | Fully reimbursable |
| `partially_covered` | Some of the amount qualifies — mixed items, or an exhausted limit |
| `rejected` | Nothing bought is covered by the category |
| `not_proof_of_payment` | The evidence does not establish that money changed hands |
| `insufficient_evidence` | A required fact is missing entirely |
| `partially_paid` | Payment was made in installments; only part has been paid |

`rejected` and `not_proof_of_payment` are easy to merge and must not be, in either direction. They answer
different questions, and each reads a different part of the labelling policy to answer it.

**`rejected` answers what was bought.** The category covers none of it. That is decided from the line
items, by resolving each item kind against the category's covered and excluded vocabularies, so the answer
is visible on the document itself.

**`not_proof_of_payment` answers whether money moved.** That is a property of the document *type*, read
from `proves_payment` in the `document_evidence` block, and it has nothing to do with what was bought. A
fiscal receipt proves its payment whatever its basket was — so a pharmacy receipt listing nothing but
medicines is `rejected`, and is not, on any reading, a failure of proof of payment.

---

## Document archetypes

Twenty-four HTML/CSS templates, sixteen Ukrainian and eight European. Layouts follow the publicly observable
conventions of each document class and jurisdiction.

### Ukraine

| # | Template | Class | Role |
|---|---|---|---|
| 1–4 | Bank in-app payment confirmations (four issuers) | `payment_confirmation` | Mainstream bank layouts |
| 5–7 | Payment-service receipts (three providers) | `payment_confirmation` | Payment-gateway layouts |
| 8 | Software cash register receipt — QR, fiscal number, VAT letters | `fiscal_receipt` | Modern fiscal document |
| 9 | Classic thermal cash register receipt (58/80 mm) | `fiscal_receipt` | Layout diversity |
| 10 | Account statement for a period, multi-transaction PDF | `bank_statement` | The statement class |
| 11 | Sole-trader invoice for services | `invoice` | The invoice class |
| 12 | Sales slip marked "not a fiscal receipt" | trap | Fiscality trap |
| 13 | Non-fiscal POS slip — RRN and auth code, no fiscal number | trap | Fiscality trap |
| 14 | Online marketplace order screenshot | linked | Proves the subject, not the payment |
| 15 | Bank receipt for an insurance premium, detailed payment purpose | `payment_confirmation` | Insurance proven by payment, not by policy |
| 16 | Act of services rendered | reference | Service-type reference |

### Europe

| # | Template | Jurisdiction | Class |
|---|---|---|---|
| 17 | Paragon fiskalny — NIP, PTU letters, thermal | PL | `fiscal_receipt` |
| 18 | Potwierdzenie przelewu | PL | `payment_confirmation` |
| 19 | Kassenbon — TSE QR, MwSt | DE | `fiscal_receipt` |
| 20 | Factura simplificada — IVA, CIF | ES | `invoice` / fiscal |
| 21 | SEPA transfer confirmation | EU | `payment_confirmation` |
| 22 | VAT invoice (English) | EU | `invoice` |
| 23 | Transfer confirmation for an insurance premium | PL | `payment_confirmation` |
| 24 | Leistungsnachweis — act of services rendered | DE | reference |

Four target classes are `payment_confirmation`, `fiscal_receipt`, `invoice` and `bank_statement`. The act of
services rendered (16, 24) is a reference type outside them: it is generated so that a classifier meets
documents that belong to none of the target classes.

---

## Imperfection catalogue

Perfect documents teach a model very little. The value of the dataset is in the ways real evidence falls
short. Each mechanism is a parameter of `claim_planner`, combined with a target verdict.

| Mechanism | Example | Verdict |
|---|---|---|
| Mixed covered and non-covered items | Gym membership plus a supplement on one receipt | `partially_covered` |
| Non-covered addition to an order | A non-qualifying item inside a qualifying order | `partially_covered` |
| Annual limit exhausted | Fourth claim exceeds what remains | `partially_covered` |
| Category does not match the policy | A purchase outside the claimed category | `rejected` |
| Date outside the active period | Transaction before or after the window | `insufficient_evidence` |
| Document does not prove payment | Invoice marked "paid: 0"; sales slip; booking confirmation | `not_proof_of_payment` |
| Paid in installments | Part of the amount settled | `partially_paid` |
| Evidence incomplete | No proof of payment, or no statement of what was bought | `insufficient_evidence` |
| Paid through an aggregator | Payee is a payment intermediary, no visible link to the merchant | requires linking |
| Documents disagree | Payment amount differs from the contract; payment predates the contract | flag |
| Subject and payment in different documents | Order screenshot plus account statement | `covered` via linking |
| Duplicate | The same RRN in two files | dedup |
| Near-duplicate counter-example | Two genuine payments seconds apart | **not** dedup |
| Fiscal-looking but not fiscal | Full requisites, marked non-fiscal | classification trap |
| Non-fiscal QR | A marketing QR on a bank receipt | trap — decode to tell |

---

## Degradation

Three capture modes, each a different composition of effects:

| Mode | Character |
|---|---|
| `screenshot` | Clean, native resolution, mild compression |
| `photo` | Perspective, uneven lighting, shadow, camera noise, motion blur |
| `scan` | Paper texture, slight rotation, dust, scanner banding |

Geometric transforms carry the bounding boxes with them, so annotations stay aligned. The label is invariant
under degradation by construction: nothing about *what the document says* changes when it is photographed
badly.

---

## Determinism

Every run is fully determined by `--seed`. The repository ships the generator, its configuration and the
seed — not the dataset. Anyone can reproduce the same dataset byte for byte.

This is why:

- exchange rates are static (a live rates API would make labels depend on the day of the run);
- fonts are vendored rather than taken from the host system (missing fonts change pixels);
- dependencies are pinned and locked;
- the Python version has an upper bound.

Any unseeded source of randomness is a bug.

---

## Extending the generator

**A new document archetype.** Add `templates/<slug>.html` and `templates/<slug>.css`. Mark every extractable
field with `data-field="<name>"` so the renderer can capture its bounding box. Register the archetype so
`claim_planner` can select it, with the document class it belongs to and the facts it can prove (subject,
payment, or both).

**A new benefit category.** Add an entry to `config/policy.yaml` with `intent`, an annual limit and the three
item buckets — `covered_items`, `excluded_items`, `ambiguous_items` — each mapping an item kind to line-item
name templates per language. Give each new item kind a price range in `config/generation.yaml`, and a
vocabulary there for every placeholder its templates name; a placeholder with no vocabulary raises rather
than being skipped. Add vendors for it in `config/vendors.json`, each with a profile that sells at least one
of its covered kinds.

**A new jurisdiction.** Add a block to `config/fiscal-rules.yaml`: VAT rates, VAT letter codes, identifier
formats with their checksum algorithms, receipt layout constants and the fiscal QR payload. Add matching
templates and at least one font covering the script.

**A different labelling policy.** Replace `config/policy.yaml` wholesale. Nothing in the pipeline hardcodes
category names, limits or thresholds.

---

## Provenance

The generator is built only from public sources: published fiscal law and document formats, general knowledge
of document layouts, published industry surveys, `Faker`, publicly known vendor names, and the author's own
receipts. No real document belonging to any third party is used as a source anywhere in this project.

Benefit categories are derived from a [public industry survey of benefits packages in the Ukrainian IT
market](https://dou.ua/lenta/articles/benefits-package-2026/) (DOU). Limits, thresholds and the verdict mix
are this project's own plausible values and describe no real employer's plan.

Everything the tool produces is synthetic: documents that were never issued, for transactions that never
happened, by people who do not exist.
