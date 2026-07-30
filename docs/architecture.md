# Architecture

How `receipt-synth` produces document images whose ground truth is known by construction.

- [Label-first generation](#label-first-generation)
- [Pipeline](#pipeline)
- [Configuration model](#configuration-model)
- [Ground truth](#ground-truth)
- [Evidence](#evidence)
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
full_name: <given name from Faker + surname from config/generation.yaml>
location: { country: UA|PL|DE|ES, city: <Faker> }
home_currencies: [UAH, ...]
languages: [uk, ...]
tax_id: <valid, checksum-correct>
family: [{ relation: spouse|child, name, birth_date }]
benefit_categories: [<drawn per config/policy.yaml>]
```

The distribution of countries, currencies and languages is controlled rather than uniform, so the resulting
dataset is balanced along those axes instead of accidentally skewed.

A personal name — a persona here, a sole trader printed by `content_builder` — is composed rather than
stored, and its surname is drawn from a deliberately narrow set of the most common surnames of the
jurisdiction, listed under `personal_names` in `config/generation.yaml`. This dataset is published, and a
rare surname on a rendered receipt points at whoever bears it however the name was produced, while a surname
carried by tens of thousands of people does not. That block states the reasoning, its source and its
frequency threshold in full; note that it is a property of the sample rather than a list of excluded names.

### 2. `claim_planner`

The label-first core. For each persona × category it draws a target verdict and an imperfection mechanism,
then selects the document archetypes that realize them — including claims whose evidence is deliberately
split across several documents.

**A plan is a list of documents.** Each entry names a template and its own date, because the documents of a
claim are not simultaneous: an invoice is issued and then settled. The planner keeps taking archetypes until
both facts a reimbursement rests on are established — what was bought, and that it was paid for — reading
what each type proves from `document_evidence` in `config/policy.yaml`. Where a single archetype proves both,
as a fiscal receipt does, the claim has one document; where none does, it takes a subject document and a
payment document. The registry currently holds three archetypes, all of them Ukrainian fiscal receipts, and
each proves both — so every claim built today has exactly one document, drawn from the three. That is a
property of the registry and not of a claim, and nothing downstream may depend on it.

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

**Not every requisite is on every document, and absence is modelled rather than tolerated.** Whether the
seller is registered for VAT is a property of the vendor (`vat_payer` in `config/vendors.json`), and it decides
two things at once: whether the seller block prints a VAT-payer number *in addition to* the identification code
every seller prints, and whether the receipt carries a VAT block at all. A seller that is not registered prints
no per-line VAT letter, no tax-summary row and nothing in their place, so `vat_letter` is legitimately `null`
and the corresponding bounding boxes are legitimately absent.

The identifier lines are worth spelling out, because a plausible reading of the published form makes them
mutually exclusive and real receipts are not: a registered company prints both, and a registered sole trader's
VAT-payer number is the same ten-digit taxpayer number its identification-code line carries. The lengths
follow the **type of person**, not the prefix. `config/fiscal-rules.yaml` records the conflict and which side
decided it.

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
  "amount_due": 1250.00,
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

`amount_due` is the receipt's `ДО СПЛАТИ` line: the total less any discount, plus cash rounding. It is
**equal to `amount` in the current version**, because both adjustments are zero — so it discriminates nothing
and its accuracy is not a meaningful metric yet. `config/labelling-schema.yaml` says so in the field's own
entry, and says why the divergence waits on a policy decision rather than on code.

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

**`covered_fraction`** is the covered amount over the total, taken from the line items of the claim's subject
documents and nothing else — see [Evidence](#evidence) for why that is not the same as every document's line
items. It reports what the *documents* cover, which is not always what the plan *pays*: a claim whose every
line is covered but whose annual limit has run out reads `covered_fraction: 1.0` and
`verdict: "partially_covered"`. That pairing is not a contradiction, it is the two facts kept apart. It is
`null` only where there is no line item to compute it from — a claim evidenced by a payment and nothing
saying what it bought — which is a different statement from `0`.

Note also that the fraction never decides the verdict. The coverage rule is strict — *any* non-covered line
makes a claim partially covered, however small — so a claim spanning enough documents cannot dilute a real
non-reimbursable article into a rounding error.

**`reimbursable_amount`** is what the plan actually pays out for the claim, in the policy's reporting
currency: the covered amount, capped by whatever is left of the annual limit. It is the only field that
distinguishes a limit-bound claim numerically, and it is what makes the pairing above readable.

**`imperfection`** names why the verdict is what it is, where the verdict alone does not say. For
`partially_covered` the causes are declared in `policy.yaml`: `mixed_items` (a non-covered line is on the
document) and `limit_exhausted` (the annual balance ran out). For `insufficient_evidence` they are
`subject_not_evidenced`, `amount_mismatch` and `payment_precedes_subject`, and for `rejected` there is one,
`outside_period` — see [Evidence](#evidence). A claim can carry more than one. The three sets are disjoint, so
a cause always determines its verdict. The converse holds for the first two verdicts only: `rejected` also
arrives through zero coverage, and a claim rejected that way carries no cause, because the verdict already
says the whole of it.

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
| `rejected` | The policy plainly does not cover the claim — nothing bought is covered by the category, or the payment falls outside the active period |
| `not_proof_of_payment` | It is not established that **money moved**: every document is of a *type* whose `proves_payment` is `false` |
| `insufficient_evidence` | It is not established **what was bought** (no document states it), or not established that the payment and the purchase are **one transaction** |
| `partially_paid` | Payment was made in installments; only part has been paid |

The evidence a claim rests on has three slots — that money moved, that a purchase was made, and that the two
are one transaction — and each row above names **its own slot**. No row is written as "the case that is not one
of the others", and no row carries an exception excluding a case another row covers. That is a rule rather
than a style: while two verdicts are defined through each other, editing the boundary of one silently moves
the boundary of the other, and nothing at the edit reveals it. It has gone wrong twice in this repository.
`verdict_notes.definitions_name_their_own_slot` in `config/labelling-schema.yaml` carries the full statement,
the test an edit has to pass, and — under `why_this_boundary_slips` — why this particular pair invites it.

The table above is sufficient on its own: everything needed to classify a claim is in it. The rest of this
section contrasts verdicts that are easy to confuse, and it is **optional reading** — that is precisely what
allows it to contrast at all. The rule governs the text you must consult to reach an answer, not the text that
checks an answer you already have.

`rejected` and `not_proof_of_payment` are easy to merge and must not be, in either direction. They answer
different questions, and each reads a different part of the labelling policy to answer it.

**`rejected` answers whether the policy covers the claim at all**, and it has two ways of answering no. By
*what* was bought: the category covers none of it, decided from the line items by resolving each item kind
against the category's covered and excluded vocabularies. By *when* it was paid: the payment falls outside the
active period. Both answers are visible on the document itself, and the second carries the cause
`outside_period` while the first carries none.

Neither is `insufficient_evidence`, and the out-of-period case is the one worth stating, because it used to be
labelled that way. `insufficient_evidence` means, by its own name, that a required fact was not established;
an out-of-period claim establishes every fact there is — the purchase happened, the proof is flawless, only
the date does not match the window. It is a case of the policy plainly not covering an expense, which is what
`rejected` is for. There is no seventh verdict for it: a named cause on a verdict whose definition already
fits says the same thing without splitting the enum.

**`not_proof_of_payment` owns the money-moved slot.** Every document of the claim is of a type whose
`proves_payment` entry in the `document_evidence` block is `false` — an invoice, an act, an order screenshot, a
non-fiscal receipt, alone or in any combination — so nothing the claim carries attests a movement of money. The
type decides it; amounts, baskets and dates are not consulted, and the verdict carries no cause because there
is one slot and one way to fail it. A fiscal receipt proves its payment whatever its basket was, so a pharmacy
receipt listing nothing but medicines is `rejected`, and is not, on any reading, a failure of proof of payment.

**`insufficient_evidence` owns the other two slots.** *What was bought* is unestablished when no document of
the claim is of a type that states it — a bare transfer, cause `subject_not_evidenced`. *One transaction* is
unestablished when a subject document and its payment both exist and fail a cross-check, causes
`amount_mismatch` and `payment_precedes_subject` (see [Evidence](#evidence)). Those two slots are the whole of
the verdict, and which one failed is read off the claim's own documents.

---

## Evidence

A claim is a list of documents, and the list has to resolve into the money the claim is about before any
verdict can be derived from it. That resolution is read off `document_evidence` in `config/policy.yaml` —
what each document *type* proves — and off nothing else.

### A claim's amount is not the sum of its documents

The dominant pair is an invoice plus the payment confirmation that settles it, and those are one movement of
money described twice. Adding them counts it twice, and the doubled figure would travel: `covered_fraction`,
`reimbursable_amount` and the cumulative limit would all be computed from it, the verdict would still come
out plausible, and the balance report would not notice, because it checks the distribution of verdicts and
not the arithmetic of amounts.

So a claim's documents are grouped into **transactions**. A document that proves payment attests to one
movement of money; a document that proves only the subject describes money some payment document already
attests, and adds none of its own. The claim's line items are the line items of its subject documents,
counted once per transaction.

Two shapes are derivable, and only two:

| Shape | Example | Amount |
|---|---|---|
| Self-contained | Several fiscal receipts | Each is its own transaction, and their money **does** add |
| One split pair | An invoice and the transfer that settles it | One transaction, described twice, counted once |

Anything else is refused rather than guessed, because the guess decides the claim's amount. A fiscal receipt
beside an invoice is either the same purchase described twice or two purchases one of which was never paid;
one invoice beside two payments leaves open which payment settles it. A **bank statement** is refused
outright: it genuinely lists several transactions while its label carries one amount for the whole document,
so nothing identifies the row a given claim is about.

### Both facts, or no reimbursement

A claim is reimbursable only when its documents establish *both* what was bought and that it was paid for.
Neither is optional and neither implies the other:

- the money-moved slot fails when every document is of a type that proves no payment — an invoice on its own —
  which is `not_proof_of_payment`;
- the what-was-bought slot fails when no document is of a type that states it — a bare transfer — which is
  `insufficient_evidence`, cause `subject_not_evidenced`.

Each is read from the types the claim carries, independently of the other, so neither answer depends on how the
other is worded.

With a fiscal receipt this is vacuously satisfied, which is exactly why it has to be checked rather than
assumed: the moment a claim can be an invoice on its own, a verdict derived from coverage alone would label
it `covered`.

### Documents that disagree

The two documents of a split pair have to describe *one* transaction. Two cross-checks say whether they do,
and each is its own defect with its own name:

- **`amount_mismatch`** — the subject document and its payment state different amounts;
- **`payment_precedes_subject`** — the payment is dated before the document it settles. Strictly before:
  paying an invoice on the day it is issued is ordinary.

Either makes the claim `insufficient_evidence`. This is the third slot: both other facts are established
separately, and the claim still does not establish that *this* payment paid for *this* subject — a linkage a
claim has to prove. It is a verdict and not a flag beside a coverage verdict, because a flag would let a claim
whose documents contradict each other come out `covered`.

These two causes are the linkage slot; the missing **subject** document is the other slot `insufficient_evidence`
owns. Together they are why the verdict still exists after the period case moved to `rejected`: in all three
something genuinely *is* unestablished.

### The period is checked on the payment

A limit is consumed when money moves, so a claim is dated by its **proof of payment** and the active period
is checked against that date alone. A December invoice paid in January is an ordinary January expense, and
refusing it for its date would be wrong. The subject document's date is checked for *order* instead, which is
the `payment_precedes_subject` defect above — a different question with a different repair.

A payment that does fall outside the window makes the claim `rejected`, cause `outside_period` — the policy
does not cover an expense paid outside its own period, which is a coverage answer and not an evidence one. See
[Verdicts](#verdicts).

---

## Document archetypes

Twenty-four rows below, sixteen Ukrainian and eight European. Layouts follow the publicly observable
conventions of each document class and jurisdiction.

**Rows are not templates one for one.** Row 8 is realized by two templates, one per paper width, so the
twenty-four rows are twenty-five templates. Three of them exist today — `ua_prro_receipt`,
`ua_prro_receipt_58mm` and `ua_rro_receipt`, the whole of the `fiscal_receipt` class for Ukraine.

The three share one body, `templates/ua_fiscal_receipt.jinja`, and differ in the paper width and in the fiscal
identity the register prints. That is deliberate and it is also a limit: real registers differ in layout by
provider, and no open sample shows the layout of a hardware receipt, so a layout difference invented between
them would be a guess on every image. What varies is what is evidenced.

### Ukraine

| # | Template | Class | Role |
|---|---|---|---|
| 1–4 | Bank in-app payment confirmations (four issuers) | `payment_confirmation` | Mainstream bank layouts |
| 5–7 | Payment-service receipts (three providers) | `payment_confirmation` | Payment-gateway layouts |
| 8 | Software cash register receipt (ПРРО) — QR, `ФН ПРРО`, VAT letters; **two templates, 80 mm and 58 mm** | `fiscal_receipt` | Modern fiscal document |
| 9 | Classic hardware cash register receipt (РРО), 80 mm — `ЗН` beside `ФН`, sequential number | `fiscal_receipt` | Register diversity |
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
| Payment outside the active period | Money moved before or after the window | `rejected` |
| Document does not prove payment | Invoice marked "paid: 0"; sales slip; booking confirmation | `not_proof_of_payment` |
| Paid in installments | Part of the amount settled | `partially_paid` |
| Evidence incomplete | No statement of what was bought — a bare transfer | `insufficient_evidence` |
| Paid through an aggregator | Payee is a payment intermediary, no visible link to the merchant | requires linking |
| Documents disagree | Payment amount differs from the contract; payment predates the contract | `insufficient_evidence` |
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
`claim_planner` can select it, with the document class it belongs to; what it proves follows from that class
through `document_evidence` in `config/policy.yaml`, and there is no per-archetype override — a template
whose evidence differs from its class must not be registered until a document's role is carried in the label
rather than derived from its type. Add a builder for it, keyed by slug in `assembler`.

Where a new archetype differs from an existing one only in what is already data — the paper width, which
requisites the device prints — share the body rather than copying it: the three Ukrainian fiscal receipts
include one `.jinja` fragment and carry only their own `<slug>.css`. Copying a template makes the evidence
comments behind its layout facts two sources of truth, and the copy is the one that goes stale. The test suite
follows includes when it checks a template, so a guard reading `<slug>.html` alone cannot quietly become
vacuous.

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
