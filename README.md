# receipt-synth

Generate synthetic document images with ground truth known by construction — fiscal receipts, payment
confirmations, invoices and bank statements across Ukraine, Poland, Germany and Spain.

Because every document is *built to satisfy* a label rather than labelled after the fact, the output carries
exact field values, per-field bounding boxes, per-line coverage flags and a claim-level verdict, with no
annotation step anywhere. Useful for document classification, key information extraction, document-boundary
detection in multi-document files, deduplication, and reimbursement-decision logic.

> **Everything this tool produces is synthetic.** The generated images depict documents that were never
> issued, for transactions that never happened, by people who do not exist. Every ground-truth record is
> stamped `"synthetic": true` with the generator version. These are training and evaluation artifacts. They
> are not, and must not be presented as, proof of payment or any other genuine record.

## Status

Early development — the pipeline is being built end to end on a single archetype before the remaining
templates are added. The interface below describes the target CLI; see [docs/architecture.md](docs/architecture.md)
for the full design.

| Component | State |
|---|---|
| Configuration model — policy, fiscal rules, FX, vendors | in place |
| Fonts, dependency pinning, reproducible environment | in place |
| `persona_generator` → `claim_planner` → `content_builder` → `renderer` → `degrader` → `assembler` | in progress |
| 24 document archetypes | 0 of 24 |
| Invariant test suite | in progress |

## Install

Requires Python 3.12.

```bash
uv sync
uv run playwright install chromium
```

## Usage

```bash
uv run generate-dataset \
  --config config/policy.yaml \
  --personas 30 --seed 42 \
  --categories-per-persona 3 --max-docs-per-category 5 \
  --out out/ --split 0.85
```

`--config` points at the labelling policy; the fiscal rules, exchange rates and vendor catalogue are read
from the same directory unless overridden.

Output:

```text
out/images/<claim>.png        rendered documents (+ degraded variants)
out/labels/<claim>.json       claim-level and document-level ground truth
out/manifest.json             full index + train/val split
out/balance-report.md         distribution over classes, verdicts, currencies, languages
```

Generation is **deterministic under `--seed`**: the repository ships the generator, its configuration and the
seed, not the dataset. Rerunning with the same seed reproduces the same dataset byte for byte.

## What you get

Each rendered document comes with a record like this — field values, the pixel box of every field, and a
coverage flag per line item:

```json
{
  "doc_id": "p001_c2_d1",
  "doc_type": "fiscal_receipt",
  "language": "uk",
  "currency": "UAH",
  "amount": 1250.00,
  "date": "2026-08-03",
  "counterparty": "<vendor>",
  "line_items": [
    { "name": "Абонемент 8 занять", "qty": 1, "price": 1000.00, "covered": true,  "vat_letter": "А" },
    { "name": "Протеїн 900 г",      "qty": 1, "price": 250.00,  "covered": false, "vat_letter": "А" }
  ],
  "field_bboxes": { "amount": [412, 690, 96, 18], "date": [24, 742, 132, 16] },
  "synthetic": true,
  "generator_version": "0.1.0"
}
```

And a claim-level record explaining the verdict:

```json
{
  "claim_id": "p001_c2",
  "verdict": "partially_covered",
  "covered_fraction": 0.80,
  "imperfection": ["mixed_items"],
  "verdict_basis": ["documents"],
  "policy_trace": ["category=sport ok", "period ok", "coverage 80% (sports nutrition excluded)"]
}
```

`verdict_basis` distinguishes verdicts derivable from the images alone from those that also need account
state — an exhausted annual limit, for instance, cannot be read off a receipt. Score document understanding
on the first subset, end-to-end behaviour on both.

## How it works

**Label-first**: the target verdict of a claim is chosen first, and documents are then constructed to realize
it — so the label never has to be inferred from an image.

```text
persona_generator → claim_planner → content_builder → renderer → degrader → assembler
```

| Stage | What it does |
|---|---|
| `persona_generator` | Synthetic people: name, country, currencies, languages, family, valid tax id |
| `claim_planner` | Picks the target verdict and the imperfection mechanism, then the document archetypes that realize them |
| `content_builder` | Fills in valid requisites (tax-id checksums, VAT letters, amount in words = digits) — or breaks them deliberately for fraud and trap archetypes |
| `renderer` | Jinja2 → HTML → Playwright screenshot, plus per-field bounding boxes from `getBoundingClientRect()` |
| `degrader` | Augraphy (paper, ink, shadows) + Albumentations (perspective, blur, JPEG) → screenshot / photo / scan modes |
| `assembler` | Dataset layout, train/val split, balance report |

Full detail, including the ground-truth schemas, the archetype catalogue and the catalogue of imperfection
mechanisms: [docs/architecture.md](docs/architecture.md).

## Configuration

See [config/README.md](config/README.md) for the full model.

| File | Role |
|---|---|
| `config/policy.yaml` | Categories, limits, coverage thresholds, active period. The labelling policy — because generation is label-first, this file *is* the ground truth. Declarative on purpose, so any downstream document-verification pipeline can load the same file and stay consistent with the labels it was trained on. |
| `config/fiscal-rules.yaml` | VAT rates and letter codes, identifier formats and checksums, receipt layout constants, fiscal QR payloads per jurisdiction. Law, not policy. |
| `config/fx-rates.yaml` | Static exchange rates. Static by design — a live rates API would break determinism. |
| `config/vendors.json` | Vendor, bank and payment-provider names per category and jurisdiction. |

Nothing in the pipeline hardcodes category names, limits or thresholds; replacing `config/policy.yaml`
retargets the generator at a different plan.

## Extending

- **A document archetype** — add `templates/<slug>.html` and `.css`, mark extractable fields with
  `data-field="<name>"`, and register the archetype with its document class.
- **A benefit category** — add an entry to `config/policy.yaml` with `intent`, an annual limit and the three
  item buckets, plus vendors in `config/vendors.json`.
- **A jurisdiction** — add a block to `config/fiscal-rules.yaml` (VAT rates and letters, identifier formats
  and checksums, layout constants, fiscal QR payload), matching templates, and a font covering the script.

Details in [docs/architecture.md](docs/architecture.md#extending-the-generator).

## Provenance

Real reimbursement documents belong to the people and businesses that issued and received them, and cannot be
collected or labelled for machine learning without their consent. This generator is built **clean-room** from
public sources only: published fiscal law and document formats, general knowledge of document layouts,
published industry surveys, `Faker`, publicly known vendor names, and the author's own receipts. No real
document belonging to any third party is used as a source anywhere in this project.

Benefit categories are derived from a [public industry survey of benefits packages in the Ukrainian IT
market](https://dou.ua/lenta/articles/benefits-package-2026/) (DOU, ~10.6k respondents), selected for document
diversity: each category owns a document type or an imperfection mechanism the others do not produce. Limits
and thresholds are this project's own plausible values and describe no real employer's plan.

## Repository layout

```text
config/                 labelling policy, fiscal rules, exchange rates, vendors
templates/              document archetypes (HTML + CSS)
fonts/                  redistributable fonts (OFL) + license texts
src/receipt_synth/      the generator
tests/                  invariant tests (VAT, amount in words, tax ids, date ↔ verdict)
docs/architecture.md    design reference
out/                    generated dataset (gitignored)
```

## Attribution and license

Built as the dataset-engineering component of a Woolf MSc Computer Science (Software Engineering) thesis.

MIT — see [LICENSE](LICENSE). Bundled fonts keep their own licenses, included under `fonts/LICENSES/`.
