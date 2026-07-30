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

Early development — the pipeline was built end to end on a single archetype and the remaining templates are
being added a document class at a time. The interface below describes the target CLI; see
[docs/architecture.md](docs/architecture.md) for the full design.

| Component | State |
|---|---|
| Configuration model — policy, fiscal rules, FX, vendors | in place |
| Fonts, dependency pinning, reproducible environment | in place |
| `persona_generator` → `claim_planner` → `content_builder` → `renderer` → `degrader` → `assembler` | in progress |
| 24 document archetypes | 3 templates, covering 2 of the 24 rows — the Ukrainian `fiscal_receipt` class |
| Invariant test suite | in progress |

The three are a software cash register (ПРРО) on 80 mm and on 58 mm paper, and a classic hardware register
(РРО), which prints a different set of fiscal requisites. One archetype row is two templates because the paper
width is one of them; [docs/architecture.md](docs/architecture.md#document-archetypes) has the catalogue.

## Install

Requires Python 3.12.

```bash
make install    # uv sync + the Chromium build the renderer needs
```

Before calling any change done, run the project's gate — lint, tests and the redaction check, in that
order:

```bash
make check
```

`make` on its own lists the targets.

## Usage

```bash
uv run generate-dataset --seed 42 --personas 30 --claims-per-persona 5 --out out/
```

Every flag:

| Flag | Default | What it does |
|---|---|---|
| `--seed` | *required* | Determines the whole run. Required rather than defaulted: a dataset is reproducible only if the number that produced it is one the caller wrote down. |
| `--out` | `out` | Output directory. |
| `--personas` | `1` | How many synthetic people to generate. |
| `--claims-per-persona` | `1` | An **upper bound**. Planning stops early once a persona has no category with an annual balance left, which is also the only way the cumulative-limit mechanism is exercised. |
| `--split` | `0.85` | Fraction of **personas** assigned to train. The partition is by persona because annual limits are cumulative per persona; it is **not** stratified. |
| `--country` | `UA` | Jurisdiction whose fiscal rules apply. |
| `--version` | — | Print the version and exit. |

The configuration is read from `config/` in the repository — there is **no `--config` flag**, and none is
planned: the generator and its policy are versioned together, and pointing the two at different revisions is
the failure the labelling contract exists to prevent.

Output:

```text
out/images/<doc_id>.png       rendered documents, degraded as they would have been captured
out/labels/<doc_id>.json      document-level ground truth
out/labels/<claim_id>.claim.json   claim-level ground truth
out/ground_truth.json         the full index: every persona, claim and document, plus the partition
```

The **balance report is printed to stdout**, not written to a file — redirect it if you want to keep it. It
covers verdicts against `verdict_mix`, imperfection causes, document classes against the per-class minimum,
currency, language, capture channels with their completeness subsets, and the train/validation partition.
Every figure carries its denominator, and a dimension with no data is reported as **absent** rather than as
zero.

[`docs/balance-report.md`](docs/balance-report.md) is that report for the **production corpus**, written for a
reader rather than for a terminal: what the run produced, which per-class figures may be quoted and which may
not, and what the corpus is not. Its machine-readable twin is profile `RP-05` under `run_profiles` in
`config/labelling-schema.yaml` — the one profile there marked authoritative, because it is the corpus a
consumer receives. The others are examples of how a quantity behaves, and the file says so where the numbers
are.

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
| `config/labelling-schema.yaml` | The contract: what is labelled, how each value is compared, and the dataset's known limitations. Read by consumers, not by the generator. |
| `config/fiscal-rules.yaml` | VAT rates and letter codes, identifier formats and checksums, receipt layout constants, fiscal QR payloads per jurisdiction. Law, not policy. |
| `config/generation.yaml` | The vocabulary that fills the placeholders of a line-item name template, retail price ranges, basket shape, mixed-basket coverage targets. Generation input — not a label and not law. |
| `config/fx-rates.yaml` | Static exchange rates. Static by design — a live rates API would break determinism. |
| `config/vendors.json` | Vendor, bank and payment-provider names per category and jurisdiction, and which item kinds each sort of outlet sells. |

Nothing in the pipeline hardcodes category names, limits or thresholds; replacing `config/policy.yaml`
retargets the generator at a different plan.

## Extending

- **A document archetype** — add `templates/<slug>.html` and `.css`, mark extractable fields with
  `data-field="<name>"`, and register the archetype with its document class.
- **A benefit category** — add an entry to `config/policy.yaml` with `intent`, an annual limit and the three
  item buckets; prices and placeholder vocabulary in `config/generation.yaml`; vendors in
  `config/vendors.json`.
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
config/                 labelling policy, labelling contract, fiscal rules, generation input,
                        exchange rates, vendors
templates/              document archetypes (HTML + CSS), plus the .jinja bodies archetypes share
fonts/                  redistributable fonts (OFL) + license texts
src/receipt_synth/      the generator
tests/                  invariant tests (VAT, amount in words, tax ids, date ↔ verdict)
docs/architecture.md    design reference
out/                    generated dataset (gitignored)
```

## Attribution and license

Built as the dataset-engineering component of a Woolf MSc Computer Science (Software Engineering) thesis.

MIT — see [LICENSE](LICENSE). Bundled fonts keep their own licenses, included under `fonts/LICENSES/`.
