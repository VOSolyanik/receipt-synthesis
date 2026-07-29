# receipt-synth — project instructions

## What this repo is

A generator of synthetic document images — fiscal receipts, payment confirmations, invoices, bank statements
across UA/PL/DE/ES — whose ground truth is known by construction, for training and evaluating
document-understanding and reimbursement-verification systems.

- **Standalone tool.** This repository contains only the generator, its configuration and its seed. It has no
  dependency on any consumer, and its public documentation must not assume one.
- License: MIT. Private during development, **public later** — write every file as if it were already public.
- Public design reference: [docs/architecture.md](../docs/architecture.md). Keep it accurate; it is what a
  stranger who clones this repo reads.

## HARD RULE — clean-room provenance

The generator is built **exclusively** from public sources:

- published fiscal law and document formats (Ukrainian cash-register law, the fiscal formats of PL/DE/ES);
- general knowledge of document layouts;
- published industry surveys (e.g. the survey cited in `config/policy.yaml`);
- `Faker`;
- LLM-generated **public** vendor lists;
- **the author's own receipts.**

**Never** use as a source: real documents belonging to any third party — an employer, a colleague, a client.
No such material exists in this project.

Insider knowledge of a real benefit plan may confirm that a choice is plausible, but it may never *be* the
source. When both are available, cite the public one and derive from it — provenance is what has to hold up,
not the result.

**Test for every structural decision:** *"Can I justify this from public sources plus my own receipts alone?"*
Yes → clean. Otherwise → contaminated; discard it and re-derive.

## Repository hygiene

- **No PII**, no secrets, no API keys — in code, config, commit messages, or documentation.
- **No real images of anyone's documents** are committed. Ever.
- Any real receipts used as a layout reference stay local and gitignored. They are PII.
- After any session that reads real receipts, scan everything about to be committed for identifiers carried
  over into comments or examples — receipt numbers, tax ids, RRNs, authorization codes, terminal ids, masked
  cards, transaction dates, names. Layout facts are the only thing that may leave such a source; concrete
  printed values must be replaced with invented ones.
- Generated dataset output is **not** committed — the repo ships the generator, config and seed, so that
  `seed → dataset` is reproducible. Rendered images and labels live in `out/` (gitignored).
- Brand names appear in templates for layout fidelity. To keep intent unambiguous, every ground-truth record
  carries `"synthetic": true` plus the generator version, and the README states plainly that the artifacts
  are synthetic and are not valid proof of payment.
- Public docs describe **this tool only**. No references to a specific downstream consumer, to an internal
  planning document, or to any write-up this work is part of, beyond the one-line attribution in the README.

## Architecture — label-first

The target verdict of a claim is chosen **first**; documents are then built to realize it. Ground truth is
therefore known by construction — no manual annotation. Do not add any step that infers labels from rendered
output; that inverts the whole design.

Pipeline, each stage a module with a clean interface:

```text
persona_generator → claim_planner → content_builder → renderer → degrader → assembler
```

Configuration is split by nature, and the split must be preserved:

- `config/policy.yaml` — **policy.** Because generation is label-first, this file *is* the ground truth.
  Declarative so that downstream consumers can load the same file and stay consistent with the labels.
- `config/fiscal-rules.yaml` — **law.** VAT letters, identifier formats, receipt layout constants.
- `config/fx-rates.yaml` — **reference data.** Static, for determinism.
- `config/vendors.json` — **data.**

Generation is **deterministic under `--seed`**. Anything that introduces unseeded randomness is a bug.

## Repository structure

```text
README.md
pyproject.toml                  # pinned deps
config/
  policy.yaml                   # categories, limits, coverage %, period — the labelling policy
  fiscal-rules.yaml             # VAT letters, identifier formats, receipt layout per jurisdiction — law
  fx-rates.yaml                 # static exchange rates — reference data, static for determinism
  vendors.json                  # public vendor lists per category and jurisdiction
  generation.yaml               # placeholder vocabulary, price ranges, basket draw inputs.
                                # Merchandise, not plan — nothing here reaches a label
  labelling-schema.yaml         # the contract: field names, types, comparison rules.
                                # NOTHING IN src/ READS IT — consumed downstream, and a
                                # loader here would be inventing a caller
  README.md                     # the configuration model
templates/                      # 24 archetypes: <slug>.html + <slug>.css
fonts/                          # redistributable fonts only (+ fonts/LICENSES/)
src/receipt_synth/
  config.py                     # loaders for the config files the generator itself reads —
                                # policy, fiscal rules, fx, vendors, generation. NOT the
                                # labelling contract. Read by stages at both ends of the
                                # pipeline, so it belongs to neither
  persona_generator.py
  claim_planner.py
  policy_engine.py              # the oracle: derives a claim's verdict from policy.yaml.
                                # Independent of any consumer's engine by design (Ф-107)
  content_builder.py            # + invariant validators
  renderer.py                   # Jinja2 + Playwright + bbox
  degrader.py                   # Augraphy + Albumentations
  assembler.py
  schemas.py                    # pydantic: persona / doc / claim
  cli.py
tests/                          # invariant unit tests (VAT, amount-in-words, tax id, date ↔ verdict)
docs/architecture.md            # public design reference
out/                            # gitignored — generated dataset
tasks/                          # gitignored — local working notes (todo.md, lessons.md)
```

## Stack (pinned)

| Layer | Tool | Pin / note |
|---|---|---|
| Language | Python | 3.12 |
| Persona data | Faker | locale `uk_UA` and others per jurisdiction |
| Templates | Jinja2 | HTML/CSS per archetype |
| Render | Playwright | Chromium; bboxes via `getBoundingClientRect()` |
| Multi-page PDF | ReportLab | only for bank statements, when needed |
| QR | segno | fiscal QRs + decorative ones (classification trap) |
| Degradation (paper) | Augraphy | — |
| Degradation (geometry) | **albumentations==2.0.8** | **MIT pin.** Do **not** switch to AlbumentationsX (AGPL) |
| Validation | pydantic | v2 |
| Fonts | vendored in `fonts/` | redistributable licenses only (OFL), license text committed |

The `albumentations==2.0.8` MIT pin is a licensing decision, not a preference: AGPL would become ambiguous if
a downstream consumer ever ships a modified library. Never relax this pin without an explicit decision.

## Code conventions

- **Code, comments, docstrings, commit messages, log output, public docs: English.**
- Domain terms keep their original form with a short gloss on first use, e.g.
  `vat_letter  # ПДВ-літера: fiscal VAT rate code printed on Ukrainian receipts (А = 20%, В = 7%)`.
- Rendered document content is of course in the document's own language (uk / pl / de / es / en) — that is
  data, not code.
- Prefer composition over inheritance; explicit over clever. Match the surrounding code.
- Small, reversible steps; one concern per commit; conventional commits (`type(scope): subject`).
- Invariant validators live next to `content_builder` and are covered by tests: tax-id checksum, VAT letter ↔
  rate, amount-in-words = numeric amount, line-item sum = total, date inside/outside period ↔ verdict.
  Fraud and trap archetypes break these **deliberately** — a broken invariant must be an explicit, labelled
  choice, never an accident.
- Run `make check` before calling anything done — lint, tests and the redaction gate, in that order. That
  one command is what "the checks are green" means here; do not substitute your own invocation.

## Notes

- `tasks/lessons.md` records corrections so they are not repeated — read it at session start, append after
  every correction. Local only, gitignored.
- `tasks/todo.md` holds the current plan as checkable items. Local only, gitignored.
