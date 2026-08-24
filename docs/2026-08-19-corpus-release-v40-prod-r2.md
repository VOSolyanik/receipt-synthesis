# v40-prod-r2 — the delivered corpus, as a release record

What was shipped to a consumer, in the form a consumer needs to prove that the directory in hand is the
directory this repository produced. `docs/balance-report.md` says what a corpus *contains*; this file says
what this corpus *is* — one command, one seed, one contract version, three digests.

Written 2026-08-24, after the corpus was consumed by the paid final measurement run in
`reimbursement-validation` on 2026-08-20. Nothing here is new evidence: every figure is either the
generator's own stdout or a digest recomputed over the shipped directory on the date of writing.

**This corpus supersedes `v38-prod` / RP-07 as the one the reported figures were measured on.** The earlier
release record, `docs/2026-08-06-corpus-release-v38-prod.md`, stays as it is — its numbers were correct when
taken and describe a directory that still exists.

## The run

| | |
| --- | --- |
| Command | `generate-dataset --seed 116 --personas 96 --claims-per-persona 8 --split 0.5` |
| Seed | 116 |
| Generator | receipt-synth 0.1.0, commit **`83aa45d`** (2026-08-19 12:08:08 +0300) |
| Contract | `config/labelling-schema.yaml` **version 40** |
| Policy | `config/policy.yaml`, `policy_version: 1` |
| Jurisdiction | UA (the default; no other is built) |
| Generated | 2026-08-19 19:14 |
| Shipped to | `$RV_DATASETS/v40-prod-r2` (outside this repository) |

🔴 **THE SEED ALONE DOES NOT REPRODUCE THIS CORPUS.** `--personas`, `--claims-per-persona` and `--split`
are part of the identity of the run: they set how many claims are ordered (96 × 8 = 768) and how the
partition falls. A `--seed 116` run at any other persona count or split fraction produces a different
directory that will not match a digest below. State all four flags or state none of them.

**The command is RP-06's and RP-07's, verbatim except for the seed.** Persona count, claims per persona and
split fraction were deliberately not refreshed, so every shift against those runs is attributable to the
seed and to the generator, and to nothing else.

**Version 40 is asserted, not tolerated.** `packages/schemas/tests/test_labelling_contract.py:639` in
`reimbursement-validation` pins `schema.version == 40`; a directory built at another contract version cannot
be proved to be this one.

**The generator did not move between generation and measurement.** `83aa45d` is the commit the corpus was
built at, it is still `main` at the time of writing, and the measurement run that consumed the corpus began
2026-08-20 — so no generator change sits between the corpus and the numbers taken on it.

### Seeds a consumer must be able to state

| Seed | Value | What it fixes |
| --- | --- | --- |
| `dataset_seed` | **116** | the corpus itself. Must match between any two artifacts being compared, or they describe different documents |
| `execution_seed` | **not reported** | the consumer's run, not this corpus. Absent from every scorecard taken on it |
| generator anchor | none since v38 | before contract version 38 the persona birth date came from `Faker.date_of_birth`, anchored on `datetime.now()`. This corpus is well past that, so a seed reproduces it |

## Size, with the denominator on every figure

| | |
| --- | --- |
| Personas | **96** |
| Claims | **758 of 768 ordered** (96 × 8) |
| Documents | **1272** |
| Image files | **1144** — 1016 carry one document, 128 carry two documents of one claim |
| Label files | **2030** — 1272 document labels + 758 claim labels |
| Documents per claim | 1.678 |
| Pages | **1310** — `page_count` is 2 on 38 documents |
| Labelled boxes | **36 854** |

The 10 claims ordered and not built ran out of annual balance: every documentable category of that persona
had its limit spent. That is the limit mechanism working, not a shortfall.

**1144 images is fewer than 1272 documents and nothing is missing.** A reader who expects one file per
document will conclude 128 documents were lost. There are four counts here and they are all different —
1144 files, 1272 documents, 1310 pages, and 1400 render calls (1272 documents + 128 bundle compositions).
⛔ Only the first two describe the corpus; the render count belongs to the pixel gate.

### Partition

| Side | Personas | Claims | Documents | Image files | Share of claims |
| --- | --- | ---: | ---: | ---: | --- |
| train | 48 | 383 | 643 | 579 | 50.5% |
| validation | 48 | **375** | **629** | **565** | 49.5% |

By **persona**, 50/50, **not stratified**. The unit is the persona because annual limits are cumulative per
persona, so a claim's verdict can depend on that persona's earlier claims; a per-claim partition would make a
validation label a function of training data. The corpus states this itself at `split.why_this_unit`.

🔴 **The validation row is the denominator of every reported figure.** Nothing is measured on the corpus as a
whole. A figure quoted against 758 or 1272 is quoted against the wrong number.

### Composition, corpus row (denominator 1272 documents)

| class | documents | share | train | validation |
| --- | ---: | --- | ---: | ---: |
| `invoice` | 571 | 44.9% | 296 | 275 |
| `payment_confirmation` | 279 | 21.9% | 138 | 141 |
| `bank_statement` | 248 | 19.5% | 130 | 118 |
| `fiscal_receipt` | 71 | 5.6% | **26** | 45 |
| `platform_receipt` | 67 | 5.3% | 37 | 30 |
| `non_fiscal_receipt` | 36 | 2.8% | **16** | **20** |
| `act` | 0 | — | — | — |
| `order_screenshot` | 0 | — | — | — |

**Per-class minimum of 30, checked on each side.** The four target classes clear it on validation —
**4 of 4** — which is what made this seed acceptable. `fiscal_receipt` is below 30 on *train* and
`non_fiscal_receipt` on both sides; ⛔ no per-class figure may be quoted for them there. `act` and
`order_screenshot` are ABSENT because no archetype builds them, so the minimum does not apply.

### Capture channels (denominator 1272 documents)

| channel | corpus | share | validation | complete |
| --- | ---: | --- | ---: | --- |
| screenshot | 338 | 26.6% | 154 | 338/338 |
| digital_pdf | 324 | 25.5% | 149 | 324/324 |
| photo | 323 | 25.4% | **172** | 304/323 |
| scan | 287 | 22.6% | 154 | 287/287 |

⚠️ **The split's channel order is not the corpus's.** On validation the commonest channel is **photo**
(172), while on the corpus it is screenshot (338) and photo is third. The partition is by persona, so
channels are not stratified. Quoting a channel distribution beside a metric means quoting the validation
column.

### Currency and language (denominator 1272 documents)

| | | |
| --- | ---: | --- |
| UAH | 1189 | 93.5% |
| EUR | **83** | 6.5% |
| uk | 1211 | 95.2% |
| en | **61** | 4.8% |

Three of the thirteen archetypes are stated in euro. This exercises the axis; it does not populate it.

### Verdicts (denominator 758 claims; validation column added for the consumer)

| verdict | corpus | validation | target share |
| --- | ---: | ---: | --- |
| `partially_covered` | 228 | 118 | 20.0% |
| `covered` | 221 | 110 | 40.0% |
| `not_proof_of_payment` | 93 | 41 | 10.0% |
| `insufficient_evidence` | 75 | 39 | 10.0% |
| `rejected` | 74 | 36 | 10.0% |
| `partially_paid` | 67 | 31 | 10.0% |

`covered` lands under its target because **102 claims drawn as `covered` were moved to
`partially_covered` by a binding annual limit** — the oracle overruling the target, which is the limit
mechanism doing its job rather than a balance failure. In the other direction 22 claims drawn as
`partially_covered` came out `covered`, a builder shortfall rather than a policy event.

### Categories (denominator 758 claims)

| category | corpus | validation |
| --- | ---: | ---: |
| `medical_insurance` | 172 | 78 |
| `mental_health` | 136 | 61 |
| `hobby` | 107 | 43 |
| `sport` | 106 | 64 |
| `professional_development` | 83 | 39 |
| `language_courses` | 79 | 42 |
| `vitamins_nutrition` | 75 | 48 |

🔴 **`professional_development` remains the category a consumer can lose whole.** It is the only category
`platform_receipt` is registered for, both variants of it, and a consumer whose taxonomy excludes that class
loses the category rather than a share of it. This was measured downstream on this corpus: the category's
automation rate came out **0.231** against 0.484–0.837 elsewhere.

## Checking a corpus against this record

🔴 **TWO DIGEST CONVENTIONS ARE IN CIRCULATION FOR THE SAME DIRECTORIES, AND BOTH ARE CORRECT.** The
manifest digest is unambiguous — one file, one SHA-256. The two tree digests are not: this repository and
the consumer's run record hash the same bytes under different rules and print different values. Stating one
without naming its rule invites a false mismatch. All values below were computed over the shipped directory
on 2026-08-24.

```text
manifest_sha256   ecff9b85e073f9ffe54ab1fccb7a0e20042433428968e8a4729dfb28099901dd
                  SHA-256 of ground_truth.json — one file, no convention needed
```

**Convention A — this repository's** (`docs/balance-report.md`): each file contributes its *name* followed
by the SHA-256 of its bytes, sorted by name.

```python
h = hashlib.sha256()
for path in sorted(directory.glob("*"), key=lambda p: p.name):
    h.update(path.name.encode())
    h.update(hashlib.sha256(path.read_bytes()).digest())
```

```text
images/   867f9baa0d3b00252ec7f73ee971b72184788c2e13de44e7c8a6f1f4cdda042f   1144 files
labels/   3019200daa816576cef3718fd07e4e40f92c02f7783f86c362009c0023a75ec0   2030 files
```

**Convention B — the consumer's run record**: SHA-256 over the *text* of a `shasum -a 256` listing, paths
sorted with `LC_ALL=C`.

```sh
find images -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256
```

```text
images/   23c3dd5f11ec441335eaeca95022245f8354451da355b5c6ae06809d6a941b90   1144 files
labels/   e4f74280f3dbf6cef8ab46aaea2722e91b04b2e59f97a140e1c354165ada3f62   2030 files
```

2030 labels is 1272 document labels plus 758 claim labels. A mismatch on the manifest, or on either tree
digest **under its own convention**, means the directory in hand is a different run and nothing on this page
applies to it.

## What this corpus is not

- not balanced across document classes, and nothing tried to make it so;
- `non_fiscal_receipt` is below the per-class minimum on both sides and `fiscal_receipt` on train — neither
  is a classifier target class, and ⛔ no per-class figure may be quoted for them there;
- contains no `act` and no `order_screenshot` at any size;
- exercises currency and language on 6.5% and 4.8% of its documents — that exercises the axes and does not
  populate them;
- the partition is not stratified, so every class landing above 30 on validation is an outcome of this run
  and not a property guaranteed at another size or seed, and the channel mix differs between the sides;
- 67.8% of claims carry more than one document — a **designed parameter of this generator**, ⛔ not a
  measured frequency of any real flow;
- every artifact is synthetic and is not valid proof of payment.
