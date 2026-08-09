# RP-07 — the delivered corpus, as a release record

What was shipped to a consumer, in the form a consumer needs to prove that the directory in hand is the
directory this repository produced. `docs/balance-report.md` says what the corpus *contains*; this file says
what the corpus *is* — one command, one seed, one contract version, three digests.

Written 2026-08-06, after the corpus was consumed by a paid measurement run in
`reimbursement-validation`. Nothing here is new evidence: every figure is either the generator's own output
or a digest recomputed over the shipped directory on that date.

## The run

| | |
| --- | --- |
| Command | `generate-dataset --seed 42 --personas 96 --claims-per-persona 8 --split 0.5` |
| Seed | 42 |
| Generator | receipt-synth 0.1.0 |
| Contract | `config/labelling-schema.yaml` **version 38** |
| Policy | `config/policy.yaml`, `policy_version: 1` |
| Jurisdiction | UA (the default; no other is built) |
| Shipped to | `~/Documents/Projects/Personal/receipt-synth-datasets/v38-prod` (outside this repository) |

**The command is RP-06's, verbatim.** Seed, persona count, claims per persona and split fraction were
deliberately not refreshed, so every shift against that run is attributable to the generator and to nothing
else.

**Version 38 is asserted, not tolerated.** `packages/schemas/tests/test_labelling_contract.py:618` pins
`schema.version == 38`; a pre-38 directory cannot be proved to be this one.

### Seeds a consumer must be able to state

| Seed | Value | What it fixes |
| --- | --- | --- |
| `dataset_seed` | **42** | the corpus itself. Must match between any two artifacts being compared, or they describe different documents |
| generator anchor | none since v38 | 🔴 before version 38 the persona birth date came from `Faker.date_of_birth`, whose anchor is `datetime.now()`. Two runs of a byte-identical command two hours apart across midnight moved 36 of 96 personas' `tax_id` and 352 of 1141 images. **RP-07 is the first corpus of this repository that a seed reproduces**; no digest recorded against an earlier profile is reproducible by anybody |

## Size, with the denominator on every figure

| | |
| --- | --- |
| Personas | **96** |
| Claims | **750 of 768 ordered** (96 × 8) |
| Documents | **1241** |
| Image files | **1122** — 1003 carry one document, 119 carry two documents of one claim |
| Label files | **1991** — 1241 document labels + 750 claim labels |
| Documents per claim | 1.65 |

The 18 claims ordered and not built ran out of annual balance: every documentable category of that persona
had its limit spent. That is the limit mechanism working, not a shortfall.

**1122 images is fewer than 1241 documents and nothing is missing.** A reader who expects one file per
document will conclude 119 documents were lost.

### Partition

| Side | Personas | Claims | Documents | Share of claims |
| --- | --- | --- | --- | --- |
| train | 48 | 373 | 631 | 49.7% |
| validation | 48 | **377** | **610** | 50.3% |

By **persona**, 50/50, **not stratified**. The realized claim share differs from the requested 50% of
*personas* because personas do not carry equal claim counts. Every claim and every document carries a side.

### Composition, corpus row (denominator 1241 documents)

| class | documents | share |
| --- | --- | --- |
| `invoice` | 523 | 42.1% |
| `bank_statement` | 261 | 21.0% |
| `payment_confirmation` | 247 | 19.9% |
| `fiscal_receipt` | 111 | 8.9% |
| `platform_receipt` | 59 | 4.8% |
| `non_fiscal_receipt` | 40 | 3.2% |

### Composition, validation row (denominator 610 documents)

The row that decides whether the corpus is usable, because nothing is measured on the corpus as a whole.

| class | documents | target class | ≥ 30 |
| --- | --- | --- | --- |
| `invoice` | 249 | yes | yes |
| `bank_statement` | 128 | yes | yes |
| `payment_confirmation` | 116 | yes | yes |
| `fiscal_receipt` | 63 | yes | yes |
| `platform_receipt` | 33 | no | yes |
| `non_fiscal_receipt` | 21 | no | **no** |

### Verdicts (denominator 750 claims; validation column added for the consumer)

| verdict | corpus | validation | target share |
| --- | --- | --- | --- |
| `partially_covered` | 231 | 111 | 20.0% |
| `covered` | 230 | 117 | 40.0% |
| `rejected` | 88 | 44 | 10.0% |
| `not_proof_of_payment` | 72 | 37 | 10.0% |
| `partially_paid` | 66 | 34 | 10.0% |
| `insufficient_evidence` | 63 | 34 | 10.0% |

First delivered corpus containing **every** member of `verdict_mix`; the shares are the dataset's balance
against it rather than shares conditional on a realizable subset.

### Categories (denominator 750 claims)

| category | annual limit | corpus | validation |
| --- | --- | --- | --- |
| `medical_insurance` | 16000 | 170 | 88 |
| `vitamins_nutrition` | 12000 | 125 | 72 |
| `mental_health` | 25000 | 116 | 48 |
| `hobby` | 10000 | 101 | 52 |
| `sport` | 12000 | 95 | 42 |
| `language_courses` | 12000 | 72 | 36 |
| `professional_development` | 16000 | 71 | 39 |

🔴 **All 33 validation `platform_receipt` documents belong to `professional_development`** — the category is
proven by that class and by no other on this side (39 documents = 33 platform receipts + 3 invoices +
3 non-fiscal receipts). A consumer whose target taxonomy excludes `platform_receipt` therefore loses that
category entirely, not partially. Measured downstream: 6 of 39 claims decided.

## Checking a corpus against this record

🔴 **TWO DIGEST CONVENTIONS ARE IN CIRCULATION FOR THE SAME DIRECTORIES, AND BOTH ARE CORRECT.** The
manifest digest is unambiguous — one file, one SHA-256. The two tree digests are not: `docs/balance-report.md`
and the consumer's run record hash the same bytes under different rules and therefore print different
values. Both were recomputed over the shipped directory on 2026-08-06 and both reproduce. Stating one
without naming its rule invites a false mismatch.

```text
manifest_sha256   e1ed5058b607bcc19b629317e64962bfd04923fba93af1ecc41791e4fd9516e6
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
images/   6a63bfae609351c64bcdc58601427432038b87262624b1ee270e654d6b8fad4f   1122 files
labels/   78f26ffd26dddb9beb2d04fd81bede2d63a065f2ad77b2120c50ff2201367d33   1991 files
```

**Convention B — the consumer's run record**: SHA-256 over the *text* of a `shasum -a 256` listing, paths
sorted with `LC_ALL=C`.

```sh
find images -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256
```

```text
images/   87fa2cdacd90ff1360148211d3dcdbeb8cfd5ef3128f0d3b68d9b6e79b801b3d   1122 files
labels/   9d88fc23e521049ef16738b9ba5b7ea8742aa680e3ac09559ebcc1a479987233   1991 files
```

1991 labels is 1241 document labels plus 750 claim labels. A mismatch on the manifest, or on either tree
digest **under its own convention**, means the directory in hand is a different run and nothing on this page
applies to it.

## What this corpus is not

Unchanged from `docs/balance-report.md`, repeated because a release record is what a consumer reads first:

- not balanced across document classes, and nothing tried to make it so;
- `non_fiscal_receipt` is below the per-class minimum on validation and `platform_receipt` on train —
  neither is a classifier target class, and no per-class figure may be quoted for them there;
- contains no `act` and no `order_screenshot` at any size;
- exercises currency and language on 3.1% of its documents (39 `eu_platform_receipt`, EUR/`en`) — that
  exercises the axis and does not populate it;
- the partition is not stratified, so every class landing above 30 on validation is an outcome of this run
  and not a property guaranteed at another size or seed;
- every artifact is synthetic and is not valid proof of payment.
