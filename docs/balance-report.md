# Balance report — the production corpus

What a run of this generator actually produced, written down so that a figure computed on the corpus
can be read with its denominator beside it. The generator prints this report to stdout; this file is
the copy that gets read by a person.

The machine-readable form of everything below is `run_profiles` profile **RP-07** in
`config/labelling-schema.yaml`. That profile is the authoritative one: it describes the corpus a
consumer receives. **RP-06 was authoritative one contract version ago and is not any more** — it
records a run made at contract version 20, before three verdicts, three `insufficient_evidence`
causes, two document classes, a fourth capture channel, a second currency and multi-document files
existed, and it is demoted in writing at its own `superseded_by` key rather than quietly. Its
numbers were correct when taken and are correct now. Everything earlier in that file is an example
of how a quantity behaves.

## The run

| | |
| --- | --- |
| Command | `generate-dataset --seed 42 --personas 96 --claims-per-persona 8 --split 0.5` |
| Seed | 42 |
| Generator | receipt-synth 0.1.0 |
| Contract | `config/labelling-schema.yaml` version 38 |
| Jurisdiction | UA (the default; no other is built) |
| Size | 96 personas · 750 of 768 claims built · 1241 documents in **1122 files** |
| Partition | by persona, 50/50, not stratified |
| Dataset location | outside this repository — the repo ships the generator, its configuration and the seed |

**The command is RP-06's, verbatim.** The seed, the persona count, the claims per persona and the
fraction were deliberately not refreshed, so that every shift against that run is attributable to
the generator and to nothing else.

The 18 claims that were ordered and not built ran out of annual balance: every documentable category
of that persona had its limit spent. That is the limit mechanism working, not a failure.

**More claims and fewer documents than RP-06 — 750 against 728, on 1241 documents against 1315 — and
the two move for the same reason.** Three of the verdicts added since that run reimburse nothing, so
an annual limit binds far later: **461 of these 750 claims consume balance**, where all but 74 of
RP-06's 728 did, and only 18 claims went unbuilt against 40. The same three verdicts are largely
single-document claims, so the corpus grew in claims and shrank in documents: 1.65 documents per
claim against 1.81.

### Checking a corpus against this report

```text
manifest_sha256   e1ed5058b607bcc19b629317e64962bfd04923fba93af1ecc41791e4fd9516e6
images_sha256     6a63bfae609351c64bcdc58601427432038b87262624b1ee270e654d6b8fad4f   1122 files
labels_sha256     78f26ffd26dddb9beb2d04fd81bede2d63a065f2ad77b2120c50ff2201367d33   1991 files
```

`manifest_sha256` is the SHA-256 of `ground_truth.json`. The other two are a SHA-256 over the files of
a directory, each contributing its name followed by the SHA-256 of its bytes, in sorted order by name:

```python
h = hashlib.sha256()
for path in sorted(directory.glob("*")):
    h.update(path.name.encode())
    h.update(hashlib.sha256(path.read_bytes()).digest())
```

1991 labels is 1241 document labels plus 750 claim labels. **1122 images is fewer than 1241
documents**, and that is not a shortfall: 119 files carry both documents of one claim. A mismatch on
any of the three means the directory in hand is a different run, and nothing below applies to it.

> **This is the first corpus of this repository that a seed reproduces.** Until version 38 the
> persona's birth date — which the РНОКПП encodes, which an invoice prints as the buyer's code — was
> drawn with `Faker.date_of_birth`, whose offset comes from the seed and whose **anchor comes from
> `datetime.now()`**. Two runs of a byte-identical command two hours apart across midnight moved 36
> of 96 personas' `tax_id` and 352 of 1141 images. No digest recorded against an earlier profile can
> be reproduced by anybody; see `run_profiles.a_digest_taken_before_version_38`.

## The threshold, checked where it is actually binding: the validation side

The minimum is **≥ 30 documents per buildable class**. Below 30, the 95% Wilson interval around an
accuracy near 0.9 is about ±0.10, so "0.91" and "0.85" are indistinguishable. Because macro-F1 weights
every class equally, a thin class swings the mean.

**The corpus row was never the constraint.** Nothing is measured on the corpus as a whole: a consumer
inspects documents on the development side and reports figures on the measurement side. So the row
that decides whether this corpus is usable is the validation one. From version 38 the generator's own
report checks it there — until then it checked the corpus row, and printed `ok` for a class that
carries 21 documents on validation.

**All four target classes clear 30 on validation. Two non-target classes do not.** Denominators: 631
train documents, 610 validation.

| class | target class | train | validation | ≥ 30 on validation |
| --- | --- | ---: | ---: | --- |
| `invoice` | yes | 274 | **249** | yes |
| `bank_statement` | yes | 133 | **128** | yes |
| `payment_confirmation` | yes | 131 | **116** | yes |
| `fiscal_receipt` | yes | 48 | **63** | yes |
| `platform_receipt` | no | 26 | **33** | yes — but **below 30 on train** |
| `non_fiscal_receipt` | no | 19 | **21** | **no** |

`classifier_metric.target_classes` names the first four, and macro-F1 is scored over those. So the
corpus is usable for the measurement it exists for.

**What is blocked, stated plainly:** no per-class figure may be quoted for `non_fiscal_receipt` on
either side, and none for `platform_receipt` on train. Nothing was reweighted to close either. The
count of a товарний чек, which only a `not_proof_of_payment` claim prints, is a consequence of
`verdict_mix`; moving it would be choosing a document mix this repository refuses to choose.

`fiscal_receipt` at 63 is the thinnest target class and remains the one that decides how small a
validation side may ever be. **Its margin is thinner than RP-06's** — 70 there, on a corpus where it
was 10.7% rather than 8.9% — because two classes that did not exist then now take documents it used
to carry.

Every verdict clears 30 on validation too, which is an outcome and not a guarantee: **the partition
is not stratified.** `insufficient_evidence` at 29 on *train* is the one cell below it; nothing is
measured on train, so it costs a consumer nothing, and it is stated rather than rounded past.

### Context: the corpus row

Denominator 1241 documents.

| class | documents | share | against RP-06 |
| --- | ---: | ---: | --- |
| `invoice` | 523 | 42.1% | 587 |
| `bank_statement` | 261 | 21.0% | 294 |
| `payment_confirmation` | 247 | 19.9% | 293 |
| `fiscal_receipt` | 111 | 8.9% | 141 |
| `platform_receipt` | 59 | 4.8% | the class did not exist |
| `non_fiscal_receipt` | 40 | 3.2% | no archetype built it |

Every shift on that table has a mechanism:

- **`invoice` −64.** `professional_development` moved off the invoice-plus-payment pair: the platform
  receipt proves both facts on one page and is confined to that category, which drew 71 claims here.
  Partly offset by the 32 bare invoices a `not_proof_of_payment` claim prints.
- **`payment_confirmation` −46 and `bank_statement` −33 — that is −79, and it is exact.** Every
  two-document claim carries exactly one payment document, and two-document claims fell 587 → 491
  (−96); 17 payment documents came back as single-document `subject_not_evidenced` claims. −96 + 17
  = −79.
- **`fiscal_receipt` −30.** The class is the single document for `vitamins_nutrition`, which drew 125
  claims here against 141 there, and inside that category the товарний чек now competes for the
  claims that prove no payment.
- **`platform_receipt` +59, `non_fiscal_receipt` +40.** Both are archetypes registered since RP-06
  (contract versions 32 and 27). Neither is a classifier target class.

`act` and `order_screenshot` have no archetype, so no run of any size contains one. That is absence,
not a count of zero.

The mix is **not balanced and was never tuned to be**. `config/policy.yaml` declares a `verdict_mix`
and deliberately declares no document mix — a share of receipts against invoices would read as an
observation about which documents claimants actually submit, and nothing here has measured that.

## Files are not documents

| | files | documents |
| --- | ---: | ---: |
| one document | 1003 | 1003 |
| two documents of one claim | 119 | 238 |
| total | **1122** | **1241** |

Drawn per eligible claim at `file_composition.bundle_share` 0.25 of the 491 two-document claims,
which predicts 123. Every earlier profile had one file per document, and a reader who expects that
will conclude this run lost 119 documents.

## Why the partition is a half, and not 85/15

**Nothing is trained on this dataset.** 85/15 comes from tasks where a model *learns* on the larger
side, and that side is large because learning consumes examples. The partition here guards against
**fitting the measurement**: a consumer looks into documents, finds where classification or
extraction errs, and adjusts accordingly — and a figure does not count on the documents that were
looked at and tuned against. Inspecting failures needs a few dozen documents.

**A half is not itself derived from that argument.** What sets the floor is the thinnest target
class: a per-class figure needs its 30 documents *on the side it is measured on*, and
`fiscal_receipt` runs at 8.9% here, so a validation side much below a third of the corpus could not
carry one.

| side | personas | claims | documents | share of claims |
| --- | ---: | ---: | ---: | ---: |
| train | 48 | 373 | 631 | 49.7% |
| validation | 48 | 377 | 610 | 50.3% |

The realized claim share differs from the requested 50% of *personas* because personas do not carry
equal claim counts. Every claim and every document carries a side; nothing is unassigned.

The unit is the **persona**: annual limits are cumulative per persona, so a claim labelled
`partially_covered` for `limit_exhausted` carries that label because of that persona's earlier claims.
Partition by claim and a validation label becomes a function of development data.

## Verdicts

Denominator 750 claims built.

| verdict | claims | share | target | drawn as this verdict |
| --- | ---: | ---: | ---: | ---: |
| `partially_covered` | 231 | 30.8% | 20.0% | 149 (19.9%) |
| `covered` | 230 | 30.7% | 40.0% | 312 (41.6%) |
| `rejected` | 88 | 11.7% | 10.0% | — |
| `not_proof_of_payment` | 72 | 9.6% | 10.0% | — |
| `partially_paid` | 66 | 8.8% | 10.0% | — |
| `insufficient_evidence` | 63 | 8.4% | 10.0% | — |

**This is the first delivered corpus containing every member of `verdict_mix`, so these shares are
the dataset's balance against it** rather than shares conditional on a realizable subset. RP-06 held
three of the six, and at least 20% of the declared mix was absent from it. Nothing is absent now, so
there is nothing to report as absent.

**`covered` and `partially_covered` are displaced by the oracle, not by the draw.** 102 claims drawn
as `covered` were labelled `partially_covered` by a binding annual limit, and 20 went the other way
because a basket sized from an estimated line value did not overrun the balance. Undo both and the
drawn populations are 312 and 149 — 41.6% and 19.9% against declared 40% and 20%. The label is
derived, never asserted.

The other four sit within 1.6 points of their declared 10%, which is inside the seed-to-seed spread:
over eight further seeds each of them ranged across about ten points.

## `partially_covered` by cause — two denominators, and the target belongs to the second

`partially_covered_causes` sizes the **draw**. The realized bucket is a different population, because
the oracle moves claims into it and every one of those arrives carrying `limit_exhausted`.

| cause | realized, of 231 | drawn, of 149 | target |
| --- | ---: | ---: | ---: |
| `mixed_items` only | 78 (33.8%) | 95 (63.8%) | 65% |
| `limit_exhausted` only | 136 (58.9%) | 54 (36.2%) | 35% |
| both on one claim | 17 (7.4%) | — | no share declared |

Read the realized column against the target and the generator looks 30 points out. It is not: 102 of
the 231 were drawn as `covered`. A target column beside a share computed on another population reads
as a miss no run of any size can close, which is why both columns are now printed.

## `insufficient_evidence` — the realized distribution of CAUSES

A verdict's share says nothing about which mechanism produced it. Denominator 63 claims; the five
causes are mutually exclusive by construction, and **all five occur** — RP-06 held two.

| cause | claims | share | target |
| --- | ---: | ---: | ---: |
| `subject_not_evidenced` | 17 | 27.0% | 20% |
| `counterparty_mismatch` | 13 | 20.6% | 20% |
| `amount_mismatch` | 11 | 17.5% | 20% |
| `payment_precedes_subject` | 11 | 17.5% | 20% |
| `subject_mismatch` | 11 | 17.5% | 20% |

## `rejected` — the realized distribution of ROUTES

Denominator 88 claims. **The date does not predict the verdict**: an out-of-period payment is one
route of two, and the 41 zero-coverage claims carry no cause at all.

| route | claims | share | target |
| --- | ---: | ---: | ---: |
| `outside_period` | 47 | 53.4% | 50% |
| `zero_coverage` | 41 | 46.6% | 50% |

Over all 750 claims, counted per occurrence: `limit_exhausted` 153, `mixed_items` 95, `outside_period`
47, `subject_not_evidenced` 17, `counterparty_mismatch` 13, `amount_mismatch` 11, `subject_mismatch`
11, `payment_precedes_subject` 11. The first two sum past their verdict's count because 17 claims
carry both.

## Capture channels, and the share of `content_complete: true` per channel

**This is the denominator of any character error rate on this corpus, and it is part of the result
rather than a caveat to it.** "CER on photo = X" is not a result; "CER on the 344 photographs of 360
whose text survived" is.

Denominator 1241 documents.

| channel | documents | share | `content_complete: true` | share complete |
| --- | ---: | ---: | ---: | ---: |
| `photo` | 360 | 29.0% | 344 of 360 | 95.6% |
| `screenshot` | 320 | 25.8% | 320 of 320 | 100% |
| `scan` | 287 | 23.1% | 287 of 287 | 100% |
| `digital_pdf` | 274 | 22.1% | 274 of 274 | 100% |

**The channels are no longer roughly equal, and that is a calibration rather than a draw.** RP-06's
three-way near-equality was a uniform placeholder; since contract version 35 a channel is drawn per
document class from `capture_mix` in `config/policy.yaml`, and `digital_pdf` — the undamaged original
— is a member. A screen-native archetype takes no draw at all, which is why `payment_confirmation` is
half screenshots.

16 photographs lost text off the frame — 10 across the right edge, 6 across the bottom, 1 across the
left. The other three channels lost none: `screenshot` and `digital_pdf` apply no geometry, and a
flatbed's margin absorbs its own skew.

Per side, which is the denominator that actually applies to a reported CER:

| side | photo | screenshot | scan | digital_pdf |
| --- | --- | --- | --- | --- |
| train | 192 of 197 | 147 of 147 | 145 of 145 | 142 of 142 |
| validation | **152 of 163** | 173 of 173 | 142 of 142 | 132 of 132 |

**Complete means the TEXT survived.** A capture that cut off a QR code or a stamp while keeping every
character counts as complete, and truthfully — the scope is `content_bbox`, which covers text and
nothing else.

## The pixel↔label gate

The first production-scale sweep of `tools/pixel_label_gate.py`, and the only figures on this page
about **pixels** rather than about labels. Denominator: **37 155 labelled boxes over 1241 documents**
— every `field_bboxes` entry the run wrote, and the gate saw exactly that many.

| statement | boxes measured | findings |
| --- | ---: | ---: |
| survival — the shipped JPEG kept the marks | 37 124 | **1** |
| evidence — a box's pixels are its own field's | **not run on this corpus** | — |

31 boxes are unmeasured for survival, each with a reason: 30 the render left blank, one lying outside
the frame.

**One box failed.** `p025_c5_d1:relevant_transaction`, peak correlation 0.549 against a `photo` floor
of 0.700. It is the corpus's smallest box, 22×14 px, on a washed-out patch spanning 46 grey levels;
325 of the corpus's photo boxes are that small, so the rate among them is 1 in 325. It was not
repaired and no floor was raised to swallow it — a floor that clears the hardest photograph could not
go red where the damage is easiest to see, which is the whole reason the floors are per channel.

**The evidence statement was not run on this corpus**, and that is stated rather than left to be
assumed: it costs three renders a page. It *was* run at the same seed and command on the corpus this
one replaced — 28 433 boxes, no findings, over the same eleven archetypes. That is evidence about the
templates, which nothing between the two runs touched; it is not a measurement of these 37 155 boxes.

## Other properties of this corpus

- **Claim shapes.** 491 claims carry two documents, 259 carry one. `linked` is true on the 491. The
  single-document share nearly doubled against RP-06's 141 of 728, by construction rather than by
  draw: a `not_proof_of_payment` claim is one document proving no payment (72 — 40 slips and 32 bare
  invoices), a `subject_not_evidenced` claim is one payment document with no subject beside it (17),
  and a platform receipt proves both facts alone (59).
- **Two currencies and two languages, barely.** EUR and `en` on 39 documents, 3.1%, all of them
  `eu_platform_receipt`; each one's claim carries the applied rate in `fx_rates`. RP-06 was UAH and
  `uk` on all 1315 of its documents and did not exercise either dimension. **3.1% exercises the axis
  and does not populate it** — a per-currency figure on this corpus is not a measurement.
- **Multi-page documents.** 46 bank statements run onto a second sheet.
- **Cross-document checks.** 0 of the 491 multi-document claims contain a fiscal receipt — by
  construction rather than by outcome.
- **Unidentified payers.** 27 of 1241 documents carry a payer of `-`; the rate follows the declared
  initiation shares, not the run's size.
- **VAT summary row on the 111 fiscal receipts**, of which 91 carry a tax row. Paper produced the
  `equals` form 67 times and the spaced form never; electronic drew both, 15 spaced and 9 equals. 20
  receipts carry no tax row because the seller is not VAT-registered. The paper/electronic split here
  follows `capture_mix` for this class, so it measures that calibration and not the world.

## What this corpus is not

Reading order matters here: the report above says what the corpus contains, and this section says what
a consumer must not conclude from it.

- It is not balanced across document classes, and nothing tried to make it so.
- Two of its six document classes are below the per-class minimum on at least one side, and neither is
  a classifier target class.
- It does not contain the `act` or `order_screenshot` classes at any size.
- It does not exercise currency or language beyond 3.1% of its documents, and no jurisdiction other
  than UA.
- Its partition is **not stratified**. Every class and verdict landing above 30 on validation is an
  outcome of this run, not a property the generator guarantees for another size or seed.
- Every artifact in it is synthetic and is not valid proof of payment.
