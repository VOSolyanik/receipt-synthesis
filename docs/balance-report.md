# Balance report — the production corpus

What a run of this generator actually produced, written down so that a figure computed on the corpus
can be read with its denominator beside it. The generator prints this report to stdout; this file is
the copy that gets read by a person.

The machine-readable form of everything below is `run_profiles` profile **RP-06** in
`config/labelling-schema.yaml`. That profile is the authoritative one: it describes the corpus a
consumer receives. **RP-05 was authoritative one contract version ago and is not any more** — it
records the first production run, at half this size and an 85/15 partition, and it is demoted in
writing at its own `superseded_by` key rather than quietly. Its numbers were correct when taken and
are correct now. Everything earlier in that file is an example of how a quantity behaves.

## The run

| | |
| --- | --- |
| Command | `generate-dataset --seed 42 --personas 96 --claims-per-persona 8 --split 0.5` |
| Seed | 42 |
| Generator | receipt-synth 0.1.0 |
| Contract | `config/labelling-schema.yaml` version 20 |
| Jurisdiction | UA (the default; no other is built) |
| Size | 96 personas · 728 of 768 claims built · 1315 documents |
| Partition | by persona, 50/50, not stratified |
| Dataset location | outside this repository — the repo ships the generator, its configuration and the seed |

The 40 claims that were ordered and not built ran out of annual balance: every documentable category
of that persona had its limit spent. That is the limit mechanism working, not a failure.

**Doubling the personas did not double the corpus.** 666 documents became **1315, not 1332**: 40
claims went unbuilt here against 25 in the half-size run. A persona stops producing claims once every
documentable category has its annual limit spent, and that is a per-persona property rather than a
rate — so corpus size is very nearly linear in personas and never exactly linear, and any figure
extrapolated from a smaller run runs slightly high.

### Checking a corpus against this report

```text
manifest_sha256   7169534b716dc5b944e9074cc272f6635547c559499e9d1f4a42ffa699f06335
images_sha256     41cf66d9a37cfb6e12517aed60b0cc8bddc2bcceb18ed671a28b598ae8d4a6bf   1315 files
labels_sha256     c592c3b3807c3fd04b50ea27eeac9d0e48bf11a513732e9fac8f8758ae82ac93   2043 files
```

`manifest_sha256` is the SHA-256 of `ground_truth.json`. The other two are a SHA-256 over the files of
a directory, each contributing its name followed by the SHA-256 of its bytes, in sorted order by name:

```python
h = hashlib.sha256()
for path in sorted(directory.glob("*")):
    h.update(path.name.encode())
    h.update(hashlib.sha256(path.read_bytes()).digest())
```

2043 labels is 1315 document labels plus 728 claim labels. A mismatch means the directory in hand is a
different run, and nothing below applies to it.

## The threshold, checked where it is actually binding: the validation side

The minimum is **≥ 30 documents per buildable class**. Below 30, the 95% Wilson interval around an
accuracy near 0.9 is about ±0.10, so "0.91" and "0.85" are indistinguishable. Because macro-F1 weights
every class equally, a thin class swings the mean.

**The corpus row was never the constraint.** Nothing is measured on the corpus as a whole: a consumer
inspects documents on the development side and reports figures on the measurement side. So the row
that decides whether this corpus is usable is the validation one.

**All four target classes clear 30 on validation.** Denominators: 653 train documents, 662 validation.

| class | train | validation | ≥ 30 on validation |
| --- | ---: | ---: | --- |
| `invoice` | 291 | **296** | yes |
| `bank_statement` | 146 | **148** | yes |
| `payment_confirmation` | 145 | **148** | yes |
| `fiscal_receipt` | 71 | **70** | yes |

This is what the two changes bought together. The same four classes on the previous production run's
validation side were 45, 21, 24 and **10** — two of them could carry no per-class figure at all.
Neither change alone would have fixed it: re-splitting 666 documents at a half yields about 26
validation receipts, still short; doubling at 85/15 leaves the same thin validation side.

`fiscal_receipt` at 70 is the thinnest class and remains the one that decides how small a validation
side may ever be.

Verdicts clear it on validation too — `covered` 180, `partially_covered` 150,
`insufficient_evidence` 36 of 366 claims — which is an outcome, not a guarantee. **The partition is
not stratified.**

### Context: the corpus row

Denominator 1315 documents.

| class | documents | share |
| --- | ---: | ---: |
| `invoice` | 587 | 44.6% |
| `bank_statement` | 294 | 22.4% |
| `payment_confirmation` | 293 | 22.3% |
| `fiscal_receipt` | 141 | 10.7% |

The mix is **not balanced and was never tuned to be**. `config/policy.yaml` declares a `verdict_mix`
and deliberately declares no document mix — a share of receipts against invoices would read as an
observation about which documents claimants actually submit, and nothing here has measured that.
Macro-F1 needs sufficient *n* per class, not equal *n*, so the spread is not a defect.

`other` is outside the threshold and reported separately: **it is empty.** Three classes named by the
vocabulary — `act`, `order_screenshot`, `non_fiscal_receipt` — have no archetype, so no run of any
size contains one. That is absence, not a count of zero.

## Why the partition is a half, and not 85/15

Recorded here rather than inherited, because a fraction adopted silently is exactly the kind of number
that later reads as authoritative.

**Nothing is trained on this dataset.** 85/15 comes from tasks where a model *learns* on the larger
side, and that side is large because learning consumes examples. The partition here guards against
something else: **fitting the measurement**. A consumer looks into documents, finds where
classification or extraction errs, and adjusts accordingly — and a figure does not count on the
documents that were looked at and tuned against. That is all the development side is for, and
inspecting failures needs a few dozen documents.

**A half is not itself derived from that argument** — the argument alone would justify a far smaller
development side. What sets the floor is the thinnest class: a per-class figure needs its 30 documents
*on the side it is measured on*, and `fiscal_receipt` runs at 10.7% here, so a validation side much
below 30% of the corpus could not carry one. A half clears that with margin, leaves the development
side an order of magnitude above what inspection needs, and requires no argument about a number nobody
measured.

The same reasoning now sets the generator's default: `--split` defaults to `0.5`. A consumer whose
pipeline really does train should pass the flag rather than inherit it.

| side | personas | claims | documents | share of claims |
| --- | ---: | ---: | ---: | ---: |
| train | 48 | 362 | 653 | 49.7% |
| validation | 48 | 366 | 662 | 50.3% |

The realized claim share differs from the requested 50% of *personas* because personas do not carry
equal claim counts. Every claim and every document carries a side; nothing is unassigned.

The unit is the **persona**: annual limits are cumulative per persona, so a claim labelled
`partially_covered` for `limit_exhausted` carries that label because of that persona's earlier claims.
Partition by claim and a validation label becomes a function of development data.

## Verdicts

Denominator 728 claims.

| verdict | claims | share | target, of the realizable subset |
| --- | ---: | ---: | ---: |
| `covered` | 338 | 46.4% | 62.5% |
| `partially_covered` | 316 | 43.4% | 25.0% |
| `insufficient_evidence` | 74 | 10.2% | 12.5% |

**At least 20% of the declared mix is absent from this corpus.** `not_proof_of_payment` (10%) and
`partially_paid` (10%) are not generated by any run today, and `rejected` carries no declared share at
all — so the shares above are conditional on the subset that can be drawn, and 20% is a lower bound on
what is missing rather than the whole of it. They are not this dataset's balance against `verdict_mix`
and must not be read as one.

`partially_covered` came out well above target because the oracle overrules the plan: 142 claims drawn
as `covered` were labelled `partially_covered` by a binding annual limit, while 30 drawn as
`partially_covered` came out `covered`. The label is derived, never asserted.

## `insufficient_evidence` — the realized distribution of CAUSES

A verdict's share says nothing about which mechanism produced it. The vocabulary names three ways of
reaching this verdict; the corpus holds two, and this is where that becomes visible.

Denominator 74 `insufficient_evidence` claims. The two cross-check causes are mutually exclusive.

| cause | claims | share | target |
| --- | ---: | ---: | ---: |
| `amount_mismatch` | 40 | 54.1% | 50% |
| `payment_precedes_subject` | 34 | 45.9% | 50% |
| `subject_not_evidenced` | 0 | — | **not drawn** |

The majority changed sides against the previous production run, where `payment_precedes_subject` led
at 53.8%. That is what a share near a half looks like under a different draw, not a change in the
generator.

`subject_not_evidenced` has no declared share, because nothing can plan a deliberately incomplete
claim; a consumer must not expect this cause in the corpus at any size. Both causes that *are* drawn
occur, which is what `config/policy.yaml` requires of the result rather than of the share.

Over all 728 claims: `limit_exhausted` 209, `mixed_items` 136, `amount_mismatch` 40,
`payment_precedes_subject` 34. The first two sum past their verdict's count because one claim may
carry both — 29 claims carry both, a combination for which the policy declares no share.

## Capture channels, and the share of `content_complete: true` per channel

**This is the denominator of any character error rate on this corpus, and it is part of the result
rather than a caveat to it.** A CER is defined only where the text it is measured against is actually
in the picture. "CER on photo = X" is not a result; "CER on the 382 photographs of 406 whose text
survived" is.

Denominator 1315 documents.

| channel | documents | share | `content_complete: true` | share complete | unmeasured |
| --- | ---: | ---: | ---: | ---: | ---: |
| `screenshot` | 456 | 34.7% | 456 of 456 | 100% | 0 |
| `scan` | 453 | 34.4% | 453 of 453 | 100% | 0 |
| `photo` | 406 | 30.9% | 382 of 406 | 94.1% | 0 |

24 photographs lost text off the frame — 19 across the right edge, 5 across the bottom. The other two
channels lost none: `screenshot` applies no geometry, and a flatbed's margin absorbs its own skew.

Per side, which is the denominator that actually applies to a reported CER:

| side | screenshot | scan | photo |
| --- | --- | --- | --- |
| train | 217 of 217 | 233 of 233 | 189 of 203 |
| validation | 239 of 239 | 220 of 220 | **193 of 203** |

193 complete photographs on validation, comfortably above the minimum — which the previous production
run's 27 of 30 was not.

Two warnings that travel with these numbers:

- **Complete means the TEXT survived.** A capture that cut off a QR code or a stamp while keeping
  every character counts as complete, and truthfully — the scope is `content_bbox`, which covers text
  and nothing else.
- **The three channels are roughly equal by construction, not by observation.** They are drawn with
  equal weight, which is a placeholder rather than a measurement of how people submit documents.
  Every figure aggregated over documents is weighted by this marginal.

## Other properties of this corpus

- **Claim shapes.** 587 claims carry two documents, 141 carry one. `linked` is true on the 587.
- **Cross-document checks.** 0 of the 587 multi-document claims contain a fiscal receipt — by
  construction rather than by outcome. A consumer measuring cross-document consistency has no case
  here in which a receipt is one of the two documents.
- **One currency and one language.** UAH and `uk` on all 1315 documents. The corpus does not exercise
  either dimension: a per-value figure computed on it is the corpus average under another name.
- **Unidentified payers.** 32 of 1315 documents carry a payer of `-`; the rate follows the declared
  initiation shares, not the run's size.
- **VAT summary row on the 141 fiscal receipts**, of which 121 carry a tax row. Paper — `photo` and
  `scan` — produced the `equals` form 79 times (47 + 32) and the spaced form never, which is the
  one-sided half of the asymmetric rule appearing in data at four times the previous run's evidence.
  Electronic drew both: 25 spaced, 17 equals. 20 receipts carry no tax row because the seller is not
  VAT-registered. The paper/electronic split here is the capture mix, so it measures that placeholder
  and not the world.

## What this corpus is not

Reading order matters here: the report above says what the corpus contains, and this section says what
a consumer must not conclude from it.

- It is not balanced across document classes, and nothing tried to make it so.
- It does not contain three of the six verdicts in the declared mix.
- It does not contain the `subject_not_evidenced` cause at any size.
- It does not exercise currency, language, or any jurisdiction other than UA.
- Its partition is **not stratified**. Every class and verdict landing above 30 on validation is an
  outcome of this run, not a property the generator guarantees for another size or seed.
- Every artifact in it is synthetic and is not valid proof of payment.
