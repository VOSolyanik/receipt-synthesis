# Cross-document fields — what two documents of one claim say about the same thing

A claim can be evidenced by more than one document, and the two documents are not independent:
they name the same seller, the same claimant, the same money, and one of them may cite the other.
**Linking documents of different types to one claim is the central problem this corpus exists to
pose**, so which fields carry that link, and whether each is a solvable problem or a giveaway, is a
property of the generator that has to be stated rather than discovered.

This page is that statement. It is derived, not remembered — see *How this table was derived*
below — and `tools/cross_document_audit.py` measures every row of it on a generated corpus, from
`reference_text` and never from the builder.

---

## The three states, and why two of them are defects

A field shared by two documents of one claim is in exactly one of three states.

| | state | what a selector built on it measures |
| --- | --- | --- |
| ① | **matches trivially** — the same characters on both pages | nothing. It scores perfectly **by construction**, and the score is a fact about the generator |
| ② | **never matches** — drawn independently on each document | nothing. It scores zero **by construction**, and that score is also a fact about the generator |
| ③ | **resolvable** — the two pages agree on one underlying fact, and reaching it takes work: a different spelling, a different format, a value embedded in free text, or an honest absence on some pairs | the system. The score is **earned** |

**① is as much a measurement defect as ②, and it is the easier one to miss**, because a report full
of perfect scores does not look like a broken instrument. In reality the seller's name on a card
statement is an acquirer's descriptor (`LIQPAY*Ukr Host`) against a legal name on the invoice
(`ТОВ «Укр Хост»`), and the amount on a payment document may include a fee the invoice does not
know about. Both of those are state ③. **The goal of this table is to move rows into column ③**;
the count of rows in ③ is the honest measure of how much of the central problem this corpus poses.

---

## Which documents can co-occur

From `claim_planner._select_documents`, a claim's evidence takes one of exactly two shapes:

* **one document that proves both facts** — a fiscal receipt. It is the whole claim, so it shares
  no field with anything and appears in no row below;
* **one subject document plus one payment document**, where `document_evidence` in
  `config/policy.yaml` says which class is which.

So the cross-document surface is the product `{subject classes} × {payment classes}`, which today is

| subject | payment |
| --- | --- |
| `invoice` | `payment_confirmation` |
| `invoice` | `bank_statement` |

and a class registered later inherits every row of this table by belonging to one of those two
sides. Nothing here is keyed on the slug.

---

## How this table was derived

Not from memory, and not from the fields that happen to disagree today:

1. take every `data-field` marker in each class's template — that is what the class actually
   **prints**, which is also what an extractor can read;
2. for each `{subject, payment}` pair above, cross the two inventories and keep every pair of
   fields that **denote the same thing in the world**: the same party, the same other document, or
   the same fact of the transaction;
3. classify each surviving row as **party identity**, **reference to another document**, or
   **shared fact**;
4. record which of the three states it is in, and check that against
   `tools/cross_document_audit.py` run on a corpus.

Step 2 is what puts `seller_bank_code` in the table even though it is printed under a caption on
one class and folded into a sentence on another, and what keeps `seller_phone` out of it: the
invoice prints one and no payment document has anywhere to print it.

---

## The table

`—` means the class does not print the field at all. A row whose payment column is `—` on **both**
payment classes is not a cross-document field and is not listed.

### Party identity — the seller (the party opposite the claimant)

| field | invoice | payment_confirmation | bank_statement | state |
| --- | --- | --- | --- | --- |
| name | `seller_name` | `payee_name` | labelled row's `counterparty` | ① matches trivially |
| tax code (ЄДРПОУ / РНОКПП) | `seller_tax_code` | `payee_code` | labelled row's `counterparty_code` | ① — **was ② until this table existed** |
| account (IBAN) | `seller_account` | `payee_account` | labelled row's `counterparty_account` | ① — **was ②** |
| bank name | in the payment-order block and again after the IBAN | inside `payee_bank`, **which is sometimes an empty captioned field** | labelled row's `counterparty_bank` | ① where both print it — **was ②** |
| bank code (МФО) | in the payment-order block | inside `payee_bank` | — (the row prints no bank code) | ① — **was ②** |
| VAT number (ПН) | — | — | — | not cross-document: only a fiscal receipt prints it, and a fiscal receipt is a whole claim |
| address, telephone | printed | — | — | not cross-document |

### Party identity — the claimant

| field | invoice | payment_confirmation | bank_statement | state |
| --- | --- | --- | --- | --- |
| name | `buyer_name` | `payer_name` | `payer` (the account holder) | ① — with an honest absence, below |
| tax code (РНОКПП) | `buyer_code` | `payer_code` | `holder_code` | ① — with the same absence |
| account (IBAN) | — | `payer_account` | `account` | **cannot co-occur**: both carriers are payment classes and a claim has one payment document |
| bank | — | `bank_name` / `bank_code` | `bank_name` / `bank_code` | cannot co-occur, as above |

The absence is real and is the closest thing to a ③ this table has today: on the
internet-acquiring mode the confirmation **does not identify the payer at all** and prints a
hyphen, so a linker matching claimants by name has to handle a party that is not named. It is
absence rather than a different spelling, so it is not yet ③ — a linker that gives up on those
pairs is not wrong.

### Reference to another document

| field | invoice | payment_confirmation | bank_statement | state |
| --- | --- | --- | --- | --- |
| invoice number | in the title, «Рахунок на оплату № N» | inside `payment_purpose` | inside the labelled row's `payment_purpose` | ③ **resolvable** — **was ②** |
| invoice date | in the title, **written in words** («від 13 січня 2026 р.») | inside `payment_purpose`, **as digits** (`13.01.2026`) | same as the confirmation | ③ **resolvable** — **was ②** |
| delivery-note (ВН) number | — | — | inside some rows' `payment_purpose` | **deliberately refers outside the claim**, below |
| agreement number | `agreement` | — | — | not cross-document |

Three things make the invoice reference a genuine ③ rather than another ①:

* it is **embedded in free text** on the payment side and in a title on the subject side, so both
  ends have to be parsed before anything can be compared;
* the **date is spelled differently on the two pages** — words against digits — so a comparison of
  the reference as a whole is a normalization problem, not a string equality;
* it is **absent on a large minority of pairs, for reasons that are mostly honest** — measured on
  RP-06, a рахунок is cited on 163 of 261 confirmations that print a purpose and on 174 of 294
  statement rows. Not every purpose line cites a document at all, and some cite a **ВН** — a
  delivery note, which is a different document class and is *not* in the claim. A linker cannot
  assume the reference is there, and one that matches «№» without reading which class is named will
  link the wrong document. That trap is deliberate and the placeholder vocabulary keeps the two
  apart: `{invoice_no}` is the claim's invoice, `{delivery_note_no}` is a document outside it.
  **"Mostly" honest, and the exception is measured below** — see *A contaminated bucket*.

### Shared facts of the transaction

| field | invoice | payment_confirmation | bank_statement | state |
| --- | --- | --- | --- | --- |
| amount | `total` | `amount` (the transfer) | labelled row's `amount` | ① matches trivially, **except where a claim is planned to disagree** |
| amount in words | `amount_in_words` | `amount_in_words`, on some documents | — | ① where both print it |
| currency | «грн» in the count line | — | «валюта UAH/980» in the account header | ① — one value in the corpus, so it distinguishes nothing |
| date | issue date | operation date | labelled row's operation date | **a relation, not an equality**: the subject is dated on or before the payment, and a claim planned for `payment_precedes_subject` inverts exactly that |
| subject / line items | `line_items` | — | — | **asymmetric by definition**: this is why the invoice proves the subject and the payment classes do not |

Two notes on `amount`, because it is the field most likely to be mistaken for a solved ③:

* **the transfer is not the largest number printed.** A confirmation with a non-zero fee prints
  `total_charged = transfer + fee` beside the transfer, and it is the transfer that equals the
  invoice. That is a real trap — but it is a trap for **extraction**, not for **linking**: once the
  right number is read, it matches byte for byte. The linking key stays ①;
* the disagreements that exist are **planned**: `insufficient_evidence` with the cause
  `amount_mismatch` is realized by making the two documents state different amounts. Those are
  labels, not noise, and an audit that counts them as failures is counting the corpus's own
  intent.

---

## What is measured, and on which run

Figures are properties of one run and are quoted with the run named — the table above is the
property, and it does not change when a run does.

`tools/cross_document_audit.py` on **RP-06**, the authoritative production corpus
(`config/labelling-schema.yaml`, `run_profiles`), generated **before** the change that this page
accompanies:

```
invoice + bank_statement — 294 pairs        invoice + payment_confirmation — 293 pairs
  field            readable  agree            field            readable  agree
  seller_name           294    294            seller_name           293    293
  seller_tax_code       294      0            seller_tax_code       293      0
  seller_account        294      0            seller_account        293      0
  seller_bank_name      294     70            seller_bank_name      222     56
  seller_bank_code        0      0            seller_bank_code      222      0
  payer_name            294    294            payer_name            293    261
  payer_tax_code        294    294            payer_tax_code        261    261
  invoice_number        174      0            invoice_number        163      0
```

Read across both shapes: **every identifier of the seller disagreed on every one of the 587 pairs
in that corpus**, and the seller's name and the claimant's identity agreed on all of them. There
was no field in state ③ at all. `seller_bank_name` agreeing on about a quarter of pairs is not
resolvability — the bank is drawn from a four-name list, so a quarter is what coincidence looks
like, and coincidence is the reason a low-cardinality field cannot carry a link.

Two of the readable counts are below the pair count for honest reasons, and neither is a defect:
👁 a confirmation's payee bank is sometimes a **caption with nothing under it**, which is why
`seller_bank_name` is readable on 222 of 293 there; and `payer_tax_code` is readable on 261,
because the internet-acquiring mode names no payer at all.

**RP-06 predates the fix and still has every ② in it.** Any figure taken on that corpus describes
that corpus and nothing else; the corpus is regenerated once, at the start of the next iteration,
and this page's figures are re-taken then.

### Before and after, on one run each

RP-06 cannot show the effect of the change, because it is not regenerated here. So the comparison is
made on a **small local run at the same seed and size**, generated once before the change and once
after — `--seed 4242 --personas 24 --claims-per-persona 8 --split 0.5`, which is a development run
and not a profile.

```text
                     BEFORE (141 pairs)          AFTER (148 pairs)
                     readable   agree            readable   agree
  seller_name          141      141                148      148
  seller_tax_code      141        0                148      148
  seller_account       141        0                148      148
  seller_bank_name     125       29                129      129
  seller_bank_code      59        0                 56       56
  payer_name           141      129                148      138
  payer_tax_code       129      129                138      138
  invoice_number        67        0                 82       82
```

⚠️ **THE TWO RUNS DO NOT HOLD THE SAME PAIRS, and 141 ≠ 148 is the proof.** Drawing the seller's
identity once per claim changes how many values the run takes from the generator, so the same seed
produces a different corpus — every draw after the first identity moves. That is the ordinary cost
of touching a seeded stream, and it means these two columns are **two runs of one configuration**,
not a before-and-after of the same documents. What they compare is the *behaviour*, and on that they
are unambiguous: every identifier of the seller went from **agreeing on none of its pairs to agreeing
on all of them**, and the invoice citation with it.

`invoice_number` being readable on 82 of 148 pairs is the ③ state doing its job, not a shortfall: a
purpose line that names no document, or names a ВН, is not a failed link. **What a consumer cannot
yet tell is which of those two it is** — a linker's recall is therefore not honestly measurable until
the label says whether a citation was there to find. That is the one contract field this change wants
and does not have.

⚠️ **These citation counts were themselves misread once, and the correction is larger than the
figure.** The audit's reader ended its pattern with a list of what may follow the number — `,`, a
date, a space, or `$` — and `$` without `re.MULTILINE` is the end of the WHOLE PAGE rather than the
end of the line. Three of the five configured purposes end in the number, so those matched only when
the purpose happened to be the last text on the document: **99 of 163 citations went unread**, and
every figure above was reported at 64 before it was caught. It never moved the finding it was written
for — nothing agreed either way, and 0 of a subset is still 0 — but coverage is exactly the quantity
that says how much of a linking score is earned, so the understatement mattered to the one column
this page exists to fill. **Describe the token you want; do not enumerate its neighbours.**

---

## 🔴 A contaminated bucket — the "honest absence" is not all honest

The ③ state above rests on a claim that has to be measured rather than assumed: that where a payment
document cites no invoice, **the absence is a real property of the domain**. Measured on RP-06, it is
mostly true and **not entirely**, and the exception is large enough to matter to any figure built on
it.

Of the 261 payment confirmations that print a purpose at all:

| | count | cites a рахунок | honest? |
| --- | ---: | --- | --- |
| «Оплата згідно рахунку № … від …» | 64 | yes | — |
| «Поповнення балансу (гарантійний внесок) згідно рахунку № …» | 59 | yes | — |
| «Оплата за послуги згідно рахунку № …» | 40 | yes | — |
| «Оплата за товар» | 44 | no | ✅ a payment for goods naming no document is ordinary |
| **«Переказ власних коштів»** | **54** | no | 🔴 **no — see below** |

**54 of the 98 non-citing confirmations — 55% of the "honest absence" bucket — say «Переказ власних
коштів», "a transfer of the payer's own funds", on a document whose PAYEE BLOCK NAMES A FIRM.** The
page contradicts itself: the purpose says money moved between the claimant's own accounts, and four
lines above it the recipient is a named company with its own tax code and IBAN. This is the same
defect the bank statement's `credit_from_self` pool already carries a comment about — *"a first
version drew this purpose against an ordinary vendor … Nothing in a label would have caught it; the
render did"* — surviving in the confirmation's pool, where nothing has drawn it yet.

⚠️ **The statement side is clean.** Its labelled row draws only from the `debit` pool, and
`credit_from_self` reaches ordinary credit rows only: of 294 labelled rows, 174 cite a рахунок and
120 cite a ВН, and none claims to be a self-transfer. The contamination is confirmation-only.

**Why it is recorded here rather than fixed here.** Repairing it changes what the documents SAY,
which is a change to the corpus, and the corpus is regenerated exactly once at the start of the next
iteration. The fix goes there, in one pass, with everything else that moves a printed value.

🔴 **And it gates the contract field this page asks for.** The proposed `cites_subject_document`
flag exists so that a linker's recall can be measured against the pairs where a citation was there to
find. Introduced today, it would return `false` for these 54 and thereby **certify them as
legitimately non-citing** — the defect would disappear inside the exception, and the denominator
whose honesty is the flag's entire purpose would be 55% contaminated in its absence half. **The flag
must not ship before the 54 are repaired**, or it legalizes them silently. That condition is written
into the field's specification, not left here.

---

## What this page does not fix

Named here rather than left for a reader to notice:

* **the seller's name is still ①.** Making it ③ means printing an acquirer's descriptor on the
  payment side against a legal name on the invoice — a real and common case — and it is a change
  to what the documents *say*, not to how identifiers are drawn. It is not in this change.
* **the amount is still ① as a linking key**, for the reason given above.
* **the claimant's account and bank cannot co-occur** while a claim has exactly one payment
  document. Nothing is drawn wrongly; there is simply nothing to compare.
* **the reference points one way only.** The payment document cites the invoice; the invoice
  carries no pointer to the payment, and no real invoice would — it is issued first.
* **the 54 self-transfer purposes above.** Measured and recorded, deliberately not repaired here,
  and blocking the `cites_subject_document` field until they are.
