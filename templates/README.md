# `templates/`

One renderable document per `<slug>.html`, its layout in `<slug>.css`. Files with a `.jinja`
extension are **fragments**, included or imported by several templates and renderable on their
own by nothing.

Most of this directory is the shipped corpus. The rest are **mock-ups**, and this file is about
them, because a reader who cannot tell the two apart will read a mock-up as a claim about the
dataset.

## Shipped

| Template | Class |
|---|---|
| `ua_prro_receipt`, `ua_prro_receipt_58mm`, `ua_rro_receipt` | `fiscal_receipt` |
| `ua_bank_payment_confirmation` | `payment_confirmation` |
| `ua_bank_statement` | `bank_statement` |
| `ua_invoice` | `invoice` |
| `ua_fiscal_receipt.jinja` + `.css` | the body and the till-roll rules the three receipts share |

Each is registered in `claim_planner.ARCHETYPES`, has a builder in `assembler._BUILDERS`, marks
every extractable element with `data-field="<name>"`, and reaches a dataset.

## Mock-ups

| Template | What it is |
|---|---|
| `ua_non_fiscal_receipt` | товарний чек — a sales slip issued without a cash register |
| `ua_bank_app_transaction` | one operation as a banking application shows it |
| `ua_bank_receipt_in_app` | `ua_bank_payment_confirmation`, captured inside that application |
| `ua_phone_chrome.jinja` + `.css` | the status bar, tab bar and back arrow those two share |
| `eu_platform_receipt` | a platform receipt in English and EUR |
| `ua_platform_receipt` | the same class in Ukrainian and UAH |
| `platform_receipt.jinja` + `.css` | the body and the page rules the last two share |

Render them with:

```sh
uv run python tools/render_mockups.py --out <a directory outside this repository>
```

### What a mock-up is, exactly

**Not registered.** No entry in `ARCHETYPES`, no builder in `_BUILDERS`, no dataset contains
one. The generator cannot produce them and no test renders them beyond the one guard that sweeps
every file in this directory.

**No labels and no bounding boxes, deliberately.** Not one `data-field` attribute appears in any
of these files — so `RenderedDocument.field_bboxes` comes back empty by construction and the
render script writes no JSON beside the images. The field names would have to come from the
labelling contract, and for two of the five the contract's relevant field is still an open
question (RC-08). Naming fields here would be inventing that answer in markup, where nothing
reviews it. **Labels and boxes arrive with the connection, not with the layout.**

**Nothing in `config/` changed.** Not the labelling contract, not the policy, not the fiscal
rules, not the vendor lists. Where a mock-up needs a Ukrainian string that config does not carry,
the string is in `tools/render_mockups.py` under a comment saying which config file it belongs in
once the archetype is connected.

### Why they are not connected

The production run measures four document classes. Adding layout variety **before** that
measurement would change what the measurement means: macro-F1 would move, and nobody could say
whether the classifier or the corpus had changed. Measure, then add, then measure again — and the
difference becomes a result about robustness to layout rather than a confound.

---

## `ua_non_fiscal_receipt` — the document the fiscality rule has never had

The consuming system's rule is that a **negative** fiscality marker overrides every positive
signal a document carries, however many. This repository implements the precedence structurally,
and **no document in the corpus prints such a marker** — the contract says so itself, at RC-08:
*"no archetype prints one yet"*. The central argument for trusting the system has no test
document. This is that document.

### Sources, counted honestly

**One independent source on what is printed**, restated by four outlets.

The State Tax Service's explanation (БЗ 109.10) is that a товарний чек must carry the wording
**«Товарний чек»** and must otherwise satisfy п. 2 розд. II of the *Положення про форму та зміст
розрахункових документів* (наказ Мінфіну № 13 від 21.01.2016) — the fiscal receipt's own form —
**except** for the fiscal number of the register and the wording **«ФІСКАЛЬНИЙ ЧЕК»**. Three
professional outlets quote that sentence verbatim and identically; a fourth states the same
requirement without quoting the explanation. ⚠️ **They are four files and one source.** The tax
service's own page returned 403 to an unauthenticated fetch, so the wording here is a quotation of
a quotation; anyone re-deriving this should open БЗ 109.10 directly.

⚠️ **One of the four shows a drawn sample rather than a photograph of a real slip**, and the
sample was not used: a drawing is a publisher's reading of the same explanation, not a second
observation. No photograph of a real Ukrainian товарний чек was found in the public search, so
**nothing on this page is evidenced by an image of a real document** — the layout is derived from
the form the explanation points at.

**A second, genuinely independent source on what the document must carry.** A товарний чек is a
primary accounting document, so ст. 9 of the law on accounting (№ 996-XIV) requires it to name the
person responsible for the operation and to bear that person's signature. A fiscal receipt carries
neither. Both lines are printed, and nothing else on the page comes from this source.

**Both sources agree on a third point that matters more than either:** the form and content of a
товарний чек **are not defined by law**. So this is a mock-up of a *convention*, not of a form,
and it can be wrong in ways a fiscal receipt cannot.

### 🔴 The finding: the negative marker is not the string the contract assumes

The contract names three kinds of negative marker — «НЕ ФІСКАЛЬНИЙ ЧЕК», «KOPIA DLA SPRZEDAWCY»,
and an empty receipt-number field. For Ukraine:

* **«НЕ ФІСКАЛЬНИЙ ЧЕК» is not evidenced on a sales document.** No public source found says that
  string is printed on one. `receipt.non_fiscal_marker` in `config/fiscal-rules.yaml` holds it,
  and this mock-up **does not use it** — the constant is left untouched, because changing it is a
  decision to be made after reading this, not a side effect of a mock-up.
* **What is evidenced is a different mechanism, in two parts.** A *positive self-identifying
  title* — «ТОВАРНИЙ ЧЕК» — standing where «ФІСКАЛЬНИЙ ЧЕК» stands on the fiscal form; and an
  *absence*: no fiscal number, no factory serial, no fiscal wording, no register maker's name, no
  online marker, no QR. The contract's own vocabulary already admits an absence as a marker, so
  this is inside its rule rather than beside it — but a field shaped as *"marker text plus its
  position"* cannot represent it, because there is no text to put in.
* The one explicit Ukrainian denial this project has ever observed — «! НЕ ФІСКАЛЬНИЙ !» on a
  ПРРО's «СЛУЖБОВИЙ ДОКУМЕНТ» — is on a **service** document (a cash movement), not a sale, and
  comes from the author's own receipts rather than a public source.

### 🔴 The shape of the slot, not the wording of the rule

Worth separating from the finding above, because it is a different kind of defect and the first of
its kind recorded here. **The rule is right and its prose is right; what cannot hold the answer is
the shape of the field.** The contract's rule already admits an absence as a negative marker — it
names an empty receipt-number field as one of the three kinds — so nothing in the wording needs
changing. But the slot RC-08 proposes is *marker text together with its position in the text*, and
an absence has neither. There is no string to store and no coordinate to store it at. A generator
filling that field for this document would have to write something that is not on the page.

Every gap this repository has recorded until now was a gap in what a document states, closable by
observing more documents or by wording a rule more carefully. This one is closed by neither: an
absence is representable only by a field whose type can say *"this requisite is not here"* —
a set of requisites found missing, or a per-requisite presence flag — and no amount of further
observation turns it into a string. So the question RC-08 has to answer first is not *which
wording* but *what shape*, and answering the second by continuing to assume the first is how a
contract comes to have a field that is always empty and never wrong.

**Is that enough to close RC-08? No — and the gap is nameable.** What is missing is an observation
of a real **sales** document that prints an explicit denial. Until one exists, RC-08's field
cannot be specified as a string with a position without that specification being an invention.
What the evidence does support is a **narrower** field that this document could exercise today: a
non-fiscal *title* in the fiscal title's slot, plus the set of fiscal requisites found absent.
Whether the contract should carry that instead is the author's decision, not this branch's.

### Narrower than reality: the slip

* **One variety of two.** The printed slip is modelled; the **handwritten** товарний чек on a
  pre-printed pad — the other real variety, and a different image problem entirely — is not.
* **One paper width.** 80 mm only; the 58 mm roll is a second template when this is connected,
  exactly as the fiscal class already splits.
* **The position of the title** at the foot follows from *"the same content as the form, minus two
  items"*, not from a sample. It is the weakest claim on the page.
* **No layout diversity between issuers of such slips**, of which nothing is known publicly.

---

## `ua_bank_app_transaction` — the strongest negative example available

A screenshot of one operation inside a banking app: the **dominant carrier in this domain**, and
absent from the generator entirely.

It looks like proof of payment to a person and carries **none of the requisites** by which proof
of payment is recognized — no document number, no authorization code, no RRN, no stamp, no
signature, no amount in words, no parties, no account, no purpose of payment. Every one of those
is printed by `ua_bank_payment_confirmation`.

Formally the class is already covered: it is `other`, and `other` routes to `in_review`, which is
the **correct** outcome. The risk is in how a system gets there — a classifier reading a large
amount, a merchant and the word «квитанція» off the interface has every reason to call it a
payment confirmation, extraction then hunts for fields that are not on the image, and the failure
surfaces as "poor extraction quality" rather than as *the wrong kind of input*. A corpus with no
such image never measures which way the classifier falls.

### The merchant descriptor

The counterparty line is **not a legal name**. It is `<PROCESSOR>*<tail>` — the string a payment
processor puts into the card network's message. Nothing in this repository has ever printed one.

* The **prefix** is a public Ukrainian processor, read from `payment_providers` and `aggregators`
  in `config/vendors.json`. Public marks, used for exactly the trade they are publicly in.
* The **tail** is a sole trader's name after the descriptor field has had it: uppercased,
  punctuation squeezed out, the legal form gone. It is composed by
  `content_builder.sole_trader_name` from a published high-frequency surname — **never an invented
  firm**, for the reason stated at the head of `config/generation.yaml`.

So the bank says `CITY24*ПОНОМАРЕНКО С.Л.` and an invoice for the same purchase says
`ФОП Пономаренко С. Л.`, and a counterparty cross-check comparing the two fails on a pair of
entirely genuine documents. ⚠️ **The normalization rule is a consumer's work, not this
repository's.** The mock-up's job is to print the phenomenon so there is something to decide about.

### Narrower than reality: the screen

* **No issuer's visual design is reproduced.** Generic dark palette, neutral type, no accent hue.
  Ukrainian banking apps do not look like this and do not look like each other; copying one would
  be a claim about that bank's interface made from screenshots that cannot be cited. What is
  modelled is the **arrangement**, which is what a document-understanding system reads.
* **One device geometry.** 1170 × 2532 — a 390 × 844 pt screen at a scale factor of 3. Real
  captures vary in both.
* **Every status-bar value is invented** — clock, battery, signal. **No carrier name is printed at
  all**, which sidesteps the question rather than answering it: no public pool of them exists here.
* The Latin-alphabet descriptor tail, which also occurs in reality, is not produced; only the
  Cyrillic form is.

---

## `ua_bank_receipt_in_app` — one document, two carriers

The **same** payment confirmation, captured inside the app: dark surround, status bar, back arrow,
a screen heading, the document as a white card, a share button, a tab bar.

It exists because it tests, directly and cheaply, that **the medium does not change the ground
truth**. Nothing else here asks that: `degrader` *damages* an image, whereas this *reframes* the
document — and a frame is what a classifier is most likely to read as a different kind of thing.
When it is connected, it is the cheapest invariance test the corpus will have.

**The body is neither rewritten nor copied.** The card holds `ua_bank_payment_confirmation` — the
shipped template, rendered from the shipped builder's context by the shipped renderer — written to
disk and embedded as a frame. Copying the markup would create two spellings of one layout, which
is the defect the labelling contract exists to prevent; extracting a shared body would edit a file
a shipped archetype renders from, which this branch does not do. The frame has a property neither
alternative has: **the inner document is the one the dataset already contains.**

🔴 **And that has a cost, which is a decision for whoever connects it.** Rendering a document
*inside* another one means the inner page is written to disk before the outer page can point at
it, and `Renderer.render` does not do that today — `tools/render_mockups.py` does it instead. The
alternative at that point is to extract the confirmation's body into a shared `.jinja` fragment,
exactly as the three fiscal receipts already share `ua_fiscal_receipt.jinja`; that is the cheaper
route once editing a shipped file is allowed, and it is **the recommendation**. The frame is what
proves the layout works without touching anything.

The render script also writes `ua_bank_payment_confirmation.png` — the A4 twin. That pair is the
archetype's whole argument, and it is meant to be looked at as a pair.

---

## `eu_platform_receipt` and `ua_platform_receipt` — the second currency and the second language

### 🔴 The corpus has one of each, and the contract says so itself

The authoritative reference profile records it as a finding rather than a footnote:
`single_valued_dimensions` — **`currency_UAH: 1315`** and **`language_uk: 1315`** out of 1315
documents — with the note that the run therefore does not **exercise** either dimension, and that
a per-currency or per-language figure computed on it is the corpus average under another name.

Foreign-currency conversion is declared core to the system. **No document in the corpus is in a
foreign currency.** These two are the first that would be, and they are a pair on purpose: a
dimension with one document in it measures the document, not the dimension.

### 🔴 Two things this class runs straight into, both of them structural

**The oracle refuses foreign currency.** `policy_engine._check_currency` raises on any document
whose currency is not `reporting_currency`, and its docstring says it deliberately offers no
conversion, because *"a rate applied here would put a number in the ground truth that nothing in
the dataset can prove"*. That is **correct** while every archetype emits UAH: a foreign-currency
document really would mean something upstream had gone wrong. The moment this archetype exists,
the same code stops being a guard against a contradiction and becomes an unmade decision — either
the ground truth carries a converted number, or the claim's amount stays in EUR and the limit
comparison moves to the consumer, where the contract's own `currency` note already puts it.
Connecting this template without settling that turns a deliberate raise into a crash on real data.

**`config/fx-rates.yaml` is read by nothing.** `config.load_fx_rates` has **no caller in `src/`**;
the only other mentions of the file in the tree are two documentation tables describing it. Its
header explains that the rates are static so that conversion stays deterministic — a property of a
conversion no code performs. `tools/render_mockups.py` is the first caller in the repository, and
even there the converted figure is printed to the console, never onto a page, because there is no
document behind it.

**Nothing on either receipt states a rate or an equivalent.** No public source found shows such a
receipt doing so, so neither does this one. That is what makes the pair useful rather than what
makes it incomplete: a claim evidenced by a EUR receipt and a UAH bank payment carries its
conversion **nowhere on paper**, which is the abstract sentence in the policy engine made concrete.

### Sources, counted honestly

**Four independent source families. No photograph of a real document among them.**

1. **📄 EU law — Article 226 of Directive 2006/112/EC**, the exhaustive list of particulars a VAT
   invoice must carry. Used *backwards*, as the fiscal form was for `ua_non_fiscal_receipt`: the
   English receipt declares itself not an invoice, so it may lack the supplier's address, the
   supplier's VAT identification number, the customer's, the tax rate and the tax amount — and it
   lacks exactly those. ⚠️ Three files quote the article; they are **one source**, the Directive.
2. **📄 UA law — Law № 1525-IX with ст. 208¹ ПКУ and Section VIII of the VAT-registration
   regulation**: a non-resident supplying electronic services to Ukrainian individuals registers
   for ПДВ, receives an individual tax number and charges 20%. Several tax-service pages and three
   professional outlets restate it; again **one source**, the law. Used to decide what is *not*
   built — see the limit below.
3. **Vendor documentation — the payment platform's own published guidance on what a receipt
   contains**: business information, receipt number and date, an itemized list, and payment
   information including the total, the method and any tax. That field set is what both variants
   print. A primary source of a different kind from the two above, and independent of them.
4. **👁 Accounting practice — the wording «This is not a VAT invoice»**. Four independent
   commentators describe it as printed on real platform receipts, explaining what it means for a
   buyer who wants to deduct the tax. ⚠️ These four are independent of each other, unlike the
   restatements in (1) and (2) — but **not one of them shows an image of a receipt carrying it**,
   so the wording is evidenced by description and never by observation.

### 🔴 What this adds to the RC-08 argument

The Ukrainian non-fiscal slip's negative marker is an **absence** — no string, no coordinate — and
the section above concludes that a field shaped as *marker text plus its position* cannot hold it.

This class supplies the other half of that argument. «This is not a VAT invoice.» **is** a string
with a position: it sits at the foot of the page, it can be quoted, and the proposed field shape
represents it exactly. So the two mock-ups together say something neither says alone — the field
shape is not wrong, it is **incomplete**. It covers the printed denial and cannot express the
denial-by-omission, and both occur. Whatever RC-08 settles on has to hold both, or it will look
correct on every English document and be silently empty on every Ukrainian one.

⚠️ And the configured constant diverges here too. `receipt.non_fiscal_marker` in the **EU block**
of `config/fiscal-rules.yaml` is `"NOT A FISCAL DOCUMENT"` — a claim about *fiscality*, a
cash-register concept. What the sources evidence is a claim about being a *tax invoice*, which is
a different thing: a receipt can be perfectly fiscal and still not be the document that lets a
buyer deduct the tax. The mock-up prints the evidenced wording; **config is untouched**.

### The pair differs on three axes, not two

Reading it as a clean two-variable comparison would credit the currency with a difference the tax
block caused. The three:

* **language** — `en` against `uk`, and it is data: every caption arrives in a `labels` dictionary
  and the shared body contains no printed word of either language;
* **currency** — EUR against UAH, with both separators read from `number_format` in
  `config/fiscal-rules.yaml`, which sets them opposite ways round for the two jurisdictions;
* **tax treatment**, which is **law and not noise**. The Ukrainian seller is a domestic company
  registered for ПДВ, so it prints its identifiers and a «У т.ч. ПДВ» line — the Ukrainian
  convention of a VAT-inclusive price — and makes no statement about not being a tax document. The
  English one carries the denial and none of those requisites.

### Narrower than reality: the platform receipt

* **EUR rather than USD**, and the reason is the repository's own vocabulary: `Country` in
  `schemas.py` holds UA, PL, DE and ES, `config/fiscal-rules.yaml` already carries an `EU` block
  with `language: en`, `currency: EUR` and `date_format: %Y-%m-%d`, and there is no US
  jurisdiction here to hang a USD document on. Both codes are in `config/fx-rates.yaml`, so the
  USD variant costs a context and no template.
* ⛔ **A foreign platform's Ukrainian receipt is not built**, and it would be the sharper artifact —
  one seller issuing both variants, differing in nothing but language and currency. Printing it
  means printing a named foreign company's Ukrainian tax registration, which is a checkable claim
  about a real firm that this branch has not checked and must not assert. The domestic seller
  states only what `config/vendors.json` already states.
* ⛔ **No issuer's visual design is reproduced.** Real platform receipts are branded and no two look
  alike; one plain layout is modelled and the diversity is a declared gap — the same limit the A4
  confirmation carries.
* **One line-item shape.** Two courses, quantity one each. Subscriptions, proration, refunds and
  coupon lines are all real and none is modelled.
* **The seller's name is a public mark and its identifiers are generated** — the same construction
  the bank confirmation already uses, where a real bank's name sits beside an invented bank code.
  Every ground-truth record carries `"synthetic": true`, and the README says what these artifacts
  are.

---

## What was looked at, and what was seen

The renders were inspected by eye, not merely rendered. What that inspection changed:

* **The non-fiscal slip drew «БЕЗГОТІВКОВА»** — a non-cash sale. 📄 A card sale obliges the seller
  to use a register, so a slip issued without one records **cash**; the payment method is now
  «ГОТІВКА», read from `config/fiscal-rules.yaml`, whose own comment says that value is printed by
  no shipped document. The mock-up is the first thing in the repository to reach it.
* **The receipt-in-app frame shows the screen heading «Квитанція № …» over a document whose own
  heading reads «Платіжна інструкція»**, both carrying the same number. That is not staged — the
  confirmation drew its own title — and it reproduces a property the confirmation's comments
  already record from real captures.
* **The white card is blank for roughly its bottom third.** That is the A4 sheet's own proportion,
  carried faithfully, and it is left alone: shortening it would mean showing a different document
  and giving up the one thing this archetype is for.
* **The app screen's category chip disagrees with what was bought.** Left as it is, and now
  documented: a bank categorizes by the payment route, so a purchase settled through a
  utility-payment aggregator lands under utilities whatever the merchant sells. A third thing on
  that screen that looks like evidence and is not.
* **The Ukrainian platform receipt drew its footer rule across the page with nothing under it.**
  It carries neither of the two footer notes, and the `<footer>` was unconditional — a line
  announcing an empty block. The whole foot is conditional now, which is the same defect and the
  same fix the A4 confirmation already records in its own comments.
* **Both platform receipts priced every line identically**, because one drawn price was repeated
  across the table. It reads as a template filling itself rather than as a purchase, and a corpus
  where every line of a document carries the same figure teaches an extractor that it only has to
  read one of them. Each line is now priced independently.
* **«У т.ч. ПДВ:» came out uppercased** by the `.caption` rule that is right for the small field
  labels above it. That label is a value `config/fiscal-rules.yaml` settled, written the way an
  invoice writes it, and a stylesheet had been overriding the casing config chose. The totals
  captions keep their own case now.
* **Both platform receipts are blank for their bottom third**, exactly as the A4 confirmation is,
  and for the same reason: it is the sheet's own proportion. Left alone.
