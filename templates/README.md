# `templates/`

One renderable document per `<slug>.html`, its layout in `<slug>.css`. Files with a `.jinja`
extension are **fragments**, included or imported by several templates and renderable on their
own by nothing.

Most of this directory is the shipped corpus. The rest are **mock-ups**, and this file is about
them, because a reader who cannot tell the two apart will read a mock-up as a claim about the
dataset.

## The five mock-ups at a glance

**Five archetypes across six renderable templates.** Render them all with one command; every
image lands in the directory given, outside this repository:

```sh
uv run python tools/render_mockups.py --out <a directory outside this repository>
```

Sizes below are from the default seed. Heights that depend on drawn content — a basket, a
purpose — move with the seed; the fixed ones are marked.

---

**1 · `ua_bank_app_transaction`** — one operation as a banking application shows it.
* **File:** `ua_bank_app_transaction.png` · 1170 × 2532 px, a phone screen at 3×.
* **Makes measurable:** the **strongest negative example for the payment class** — it looks like
  proof of payment and carries none of the requisites by which proof of payment is recognized.
* **Look for:** the counterparty line — `CITY24*ПОНОМАРЕНКО С.Л.`, a processor prefix and a
  mangled tail, which no document here has ever printed. Then check the screen for a document
  number, an authorization code, an RRN, a stamp, a signature, a purpose: **none of them is
  there.** The category chip also disagrees with what was bought, on purpose.

**2 · `ua_bank_receipt_in_app`** — the same payment confirmation, captured inside the app.
* **Files:** `ua_bank_receipt_in_app.png` · 1170 × 2532 px, **beside** `ua_bank_payment_confirmation.png`
  · 794 × 1123 px — the A4 twin. **The pair is the artifact; look at both.**
* **Makes measurable:** that **the medium does not change the ground truth.** One document, two
  carriers, and the labels must agree.
* **Look for:** that the white card really is the same document as the A4 sheet, field for field.
  Then the screen heading «Квитанція № …» sitting over a document whose own heading reads
  «Платіжна інструкція» — same number, two wordings, and not staged.

**3 · `eu_platform_receipt`** and **4 · `ua_platform_receipt`** — a platform receipt, in English
and EUR, and the same class in Ukrainian and UAH.
* **Files:** `eu_platform_receipt.png` and `ua_platform_receipt.png` · both 794 × 1123 px, fixed.
* **Makes measurable:** the **currency and language dimensions**, which the corpus does not
  exercise at all — the contract's own reference profile records `currency_UAH: 1315` and
  `language_uk: 1315` out of 1315 documents. Also **the other half of RC-08**: the English one
  carries a negative marker that IS a string with a position, where the Ukrainian slip's was an
  absence.
* **Look for:** «This is not a VAT invoice.» at the foot of the English page, and its **absence**
  on the Ukrainian one, which prints a ПДВ line instead. Then the number formats, which run
  opposite ways — `397.05` against `5 713,03`. Nothing on either page states a rate: the
  conversion this pair implies is **nowhere on paper**, and that is deliberate.

**5 · `ua_insurance_contract`** — a three-page voluntary health insurance contract.
* **File:** `ua_insurance_contract.png` · 794 × 3401 px, fixed — three A4 sheets in one image.
* **Makes measurable:** **«one page = one document», broken in one direction** — one document
  across several pages. Every other archetype here renders exactly one page, so a file-splitting
  step would otherwise score a perfect result against a guarantee.
* **Look for:** the page footers «Сторінка N з 3» beside the document's own number, and pages two
  and three with **no title, no parties and no requisites** — a running head and clauses that
  begin mid-numbering. Also the salience trap: the largest figure is the **sum insured**, and the
  money that moved is the **premium** two rows below, roughly forty times smaller.

**6 · `ua_claim_bundle`** — an invoice and the bank confirmation that settled it, in one file.
* **Files:** `ua_claim_bundle.png` · 794 × 2262 px, **beside** `ua_claim_bundle.page1_invoice.png`
  and `ua_claim_bundle.page2_confirmation.png` — the file, and the two documents it is made of.
* **Makes measurable:** **«one page = one document», broken in the other direction** — two
  documents in one file. Both directions, or a segmentation figure is not a figure.
* **Look for:** what marks the boundary between the two documents — **nothing but the grey gap.**
  No cover page, no continuous pagination, no unifying header. Then the linkage that makes them
  one claim: the payment purpose names the invoice on the page above it, the amounts match, and
  the two pages give the same firm one ЄДРПОУ and one IBAN.
  🔴 **That last part is what this file revealed and what has since been fixed:** the two pages
  used to name one firm by two different ЄДРПОУ and two different IBANs — the **shipped
  generator's** behaviour, not the mock-up's. It is written up below, with the old figures kept as
  the record of what the defect looked like.

---

### What this branch deliberately does not contain

**No labels and no bounding boxes** — not one `data-field` attribute in any of the six
templates, so `field_bboxes` comes back empty by construction and no JSON is written beside any
image. **No connection to the dataset** — nothing is registered in `ARCHETYPES`, no builder exists
in `_BUILDERS`, and no run of any size contains one of these documents.

⚠️ **THAT WAS TRUE OF SEVEN TEMPLATES AND IS TRUE OF SIX.** `ua_non_fiscal_receipt` has been
**connected**: it is registered, it has a builder, it carries `data-field` attributes, its
Ukrainian strings moved into `config/fiscal-rules.yaml`, and a run contains one wherever a claim
was planned as `not_proof_of_payment`. What it demonstrated as a mock-up is unchanged and is kept
below, because the finding — that the negative marker is not the string the contract assumes — is
what RC-08 still turns on. The connection did **not** decide RC-08: no field records a marker, and
the absence is carried by `has_fiscal_number` / `has_qr` / `qr_is_fiscal` instead.

## Shipped

| Template | Class |
|---|---|
| `ua_prro_receipt`, `ua_prro_receipt_58mm`, `ua_rro_receipt` | `fiscal_receipt` |
| `ua_bank_payment_confirmation` | `payment_confirmation` |
| `ua_bank_statement` | `bank_statement` |
| `ua_invoice` | `invoice` |
| `ua_non_fiscal_receipt` | `non_fiscal_receipt` |
| `ua_fiscal_receipt.jinja` + `.css` | the body and the till-roll rules the three receipts share |

Each is registered in `claim_planner.ARCHETYPES`, has a builder in `assembler._BUILDERS`, marks
every extractable element with `data-field="<name>"`, and reaches a dataset.

## Mock-ups

| Template | What it is |
|---|---|
| `ua_bank_app_transaction` | one operation as a banking application shows it |
| `ua_bank_receipt_in_app` | `ua_bank_payment_confirmation`, captured inside that application |
| `ua_phone_chrome.jinja` + `.css` | the status bar, tab bar and back arrow those two share |
| `eu_platform_receipt` | a platform receipt in English and EUR |
| `ua_platform_receipt` | the same class in Ukrainian and UAH |
| `platform_receipt.jinja` + `.css` | the body and the page rules the last two share |
| `ua_insurance_contract` | a three-page insurance contract — one document, several pages |
| `ua_claim_bundle` | an invoice and the confirmation that settled it — one file, two documents |

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
labelling contract. Naming fields here would be inventing that answer in markup, where nothing
reviews it. **Labels and boxes arrive with the connection, not with the layout** — which is
exactly what happened to `ua_non_fiscal_receipt` when it was connected.

**Nothing in `config/` changed.** Not the labelling contract, not the policy, not the fiscal
rules, not the vendor lists. Where a mock-up needs a Ukrainian string that config does not carry,
the string is in `tools/render_mockups.py` under a comment saying which config file it belongs in
once the archetype is connected.

### Why they are not connected

The first production run measured four document classes. Adding layout variety **before** that
measurement would change what the measurement means: macro-F1 would move, and nobody could say
whether the classifier or the corpus had changed. Measure, then add, then measure again — and the
difference becomes a result about robustness to layout rather than a confound.

⚠️ **`ua_non_fiscal_receipt` was connected after that measurement and deliberately**, because it
is not layout variety: it is the only document that makes a RULE testable — a negative fiscality
signal overriding positive ones — and the rule has no test case without it. The reasoning above is
unchanged and still governs the five templates left here; a figure measured before the class
existed is a figure about a corpus of four classes, which is what every profile in
`config/labelling-schema.yaml` says of itself.

---

## `ua_non_fiscal_receipt` — the document the fiscality rule had never had

**Now shipped.** The section is kept whole because the sources, the finding and the narrowings
below are what the archetype was built from and what RC-08 still turns on — none of them changed
when it was connected. Where the text says "mock-up", read "this page": the layout it describes is
the layout a run renders.

The consuming system's rule is that a **negative** fiscality marker overrides every positive
signal a document carries, however many. This repository implements the precedence structurally,
and until this archetype was connected **no document in the corpus carried anything negative to
override with** — the contract says so itself, at RC-08: *"no archetype prints one yet"*. The
central argument for trusting the system had no test document. This is that document.

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
* **One paper width.** 80 mm only; the 58 mm roll would be a second template, exactly as the
  fiscal class already splits — and it is a real narrowing now that runs contain this class.
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

## 🔴 A decision this branch surfaced and must not make: foreign currency in the oracle

**Recorded here rather than implemented, and rather than filed anywhere else.** The change it
describes lands in `policy_engine.py`, which is the oracle — connecting an archetype and editing
the component the whole labelling argument rests on, both of which this branch is forbidden. It
is not in `docs/` either: that directory is the public design reference, and an unmade decision
put there would read as settled design. It sits beside the mock-up that surfaced it.

### What the code does today, and why it was right

`_check_currency` in `src/receipt_synth/policy_engine.py` (line 759 at the time of writing)
**raises** on any document whose `currency` is not `reporting_currency`. Its own docstring says
the refusal is deliberate and that no conversion is offered at any rate from any source, because
*"a rate applied here would put a number in the ground truth that nothing in the dataset can
prove"*. The raise message goes further: *"Every archetype this generator has emits UAH, so this
document should not exist."*

**That was correct, and it was not a stub.** While every archetype emitted UAH, a foreign-currency
document could only mean something upstream had gone wrong — a jurisdiction wired to the wrong
currency, a document attached to the wrong persona. Refusing loudly was the right response to a
contradiction, and converting silently would have hidden it.

### What changed

`eu_platform_receipt` is a document in EUR that is **correct**. The same code therefore stops
being a guard against a contradiction and becomes an **unmade decision**: the condition its
message asserts — that no archetype emits anything but UAH — is a fact about the registry, and the
registry is exactly what connecting this archetype changes. Whichever way the decision goes, that
message becomes false and has to be rewritten; it is not a comment, it is what a maintainer reads
when the run stops.

Note also what did **not** change: the reason the docstring gives is still sound. Nothing on
either platform receipt states a rate or an equivalent, so a converted number really is unprovable
*from the documents*. The question is whether the documents are the only thing allowed to prove it.

### Resolution A — convert at build time, and record the applied rate in the label

`config/fx-rates.yaml` **already exists, is static by design for determinism, and is already
vendored into the consuming MVP.** That is what makes this different from applying an invented
number: the rate is a **shared, checkable input** that both sides load, so a consumer can
reproduce the conversion exactly rather than take a figure on trust. The generator converts using
that table at the document's date, and the policy engine compares against the limit in
`reporting_currency` as it does now.

🔴 **Strictly conditional on the label recording the rate that was applied** — the rate itself, not
only its result, and beside the original amount and currency, which the contract already keeps per
document. Without that the objection in the docstring stands unchanged, and it stands *more*
sharply than before: a rates file can be edited, and a label carrying only a converted figure
becomes silently unreproducible the moment it is. The rate in the label is what turns
"unprovable" into "derived from a stated input".

Consequences to accept with it:

* **The contract gains fields and a version.** At least the applied rate; probably its source and
  that file's `version`. This is a contract change, not an engine change with a contract
  side-effect.
* **The converted amount is a derived field and not an extraction target.** It is on no document
  and no extractor can read it, so a per-field score on it measures nothing — the same trap the
  contract already names for `amount_due` while that field cannot diverge from `total`. It has to
  be marked as such or somebody will quote an F1 for it.
* **`jitter` has to stay disabled or be recorded too.** The file offers seeded rate variation and
  it is off; switching it on without the applied rate in the label reintroduces exactly the
  unreproducibility this resolution exists to avoid.

### Resolution B — leave the amount in its own currency, and move the comparison to the consumer

The engine stops refusing and stops converting: the label keeps `amount` and `currency` per
document, as it already does, and the limit comparison happens at approval time in the consuming
system. **The contract's own `currency` note already describes this arrangement** — *"the consumer
converts to a base currency at approval time using the transaction date, which is why the original
amount, currency and date are all kept per document"* — so this resolution is the one the contract
is currently written for.

Consequences to accept with it:

* **No new fields and no version bump.** The cheapest of the two by a wide margin.
* **The verdict path is not exercised on foreign currency.** The oracle cannot compute coverage
  against a UAH limit without converting, so such a claim gets no limit-bound verdict — it can
  exercise classification and extraction and nothing further. That is a coherent staged answer,
  but it must be *stated*, because it means the currency dimension is exercised on two of the
  three things the system does and not on the third.
* **It concedes the thing the archetype was built to test.** Foreign-currency conversion is
  declared core; under B it stays untested end to end inside this repository, and the evidence for
  it moves to the consumer's own tests.

### Where the evidence in this repository already leans, and who decides

A lean and not a decision: the repository has already paid for the input Resolution A needs.
`fx-rates.yaml` is committed, static, versioned and vendored downstream, and the only thing
standing between it and a defensible conversion is a label field. Resolution B is cheaper and
gives up the measurement.

**The choice is the author's, in T3, together with connecting the archetype** — the two are one
decision, because the archetype cannot be registered without the engine having an answer, and the
answer costs nothing until it is.

---

## `ua_insurance_contract` — the control that makes «one page = one document» falsifiable

### 🔴 Why a multi-page document at all

**Every archetype in this repository renders exactly one page.** A corpus built from them cannot
contradict the rule *one page is one document* — so a later step that splits a submitted file into
logical documents would score a **perfect result on it while never having been tested**. A
measurement whose ideal is guaranteed by the construction of the data measures the construction,
not the system.

This document is the control. **One document, three pages**, in one file. Without something like
it in the corpus, a segmentation figure is not a weak result — it is not a result.

### ⛔ It is explicitly not evidence, and it cannot be labelled today

`medical_insurance` in `config/policy.yaml` states in as many words that a claim of that category
is proven by *a bank payment confirmation whose payment purpose names the policy and the insurer*,
and that **«the policy document itself is out of scope and is never parsed»**. So this archetype
belongs to the catch-all class and to nothing else: it exists to be **segmented and classified**,
never read for a verdict.

🔴 And the generator could not label it as the catch-all even if it were connected: `DocType` in
`schemas.py` has **no `other` member and no `contract` member**, both of which the consumer's class
vocabulary names. That is **RC-14**, still `blocked_on: decision`. So this mock-up sits on the far
side of an open contract question, exactly as `ua_non_fiscal_receipt` sits on the far side of
RC-08 — and the same discipline applies: no `data-field` here either, because naming fields would
be inventing the answer in markup.

### Sources, counted honestly

**Two independent legal source families. No photograph of a real contract among them, and the
clause prose is invented.**

1. **📄 Article 89(2) of the law on insurance (№ 1909-IX)** lists **nineteen** particulars an
   insurance contract must contain — the document's name, the insurer's name and address, the
   policyholder's, the subject and object of the insurance, the sum insured, the tariff, the list
   of insured risks, the list of exclusions, the term and territory, the premium with its payment
   procedure, the amendment and termination procedure, the payout procedure and timelines, the
   grounds for refusing a payout, the parties' rights and obligations, and the dispute-resolution
   procedure. **That is what makes three pages evidenced rather than chosen.** A one-page
   insurance contract would be the invention here; nineteen mandatory particulars do not fit on a
   sheet, and the layout places them across three.
2. **📄 Article 979 of the Civil Code** makes the **contract** the thing and allows it to be
   *issued* as a поліс or сертифікат — so a поліс is a **form** of the contract, not a separate
   document class. The document is therefore titled as a contract and the word «поліс» appears
   nowhere on it: printing both would suggest the corpus holds two classes where the law holds one.

⚠️ **The clause prose is invented and short**, and both halves are deliberate. Invented is safe —
a contract clause is a text, not a mark under which a firm trades, and no real document was read
to produce any of it. Short is a **declared limit**: what is modelled is the page structure and
where the required particulars fall, which is what a segmenter and a classifier read. Real
contract prose runs several times longer, and padding a mock-up to look right would mean inventing
legal text at length to no measurable end.

### What carries the segmentation signal

Three things, and none of them is enough alone:

* **The continuation pages carry no title, no parties and no requisites** — a running head naming
  the document, and clauses that begin mid-numbering. That is what a wrongly-cut page looks like.
* **A page footer on every sheet**, carrying «Сторінка N з 3» *beside the document's own number*.
  The count says how many pages the document has; the number says **which** document they belong
  to, which is what a file holding two documents would have to be split on.
* **One `data-document` root for three sheets.** The renderer frames the image on that element, so
  the image is the whole **file**. A root per sheet would have produced three documents, which is
  the confusion this archetype exists to break.

### 🔴 A salience trap of a different order

The largest figure on the document is the **sum insured**, and it is not an amount anyone paid.
The premium is what money changed hands for, and it is roughly forty times smaller and printed two
rows below. The policy puts this document out of scope precisely so nothing has to read either —
but a system that has not applied that rule and reaches for the most prominent figure is wrong **by
a factor, not by a margin**. The payment confirmation carries the same trap as a fee; here the gap
is an order of magnitude.

### Narrower than reality: the contract

* **A PNG of stacked sheets, not a PDF.** A real multi-page document reaches a verifier as a PDF,
  and this repository already pins ReportLab for exactly that. Producing the container is part of
  connecting the archetype; the mock-up shows the layout and the page structure, not the file
  format. The grey field between sheets asserts no particular viewer — a reproduced toolbar would.
* ⛔ **There is no pagination engine.** A sheet has a fixed height and **clips**; which section
  lands on which page is decided in `tools/render_mockups.py`, not by the stylesheet. A section
  that grows past its sheet is a defect to be caught by eye, and the render has to be looked at
  whenever the clause list changes.
* 🔴 **ONE document per file — the other half of the control is NOT built.** This falsifies *one
  page = one document* in one direction only: one document spanning several pages. The opposite
  case — **one file holding two documents**, a contract followed by its appendix or an act — is
  just as real, just as common in what people submit, and would break the rule the other way. It
  is named here rather than built, because it is a second archetype and this session was asked for
  one. A segmentation measurement wants both.
* **One layout and one insurer.** Real insurers' contract layouts differ; one arrangement of the
  required particulars is modelled, on the same terms as the A4 confirmation's declared gap.
* **No stamp.** The signature rules are printed empty, which is 👁 the state such a scan is usually
  in before signing; a scan of a *signed* contract is a further variant not produced here.

---

## `ua_claim_bundle` — one file, two documents

`ua_insurance_contract` breaks *one page = one document* in one direction: one document across
three pages. This breaks it in the other. A file-splitting step measured on a corpus holding only
the first is **still scoring against a guarantee** — every cut it needs to make is a cut it never
has to refuse. Both directions, or the measurement is not a measurement.

👁 **And it is an observed pattern rather than a hypothesis.** Claimants really do staple the
documents of one claim into a single PDF before submitting them.

### Sources, counted honestly

**No new public source, and one practitioner observation. Saying so is the point of counting.**

* The *stitching* is 👁 **one observation of a submission practice, made by this project's author
  in their own workplace**. It is not a public source, it is not a document, and **no real
  document was read, held or copied to produce it** — what was observed is that people combine
  files, which is a habit rather than an artifact. One observation is thin evidence and it is
  recorded as one.
* The two documents *inside* the file need no new sources, because they are not new documents:
  they are `ua_invoice` and `ua_bank_payment_confirmation`, whose layouts are already evidenced
  where those archetypes are, and whose evidence this file neither adds to nor weakens.

### 🔴 The stitching adds nothing, and that is the whole difficulty

Compare the contract: it numbers its own pages, repeats its own number on each of them, and
carries a running head. **The document tells you which pages belong together.**

Here there is no cover page, no continuous pagination, no unifying header and no page numbering
across the file. Each page carries only its own document's marks, and **nothing on the file says
the two are different documents**. The only mark the stitching leaves is the gap between sheets —
one CSS rule, and the entire visual difference between this file and the two documents in it.

That asymmetry is why both archetypes are needed. One is a document that *insists* its pages are
one thing; the other is a file that says *nothing at all* about what its pages are.

### The two pages are one claim by construction, not by coincidence

Six things tie them, and **every one now comes from the builders** rather than from the render
script: one vendor instance passed to both; one `PartyIdentity` passed to both, so the firm named
on both pages carries one tax code and one account; one buyer named identically; the invoice's own
total handed to the confirmation as its transfer, which is the order the assembler uses; the
confirmation dated four days after the invoice, because an invoice is issued and then settled; and
the invoice's own number and date written into the payment purpose by the builder that was told
what the payment cites.

🔴 **Two of those six used to be patched here by hand, and removing the patch is the point.** The
script set the payment purpose itself, because the builder drew that number independently and
would otherwise have printed a purpose naming an invoice that is not in the file. A mock-up that
repairs what the shipped builders get wrong **reports a link the generator does not produce** —
and every eye spent checking the picture is spent confirming the patch rather than the generator.
The one thing it could not patch was the pair of identifiers, and that is what made the defect
visible.

The render script still makes one choice about what to *photograph*: it redraws the confirmation
until the drawn purpose is one that names a document at all. Two of the five configured purposes
name none, which is deliberate and is a real part of the corpus — but a bundle showing one of them
shows a pair with nothing tying its pages. **The number is still the builder's.**

### 🔴 What putting the two on one page revealed about the shipped generator — and what it prints now

**This section is kept as the record of a fixed defect.** The figures below are what the broken
builders produced; they are not what the current mock-up shows, and they are preserved because
they are the only account of what the defect looked like on a page.

**Then.** The invoice and the confirmation printed different identifiers for the same payee: the
invoice gave the seller `Код 15122186` and an IBAN, and the confirmation gave the same named firm
`Код 14607473` and a different IBAN. Same name, two ЄДРПОУ, two accounts.

⚠️ **That was never a property of the mock-up. It was a property of the shipped builders**, and it
reached the production corpus: `assembler` resolved a vendor once and passed that instance to both
builders, while `build_invoice` and `build_payment_confirmation` each called `generate_edrpou` and
`generate_iban` on their own. Measured afterwards on the delivered corpus, from the printed text of
every pair: **587 invoice-and-payment pairs, 587 disagreements** on the tax code and on the IBAN,
and 587 agreements on the name.

What made it worth reporting rather than shrugging at is that **the constraint was recognized and
solved for one field and not extended to its neighbours**. `resolve_vendor`'s own docstring says a
draw repeated per document *"would print two different sellers on two documents of one purchase"*,
and fixed the NAME once per vendor instance for that exact reason. The identifiers were left drawn
per document, and no test asked whether two documents of one claim agree on them.

The consequence was concrete: a downstream counterparty cross-check comparing tax codes would have
failed on **every genuine invoice-and-payment pair in the corpus**. A team measuring such a check
would have read its own correct implementation as broken — or, worse, concluded that counterparty
codes are not worth comparing.

**Now.** The seller's tax code, account and bank are drawn once per claim as a `PartyIdentity` and
carried to every builder, and the payment's purpose cites the claim's own invoice. Both pages of
the current bundle print one code and one account for one firm, and the confirmation names the
invoice on the sheet before it. The derivation of which fields this covers, which it does not, and
which are still trivially equal is `docs/cross-document-fields.md`; `tools/cross_document_audit.py`
measures any corpus against it.

⚠️ **The corpus delivered before this change still has the defect in it.** It is regenerated once,
at the start of the next iteration, and every figure quoted from it until then describes a corpus
that no longer reflects the generator.

### 👁 A domain observation: «which amount» has no answer without a key

Three documents in this repository now show the same thing in three different shapes, and it is
worth stating as a property of the domain rather than as a requirement on any system.

* **The bank statement.** Already measured: the shoulder returned the **first row** of the table
  instead of the relevant one. Many candidate numbers, one of them the claim's.
* **The insurance contract.** The largest figure is the sum insured; the money that moved is the
  premium, roughly forty times smaller and two rows below. Two candidates, and the salient one is
  the wrong one — wrong by a factor rather than by a margin.
* **This bundle.** The invoice's total and the confirmation's transfer are the **same number**,
  both correct, and they describe **one** movement of money. The failure here is not picking the
  wrong number; it is picking the right one **twice**. The architecture already states the rule —
  *a pair is one transaction, counted once* — and until now no artifact could test it.

The common factor is not that these documents are long or multi-row. It is that **a document does
not carry the key that says which of its numbers a claim is about.** The key belongs to the claim,
not to the paper: the statement needs to know which row, the contract which of two figures, the
bundle that two figures are one event. A pipeline that answers "which amount" from the document
alone is answering a question the document was never asked.

This is an observation, not a demand. What follows from it is a decision about where that key
comes from, and that decision is not this branch's.

### Narrower than reality: the bundle

* **Two documents, in the order a claimant would put them.** Three or more, and orders other than
  invoice-then-payment, are equally real and not produced.
* **Both pages are exactly A4**, because both inner archetypes are. A stitched file whose pages
  are different sizes — a phone screenshot bound after a scanned sheet — is common and not modelled.
* **Stacked PNG sheets rather than a PDF**, on the same terms as the contract: this repository
  pins ReportLab for the container, and producing it is part of connecting the archetype.
* **Nothing is degraded.** A real stitched file is usually a mix — one page scanned, one exported
  clean — and that mix is itself a segmentation cue. Here both pages are pristine renders.

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
* **The contract's second page was half empty and its third ended in mid-air.** Two separate
  defects from one render: the clause sections were unevenly distributed, and the signature block
  sat directly under the last clause instead of at the foot of the sheet. A page that ends with a
  large blank area teaches a segmenter that white space marks a page break, which is exactly the
  wrong lesson from a control built to test page breaks. The sections were rebalanced, one moved
  to the second page, and the signatures are now pushed to the foot — where the whitespace above
  them is a real property of a signed contract rather than a layout accident.
* **The contract's subject, object and territory were set in the monospaced face**, flush right
  with the figures. They are sentences, not data, and monospace made them read as data. The face
  is now chosen per row by a flag from the context rather than by position in the table.
* **The bundle's two pages named the same firm and gave it two different tax codes and two
  different IBANs.** Reading the file top to bottom is what made it visible — the two documents
  are correct apart and contradict each other together. It was the shipped builders' behaviour and
  not the mock-up's, and it reached the production corpus. **Since fixed in `src/`**: the seller's
  identity is drawn once per claim, and this mock-up's own hand-patched payment purpose was
  removed with it, because a fixture that repairs what the builders get wrong hides the thing it
  was built to show. See `docs/cross-document-fields.md`.
* **The invoice page is blank for its lower half.** That is the shipped `ua_invoice` archetype's
  own proportion with a three-line basket, carried through unchanged — the bundle neither caused
  it nor hides it, and a stitched file really does contain whatever its parts look like.
