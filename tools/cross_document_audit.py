#!/usr/bin/env python3
"""Measure, on a generated corpus, whether the documents of one claim agree with each other.

    uv run python tools/cross_document_audit.py out

🔴 IT READS THE OUTPUT AND NOT THE BUILDER. Every value below is pulled out of
`reference_text` — the printed characters of the page, recorded by the layout engine that
produced the image — with the regexes a consumer would have to write. A comparison that read
the builder's own fields instead could not fail: the two sides would be the same object, and
the whole class of defect this tool exists for is precisely a field that is *drawn twice*
rather than *printed wrong*.

WHAT A ROW MEANS, AND WHY THERE ARE THREE OUTCOMES RATHER THAN TWO. A cross-document field is
only useful for linking documents to one claim if it can be *resolved* — which means it has to
be capable of both agreeing and disagreeing, and of being read from two different renderings.
So each row reports:

    readable   pairs where the field could be read off BOTH documents
    agree      pairs where the two readings are equal

and the useful states are then:

    agree == readable == pairs   the field matches TRIVIALLY. A selector built on it scores
                                 perfectly by construction and measures nothing.
    agree == 0                   the field NEVER matches. A selector built on it scores zero by
                                 construction and measures nothing either.
    0 < agree < readable, and    the field is resolvable and the score is EARNED. This is the
    readable < pairs             only state worth having, and it needs the field to be absent or
                                 differently spelled on some pairs for honest reasons.

`docs/cross-document-fields.md` holds the derivation of which fields belong here at all.
Nothing in `src/` imports this file: it is an instrument for auditing a run, not a stage.

Exits 0 when the corpus was read, 2 on a usage error. It reports; it does not judge — the
question of which state a row *should* be in is a design decision and lives in the doc.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- reading the page --
#
# One regex per printed block, written against the document's own layout. `Код банку` is
# never `Код`: the МФО is six digits under a caption that carries eight and ten elsewhere on
# the same page, so every code pattern states its length rather than trusting the caption.

INVOICE_PAYEE = re.compile(r"Одержувач\t(.+?)\tКод\t(\d{8,10})\b")
INVOICE_PAYEE_BANK = re.compile(r"Банк одержувача\t(.+?)\tКод банку\t(\d{6})\b")
INVOICE_ACCOUNT = re.compile(r"КРЕДИТ рах\. №\t(UA\d{27})\b")
INVOICE_TITLE = re.compile(r"Рахунок на оплату № *(\S+) від (.+?) р\.")
INVOICE_BUYER = re.compile(r"Покупець:\n(.+?), РНОКПП (\d{10})\b")

CONFIRMATION_PARTY = re.compile(
    r"^(Платник|Отримувач)\n(.+?)\n(?:Код (\d{8,10})\n)?(?:IBAN (UA\d{27})\n)?"
    r"(?:Банк отримувача (.+?), Код банку (\d{6})\n)?",
    re.MULTILINE,
)
STATEMENT_HOLDER = re.compile(r"Клієнт (.+?), РНОКПП (\d{10})\b")

# What a purpose line cites. 🔴 `рахунок` and `ВН` are DIFFERENT DOCUMENT CLASSES — an invoice
# and a delivery note — so a pattern matching «№» alone would report a delivery note as the
# claim's invoice and call a wrong link a right one.
#
# ⚠️ THE NUMBER IS MATCHED BY ITS OWN SHAPE, not by what follows it. The first version ended
# `(?:,| від <date>|$| )` — a list of the things that can come after the number — and `$` without
# `re.MULTILINE` is the END OF THE WHOLE PAGE, not the end of the line. So a purpose ending in the
# number, which is what three of the five configured templates produce, matched only when it
# happened to be the last text on the document: **99 of 163 citations went unread**. It never
# affected the finding it was written for — nothing agreed either way — but it understated the
# COVERAGE threefold, and coverage is the figure that says how much of a linking score is earned.
# Describe the token you want; do not enumerate its neighbours.
CITED_INVOICE = re.compile(r"рахунку № *([0-9A-Za-z/-]+)")


@dataclass(frozen=True)
class Row:
    """One cross-document field, as the audit reports it."""

    key: str
    subject: str | None
    payment: str | None


def _invoice_fields(text: str) -> dict[str, str | None]:
    payee = INVOICE_PAYEE.search(text)
    bank = INVOICE_PAYEE_BANK.search(text)
    account = INVOICE_ACCOUNT.search(text)
    title = INVOICE_TITLE.search(text)
    buyer = INVOICE_BUYER.search(text)
    return {
        "seller_name": payee.group(1) if payee else None,
        "seller_tax_code": payee.group(2) if payee else None,
        "seller_bank_name": bank.group(1) if bank else None,
        "seller_bank_code": bank.group(2) if bank else None,
        "seller_account": account.group(1) if account else None,
        "invoice_number": title.group(1) if title else None,
        "payer_name": buyer.group(1) if buyer else None,
        "payer_tax_code": buyer.group(2) if buyer else None,
    }


def _confirmation_fields(text: str) -> dict[str, str | None]:
    """The two party blocks of a confirmation, read by caption.

    ⚠️ THE PAYER BLOCK CAN BE A HYPHEN. On the internet-acquiring mode the payer is not
    identified at all, and a pattern that simply took "the next Код after Платник" would run
    on into the RECIPIENT's block and report the payee's code as the payer's. That is not a
    hypothetical: the first version of this reader did exactly that and manufactured 32
    disagreements out of a corpus that had none.
    """
    parties = {
        match.group(1): match for match in CONFIRMATION_PARTY.finditer(text + "\n")
    }
    payer, payee = parties.get("Платник"), parties.get("Отримувач")
    purpose = CITED_INVOICE.search(text)
    return {
        "seller_name": payee.group(2) if payee else None,
        "seller_tax_code": payee.group(3) if payee else None,
        "seller_bank_name": payee.group(5) if payee else None,
        "seller_bank_code": payee.group(6) if payee else None,
        "seller_account": payee.group(4) if payee else None,
        "invoice_number": purpose.group(1) if purpose else None,
        "payer_name": payer.group(2) if payer else None,
        "payer_tax_code": payer.group(3) if payer else None,
    }


def _statement_fields(text: str, relevant: str | None) -> dict[str, str | None]:
    """The LABELLED row of a statement, located by the operation number the label points at.

    The page carries a dozen rows naming a dozen other counterparties; only one of them is the
    claim's. `relevant_transaction` is what the label uses to say which, so the audit joins the
    same way a consumer would, rather than searching for the vendor's name — searching by name
    would presuppose the very agreement being measured.
    """
    holder = STATEMENT_HOLDER.search(text)
    fields: dict[str, str | None] = {
        "seller_name": None,
        "seller_tax_code": None,
        "seller_bank_name": None,
        "seller_bank_code": None,
        "seller_account": None,
        "invoice_number": None,
        "payer_name": holder.group(1) if holder else None,
        "payer_tax_code": holder.group(2) if holder else None,
    }
    if relevant is None:
        return fields

    # A row runs from its operation number to the blank line before the next one. 👁 The
    # counterparty block prints name, code, IBAN and bank on four lines in that order.
    block = re.search(
        rf"^{re.escape(relevant)}\t.*?(?=\n\n|\Z)", text, re.MULTILINE | re.DOTALL
    )
    if block is None:
        return fields
    body = block.group(0)
    code = re.search(r"\n(\d{8,10})\n", body)
    account = re.search(r"\b(UA\d{27})\b", body)
    purpose = CITED_INVOICE.search(body)
    lines = [line for line in body.split("\n") if line.strip()]
    # 👁 The row wraps over several lines: number and date, then the time, then the money columns
    # with the purpose and the counterparty's NAME as the last tab-separated field of that line,
    # then the code, the account and the bank. So the name is taken from the line carrying the
    # money columns — identified by its tab count — and not from the first line, which holds the
    # operation date and would silently report a date as a counterparty.
    money_line = next((line for line in lines if line.count("\t") >= 3), None)
    fields |= {
        "seller_name": money_line.split("\t")[-1].strip() if money_line else None,
        "seller_tax_code": code.group(1) if code else None,
        "seller_account": account.group(1) if account else None,
        "seller_bank_name": lines[-1].strip() if lines else None,
        "invoice_number": purpose.group(1) if purpose else None,
    }
    return fields


_SUBJECT_TYPES = {"invoice"}
_PAYMENT_TYPES = {"payment_confirmation", "bank_statement"}
FIELDS = (
    "seller_name",
    "seller_tax_code",
    "seller_account",
    "seller_bank_name",
    "seller_bank_code",
    "payer_name",
    "payer_tax_code",
    "invoice_number",
)


def fields_of(document: dict) -> dict[str, str | None]:
    doc_type = document["doc_type"]
    text = document["reference_text"]
    if doc_type == "invoice":
        return _invoice_fields(text)
    if doc_type == "payment_confirmation":
        return _confirmation_fields(text)
    if doc_type == "bank_statement":
        return _statement_fields(text, document.get("relevant_transaction"))
    raise ValueError(f"no reader for document type {doc_type!r}")


def audit(corpus: Path) -> dict:
    ground_truth = json.loads((corpus / "ground_truth.json").read_text())
    by_claim: dict[str, list[dict]] = collections.defaultdict(list)
    for document in ground_truth["documents"]:
        by_claim[document["doc_id"].rsplit("_d", 1)[0]].append(document)

    shapes: dict[str, dict] = {}
    for claim_id, documents in sorted(by_claim.items()):
        subject = next((d for d in documents if d["doc_type"] in _SUBJECT_TYPES), None)
        payment = next((d for d in documents if d["doc_type"] in _PAYMENT_TYPES), None)
        if subject is None or payment is None:
            continue
        shape = f"{subject['doc_type']} + {payment['doc_type']}"
        report = shapes.setdefault(
            shape,
            {
                "pairs": 0,
                "rows": {
                    key: {"readable": 0, "agree": 0, "examples": []} for key in FIELDS
                },
            },
        )
        report["pairs"] += 1
        left, right = fields_of(subject), fields_of(payment)
        for key in FIELDS:
            a, b = left[key], right[key]
            if a is None or b is None:
                continue
            row = report["rows"][key]
            row["readable"] += 1
            if a == b:
                row["agree"] += 1
            elif len(row["examples"]) < 3:
                row["examples"].append(f"{claim_id}: {a!r} / {b!r}")
    return shapes


def state_of(readable: int, agree: int) -> str:
    """What the two counts say about this row, and NOTHING BEYOND THEM.

    ⚠️ These are not the three design states of `docs/cross-document-fields.md`. Counts cannot
    tell resolvable from trivial: a field printed identically on both pages and a field that
    has to be parsed out of a free-text purpose line both come back ALWAYS AGREES, and a field
    drawn from a four-value list agrees a quarter of the time by coincidence. Which state a row
    is in is a judgement about HOW the value has to be read, and the doc makes it; this
    function reports what was counted so the doc can be checked against it.
    """
    if readable == 0:
        return "NOT READABLE"
    if agree == 0:
        return "NEVER AGREES"
    if agree == readable:
        return "ALWAYS AGREES"
    return "SOMETIMES AGREES"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_document_audit",
        description=(
            "Report, per cross-document field, how often the two documents of one claim "
            "agree — read from reference_text, never from the builder"
        ),
    )
    parser.add_argument("corpus", type=Path, help="a directory holding ground_truth.json")
    parser.add_argument(
        "--examples", action="store_true", help="print up to three disagreements per row"
    )
    args = parser.parse_args(argv)

    if not (args.corpus / "ground_truth.json").is_file():
        parser.error(f"{args.corpus} holds no ground_truth.json")

    shapes = audit(args.corpus)
    if not shapes:
        print(f"{args.corpus}: no claim carries a subject document and a payment document")
        return 0

    for shape, report in sorted(shapes.items()):
        pairs = report["pairs"]
        print(f"\n{shape} — {pairs} pairs")
        print(f"  {'field':<18}{'readable':>10}{'agree':>8}   state")
        for key in FIELDS:
            row = report["rows"][key]
            state = state_of(row["readable"], row["agree"])
            print(f"  {key:<18}{row['readable']:>10}{row['agree']:>8}   {state}")
            if args.examples:
                for example in row["examples"]:
                    print(f"      {example}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
