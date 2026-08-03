"""One bank name, one bank code — everywhere it is drawn, not only within one call.

🔴 THE DEFECT THIS FILE EXISTS FOR: `draw_party_identity`, `build_payment_confirmation` and
`build_bank_statement` each used to draw a bank's NAME and its МФО (`bank_code`) INDEPENDENTLY.
Every single call was internally consistent — the account it built really did carry the code it
drew — so nothing local to one call could see the problem. What broke is the fact TWO DIFFERENT
calls have no reason to agree: measured on a delivered run, 7 claims of 60 printed one real bank
name with two different codes, because the payer's own bank (drawn in the payment document) and
the claim's payee bank (drawn in `draw_party_identity`) happened to name the same institution.

THE FIX IS A LOOKUP, NOT A STRICTER DRAW. `config.bank_codes` is a fixed, invented name → code
table (`config/vendors.json` `banks.<country>[].bank_code`); every site above draws the NAME with
`rng.choice(banks(country))` exactly as before and then looks the CODE up, so a code can no longer
differ between two draws of the same name — not because the odds improved, but because there is
only one value to draw from.

Every expectation below is checked against `config.bank_codes` directly — the table IS the known
answer here, unlike a drawn value, because it is not seeded: it is a fixed fact of the
configuration and every builder is required to agree with it.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal

from receipt_synth.config import bank_codes, banks
from receipt_synth.content_builder import (
    build_bank_statement,
    build_payment_confirmation,
    draw_party_identity,
    resolve_vendor,
)

VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
CLAIMANT = "Ковальчук Олена Петрівна"
CLAIMANT_CODE = "2345678901"
WHEN = datetime(2026, 6, 11, 14, 33)
# Enough seeds that, with a FOUR-name pool, the same name is drawn twice across independent calls
# many times over — see `test_content_builder`'s sibling note on cardinality. Four seeds proved too
# few for an equality test elsewhere in this repository (`test_cross_document_identity.SEEDS`); 40
# is comfortably past the point of relying on luck, and the test below asserts the coincidence
# actually occurred rather than assuming it did.
SEEDS = range(40)


def _draws(seed: int):
    """One identity, one confirmation issuer and one statement issuer, drawn the way the
    assembler draws them — the identity ONCE per claim, the payment document's own bank as a
    SEPARATE call, exactly the two independent draws the defect lived between."""
    rng = random.Random(seed)
    vendor = resolve_vendor(rng, dict(VENDOR), "UA")
    identity = draw_party_identity(rng, vendor, "UA")
    confirmation = build_payment_confirmation(
        rng,
        issued_at=WHEN,
        vendor=vendor,
        identity=identity,
        payer_name=CLAIMANT,
        payer_tax_id=CLAIMANT_CODE,
        amount=Decimal("100.00"),
        initiation="transfer",
    )
    rng2 = random.Random(seed + 10_000)
    vendor2 = resolve_vendor(rng2, dict(VENDOR), "UA")
    identity2 = draw_party_identity(rng2, vendor2, "UA")
    statement = build_bank_statement(
        rng2,
        issued_at=WHEN,
        vendor=vendor2,
        identity=identity2,
        payer_name=CLAIMANT,
        payer_tax_id=CLAIMANT_CODE,
        amount=Decimal("50.00"),
    )
    return identity, confirmation, statement


# ---------------------------------------------------------------- the table itself --


def test_the_tables_codes_are_six_digits_and_unique():
    """The invented codes have to be shaped like a real МФО (six digits, config/fiscal-rules.yaml
    `identifiers.bank_code`) and distinct — a repeated code would make two different real banks
    print as one for any consumer keying on it, which defeats the point of inventing one per name.
    """
    codes = bank_codes("UA")
    assert set(codes) == set(banks("UA")), "the table names a different set of banks than banks()"
    assert len(codes) >= 4, "too few banks to ever exercise a coincidence"
    for name, code in codes.items():
        assert code.isdigit() and len(code) == 6, f"{name!r} carries {code!r}, not six digits"
    assert len(set(codes.values())) == len(codes), (
        f"two names share a code: {sorted(codes.items())}"
    )


# ---------------------------------------------------------------- each site, against the table --


def test_a_drawn_identity_takes_its_code_from_the_table():
    for seed in SEEDS:
        rng = random.Random(seed)
        vendor = resolve_vendor(rng, dict(VENDOR), "UA")
        identity = draw_party_identity(rng, vendor, "UA")
        assert identity.bank_code == bank_codes("UA")[identity.bank_name]
        assert identity.account[4:10] == identity.bank_code


def test_a_confirmations_own_issuer_takes_its_code_from_the_table():
    for seed in SEEDS:
        identity, confirmation, _ = _draws(seed)
        assert confirmation.bank_code == bank_codes("UA")[confirmation.bank_name]


def test_a_statements_own_issuer_takes_its_code_from_the_table():
    for seed in SEEDS:
        _, _, statement = _draws(seed)
        assert statement.bank_code == bank_codes("UA")[statement.bank_name]
        assert statement.account[4:10] == statement.bank_code


# ------------------------------------------------------- the cross-call invariant itself --


def test_the_same_bank_name_never_carries_two_codes_across_independent_draws():
    """🔴 THE ROW THIS FILE IS ACTUALLY FOR. `draw_party_identity` (the claim's payee) and
    `build_payment_confirmation` (the same claim's payer, issued by the payer's own bank) are TWO
    INDEPENDENT DRAWS from the same four-name pool — nothing ties them together except that both
    now read the same table. Across many seeds the two coincide on a name repeatedly; when they
    do, this asserts the code came out identical, and it also asserts that the coincidence
    happened, so a pool change that stopped producing it could not turn this into a check of
    nothing (the fixture would fail on `collisions` before it could pass on silence).
    """
    seen: dict[str, str] = {}
    collisions = 0
    for seed in SEEDS:
        identity, confirmation, statement = _draws(seed)
        for name, code in (
            (identity.bank_name, identity.bank_code),
            (confirmation.bank_name, confirmation.bank_code),
            (statement.bank_name, statement.bank_code),
        ):
            if name in seen:
                collisions += 1
                assert seen[name] == code, (
                    f"seed {seed}: {name!r} was first seen with code {seen[name]!r}, now prints "
                    f"{code!r}"
                )
            else:
                seen[name] = code
    assert collisions > 10, (
        f"only {collisions} coincidences in {len(list(SEEDS))} seeds — not enough independent "
        "draws landed on the same name to be a positive control for this test"
    )
