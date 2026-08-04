"""The banking application's two carriers of the payment_confirmation class.

Two archetypes, one phone, opposite arguments. `ua_bank_app_transaction` is the negative
example by LACK of requisites — it looks like proof of payment and carries none of the
marks proof of payment is recognized by — so its tests are mostly about what is absent,
checked on the collected fields and the label. `ua_bank_receipt_in_app` is the A4
confirmation itself inside the app's frame, so its tests are about IDENTITY: the same
builder, the same label, one document on two carriers — asserted by construction, not by
resemblance.

The weighted archetype draw is tested here too, because these archetypes are why it
exists: four renderings of one class role, drawn at the shares generation.yaml declares.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal

import pytest

from receipt_synth.claim_planner import (
    ARCHETYPES,
    Evidence,
    _draw_archetype,
    evidence_of,
)
from receipt_synth.config import archetype_draw_weights, load_vendors
from receipt_synth.content_builder import (
    build_app_transaction,
    build_bank_receipt_in_app,
    build_payment_confirmation,
    descriptor_prefixes,
    draw_party_identity,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, Direction, DocType

VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
WHEN = datetime(2026, 6, 15, 13, 46, 0)


def identity(seed: int = 606):
    return draw_party_identity(random.Random(seed), VENDOR, "UA")


def make_transaction(seed: int = 20260615, amount: str = "600.00"):
    return build_app_transaction(
        random.Random(seed),
        issued_at=WHEN,
        vendor=VENDOR,
        identity=identity(),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        amount=Decimal(amount),
    )


def make_framed(seed: int = 20260417):
    return build_bank_receipt_in_app(
        random.Random(seed),
        issued_at=WHEN,
        vendor=VENDOR,
        identity=identity(),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        amount=Decimal("600.00"),
    )


# ================================================ registration ==


def test_both_carriers_are_registered_as_the_confirmation_class():
    """Neither is a new document type — the four classifier target classes stay four —
    and both carry the type's default evidence, there being no per-archetype override."""
    for slug in ("ua_bank_app_transaction", "ua_bank_receipt_in_app"):
        archetype = ARCHETYPES[slug]
        assert archetype.doc_type is DocType.PAYMENT_CONFIRMATION, slug
        assert evidence_of(archetype) == Evidence(False, True), slug


def test_the_payment_pool_is_fully_listed_in_the_share_table():
    """The all-or-none rule of `_draw_archetype` bites the day someone registers a fifth
    payment archetype without declaring its share; this is the tripwire that says so at
    the registry rather than mid-run."""
    weights = archetype_draw_weights()
    payment_slugs = {
        slug
        for slug, archetype in ARCHETYPES.items()
        if evidence_of(archetype) == Evidence(False, True)
    }
    assert payment_slugs, "no payment archetypes — this test would assert nothing"
    missing = sorted(payment_slugs - set(weights))
    assert not missing, f"payment archetypes without a declared share: {missing}"


def test_a_half_declared_pool_is_refused_and_an_undeclared_one_draws_uniformly():
    """The all-or-none rule itself, on hand-built pools: a table listing SOME members of
    a pool is a distribution nobody chose."""
    listed = ARCHETYPES["ua_bank_payment_confirmation"]
    unlisted = ARCHETYPES["ua_prro_receipt"]

    with pytest.raises(ValueError, match="half-declared"):
        _draw_archetype(random.Random(1), [listed, unlisted])

    fiscal = [ARCHETYPES["ua_prro_receipt"], ARCHETYPES["ua_rro_receipt"]]
    drawn = {_draw_archetype(random.Random(seed), fiscal).slug for seed in range(20)}
    assert drawn == {"ua_prro_receipt", "ua_rro_receipt"}


def test_the_declared_weights_are_actually_applied():
    """Not a statistical test: with 40 seeds, an 8% archetype and a 50% one must separate
    visibly, and a uniform draw over four (25% each) puts the counts far from both. The
    exact shares are the author's parameters; what this pins is that the table reaches
    the draw at all — a mutation replacing the weighted draw with `rng.choice` survives
    every other test in this file."""
    payments = [
        archetype
        for archetype in ARCHETYPES.values()
        if evidence_of(archetype) == Evidence(False, True)
    ]
    counts: dict[str, int] = {}
    for seed in range(200):
        slug = _draw_archetype(random.Random(seed), payments).slug
        counts[slug] = counts.get(slug, 0) + 1
    # 0.50 vs 0.08 over 200 draws: the statement must dominate the app screen by a wide
    # margin, and uniform (50 each) fails the first inequality.
    assert counts.get("ua_bank_statement", 0) > 70
    assert counts.get("ua_bank_app_transaction", 0) < 40


# ================================================ the transaction screen ==


def test_the_screen_carries_none_of_the_requisites_proof_is_recognized_by():
    """The archetype's whole property, on the label: no document code, no authorization
    code, no purpose, no fee, no payer — and the direction the printed minus states."""
    truth = make_transaction().ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.doc_type is DocType.PAYMENT_CONFIRMATION
    assert truth.payment_purpose is None
    assert truth.auth_code is None
    assert truth.fee is None
    assert truth.payer is None
    assert truth.direction is Direction.DEBIT
    assert truth.line_items == []


def test_the_descriptor_is_a_public_prefix_and_the_label_keeps_the_bare_name():
    """🔴 The split the archetype exists for: the PAGE prints `PREFIX*NAME` — the
    counterparty as the card network carries it — while the LABEL keeps the bare trade
    name every other document of the claim agrees on. The prefix comes from the public
    processor pools of config/vendors.json, never invented."""
    document = make_transaction()
    prefix, _, tail = document.merchant_descriptor_text.partition("*")
    assert prefix in descriptor_prefixes()
    assert tail == VENDOR["name"].upper()
    assert document.counterparty == VENDOR["name"]


def test_the_amount_is_the_claims_and_never_drawn():
    with pytest.raises(ValueError, match="amount is the claim's"):
        build_app_transaction(
            random.Random(1),
            issued_at=WHEN,
            vendor=VENDOR,
            identity=identity(),
            payer_name="X",
            payer_tax_id="1",
            amount=None,
        )


def test_the_same_seed_builds_the_same_screen():
    assert make_transaction() == make_transaction()


# ================================================ the framed receipt ==


def test_the_inner_document_is_the_a4_confirmation_itself():
    """THE CARRIER-INVARIANCE ARGUMENT, by construction: the framed builder consumes the
    same keywords through the same inner builder, so the same seed yields the SAME
    confirmation — equal content object, equal label — and the frame adds only chrome
    draws after it. If these two ever diverge, the archetype has lost its reason to
    exist."""
    seed = 20260417
    direct = build_payment_confirmation(
        random.Random(seed),
        issued_at=WHEN,
        vendor=VENDOR,
        identity=identity(),
        payer_name="Ковальчук Олена Петрівна",
        payer_tax_id="2345678901",
        amount=Decimal("600.00"),
    )
    framed = make_framed(seed)
    assert framed.inner == direct

    kwargs = dict(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert framed.ground_truth(**kwargs) == direct.ground_truth(**kwargs)


def test_the_frame_collects_the_documents_fields_and_its_own_chrome(tmp_path):
    """One render, one coordinate space: every field of the inner document is collected
    from the framed image, plus the chrome's `screen_title` — and the document's own
    text is in the frame's reference text, which an embedded raster could never give."""
    framed = make_framed()
    with Renderer() as renderer:
        on_paper = renderer.render(
            "ua_bank_payment_confirmation",
            framed.inner.render_context(),
            tmp_path / "paper.png",
        )
        on_screen = renderer.render(
            "ua_bank_receipt_in_app", framed.render_context(), tmp_path / "screen.png"
        )
    assert set(on_paper.field_bboxes) <= set(on_screen.field_bboxes)
    assert "screen_title" in on_screen.field_bboxes
    assert framed.inner.document_code in on_screen.reference_text
    # The screen heading repeats the document's number under the APP'S word for it.
    assert f"Квитанція № {framed.inner.document_code}" in on_screen.reference_text


def test_the_same_seed_builds_the_same_framed_receipt():
    assert make_framed() == make_framed()


# ================================================ the vendor pools stay honest ==


def test_the_descriptor_prefix_pools_are_the_public_processor_lists():
    """The prefixes are read from `payment_providers` and `aggregators` — public
    Ukrainian processors already named in vendors.json for exactly their trade — and
    never written in code, where an invented prefix would be an invented mark."""
    vendors = load_vendors()
    expected = sorted(
        entry["display_name"].upper() for entry in vendors["payment_providers"]["UA"]
    ) + sorted(entry["name"].upper() for entry in vendors["aggregators"]["UA"])
    assert sorted(descriptor_prefixes()) == sorted(expected)
