"""The VAT summary row takes two forms, and the medium decides which — asymmetrically.

👁 the evidence, which is what this whole module is about:

    A — delivered online — `ПДВ А 20%`     (spaced)
    B — paper            — `ПДВ А=20,00%`  (equals)
    C — paper            — `ПДВ А=20,00%`  (equals)

three of the author's own receipts, each a separate installation, which is what makes them the
strongest evidence available for a question about print format. So:

    paper       → the equals form only.  Two independent observations, no counterexample.
    electronic  → both, drawn.           One observation, and one cannot support a rule.

🔴 the asymmetry is the property under test. A symmetric rule — "electronic ⇒ spaced" — would read
better and rest on n = 1, which is the over-claim that produced the single unconditional form this
replaces. The tests below therefore assert the paper side exhaustively and the electronic side as a
draw, and one of them exists only to fail if somebody tidies the two into agreement.

⚠️ nothing asserted the printed row before this module. The form was changed once already, on a
denominator that turned out to be wrong, and no test read the string it printed — so the change was
invisible to the suite in both directions.
"""

from __future__ import annotations

import random
import re
from datetime import datetime

import pytest

from receipt_synth.config import jurisdiction
from receipt_synth.content_builder import (
    build_prro_receipt,
    draw_party_identity,
    resolve_vendor,
)
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture, Medium

UA = jurisdiction("UA")
FORMS = UA["tax_line_label_forms"]
BY_MEDIUM = UA["tax_line_forms_by_medium"]

PAYER = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
NON_PAYER = {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False}
WHEN = datetime(2026, 6, 3, 14, 22, 51)

# Enough seeds that a draw with any plausible weighting shows both outcomes. Stated as a constant
# because two tests below report it as their denominator.
SEEDS = 60


def make(seed: int, capture: Capture, vendor: dict = PAYER):
    rng = random.Random(seed)
    resolved = resolve_vendor(rng, vendor, "UA")
    return build_prro_receipt(
        rng,
        category_id="vitamins_nutrition",
        issued_at=WHEN,
        vendor=resolved,
        identity=draw_party_identity(rng, resolved, "UA"),
        address="м. Київ",
        capture=capture,
    )


def forms_over_seeds(capture: Capture) -> list[str | None]:
    return [make(seed, capture).vat_row_form for seed in range(SEEDS)]


# ------------------------------------------------------- the medium of a channel --


def test_every_capture_channel_declares_what_it_captures():
    """A channel with no declared medium would be treated as a screen by omission, and the receipt
    requisite that depends on it would be chosen by an oversight. The sweep's denominator is the
    whole enum, so a channel added later is covered without this test being touched."""
    for capture in Capture:
        assert isinstance(capture.medium, Medium), capture

    assert {c.medium for c in Capture} == {Medium.PAPER, Medium.ELECTRONIC}, (
        "one of the two media is unreachable, so half the rule below cannot be exercised"
    )
    assert Capture.PHOTO.medium is Medium.PAPER
    assert Capture.SCAN.medium is Medium.PAPER
    assert Capture.SCREENSHOT.medium is Medium.ELECTRONIC


def test_every_medium_has_forms_declared_for_it():
    """Config completeness, over the media the enum can produce rather than over the map's own
    keys — a map checked against itself would pass however incomplete it was."""
    for medium in {capture.medium for capture in Capture}:
        assert BY_MEDIUM.get(medium.value), medium
        for form in BY_MEDIUM[medium.value]:
            assert form in FORMS, f"{medium.value} may draw {form!r}, which is not a declared form"


# --------------------------------------------------------------- the asymmetry --


@pytest.mark.parametrize("capture", [Capture.PHOTO, Capture.SCAN])
def test_paper_takes_the_equals_form_and_never_the_other(capture):
    """👁 two independent paper installations, zero counterexamples — so this side is exhaustive
    rather than a draw. The combination paper + spaced does not occur in the data because it was
    never observed, which is the concrete thing the asymmetry buys."""
    drawn = set(forms_over_seeds(capture))

    assert drawn == {"equals"}, (
        f"{capture.value} produced {sorted(drawn)} over {SEEDS} seeds; paper takes one form"
    )


def test_an_electronic_receipt_draws_both_forms():
    """👁 one electronic observation, so the rule refuses to choose: both forms occur.

    This is the test that would go red if somebody made the rule symmetric — "electronic ⇒ spaced"
    reads better and would rest on a single receipt. The counts are reported with their denominator
    rather than asserted as a ratio: `rng.choice` over the permitted forms is uniform by
    consequence, and pinning a proportion would state a frequency nothing has measured.
    """
    drawn = forms_over_seeds(Capture.SCREENSHOT)
    seen = {form: drawn.count(form) for form in set(drawn)}

    assert set(drawn) == set(FORMS), (
        f"over {SEEDS} seeds an electronic receipt produced {seen}, not both declared forms"
    )
    for form, count in seen.items():
        assert count > 0, form


def test_the_two_forms_are_different_strings():
    """The guard the two tests above rest on. Were the formats made identical — by a tidy-up, or by
    someone resolving the asymmetry — every assertion here would still pass while measuring
    nothing, because both media would print the same row."""
    assert FORMS["equals"] != FORMS["spaced"]
    assert "=" in FORMS["equals"] and "=" not in FORMS["spaced"]


# ----------------------------------------------------- the label and the page --


@pytest.mark.parametrize("capture", list(Capture))
def test_the_label_carries_the_form_that_was_chosen(capture):
    """The point of labelling it: a consumer can filter on it. A system that learned one form fails
    on the other, and without this field nothing in the labels would explain why."""
    for seed in range(12):
        receipt = make(seed, capture)
        record = receipt.ground_truth(
            doc_id="d", source_file="d.png", capture=capture, field_bboxes={}
        )
        assert record.vat_row_form == receipt.vat_row_form
        assert record.vat_row_form in BY_MEDIUM[capture.medium.value]


def test_a_seller_that_is_not_registered_has_no_form_at_all():
    """`None` is the same statement the empty `tax_lines` makes: there is no tax block to take a
    form. A non-payer's receipt would otherwise carry a label field describing a row it does not
    print — the defect that gave the statement its second money column."""
    for capture in Capture:
        receipt = make(1, capture, vendor=NON_PAYER)
        assert receipt.tax_lines == []
        assert receipt.vat_row_form is None
        assert receipt.ground_truth(
            doc_id="d", source_file="d.png", capture=capture, field_bboxes={}
        ).vat_row_form is None


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_printed_row_is_the_form_the_label_declares(form):
    """🔴 the label and the page, checked against each other. A form recorded in the label while a
    different one is printed is exactly the class of defect the cross-document check found on the
    counterparty: internally consistent on each side and wrong between them.

    Rendered rather than read off `render_context`, so the string asserted is the one that reaches
    the image.
    """
    capture = next(
        c for c in Capture if form in BY_MEDIUM[c.medium.value]
    )
    receipt = next(
        r for r in (make(seed, capture) for seed in range(SEEDS)) if r.vat_row_form == form
    )
    with Renderer() as renderer:
        html = renderer.build_html("ua_prro_receipt", receipt.render_context())

    printed = re.sub(r"<[^>]+>", " ", html)
    expected = FORMS[form].format(
        name=UA["vat_letters"]["А"]["name"],
        letter="А",
        rate=20.0,
        rate_2dp=f"{20.0:.2f}".replace(".", receipt.decimal_separator),
    )
    assert expected in printed, f"{expected!r} is not on the page for form {form!r}"

    other = next(f for f in FORMS if f != form)
    unexpected = FORMS[other].format(
        name=UA["vat_letters"]["А"]["name"],
        letter="А",
        rate=20.0,
        rate_2dp=f"{20.0:.2f}".replace(".", receipt.decimal_separator),
    )
    assert unexpected not in printed, f"both forms are on one page: {unexpected!r}"
