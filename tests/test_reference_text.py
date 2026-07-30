"""`reference_text`: every printed character, in reading order, from before rasterization.

🔴 WHY IT IS NOT THE FIELD BOXES, which is the property this whole module is about. `field_bboxes`
covers the LABELLED fields; `reference_text` covers ALL printed text. A document whose every
labelled field survived a crop while the footer carrying the fiscal wording was lost would report
as complete measured on the fields alone — and would then hand a system a falsely low character
error rate for text it never read.

GROUND TRUTH BY CONSTRUCTION, NOT BY ANNOTATION. It is read from the layout engine that produced
the image, in the same pass, before the screenshot. Nothing here was transcribed from a picture.

`content_bbox` is its geometric counterpart: where that text IS. ⚠️ Text and only text — a QR, a
stamp and a signature are ink it does not cover — so the two describe the same thing and a later
measurement can use them together without one covering more than the other.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from html import unescape

import pytest

from receipt_synth.claim_planner import ARCHETYPES
from receipt_synth.renderer import Renderer
from receipt_synth.schemas import Capture
from test_renderer import REGISTERED_SLUGS, context_for

# Every field marker a template can carry that is NOT text — a QR is a picture and a stamp is drawn
# geometry. `content_bbox` covers text, so these are the boxes it is not required to contain, and
# naming them is what keeps the containment assertion below honest rather than loose.
NON_TEXT_FIELDS = frozenset({"qr", "stamp"})


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as instance:
        yield instance


@pytest.fixture(scope="module")
def rendered(renderer, tmp_path_factory):
    """Every registered archetype, rendered once. The sweep's denominator is the registry, so an
    archetype added later is covered without this file being touched."""
    out = tmp_path_factory.mktemp("reference")
    return {
        slug: renderer.render(slug, context_for(slug), out / f"{slug}.png")
        for slug in REGISTERED_SLUGS
    }


def printed_values(markup: str) -> list[str]:
    """The inner text of every marked field, as a reader would see it.

    ⚠️ ENTITIES ARE UNESCAPED, and the first version of this helper did not: Jinja autoescapes, so
    an apostrophe reaches the markup as `&#39;` while `innerText` returns the character. Comparing
    the two raw made this file report a generator defect that was its own — the amount in words
    was "missing" from the page it is printed on.
    """
    return [
        unescape(re.sub(r"<[^>]+>", "", body)).strip()
        for body in re.findall(r'data-field="[^"]+"[^>]*>(.*?)<', markup, flags=re.DOTALL)
    ]


# ------------------------------------------------------------ it exists, everywhere --


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_every_archetype_carries_its_text(slug, rendered):
    """No archetype may produce a page whose text nobody recorded. Parametrized over the registry
    rather than over a list, so a template registered later cannot slip past."""
    result = rendered[slug]

    assert result.reference_text.strip(), f"{slug} produced no reference text"
    assert len(result.reference_text) > 100, f"{slug} produced {len(result.reference_text)} chars"
    assert "\n" in result.reference_text, "reading order is lost — the text is one line"


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_text_is_what_a_reader_sees_and_not_the_markup(slug, rendered):
    """🔴 `innerText`, NOT `textContent`, and the difference is measurable rather than stylistic.

    `textContent` returns the source order of every node, including whitespace the markup is
    indented with and elements CSS never paints. `innerText` is what the layout engine decided a
    reader sees. Since the premise of this dataset is that the label describes the IMAGE, the text
    has to come from the same authority that produced the image.

    MEASURED ON ONE RECEIPT, which is where these bounds come from: `innerText` gave 536 characters
    over 39 lines with ZERO runs of four spaces and ZERO indented lines; `textContent` gave 1243
    characters over 134 lines with 102 and 121 of them. A mutation swapping the two survived a test
    that only asked whether a newline appeared anywhere.

    ⚠️ BLANK LINES ARE NOT ONE OF THE SIGNALS, and the first version of this test used them. The
    bank statement's `innerText` legitimately carries 18 — its table has empty cells, and an empty
    cell is an empty line to a reader. Measured across the registry: 0 blank lines on five
    archetypes and 18 on the statement, against 0 four-space runs and 0 indented lines on ALL SIX.
    A signal that fires on a correct document is not a signal.
    """
    text = rendered[slug].reference_text

    assert not re.search(r" {4,}", text), f"{slug}: four-space runs — that is source indentation"
    assert not [ln for ln in text.splitlines() if ln.startswith("  ")], (
        f"{slug}: indented lines — the text appears to be `textContent` rather than `innerText`"
    )


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_text_covers_more_than_the_labelled_fields(slug, rendered):
    """🔴 THE PROPERTY THE FIELD BOXES CANNOT GIVE. Every archetype prints text that no `data-field`
    marks — captions, headings, the fiscal wording, a footer — and that text is exactly what a
    crop-detection measurement would otherwise miss.

    Asserted as a strict inequality with both counts reported, because "the text is non-empty" would
    pass on a page whose text was nothing but its fields.
    """
    result = rendered[slug]
    words = set(re.findall(r"\w+", result.reference_text))
    field_words = set(re.findall(r"\w+", " ".join(result.field_bboxes)))

    assert words, f"{slug}: no words in the reference text"
    unmarked = words - field_words
    assert len(unmarked) > len(result.field_bboxes) / 2, (
        f"{slug}: only {len(unmarked)} words of {len(words)} lie outside the "
        f"{len(result.field_bboxes)} field names — the page looks like nothing but its fields"
    )


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_every_printed_field_value_appears_in_the_text(slug, renderer, rendered):
    """The text is the WHOLE page, so every value a field marks has to be in it. Checked against the
    markup rather than against the label, so it compares two independent renderings of the same
    document rather than the label with itself."""
    html = renderer.build_html(slug, context_for(slug))
    values = [v for v in printed_values(html) if v and len(v) > 3 and "<" not in v]
    assert values, f"{slug}: no field values to check"

    text = rendered[slug].reference_text
    missing = [v for v in values if v not in text]
    assert not missing, (
        f"{slug}: {len(missing)} of {len(values)} printed field values are absent from the "
        f"reference text: {missing[:3]}"
    )


# ------------------------------------------------------------------ the extent --


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_content_extent_is_inside_the_page_and_not_the_page(slug, rendered):
    """It is the extent of the INK, not of the paper. A box equal to the image would be useless to
    a crop measurement — every capture would contain it — so the assertion is that it is strictly
    smaller in at least one direction, which is true of any real document with a margin."""
    result = rendered[slug]
    x, y, w, h = result.content_bbox

    assert w > 0 and h > 0, f"{slug}: degenerate content extent"
    assert x >= 0 and y >= 0
    assert x + w <= result.width and y + h <= result.height
    assert (w < result.width) or (h < result.height), (
        f"{slug}: the content extent is the whole page, so it measures nothing"
    )


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_the_content_extent_contains_every_text_field(slug, rendered):
    """The two describe the same thing, so the box that covers all text covers every TEXT field.

    ⚠️ NON-TEXT MARKERS ARE EXCLUDED BY NAME rather than by a tolerance: a QR is a picture and a
    stamp is drawn geometry, and neither is text. Listing them is what makes this assertion strict
    for everything else instead of loose for everything.
    """
    result = rendered[slug]
    cx, cy, cw, ch = result.content_bbox

    for name, (x, y, w, h) in result.field_bboxes.items():
        if name in NON_TEXT_FIELDS:
            continue
        assert x >= cx - 1 and y >= cy - 1, f"{slug}: {name} starts outside the content extent"
        assert x + w <= cx + cw + 1 and y + h <= cy + ch + 1, (
            f"{slug}: {name} ends outside the content extent"
        )


def test_a_non_text_marker_really_is_outside_the_text_extent(renderer, tmp_path):
    """The exclusion above must EXCLUDE something, or it is a list that weakens an assertion for no
    reason. The confirmation's QR sits below its last line of text, so its box genuinely falls
    outside — which is why the exemption exists rather than being defensive."""
    from receipt_synth.content_builder import build_payment_confirmation, resolve_vendor

    vendor = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
    for seed in range(40):
        rng = random.Random(seed)
        confirmation = build_payment_confirmation(
            rng,
            issued_at=datetime(2026, 4, 17, 11, 3, 9),
            vendor=resolve_vendor(rng, vendor, "UA"),
            payer_name="Ковальчук Олена Петрівна",
            payer_tax_id="2345678901",
        )
        if confirmation.qr_payload:
            break
    else:  # pragma: no cover - the share makes this unreachable
        pytest.fail("no confirmation in 40 seeds carried a QR")

    # The module's own renderer, not a second one: Playwright's sync API refuses to start inside a
    # loop it already owns, and a nested `Renderer()` is what that error means.
    result = renderer.render(
        "ua_bank_payment_confirmation", confirmation.render_context(), tmp_path / "c.png"
    )
    qx, qy, qw, qh = result.field_bboxes["qr"]
    cx, cy, cw, ch = result.content_bbox

    assert not (qy >= cy and qy + qh <= cy + ch), (
        "the QR lies inside the text extent, so excluding it from the containment check above "
        "excludes nothing and that check is weaker than it looks"
    )


# ---------------------------------------------------------------- determinism --


def test_the_same_document_yields_the_same_text(renderer, tmp_path):
    """One seed, one page, one transcript. It is read from the layout engine, so a difference here
    would mean the layout itself was not deterministic — which would move the image too."""
    context = context_for("ua_prro_receipt")
    first = renderer.render("ua_prro_receipt", context, tmp_path / "a.png")
    second = renderer.render("ua_prro_receipt", context, tmp_path / "b.png")

    assert first.reference_text == second.reference_text
    assert first.content_bbox == second.content_bbox


def test_the_reserved_key_cannot_collide_with_a_field(renderer, tmp_path):
    """The content extent rides through the degrader under a reserved key so it gets the SAME
    geometric transform as the field boxes. If a template ever marked a field by that name the two
    would silently merge, and the label would carry a field box in place of the extent."""
    from receipt_synth import assembler

    assert assembler._CONTENT_BBOX_KEY.startswith("__")
    for slug in REGISTERED_SLUGS:
        assert assembler._CONTENT_BBOX_KEY not in rendered_fields(renderer, slug), slug


def test_the_content_extent_comes_from_the_degrader_and_not_from_the_clean_render():
    """🔴 A POSITIONAL QUESTION, ANSWERED WITH `ast` RATHER THAN BY RUNNING ANYTHING.

    The extent must travel through the degrader's geometry with the field boxes, or it will drift
    from them the moment a real geometric step arrives. TODAY IT CANNOT BE OBSERVED IN OUTPUT: the
    only calibrated capture channel is `screenshot`, which applies no geometry, so a version that
    took the box straight from the clean render produces identical labels — a mutation doing
    exactly that survived every behavioural test, and it survived for that reason rather than for a
    weak assertion.

    So the property is asserted where it is decided: `content_bbox` is bound from the DEGRADED
    result, and never from `clean`. Reading the source is the only instrument that can see this
    while the difference is invisible at runtime — the same reasoning `lessons.md` records for
    questions about where something lives.
    """
    import ast
    import inspect

    from receipt_synth import assembler

    tree = ast.parse(inspect.getsource(assembler._build_document))
    bindings = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "content_bbox" for t in node.targets)
    ]
    assert len(bindings) == 1, f"{len(bindings)} bindings of `content_bbox` — expected exactly one"

    source = ast.unparse(bindings[0].value)
    assert "clean" not in source, (
        f"`content_bbox` is bound from {source!r}, which bypasses the degrader's geometry; it must "
        "come from the transformed boxes"
    )
    assert "boxes" in source and "pop" in source, (
        f"`content_bbox` is bound from {source!r}; it should be popped out of the degraded boxes"
    )


def rendered_fields(renderer: Renderer, slug: str) -> set[str]:
    html = renderer.build_html(slug, context_for(slug))
    return set(re.findall(r'data-field="([^"]+)"', html))


def test_the_label_carries_both_and_they_survive_the_degrader(renderer, tmp_path):
    """End to end: what the renderer read reaches the label, and the extent comes back from the
    degrader's geometry pipeline rather than bypassing it. `capture` is a screenshot here, which
    applies no geometry — so the box is unchanged, and the assertion is that it went THROUGH."""
    from receipt_synth import assembler
    from receipt_synth.claim_planner import ClaimPlan, DocumentPlan, evidence_of
    from receipt_synth.persona_generator import generate_persona
    from receipt_synth.schemas import Country, Verdict

    receipt = next(a for a in ARCHETYPES.values() if evidence_of(a) == (True, True))
    when = datetime(2026, 6, 15, 12, 0)
    plan = ClaimPlan(
        claim_id="p001_c1", persona_id="p001", category=receipt.categories[0],
        verdict=Verdict.COVERED, issued_at=when,
        documents=(DocumentPlan(archetype=receipt, issued_at=when),),
    )
    document = assembler._build_document(
        random.Random(7),
        persona=generate_persona(random.Random(4), persona_id="p001", country=Country.UA),
        plan=plan, document_plan=plan.documents[0],
        vendor={"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy",
                "vat_payer": True},
        doc_id="p001_c1_d1", renderer=renderer, out_dir=tmp_path,
    )

    assert document.reference_text.strip()
    assert document.content_bbox is not None
    assert assembler._CONTENT_BBOX_KEY not in document.field_bboxes, (
        "the reserved key reached the label — it must be removed after the transform"
    )
    assert document.capture is Capture.SCREENSHOT
