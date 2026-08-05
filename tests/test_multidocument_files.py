"""One file holding BOTH documents of one claim — the composition, its geometry and its labels.

🔴 WHY THIS FILE EXISTS. `tests/test_statement_pagination.py` breaks «one page is one document» in
one direction: one document across two sheets. This is the other direction — two documents in one
file. A step that splits a submitted file into logical documents scores a perfect result on a
corpus where every file holds exactly one, because every cut it needs to make is a cut it never
has to refuse, and running a document onto MORE pages never asks it to cut either. Both
directions, or a segmentation figure is not a figure.

⛔ ONLY THE DOCUMENTS OF ONE CLAIM ARE EVER BUNDLED. Documents of DIFFERENT claims in one file is
a deliberately unsupported pattern — see `assembler.documents_share_one_file`. Nothing here
exercises it, and nothing should be added that does.

WHAT IS ASSERTED, and each is a different way the composition could be right in the picture and
wrong in the label:

  * the DECISION — off at a share of nought, on for every eligible claim at one, never for a
    claim of a single document whatever the share, and a function of the seed alone;
  * the EMBEDDING is 1:1 — a region's box is the size of the image inside it, so a scale drift
    cannot silently multiply every coordinate downstream of it;
  * the OFFSET — a document's boxes move into the file's coordinates by its region's origin;
  * the CARRY — the regions and the offset field boxes go through ONE degrader call and ONE
    transform, which is the failure that is invisible in every other measurement;
  * the LABELS — one file name shared by both documents, no per-document image left behind, and
    NOTHING but carriage different from the same claim built unbundled at the same seed.

⚠️ NO MODULE-SCOPED `Renderer` HERE, deliberately: several tests below drive `generate_dataset`,
which opens a browser of its own, and Playwright's sync API refuses two live instances in one
process (see the same warning in tests/test_render_regions.py). The fixture is function-scoped so
that no renderer of this module is ever alive while a run is going on.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("albumentations")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402

from receipt_synth import assembler  # noqa: E402
from receipt_synth.assembler import (  # noqa: E402
    BUNDLE_TEMPLATE,
    bundled_boxes,
    documents_share_one_file,
    generate_dataset,
    take_bundled_boxes,
)
from receipt_synth.config import file_composition_share  # noqa: E402
from receipt_synth.content_builder import (  # noqa: E402
    build_bank_statement,
    build_invoice,
    draw_party_identity,
    generate_rnokpp,
    resolve_vendor,
)
from receipt_synth.degrader import carry_boxes  # noqa: E402
from receipt_synth.renderer import RenderedDocument, Renderer  # noqa: E402

# The run every end-to-end assertion below is taken on. Two personas of three claims is the
# smallest shape that plans several two-document claims — the eligible ones — while staying
# inside one browser session and a handful of renders.
SEED = 20260803
PERSONAS = 2
CLAIMS = 3


def contains(outer, inner) -> bool:
    """Whether `inner` lies wholly inside `outer`, both in the COCO convention."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ix + iw <= ox + ow and iy + ih <= oy + oh


@pytest.fixture
def renderer():
    """Function-scoped — see the module docstring. A browser per test is the price of driving
    `generate_dataset` from the same file."""
    with Renderer() as instance:
        yield instance


# --------------------------------------------------------------- the decision --


def test_the_share_of_nought_bundles_nothing(monkeypatch):
    """The knob has to be able to turn the case OFF completely: a run reproducing an earlier
    corpus needs a way back to the composition that corpus had."""
    monkeypatch.setattr(assembler, "file_composition_share", lambda name: 0.0)

    assert not any(
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=2)
        for n in range(60)
    )


def test_the_share_of_one_bundles_every_claim_that_has_two_documents(monkeypatch):
    """And the other end, which is what makes the parameter load-bearing rather than a constant
    that happens to be small."""
    monkeypatch.setattr(assembler, "file_composition_share", lambda name: 1.0)

    assert all(
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=2)
        for n in range(60)
    )


def test_a_claim_of_one_document_never_bundles_however_high_the_share(monkeypatch):
    """A file holding one document is not a bundle of one. ⛔ And the other way of giving such a
    claim a companion — a file holding a document of ANOTHER claim — is the pattern this
    generator deliberately does not produce."""
    monkeypatch.setattr(assembler, "file_composition_share", lambda name: 1.0)

    assert not any(
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=1)
        for n in range(60)
    )


def test_the_declared_share_leaves_the_single_document_file_the_strong_majority():
    """The composition decision itself, asserted rather than left to the file. Every
    classification and extraction figure this corpus has produced was measured on files holding
    ONE document, and a share that made the bundle common would change what those numbers describe
    without anybody editing them."""
    bundled = [
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=2)
        for n in range(400)
    ]

    assert file_composition_share("bundle") <= 0.3
    assert bundled.count(False) / len(bundled) > 0.6


def test_the_decision_is_a_function_of_the_seed_and_the_claim_alone():
    """Determinism, and the property that keeps it cheap: the decision is derived from the run
    seed and the claim id — the device `assign_splits` uses — so it takes nothing out of the
    generator's own draw stream, and the documents of a seed are the same documents whether or not
    they end up in one file."""
    first = [
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=2)
        for n in range(40)
    ]
    again = [
        documents_share_one_file(seed=SEED, claim_id=f"p001_c{n}", document_count=2)
        for n in range(40)
    ]
    other_seed = [
        documents_share_one_file(seed=SEED + 1, claim_id=f"p001_c{n}", document_count=2)
        for n in range(40)
    ]

    assert first == again
    assert first != other_seed, "the run seed does not reach the decision"


# -------------------------------------------------------------- the template --


def test_the_bundle_template_marks_one_region_per_document_and_labels_nothing():
    """One `data-document` root for the FILE and one `data-region` per sheet, named in claim order
    — and no `data-field` anywhere: the fields of a bundled document are inside the embedded
    image, and a marker here would add a second box under a name that document already uses.

    Asserted on the RENDERED markup rather than on the file, for the reason
    `test_renderer.template_source` gives for stripping Jinja comments: a rule quoted in a comment
    is not markup, and the region names are a loop variable until something renders them.
    ⚠️ `build_html` needs no browser — it is Jinja and a stylesheet — so this costs no session.
    """
    markup = Renderer().build_html(
        BUNDLE_TEMPLATE,
        {
            "sheets": [
                {"region": f"doc_{n}", "url": "file:///nowhere.png", "width": 10, "height": 10}
                for n in (1, 2)
            ],
            "qr_payload": None,
        },
    )

    assert markup.count("data-document") == 1
    assert 'data-region="doc_1"' in markup and 'data-region="doc_2"' in markup
    assert "data-field" not in markup


# ------------------------------------------------ the embedding, at natural size --


def solid_sheet(path: Path, *, width: int, height: int, colour) -> RenderedDocument:
    """A rendered document that is one flat colour — enough to be embedded and recognized."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    cv2.imwrite(str(path), image)
    return RenderedDocument(
        image_path=path,
        width=width,
        height=height,
        field_bboxes={"amount": (10.0, 20.0, 30.0, 10.0)},
        region_bboxes={},
        reference_text="x",
        content_bbox=(5.0, 5.0, float(width - 10), float(height - 10)),
    )


def test_each_document_is_embedded_at_natural_size_and_is_really_on_the_page(
    renderer, tmp_path
):
    """🔴 THE 1:1 ASSERTION, MEASURED RATHER THAN TRUSTED. Every coordinate a bundled label
    carries is a per-document coordinate plus its region's origin, which holds only while the
    embedded image is drawn at its own size. A scale drift would leave every box in the file
    plausible and wrong.

    The COLOURS are the second half of it: an `<img>` that failed to load keeps whatever box its
    width and height attributes give it, so the size of a region cannot by itself tell a missing
    document from a present one. The pixel in the middle of it can.
    """
    colours = ((0, 0, 255), (255, 0, 0))
    sheets = [
        solid_sheet(tmp_path / "doc_1.png", width=300, height=200, colour=colours[0]),
        solid_sheet(tmp_path / "doc_2.png", width=240, height=120, colour=colours[1]),
    ]

    composed = assembler._compose_bundle(renderer, sheets, output_path=tmp_path / "bundle.png")

    image = cv2.imread(str(composed.image_path))
    for index, (sheet, colour) in enumerate(zip(sheets, colours, strict=True), start=1):
        x, y, width, height = (int(value) for value in composed.region_bboxes[f"doc_{index}"])
        assert (width, height) == (sheet.width, sheet.height), "the sheet is not at natural size"
        assert tuple(image[y + height // 2, x + width // 2]) == colour, "the sheet is not there"


def test_the_regions_are_in_claim_order_and_do_not_overlap(renderer, tmp_path):
    """`doc_1` above `doc_2`, both inside the file. The order is the claim's — the subject
    document first, the payment document second — so a file read top to bottom is the claim read
    in the order it was assembled."""
    sheets = [
        solid_sheet(tmp_path / "doc_1.png", width=300, height=200, colour=(0, 0, 255)),
        solid_sheet(tmp_path / "doc_2.png", width=300, height=160, colour=(255, 0, 0)),
    ]

    composed = assembler._compose_bundle(renderer, sheets, output_path=tmp_path / "bundle.png")

    first, second = composed.region_bboxes["doc_1"], composed.region_bboxes["doc_2"]
    for region in (first, second):
        assert contains((0, 0, composed.width, composed.height), region)
    assert first[1] + first[3] <= second[1], "the second document begins before the first ends"


def test_a_sheet_that_did_not_come_out_at_natural_size_is_refused(
    renderer, tmp_path, monkeypatch
):
    """⛔ NO SILENT RESCALE. The refusal is what makes 1:1 a guarantee rather than a hope: a
    stylesheet that one day gave `.doc` a width of its own would otherwise multiply every
    coordinate of every bundled document by a factor nothing in the label records."""
    sheets = [solid_sheet(tmp_path / "doc_1.png", width=300, height=200, colour=(0, 0, 255))]
    real = Renderer.render

    def shrunk(self, template_name, context, output_path):
        result = real(self, template_name, context, output_path)
        x, y, width, height = result.region_bboxes["doc_1"]
        return replace(result, region_bboxes={"doc_1": (x, y, width / 2, height)})

    monkeypatch.setattr(Renderer, "render", shrunk)

    with pytest.raises(ValueError, match="natural size"):
        assembler._compose_bundle(renderer, sheets, output_path=tmp_path / "bundle.png")


# --------------------------------------------------- the offset and the carry --

# The known-answer fixture of tests/test_bbox_gate.py, reused for the reason
# tests/test_statement_pagination.py reuses it: the claim here is that a `file_region`, a page
# region and an offset field box go through the SAME arithmetic, so all three have to be measured
# with the same rectangle against the same hand-computed answer.
WIDTH, HEIGHT = 200, 100
MARKER = (20.0, 30.0, 40.0, 10.0)
WHOLE_IMAGE = (0.0, 0.0, float(WIDTH), float(HEIGHT))


def fixture_image() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x, y, w, h = (int(value) for value in MARKER)
    image[y : y + h, x : x + w] = 255
    return image


def test_a_documents_boxes_are_moved_by_its_regions_origin():
    """The offset, on its own and before any degradation: a box at (20, 30) inside a document
    whose sheet starts at (7, 500) in the file is at (27, 530) in the file. Sizes are untouched —
    the document is embedded, not scaled."""
    merged = bundled_boxes([((7.0, 500.0, 300.0, 200.0), {"amount": MARKER})])

    assert merged["__doc_1__amount"] == (27.0, 530.0, 40.0, 10.0)
    assert merged["__file_region_1__"] == (7.0, 500.0, 300.0, 200.0)


def test_a_file_region_moves_exactly_as_the_boxes_inside_it_do():
    """🔴 THE FAILURE THIS TEST EXISTS FOR IS INVISIBLE EVERYWHERE ELSE. Every capture channel
    that has geometry DRAWS it, so a region moved by a second call to the degrader is moved by a
    second draw and lands somewhere plausible and wrong — which reads downstream as a poor
    segmenter and never as a coordinate defect.

    HAND-COMPUTED, from tests/test_bbox_gate.py's own geometry: a horizontal flip of a 200-wide
    image maps column c to 199 - c, so the marker's columns 20 … 59 become 140 … 179 —

        (20, 30, 40, 10)  ->  (140, 30, 40, 10)

    The document sits at the origin of the file here, so its field box, its content extent and its
    page region start at the marker and must all end at the flipped marker; the file region is the
    whole image and must come back as the whole image.
    """
    merged = bundled_boxes(
        [
            (
                WHOLE_IMAGE,
                {
                    "amount": MARKER,
                    assembler._CONTENT_BBOX_KEY: MARKER,
                    "__page_region_1__": MARKER,
                    "__page_region_2__": MARKER,
                },
            )
        ]
    )

    _, moved = carry_boxes([A.HorizontalFlip(p=1.0)], fixture_image(), merged, seed=0)
    carried = take_bundled_boxes(moved, index=1, page_count=2)

    assert carried.field_bboxes == {"amount": (140.0, 30.0, 40.0, 10.0)}
    assert carried.content_bbox == (140.0, 30.0, 40.0, 10.0)
    assert carried.page_regions == [(140.0, 30.0, 40.0, 10.0)] * 2
    assert carried.file_region == WHOLE_IMAGE


def test_two_documents_naming_the_same_field_keep_their_own_boxes():
    """🔴 TWO DOCUMENTS OF ONE CLAIM PRINT THE SAME FIELD NAMES — both carry an `amount` — and
    both have to survive one merged call. A flat merge would leave the file with a single
    `amount` box, silently, and the label of whichever document lost would point at the other's
    ink."""
    other = (100.0, 10.0, 20.0, 20.0)
    merged = bundled_boxes(
        [
            (WHOLE_IMAGE, {"amount": MARKER, assembler._CONTENT_BBOX_KEY: MARKER}),
            (WHOLE_IMAGE, {"amount": other, assembler._CONTENT_BBOX_KEY: other}),
        ]
    )

    _, moved = carry_boxes([A.NoOp(p=1.0)], fixture_image(), merged, seed=0)
    first = take_bundled_boxes(moved, index=1, page_count=1)
    second = take_bundled_boxes(moved, index=2, page_count=1)

    assert first.field_bboxes == {"amount": MARKER}
    assert second.field_bboxes == {"amount": other}
    assert not moved, "the split left a reserved key behind"


def test_no_reserved_key_of_the_bundle_reaches_a_label():
    """⚠️ THE MERGE ADDS TWO MORE RESERVED NAMES to the two `tracked_boxes` already carries, and
    a key this split forgot would reach a consumer as a `data-field` called `__doc_1__amount` — a
    field no template prints and no requirement names."""
    merged = bundled_boxes(
        [
            (
                WHOLE_IMAGE,
                {
                    "amount": MARKER,
                    assembler._CONTENT_BBOX_KEY: MARKER,
                    "__page_region_1__": MARKER,
                    "__page_region_2__": MARKER,
                },
            )
        ]
    )

    _, moved = carry_boxes([A.NoOp(p=1.0)], fixture_image(), merged, seed=0)
    carried = take_bundled_boxes(moved, index=1, page_count=2)

    assert not [name for name in carried.field_bboxes if name.startswith("__")]
    assert not moved


def test_the_split_refuses_a_document_the_merge_never_carried():
    """⛔ NO SILENT EMPTY DOCUMENT. A file whose second region never reached the degrader would
    otherwise be labelled as a document with no boxes at all — a record that looks like a page
    nothing was extracted from rather than like the defect it is."""
    merged = bundled_boxes([(WHOLE_IMAGE, {"amount": MARKER, assembler._CONTENT_BBOX_KEY: MARKER})])

    with pytest.raises(ValueError, match="__file_region_2__"):
        take_bundled_boxes(merged, index=2, page_count=1)


# ---------------------------------------------------------------- the whole run --


@pytest.fixture(scope="module")
def bundled_run(tmp_path_factory):
    """A run whose every eligible claim is bundled, so what follows measures the path rather than
    a seed's luck. The share is the only thing turned up — which is what
    `test_bundling_changes_nothing_but_carriage` then rests on."""
    out = tmp_path_factory.mktemp("bundled")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "file_composition_share", lambda name: 1.0)
        dataset = generate_dataset(
            seed=SEED,
            out_dir=out,
            train_fraction=0.5,
            personas=PERSONAS,
            claims_per_persona=CLAIMS,
        )
    return dataset, out


@pytest.fixture(scope="module")
def single_run(tmp_path_factory):
    """The same seed with the share at nought — one file per document, which is the corpus as it
    was before this path existed."""
    out = tmp_path_factory.mktemp("single")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "file_composition_share", lambda name: 0.0)
        dataset = generate_dataset(
            seed=SEED,
            out_dir=out,
            train_fraction=0.5,
            personas=PERSONAS,
            claims_per_persona=CLAIMS,
        )
    return dataset, out


def documents_of(dataset, claim):
    index = {document.doc_id: document for document in dataset.documents}
    return [index[doc_id] for doc_id in claim.documents]


def bundled_claims(dataset):
    return [claim for claim in dataset.claims if len(claim.documents) > 1]


def test_both_documents_of_a_bundled_claim_name_one_file_that_exists(bundled_run):
    """The whole of what a multi-document file is, in the label: two documents, one
    `source_file`, and that file on disk."""
    dataset, out = bundled_run
    claims = bundled_claims(dataset)
    assert claims, "this run planned no claim of two documents — every assertion here is vacuous"

    for claim in claims:
        documents = documents_of(dataset, claim)
        assert {document.source_file for document in documents} == {f"{claim.claim_id}.png"}
        assert (out / "images" / f"{claim.claim_id}.png").is_file()


def test_no_per_document_image_of_a_bundled_claim_survives(bundled_run):
    """The clean render of each document is an INTERMEDIATE. A per-document PNG left in `images/`
    would be a second file carrying the same document, with the label pointing at neither."""
    dataset, out = bundled_run

    for claim in bundled_claims(dataset):
        for doc_id in claim.documents:
            assert not (out / "images" / f"{doc_id}.png").exists(), doc_id
    assert {path.name for path in (out / "images").iterdir()} == {
        document.source_file for document in dataset.documents
    }


def test_every_field_box_lies_inside_its_own_documents_region(bundled_run):
    """🔴 THE OFFSET, MEASURED ON THE SHIPPED LABEL. Each document's boxes were rendered in its
    own coordinates and moved into the file's; one that stayed behind would point into the other
    document, and every extraction figure taken from it would be wrong in a way that reads as a
    bad extractor."""
    dataset, _ = bundled_run

    for claim in bundled_claims(dataset):
        for document in documents_of(dataset, claim):
            assert document.file_region is not None, document.doc_id
            for name, box in document.field_bboxes.items():
                assert contains(document.file_region, box), f"{document.doc_id}.{name}"


def test_the_regions_of_one_file_stay_in_claim_order_and_barely_touch(bundled_run):
    """The first document of the claim above the second, on the SHIPPED label — and each region
    still a rectangle with area, somewhere on the file.

    ⚠️ NOT «DISJOINT», AND THE CAVEAT IS THE DEGRADER'S RATHER THAN THIS PATH'S. A region is
    carried through the capture as four corners and rebuilt as their axis-aligned HULL
    (`degrader.carry_boxes`), so a rotation of a degree or two makes every box a little larger
    than the ink it covers and two stacked sheets can overlap by a few pixels. The composed file
    itself has no overlap at all — asserted above, before any capture — so what is bounded here is
    the hull's own inflation, at a few percent of a sheet.
    """
    dataset, out = bundled_run

    for claim in bundled_claims(dataset):
        documents = documents_of(dataset, claim)
        image = cv2.imread(str(out / "images" / documents[0].source_file))
        height, width = image.shape[:2]
        above, below = (document.file_region for document in documents)

        for region in (above, below):
            assert region[2] > 0 and region[3] > 0, claim.claim_id
            assert region[0] < width and region[1] < height, "a region is off the file entirely"
        assert above[1] < below[1], f"{claim.claim_id}: the claim's order did not survive"
        overlap = max(0.0, above[1] + above[3] - below[1])
        assert overlap < 0.1 * min(above[3], below[3]), f"{claim.claim_id}: overlap {overlap}"


def test_a_bundled_claim_label_is_shaped_exactly_as_an_unbundled_one(bundled_run, single_run):
    """The claim record is about EVIDENCE, not about files: the same claim carries the same
    documents in the same order however they were carried."""
    bundled, _ = bundled_run
    single, _ = single_run

    for one, other in zip(bundled.claims, single.claims, strict=True):
        assert one.model_dump() == other.model_dump()


# Everything a bundled label may legitimately differ in, and nothing else. The first five are
# geometry; `source_file` is the file itself; `capture` is the file's single channel — a file is
# captured once, so the second document is carried on the channel the file was captured on.
_CARRIAGE = {
    "file_region",
    "page_regions",
    "field_bboxes",
    "content_bbox",
    "content_lost_edges",
    "content_complete",
    "source_file",
    "capture",
}


def test_bundling_changes_nothing_but_carriage(bundled_run, single_run):
    """🔴 THE LABEL INTEGRITY CLAIM, AND THE REASON THE DECISION IS SEEDED INDEPENDENTLY.
    Bundling is a decision about CARRIAGE: at one seed the same documents are produced — the same
    amounts, dates, parties, verdicts and causes — whether they are carried in one file or two.
    The comparison is possible at all only because the decision takes nothing out of the
    generator's draw stream."""
    bundled, _ = bundled_run
    single, _ = single_run

    assert [document.doc_id for document in bundled.documents] == [
        document.doc_id for document in single.documents
    ]
    for one, other in zip(bundled.documents, single.documents, strict=True):
        assert {
            key: value for key, value in one.model_dump().items() if key not in _CARRIAGE
        } == {key: value for key, value in other.model_dump().items() if key not in _CARRIAGE}


def test_an_unbundled_run_is_the_corpus_as_it_was(single_run):
    """The share of nought has to reproduce the earlier composition exactly: one file per
    document, named after the document, and no `file_region` anywhere."""
    dataset, out = single_run

    for document in dataset.documents:
        assert document.source_file == f"{document.doc_id}.png"
        assert document.file_region is None
        assert (out / "images" / document.source_file).is_file()


def test_the_written_label_carries_the_region_a_consumer_reads(bundled_run):
    """Through the file a consumer actually opens rather than through the in-memory record: a
    `file_region` that failed to serialize would be segmentation ground truth nobody receives."""
    dataset, out = bundled_run
    claim = bundled_claims(dataset)[0]
    document = documents_of(dataset, claim)[0]

    written = json.loads((out / "labels" / f"{document.doc_id}.json").read_text(encoding="utf-8"))

    assert written["source_file"] == f"{claim.claim_id}.png"
    assert written["file_region"] == list(document.file_region)


def test_a_bundled_run_is_reproducible_to_the_byte(tmp_path):
    """Determinism under `--seed` over the path that composes a third image out of two others:
    same seed, same decisions, same pixels, same labels."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "file_composition_share", lambda name: 1.0)
        first = generate_dataset(
            seed=SEED, out_dir=tmp_path / "a", train_fraction=0.5, personas=1, claims_per_persona=1
        )
        second = generate_dataset(
            seed=SEED, out_dir=tmp_path / "b", train_fraction=0.5, personas=1, claims_per_persona=1
        )

    assert first.as_manifest() == second.as_manifest()
    for document in first.documents:
        name = document.source_file
        assert (tmp_path / "a" / "images" / name).read_bytes() == (
            tmp_path / "b" / "images" / name
        ).read_bytes()


# ---------------------------------------- the combination with a paginated document --


def test_a_paginated_document_inside_a_bundle_carries_both_relations(renderer, tmp_path):
    """🔴 THE TWO RELATIONS ARE INDEPENDENT, AND THE COMBINATION HAS TO WORK. A statement that
    runs onto a second sheet, bundled with the invoice it settles, carries a `file_region` saying
    where the document is in the file AND `page_regions` saying where each of its sheets is —
    both in file coordinates, both moved by the one transform.

    FORCED RATHER THAN SEED-HUNTED. The pagination draw and the bundling decision are
    independent, so a seed producing both is a coincidence a test would then depend on; the two
    documents are built here exactly as the assembler builds them and handed to the same
    bundling step.
    """
    rng = random.Random(11)
    vendor = resolve_vendor(
        rng,
        {
            "name": "Освітній центр «Проспект»",
            "legal_form": "TOV",
            "profile": "training_centre",
            "vat_payer": True,
        },
        "UA",
    )
    identity = draw_party_identity(rng, vendor, "UA")
    holder, holder_code = "Ковальчук Олена Петрівна", generate_rnokpp(random.Random(3))
    when = datetime(2026, 5, 12, 14, 33)

    invoice = build_invoice(
        rng,
        category_id="professional_development",
        issued_at=when,
        vendor=vendor,
        identity=identity,
        buyer_name=holder,
        buyer_tax_id=holder_code,
    )
    statement = build_bank_statement(
        rng,
        issued_at=when,
        vendor=vendor,
        identity=identity,
        payer_name=holder,
        payer_tax_id=holder_code,
        amount=invoice.total,
        cites=invoice.reference,
        pages=2,
    )
    assert statement.page_count == 2, "the statement under test is not paginated"

    staging = tmp_path / "staging"
    staging.mkdir()
    builts = [
        _built(renderer, "ua_invoice", invoice, doc_id="p001_c1_d1", staging=staging),
        _built(renderer, "ua_bank_statement", statement, doc_id="p001_c1_d2", staging=staging),
    ]

    labels = assembler._bundle_one_file(
        builts, claim_id="p001_c1", renderer=renderer, out_dir=tmp_path / "out", staging=staging
    )

    subject, payment = labels
    assert subject.page_count == 1 and subject.page_regions is None
    assert payment.page_count == 2
    assert payment.page_regions is not None and len(payment.page_regions) == 2
    for region in payment.page_regions:
        assert contains(payment.file_region, region), "a sheet lies outside its own document"
    assert contains(payment.file_region, payment.content_bbox)
    for name, box in payment.field_bboxes.items():
        assert contains(payment.file_region, box), name
    assert not contains(subject.file_region, payment.file_region)


def test_a_run_that_paginates_and_bundles_everything_labels_both_relations(tmp_path):
    """The same combination THROUGH THE PIPELINE, because the hand-built one above cannot see the
    loop that leads to it — and the loop is where it broke: the amount a payment settles is read
    off the subject's record, and asking every bundled document for a record instead of only the
    subject refuses on the first paginated statement. Every test above stayed green; this one is
    what says so.

    Both shares are turned up, so the run is one where every eligible claim bundles and every
    statement runs onto a second sheet. Nothing else about it is unusual.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler, "file_composition_share", lambda name: 1.0)
        patch.setattr("receipt_synth.content_builder.bank_statement_share", lambda name: 1.0)
        dataset = generate_dataset(
            seed=SEED, out_dir=tmp_path, train_fraction=0.5, personas=1, claims_per_persona=3
        )

    paginated = [
        document
        for claim in bundled_claims(dataset)
        for document in documents_of(dataset, claim)
        if document.page_count > 1
    ]
    assert paginated, "no bundled claim of this run carries a paginated document"
    for document in paginated:
        assert document.file_region is not None
        assert len(document.page_regions) == document.page_count
        for region in document.page_regions:
            assert contains(document.file_region, region), document.doc_id


def _built(renderer, slug, document, *, doc_id, staging):
    """One document rendered clean into the staging directory — what `_render_document` hands the
    bundling step, without the draws a run would make around it."""
    return assembler._BuiltDocument(
        doc_id=doc_id,
        document=document,
        reference=getattr(document, "reference", None),
        capture=assembler.CAPTURE_DRAW_ORDER[0],
        degrade_seed=5,
        clean=renderer.render(slug, document.render_context(), staging / f"{doc_id}.png"),
    )
