"""`DocGroundTruth`'s region/page fields: two independent relations.

ONE DOCUMENT ACROSS SEVERAL PAGES (`page_count` / `page_regions`) and SEVERAL DOCUMENTS
IN ONE FILE (`file_region`) are orthogonal — a file can hold more than this document, this
document can span more than one page inside it, and the two can combine. Nothing here
draws or builds a multi-document file yet; this only pins the label shape and the
invariant between `page_count` and `page_regions`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from receipt_synth.schemas import Capture, DocGroundTruth, DocType


def a_document(**overrides):
    """A minimal, otherwise-valid `DocGroundTruth` — see `test_reference_text.a_document`,
    which this mirrors so the two files do not each invent their own minimal record."""
    fields = {
        "doc_id": "d1", "source_file": "d1.png", "doc_type": DocType.FISCAL_RECEIPT,
        "language": "uk", "currency": "UAH", "amount": Decimal("10.00"),
        "date": date(2026, 6, 15), "counterparty": "Аптека", "line_items": [],
        "has_qr": True, "qr_is_fiscal": True, "has_fiscal_number": True,
        "capture": Capture.PHOTO,
    }
    return DocGroundTruth(**(fields | overrides))


# ------------------------------------------------------------------- defaults --


def test_the_default_is_one_page_and_no_file_region():
    """Every existing call site constructs a `DocGroundTruth` without these fields, and must
    keep meaning "the document IS the whole file, on one page" without being touched."""
    document = a_document()

    assert document.file_region is None
    assert document.page_count == 1
    assert document.page_regions is None


# ------------------------------------------------------------- multi-page, one file --


def test_a_multi_page_document_carries_one_region_per_page():
    document = a_document(
        page_count=2,
        page_regions=[(0.0, 0.0, 100.0, 200.0), (0.0, 200.0, 100.0, 200.0)],
    )

    assert document.page_count == 2
    assert len(document.page_regions) == 2
    assert document.file_region is None


def test_page_count_above_one_without_page_regions_is_rejected():
    with pytest.raises(ValidationError):
        a_document(page_count=2)


def test_page_regions_shorter_than_page_count_is_rejected():
    with pytest.raises(ValidationError):
        a_document(page_count=3, page_regions=[(0.0, 0.0, 10.0, 10.0)])


def test_page_regions_longer_than_page_count_is_rejected():
    with pytest.raises(ValidationError):
        a_document(
            page_count=1,
            page_regions=[(0.0, 0.0, 10.0, 10.0), (0.0, 10.0, 10.0, 10.0)],
        )


def test_page_regions_present_with_the_default_page_count_is_rejected():
    """`page_count` defaults to 1, and a `page_regions` list must not sneak in beside it —
    the mismatch is on the PAIR, not only on an explicitly stated `page_count`."""
    with pytest.raises(ValidationError):
        a_document(page_regions=[(0.0, 0.0, 10.0, 10.0)])


# ---------------------------------------------------------- one document, several in a file --


def test_a_document_sharing_its_file_carries_its_own_rectangle():
    document = a_document(file_region=(10.0, 10.0, 300.0, 150.0))

    assert document.file_region == (10.0, 10.0, 300.0, 150.0)
    assert document.page_count == 1
    assert document.page_regions is None


# ----------------------------------------------------------------------- both at once --


def test_a_multi_page_document_inside_a_multi_document_file_validates():
    """The combination the brief calls out explicitly: both relations set at once, and
    valid, even though today's generator produces neither by itself yet."""
    document = a_document(
        file_region=(0.0, 0.0, 400.0, 600.0),
        page_count=2,
        page_regions=[(0.0, 0.0, 400.0, 300.0), (0.0, 300.0, 400.0, 300.0)],
    )

    assert document.file_region == (0.0, 0.0, 400.0, 600.0)
    assert document.page_count == 2
    assert len(document.page_regions) == 2
