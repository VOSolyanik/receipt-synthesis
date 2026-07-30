"""The comparison rules as cases, and the one property this generator can check about them.

🔴 WHAT THIS FILE CAN AND CANNOT DO, said first because the distinction decides every test below.

THIS GENERATOR HAS NO NORMALIZER. It emits one side of a comparison and never performs one, so
there is nothing here to run the vectors against — they are for whoever writes the other side.
Testing them by implementing a normalizer here would be testing an implementation written to match
the file, which proves the file agrees with itself.

WHAT IS CHECKABLE IS THE FIXED POINT: every value this dataset emits must ALREADY BE in the
canonical form its class declares, so that `normalize(label) == label` holds by inspection. That is
a real property of the corpus, it is what makes the comparison rules usable at all, and it fails
loudly if a builder ever emits an untrimmed name or an unrounded amount.

⚠️ THE FORM IS NOT THE WHOLE RULE. Casefolding and legal-form stripping are applied to BOTH SIDES at
comparison time; they are not properties a label is in. So the fixed point covers the
canonicalizing clauses — NFC, trimming, collapsing, rounding, membership — and the comparison-only
clauses are named where they are excluded rather than quietly dropped.
"""

from __future__ import annotations

import json
import re
import unicodedata

import pytest
import yaml

from receipt_synth.config import CONFIG_DIR, load_policy

CONTRACT = yaml.safe_load((CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8"))
NORMALIZATION = CONTRACT["normalization"]
VECTORS = CONTRACT["conformance_vectors"]["vectors"]

EMITTED_CLASSES = {
    name for name, spec in NORMALIZATION.items() if spec.get("status") == "emitted"
}


# ------------------------------------------------------------- completeness --


def test_every_emitted_normalization_class_has_vectors():
    """🔴 THE COMPLETENESS PROPERTY, and its denominator is section 2 of the contract — the
    `normalization` rules themselves — rather than a list here, so a class added there without
    cases fails rather than being silently untested.

    `emitted` only: `open_enum` is `divergent` and `identifier` is a forward contract, so neither
    describes a comparison a consumer can run against this dataset today.
    """
    assert EMITTED_CLASSES, "no emitted normalization classes — this test asserts nothing"

    missing = sorted(EMITTED_CLASSES - set(VECTORS))
    assert not missing, (
        f"{len(missing)} of {len(EMITTED_CLASSES)} emitted classes carry no conformance vector: "
        f"{missing}"
    )


def test_no_vector_describes_a_class_that_does_not_exist():
    """The other direction. A vector for a class section 2 does not declare would be a rule nobody
    can look up, and it would go on passing after the class was renamed."""
    unknown = sorted(set(VECTORS) - set(NORMALIZATION))
    assert not unknown, f"{unknown} have vectors and no normalization rule"


@pytest.mark.parametrize("name", sorted(VECTORS))
def test_every_vector_is_complete(name):
    """Each case says what goes in, what the label holds, what the rule decides, and WHY. A vector
    without a reason is a fact nobody can maintain: when a rule changes, the reason is what says
    whether the case moved with it."""
    cases = VECTORS[name]
    assert cases, f"{name} declares an empty vector list"

    for case in cases:
        assert set(case) >= {"input", "label", "equal", "why"}, (name, case)
        assert isinstance(case["equal"], bool) or case["equal"] is None, (name, case)
        # A bound rather than mere non-emptiness, because "yes" is not a reason — but a LOW one:
        # "identical bytes" is a complete explanation of an identity vector, and a threshold that
        # rejected it would push a writer into padding rather than into explaining.
        assert len(case["why"]) >= 10, (name, case)


def test_at_least_one_class_declares_a_case_it_calls_unequal():
    """A vector set of nothing but matches would pass any implementation that returned `true`
    unconditionally. Checked over the whole file with its denominator, because the property is
    about the SET of vectors rather than about any one class."""
    unequal = sum(1 for cases in VECTORS.values() for c in cases if c["equal"] is False)
    total = sum(len(cases) for cases in VECTORS.values())

    assert unequal >= len(VECTORS), (
        f"only {unequal} of {total} vectors across {len(VECTORS)} classes assert INEQUALITY; a set "
        "of matches alone cannot distinguish a working implementation from one that always agrees"
    )


def test_the_exact_classes_agree_with_their_own_rule():
    """🔴 THE ONE PLACE A VECTOR'S ANSWER CAN BE DERIVED RATHER THAN TRUSTED.

    Four classes state a rule simple enough to evaluate from its own words: `identity` compares
    bytes, and `closed_enum`, `policy_vocabulary` and `vat_letter` compare code points exactly after
    NFC. For those, `equal` is not an opinion — it follows — so a vector claiming otherwise is
    wrong on its face.

    THIS IS NOT THE NORMALIZER THIS FILE REFUSES TO WRITE. The classes with judgement in them —
    stripping a legal form, parsing a date, rounding money — are untouched here precisely because
    implementing them would mean testing an implementation written to match the file. What is
    checked is only where the rule leaves no room.

    Found by a mutation: flipping the Latin-A / Cyrillic-А vector to `equal: true` — a claim that
    contradicts `do not transliterate` — survived a sweep that only counted how many vectors
    asserted inequality.
    """
    exact = {"identity", "closed_enum", "policy_vocabulary", "vat_letter"}
    assert exact <= set(VECTORS), sorted(exact - set(VECTORS))

    for name in sorted(exact):
        for case in VECTORS[name]:
            left, right = str(case["input"]), str(case["label"])
            if name != "identity":
                left = unicodedata.normalize("NFC", left)
                right = unicodedata.normalize("NFC", right)
            assert case["equal"] is (left == right), (
                f"{name}: {case['input']!r} vs {case['label']!r} is declared "
                f"equal={case['equal']}, but this class compares exactly and they are "
                f"{'the same' if left == right else 'different'}"
            )


# --------------------------------------------- the un-normalized input, checked --


def test_the_unicode_vector_is_genuinely_not_normalized():
    """🔴 THE ESCAPE SEQUENCE IS CHECKED, NOT TRUSTED.

    A vector for Unicode normalization needs an input that is NOT normalized. Any editor, any copy
    through a terminal, any reformatting of the contract would silently normalize it — leaving a
    case that passes without testing anything while still looking like a test. It is therefore
    written `\\u0456\\u0308` in the file, and this test asserts the value that reaches a reader is
    decomposed and that its label is the composed form of the same word.
    """
    decomposed = [
        case for case in VECTORS["text_name"]
        if not unicodedata.is_normalized("NFC", str(case["input"]))
    ]
    assert len(decomposed) == 1, (
        f"{len(decomposed)} text_name vectors carry a non-NFC input; the rule needs exactly one, "
        "and a second would probably be an accident"
    )

    case = decomposed[0]
    assert unicodedata.normalize("NFC", case["input"]) == case["label"]
    assert case["input"] != case["label"], "the two spellings are identical — nothing is tested"
    assert case["equal"] is True, "NFC is the first clause of every rule, so they are equal"


def test_the_contract_stores_that_input_as_an_escape_and_not_as_a_character():
    """The file on disk, read as TEXT rather than parsed — a positional question about what the
    bytes contain, which the parsed value cannot answer: by the time YAML has decoded the escape,
    both spellings look the same."""
    raw = (CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8")

    assert "\\u0456\\u0308" in raw, (
        "the decomposed input is no longer stored as an escape sequence, so an editor has "
        "normalized it and the vector now tests nothing"
    )


# ------------------------------------------------------- the fixed point, on a run --


def canonical_failures(record: dict) -> list[str]:
    """Every way a label record departs from the canonical form its classes declare.

    One function rather than a test per class so the sweep's denominator is the record itself: a
    field added to the model without a rule shows up as an unchecked key rather than as nothing.
    """
    problems = []

    def text_is_canonical(value: str, where: str) -> None:
        if not unicodedata.is_normalized("NFC", value):
            problems.append(f"{where}: not NFC")
        if value != value.strip():
            problems.append(f"{where}: untrimmed")
        if re.search(r"\s{2,}", value):
            problems.append(f"{where}: uncollapsed whitespace")

    for key in ("counterparty", "payer", "payment_purpose"):
        value = record.get(key)
        if isinstance(value, str):
            text_is_canonical(value, key)
    for index, item in enumerate(record.get("line_items", [])):
        text_is_canonical(item["name"], f"line_items[{index}].name")
        if item["vat_letter"] is not None and not unicodedata.is_normalized(
            "NFC", item["vat_letter"]
        ):
            problems.append(f"line_items[{index}].vat_letter: not NFC")

    # `party_name` additionally makes the BARE name authoritative, so a label carrying a printed
    # legal form is not in canonical form. Checked on the prefixes content_builder can print, and
    # on the TRIMMED value: the first version tested the raw string, so a name that was both
    # untrimmed and prefixed reported only the whitespace and the more serious fault was masked by
    # the lesser one.
    counterparty = record.get("counterparty")
    if isinstance(counterparty, str):
        for prefix in ("ТОВ ", "ФОП ", "ПрАТ "):
            if counterparty.strip().startswith(prefix):
                problems.append(f"counterparty: carries the legal form {prefix.strip()!r}")

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", record["date"]):
        problems.append(f"date: {record['date']!r} is not ISO")

    for key in ("amount", "amount_due", "fee", "total_charged", "reimbursable_amount"):
        value = record.get(key)
        if isinstance(value, int | float) and round(value, 2) != value:
            problems.append(f"{key}: {value!r} is not at two decimal places")

    for name, box in record.get("field_bboxes", {}).items():
        if len(box) != 4 or any(float(v) != int(v) for v in box):
            problems.append(f"field_bboxes[{name}]: {box!r} is not four whole pixels")

    return problems


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    """One real run's labels. The fixed point is a property of what the generator EMITS, so it is
    checked on emitted records rather than on hand-built ones."""
    from receipt_synth.assembler import generate_dataset

    out = tmp_path_factory.mktemp("conformance")
    generate_dataset(seed=20260803, out_dir=out, personas=2, claims_per_persona=4)
    return [json.loads(p.read_text()) for p in sorted((out / "labels").glob("*.json"))
            if not p.name.endswith(".claim.json")]


def test_every_emitted_value_is_already_in_its_canonical_form(labels):
    """🔴 `normalize(label) == label`, IN THE ONE FORM THIS REPOSITORY CAN CHECK IT.

    Every canonicalizing clause of every rule — NFC, trimming, collapsing, ISO dates, two decimal
    places, whole-pixel boxes, the bare trading name — must already hold of what is emitted. If it
    does not, a consumer applying the rule changes the label, and the two sides of the comparison
    are no longer the same thing.

    ⚠️ CASEFOLDING IS NOT CHECKED and is not a defect: it is applied to BOTH sides at comparison
    time, so the label is deliberately not in a casefolded form. Naming the exclusion is the point —
    a fixed-point test that quietly included it would demand lower-case counterparties.
    """
    assert labels, "the run produced no document labels"

    failures = {
        record["doc_id"]: problems
        for record in labels
        if (problems := canonical_failures(record))
    }
    assert not failures, (
        f"{len(failures)} of {len(labels)} emitted records are not in canonical form: "
        f"{dict(list(failures.items())[:3])}"
    )


def test_the_canonical_check_can_fail(labels):
    """The tripwire's own tripwire. A sweep that cannot detect a violation passes forever, and this
    file exists because of exactly that class of defect elsewhere in the repository."""
    broken = dict(labels[0])
    broken["counterparty"] = f"  ТОВ «{broken['counterparty']}»  "
    broken["date"] = "15.06.2026"

    problems = canonical_failures(broken)
    assert any("untrimmed" in p for p in problems)
    assert any("legal form" in p for p in problems)
    assert any("not ISO" in p for p in problems)


def test_every_enum_value_emitted_is_in_its_declared_vocabulary(labels):
    """The membership half of the fixed point, which the string checks above cannot cover: a value
    that is trimmed and NFC and still outside its vocabulary is an error rather than a near miss,
    and that is what `closed_enum` and `policy_vocabulary` say."""
    doc_types = {entry["name"] for entry in CONTRACT["document_label"]["fields"]}
    assert "doc_type" in doc_types, "the field list moved — this sweep is reading the wrong block"

    kinds = {
        kind
        for category in load_policy()["categories"]
        for bucket in ("covered_items", "excluded_items", "ambiguous_items")
        for kind in (category.get(bucket) or {})
    }
    assert kinds, "no item kinds in the policy — the sweep below would assert nothing"

    for record in labels:
        assert record["doc_type"] in CONTRACT["document_types"], record["doc_id"]
        assert record["capture"] in {"screenshot", "photo", "scan"}, record["doc_id"]
        for item in record["line_items"]:
            assert item["item_kind"] in kinds, (record["doc_id"], item["item_kind"])
