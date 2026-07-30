"""Every index `config/labelling-schema.yaml` offers a consumer resolves.

🔴 WHY THIS FILE EXISTS, because it decides what belongs in it. The contract stored several of
its indexes as PROSE — a rule's `applies_to` list, a status slot holding a sentence, a
`generator:` cell mixing field names with commentary — and prose is not checked. A
hand-maintained reverse index over the `normalize:` keys diverged from them EIGHT TIMES while
under active attention, including inside the very version that added the fields it missed. The
repair in contract version 21 was structural: delete the duplicate index, make the remaining ones
data, and hold them here.

⚠️ SO THESE ARE NOT TESTS OF THE GENERATOR. `src/` reads nothing in that file; every assertion
below is about the contract's internal consistency, and the denominator of each is derived from
the file itself rather than listed here — a field, a rule or a table row added there is covered
without anyone remembering to extend a list.

WHAT THEY CANNOT SEE, said plainly: whether a rule is the RIGHT rule for a field. That is a
judgement no index check reaches, and `conformance_vectors` is where it is pinned instead.
"""

from __future__ import annotations

import pytest
import yaml

from receipt_synth.config import CONFIG_DIR

CONTRACT = yaml.safe_load((CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8"))

NORMALIZATION = CONTRACT["normalization"]
STATUS_VOCABULARY = CONTRACT["status_vocabulary"]

# The four blocks that carry a `fields:` list of label field records. Named here because the
# contract has no key that enumerates them; everything else below is derived.
FIELD_BLOCKS = ("document_label", "line_item", "claim_label", "persona_record")


def _fields() -> list[tuple[str, dict]]:
    return [(block, field) for block in FIELD_BLOCKS for field in CONTRACT[block]["fields"]]


def _statuses(node, path: str = "") -> list[tuple[str, str]]:
    """Every `status:` value anywhere in the contract, with the path that carries it.

    A structural walk rather than a lookup of the places a status is expected: the whole point is
    to find the slot nobody remembered.
    """
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "status" and isinstance(value, str):
                found.append((path, value))
            found.extend(_statuses(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_statuses(value, f"{path}[{index}]"))
    return found


# ----------------------------------------------------------------- the forward index --


def test_every_field_normalize_names_a_real_rule():
    """🔴 THE ONE THAT BLOCKS A CONSUMER. `normalization[field["normalize"]]` is how a scorecard
    looks a field's comparison rule up, so a `normalize:` value that is not a key of
    `normalization` raises `KeyError` on a contract that is otherwise valid — and it raises at the
    consumer, on a file it vendored, with nothing here having gone red.

    Two of the 46 fields carrying the key held a sentence rather than a rule name until contract
    version 21: `document_label.line_items` said "see `line_item`" and `claim_label.documents`
    said "identity per element".
    """
    carriers = [(block, field) for block, field in _fields() if "normalize" in field]
    assert carriers, "no field carries a `normalize:` key — this test asserts nothing"

    unknown = [
        f"{block}.{field['name']} -> {field['normalize']!r}"
        for block, field in carriers
        if field["normalize"] not in NORMALIZATION
    ]
    assert not unknown, (
        f"{len(unknown)} of {len(carriers)} fields carrying `normalize:` name no rule in "
        f"`normalization`: {unknown}"
    )


def test_every_normalization_rule_is_named_by_at_least_one_field():
    """The other direction, and the only thing the deleted `applies_to` index could ever have
    revealed: an ORPHAN RULE. A rule no field names is either a field that lost its `normalize:`
    key or a rule kept past the removal of what it governed, and both read as maintained.

    Written against the rules rather than against a list here, so a rule added there is covered.
    """
    named = {field["normalize"] for _, field in _fields() if "normalize" in field}
    orphans = sorted(set(NORMALIZATION) - named)
    assert not orphans, (
        f"{len(orphans)} of {len(NORMALIZATION)} normalization rules are named by no field: "
        f"{orphans}"
    )


def test_a_rule_that_delegates_to_another_names_a_real_one():
    """`element_rule` is the one thing the deleted reverse index held that no field holds: the
    members of an unordered string list are compared as `closed_enum` says, and a consumer reading
    the forward index alone would learn only that the array is a set. It is a rule name for the
    same reason `normalize:` is one — so it is looked up rather than parsed.
    """
    delegating = {
        name: spec["element_rule"] for name, spec in NORMALIZATION.items() if "element_rule" in spec
    }
    assert delegating, "no rule delegates — this test asserts nothing"

    unknown = {name: rule for name, rule in delegating.items() if rule not in NORMALIZATION}
    assert not unknown, f"{unknown} delegate to a rule `normalization` does not declare"


# ------------------------------------------------------------- the status vocabulary --


def test_every_status_is_a_word_of_the_vocabulary():
    """🔴 `status:` IS A RESERVED WORD WITH EXACTLY ONE MEANING. A consumer branches on it, so a
    value outside `status_vocabulary` compares against nothing — and a whole SENTENCE in the slot
    ("done in version 10, and NARROWER than it was asked for") is not a further vocabulary word
    but a value no branch can reach.

    Three values were undeclared before contract version 21: `resolved`, in use since the file's
    first commit, and two sentences in `required_changes`. The first joined the vocabulary; the
    two sentences moved to `progress:`, which is prose and is read rather than indexed.
    """
    statuses = _statuses(CONTRACT)
    assert statuses, "no `status:` key found anywhere — this test asserts nothing"

    outside = sorted({(path, value) for path, value in statuses if value not in STATUS_VOCABULARY})
    assert not outside, (
        f"{len(outside)} of {len(statuses)} `status:` values are not one of the "
        f"{len(STATUS_VOCABULARY)} words of `status_vocabulary` "
        f"({sorted(STATUS_VOCABULARY)}): {outside}"
    )


def test_every_vocabulary_word_is_used():
    """A declared word nothing carries is a distinction the file no longer draws, and it invites a
    consumer to write a branch that can never be taken."""
    in_use = {value for _, value in _statuses(CONTRACT)}
    unused = sorted(set(STATUS_VOCABULARY) - in_use)
    assert not unused, f"{unused} are declared in `status_vocabulary` and carried by nothing"


# ------------------------------------------------- the prd_required_fields generator column --


def _prd_rows() -> list[tuple[str, dict]]:
    return [
        (doc_type, row)
        for doc_type, block in CONTRACT["document_types"].items()
        for row in block.get("prd_required_fields", [])
    ]


@pytest.mark.parametrize("doc_type", sorted(CONTRACT["document_types"]))
def test_the_generator_column_is_a_list_of_names(doc_type):
    """🔴 PUNCTUATION WAS STRUCTURAL WHILE THE CELL WAS PROSE. `has_qr, qr_is_fiscal (two
    booleans)` rearranged into `has_qr (boolean), qr_is_fiscal` reads as a stylistic edit and
    drops a field from the schema — a reader sees tidying, a parser sees one fewer field. As a
    list the same rearrangement is a no-op or a parse error, never a silent loss.

    Parametrized over the document types the contract declares, not over the four that have such a
    table today, so a fifth is covered on arrival.
    """
    rows = CONTRACT["document_types"][doc_type].get("prd_required_fields", [])
    for row in rows:
        assert "generator" in row, (doc_type, row["prd"])
        cell = row["generator"]
        assert isinstance(cell, list), (
            f"{doc_type}: `generator:` for {row['prd'][:40]!r} is {type(cell).__name__}, not a "
            f"list — prose in this column makes punctuation structural"
        )
        assert all(isinstance(name, str) for name in cell), (doc_type, row["prd"], cell)


def test_every_generator_name_is_a_real_label_field():
    """The teeth on the list: a name that is not a `document_label` field is a typo or a rename
    that left this table behind, and the list shape alone would not catch either.

    `document_label` is the right denominator because the column names what the generator emits
    PER DOCUMENT, which is that record and no other.
    """
    label_fields = {field["name"] for field in CONTRACT["document_label"]["fields"]}
    rows = _prd_rows()
    assert rows, "no `prd_required_fields` rows found — this test asserts nothing"

    unknown = [
        f"{doc_type}.{name}"
        for doc_type, row in rows
        for name in row["generator"]
        if name not in label_fields
    ]
    assert not unknown, (
        f"{len(unknown)} names across {len(rows)} rows are not `document_label` fields: {unknown}"
    )
