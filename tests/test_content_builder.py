"""The assembled ПРРО receipt — every invariant holding at once, on generated content.

The per-invariant suites pin each rule in isolation. This one asserts that the builder
actually respects all of them together, over many seeds, which is the only way a drift
between two individually-correct pieces shows up.
"""

from __future__ import annotations

import inspect
import random
import re
from datetime import date, datetime
from decimal import Decimal

import pytest
from faker import Faker

from receipt_synth import config
from receipt_synth.claim_planner import ARCHETYPES
from receipt_synth.config import (
    acquirers,
    category,
    fiscal_makers,
    high_frequency_surnames,
    jurisdiction,
    load_generation,
    load_policy,
    load_vendors,
    price_range,
    unprintable_item_kinds,
    vendor_profile,
)
from receipt_synth.content_builder import (
    MAX_LINE_ITEMS,
    _fill_placeholders,
    build_prro_receipt,
    estimated_line_value,
    is_valid_rnokpp,
    legal_name,
    personal_surname,
    resolve_vendor,
    sellable_kinds,
    sole_trader_name,
    validate_amount_in_words,
    validate_line_item_sum,
    validate_vat_letter,
    vendor_can_carry,
)
from receipt_synth.policy_engine import (
    covered_total,
    document_evidence,
    resolved_coverage,
    verdict_for,
)
from receipt_synth.schemas import Capture, DocType, Verdict

ISSUED_AT = datetime(2026, 8, 3, 14, 22, 51)
# `vat_payer` is as load-bearing as `legal_form`: it decides the seller's tax-identifier line
# and whether the receipt has a VAT block at all. A pharmacy chain is registered; a sole
# trader on the simplified system is not. The variety itself is exercised in
# tests/test_vat_payer_status.py — here the flag is only what these fixtures need to build.
VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV", "profile": "pharmacy", "vat_payer": True}
# Sole-trader names are drawn, never written down — so the fixtures resolve one instead of
# stating it, exactly as the assembler does when it picks a vendor for a claim.
SOLE_TRADER = resolve_vendor(
    random.Random(11),
    {"legal_form": "FOP", "profile": "nutrition_practice", "vat_payer": False},
    "UA",
)
# A profile that sells nothing its category excludes. `private_tutor` is one on purpose —
# a tutor sells lessons and no goods — and it is the residual leak recorded under
# `known_limitations` in config/labelling-schema.yaml.
COVERED_ONLY = resolve_vendor(
    random.Random(12),
    {"legal_form": "FOP", "profile": "private_tutor", "vat_payer": False},
    "UA",
)

# ФОП Прізвище І. П. — surname, then two initials, as a Ukrainian document prints it.
SOLE_TRADER_NAME = re.compile(r"[А-ЯЇІЄҐ][а-яїієґ'’\-]+ [А-ЯЇІЄҐ]\. [А-ЯЇІЄҐ]\.")


def build(seed: int, vendor: dict = VENDOR, **kwargs):
    return build_prro_receipt(
        random.Random(seed),
        category_id="vitamins_nutrition",
        issued_at=ISSUED_AT,
        vendor=vendor,
        **kwargs,
    )


# ------------------------------------------------------------------ contract --


def test_is_deterministic_under_seed():
    assert build(42) == build(42)


def test_different_seeds_give_different_receipts():
    """If they did not, the seed would not be reaching the content and a whole dataset
    would be one receipt repeated."""
    assert len({build(s).receipt_number for s in range(20)}) > 1
    assert len({build(s).total for s in range(20)}) > 1


def test_has_at_least_two_line_items():
    for seed in range(20):
        assert len(build(seed).line_items) >= 2


# ---------------------------------------------------------------- invariants --


@pytest.mark.parametrize("seed", range(30))
def test_all_invariants_hold(seed):
    receipt = build(seed)

    assert validate_line_item_sum(receipt.line_items, receipt.total)
    assert validate_amount_in_words(receipt.amount_in_words, receipt.total)
    # A registered ПДВ payer prints its identification code under "ІД" AND its VAT-payer number
    # under "ПН" — two lines, not one. Which lines a seller carries, their lengths and the whole
    # of the VAT block are tests/test_vat_payer_status.py.
    assert receipt.seller.tax_code_label == "ІД"
    assert receipt.seller.vat_number is not None
    for item in receipt.line_items:
        assert validate_vat_letter(item.item_kind, item.vat_letter, "UA")


@pytest.mark.parametrize("seed", range(10))
def test_a_sole_trader_prints_a_rnokpp_and_no_quotes(seed):
    """A non-payer prints its own tax number, and for a ФОП that number is the РНОКПП: there
    is no ЄДРПОУ to print, so an eight-digit code would name the seller with an identifier no
    register could resolve to them. A sole trader's name is a person's, printed without
    quotes."""
    receipt = build(seed, vendor=SOLE_TRADER)

    assert receipt.seller.tax_code_label == "ІД"
    assert is_valid_rnokpp(receipt.seller.tax_code)
    assert legal_name(receipt.seller) == f"ФОП {SOLE_TRADER['name']}"
    assert SOLE_TRADER_NAME.fullmatch(SOLE_TRADER["name"]), SOLE_TRADER["name"]
    assert legal_name(build(seed).seller) == "ТОВ «Аптека АНЦ»"


@pytest.mark.parametrize("seed", range(30))
def test_covered_only_receipt_contains_no_excluded_item(seed):
    """The label-first knob for this skeleton: the planner asked for `covered`, so the
    builder may draw only from the category's covered items. A single excluded line
    would silently make the claim partially covered and the label wrong."""
    covered_kinds = set(category("vitamins_nutrition")["covered_items"])
    receipt = build(seed, covered_only=True)

    assert all(item.covered for item in receipt.line_items)
    assert {item.item_kind for item in receipt.line_items} <= covered_kinds


# ---------------------------------------------------------- mixed baskets ----


def mixed(seed: int, coverage_target: str = "0.7", **kwargs):
    return build(seed, covered_only=False, coverage_target=Decimal(coverage_target), **kwargs)


@pytest.mark.parametrize("seed", range(30))
def test_a_mixed_basket_draws_from_both_buckets(seed):
    spec = category("vitamins_nutrition")
    receipt = mixed(seed)

    kinds = {item.item_kind for item in receipt.line_items}
    assert kinds & set(spec["covered_items"]), "no covered line"
    assert kinds & set(spec["excluded_items"]), "no non-covered line"
    assert kinds.isdisjoint(spec["ambiguous_items"]), (
        "policy.yaml states no coverage answer for the ambiguous bucket, so a line drawn "
        "from it would carry a `covered` flag this generator invented"
    )


@pytest.mark.parametrize("seed", range(30))
def test_the_covered_flag_agrees_with_the_bucket_the_line_came_from(seed):
    """The flag is the label. A line from `excluded_items` marked covered would be a
    wrong per-line label *and* a wrong claim verdict, from one mistake."""
    spec = category("vitamins_nutrition")
    for item in mixed(seed).line_items:
        assert item.covered is (item.item_kind in spec["covered_items"])
        assert item.covered is not (item.item_kind in spec["excluded_items"])


@pytest.mark.parametrize("seed", range(40))
def test_a_mixed_basket_lands_on_the_partially_covered_side_of_the_threshold(seed):
    """What the builder actually owes the planner. The realized ratio only approaches the
    target — prices are clamped into each item kind's own range so the receipt stays
    plausible — but the verdict it produces has to be the one that was asked for. Under
    the STRICT rule that is guaranteed by the existence of a non-covered line rather than
    by where the ratio lands, which is exactly why the clamp is tolerable."""
    receipt = mixed(seed)
    items = receipt.line_items
    covered = covered_total("vitamins_nutrition", items)
    every_line_covered = all(resolved_coverage("vitamins_nutrition", items))

    assert not every_line_covered
    assert verdict_for(covered, receipt.total, every_line_covered=every_line_covered) is (
        Verdict.PARTIALLY_COVERED
    )


@pytest.mark.parametrize("target", ["0.3", "0.5", "0.7", "0.9"])
def test_the_realized_coverage_tracks_the_requested_one(target):
    """Averaged over seeds, because a single basket is a draw. Within 0.15 of the target:
    the clamp on prices is what stops it being exact, and closer than that is not
    something a label depends on — the verdict turns on the non-covered line existing, not
    on the ratio."""
    wanted = Decimal(target)
    realized = [
        covered_total("vitamins_nutrition", r.line_items) / r.total
        for r in (mixed(seed, coverage_target=target) for seed in range(40))
    ]
    mean = sum(realized) / len(realized)
    assert abs(mean - wanted) < Decimal("0.15"), f"asked {wanted}, got {mean}"


@pytest.mark.parametrize("seed", range(20))
def test_a_mixed_basket_still_satisfies_every_document_invariant(seed):
    receipt = mixed(seed)
    assert validate_line_item_sum(receipt.line_items, receipt.total)
    assert validate_amount_in_words(receipt.amount_in_words, receipt.total)
    assert sum(line.gross for line in receipt.tax_lines) == receipt.total
    for item in receipt.line_items:
        assert validate_vat_letter(item.item_kind, item.vat_letter, "UA")
        assert item.price > 0


def test_the_non_covered_line_is_not_always_the_last_one():
    """Position is learnable. If every mixed receipt put its non-covered line at the
    bottom, a consumer could score well on this dataset without reading the line."""
    positions = set()
    for seed in range(40):
        items = mixed(seed).line_items
        positions |= {i for i, item in enumerate(items) if not item.covered}
    assert len(positions) > 1


def test_a_mixed_basket_is_deterministic_under_seed():
    assert mixed(7) == mixed(7)


def test_a_mixed_basket_refuses_to_guess_its_coverage():
    """The planner chose the verdict; a default here would let the builder choose one."""
    with pytest.raises(ValueError):
        build(1, covered_only=False)


@pytest.mark.parametrize("target", ["0", "1", "-0.5", "1.5"])
def test_a_coverage_target_outside_the_open_unit_interval_is_refused(target):
    """At 1 there is no non-covered line and at 0 there is no covered one; both are other
    verdicts, reached by other mechanisms."""
    with pytest.raises(ValueError):
        mixed(1, coverage_target=target)


def test_a_covered_only_basket_refuses_a_coverage_target():
    with pytest.raises(ValueError):
        build(1, covered_only=True, coverage_target=Decimal("0.7"))


def test_a_receipt_longer_than_the_cap_is_refused():
    with pytest.raises(ValueError):
        build(1, item_count=MAX_LINE_ITEMS + 1)


def test_a_basket_can_be_sized_up_to_the_cap():
    """The planner sizes a basket upward to overrun an annual limit. If the distinct-name
    draw ran out of attempts first, the overrun would silently not happen."""
    receipt = build(3, item_count=MAX_LINE_ITEMS)
    assert len(receipt.line_items) == MAX_LINE_ITEMS
    assert len({item.name for item in receipt.line_items}) == MAX_LINE_ITEMS


# ----------------------------------------------------- sole-trader identity --


def test_a_sole_trader_name_is_drawn_in_the_printed_form():
    """Surname plus two initials, which is how a Ukrainian document prints a ФОП."""
    for seed in range(30):
        name = sole_trader_name(random.Random(seed), "UA")
        assert SOLE_TRADER_NAME.fullmatch(name), name


def test_a_drawn_surname_comes_from_the_narrowed_pool_and_not_from_faker():
    """🔴 The positive test for the path taken, not the absence of a symptom.

    `personal_surname` falls back to Faker where a language declares no pool, and a fallback
    that silently swallowed the Ukrainian pool would still produce a plausible ФОП name —
    nothing downstream would notice. So this asserts that every surname drawn for UA is IN
    the configured set, which the fallback cannot satisfy: Faker's uk_UA pool and this set
    overlap only partly, and its 524 surnames include the rare ones the narrowing removed.

    Asserting a specific surname is absent would be the weaker test. It would only fail to
    find the one name it names, while this fails for any surname from outside the pool.
    """
    pool = set(high_frequency_surnames("uk"))
    assert pool, "config/generation.yaml declares no Ukrainian surname pool"

    drawn = {sole_trader_name(random.Random(seed), "UA").split()[0] for seed in range(300)}
    assert drawn <= pool, sorted(drawn - pool)


def test_the_faker_fallback_stands_for_a_language_with_no_pool():
    """The other side of the branch, so the fallback is a tested path and not dead code.

    Only `uk` declares a pool today. A jurisdiction without one must still get a surname —
    `generate_persona` accepts PL, DE and ES — and it comes from Faker's own locale pool.
    """
    assert high_frequency_surnames("pl") == ()

    fake = Faker("pl_PL")
    fake.seed_instance(7)
    surname = personal_surname(random.Random(7), fake, "PL", female=False)
    assert surname and surname not in set(high_frequency_surnames("uk"))


def test_drawn_sole_trader_names_are_deterministic_and_varied():
    """Deterministic, or a seed stops reproducing a run. Varied, or the draw has replaced a
    stored constant with a computed one."""
    assert sole_trader_name(random.Random(5), "UA") == sole_trader_name(random.Random(5), "UA")
    assert len({sole_trader_name(random.Random(s), "UA") for s in range(40)}) > 30


def test_a_sole_trader_name_is_drawn_once_per_vendor_instance_and_carried():
    """🔴 The constraint that lost its guarantee when the name stopped being a constant.

    Two documents of one claim are issued by one seller, so they must print one name. While
    the name was a stored string this held by the nature of the type — a constant cannot
    differ from itself — and nothing had to enforce it or test it. A drawn name can differ,
    and the requirement did not change, so the guarantee has to be built and checked.

    The vendor instance is resolved ONCE by the assembler when it picks a vendor for the
    claim, and carried into every document; here the same instance is asked for two receipts
    with different generators, which is what a second document of the claim would do.
    """
    vendor = resolve_vendor(
        random.Random(3), {"legal_form": "FOP", "profile": "pharmacy", "vat_payer": False}, "UA"
    )

    first = build(1, vendor=vendor)
    second = build(2, vendor=vendor)

    assert first.seller.name == second.seller.name
    assert first.receipt_number != second.receipt_number, "otherwise this proves nothing"
    truths = [
        r.ground_truth(doc_id=f"d{i}", source_file=f"d{i}.png",
                       capture=Capture.SCREENSHOT, field_bboxes={})
        for i, r in enumerate((first, second))
    ]
    assert truths[0].counterparty == truths[1].counterparty


def test_the_document_builder_cannot_choose_a_vendor_of_its_own():
    """The other half of the constraint, and the half a unit test cannot reach by building.

    A claim yields one document today, so no test can catch a per-document redraw by
    comparing two of them through the assembler — the second document does not exist yet.
    What can be stated now is that `_build_document` is unable to pick a vendor at all: it
    takes one, and `_pick_vendor` is not in its code. Whoever adds the second document is
    then forced to decide where the vendor comes from instead of getting a fresh one by
    default, which is exactly how this guarantee was lost the first time.
    """
    from receipt_synth import assembler

    assert "vendor" in inspect.signature(assembler._build_document).parameters
    assert "_pick_vendor" not in assembler._build_document.__code__.co_names


def test_resolving_a_vendor_twice_is_what_would_break_it():
    """The failure the constraint above guards against, shown rather than described: resolve
    the same entry twice and the two names differ. That is why the resolve happens where the
    vendor is chosen and not where a document is built."""
    entry = {"legal_form": "FOP", "profile": "pharmacy"}
    names = {resolve_vendor(random.Random(s), dict(entry), "UA")["name"] for s in range(10)}
    assert len(names) > 1


def test_a_firm_keeps_the_name_it_states():
    """Only entries without a name are drawn for. A ТОВ trades under a mark, and resolving
    it must not overwrite it."""
    assert resolve_vendor(random.Random(1), VENDOR, "UA") == VENDOR


def test_a_nameless_entry_that_is_not_a_person_is_refused():
    """Absence of a name means "trades under a person's name". A limited company with no
    name is a hole in the catalogue, not an instruction to invent a person."""
    with pytest.raises(ValueError, match="trades under a mark"):
        resolve_vendor(random.Random(1), {"legal_form": "TOV", "profile": "pharmacy"}, "UA")


def test_no_personal_name_is_stored_in_the_vendor_catalogue():
    """The liability this replaced. A stored invented personal name is an unverified claim
    that no real person trades under it, and it has to be re-checked as the world changes;
    a drawn one makes no claim at all. `EK` is excluded: a German registered sole merchant
    may trade under a business designation, so for that form a stated name is legitimate.
    """
    drawn_forms = {"FOP", "JDG", "AUTONOMO"}
    stored = [
        f"{country}/{category_id}: {vendor['name']}"
        for country, category_id, vendor in every_vendor()
        if vendor["legal_form"] in drawn_forms and "name" in vendor
    ]
    assert not stored, f"sole-trader names must be drawn, not written down: {stored}"


# ------------------------------------------------------- vocabulary coverage --


def every_template():
    """(category id, bucket, item kind, language, template) for the whole policy."""
    for spec in load_policy()["categories"]:
        for bucket in ("covered_items", "excluded_items", "ambiguous_items"):
            for kind, by_language in spec.get(bucket, {}).items():
                for language, templates in by_language.items():
                    for template in templates:
                        yield spec["id"], bucket, kind, language, template


def fills(template: str, kind: str, language: str) -> bool:
    try:
        _fill_placeholders(template, kind, random.Random(0), language)
    except KeyError:
        return False
    return True


def test_every_name_template_in_the_policy_can_be_filled():
    """The gate that stops the vocabulary narrowing again.

    A template whose placeholder had no vocabulary used to be skipped, silently, which
    made the printed vocabulary of the dataset a subset of the one policy.yaml declares —
    with nothing anywhere saying which subset. Filling every template is now the invariant,
    and a new template naming a new placeholder fails here rather than at generation time
    on a machine nobody is watching.

    The exception is enumerated, not open: `unprintable_item_kinds` in generation.yaml names
    the kinds that cannot be filled and why, and the test below holds that list to being
    true. A kind may therefore be absent from the dataset, but never quietly.
    """
    declared = unprintable_item_kinds()
    unfillable = [
        f"{category_id}.{bucket}.{kind}[{language}]: {template}"
        for category_id, bucket, kind, language, template in every_template()
        if kind not in declared and not fills(template, kind, language)
    ]
    assert not unfillable, "\n".join(unfillable)


def test_every_kind_declared_unprintable_really_cannot_be_printed():
    """The other direction, and the one that stops the declaration becoming a dumping
    ground. An entry that has quietly become printable — because somebody filled its
    vocabulary — would go on excluding a kind from every dataset for a reason that no longer
    holds, and the exclusion would look deliberate. Each entry must also carry its reason,
    because a bare list of kinds records what was done and not why."""
    reasons = load_generation()["unprintable_item_kinds"]
    templates: dict[str, list[str]] = {}
    for _, _, kind, language, template in every_template():
        if language == "uk":
            templates.setdefault(kind, []).append(template)

    for kind, reason in reasons.items():
        assert kind in templates, f"{kind!r} declared unprintable, but policy.yaml has no such kind"
        assert len(reason.split()) > 5, f"{kind!r} is declared unprintable with no reason"
        assert not all(fills(t, kind, "uk") for t in templates[kind]), (
            f"{kind!r} is declared unprintable but every template of it now fills — remove "
            "it from unprintable_item_kinds in config/generation.yaml"
        )


def test_an_unprintable_kind_is_never_offered_to_a_draw():
    """Declaring a kind unprintable and then drawing it would raise mid-run. The filter is
    in `sellable_kinds`, so it holds for every bucket and every vendor at once."""
    declared = unprintable_item_kinds()
    assert declared, "this test asserts nothing if nothing is declared"

    for spec in load_policy()["categories"]:
        for bucket in ("covered_items", "excluded_items"):
            for country, category_id, vendor in every_vendor():
                if category_id != spec["id"]:
                    continue
                resolved = vendor if "name" in vendor else {**vendor, "name": "x"}
                offered = set(sellable_kinds(spec[bucket], resolved))
                assert not offered & declared, f"{country}/{category_id}: {offered & declared}"


def test_every_excluded_kind_of_a_generating_category_can_be_drawn():
    """The positive counterpart to the declaration above, and the invariant KL-02 in
    config/labelling-schema.yaml is measured against.

    A kind declared unprintable is excluded from every draw, so a NON-COVERED kind that lands
    in that list narrows the non-covered vocabulary of every document the generator produces —
    which is a property of the corpus that no label mentions. `hygiene` was in it, so the
    excluded vocabulary of the only category with an archetype was three kinds where the policy
    declares four, and the head-noun families a consumer could learn were one fewer than the
    label space says. The test asserts the whole of the category's `excluded_items` is
    reachable, so a kind falling out of the dataset cannot go unnoticed again.

    Scoped to the (jurisdiction, category) pairs some registered archetype carries: a category
    no template can document reaches no dataset either way, and holding one to this would fail
    on the PL / DE / ES lists, which are seeded rather than filled. The union is taken over the
    vendors of THAT jurisdiction only — pooling every country's vendors would let a Polish entry
    cover a kind no Ukrainian receipt can print.

    AND SCOPED TO ARCHETYPES THAT STATE WHAT WAS BOUGHT, which is narrower than "registered" now
    that the registry holds a second class. A line item reaches a document only through a document
    that lists items: a bank payment confirmation carries every benefit category — nothing on it
    can contradict one, since 👁 it lists nothing at all — so scoping by registration alone would
    ask which vendors can print the excluded kinds of SEVEN categories onto a document that
    prints no kinds whatever. The property is about baskets, so the scope is documents with
    baskets.
    """
    documented = {
        (archetype.country.value, category_id)
        for archetype in ARCHETYPES.values()
        if document_evidence(archetype.doc_type).proves_subject
        for category_id in archetype.categories
    }
    assert documented, "no archetype registered — this test would assert nothing"

    for country, category_id in sorted(documented):
        spec = category(category_id)
        sellers = {
            kind
            for vendor_country, vendor_category, vendor in every_vendor()
            if (vendor_country, vendor_category) == (country, category_id)
            # A vendor with no stored name trades under a drawn one; the name plays no part in
            # what it sells, so any placeholder does.
            for kind in sellable_kinds(
                spec["excluded_items"], vendor if "name" in vendor else {**vendor, "name": "x"}
            )
        }
        missing = sorted(set(spec["excluded_items"]) - sellers)
        assert not missing, (
            f"{country}/{category_id}: no vendor can put these excluded kinds on a receipt, so "
            "they are absent from every document and the non-covered vocabulary is narrower "
            f"than policy.yaml declares: {missing}"
        )


def test_a_placeholder_with_no_vocabulary_fails_loudly():
    """The other half of the same rule: unfilled must raise, not skip. If this ever
    returned a value or silently dropped the template, the sweep above would pass on a
    dataset whose vocabulary had quietly shrunk."""
    with pytest.raises(KeyError, match="unheard_of"):
        _fill_placeholders("Товар {unheard_of}", "vitamin_complex", random.Random(0))


def test_a_filled_name_carries_no_leftover_brace():
    declared = unprintable_item_kinds()
    for _, _, kind, language, template in every_template():
        if kind in declared:
            continue
        name = _fill_placeholders(template, kind, random.Random(1), language)
        assert "{" not in name and "}" not in name, name


# ---------------------------------------------------------- vendor affinity --


def test_a_pharmacy_does_not_sell_a_nutrition_plan():
    """The affinity this step exists for. `nutritionist_visit` is a covered kind of the
    same category, so nothing but the vendor profile keeps it off a pharmacy receipt."""
    covered = category("vitamins_nutrition")["covered_items"]

    assert "nutritionist_visit" in covered
    assert "nutritionist_visit" not in sellable_kinds(covered, VENDOR)
    assert sellable_kinds(covered, SOLE_TRADER) == ["nutritionist_visit"]

    for seed in range(30):
        kinds = {item.item_kind for item in build(seed).line_items}
        assert "nutritionist_visit" not in kinds


def test_a_covered_only_vendor_cannot_carry_a_mixed_basket():
    """It sells nothing the category excludes, so there is no non-covered line to print.
    The assembler asks this before choosing a vendor; the builder refuses if asked anyway,
    because a mixed basket without a non-covered line would be labelled a verdict it does
    not realize."""
    assert vendor_can_carry(COVERED_ONLY, "language_courses", mixed=False)
    assert not vendor_can_carry(COVERED_ONLY, "language_courses", mixed=True)

    with pytest.raises(ValueError, match="mixed basket"):
        build_prro_receipt(
            random.Random(1),
            category_id="language_courses",
            issued_at=ISSUED_AT,
            vendor=COVERED_ONLY,
            covered_only=False,
            coverage_target=Decimal("0.7"),
        )


@pytest.mark.parametrize("seed", range(20))
def test_a_nutrition_practice_can_print_a_non_covered_line(seed):
    """Otherwise its receipts would be fully covered by construction, and the counterparty
    would foretell the label for every document it issued.

    A nutritionist stocking a home blood-pressure monitor and glucose test strips is
    ordinary practice — both are over-the-counter retail devices a patient is asked to
    self-monitor with — and `medical_device` is already an excluded kind of this category
    with templates for exactly those two articles. No policy change was needed to close
    the leak, only an honest reading of what such a practice sells.
    """
    assert vendor_can_carry(SOLE_TRADER, "vitamins_nutrition", mixed=True)
    assert vendor_can_carry(VENDOR, "vitamins_nutrition", mixed=True)

    receipt = mixed(seed, vendor=SOLE_TRADER)
    kinds = {item.item_kind for item in receipt.line_items if not item.covered}
    assert kinds == {"medical_device"}


def test_estimated_line_value_is_inside_the_price_ranges_it_summarizes():
    """It sizes a basket before the basket is drawn, so it only has to be the right order
    of magnitude — but a value outside every range would mean it summarizes nothing."""
    value = estimated_line_value("vitamins_nutrition")
    assert Decimal("90.00") < value < Decimal("1500.00")


@pytest.mark.parametrize("seed", range(30))
def test_tax_lines_account_for_every_line_item(seed):
    """Each VAT group's gross is the sum of its lines, and the groups partition the
    receipt — so the tax block totals back to the receipt total."""
    receipt = build(seed)

    assert sum(line.gross for line in receipt.tax_lines) == receipt.total
    assert {line.letter for line in receipt.tax_lines} == {
        item.vat_letter for item in receipt.line_items
    }


@pytest.mark.parametrize("seed", range(30))
def test_vat_amount_is_the_tax_inside_the_gross(seed):
    """Ukrainian receipts print VAT-inclusive prices, so the tax is extracted from the
    gross rather than added to it: vat = gross − gross / (1 + rate/100)."""
    for line in build(seed).tax_lines:
        rate = Decimal(str(line.rate))
        expected = (line.gross - line.gross / (1 + rate / 100)).quantize(Decimal("0.01"))
        assert line.vat == expected
        assert line.vat < line.gross


def test_both_vat_groups_are_reachable():
    """A pharmacy basket mixes 7% and 20% lines. If every receipt came out with a single
    tax line, the tax block would never be exercised in the dataset."""
    group_counts = {len(build(seed).tax_lines) for seed in range(40)}
    assert max(group_counts) >= 2


# ------------------------------------------------------------------- fiscals --


def test_receipt_number_is_an_eleven_character_alphanumeric_id():
    """Modern ПРРО issue a short alphanumeric id, not a sequential number — a field a
    consumer has to extract, so the shape matters."""
    for seed in range(20):
        assert re.fullmatch(r"[A-Za-z0-9]{11}", build(seed).receipt_number)


def test_fiscal_device_number_is_ten_digits():
    value = build(1).fiscal_device_number
    assert len(value) == 10
    assert value.isdigit()


def test_the_title_is_the_bare_wording_while_no_tag_pairing_is_public():
    """REWRITTEN, AND THE OLD ASSERTION PINNED SOMETHING UNGROUNDED. It required a tagged title
    to occur — "ФІСКАЛЬНИЙ ЧЕК" plus a short abbreviation — over a pool of tags paired with
    NOTHING, which was harmless only while no maker was printed beside them.

    👁 The tag itself is observed and the observation stands, recorded under
    `verified_against_own_receipts` in config/fiscal-rules.yaml. What no published source gives is
    the PAIRING: which provider prints which tag. Now that 📄 line 35's maker name is printed, a
    tag beside a NAMED provider would assert a pairing nobody established, so none is printed and
    every title is the bare wording.

    This is the tripwire for that regression: a tag reappearing without a pairing to justify it
    turns this red. The mechanism is not dead — the patched-pool test below exercises it — so this
    asserts a state of the DATA, not a missing feature.
    """
    titles = {build(seed).title for seed in range(40)}
    assert titles == {"ФІСКАЛЬНИЙ ЧЕК"}, (
        "a title carries a provider tag; no public source pairs a tag with a provider, so a "
        "tag printed beside the maker's name asserts a pairing nobody established"
    )


# ------------------------------------------------- which kind of cash register --
#
# ⚠️ «ЗН» and «ФН» are NOT a pair. 👁 The observed ПРРО receipts print the fiscal number
# alone; 👁 the two published hardware samples print the factory serial as well. The set of
# fiscal identity lines follows the kind of register, and the tests below are what stops the
# generator from printing a combination no observed receipt of either kind carries.


def registrars() -> dict:
    return jurisdiction("UA")["receipt"]["registrars"]


def test_only_a_hardware_register_prints_a_factory_serial():
    """The pair that is not a pair. A ПРРО has no заводський номер to print — it is software —
    so «ЗН» on its receipt would be a requisite the device cannot have; a hardware register
    prints both lines. Both directions are asserted, because a builder that printed «ЗН» on
    everything and one that printed it on nothing would each satisfy only one of them."""
    for seed in range(10):
        assert build(seed).device_serial is None, "a ПРРО has no factory serial"

        serial = build(seed, registrar="rro").device_serial
        assert serial is not None, "a hardware register prints its factory serial"
        assert re.fullmatch(
            jurisdiction("UA")["identifiers"]["device_serial"]["pattern"], serial
        ), f"{serial!r} does not match the configured ЗН pattern"


def test_the_fiscal_number_prefix_follows_the_kind_of_register():
    """👁 A ПРРО prints «ФН ПРРО», a hardware register prints «ФН». Read from config rather
    than restated: the label was a literal in the template, which is how the hardware
    archetype would have printed a ПРРО's prefix."""
    for kind, rules in registrars().items():
        assert build(1, registrar=kind).fiscal_number_label == rules["fiscal_number_label"]
    assert len({rules["fiscal_number_label"] for rules in registrars().values()}) == 2, (
        "the two prefixes are the same string, so this test discriminates nothing"
    )


def test_the_receipt_number_format_follows_the_kind_of_register():
    """👁 A ПРРО issues a short alphanumeric id; a hardware register counts its receipts, so
    the number is a short sequence of digits. Both patterns are read from
    `receipt.receipt_number` in config/fiscal-rules.yaml, so a length written into the builder
    beside the pattern cannot drift from it — the alphanumeric length used to be an `11` in
    code with a `{11}` in config and nothing tying them together.
    """
    patterns = jurisdiction("UA")["receipt"]["receipt_number"]
    for kind, rules in registrars().items():
        pattern = patterns[rules["receipt_number_format"]]["pattern"]
        for seed in range(10):
            number = build(seed, registrar=kind).receipt_number
            assert re.fullmatch(pattern, number), f"{kind}: {number!r} against {pattern}"

    # The sequential format states a RANGE of lengths, and a corpus printing one of them would
    # teach a consumer that width rather than the field.
    lengths = {len(build(seed, registrar="rro").receipt_number) for seed in range(40)}
    assert len(lengths) > 1, "every hardware receipt number is the same length"


def test_only_a_prro_prints_the_online_marker():
    """📄 The mode marker is line 31 of the published form and a ПРРО requisite. A hardware
    receipt carries no such line, and `None` is what says so — an empty string would render as
    a blank line with a bounding box pointing at nothing."""
    assert build(1).mode_marker == "Онлайн"
    assert build(1, registrar="rro").mode_marker is None


def test_the_title_tag_and_the_maker_name_come_from_one_pool_entry():
    """📄 Line 35 of the form is ONE requisite — the wording «ФІСКАЛЬНИЙ ЧЕК» together with the
    maker's name — and 👁 the title may also carry the maker's own short tag. So the tag and the
    name are two printed forms of one fact, and this is the test that they cannot name different
    makers: drawn from two independent lists, a receipt could show one provider's tag above
    another provider's name.

    Read from `config.fiscal_makers`, which is where the pairing now lives: each kind of register
    names a pool of config/vendors.json and an entry carries both fields. The pools differ per
    kind, so a maker drawn for a hardware receipt cannot come from the software list.

    ⚠️ THIS TEST CANNOT DISCRIMINATE A MISPAIRING ON TODAY'S DATA, and saying so is the point.
    Every entry's tag is empty, so ("", any name in the pool) satisfies the assertion and code
    that printed one entry's tag beside another's name would pass here. Verified rather than
    assumed: overriding the drawn name with the pool's first entry left this test green. What it
    still catches is a maker drawn from OUTSIDE the pool. The mispairing itself is caught by
    `test_a_configured_tag_is_printed_beside_its_own_maker`, on a pool whose tags differ — which
    is the only shape in which the failure is observable at all.
    """
    for kind, rules in registrars().items():
        entries = set(fiscal_makers(rules["maker_pool"], "UA"))
        assert entries, f"{kind} draws from an empty maker pool"
        for seed in range(30):
            receipt = build(seed, registrar=kind)
            tag = receipt.title.removeprefix(jurisdiction("UA")["receipt"]["title"]).strip()
            assert (tag, receipt.provider_name) in entries, (
                f"{kind}: title tag {tag!r} and maker {receipt.provider_name!r} are not one "
                f"entry of the {rules['maker_pool']!r} pool"
            )


def test_the_two_kinds_of_register_draw_makers_from_different_pools():
    """A ПРРО is published by a software provider; a hardware register is built by a manufacturer
    entered in the state register. Printing a software provider as the maker of a physical device
    would be false about both, and one shared pool is all it would take."""
    pools = {kind: rules["maker_pool"] for kind, rules in registrars().items()}
    assert len(set(pools.values())) == len(pools), f"two kinds share a maker pool: {pools}"

    names = {
        kind: {name for _, name in fiscal_makers(pool, "UA")} for kind, pool in pools.items()
    }
    software, hardware = names["prro"], names["rro"]
    assert software and hardware
    assert not software & hardware, f"a maker is in both pools: {software & hardware}"

    for seed in range(20):
        assert build(seed, registrar="prro").provider_name in software
        assert build(seed, registrar="rro").provider_name in hardware


def test_a_configured_tag_is_printed_beside_its_own_maker():
    """The tag mechanism, exercised on a PATCHED pool because no real entry carries a tag.

    TWO JOBS, and the second is the one the real-data test above cannot do. First, the wiring:
    without this the mechanism would be untested for as long as the data declines to use it, and
    the bare-wording tripwire would pass equally on a build that had lost the ability to print a
    tag at all. Second, THE MISPAIRING — a pool whose two entries carry DIFFERENT tags is the only
    shape in which one maker's tag beside another's name is observable, because with every tag
    empty the mispaired result is indistinguishable from the correct one.

    Both entries are exercised over several seeds, so the assertion is about the pairing rather
    than about which entry a single draw happened to take.
    """
    from receipt_synth import content_builder

    tagged = (("AA", "Каса Альфа"), ("BB", "Каса Бета"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(content_builder, "fiscal_makers", lambda pool, country: tagged)
        printed = {
            (
                receipt.title.removeprefix(jurisdiction("UA")["receipt"]["title"]).strip(),
                receipt.provider_name,
            )
            for receipt in (build(seed) for seed in range(20))
        }

    assert printed <= set(tagged), f"a tag was printed beside another maker's name: {printed}"
    assert printed == set(tagged), (
        f"only {printed} of {set(tagged)} was ever drawn, so the pairing is pinned for one entry"
    )


def test_a_maker_name_is_always_printed():
    """👁 11 of 11 open receipts carry a name immediately after the fiscal wording, which
    makes it the best-evidenced fact of the foot block. An entry with an empty name would
    print the wording alone and satisfy every other test here."""
    for kind in registrars():
        for seed in range(10):
            assert build(seed, registrar=kind).provider_name.strip()


def test_an_unknown_kind_of_register_is_refused_by_name():
    """The kinds are keys of config/fiscal-rules.yaml, and a slug mapped to a kind that file
    does not declare has to say which kinds exist rather than fail on a `KeyError` inside the
    draw, three statements away from the mistake."""
    with pytest.raises(ValueError, match="declares no registrar 'krro'"):
        build(1, registrar="krro")


def test_acquiring_block_matches_the_configured_patterns():
    """Patterns are read from fiscal-rules.yaml rather than restated, so this test is
    what ties the generated values to the configuration. Restating them here would let
    the two drift apart while both looked correct."""
    patterns = {
        field["key"]: field["pattern"]
        for field in jurisdiction("UA")["acquiring_block"]["fields"]
        if "pattern" in field
    }
    assert patterns, "the UA acquiring block should declare patterns"

    for seed in range(10):
        acq = build(seed).acquiring
        for key, pattern in patterns.items():
            assert re.fullmatch(pattern, getattr(acq, key)), f"{key} does not match {pattern}"


def test_the_skeleton_always_pays_by_card():
    """A card payment is the more informative document: the acquiring block carries the
    RRN, which is the deduplication key. Cash receipts have no acquiring block at all
    and are a separate variation, not a coin flip inside this one."""
    receipt = build(3)
    assert receipt.acquiring is not None
    assert receipt.payment_method == "БЕЗГОТІВКОВА"


def test_qr_payload_is_the_tax_authority_verification_url():
    receipt = build(5)
    assert receipt.qr_payload.startswith("https://cabinet.tax.gov.ua/")
    assert receipt.receipt_number in receipt.qr_payload
    assert receipt.fiscal_device_number in receipt.qr_payload
    assert f"{receipt.total:.2f}" in receipt.qr_payload


def test_time_is_printed_with_dashes_not_colons():
    """ПРРО software prints 14-22-51, not 14:22:51."""
    assert build(1).render_context()["time"] == "14-22-51"


def test_decimal_separator_varies_between_receipts():
    """Vendors differ; both variants must appear or a consumer only ever sees one."""
    separators = {build(seed).decimal_separator for seed in range(40)}
    assert separators == {".", ","}


# -------------------------------------------------------------- ground truth --


def test_ground_truth_matches_the_receipt():
    receipt = build(9)
    truth = receipt.ground_truth(
        doc_id="p001_c1_d1",
        source_file="p001_c1_d1.png",
        capture=Capture.SCREENSHOT,
        field_bboxes={"total": (1.0, 2.0, 3.0, 4.0)},
    )

    assert truth.doc_type is DocType.FISCAL_RECEIPT
    assert truth.language == "uk"
    assert truth.currency == "UAH"
    assert truth.amount == receipt.total
    assert truth.date == date(2026, 8, 3)
    assert truth.counterparty == receipt.seller.name
    assert truth.line_items == receipt.line_items
    assert truth.field_bboxes == {"total": (1.0, 2.0, 3.0, 4.0)}


def test_ground_truth_carries_its_provenance():
    truth = build(9).ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.synthetic is True
    assert truth.generator_version


def test_a_fiscal_receipt_declares_a_fiscal_qr_and_a_fiscal_number():
    """This archetype is the honest one. The traps that look like it but are not
    (templates 12 and 13) flip exactly these flags."""
    truth = build(9).ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.has_qr
    assert truth.qr_is_fiscal
    assert truth.has_fiscal_number


# ------------------------------------------------ configuration integrity ----


def all_item_kinds() -> set[str]:
    return {kind for _, _, kind, _, _ in every_template()}


def every_vendor():
    """(country, category id, vendor) over config/vendors.json."""
    for country, by_category in load_vendors()["vendors"].items():
        if country.startswith("$"):
            continue
        for category_id, vendors in by_category.items():
            for vendor in vendors:
                yield country, category_id, vendor


def test_every_vendor_profile_names_item_kinds_the_policy_declares():
    """A profile is a list of item kinds, and a typo in one silently narrows what the
    vendor sells instead of failing — the shop would simply never stock the misspelled
    kind, and the receipt would look fine."""
    known = all_item_kinds()
    profiles = load_vendors()["vendor_profiles"]

    for slug, kinds in profiles.items():
        if slug.startswith("$"):
            continue
        assert set(kinds) <= known, f"profile {slug!r} names unknown kinds: {set(kinds) - known}"


def test_every_item_kind_the_policy_declares_is_sold_by_some_profile():
    """The other direction of the profile check above, and the one that catches content
    disappearing rather than misspelled.

    A kind no profile sells cannot be printed by anybody. It keeps its templates, its brand
    vocabulary and its price range, all of them dead — and every run comes out missing it
    with nothing anywhere saying which kind went. That is the same failure the `{brand}`
    skip was removed to end, one layer further out: `hardware` was declared with three
    templates and sold by no profile, and a run printed it zero times without a word.
    """
    sold = set().union(
        *(
            set(kinds)
            for slug, kinds in load_vendors()["vendor_profiles"].items()
            if not slug.startswith("$")
        )
    )
    orphans = sorted(all_item_kinds() - sold)
    assert not orphans, f"declared by policy.yaml, sold by no profile: {orphans}"


def test_every_covered_kind_can_appear_on_a_partially_covered_document():
    """Item-kind leakage: a covered kind sold only by profiles that stock nothing the
    category excludes can never share a document with a non-covered line.

    The tell is then the LINE ITEM rather than the counterparty, which is worse — a
    consumer picks it up without ever looking at the seller. `online_course` had exactly
    this shape: both profiles selling it were covered-only, so an online course on a
    receipt guaranteed a fully covered claim.

    Stated over profiles rather than over vendors because it is a property of the affinity
    data. Whether a given jurisdiction has a vendor with that profile is a separate
    question, asked below for the jurisdictions that can actually generate.

    Kinds declared unprintable are removed from the profiles first: a seller whose only
    excluded article cannot be printed is a covered-only seller in practice, and counting it
    as mixed-capable would hide the leak behind a line that never appears.
    """
    declared = unprintable_item_kinds()
    profiles = {
        slug: set(kinds) - declared
        for slug, kinds in load_vendors()["vendor_profiles"].items()
        if not slug.startswith("$")
    }
    unreachable = []
    for spec in load_policy()["categories"]:
        excluded = set(spec["excluded_items"])
        for kind in spec["covered_items"]:
            if not any(kind in sells and sells & excluded for sells in profiles.values()):
                unreachable.append(f"{spec['id']}.{kind}")
    assert not unreachable, (
        "no profile sells these covered kinds alongside anything the category excludes, so "
        f"the line item alone foretells a fully covered claim: {unreachable}"
    )


def test_a_generating_jurisdiction_leaves_no_covered_kind_out_or_unmixable():
    """The same two invariants where they bite: the vendor lists of a jurisdiction that has
    a registered archetype, and therefore actually produces documents.

    Read from `ARCHETYPES` rather than hardcoded, so a jurisdiction whose first template
    lands is held to this the same day. PL, DE and ES are seeded rather than filled and
    generate nothing, so holding their three-entry lists to it now would fail on data that
    reaches no dataset.
    """
    countries = {archetype.country.value for archetype in ARCHETYPES.values()}
    assert countries, "no archetype registered — this test would assert nothing"

    for country in sorted(countries):
        for spec in load_policy()["categories"]:
            vendors = load_vendors()["vendors"][country].get(spec["id"], [])
            for kind in sorted(spec["covered_items"]):
                sellers = [v for v in vendors if kind in vendor_profile(v["profile"])]
                assert sellers, f"{country}/{spec['id']}: no vendor sells {kind!r}"
                assert any(
                    vendor_can_carry(v, spec["id"], mixed=True) for v in sellers
                ), (
                    f"{country}/{spec['id']}: every vendor selling {kind!r} is covered-only, "
                    "so that line item foretells the label"
                )


def test_every_vendor_can_sell_something_its_category_covers():
    """Otherwise the vendor is unusable: every receipt this generator builds carries at
    least one covered line, whatever the verdict."""
    for country, category_id, vendor in every_vendor():
        assert vendor_can_carry(vendor, category_id, mixed=False), (
            f"{country}/{category_id}: {vendor['name']!r} sells nothing the category covers"
        )


def test_every_category_has_a_vendor_that_can_carry_a_mixed_basket():
    """`partially_covered` by `mixed_items` is a fifth of the target verdict mix. A
    category whose vendors all sell services only could never realize it, and the failure
    would surface as an assembler exception mid-run rather than as a gap in the data."""
    seen = {(country, category_id) for country, category_id, _ in every_vendor()}
    for country, category_id in sorted(seen):
        assert any(
            vendor_can_carry(vendor, category_id, mixed=True)
            for c, cat, vendor in every_vendor()
            if (c, cat) == (country, category_id)
        ), f"{country}/{category_id}: no vendor sells anything the category excludes"


def test_price_ranges_are_ordered_and_within_a_coarse_sanity_band():
    """Ordering, and an outer band no article this generator prints leaves.

    NOT a guard against a range written in minor units by habit, and it was described as
    one until a reviewer wrote `stationery: ["1500.00", "8000.00"]` and watched the suite
    stay green. A hundredfold slip on a cheap kind lands inside the band, so the band
    cannot see it; a third of the kinds are priced under 1000 UAH and are invisible to it
    entirely. The only assertion that would catch every case is a per-kind band, which is
    this file restated in a test — two copies of the prices, and the copy in the test is
    the one nobody updates.

    What actually removes that class of error is upstream: prices are stated on the same
    scale as `annual_limit` in policy.yaml, so there is no second scale to slip into. The
    next test adds the one independent check that exists.
    """
    for kind in sorted(all_item_kinds() | {"default"}):
        low, high = price_range(kind)
        assert Decimal(0) < low < high, f"{kind}: {low}..{high}"
        assert Decimal("1.00") <= low and high <= Decimal("100000.00"), f"{kind}: {low}..{high}"


def test_no_covered_article_costs_more_in_one_line_than_a_year_of_the_benefit():
    """The one price check with an anchor outside config/generation.yaml: `annual_limit`.

    A covered kind whose CHEAPEST form already exceeds what the plan allows for a whole year
    is wrong on its own terms — no claim in that category could ever be fully reimbursed,
    and every document would be `partially_covered` by an exhausted limit. It also happens
    to catch a hundredfold slip on 20 of the 21 covered kinds, which is where the previous
    test's claim should have lived. It says nothing about excluded or ambiguous kinds: the
    plan sets no limit on what it does not pay for.
    """
    for spec in load_policy()["categories"]:
        limit = Decimal(str(spec["annual_limit"]))
        for kind in sorted(spec["covered_items"]):
            low, _ = price_range(kind)
            assert low < limit, (
                f"{spec['id']}.{kind}: cheapest line {low} exceeds the annual limit {limit}"
            )


def test_a_price_finer_than_the_currency_is_refused(monkeypatch):
    """Truncating the third decimal place silently would be the same class of mistake the
    single scale exists to prevent — a price quietly other than the one that was written."""
    monkeypatch.setattr(
        config,
        "load_generation",
        lambda: {"price_ranges": {"default": ["1.00", "2.00"],
                                  "ranges": {"vitamin_complex": ["1.005", "2.00"]}}},
    )
    config.price_range.cache_clear()
    try:
        with pytest.raises(ValueError, match="two decimal places"):
            config.price_range("vitamin_complex")
    finally:
        config.price_range.cache_clear()


def test_every_item_kind_has_a_price_range_of_its_own():
    """The `default` exists so that adding a kind to policy.yaml does not break generation
    — not so that a kind can stay on it. A kind priced by the fallback is a kind whose
    prices nobody chose."""
    priced = set(load_generation()["price_ranges"]["ranges"])
    assert all_item_kinds() <= priced, f"no price range for {sorted(all_item_kinds() - priced)}"


def test_acquirer_names_are_distinct():
    names = acquirers("UA")
    assert len(names) > 1, "one acquirer would put the same bank on every receipt"
    assert len(set(names)) == len(names)


def test_line_item_names_carry_no_unresolved_placeholders():
    """Templates in policy.yaml contain {dose}, {n}, {brand}. A leftover brace would be
    printed literally onto the receipt image."""
    for seed in range(30):
        for item in build(seed).line_items:
            assert "{" not in item.name and "}" not in item.name
