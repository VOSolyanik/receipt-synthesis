"""The payment date is defined by two files, and the boundary between them is the property.

`config/policy.yaml` names the concept — the date the funds left the payer's account — and no
printed caption. `config/labelling-schema.yaml` maps the printed captions onto that concept and
defines no concept of its own. Each half is inert without the other, which is exactly why the
split needs a guard: nothing inside either file reveals that the other one moved.

Why it is worth a test rather than a convention. A downstream consumer builds its own verdict
engine against policy.yaml. Were the concept unnamed there, two independent and equally correct
engines would pick different printed dates legitimately — and since the benefit period is checked
on this value and on no other, they would disagree about the verdict while every field either one
extracted looked correct. A caption drifting into policy.yaml is the same defect from the other
side: the policy would then carry a fact about one bank's layout, and a consumer would have two
places to read the answer from, which is one place too many.

What this file does not claim. It checks that the two halves are on the right sides and that the
refused captions each carry a reason of their own. It cannot check that the order of preference is
correct — that is a judgement about the meaning of the terms, argued in the contract itself, and
the only thing a test can do about it is make a silent change to it impossible.

The one test that reads the contract. No code in `src/` reads config/labelling-schema.yaml, and
adding a loader there would be inventing a caller. This file reads it because the property under
test lives in neither file alone; it is not a loader and no pipeline stage calls it. The head of
the contract says so too.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# The six date concepts a real Ukrainian bank confirmation prints, as the contract names them.
PRIMARY_CAPTIONS = frozenset(
    {
        "Дата виконання",
        "Дата проведення",
        "Дата операції",
        "Дата валютування",
        "Дата складання",
        "Дата обробки",
    }
)


def contract_payment_date() -> dict:
    """The mapping block of the labelling contract, resolved by path."""
    contract = yaml.safe_load((CONFIG_DIR / "labelling-schema.yaml").read_text(encoding="utf-8"))
    return contract["document_types"]["payment_confirmation"]["payment_date"]


def declared_captions() -> list[str]:
    """Every printed caption the contract names, accepted and refused, with its variants."""
    block = contract_payment_date()
    captions: list[str] = []
    for entry in block["accept_in_order"] + block["never"]:
        captions.append(entry["caption"])
        captions.extend(entry["also_printed_as"])
    return captions


# --------------------------------------------------------------- the policy side --


def test_the_policy_names_the_payment_date_concept():
    """`period.payment_date` exists and says whose account the funds left.

    Both halves are load-bearing. Without the key, "the payment date" is undefined and two
    engines may differ. With the key naming the wrong end of the transfer — the date the
    recipient acquires the funds, which is a real printed caption and a different day — the
    definition is present and wrong, which is worse than absent.
    """
    policy = yaml.safe_load((CONFIG_DIR / "policy.yaml").read_text(encoding="utf-8"))
    concept = policy["period"]["payment_date"]

    assert isinstance(concept, str) and concept.strip()
    assert "payer's account" in concept.lower(), (
        "config/policy.yaml must name the account the funds LEFT. A definition that names the "
        f"recipient's end instead is a different day: {concept!r}"
    )


def test_the_policy_names_no_printed_caption():
    """No caption from the contract's mapping appears anywhere in policy.yaml.

    The captions are derived from the contract rather than listed here, so the sweep covers
    whatever the mapping currently names and cannot fall behind it. Its denominator is asserted
    first: were the mapping block emptied, an absence check over nothing would pass.
    """
    captions = declared_captions()

    assert PRIMARY_CAPTIONS.issubset(set(captions)), (
        "the contract no longer names all six printed date concepts, so the sweep below would "
        f"be checking policy.yaml against a shorter list: {sorted(set(captions))}"
    )

    policy_text = (CONFIG_DIR / "policy.yaml").read_text(encoding="utf-8")
    leaked = sorted({caption for caption in captions if caption in policy_text})

    assert not leaked, (
        f"{len(leaked)} of {len(captions)} captions checked appear in config/policy.yaml. A "
        "caption is a property of one bank's layout: it belongs to the labelling contract, and "
        f"a second home for it is a second answer — {leaked}"
    )


# ------------------------------------------------------------- the contract side --


def test_the_contract_gives_the_caption_preference_order():
    """The three accepted captions, in the decided order, exactly.

    Order is the whole content of the decision: all three are read as the concept, and which one
    wins when a document prints two of them is what stops two implementations differing. A
    reordering changes the extracted date on any document carrying more than one caption, and
    5 of the 8 documents reviewed carry more than one.
    """
    accepted = [entry["caption"] for entry in contract_payment_date()["accept_in_order"]]

    assert accepted == ["Дата виконання", "Дата проведення", "Дата операції"]


def test_each_refused_caption_states_its_own_reason():
    """The three refusals, each carrying a reason that does not lean on another caption.

    A refusal written as "not the one above" is a defect even while it is accurate: editing
    either entry then moves the other, and nothing local to the edit shows it. So each `why`
    must name the mechanism that disqualifies its own caption — an accounting convention, the
    date of the obligation, an internal bank event — and must not identify itself by naming a
    sibling.
    """
    refused = contract_payment_date()["never"]

    assert [entry["caption"] for entry in refused] == [
        "Дата валютування",
        "Дата складання",
        "Дата обробки",
    ]

    others = {entry["caption"] for entry in refused} | PRIMARY_CAPTIONS
    for entry in refused:
        reason = entry["why"]
        assert reason.strip(), f"{entry['caption']} is refused without a reason"

        borrowed = sorted(
            caption for caption in others - {entry["caption"]} if caption in reason
        )
        assert not borrowed, (
            f"the refusal of {entry['caption']} works by naming {borrowed}. Each caption must "
            "be disqualified by its own meaning, or an edit to one silently moves the other."
        )
