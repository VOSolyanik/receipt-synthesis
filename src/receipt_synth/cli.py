"""Command line entry point.

    generate-dataset --seed 20260803 --split 0.5 --out out

The seed is required rather than defaulted. A dataset is only reproducible if the number
that produced it is something the caller chose and can write down; a default would let a
run look reproducible without anyone having recorded what to reproduce it with.

`--split` is required for the same reason, and it is the same reason rather than a similar
one. The split fraction is a decision about the measurement — which documents a figure may
be quoted on — so a default would let a run be performed without that decision ever having
been declared, and the resulting partition would carry the authority of something chosen.
The argument for a half is guidance on what to pass, not the reason for a default; it lives
in README.md's flag table and in docs/architecture.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_synth import __version__
from receipt_synth.assembler import (
    balance_report,
    generate_dataset,
)
from receipt_synth.schemas import Country


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-dataset",
        description=(
            "Generate synthetic benefit-reimbursement documents whose ground truth is "
            "known by construction. Output is not valid proof of payment."
        ),
    )
    parser.add_argument("--seed", type=int, required=True, help="seed; determines the whole run")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("--personas", type=int, default=1, help="how many personas to generate")
    parser.add_argument(
        "--claims-per-persona",
        type=int,
        default=1,
        help=(
            "upper bound on claims per persona; planning stops early once a persona has "
            "no benefit category with an annual balance left"
        ),
    )
    parser.add_argument(
        "--split",
        type=float,
        required=True,
        metavar="TRAIN_FRACTION",
        help=(
            "fraction of PERSONAS assigned to train, the rest to validation; required, because "
            "the partition decides which documents a figure may be quoted on and a run must not "
            "be performed without that having been declared. Pass 0.5 unless your consumer really "
            "does train: nothing is trained on this dataset, and the measurement side has to be "
            "big enough to carry a per-class figure for the thinnest class. The partition is by "
            "persona because annual limits are cumulative per persona, and it is not stratified"
        ),
    )
    parser.add_argument(
        "--country",
        type=Country,
        choices=list(Country),
        default=Country.UA,
        help="jurisdiction whose fiscal rules apply",
    )
    parser.add_argument("--version", action="version", version=f"receipt-synth {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset = generate_dataset(
        seed=args.seed,
        out_dir=args.out,
        personas=args.personas,
        claims_per_persona=args.claims_per_persona,
        country=args.country,
        train_fraction=args.split,
    )

    print(
        f"receipt-synth {__version__} — seed {dataset.seed}\n"
        # "of N ordered" here as well as in the report below: this line is the one that
        # gets copied into a summary, and a bare claim count reads as the number asked for.
        f"  {len(dataset.personas)} persona(s), {len(dataset.claims)} of "
        f"{dataset.claims_ordered} claim(s), {len(dataset.documents)} document(s)\n"
        f"  images: {args.out / 'images'}\n"
        f"  labels: {args.out / 'labels'}\n"
        f"  manifest: {args.out / 'ground_truth.json'}\n"
    )
    print(balance_report(dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
