"""Command line entry point.

    generate-dataset --seed 20260803 --out out

The seed is required rather than defaulted. A dataset is only reproducible if the number
that produced it is something the caller chose and can write down; a default would let a
run look reproducible without anyone having recorded what to reproduce it with.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_synth import __version__
from receipt_synth.assembler import balance_report, generate_dataset
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
