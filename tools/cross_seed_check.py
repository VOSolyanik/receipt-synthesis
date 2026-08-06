#!/usr/bin/env python3
"""Fail the build if a corpus is anything other than a function of its seed.

The CLI says the seed "determines the whole run". That sentence was false once already:
the persona date of birth was drawn from `datetime.now()`, so a corpus was a function of
the seed AND the calendar day, and nothing in the test suite could notice — every test ran
on a single day, where the two are indistinguishable.

This check makes them distinguishable. It generates the same seed twice under two time
zones that are 26 hours apart, so their local dates ALWAYS differ no matter what hour the
check runs at, and then compares the corpora byte for byte:

    Etc/GMT+12  is UTC-12   (the sign in these zone names is inverted, per POSIX)
    Etc/GMT-14  is UTC+14

Two properties are asserted, and both are needed. Equality alone would also be satisfied
by a generator that ignores its seed and emits a constant, so a third run under a
neighbouring seed has to come out different.

    same seed, different calendar day  ->  identical corpus
    neighbouring seed                  ->  different corpus

Each run is a separate process, deliberately. `TZ` is read by the C library when the
process starts, so setting it in-process would not move the calendar day; and a fresh
process also rules out module-level state surviving between runs.

The comparison covers the manifest, the labels and the rendered images. Rendering is
already asserted to be byte-identical for identical input (tests/test_renderer.py), so an
image that moves here is a real finding, not noise.

    uv run python tools/cross_seed_check.py

Exits 0 when the corpus is a function of the seed, 1 when it is not.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 26 hours apart, so the local date differs at every hour of the day.
EARLIER_ZONE = "Etc/GMT+12"
LATER_ZONE = "Etc/GMT-14"

#: Small on purpose. The check is about whether the corpus moves, not about how big it is;
#: two personas already exercise personas, claims, documents, labels and the renderer.
DEFAULT_PERSONAS = 2
DEFAULT_CLAIMS_PER_PERSONA = 2
DEFAULT_SEED = 424242


@dataclass(frozen=True)
class Corpus:
    """One generated corpus, reduced to the digests that are compared."""

    label: str
    manifest_and_labels: str
    images: str

    def __str__(self) -> str:
        return f"{self.label}: labels {self.manifest_and_labels[:16]} images {self.images[:16]}"


def _digest_tree(root: Path) -> str:
    """Hash every file under `root`, path included, in a stable order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(out_dir: Path, *, seed: int, timezone: str, personas: int, claims: int) -> None:
    """Run the generator in a fresh process under `timezone`."""
    environment = dict(os.environ, TZ=timezone)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "receipt_synth.cli",
            "--seed",
            str(seed),
            "--split",
            "0.5",
            "--personas",
            str(personas),
            "--claims-per-persona",
            str(claims),
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )


def build(
    out_dir: Path, label: str, *, seed: int, timezone: str, personas: int, claims: int
) -> Corpus:
    generate(out_dir, seed=seed, timezone=timezone, personas=personas, claims=claims)
    manifest = hashlib.sha256((out_dir / "ground_truth.json").read_bytes())
    manifest.update(_digest_tree(out_dir / "labels").encode())
    return Corpus(label, manifest.hexdigest(), _digest_tree(out_dir / "images"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_seed_check",
        description="Assert that a corpus is a function of its seed and of nothing else.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed under test")
    parser.add_argument("--personas", type=int, default=DEFAULT_PERSONAS)
    parser.add_argument("--claims-per-persona", type=int, default=DEFAULT_CLAIMS_PER_PERSONA)
    args = parser.parse_args(argv)

    shared = {"personas": args.personas, "claims": args.claims_per_persona}

    with tempfile.TemporaryDirectory(prefix="cross-seed-") as tmp:
        work = Path(tmp)
        early = build(
            work / "early", f"seed {args.seed} @ {EARLIER_ZONE}",
            seed=args.seed, timezone=EARLIER_ZONE, **shared,
        )
        late = build(
            work / "late", f"seed {args.seed} @ {LATER_ZONE}",
            seed=args.seed, timezone=LATER_ZONE, **shared,
        )
        neighbour = build(
            work / "neighbour", f"seed {args.seed + 1} @ {EARLIER_ZONE}",
            seed=args.seed + 1, timezone=EARLIER_ZONE, **shared,
        )

    for corpus in (early, late, neighbour):
        print(f"cross-seed: {corpus}")

    failures: list[str] = []
    if (early.manifest_and_labels, early.images) != (late.manifest_and_labels, late.images):
        failures.append(
            f"the same seed built different corpora on different calendar days "
            f"({EARLIER_ZONE} vs {LATER_ZONE}) — something reads the clock rather than the seed"
        )
    if (early.manifest_and_labels, early.images) == (
        neighbour.manifest_and_labels,
        neighbour.images,
    ):
        failures.append(
            f"seeds {args.seed} and {args.seed + 1} built the same corpus — "
            f"the seed is not reaching the generator"
        )

    if failures:
        for failure in failures:
            print(f"cross-seed: FAIL — {failure}", file=sys.stderr)
        return 1

    print("cross-seed: the corpus is a function of the seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
