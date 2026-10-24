"""Explicit-path CLI for canonical CLEVR PUB-R1.1 statistics reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.clevr_reporting import (
    PredictionSource,
    build_statistics_bundle,
    publish_statistics_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute and publish one CLEVR PUB-R1.1 statistics bundle.",
        allow_abbrev=False,
    )
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument(
        "--prediction-file",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bundle = build_statistics_bundle(
        args.audit_dir,
        tuple(PredictionSource(path) for path in args.prediction_file),
    )
    publish_statistics_bundle(args.output_dir, bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
