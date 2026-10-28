"""Explicit CLI for user-authorized CLEVR PUB-R1.1 operations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from src.clevr_runner import (
    CLEVRRunnerError,
    _absolute_directory,
    _explicit_output_root,
    _normalize_methods,
    _setup_clevr_runtime,
    _validate_request,
    evaluate_clevr_checkpoint,
    preflight_clevr_run,
    train_clevr_seed,
)


class _CLIError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIError(message)


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", required=True, choices=("smoke", "pilot", "formal"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True, action="append")
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device-index", required=True, type=int)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description=(
            "Inspect or start an explicit CLEVR PUB-R1.1 operation. Invoking "
            "train/evaluate starts user-authorized GPU work; base-test evaluation "
            "is separate from training."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Validate configuration and train/dev dataset metadata only."
    )
    _add_request_arguments(inspect)
    train = commands.add_parser(
        "train", help="Start one explicitly authorized GPU training invocation."
    )
    _add_request_arguments(train)
    train.add_argument("--output-root", required=True, type=Path)
    train.add_argument("--resume", action="append", default=[], metavar="METHOD=PATH")
    evaluate = commands.add_parser(
        "evaluate", help="Start one separately authorized formal base-test evaluation."
    )
    _add_request_arguments(evaluate)
    evaluate.add_argument("--output-root", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    return parser


def _request(args: argparse.Namespace) -> tuple[str, ...]:
    methods = _normalize_methods(tuple(args.method))
    _validate_request(
        spec_phase=args.phase, seed=args.seed, methods=methods,
        audit_dir=args.audit_dir, data_root=args.data_root,
        device_index=args.device_index,
    )
    return methods


def _resume_mapping(values: Sequence[str], methods: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if type(value) is not str or "=" not in value:
            raise _CLIError("--resume requires METHOD=PATH")
        method, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if method not in methods or method in result:
            raise _CLIError("--resume method must be selected and unique")
        if not raw_path or not path.is_absolute() or not path.is_file():
            raise _CLIError("--resume path must be an explicit existing absolute file")
        result[method] = path
    return result


def _emit(stream: object, document: dict[str, object]) -> None:
    print(json.dumps(document, allow_nan=False, sort_keys=True), file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        methods = _request(args)
        if args.command == "inspect":
            result = preflight_clevr_run(
                spec_phase=args.phase, seed=args.seed, methods=methods,
                audit_dir=args.audit_dir, data_root=args.data_root,
                device_index=args.device_index,
            )
            _emit(sys.stdout, {
                "command": "inspect", "methods": list(result.methods),
                "phase": args.phase, "seed": args.seed, "status": "ok",
                "vocab_size": result.vocab_size,
            })
            return 0
        if args.command == "train":
            _explicit_output_root(args.output_root)
            resumes = _resume_mapping(args.resume, methods)
            train_clevr_seed(
                spec_phase=args.phase, seed=args.seed, methods=methods,
                audit_dir=args.audit_dir, data_root=args.data_root,
                output_root=args.output_root, device_index=args.device_index,
                resume_checkpoints=resumes,
            )
            _emit(sys.stdout, {"command": "train", "status": "complete"})
            return 0
        if args.command != "evaluate" or len(methods) != 1:
            raise _CLIError("evaluate requires exactly one --method")
        _absolute_directory(args.output_root, "output_root")
        if args.phase != "formal":
            raise _CLIError("evaluate requires --phase formal")
        if (not args.checkpoint.is_absolute() or not args.checkpoint.is_file()):
            raise _CLIError("--checkpoint must be an explicit existing absolute file")
        path = evaluate_clevr_checkpoint(
            spec_phase=args.phase, seed=args.seed, method=methods[0],
            audit_dir=args.audit_dir, data_root=args.data_root,
            output_root=args.output_root, device_index=args.device_index,
            checkpoint=args.checkpoint, partition="base_test",
        )
        _emit(sys.stdout, {
            "command": "evaluate", "predictions": str(path), "status": "complete",
        })
        return 0
    except (_CLIError, CLEVRRunnerError, ValueError) as exc:
        _emit(sys.stderr, {"error": str(exc), "type": type(exc).__name__})
        return 2
    except Exception as exc:
        _emit(sys.stderr, {"error": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
