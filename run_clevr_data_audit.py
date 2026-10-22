"""Build reviewable manifests from an explicit local CLEVR v1.0 root."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from src.clevr_audit import (
    AuditResult,
    CLEVRDataError,
    SelectedQuestion,
    build_audit,
    load_source_index,
)


_PROTOCOL_VERSION = "PUB-R1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit user-provided CLEVR train/val files without training."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Explicit local CLEVR_v1.0 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for the four audit artifacts.",
    )
    return parser


def _validate_output_target(output_dir: Path) -> Path:
    output = output_dir.expanduser().absolute()
    if output.is_symlink():
        raise CLEVRDataError(f"output directory must not be a symlink: {output}")
    output = output.resolve()
    if output.exists():
        if not output.is_dir():
            raise CLEVRDataError(f"output path is not a directory: {output}")
        try:
            nonempty = next(output.iterdir(), None) is not None
        except OSError as exc:
            raise CLEVRDataError(f"cannot inspect output directory {output}: {exc}") from exc
        if nonempty:
            raise CLEVRDataError(f"refusing to overwrite nonempty output directory: {output}")
    return output


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


def _selected_payload(
    selected: SelectedQuestion, data_root: Path
) -> dict[str, object]:
    try:
        relative_image_path = selected.image_path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise CLEVRDataError(
            f"selected image path is outside the explicit data root: {selected.image_path}"
        ) from exc
    return {
        "partition": selected.partition,
        "official_split": selected.official_split,
        "image_index": selected.image_index,
        "image_filename": selected.image_filename,
        "image_path": relative_image_path,
        "question_index": selected.question_index,
        "source_position": selected.source_position,
        "question": selected.question,
        "encoded_tokens": list(selected.encoded_tokens),
        "answer": selected.answer,
        "relation": selected.relation,
    }


def _write_artifacts(directory: Path, audit: AuditResult, data_root: Path) -> None:
    _write_json(
        directory / "image_partitions.json",
        {
            "protocol_version": _PROTOCOL_VERSION,
            "partitions": {
                partition: [asdict(image) for image in getattr(audit.image_partitions, partition)]
                for partition in ("train", "dev", "base_test", "confirmation")
            },
        },
    )
    _write_json(
        directory / "vocabulary.json",
        {
            "protocol_version": _PROTOCOL_VERSION,
            "token_to_id": dict(audit.vocabulary.token_to_id),
            "id_to_token": list(audit.vocabulary.id_to_token),
        },
    )
    with (directory / "selected_questions.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for selected in audit.selected_questions:
            json.dump(
                _selected_payload(selected, data_root),
                output_file,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            output_file.write("\n")

    cell_counts = []
    for cell in audit.cell_counts:
        payload = asdict(cell)
        payload["accuracy"] = "NA" if cell.accuracy is None else cell.accuracy
        cell_counts.append(payload)
    _write_json(
        directory / "audit_report.json",
        {
            "protocol_version": _PROTOCOL_VERSION,
            "environment": {
                "python": platform.python_version(),
                "pillow": version("Pillow"),
                "numpy": version("numpy"),
            },
            "partition_summaries": [
                asdict(summary) for summary in audit.partition_summaries
            ],
            "cell_counts": cell_counts,
            "exclusion_counts": [
                asdict(exclusion) for exclusion in audit.exclusion_counts
            ],
            "source_summaries": [
                asdict(summary) for summary in audit.source_summaries
            ],
            "file_summaries": [
                asdict(summary) for summary in audit.file_summaries
            ],
        },
    )


def _publish_atomically(output: Path, audit: AuditResult, data_root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        _write_artifacts(temporary, audit, data_root)
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _validate_output_target(args.output_dir)
    source = load_source_index(args.data_root)
    audit = build_audit(source)
    _publish_atomically(output, audit, source.data_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CLEVRDataError as exc:
        raise SystemExit(f"CLEVR data audit failed: {exc}") from exc
