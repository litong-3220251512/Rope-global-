"""Direct, non-hashed data provenance shared by runner and report boundaries."""

from collections.abc import Mapping
from pathlib import Path


AUDIT_ARTIFACT_NAMES = (
    "selected_questions.jsonl", "vocabulary.json", "audit_report.json",
    "image_partitions.json",
)


def validate_audit_snapshot(value: object) -> dict[str, str]:
    if (not isinstance(value, Mapping) or set(value) != set(AUDIT_ARTIFACT_NAMES)
            or any(type(text) is not str or not text for text in value.values())):
        raise ValueError("paired data requires all four original audit texts")
    return {name: value[name] for name in AUDIT_ARTIFACT_NAMES}


def run_data_contract(
    config: object, *, method: str, seed: int, phase: str, target_updates: int,
) -> tuple[dict[str, str], str | None]:
    if not isinstance(config, Mapping):
        raise ValueError("paired data run_config must be a mapping")
    expected = {"protocol_version": "PUB-R1.1", "method": method, "seed": seed,
                "phase": phase, "target_updates": target_updates}
    for name, value in expected.items():
        if type(config.get(name)) is not type(value) or config[name] != value:
            raise ValueError(f"paired data run_config {name} disagrees")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("paired data run_config lacks data provenance")
    snapshot = validate_audit_snapshot(data.get("audit_artifacts"))
    root = data.get("data_root")
    if method == "Q-only":
        if root is not None:
            raise ValueError("Q-only paired data must not have an RGB root")
    elif type(root) is not str or not root or not Path(root).is_absolute():
        raise ValueError("paired data requires an explicit absolute RGB root")
    return snapshot, root
