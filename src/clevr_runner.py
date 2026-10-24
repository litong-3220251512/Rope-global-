"""Dependency-injected, artifact-free PUB-R1.1 computation engine.

A failed update is not rollback-safe. Recover from an explicit completed
checkpoint; only a returned result denotes a successfully committed update.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.clevr_dataset import CLEVRSampleIdentity
from src.clevr_run_artifacts import CLEVRPredictionRecord
from src.clevr_sampling import CLEVRIndexStream


class CLEVRRunnerError(RuntimeError):
    """The injected computation violates the frozen CLEVR contract."""


@dataclass(frozen=True, slots=True)
class CLEVRUpdateResult:
    dataset_indices: tuple[int, ...]
    question_indices: tuple[int, ...]
    loss_sum: float
    loss_mean: float
    gradient_norm: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class CLEVREvaluationResult:
    partition: str
    total_questions: int
    correct: int
    loss_sum: float
    cross_entropy: float
    accuracy: float
    rows: tuple[CLEVRPredictionRecord, ...]


def _require_method(method: str) -> None:
    if type(method) is not str or method not in ("G", "B", "M10", "Q-only"):
        raise CLEVRRunnerError("method must be G, B, M10, or Q-only")


def _question_index(dataset: object, index: int, partition: str) -> int:
    identity = dataset.sample_identity(index)
    if (not isinstance(identity, CLEVRSampleIdentity)
            or identity.partition != partition
            or type(identity.question_index) is not int or identity.question_index < 0):
        raise CLEVRRunnerError("sample identity has an invalid partition or question_index")
    return identity.question_index


def _model_batch(
    batch: Mapping[str, Tensor], *, size: int, method: str, device: torch.device,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    fields = {"question_tokens", "question_length", "answer"}
    if method != "Q-only":
        fields.add("image")
    if not isinstance(batch, Mapping) or set(batch) != fields:
        raise CLEVRRunnerError("batch keys disagree with the method's model contract")
    if any(not isinstance(value, Tensor) for value in batch.values()):
        raise CLEVRRunnerError("batch values must be tensors")
    tokens, lengths, labels = (batch[key] for key in ("question_tokens", "question_length", "answer"))
    if (tokens.dtype != torch.int64 or tokens.ndim != 2
            or tokens.shape[0] != size or tokens.shape[1] < 2):
        raise CLEVRRunnerError("question_tokens must be int64 [N,T], T >= 2")
    if (lengths.dtype != torch.int64 or lengths.shape != (size,)
            or bool((lengths < 2).any()) or bool((lengths > tokens.shape[1]).any())):
        raise CLEVRRunnerError("question_length must be int64 [N] within token bounds")
    if (labels.dtype != torch.int64 or labels.shape != (size,)
            or bool(((labels != 0) & (labels != 1)).any())):
        raise CLEVRRunnerError("answer must be int64 [N] with binary labels")
    image = None
    if method != "Q-only":
        image = batch["image"]
        if (image.dtype != torch.float32 or image.shape != (size, 3, 224, 224)
                or not bool(torch.isfinite(image).all())):
            raise CLEVRRunnerError("image must be finite float32 [N,3,224,224]")
        image = image.to(device)
    return image, tokens.to(device), lengths.to(device), labels.to(device)


def _logits(model: nn.Module, image: Tensor | None, tokens: Tensor, lengths: Tensor) -> Tensor:
    logits = model(image, tokens, lengths)
    if (not isinstance(logits, Tensor) or logits.dtype != torch.float32
            or logits.shape != (len(tokens), 2) or not bool(torch.isfinite(logits).all())):
        raise CLEVRRunnerError("logits must be finite float32 [N,2]")
    return logits


def _finite_scalar(value: Tensor, name: str) -> float:
    result = float(value.detach().item())
    if not math.isfinite(result):
        raise CLEVRRunnerError(f"{name} must be finite")
    return result


def run_clevr_update(
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    dataset: object,
    collate_fn: Callable[[Sequence[Mapping[str, Tensor]]], Mapping[str, Tensor]],
    method: str,
    device: torch.device,
) -> CLEVRUpdateResult:
    """Accumulate eight contiguous micro-batches, then commit exactly 128 indices.

    The caller owns model mode and device placement. No checkpoint, journal,
    retry, or rollback is attempted, even if a dependency raises after mutation.
    """
    optimizer.zero_grad(set_to_none=True)
    _require_method(method)
    if len(optimizer.param_groups) != 2:
        raise CLEVRRunnerError("optimizer must expose exactly two parameter groups")
    rates = [group["lr"] for group in optimizer.param_groups]
    if (any(type(rate) not in (int, float) or not math.isfinite(rate) for rate in rates)
            or rates[0] != rates[1]):
        raise CLEVRRunnerError("optimizer groups must share one finite learning rate")
    learning_rate = float(rates[0])
    if len(dataset) != sampler.dataset_size:
        raise CLEVRRunnerError("dataset size disagrees with sampler")
    indices = sampler.preview(128)
    if (type(indices) is not tuple or len(indices) != 128
            or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices)):
        raise CLEVRRunnerError("sampler preview must contain 128 valid dataset indices")
    question_indices = tuple(_question_index(dataset, index, "train") for index in indices)
    loss_sum = 0.0
    for offset in range(0, 128, 16):
        samples = [dataset[index] for index in indices[offset:offset + 16]]
        image, tokens, lengths, labels = _model_batch(
            collate_fn(samples), size=16, method=method, device=device,
        )
        logits = _logits(model, image, tokens, lengths)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        loss_sum += _finite_scalar(loss, "loss") * 16
        (loss / 8).backward()
    if not math.isfinite(loss_sum):
        raise CLEVRRunnerError("accumulated loss must be finite")
    try:
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise CLEVRRunnerError("gradient clipping failed") from exc
    gradient_norm = _finite_scalar(norm, "gradient norm")
    # Validate and construct all metadata before the irreversible step sequence.
    result = CLEVRUpdateResult(indices, question_indices, loss_sum, loss_sum / 128,
                               gradient_norm, learning_rate)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    sampler.commit(indices)
    return result


def evaluate_clevr_model(
    *,
    model: nn.Module,
    dataset: object,
    collate_fn: Callable[..., Mapping[str, Tensor]],
    method: str,
    partition: str,
    device: torch.device,
) -> CLEVREvaluationResult:
    """Return complete question-weighted metrics and sorted CPU prediction rows.

    This seam never publishes files or chooses checkpoints. The caller must
    separately authorize and assemble any production evaluation.
    """
    was_training = model.training
    try:
        model.eval()
        _require_method(method)
        if type(partition) is not str or partition not in ("train", "dev", "base_test"):
            raise CLEVRRunnerError("evaluation partition must be train, dev, or base_test")
        size = len(dataset)
        if size == 0:
            raise CLEVRRunnerError("evaluation dataset must not be empty")
        seen: set[int] = set()
        rows: list[CLEVRPredictionRecord] = []
        loss_sum = 0.0
        correct = 0
        with torch.inference_mode():
            for offset in range(0, size, 16):
                indices = range(offset, min(offset + 16, size))
                question_indices = []
                for index in indices:
                    question_index = _question_index(dataset, index, partition)
                    if question_index in seen:
                        raise CLEVRRunnerError("duplicate evaluation question_index")
                    seen.add(question_index)
                    question_indices.append(question_index)
                samples = [dataset[index] for index in indices]
                image, tokens, lengths, labels = _model_batch(
                    collate_fn(samples), size=len(indices), method=method, device=device,
                )
                logits = _logits(model, image, tokens, lengths)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                loss_sum += _finite_scalar(loss, "evaluation loss")
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                cpu_logits = logits.detach().cpu().tolist()
                cpu_labels = labels.detach().cpu().tolist()
                rows.extend(CLEVRPredictionRecord(question_index, label, tuple(values))
                            for question_index, label, values in zip(question_indices, cpu_labels, cpu_logits))
        if not math.isfinite(loss_sum) or len(rows) != size:
            raise CLEVRRunnerError("evaluation metrics or row coverage are incomplete")
        return CLEVREvaluationResult(
            partition, size, correct, loss_sum, loss_sum / size, correct / size,
            tuple(sorted(rows, key=lambda row: row.question_index)),
        )
    finally:
        model.train(was_training)
