"""Paired, from-scratch CIFAR-100 RoPE experiment with a held-out test set."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import timm
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from timm.data import Mixup

from src.cifar_data import CIFARDataConfig, build_datasets, build_test_dataset
from src.classification import RoPEClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def digest_tensors(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def learning_rate(step, steps, warmup_steps, peak, minimum=5e-6, warmup_start=5e-7):
    if step < warmup_steps:
        return warmup_start + (peak - warmup_start) * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(steps - warmup_steps - 1, 1)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def optimizer_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("cls_token"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_loader(dataset, args, training=False, epoch_seed=None):
    generator = torch.Generator().manual_seed(epoch_seed or args.split_seed)
    sampler = RandomSampler(dataset, generator=generator) if training else None
    return DataLoader(
        dataset, batch_size=args.batch_size if training else args.eval_batch_size,
        sampler=sampler, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=training, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed((epoch_seed or args.split_seed) + 1),
        persistent_workers=False,
    )


@torch.no_grad()
def evaluate(model, loader, device, lam=None, predictions=False):
    model.eval()
    correct1 = correct5 = count = 0
    loss_sum = 0.0
    saved_predictions, saved_targets = [], []
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images, lam=lam)
        logits = logits.float()
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        top = logits.topk(5, dim=1).indices
        correct1 += int(top[:, 0].eq(targets).sum())
        correct5 += int(top.eq(targets[:, None]).any(dim=1).sum())
        count += len(targets)
        if predictions:
            saved_predictions.append(top[:, 0].cpu().numpy())
            saved_targets.append(targets.cpu().numpy())
    result = {"top1": correct1 / count, "top5": correct5 / count,
              "cross_entropy": loss_sum / count, "count": count}
    if predictions:
        return result, np.concatenate(saved_predictions), np.concatenate(saved_targets)
    return result


def model_kwargs(args, mode):
    return dict(mode=mode, fixed_lambda=args.fixed_lambda, image_size=args.image_size,
                patch_size=args.patch_size, embed_dim=args.embed_dim, depth=12,
                heads=args.heads, num_classes=100, init_scale=1e-4,
                drop_path_rate=args.drop_path, attention_backend=args.attention_backend)


def train_branch(args, bundle, mode, seed, common_state, output, device, events):
    name = f"{mode}_seed{seed}"
    model = RoPEClassifier(**model_kwargs(args, mode))
    model.load_shared_state_dict(common_state)
    if digest_tensors(model.shared_state_dict()) != digest_tensors(common_state):
        raise AssertionError("Non-PE initialization mismatch")
    if list(model.position.parameters()):
        raise AssertionError("This experiment requires fixed PE, with no trainable lambda")
    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8, fused=True)
    mixup = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                  prob=args.mixup_prob, switch_prob=0.5, mode="batch",
                  label_smoothing=args.smoothing, num_classes=100)
    steps_per_epoch = len(bundle.train) // args.batch_size
    if steps_per_epoch < 1:
        raise ValueError("Training split must contain at least one full batch")
    history = []
    start_epoch = 0
    last_path = output / f"{name}_last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
    training_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    initial = None
    if not start_epoch:
        initial = evaluate(model, make_loader(bundle.dev, args), device)
        events({"event": "branch_initial", "mode": mode, "seed": seed, "dev": initial})
    for epoch in range(start_epoch, args.epochs):
        # Explicit epoch seeds give paired data, mixing, dropout and resumability.
        epoch_seed = seed + 10007 * (epoch + 1)
        seed_everything(epoch_seed)
        loader = make_loader(bundle.train, args, training=True, epoch_seed=epoch_seed)
        model.train()
        loss_sum, count, gradient_sum = 0.0, 0, 0.0
        first_batch_sha = mixed_targets_sha = None
        epoch_start = time.monotonic()
        for batch, (images, targets) in enumerate(loader):
            if batch == 0:
                first_batch_sha = digest_tensors({"images": images, "targets": targets})
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images, mixed_targets = mixup(images, targets)
            if batch == 0:
                mixed_targets_sha = digest_tensors({"mixed_targets": mixed_targets})
            current_lr = learning_rate(epoch * steps_per_epoch + batch,
                                       args.epochs * steps_per_epoch,
                                       args.warmup_epochs * steps_per_epoch,
                                       args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = training_model(images)
                loss = -(mixed_targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {name} epoch {epoch + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                                       error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            count += len(images)
            gradient_sum += float(grad_norm)
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - epoch_start
        dev = evaluate(model, make_loader(bundle.dev, args), device)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / count,
                  "train_examples": count, "lr": current_lr,
                  "gradient_norm_mean_before_clip": gradient_sum / steps_per_epoch,
                  "train_seconds": train_seconds, "dev": dev,
                  "first_batch_sha256": first_batch_sha,
                  "mixed_targets_sha256": mixed_targets_sha}
        history.append(record)
        events({"event": "epoch", "mode": mode, "seed": seed, **record})
        write_json(output / f"{name}_history.json", history)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                       "epoch": epoch + 1, "mode": mode, "seed": seed,
                                       "history": history, "model_kwargs": model_kwargs(args, mode)})
    result = {"mode": mode, "seed": seed, "final_dev": history[-1]["dev"],
              "epochs": len(history), "initial_dev": initial,
              "segment_seconds": time.monotonic() - started,
              "training_seconds": sum(row["train_seconds"] for row in history),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "checkpoint": last_path.name}
    write_json(output / f"{name}_result.json", result)
    del training_model, model, optimizer
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["G", "B", "C"], default=["G", "B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261015])
    parser.add_argument("--split-seed", type=int, default=20261015)
    parser.add_argument("--dev-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--fixed-lambda", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--augmentation", choices=["deit", "threeaugment", "basic"], default="deit")
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--attention-backend", choices=["manual", "sdpa"], default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    # Only for pipeline verification, never label as a benchmark result.
    parser.add_argument("--smoke-train-limit", type=int, default=0)
    parser.add_argument("--smoke-dev-limit", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("modes and seeds must be unique")
    if not 0 <= args.warmup_epochs < args.epochs or args.batch_size % 2:
        parser.error("require 0 <= warmup < epochs and an even mixup batch size")
    if args.skip_test is False and (args.smoke_train_limit or args.smoke_dev_limit):
        parser.error("smoke subsets require --skip-test")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one selected CUDA GPU")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    serialized_args = {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items() if key != "resume"}
    sources = [Path(__file__), Path("src/classification.py"), Path("src/local_rope.py"),
               Path("src/segmentation.py"), Path("src/cifar_data.py"),
               Path("vendor/rope-vit/models/vit_rope.py"), Path("vendor/rope-vit/deit/models_v2.py")]
    hashes = {str(path.relative_to(Path.cwd()) if path.is_absolute() else path):
              hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    config = CIFARDataConfig(root=args.data_root, image_size=args.image_size,
                            split_seed=args.split_seed, dev_per_class=args.dev_per_class,
                            augmentation=args.augmentation, color_jitter=args.color_jitter,
                            random_erasing=args.random_erasing)
    bundle = build_datasets(config)
    if args.smoke_train_limit or args.smoke_dev_limit:
        from torch.utils.data import Subset
        if args.smoke_train_limit:
            bundle.train = Subset(bundle.train, range(args.smoke_train_limit))
        if args.smoke_dev_limit:
            bundle.dev = Subset(bundle.dev, range(args.smoke_dev_limit))
    manifest = {"args": serialized_args, "source_sha256": hashes,
                "dataset": bundle.metadata, "train_indices": list(map(int, bundle.train_indices)),
                "dev_indices": list(map(int, bundle.dev_indices)), "pretrained": False,
                "effective_train_images": len(bundle.train), "effective_dev_images": len(bundle.dev),
                "engineering_smoke_only": bool(args.smoke_train_limit or args.smoke_dev_limit),
                "selection_rule": "fixed final epoch; no test-based selection",
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "timm": str(timm.__version__), "python": sys.version,
                "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(), "primary_metric": "test top1 accuracy",
                "determinism": "paired explicit epoch seeds; bitwise CUDA determinism is not enforced"}
    if args.resume:
        existing = json.loads((output / "manifest.json").read_text())
        if existing["args"] != serialized_args or existing["source_sha256"] != hashes:
            raise ValueError("Resume requires identical arguments and training sources")
    else:
        write_json(output / "manifest.json", manifest)
        for source in sources:
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    with (output / "events.jsonl").open("a", buffering=1) as event_file:
        def events(event):
            message = json.dumps({"time": time.time(), **event}, ensure_ascii=False)
            print(message, flush=True)
            event_file.write(message + "\n")

        events({"event": "start", "args": serialized_args})
        results = {}
        for seed in args.seeds:
            seed_everything(seed)
            common_model = RoPEClassifier(**model_kwargs(args, "G"))
            common_state = common_model.shared_state_dict()
            initial_path = output / f"shared_initial_seed{seed}.pt"
            if initial_path.exists():
                stored = torch.load(initial_path, map_location="cpu", weights_only=True)
                if digest_tensors(stored) != digest_tensors(common_state):
                    raise AssertionError("Resumed initialization mismatch")
            else:
                save_checkpoint(initial_path, common_state)
            events({"event": "initialization", "seed": seed,
                    "sha256": digest_tensors(common_state),
                    "parameters": sum(p.numel() for p in common_model.parameters())})
            del common_model
            for mode in args.modes:
                name = f"{mode}_seed{seed}"
                results[name] = train_branch(args, bundle, mode, seed, common_state,
                                              output, device, events)
            reference = json.loads((output / f"{args.modes[0]}_seed{seed}_history.json").read_text())
            for mode in args.modes[1:]:
                candidate = json.loads((output / f"{mode}_seed{seed}_history.json").read_text())
                for a, b in zip(reference, candidate, strict=True):
                    for field in ("first_batch_sha256", "mixed_targets_sha256"):
                        if a[field] != b[field]:
                            raise AssertionError(f"Paired augmentation mismatch: {mode}, {field}")
        # Test data is first instantiated only after every training branch completes.
        if not args.skip_test:
            test = build_test_dataset(config)
            test_loader = make_loader(test, args)
            for name, result in results.items():
                checkpoint = torch.load(output / result["checkpoint"], map_location="cpu", weights_only=True)
                model = RoPEClassifier(**checkpoint["model_kwargs"])
                model.load_state_dict(checkpoint["model"], strict=True)
                model.to(device)
                metric, predictions, targets = evaluate(model, test_loader, device, predictions=True)
                result["test"] = metric
                np.savez_compressed(output / f"{name}_test_predictions.npz", predictions=predictions, targets=targets)
                events({"event": "test_final", "name": name, **metric})
                if result["mode"] != "G":
                    metric_zero, predictions_zero, _ = evaluate(model, test_loader, device,
                                                                lam=0.0, predictions=True)
                    result["test_lambda0_intervention"] = metric_zero
                    np.savez_compressed(output / f"{name}_test_lambda0_predictions.npz",
                                        predictions=predictions_zero, targets=targets)
                del model
                torch.cuda.empty_cache()
        write_json(output / "results.json", results)
        write_json(output / "summary.json", {"status": "complete", "from_scratch": True,
                                              "paired_first_batch_hashes_match": True,
                                              "results": results})
        events({"event": "complete", "results": results})


if __name__ == "__main__":
    main()
