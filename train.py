from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from siblingrestore.baselines import MODELS as BASELINE_MODELS
from siblingrestore.data import PairDataset, SiblingGroupDataset
from siblingrestore.inference import restore_tiled
from siblingrestore.losses import (
    adaptive_frozen_anchor_loss,
    charbonnier,
    frozen_anchor_loss,
    gentle_identity_loss,
    gradient_loss,
    identity_safe_distillation,
    multi_scale_anchor_loss,
    multi_scale_sibling_consensus,
    reliability_guided_sibling_distillation,
    optional_loss,
    project_conflicting_gradient,
    sibling_output_consistency,
    source_contrastive_loss,
    verifier_sibling_consensus,
)
from siblingrestore.metrics import psnr, ssim
from siblingrestore.model import SiblingRestormer
from siblingrestore.verifier import load_verifier_checkpoint


MODES = ("independent", "group", "sibling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-diagnostic-interval", type=int)
    parser.add_argument("--eval-items", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in (
        ("mode", args.mode),
        ("output_dir", str(args.output_dir) if args.output_dir else None),
        ("data_root", str(args.data_root) if args.data_root else None),
        ("max_steps", args.max_steps),
        ("seed", args.seed),
        ("device", args.device),
        ("crop_size", args.crop_size),
        ("num_workers", args.num_workers),
        ("gradient_diagnostic_interval_steps", args.gradient_diagnostic_interval),
        ("smoke_eval_items", args.eval_items),
    ):
        if value is not None:
            config[key] = value
    if config["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_model(config: dict[str, object]) -> torch.nn.Module:
    family = str(config.get("model_family", ""))
    if family:
        if family not in BASELINE_MODELS:
            raise ValueError(f"unknown model_family: {family}")
        return BASELINE_MODELS[family]()
    model_config = config["model"]
    return SiblingRestormer(
        dim=int(model_config["dim"]),
        blocks_per_level=tuple(model_config["blocks_per_level"]),
        heads=tuple(model_config["heads"]),
        projection_dim=int(model_config["projection_dim"]),
        degradation_conditioned=bool(model_config.get("degradation_conditioned", False)),
        identity_mode=model_config.get("identity_mode", None),
        identity_source_count=int(model_config.get("identity_source_count", 71)),
        refinement_type=str(model_config.get("refinement_type", "none")),
        refinement_gate_max=float(model_config.get("refinement_gate_max", 0.15)),
    )


def make_train_dataset(config: dict[str, object]):
    root = Path(config["data_root"])
    common = {
        "root": root,
        "split": "train",
        "crop_size": int(config["crop_size"]),
        "seed": int(config["seed"]),
    }
    if config["mode"] == "independent":
        return PairDataset(
            **common,
            max_items=int(config.get("smoke_train_items", 0)),
        )
    return SiblingGroupDataset(
        **common,
        sibling_count=int(config["sibling_count"]),
        max_sources=int(config.get("smoke_train_sources", 0)),
    )


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def _latent_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    latent = getattr(model, "latent", None)
    if latent is None:
        return []
    return list(latent[-1].parameters())


def _gradient_stats(
    loss: torch.Tensor | None, parameters: list[torch.nn.Parameter]
) -> tuple[torch.Tensor | None, float | None]:
    if loss is None or not parameters:
        return None, None
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
    )
    flat = [g.detach().float().reshape(-1) for g in gradients if g is not None]
    if not flat:
        return None, 0.0
    vector = torch.cat(flat)
    return vector, float(torch.linalg.vector_norm(vector).cpu())


def _aligned_gradient_vector(
    loss: torch.Tensor | None, parameters: list[torch.nn.Parameter]
) -> torch.Tensor | None:
    """Parameter-aligned concatenated gradient, replacing gaps with zeros."""
    if loss is None or not parameters:
        return None
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
    )
    flat = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            flat.append(torch.zeros_like(parameter).float().reshape(-1))
        else:
            flat.append(gradient.detach().float().reshape(-1))
    return torch.cat(flat)


def _cosine(first: torch.Tensor | None, second: torch.Tensor | None) -> float | None:
    if first is None or second is None:
        return None
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return None
    return float((torch.dot(first, second) / denominator).cpu())


def train_step(
    model: SiblingRestormer,
    batch: dict[str, object],
    mode: str,
    weights: dict[str, float],
    device: torch.device,
    verifier: torch.nn.Module | None = None,
    anchor_config: dict[str, object] | None = None,
    teacher_model: torch.nn.Module | None = None,
    global_step: int = 0,
    warmup_steps: int = 0,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor | None]]:
    source_weight = float(weights["source"])
    output_weight = float(weights["output"])
    degradation_weight = float(weights["degradation"])
    if mode == "independent":
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)
        output = model(degraded)
        restored = output["restored"] if isinstance(output, dict) else output
        rec = charbonnier(restored, clean)
        grad = gradient_loss(restored, clean)
        total = rec + weights["gradient"] * grad
        return total, {
            "reconstruction": float(rec.detach()), "gradient": float(grad.detach()),
            "source_contrastive": 0.0, "output_consistency": 0.0,
            "degradation_classification": 0.0, "identity_classification": 0.0,
            "anchor": 0.0, "weighted_anchor": 0.0,
            "weighted_source": 0.0,
            "weighted_output": 0.0, "weighted_degradation": 0.0,
            "weighted_identity": 0.0,
            "total_loss": float(total.detach()),
        }, {"reconstruction": rec, "gradient_objective": grad, "source": None, "output": None, "degradation": None, "identity": None, "anchor": None}

    degraded = batch["degraded"].to(device)
    clean = batch["clean"].to(device)
    degradation_ids = batch["degradation_ids"].to(device)
    class_ids = batch["class_id"].to(device)
    batch_size, siblings, channels, height, width = degraded.shape
    flat_degraded = degraded.reshape(batch_size * siblings, channels, height, width)
    flat_clean = clean[:, None].expand(-1, siblings, -1, -1, -1).reshape_as(flat_degraded)
    output = model(flat_degraded)
    if not isinstance(output, dict):
        raise ValueError("baseline models must be trained with mode=independent")
    restored = output["restored"]
    rec = charbonnier(restored, flat_clean)
    grad = gradient_loss(restored, flat_clean)
    total = rec + weights["gradient"] * grad
    source = output_consistency = degradation = identity = anchor = None
    safe_identity = None
    safe_identity_active = 0.0
    adaptive_diagnostics: dict[str, float] = {}

    if mode == "sibling" and (source_weight > 0 or output_weight > 0 or degradation_weight > 0):
        content = output["content"].reshape(batch_size, siblings, -1)
        restored_group = restored.reshape(batch_size, siblings, channels, height, width)
        if source_weight > 0:
            clean_anchor = model.encode_content(clean)
            source = optional_loss(source_weight, lambda: source_contrastive_loss(
                content, clean_anchor, class_ids,
                temperature=weights["temperature"],
                same_class_negative_weight=weights["same_class_negative_weight"],
            ))
            total = total + source_weight * source
        if output_weight > 0:
            output_consistency = sibling_output_consistency(restored_group)
            total = total + output_weight * output_consistency
        if degradation_weight > 0:
            degradation = F.cross_entropy(output["degradation_logits"], degradation_ids.reshape(-1))
            total = total + degradation_weight * degradation
    identity_weight = float(weights.get("identity", 0.0))
    anchor_weight = float(weights.get("anchor", 0.0))
    if anchor_weight > 0:
        if mode != "sibling" or verifier is None:
            raise ValueError("anchor loss requires sibling mode and a frozen verifier")
        with torch.autocast(device_type=device.type, enabled=False):
            if anchor_config is not None and anchor_config.get("type") == "multi_scale":
                with torch.no_grad():
                    clean_features = verifier.forward_stages(clean.float())
                # FP32 re-forward of the restoration model: under AMP the
                # restored tensor is FP16 and the small MSA gradients are
                # truncated during backprop through it. An explicit FP32
                # forward keeps the multi-scale anchor gradients intact.
                output_fp32 = model(flat_degraded.float())
                restored_fp32 = output_fp32["restored"] if isinstance(output_fp32, dict) else output_fp32
                restored_features = verifier.forward_stages(restored_fp32)
                clean_features = [feature.repeat_interleave(siblings, dim=0) for feature in clean_features]
                anchor = multi_scale_anchor_loss(
                    restored_features,
                    clean_features,
                    weights=tuple(float(value) for value in anchor_config.get("weights", [0.4, 0.35, 0.25])),
                )
            elif anchor_config is not None and anchor_config.get("type") == "msc":
                # Multi-Scale Sibling Verifier Consensus: same-source siblings
                # share per-channel normalized stage features (pull) and
                # different sources are separated in embedding space with
                # same-class hard negatives weighted harder (push). No clean
                # anchor, so it does not fight pixel reconstruction. FP32
                # re-forward avoids AMP gradient truncation through the small
                # verifier feature gradients.
                output_fp32 = model(flat_degraded.float())
                restored_fp32 = output_fp32["restored"] if isinstance(output_fp32, dict) else output_fp32
                stage_features = verifier.forward_stages(restored_fp32)
                embeddings = verifier.embed(restored_fp32)
                anchor = multi_scale_sibling_consensus(
                    stage_features,
                    embeddings,
                    class_ids,
                    siblings,
                    pull_weights=tuple(float(value) for value in anchor_config.get("pull_weights", [0.4, 0.35, 0.25])),
                    push_margin=float(anchor_config.get("push_margin", 0.3)),
                    hard_negative_weight=float(anchor_config.get("hard_negative_weight", 2.0)),
                )
            elif anchor_config is not None and anchor_config.get("type") == "rsd":
                # Reliability-Guided Sibling Distillation: reliable siblings
                # form a dynamic detached prototype teacher; unreliable
                # siblings are distilled toward it (asymmetric), and
                # prototype-level hard negatives separate same-class sources.
                # FP32 re-forward avoids AMP gradient truncation.
                output_fp32 = model(flat_degraded.float())
                restored_fp32 = output_fp32["restored"] if isinstance(output_fp32, dict) else output_fp32
                restored_embedding = verifier.embed(restored_fp32)
                anchor = reliability_guided_sibling_distillation(
                    restored_embedding,
                    class_ids,
                    siblings,
                    push_margin=float(anchor_config.get("push_margin", 0.3)),
                    hard_negative_weight=float(anchor_config.get("hard_negative_weight", 2.0)),
                )
            elif anchor_config is not None and anchor_config.get("type") == "sibling_consensus":
                # Same-source sibling pull + cross-source push in the frozen
                # verifier embedding space. Does not pull toward the clean
                # anchor, so it avoids the pixel-fidelity conflict of anchor
                # losses while enforcing cross-degradation identity consensus.
                restored_embedding = verifier.embed(restored.float())
                restored_grouped = restored_embedding.reshape(batch_size, siblings, -1)
                anchor = verifier_sibling_consensus(
                    restored_grouped,
                    margin=float(anchor_config.get("margin", 0.3)),
                )
            elif anchor_config is not None and anchor_config.get("type") == "adaptive":
                # Adaptive Anchor: per-sample weight based on source margin.
                # Samples with high source confusion risk (small/negative margin
                # to hard negative) get larger anchor weight; well-separated
                # samples get zero weight.  Weight is detached.
                with torch.no_grad():
                    clean_embedding_unique = verifier.embed(clean.float())  # [B, D]
                restored_embedding = verifier.embed(restored.float())  # [B*K, D]
                source_ids_per_sample = (
                    torch.arange(batch_size, device=device)
                    .repeat_interleave(siblings)
                )  # [B*K]
                anchor, adaptive_diagnostics = adaptive_frozen_anchor_loss(
                    restored_embedding,
                    clean_embedding_unique,
                    source_ids_per_sample,
                    tau=float(anchor_config.get("tau", 0.10)),
                )
            else:
                with torch.no_grad():
                    clean_embedding = verifier.embed(clean.float())
                restored_embedding = verifier.embed(restored.float())
                clean_embedding = clean_embedding.repeat_interleave(siblings, dim=0)
                anchor = frozen_anchor_loss(restored_embedding, clean_embedding)
        total = total + anchor_weight * anchor
    safe_weight = float(weights.get("identity_safe", 0.0))
    if mode == "sibling" and safe_weight > 0 and teacher_model is not None:
        if verifier is None:
            raise ValueError("identity_safe requires a frozen verifier")
        ramp = 1.0 if warmup_steps <= 0 else min(1.0, float(global_step) / float(warmup_steps))
        with torch.autocast(device_type=device.type, enabled=False), torch.no_grad():
            teacher_output = teacher_model(flat_degraded.float())
            teacher_restored = teacher_output["restored"] if isinstance(teacher_output, dict) else teacher_output
            teacher_embedding = verifier.embed(teacher_restored)
            clean_embedding = verifier.embed(clean.float()).repeat_interleave(siblings, dim=0)
        student_embedding = verifier.embed(restored.float())
        safe_identity, active = identity_safe_distillation(
            student_embedding, teacher_embedding, clean_embedding,
            margin=float(weights.get("identity_safe_margin", 0.0)),
        )
        safe_weight *= ramp
        total = total + safe_weight * safe_identity
        safe_identity_active = float(active)
    if mode == "sibling" and identity_weight > 0:
        if model.identity_mode == "classify":
            target = batch["source_ids"].to(device).repeat_interleave(siblings)
            identity = F.cross_entropy(output["identity_logits"], target)
        elif model.identity_mode == "branch":
            target = batch["source_ids"].to(device).repeat_interleave(siblings)
            identity = F.cross_entropy(output["identity_logits"], target)
        elif model.identity_mode == "contrast":
            identity = gentle_identity_loss(
                output["identity_feature"].reshape(batch_size, siblings, -1),
                margin=float(weights.get("identity_margin", 0.3)),
            )
        else:
            raise ValueError(f"identity_weight>0 requires identity_mode, got {model.identity_mode!r}")
        total = total + identity_weight * identity
    values = {
        "reconstruction": float(rec.detach()), "gradient": float(grad.detach()),
        "source_contrastive": float(source.detach()) if source is not None else 0.0,
        "output_consistency": float(output_consistency.detach()) if output_consistency is not None else 0.0,
        "degradation_classification": float(degradation.detach()) if degradation is not None else 0.0,
        "identity_classification": float(identity.detach()) if identity is not None else 0.0,
        "anchor": float(anchor.detach()) if anchor is not None else 0.0,
        "weighted_anchor": float(anchor_weight * anchor.detach()) if anchor is not None else 0.0,
        "weighted_source": float(source_weight * source.detach()) if source is not None else 0.0,
        "weighted_output": float(output_weight * output_consistency.detach()) if output_consistency is not None else 0.0,
        "weighted_degradation": float(degradation_weight * degradation.detach()) if degradation is not None else 0.0,
        "weighted_identity": float(identity_weight * identity.detach()) if identity is not None else 0.0,
        "identity_safe": float(safe_identity.detach()) if safe_identity is not None else 0.0,
        "weighted_identity_safe": float(safe_weight * safe_identity.detach()) if safe_identity is not None else 0.0,
        "identity_safe_active_ratio": safe_identity_active,
        "total_loss": float(total.detach()),
        **{f"adaptive_{k}": v for k, v in adaptive_diagnostics.items()},
    }
    return total, values, {"reconstruction": rec, "gradient_objective": grad, "source": source, "output": output_consistency, "degradation": degradation, "identity": identity, "anchor": anchor, "identity_safe": safe_identity}


@torch.no_grad()
def evaluate(
    model: SiblingRestormer,
    data_root: Path,
    device: torch.device,
    seed: int,
    workers: int,
    split: str = "val",
    max_items: int = 0,
    tile_size: int = 512,
    tile_overlap: int = 32,
) -> dict[str, object]:
    dataset = PairDataset(data_root, split, crop_size=None, seed=seed, max_items=max_items)
    loader = make_loader(dataset, batch_size=1, shuffle=False, workers=workers)
    model.eval()
    aggregate: list[dict[str, float]] = []
    by_degradation: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)
        mask = batch["mask"].to(device)
        restored = restore_tiled(model, degraded, tile_size, tile_overlap)
        row = {
            "psnr": psnr(restored, clean, mask),
            "ssim": ssim(restored, clean, mask),
        }
        aggregate.append(row)
        by_degradation[batch["degradation"][0]].append(row)

    def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("psnr", "ssim")
        }

    return {
        "count": len(aggregate),
        "aggregate": mean_rows(aggregate),
        "by_degradation": {
            degradation: mean_rows(rows)
            for degradation, rows in sorted(by_degradation.items())
        },
        "metric_note": "SSIM is a fast global masked diagnostic; replace with windowed SSIM for paper tables.",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(config).to(device)
    init_from = config.get("init_from")
    if init_from:
        initialization = torch.load(Path(init_from), map_location="cpu", weights_only=False)
        initial_state = initialization["model"]
        if str(config["model"].get("refinement_type", "none")) == "alcrb_safe":
            # Reuse the trained M3 ALCRB weights inside the safe wrapper.
            initial_state = {
                (f"refine_module.alcrb.{key[len('refine_module.'):]}")
                if key.startswith("refine_module.") else key: value
                for key, value in initial_state.items()
            }
        missing, unexpected = model.load_state_dict(initial_state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected initialization parameters: {unexpected}")
        config["initialization_missing_keys"] = list(missing)
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    anchor_weight = float(weights.get("anchor", 0.0))
    verifier = None
    verifier_metadata = None
    teacher_model = None
    pcgrad_enabled = False
    safe_weight = float(weights.get("identity_safe", 0.0))
    if anchor_weight > 0 or safe_weight > 0:
        verifier_config = config.get("frozen_verifier")
        if not isinstance(verifier_config, dict) or not verifier_config.get("checkpoint"):
            raise ValueError("positive anchor weight requires frozen_verifier.checkpoint")
        verifier, verifier_metadata = load_verifier_checkpoint(
            Path(verifier_config["checkpoint"]),
            device,
            expected_sha256=verifier_config.get("sha256"),
        )
        pcgrad_enabled = bool(verifier_config.get("pcgrad", False))
        config["frozen_verifier_fingerprint"] = verifier_metadata
    teacher_path = config.get("identity_safe_teacher")
    if safe_weight > 0:
        if not teacher_path:
            raise ValueError("identity_safe requires identity_safe_teacher")
        teacher_checkpoint = torch.load(Path(teacher_path), map_location="cpu", weights_only=False)
        teacher_model = make_model(teacher_checkpoint["config"]).to(device).eval()
        teacher_model.load_state_dict(teacher_checkpoint["model"])
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    all_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    max_steps = int(config.get("max_steps", 0))
    planned_steps = max_steps or int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config.get("scheduler_t_max", planned_steps)), 1),
        eta_min=float(config.get("min_learning_rate", 0.0)),
    )
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dataset = make_train_dataset(config)
    loader = make_loader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        workers=int(config["num_workers"]),
    )
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    source_weight = weights["source"]
    output_weight = weights["output"]
    degradation_weight = weights["degradation"]
    source_to_id: dict[str, int] = {}
    if float(weights.get("identity", 0.0)) > 0:
        groups = json.loads(
            (Path(config["data_root"]) / "metadata" / "source_groups.json").read_text(encoding="utf-8")
        )
        train_sources = sorted(group["source_id"] for group in groups if group["split"] == "train")
        source_to_id = {source_id: index for index, source_id in enumerate(train_sources)}
        if int(config["model"].get("identity_source_count", 71)) != len(train_sources):
            raise ValueError("identity_source_count must match the number of train sources")
    log_path = output_dir / "train_log.jsonl"
    best_psnr = float("-inf")
    step = 0
    best_step = None
    last_validation_psnr = None
    last_step = None
    start_epoch = 0
    resume_path = args.resume
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not bool(config.get("reset_scheduler", False)):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        best_step = checkpoint.get("best_step")
        last_validation_psnr = checkpoint.get("last_validation_psnr")
        last_step = checkpoint.get("last_step")
        restore_rng_state(checkpoint.get("rng_state"))
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    start = time.time()
    validation_interval = int(config.get("validation_interval_steps", 500))
    next_validation = ((step // validation_interval) + 1) * validation_interval
    diagnostic_interval = int(config.get("gradient_diagnostic_interval_steps", 200))
    latent_parameters = _latent_parameters(model)

    def run_validation(epoch: int) -> None:
        nonlocal best_psnr, best_step, last_validation_psnr, last_step, next_validation
        validation = evaluate(
            model,
            Path(config["data_root"]),
            device,
            int(config["seed"]),
            int(config["num_workers"]),
            split=str(config.get("validation_split", "val")),
            max_items=int(config.get("smoke_eval_items", 0)),
            tile_size=int(config.get("eval_tile_size", 512)),
            tile_overlap=int(config.get("eval_tile_overlap", 32)),
        )
        last_validation_psnr = float(validation["aggregate"]["psnr"])
        last_step = step
        validation_record = {"epoch": epoch, "step": step, **validation}
        with (output_dir / "validation_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
        is_best = last_validation_psnr > best_psnr
        if is_best:
            best_psnr = last_validation_psnr
            best_step = step
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "validation": validation,
            "best_psnr": best_psnr,
            "best_step": best_step,
            "last_validation_psnr": last_validation_psnr,
            "last_step": last_step,
            "next_validation": next_validation,
            "rng_state": capture_rng_state(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "last_validation.json").write_text(
            json.dumps(validation_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if is_best:
            (output_dir / "best_validation.json").write_text(
                json.dumps(validation_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps({
            "mode": config["mode"], "epoch": epoch, "step": step,
            "val_psnr": last_validation_psnr, "val_ssim": validation["aggregate"]["ssim"],
        }), flush=True)
        while next_validation <= step:
            next_validation += validation_interval

    for epoch in range(start_epoch, int(config["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        for batch in loader:
            if source_to_id and "source_ids" not in batch:
                batch["source_ids"] = torch.tensor(
                    [source_to_id[source_id] for source_id in batch["source_id"]],
                    dtype=torch.long,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, components, losses = train_step(
                    model, batch, str(config["mode"]), weights, device,
                    verifier=verifier, anchor_config=config.get("anchor_config"),
                    teacher_model=teacher_model, global_step=step,
                    warmup_steps=int(config.get("identity_safe_warmup_steps", 3000)),
                )
            diagnostic = {
                "grad_cos_rec_source": None, "grad_cos_rec_output": None,
                "grad_cos_rec_degradation": None, "grad_cos_rec_identity": None,
                "grad_cos_rec_anchor": None,
                "grad_norm_rec": None,
                "grad_norm_source": None, "grad_norm_output": None,
                "grad_norm_degradation": None, "grad_norm_identity": None,
                "grad_norm_anchor": None,
                "grad_ratio_source_to_rec": None,
                "grad_ratio_output_to_rec": None, "grad_ratio_degradation_to_rec": None,
                "grad_ratio_identity_to_rec": None,
                "grad_ratio_anchor_to_rec": None,
                "weighted_grad_norm_source": None, "weighted_grad_norm_output": None,
                "weighted_grad_norm_degradation": None, "weighted_grad_norm_identity": None,
                "weighted_grad_norm_anchor": None,
                "weighted_grad_ratio_source_to_rec": None,
                "weighted_grad_ratio_output_to_rec": None,
                "weighted_grad_ratio_degradation_to_rec": None,
                "weighted_grad_ratio_identity_to_rec": None,
                "weighted_grad_ratio_anchor_to_rec": None,
                "pcgrad_projection_applied": None, "pcgrad_dot_before": None,
                "pcgrad_dot_after": None, "pcgrad_anchor_norm_before": None,
                "pcgrad_anchor_norm_after": None,
            }
            if diagnostic_interval > 0 and (step + 1) % diagnostic_interval == 0:
                rec_vector, rec_norm = _gradient_stats(losses["reconstruction"], latent_parameters)
                source_vector, source_norm = _gradient_stats(losses["source"], latent_parameters)
                output_vector, output_norm = _gradient_stats(losses["output"], latent_parameters)
                degradation_vector, degradation_norm = _gradient_stats(losses["degradation"], latent_parameters)
                identity_vector, identity_norm = _gradient_stats(losses["identity"], latent_parameters)
                diagnostic.update({
                    "grad_cos_rec_source": _cosine(rec_vector, source_vector),
                    "grad_cos_rec_output": _cosine(rec_vector, output_vector),
                    "grad_cos_rec_degradation": _cosine(rec_vector, degradation_vector),
                    "grad_cos_rec_identity": _cosine(rec_vector, identity_vector),
                    "grad_norm_rec": rec_norm, "grad_norm_source": source_norm,
                    "grad_norm_output": output_norm, "grad_norm_degradation": degradation_norm,
                    "grad_norm_identity": identity_norm,
                    "grad_ratio_source_to_rec": source_norm / rec_norm if source_norm is not None and rec_norm else None,
                    "grad_ratio_output_to_rec": output_norm / rec_norm if output_norm is not None and rec_norm else None,
                    "grad_ratio_degradation_to_rec": degradation_norm / rec_norm if degradation_norm is not None and rec_norm else None,
                    "grad_ratio_identity_to_rec": identity_norm / rec_norm if identity_norm is not None and rec_norm else None,
                    "weighted_grad_norm_source": source_norm * source_weight if source_norm is not None else None,
                    "weighted_grad_norm_output": output_norm * output_weight if output_norm is not None else None,
                    "weighted_grad_norm_degradation": degradation_norm * degradation_weight if degradation_norm is not None else None,
                    "weighted_grad_norm_identity": identity_norm * float(weights.get("identity", 0.0)) if identity_norm is not None else None,
                    "weighted_grad_ratio_source_to_rec": source_norm * source_weight / rec_norm if source_norm is not None and rec_norm else None,
                    "weighted_grad_ratio_output_to_rec": output_norm * output_weight / rec_norm if output_norm is not None and rec_norm else None,
                    "weighted_grad_ratio_degradation_to_rec": degradation_norm * degradation_weight / rec_norm if degradation_norm is not None and rec_norm else None,
                    "weighted_grad_ratio_identity_to_rec": identity_norm * float(weights.get("identity", 0.0)) / rec_norm if identity_norm is not None and rec_norm else None,
                })
                if anchor_weight > 0 and losses["anchor"] is not None and losses["reconstruction"] is not None:
                    rec_objective = losses["reconstruction"] + weights["gradient"] * losses["gradient_objective"]
                    anchor_vector = _aligned_gradient_vector(losses["anchor"] * anchor_weight, all_parameters)
                    rec_obj_vector = _aligned_gradient_vector(rec_objective, all_parameters)
                    rec_obj_norm = float(torch.linalg.vector_norm(rec_obj_vector)) if rec_obj_vector is not None else None
                    anchor_norm = float(torch.linalg.vector_norm(anchor_vector)) if anchor_vector is not None else None
                    diagnostic.update({
                        "grad_cos_rec_anchor": _cosine(rec_obj_vector, anchor_vector),
                        "grad_norm_rec": rec_obj_norm,
                        "grad_norm_anchor": anchor_norm,
                        "grad_ratio_anchor_to_rec": anchor_norm / rec_obj_norm if anchor_norm is not None and rec_obj_norm else None,
                        "weighted_grad_norm_anchor": anchor_norm,
                        "weighted_grad_ratio_anchor_to_rec": anchor_norm / rec_obj_norm if anchor_norm is not None and rec_obj_norm else None,
                    })
            if pcgrad_enabled and anchor_weight > 0 and losses["anchor"] is not None:
                base = losses["reconstruction"] + weights["gradient"] * losses["gradient_objective"]
                for key in ("output", "degradation"):
                    if losses.get(key) is not None:
                        base = base + weights[key] * losses[key]
                rec_obj_vector = _aligned_gradient_vector(base, all_parameters)
                anchor_vector = _aligned_gradient_vector(losses["anchor"] * anchor_weight, all_parameters)
                projected, applied, before, after = project_conflicting_gradient(anchor_vector, rec_obj_vector)
                scaler.scale(base).backward()
                scaler.unscale_(optimizer)
                offset = 0
                with torch.no_grad():
                    for parameter in all_parameters:
                        numel = parameter.numel()
                        segment = projected[offset : offset + numel].reshape_as(parameter).float()
                        if parameter.grad is None:
                            parameter.grad = segment.clone()
                        else:
                            parameter.grad.add_(segment)
                        offset += numel
                if float(config["gradient_clip"]) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                diagnostic.update({
                    "pcgrad_projection_applied": applied,
                    "pcgrad_dot_before": before,
                    "pcgrad_dot_after": after,
                    "pcgrad_anchor_norm_before": float(torch.linalg.vector_norm(anchor_vector)),
                    "pcgrad_anchor_norm_after": float(torch.linalg.vector_norm(projected)),
                })
            else:
                scaler.scale(loss).backward()
                if float(config["gradient_clip"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            step += 1
            record = {
                "epoch": epoch,
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **components,
                **diagnostic,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if step >= next_validation or (max_steps and step >= max_steps):
                model.eval()
                run_validation(epoch)
                model.train()
            if max_steps and step >= max_steps:
                break
        # Validate at the epoch boundary only if no step-aligned validation ran.
        if step >= next_validation or (max_steps and step >= max_steps):
            model.eval()
            run_validation(epoch)
            model.train()
        if max_steps and step >= max_steps:
            break

    summary = {
        "status": "smoke_only" if max_steps and max_steps <= 5 else "formal_12k" if max_steps == 12000 else "exploratory",
        "mode": config["mode"],
        "device": str(device),
        "steps": step,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - start,
        "best_validation_psnr": best_psnr if best_psnr > float("-inf") else None,
        "best_step": best_step,
        "last_validation_psnr": last_validation_psnr,
        "last_step": last_step,
        "pcgrad": pcgrad_enabled,
        "frozen_verifier_fingerprint": verifier_metadata,
        "warning": "Do not treat smoke or exploratory metrics as publication evidence.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
