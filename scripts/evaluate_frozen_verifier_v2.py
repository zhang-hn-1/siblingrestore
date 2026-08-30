"""
冻结 verifier 评估 v2（正式实验管线）。

相对 v1 的改进（对应 LARGE_SCALE_EXPERIMENT_DESIGN.md 第 5 节）：
  1. 流式：不保留全部恢复图，sibling/cross L1 在线计算（大 split 不爆内存）；
  2. 断点续跑：每 source 一行 <output>.sources.jsonl（含每 view 全 score 向量）+
     每 source 均值 <output>.means/*.pt，重跑自动跳过已完成 source 且聚合口径完整；
  3. 指纹：报告内含 verifier sha / 数据包 audit sha / 模型 config sha，汇总表口径强制一致；
  4. LPIPS（--lpips，256x256 协议，AlexNet）；
  5. 按退化聚合（PSNR/SSIM/top1/margin/AUC/EER 每退化一份，论文表 2 直接可用）；
  6. anchor bank 缓存：同一 (verifier, split, 数据包, tile 参数) 只算一次；
  7. noise（Method B）尺寸不匹配：clean Lanczos 对齐到 degraded 后评估；
     恢复视图统一 resize 到 clean 原生尺寸后再算 sibling/cross L1；
  8. --save-pairs 才输出全 pair CSV（大 split 默认不落盘）。

用法：
  python scripts/evaluate_frozen_verifier_v2.py \
      --checkpoint runs/.../best.pt --verifier runs/.../verifier_evaluator/best.pt \
      --data-root data/plamd_sfr_v1 \
      --output results/campaigns/c001/ours_dim48_anchor.13.val.json \
      --split val --device cuda --method ours_dim48_anchor --seed 13 --lpips
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siblingrestore.data import align_pair_shape, paired_eval_pad
from siblingrestore.inference import restore_tiled
from siblingrestore.metrics import psnr, ssim
from siblingrestore.verifier import load_verifier_checkpoint

DEGS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    result = list(range(0, length - tile_size + 1, stride))
    if result[-1] != length - tile_size:
        result.append(length - tile_size)
    return result


def verifier_embed_tiled(verifier, image: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("verifier_embed_tiled expects [1,C,H,W]")
    _, _, height, width = image.shape
    pad_h, pad_w = (8 - height % 8) % 8, (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    tiles = []
    for top in positions(padded.shape[-2], tile_size, overlap):
        for left in positions(padded.shape[-1], tile_size, overlap):
            tiles.append(padded[..., top : top + tile_size, left : left + tile_size])
    batch = torch.cat(tiles, dim=0)
    embeddings = verifier.embed(batch)
    return F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)


def binary_auc(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def eer(scores: list[float], labels: list[int]) -> float:
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int8)
    positives = int(labels_np.sum())
    negatives = int(labels_np.size - positives)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(-scores_np, kind="stable")
    sorted_scores = scores_np[order]
    sorted_labels = labels_np[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    # A tied score threshold includes the whole tie group, matching score >=.
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    fpr = cumulative_fp[group_ends] / negatives
    fnr = (positives - cumulative_tp[group_ends]) / positives
    return float(np.min((fpr + fnr) / 2.0))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_anchor_bank(verifier, groups, data_root, cache_root, verifier_sha, split, tile_size, tile_overlap, device):
    """anchor bank 缓存：同一 (verifier, split, 数据包, tile 参数) 只算一次。"""
    source_key_all = hashlib.sha256(
        "|".join(g["source_id"] for g in groups).encode()).hexdigest()[:12]
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"anchors_{verifier_sha[:12]}_{split}_{source_key_all}_t{tile_size}_o{tile_overlap}.pt"
    if path.exists():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return obj["bank"], obj["source_ids"]
    anchors = []
    with torch.no_grad():
        for group in groups:
            anchors.append(verifier_embed_tiled(
                verifier, read_rgb(data_root / group["clean_path"]).unsqueeze(0).to(device),
                tile_size, tile_overlap).cpu())
    bank = torch.cat(anchors)
    torch.save({"bank": bank, "source_ids": [g["source_id"] for g in groups],
                "params": {"split": split, "tile_size": tile_size, "tile_overlap": tile_overlap,
                           "verifier_sha": verifier_sha, "sources": len(groups)}}, path)
    return bank, [g["source_id"] for g in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--save-pairs", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    restoration = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from train import make_model
    model = make_model(restoration["config"]).to(device).eval()
    model.load_state_dict(restoration["model"])
    verifier, fingerprint = load_verifier_checkpoint(args.verifier, device)
    verifier.eval()

    data_root = Path(args.data_root)
    audit = json.loads((data_root / "metadata" / "audit.json").read_text(encoding="utf-8"))
    data_pkg_hash = audit.get("index_sha256", "unknown")[:12]
    config_hash = hashlib.sha256(json.dumps(restoration["config"], sort_keys=True).encode()).hexdigest()[:12]

    groups = [g for g in json.loads((data_root / "metadata" / "source_groups.json").read_text(encoding="utf-8"))
              if g["split"] == args.split]
    groups = groups[args.source_offset:]
    if args.max_sources:
        groups = groups[: args.max_sources]

    bank, bank_source_ids = load_anchor_bank(
        verifier, groups, data_root, data_root / ".." / "cache" / "anchors",
        fingerprint["sha256"], args.split, args.tile_size, args.tile_overlap, device)
    bank_index = {sid: i for i, sid in enumerate(bank_source_ids)}

    lpips_fn = None
    if args.lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] lpips 不可用：{exc}；跳过 LPIPS", file=sys.stderr)

    out_path = Path(args.output)
    sources_log = out_path.with_suffix(out_path.suffix + ".sources.jsonl")
    means_dir = out_path.with_suffix(out_path.suffix + ".means")

    # ---- 断点加载：已完成 source 的完整记录（含每 view 全 score 向量）----
    done: dict[str, dict] = {}
    if sources_log.exists():
        for line in sources_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["source_id"]] = rec

    rows: list[dict] = []
    means: list[torch.Tensor] = []
    sibling_l1_total = 0.0
    per_deg_scores: dict[str, dict[str, tuple[list, list]]] = {
        d: {"input": ([], []), "restored": ([], [])} for d in DEGS
    }
    pair_input, pair_restored = [], []  # 仅 --save-pairs 时使用

    def consume(rec: dict) -> None:
        """把已完成 source 的记录回放进聚合状态（live 与 resume 共用）。"""
        nonlocal sibling_l1_total
        rows.extend(rec["rows"])
        sibling_l1_total += rec["sibling_l1"]
        mean_pt = means_dir / f"{rec['source_id'].replace('/', '_')}.pt"
        if mean_pt.exists():
            means.append(torch.load(mean_pt, map_location="cpu", weights_only=False)["mean"])
        for view in rec["views"]:
            deg = view["degradation"]
            own_idx = bank_index[rec["source_id"]]
            for qt in ("input", "restored"):
                scores = view[f"{qt}_scores"]
                scores_list, labels_list = per_deg_scores[deg][qt]
                scores_list.extend(scores)
                labels_list.extend(int(i == own_idx) for i in range(len(scores)))
            if args.save_pairs:
                for qt in ("input", "restored"):
                    target = pair_input if qt == "input" else pair_restored
                    scores = view[f"{qt}_scores"]
                    target.extend((rec["source_id"], bank_source_ids[i], int(i == own_idx), s)
                                  for i, s in enumerate(scores))

    for rec in done.values():
        consume(rec)

    # ---- 主循环 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = sources_log.open("a", encoding="utf-8")
    try:
        for group in groups:
            source_id = group["source_id"]
            if source_id in done:
                continue
            clean_original = read_rgb(data_root / group["clean_path"])
            clean_h, clean_w = int(clean_original.shape[-2]), int(clean_original.shape[-1])
            restored_views, view_records = [], []
            with torch.no_grad():
                for degradation in sorted(group["degraded"]):
                    degraded_original = read_rgb(data_root / group["degraded"][degradation])
                    degraded, clean = align_pair_shape(degraded_original, clean_original)
                    degraded_pad, clean_pad, mask = paired_eval_pad(degraded, clean)
                    restored_padded = restore_tiled(model, degraded_pad.unsqueeze(0).to(device),
                                                    args.tile_size, args.tile_overlap).cpu()
                    restored = restored_padded[..., : degraded.shape[-2], : degraded.shape[-1]].squeeze(0)
                    # 统一到 clean 原生尺寸（noise 对 restored 在 noise 尺寸）
                    restored_aligned = (restored if (int(restored.shape[-2]), int(restored.shape[-1])) == (clean_h, clean_w)
                                        else F.interpolate(restored.unsqueeze(0), size=(clean_h, clean_w),
                                                           mode="bilinear", align_corners=False).squeeze(0))
                    input_embedding = verifier_embed_tiled(verifier, degraded_original.unsqueeze(0).to(device),
                                                           args.tile_size, args.tile_overlap).cpu()
                    restored_embedding = verifier_embed_tiled(verifier, restored_aligned.unsqueeze(0).to(device),
                                                              args.tile_size, args.tile_overlap).cpu()
                    input_scores = (input_embedding @ bank.T).squeeze(0)
                    restored_scores = (restored_embedding @ bank.T).squeeze(0)
                    own_idx = bank_index[source_id]
                    input_own = float(input_scores[own_idx])
                    restored_own = float(restored_scores[own_idx])
                    input_impostor = max((float(input_scores[i]) for i in range(len(bank_index)) if i != own_idx), default=float("nan"))
                    restored_impostor = max((float(restored_scores[i]) for i in range(len(bank_index)) if i != own_idx), default=float("nan"))
                    if lpips_fn is not None:
                        with torch.no_grad():
                            x = F.interpolate(restored_aligned.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
                            y = F.interpolate(clean_original.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
                            lpips_value = float(lpips_fn((x * 2 - 1).to(device), (y * 2 - 1).to(device)))
                    else:
                        lpips_value = float("nan")
                    row = {"method": args.method, "seed": args.seed, "source_id": source_id,
                           "subcategory": group["subcategory"], "degradation": degradation,
                           "psnr": psnr(restored_padded, clean_pad.unsqueeze(0), mask.unsqueeze(0)),
                           "ssim": ssim(restored_padded, clean_pad.unsqueeze(0), mask.unsqueeze(0)),
                           "lpips": lpips_value,
                           "input_to_clean_top1": int(input_scores.argmax() == own_idx),
                           "restored_to_clean_top1": int(restored_scores.argmax() == own_idx),
                           "input_own_anchor_cos": input_own, "restored_own_anchor_cos": restored_own,
                           "input_nearest_impostor_cos": input_impostor, "restored_nearest_impostor_cos": restored_impostor,
                           "input_margin": input_own - input_impostor, "restored_margin": restored_own - restored_impostor,
                           "identity_gain_margin": (restored_own - input_own) - (restored_impostor - input_impostor)}
                    view_records.append({"degradation": degradation, "row": row,
                                         "input_scores": input_scores.tolist(),
                                         "restored_scores": restored_scores.tolist()})
                    restored_views.append(restored_aligned)
            views = torch.stack(restored_views)
            pairs = list(combinations(range(len(views)), 2))
            sibling_l1 = float(sum(F.l1_loss(views[a], views[b], reduction="mean") for a, b in pairs) / len(pairs))
            mean = F.interpolate(views.mean(dim=0).unsqueeze(0), size=(256, 256),
                                 mode="bilinear", align_corners=False).squeeze(0)
            rec = {"source_id": source_id, "rows": [v["row"] for v in view_records],
                   "views": [{k: v[k] for k in ("degradation", "input_scores", "restored_scores")} for v in view_records],
                   "sibling_l1": sibling_l1}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            means_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"mean": mean, "sibling_l1": sibling_l1},
                       means_dir / f"{source_id.replace('/', '_')}.pt")
            consume(rec)
            print(f"[{args.method} {args.split}] {len(done) + len(rows) // 7}/{len(groups)}  {source_id}", file=sys.stderr)
    finally:
        handle.close()

    if len(done) + len(rows) // 7 < len(groups):
        raise SystemExit(f"未完成：{(len(done) + len(rows) // 7)}/{len(groups)}，重跑自动续传")

    # ---- 全局聚合 ----
    def agg_metrics(input_scores: list[float], input_labels: list[int],
                    restored_scores: list[float], restored_labels: list[int]) -> dict:
        return {"input_roc_auc": binary_auc(input_scores, input_labels),
                "input_eer": eer(input_scores, input_labels),
                "restored_roc_auc": binary_auc(restored_scores, restored_labels),
                "restored_eer": eer(restored_scores, restored_labels),
                "roc_auc_gain": binary_auc(restored_scores, restored_labels) - binary_auc(input_scores, input_labels),
                "eer_gain": eer(input_scores, input_labels) - eer(restored_scores, restored_labels)}

    global_scores = {qt: ([], []) for qt in ("input", "restored")}
    per_degradation = {}
    for deg, d in per_deg_scores.items():
        for qt in ("input", "restored"):
            global_scores[qt][0].extend(d[qt][0])
            global_scores[qt][1].extend(d[qt][1])
        deg_rows = [r for r in rows if r["degradation"] == deg]
        per_degradation[deg] = {
            "psnr": float(np.mean([r["psnr"] for r in deg_rows])),
            "ssim": float(np.mean([r["ssim"] for r in deg_rows])),
            "lpips": float(np.mean([r["lpips"] for r in deg_rows])),
            "restored_to_clean_top1": float(np.mean([r["restored_to_clean_top1"] for r in deg_rows])),
            **agg_metrics(*d["input"], *d["restored"]),
        }
    aggregate = {key: float(np.mean([r[key] for r in rows])) for key in
                 ("psnr", "ssim", "input_to_clean_top1", "restored_to_clean_top1",
                  "input_own_anchor_cos", "restored_own_anchor_cos",
                  "input_nearest_impostor_cos", "restored_nearest_impostor_cos",
                  "input_margin", "restored_margin", "identity_gain_margin")}
    aggregate["lpips"] = float(np.mean([r["lpips"] for r in rows]))
    aggregate.update(agg_metrics(*global_scores["input"], *global_scores["restored"]))
    # Exact cross-source L1 aggregation via the sorted-values identity:
    # for each feature, sum_i (2*i-n+1) * sorted(x)_i.  This is equivalent
    # to all 8,128 pairwise mean-L1 values without Python pair loops or cdist
    # fallbacks in older PyTorch builds.
    means_t = torch.stack(means).flatten(1).to(device)
    source_count, feature_count = means_t.shape
    sorted_means, _ = torch.sort(means_t, dim=0)
    coefficients = (2 * torch.arange(source_count, device=device, dtype=means_t.dtype)
                    - source_count + 1).unsqueeze(1)
    cross_sum = (sorted_means * coefficients).sum()
    pair_count = source_count * (source_count - 1) // 2
    cross = [float((cross_sum / max(pair_count * feature_count, 1)).detach().cpu())]
    aggregate["sibling_output_l1"] = sibling_l1_total / max(len(groups), 1)
    aggregate["cross_source_output_l1"] = float(np.mean(cross))
    aggregate["intra_inter_output_ratio"] = aggregate["sibling_output_l1"] / max(aggregate["cross_source_output_l1"], 1e-9)

    report = {"status": "independent_verifier_evaluation", "eval_version": "v2",
              "split": args.split, "sources": len(groups), "restored_views": len(rows),
              "verifier_fingerprint": fingerprint,
              "data_pkg_fingerprint": {"audit_index_sha256": audit.get("index_sha256"),
                                       "audit_groups_sha256": audit.get("groups_sha256")},
              "config_fingerprint": {"sha256": config_hash, "config": restoration["config"]},
              "lpips_protocol": "alex_256x256" if lpips_fn is not None else None,
              "aggregate": aggregate, "per_degradation": per_degradation,
              "notes": ["Headline identity metrics use an independent frozen verifier.",
                        "Test remains sealed unless explicitly requested after validation selection.",
                        "noise pairs: clean aligned to degraded native size (Lanczos) per Method B protocol."]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_pairs:
        with out_path.with_suffix(out_path.suffix + "_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["query_type", "query_source_id", "anchor_source_id", "label", "score"])
            for qt, pairs in (("input", pair_input), ("restored", pair_restored)):
                for q, a, label, score in pairs:
                    writer.writerow([qt, q, a, label, score])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
