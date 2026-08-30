from __future__ import annotations

"""Dedicated training entry for the R2R (cscxwang/R2R) baseline.

R2R's degradation-memory protocol requires (degraded, clean, degradation ids)
per sample, which the shared train.py single-input loop does not provide, so
this script drives R2R with the same PairDataset + Charbonnier/gradient loss
used by the other baselines while feeding R2R's native inputs.

R2R maps its 5 memory classes to our 7 degradations as:
  noise/inpainting/snow -> denoise(0), rain -> derain(1), haze -> dehaze(2),
  blur -> deblur(3), lowlight -> lowlight(4).
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from siblingrestore.data import PairDataset
from siblingrestore.losses import charbonnier, gradient_loss
from siblingrestore.metrics import psnr, ssim

# our degradation -> R2R memory class id
DEG_TO_R2R = {"blur": 3, "haze": 2, "inpainting": 0, "lowlight": 4,
              "noise": 0, "rain": 1, "snow": 0}
R2R_NUM_CLASSES = 5


def load_r2r_model() -> torch.nn.Module:
    import importlib.machinery
    import sys
    import types
    from pathlib import Path as P

    r2r_dir = P(__file__).parent / "siblingrestore" / "official" / "third_party" / "r2r_net"
    sys.path.insert(0, str(r2r_dir.parent))
    net = types.ModuleType("net")
    net.__spec__ = importlib.machinery.ModuleSpec("net", None)
    sys.modules["net"] = net
    for sub in ("feature_bank_5D", "model_5D"):
        spec = importlib.util.spec_from_file_location(f"net.{sub}", r2r_dir / f"{sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"net.{sub}"] = module
        spec.loader.exec_module(module)
    sys.path.insert(0, str(r2r_dir))
    from model_5D import R2R
    return R2R(is_train=True, train_mode="finetune", num_classes=R2R_NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-interval-steps", type=int, default=500)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_r2r_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.min_lr
    )

    dataset = PairDataset(args.data_root, "train", args.crop_size, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    val_dataset = PairDataset(args.data_root, "val", None, args.seed)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    log_path = out / "train_log.jsonl"
    best_psnr = float("-inf")
    best_step = None
    step = 0
    next_validation = args.validation_interval_steps
    start = time.time()

    while step < args.max_steps:
        model.train()
        for batch in loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            # R2R needs degradation ids for the memory interaction.
            ids = torch.tensor(
                [DEG_TO_R2R[str(batch["degradation"][i])] + 2 for i in range(len(degraded))],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            # finetune-mode forward returns (restored, scores, gt_ids)
            restored, scores, gt_ids = model(
                degraded, clean=clean, ids=ids, interact_label=None
            )
            rec = charbonnier(restored, clean)
            grad = gradient_loss(restored, clean)
            total = rec + 0.05 * grad
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "step": step, "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "reconstruction": float(rec.detach()), "total_loss": float(total.detach()),
                }) + "\n")
            if step >= args.max_steps:
                break

        if step >= next_validation or step >= args.max_steps:
            model.eval()
            agg = []
            with torch.no_grad():
                for vb in val_loader:
                    vd = vb["degraded"].to(device)
                    vc = vb["clean"].to(device)
                    vids = torch.tensor([DEG_TO_R2R[str(vb["degradation"][0])] + 2], device=device)
                    out_img, *_ = model(vd, clean=vd, ids=vids, interact_label=None)
                    # R2R pads the input to a multiple of padder_size; crop the
                    # restored output back to the clean image's size.
                    _, _, h, w = vc.shape
                    out_img = out_img[..., :h, :w]
                    agg.append(psnr(out_img, vc, vb["mask"].to(device)))
            val_psnr = float(torch.tensor(agg).mean())
            print(json.dumps({"mode": "r2r", "epoch": step // len(loader), "step": step,
                              "val_psnr": val_psnr}), flush=True)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_step = step
                torch.save({"model": model.state_dict(), "step": step,
                            "best_psnr": best_psnr}, out / "best.pt")
            torch.save({"model": model.state_dict(), "step": step,
                        "best_psnr": best_psnr}, out / "last.pt")
            while next_validation <= step:
                next_validation += args.validation_interval_steps
            if step >= args.max_steps:
                break

    summary = {"status": "exploratory", "mode": "independent", "steps": step,
               "best_validation_psnr": best_psnr if best_psnr > float("-inf") else None,
               "best_step": best_step,
               "elapsed_seconds": time.time() - start}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
