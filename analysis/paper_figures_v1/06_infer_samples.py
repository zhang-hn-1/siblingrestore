"""06_infer_samples.py —— 用已有 checkpoint 对选定案例补推理恢复图。

方法与官方 v2 完全一致：align_pair_shape -> paired_eval_pad -> restore_tiled(512/32)，
noise 尺寸不一致时恢复图 bilinear 回到 clean 原生尺寸；退化输入同样统一到 clean 尺寸，
保证同源各图空间对齐（图 2/3/5 需求）。
仅处理 case_plan.json 涉及的少量视图（图 2 两个来源 x 7 退化，图 3 三个案例，图 5 一个来源）。

用法: python analysis/paper_figures_v1/06_infer_samples.py [--device cuda:3] [--force]
输出: outputs/paper_figures_v1/sample_restored/<method>/...png 与 _ref/...png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import DEGRADATIONS, METHODS, OUT, ROOT  # noqa: E402

# 需要推理的方法（A5/B1 为论文核心；dehazeformer 作为图 5 外部基线代表）
INFER_METHODS = ["A5", "B1", "dehazeformer"]
DEST = OUT / "sample_restored"
METHOD_BY_ID = {m["id"]: m for m in METHODS}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def to_png(t: torch.Tensor, path: Path) -> None:
    t = t.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(t).save(path)


def load_model(mid: str, device: torch.device):
    from train import make_model
    m = METHOD_BY_ID[mid]
    ck = torch.load(ROOT / m["ckpt"], map_location="cpu", weights_only=False)
    model = make_model(ck["config"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


def restore_one(model, degraded_path: Path, clean_path: Path, device, tile=512, overlap=32):
    """返回 (degraded_disp, restored_disp, clean_disp) 均在 clean 原生尺寸。"""
    from siblingrestore.data import align_pair_shape, paired_eval_pad
    from siblingrestore.inference import restore_tiled
    with torch.no_grad():
        dg_orig = read_rgb(degraded_path)
        cl_orig = read_rgb(clean_path)
        ch, cw = int(cl_orig.shape[-2]), int(cl_orig.shape[-1])
        dg, clean = align_pair_shape(dg_orig, cl_orig)
        dg_pad, clean_pad, mask = paired_eval_pad(dg, clean)
        restored_pad = restore_tiled(model, dg_pad.unsqueeze(0).to(device), tile, overlap)
        restored = restored_pad[..., : dg.shape[-2], : dg.shape[-1]].squeeze(0).cpu()
        # 统一到 clean 原生尺寸
        if (int(restored.shape[-2]), int(restored.shape[-1])) != (ch, cw):
            restored = F.interpolate(restored.unsqueeze(0), size=(ch, cw),
                                     mode="bilinear", align_corners=False).squeeze(0)
        if (int(dg_orig.shape[-2]), int(dg_orig.shape[-1])) != (ch, cw):
            dg_disp = F.interpolate(dg_orig.unsqueeze(0), size=(ch, cw),
                                    mode="bilinear", align_corners=False).squeeze(0)
        else:
            dg_disp = dg_orig
        return dg_disp, restored, cl_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "_ref").mkdir(parents=True, exist_ok=True)

    groups = json.loads((ROOT / "data/plamd_sfr_v1/metadata/source_groups.json").read_text())
    gmap = {g["source_id"]: g for g in groups}
    plan = json.loads((OUT / "cases/case_plan.json").read_text())

    # 任务集合：source_id -> set(deg)
    tasks: dict[str, set[str]] = {}
    for src in plan["fig2_sources"]:
        tasks.setdefault(src["source_id"], set()).update(DEGRADATIONS)
    for c in plan["fig3_cases"]:
        tasks.setdefault(c["source_id"], set()).add(c["degradation"])
    f5 = plan["fig5"]
    tasks.setdefault(f5["source_id"], set()).add(f5["degradation"])

    models = {mid: load_model(mid, device) for mid in INFER_METHODS}
    print("models loaded:", list(models))

    def safe(s: str) -> str:
        return s.replace("/", "_").replace(" ", "_")

    for sid, degs in tasks.items():
        g = gmap[sid]
        clean_p = ROOT / "data" / "plamd_sfr_v1" / g["clean_path"]
        cl = read_rgb(clean_p)
        to_png(cl, DEST / "_ref" / f"{safe(sid)}__clean.png")
        for d in sorted(degs):
            dg_path = ROOT / "data" / "plamd_sfr_v1" / g["degraded"][d]
            for mid, model in models.items():
                out_p = DEST / mid / f"{safe(sid)}__{d}.png"
                if out_p.exists() and not args.force:
                    continue
                dg_disp, restored, _ = restore_one(model, dg_path, clean_p, device)
                to_png(restored, out_p)
                to_png(dg_disp, DEST / "_ref" / f"{safe(sid)}__{d}__input.png")
                print("restored", mid, d, sid[:40], out_p.name, flush=True)
        print("source done", sid[:60], flush=True)

    # fig5 局部窗口：|degraded-clean| 最大误差（限中央 3/4），192x192，回填 plan
    f5_g = gmap[f5["source_id"]]
    dg_p = ROOT / "data" / "plamd_sfr_v1" / f5_g["degraded"][f5["degradation"]]
    clean_p = ROOT / "data" / "plamd_sfr_v1" / f5_g["clean_path"]
    dg_img = read_rgb(dg_p)
    cl_img = read_rgb(clean_p)
    if dg_img.shape != cl_img.shape:
        dg_img = F.interpolate(dg_img.unsqueeze(0), size=cl_img.shape[-2:],
                               mode="bilinear", align_corners=False).squeeze(0)
    err = (dg_img - cl_img).abs().mean(dim=0).numpy()  # [H,W]
    H, W = err.shape
    wh = 192
    y0 = int(H * 0.125); y1 = int(H * 0.875) - wh
    x0 = int(W * 0.125); x1 = int(W * 0.875) - wh
    best = None
    for yy in range(y0, y1 + 1, 8):
        for xx in range(x0, x1 + 1, 8):
            s = err[yy:yy + wh, xx:xx + wh].sum()
            if best is None or s > best[0]:
                best = (s, yy, xx)
    _, cy, cx = best
    plan["fig5"]["crop"] = [int(cy), int(cy + wh), int(cx), int(cx + wh)]
    (OUT / "cases/case_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("fig5 crop set:", plan["fig5"]["crop"])


if __name__ == "__main__":
    main()
