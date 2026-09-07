"""paper_figures_v1 公共定义：路径、方法目录、评测协议常量。

数据协议（与 scripts/evaluate_frozen_verifier_v2.py 一致）：
  split = test, 256 sources x 7 degradations = 1792 restored views
  verifier = runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt
  verifier_sha256 = d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6
  lpips = alex_256x256, psnr/ssim = 官方 v2 masked 口径, [0,1] RGB float
  noise = clean Lanczos 对齐到 noise 原生尺寸
所有方法都使用同一评测脚本与同一 verifier，测试样本完全一致。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis" / "paper_figures_v1"
OUT = ROOT / "outputs" / "paper_figures_v1"

VERIFIER_SHA = "d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6"
VERIFIER_PATH = ROOT / "runs" / "campaigns" / "c001_sfr_v1" / "verifier_evaluator" / "best.pt"

DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")

# ---------------------------------------------------------------------------
# 方法目录：cohort 用于区分训练批次（不同批次不直接横比，保留各自训练元信息）。
# result_prefix = results 文件前缀（相对 ROOT），*.test.json / *.test.csv /
#                 *.test.json.sources.jsonl / *.test.json.means 由该前缀派生。
# ckpt = best.pt 相对 ROOT 路径（供指纹与补推理使用）。
# ---------------------------------------------------------------------------

def _m(mid: str, cohort: str, result_prefix: str, ckpt: str, model_family: str,
       config_note: str, is_ours: bool = False, group: str | None = None):
    return {
        "id": mid, "cohort": cohort, "result_prefix": result_prefix, "ckpt": ckpt,
        "model_family": model_family, "config_note": config_note,
        "is_ours": is_ours, "group": group,
    }

METHODS = [
    # ---- 核心消融 (ablation_core)，mode=sibling 同源分组采样, max_steps=20000, seed13 ----
    _m("A0", "ablation_core", "results/ablation_core/A0_recon.13", "runs/ablation_core/A0_recon_seed13/best.pt",
       "siblingrestormer", "rec+grad; mode=sibling; 20k", is_ours=True),
    _m("B1", "ablation_core", "results/ablation_core/B1_deg_only.13", "runs/ablation_core/B1_deg_only_seed13/best.pt",
       "siblingrestormer", "rec+grad+deg; mode=sibling; 20k", is_ours=True),
    _m("B2", "ablation_core", "results/ablation_core/B2_anchor_only.13", "runs/ablation_core/B2_anchor_only_seed13/best.pt",
       "siblingrestormer", "rec+grad+anchor; mode=sibling; 20k", is_ours=True),
    _m("A5", "ablation_core", "results/ablation_core/A5_no_sibling.13", "runs/ablation_core/A5_no_sibling_seed13/best.pt",
       "siblingrestormer", "rec+grad+deg+anchor; mode=sibling; 20k", is_ours=True),
    _m("A1", "ablation_core", "results/ablation_core/A1_sibling.13", "runs/ablation_core/A1_sibling_seed13/best.pt",
       "siblingrestormer", "rec+grad+sibling; mode=sibling; 20k", is_ours=True),
    _m("A2", "ablation_core", "results/ablation_core/A2_degradation.13", "runs/ablation_core/A2_degradation_seed13/best.pt",
       "siblingrestormer", "rec+grad+deg; mode=sibling; 20k", is_ours=True),
    _m("A3", "ablation_core", "results/ablation_core/A3_full_components.13", "runs/ablation_core/A3_full_components_seed13/best.pt",
       "siblingrestormer", "rec+grad+deg+anchor+sibling; mode=sibling; 20k", is_ours=True),
    _m("C1", "ablation_core", "results/ablation_core/C1_adaptive_anchor_only.13", "runs/ablation_core/C1_adaptive_anchor_only_seed13/best.pt",
       "siblingrestormer", "rec+grad+adaptive_anchor; mode=sibling; 20k", is_ours=True),
    _m("C2", "ablation_core", "results/ablation_core/C2_deg_adaptive_anchor.13", "runs/ablation_core/C2_deg_adaptive_anchor_seed13/best.pt",
       "siblingrestormer", "rec+grad+deg+adaptive_anchor; mode=sibling; 20k", is_ours=True),

    # ---- c005 官方对比 (外部基线 + dim48/dim64 家族), mode=independent, max_steps=12000 ----
    _m("dehazeformer", "c005", "results/campaigns/c005_official_group1/dehazeformer.13",
       "runs/campaigns/c005_official_group1/dehazeformer/13/best.pt", "dehazeformer", "12k independent"),
    _m("restormer", "c005", "results/campaigns/c005_official_group2/restormer.13",
       "runs/campaigns/c005_official_group2/restormer/13/best.pt", "restormer", "12k independent"),
    _m("promptir", "c005", "results/campaigns/c005_official_group2/promptir.13",
       "runs/campaigns/c005_official_group2/promptir/13/best.pt", "promptir", "12k independent"),
    _m("ours_dim48_anchor", "c005", "results/campaigns/c005_official_group3/ours_dim48_anchor.13",
       "runs/campaigns/c005_official_group3/ours_dim48_anchor/13/best.pt", "siblingrestormer", "12k independent", is_ours=True),
    _m("ours_dim64_anchor", "c005", "results/campaigns/c005_official_group1/ours_dim64_anchor.13",
       "runs/campaigns/c005_official_group1/ours_dim64_anchor/13/best.pt", "siblingrestormer", "12k independent", is_ours=True),
    _m("airnet", "c005", "results/campaigns/c005_official_group1/airnet.13",
       "runs/campaigns/c005_official_group1/airnet/13/best.pt", "airnet", "12k independent"),
    _m("clearair", "c005", "results/campaigns/c005_official_group1/clearair.13",
       "runs/campaigns/c005_official_group1/clearair/13/best.pt", "clearair", "12k independent"),
    _m("dfpir", "c005", "results/campaigns/c005_official_group1/dfpir.13",
       "runs/campaigns/c005_official_group1/dfpir/13/best.pt", "dfpir", "12k independent"),
    _m("ffanet", "c005", "results/campaigns/c005_official_group1/ffanet.13",
       "runs/campaigns/c005_official_group1/ffanet/13/best.pt", "ffanet", "12k independent"),
    _m("grl", "c005", "results/campaigns/c005_official_group3/grl.13",
       "runs/campaigns/c005_official_group3/grl/13/best.pt", "grl", "12k independent"),
    _m("prenet", "c005", "results/campaigns/c005_official_group1/prenet.13",
       "runs/campaigns/c005_official_group1/prenet/13/best.pt", "prenet", "12k independent"),
    _m("r2r", "c005", "results/campaigns/c005_official_group1/r2r.13",
       "runs/campaigns/c005_official_group1/r2r/13/best.pt", "r2r", "12k independent"),
    _m("swinir", "c005", "results/campaigns/c005_official_group1/swinir.13",
       "runs/campaigns/c005_official_group1/swinir/13/best.pt", "swinir", "12k independent"),
    _m("transweather", "c005", "results/campaigns/c005_official_group1/transweather.13",
       "runs/campaigns/c005_official_group1/transweather/13/best.pt", "transweather", "12k independent"),
    _m("uformer", "c005", "results/campaigns/c005_official_group1/uformer.13",
       "runs/campaigns/c005_official_group1/uformer/13/best.pt", "uformer", "12k independent"),
    # ---- 12k 模块主线（仅含 test 结果的模型：M3 与 safe-refine） ----
    _m("M3_alcrb", "psnr_modules_12k", "results/psnr_modules_12k/M3_alcrb.13",
       "runs/psnr_modules_12k/M3_alcrb/13/best.pt", "siblingrestormer", "12k, formal, M3 ALCRB", is_ours=True),
    _m("m3_alcrb_safe_seed13", "psnr_safe_refine", "results/psnr_safe_refine/m3_alcrb_safe_seed13.13",
       "weights/psnr_safe_refine/m3_alcrb_safe_seed13/best.pt", "siblingrestormer", "12k, safe-refine (M3 init)", is_ours=True),
]

METHOD_BY_ID = {m["id"]: m for m in METHODS}
# 主表/图 1 聚焦方法
MAIN_METHODS = ("A0", "B1", "B2", "A5")
OURLIKE = ("A5", "B1")  # 图1 核心比较

# 汇总指标文件（结果 JSON 提供 aggregate/per_degradation 与 verifier 指纹）
def json_path(m: dict) -> Path:
    return ROOT / (m["result_prefix"] + ".test.json")

def csv_path(m: dict) -> Path:
    return ROOT / (m["result_prefix"] + ".test.csv")

def sources_path(m: dict) -> Path:
    return ROOT / (m["result_prefix"] + ".test.json.sources.jsonl")

def means_dir(m: dict) -> Path:
    return ROOT / (m["result_prefix"] + ".test.json.means")

if __name__ == "__main__":
    for m in METHODS:
        missing = [str(p) for p in (json_path(m), csv_path(m), sources_path(m), ROOT / m["ckpt"])
                   if not p.exists()]
        print(("OK  " if not missing else "MISS"), m["id"], (", ".join(missing) if missing else ""))
