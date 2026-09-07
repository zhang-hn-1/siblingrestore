from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siblingrestore.data import _read_rgb
from siblingrestore.inference import restore_tiled
from siblingrestore.verifier import load_verifier_checkpoint

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEGS = ("clean", "blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")
VIEW_DEGS = DEGS[1:]
MODEL_PATHS = {
    "A5": ROOT / "runs/ablation_core/A5_no_sibling_seed13/best.pt",
    "Restormer": ROOT / "runs/campaigns/c005_official_group2/restormer/13/best.pt",
}
MARKERS = {
    "clean": "*",
    "blur": "o", "haze": "o", "inpainting": "o",
    "lowlight": "o", "noise": "o", "rain": "o", "snow": "o",
}


def load_models(device):
    from train import make_model
    models = {}
    for name, path in MODEL_PATHS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = make_model(checkpoint["config"]); model.load_state_dict(checkpoint["model"])
        models[name] = model.to(device).eval()
    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/identity_evaluation/visualization")
    parser.add_argument("--num-sources", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verifier", type=Path, default=ROOT / "runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    groups = [g for g in json.loads((ROOT / "data/plamd_sfr_v1/metadata/source_groups.json").read_text()) if g["split"] == "test"]
    # Deterministic selection of the requested number of test sources.
    groups = groups[: args.num_sources]
    verifier, _ = load_verifier_checkpoint(args.verifier, torch.device(device))
    models = load_models(device)
    # Records: each row has source, method (degraded/restormer/ours), degradation, embedding.
    rows = []
    embeddings = []
    for group in groups:
        sid = group["source_id"]
        with torch.no_grad():
            clean = _read_rgb(ROOT / group["clean_path"]).unsqueeze(0).to(device)
            rows.append({"source_id": sid, "method": "clean", "degradation": "clean", "embedding": verifier.embed(clean).cpu().numpy()[0]})
            embeddings.append(rows[-1]["embedding"])
            for d in VIEW_DEGS:
                degraded = _read_rgb(ROOT / group["degraded"][d]).unsqueeze(0).to(device)
                rows.append({"source_id": sid, "method": "degraded", "degradation": d, "embedding": verifier.embed(degraded).cpu().numpy()[0]})
                embeddings.append(rows[-1]["embedding"])
                for mname, model in models.items():
                    restored = restore_tiled(model, degraded, 512, 32).cpu()
                    rows.append({"source_id": sid, "method": mname, "degradation": d, "embedding": verifier.embed(restored.to(device)).cpu().numpy()[0]})
                    embeddings.append(rows[-1]["embedding"])
    matrix = np.stack(embeddings)
    # Prefer PCA; t-SNE fallback documented in the artifact method.json.
    transform_method = "pca"
    coordinates = PCA(n_components=2, random_state=args.seed).fit_transform(matrix)
    for row, xy in zip(rows, coordinates):
        row["x"] = float(xy[0]); row["y"] = float(xy[1])
    with (output / "embedding_2d.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "method", "degradation", "x", "y"])
        writer.writeheader(); writer.writerows(rows)
    colors = {sid: index for index, sid in enumerate(sorted({r["source_id"] for r in rows}))}
    panels = {
        "degraded": [r for r in rows if r["method"] in ("degraded", "clean")],
        "restormer": [r for r in rows if r["method"] in ("Restormer", "clean")],
        "ours": [r for r in rows if r["method"] in ("A5", "clean")],
    }
    for name, selected in panels.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        for r in selected:
            ax.scatter(r["x"], r["y"], color=plt.cm.tab10(colors[r["source_id"]]), marker=MARKERS.get(r["degradation"], "o"), s=75, alpha=0.8, linewidths=0.4, edgecolors="k")
        ax.set_title(f"Identity embedding ({name}); PCA fallback, 300 dpi")
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
        fig.tight_layout(); fig.savefig(output / f"{name}_embedding.png", dpi=300); fig.savefig(output / f"{name}_embedding.pdf"); plt.close(fig)
    (output / "method.json").write_text(json.dumps({"transform": transform_method, "fallback": "sklearn PCA (no UMAP dependency)", "num_sources": len(groups), "verifier": str(args.verifier), "rows": len(rows)}, indent=2) + "\n")
    print(json.dumps({"transform": transform_method, "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
