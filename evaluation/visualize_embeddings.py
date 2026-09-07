from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from siblingrestore.data import _read_rgb
from siblingrestore.inference import restore_tiled
from siblingrestore.verifier import load_verifier_checkpoint
from train_cross_verifier import CrossVerifier

DEGS = ("clean", "blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/identity_evaluation/visualization"))
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device in ("auto", "cuda") and __import__("torch").cuda.is_available() else "cpu"
    import torch
    root = ROOT / "data/plamd_sfr_v1"
    groups = [g for g in json.loads((root / "metadata/source_groups.json").read_text()) if g["split"] == "test"][:args.sources]
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    verifier, _ = load_verifier_checkpoint(ROOT / "runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt", torch.device(device))
    rows = []
    for group in groups:
        sid = group["source_id"]
        with torch.no_grad():
            clean = _read_rgb(root / group["clean_path"]).unsqueeze(0).to(device)
            clean_embedding = verifier.embed(clean).cpu().numpy()[0]
        rows.append({"source_id": sid, "query_type": "clean", "degradation": "clean", "embedding": clean_embedding})
        for d in DEGS[1:]:
            with torch.no_grad():
                image = _read_rgb(root / group["degraded"][d]).unsqueeze(0).to(device)
                emb = verifier.embed(image).cpu().numpy()[0]
            rows.append({"source_id": sid, "query_type": "degraded", "degradation": d, "embedding": emb})
    matrix = np.stack([r["embedding"] for r in rows])
    method = "pca"
    coordinates = PCA(n_components=2, random_state=13).fit_transform(matrix)
    for row, xy in zip(rows, coordinates): row.update({"x": float(xy[0]), "y": float(xy[1]), "embedding": None})
    with (output / "embedding_2d.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "query_type", "degradation", "x", "y"]); writer.writeheader(); writer.writerows(rows)
    plt.figure(figsize=(8, 6))
    source_colors = {sid: i for i, sid in enumerate(sorted({r["source_id"] for r in rows}))}
    for row in rows:
        plt.scatter(row["x"], row["y"], c=[source_colors[row["source_id"]]], marker="o" if row["query_type"] == "degraded" else "*", cmap="tab10", s=55, alpha=0.85)
    plt.title(f"Frozen verifier identity embedding ({method.upper()} fallback)"); plt.tight_layout()
    plt.savefig(output / "degraded_embedding.png", dpi=300); plt.savefig(output / "degraded_embedding.pdf"); plt.close()
    (output / "method.json").write_text(json.dumps({"method": method, "num_sources": len(groups), "verifier": "runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt"}, indent=2) + "\n")
    print(json.dumps({"method": method, "rows": len(rows), "output": str(output)}, indent=2))

if __name__ == "__main__": main()
