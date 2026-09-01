from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from siblingrestore.inference import restore_tiled
from siblingrestore.model import SiblingRestormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]
    model = SiblingRestormer(
        dim=int(config["dim"]),
        blocks_per_level=tuple(config["blocks_per_level"]),
        heads=tuple(config["heads"]),
        projection_dim=int(config["projection_dim"]),
        degradation_conditioned=bool(config.get("degradation_conditioned", False)),
        identity_mode=config.get("identity_mode", None),
        identity_source_count=int(config.get("identity_source_count", 71)),
        refinement_type=str(config.get("refinement_type", "none")),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    with Image.open(args.input) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    height, width = tensor.shape[-2:]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode).unsqueeze(0).to(device)
    with torch.no_grad():
        restored = restore_tiled(model, tensor, args.tile_size, args.tile_overlap)
    restored = restored[0, :, :height, :width].clamp(0, 1).mul(255).byte()
    output = restored.permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output)}))


if __name__ == "__main__":
    main()
