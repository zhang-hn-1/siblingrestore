from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from siblingrestore.model import SiblingRestormer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "plamd_vari_grip_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads((DATA / "metadata" / "audit.json").read_text(encoding="utf-8"))
    image_count = sum(1 for path in (DATA / "images").rglob("*") if path.is_file())
    checks = {
        "audit_passed": audit["passed"] is True,
        "image_count_700": image_count == 700,
        "index_sha256": sha256(DATA / "metadata" / "pilot_index.csv")
        == audit["index_sha256"],
        "groups_sha256": sha256(DATA / "metadata" / "source_groups.json")
        == audit["groups_sha256"],
    }
    model = SiblingRestormer(
        dim=8, blocks_per_level=(1, 1, 1, 1, 1), heads=(1, 2, 4), projection_dim=32
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64))["restored"]
    checks["model_forward_64x64"] = tuple(output.shape) == (1, 3, 64, 64)
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "images": image_count,
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
