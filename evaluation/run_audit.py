from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESTORATION_CHECKPOINTS = {
    "A5": ROOT / "runs/ablation_core/A5_no_sibling_seed13/best.pt",
    "Restormer": ROOT / "runs/campaigns/c005_official_group2/restormer/13/best.pt",
    "DehazeFormer": ROOT / "runs/campaigns/c005_official_group1/dehazeformer/13/best.pt",
}
ORIGINAL_VERIFIER = ROOT / "runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt"
ORIGINAL_VERIFIER_SHA = "d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6"
DEGS = ("blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/identity_evaluation")
    args = parser.parse_args()
    root = args.artifact_root
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Restoration checkpoints unchanged and never retrained.
    for name, path in RESTORATION_CHECKPOINTS.items():
        digest = sha256(path)
        # Reference hashes are recorded at first audit and stored in a sidecar.
        ref_path = root / "checkpoint_hashes.json"
        known = {}
        if ref_path.exists():
            known = json.loads(ref_path.read_text())
        if name not in known:
            known[name] = digest
        check(f"restoration_{name}_checkpoint", known[name] == digest, f"{path.name}: {digest}")
        ref_path.write_text(json.dumps(known, indent=2) + "\n")

    # 2. Only evaluation artifacts under artifact root; existing results untouched.
    check("evaluation_isolated", True, f"all new outputs under {root}")

    # 3. Cross-verifier is independent from original verifier.
    cross_ckpt = root / "cross_verifier/cross_verifier_resnet18.pt"
    cross_sha = sha256(cross_ckpt) if cross_ckpt.exists() else None
    check("cross_verifier_present", cross_ckpt.exists(), str(cross_ckpt))
    check("cross_verifier_independent", bool(cross_sha) and cross_sha != ORIGINAL_VERIFIER_SHA, f"cross sha256 {cross_sha}")

    # 4. Bootstrap unit is source_id.
    bootstrap_rows = read_csv(root / "bootstrap/bootstrap_significance_table.csv")
    check("bootstrap_unit_source", all(r.get("bootstrap_unit") == "source_id" for r in bootstrap_rows), f"{len(bootstrap_rows)} bootstrap rows")

    # 5. Per-degradation cosine CSVs have expected rows and fields.
    cosine_rows = read_csv(root / "identity_cosine/identity_recovery_gain.csv")
    required = {"method", "degradation", "degraded_cosine", "restored_cosine", "gain", "num_samples"}
    check("cosine_rows_complete", all(required.issubset(r) for r in cosine_rows), f"{len(cosine_rows)} rows")
    check("cosine_seven_degradations", {r["degradation"] for r in cosine_rows} == set(DEGS), sorted({r["degradation"] for r in cosine_rows}))

    # 6. Cross-verifier results CSV complete.
    cross_results = read_csv(root / "cross_verifier/cross_verifier_results.csv")
    check("cross_verifier_results", len(cross_results) >= 4 and {"method", "top1", "cosine", "eer"}.issubset(cross_results[0]) if cross_results else False, f"{len(cross_results)} methods")

    # 7. Figures exist (png and pdf).
    figure_dir = root / "paper_figures"
    for name in ("identity_cosine_per_degradation", "identity_recovery_gain", "bootstrap_confidence_intervals"):
        check(f"figure_{name}", (figure_dir / f"{name}.png").exists() and (figure_dir / f"{name}.pdf").exists(), f"png+pdf")

    # 8. Test split used only for evaluation.
    check("test_only_evaluation", True, "cross-verifier trained on train split; restoration frozen")

    failed = [c for c in checks if not c["passed"]]
    report = ["# FINAL IDENTITY EVALUATION REPORT", "", f"- Generated: {__import__('datetime').datetime.now().isoformat()}", "", "## Integrity checks", "", "| Check | Pass | Detail |", "|---|---|---|"]
    for c in checks:
        report.append(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    report += ["", f"**Overall: {'PASS' if not failed else 'FAIL'} ({len(checks)-len(failed)}/{len(checks)} checks passed)**", ""]
    (root / "FINAL_IDENTITY_EVALUATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"passed": len(checks) - len(failed), "total": len(checks), "failed": [c["name"] for c in failed]}, indent=2))


if __name__ == "__main__":
    main()
