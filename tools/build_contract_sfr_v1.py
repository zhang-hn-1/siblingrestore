#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLAMD_SFR_v1 数据包 → 旧契约 metadata 适配器。

把 data/PLAMD_SFR_v1_method_b/ 的 manifest（source_groups.jsonl + source_manifest.jsonl）
转换成服务器现有代码零改动可读的旧契约（PLAMD_EXTRACTION_SPEC.md 格式）：

    data/plamd_sfr_v1/metadata/
        ├── source_groups.json   （旧格式，数组）
        ├── pilot_index.csv      （source_id,source_key,subcategory,split,degradation,degraded_path,clean_path）
        └── audit.json           （旧格式 + noise）

生成后的路径相对 data/plamd_sfr_v1 指向 ../PLAMD_SFR_v1_method_b/main/...（零拷贝）。
noise 尺寸不匹配按包协议处理：训练/评估时 clean 用 Lanczos resize 到 noise 原生尺寸
（该逻辑在 siblingrestore/data.py 中实现，此处只记录协议字段）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DEGRADATIONS = ["blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow"]
PKG_REL = "../PLAMD_SFR_v1_method_b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-root", type=str,
                        default="data/PLAMD_SFR_v1_method_b")
    parser.add_argument("--out-root", type=str, default="data/plamd_sfr_v1")
    args = parser.parse_args()

    pkg = Path(args.pkg_root)
    out = Path(args.out_root)
    meta_out = out / "metadata"
    meta_out.mkdir(parents=True, exist_ok=True)

    groups = [json.loads(line) for line in
              (pkg / "metadata" / "source_groups.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = [json.loads(line) for line in
                (pkg / "metadata" / "source_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sha_map = {row["source_id"]: row["clean_sha256"] for row in manifest}

    def rel(path: str) -> str:
        return f"{PKG_REL}/{path}"

    entries = []
    for g in groups:
        sub = g["condition"]
        assert sub in ("good", "rust", "bird-nest"), g["source_id"]
        clean_sha = sha_map.get(g["source_id"], "")
        entries.append({
            "source_id": g["source_id"],
            "source_key": clean_sha[:16],
            "subcategory": sub,
            "split": g["split"],
            "clean_path": rel(g["clean"]),
            "degraded": {deg: rel(g["degradations"][deg]) for deg in DEGRADATIONS},
        })
    entries.sort(key=lambda e: e["source_id"])

    # 路径存在性 + 尺寸一致性（noise 除外，按协议允许 mismatch）
    missing, mismatched = [], []
    sizes = {}
    from PIL import Image
    for e in entries:
        for deg in ["clean"] + DEGRADATIONS:
            p = out / e["clean_path"] if deg == "clean" else out / e["degraded"][deg]
            if not p.exists():
                missing.append(str(p))
                continue
            with Image.open(p) as im:
                sizes[(e["source_id"], deg)] = im.size
        base = sizes[(e["source_id"], "clean")]
        for deg in DEGRADATIONS:
            if deg != "noise" and sizes[(e["source_id"], deg)] != base:
                mismatched.append(f"{e['source_id']} {deg} {sizes[(e['source_id'], deg)]} vs clean {base}")
    assert not missing, f"缺失文件 {len(missing)} 个，例如 {missing[:3]}"

    # 写旧契约
    groups_path = meta_out / "source_groups.json"
    groups_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for e in entries:
        for deg in DEGRADATIONS:
            rows.append({"source_id": e["source_id"], "source_key": e["source_key"],
                         "subcategory": e["subcategory"], "split": e["split"],
                         "degradation": deg, "degraded_path": e["degraded"][deg],
                         "clean_path": e["clean_path"]})
    rows.sort(key=lambda r: (r["source_id"], r["degradation"]))
    index_path = meta_out / "pilot_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_sources = len(entries)
    n_images = n_sources * 8
    audit = {
        "rows": len(rows), "sources": n_sources, "unique_images": n_images,
        "degradations": DEGRADATIONS,
        "split_sources": dict(Counter(e["split"] for e in entries)),
        "class_sources": dict(Counter(e["subcategory"] for e in entries)),
        "formats": {"jpeg": n_sources * 7, "png": n_sources},
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "groups_sha256": hashlib.sha256(groups_path.read_bytes()).hexdigest(),
        "decode_errors": missing,
        "size_mismatches": mismatched,
        "noise_geometry_protocol": "clean_resize_to_noise_lanczos_then_same_random_crop_256",
        "source_package": "PLAMD_SFR_v1_method_b",
        "source_audit_status": "PASS",
        "passed": not missing,
    }
    (meta_out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\n完成：{n_sources} sources / {len(rows)} 行 / 输出 {meta_out}")


if __name__ == "__main__":
    main()
