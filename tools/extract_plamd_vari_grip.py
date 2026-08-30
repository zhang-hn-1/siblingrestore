#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLAMD 全量 vari-grip 数据包提取工具（正式论文数据集，7 退化含 noise）。

用法（详见 tools/EXTRACT_GUIDE.md）：

  1) 扫描原始数据内部结构（确认 vari-grip 子树、子类别目录）：
     python extract_plamd_vari_grip.py --scan --zips blur.zip haze.zip ... snow.zip

  2) 正式提取 + 校验 + 打包：
     python extract_plamd_vari_grip.py --extract \
         --zips /data/plamd/*.zip \
         --outdir /data/out/plamd_vari_grip_2000_v1 \
         --split-seed 2026 --max-sources 2000

  若原始数据已解压为目录（含 blur/、original/ 等顶层目录），用 --raw-root 代替 --zips。

输出严格遵循 PLAMD_EXTRACTION_SPEC.md 契约，唯一扩展：退化列表加入 noise（7 种）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DEGRADATIONS = ["blur", "haze", "inpainting", "lowlight", "noise", "rain", "snow"]
ALLOWED_SUBCATEGORIES = ("good", "rust", "bird-nest")
DEFAULT_CATEGORY = "vari-grip"
DEFAULT_EXCLUDE = ("overallL_2K", "component_mix_1K")

# zip 内 {退化}/ 目录与 original/ 目录的顶层前缀
DEG_TOP = "degraded"
ORIG_TOP = "original"

ZIP_MD5_EXPECTED = {
    "blur.zip": "ec427a9adeda277070b907110a9f8cc7",
    "haze.zip": "30454d5062e40a097ca59694e020d040",
    "inpainting.zip": "9f4365c0bb7cf8f8f6f8588d88dc3250",
    "lowlight.zip": "3aa578bebbb3bde332d1dfe7cb2bc757",
    "noise.zip": "368caf8eeccc653f01beb0f6fb8649b9",
    "rain.zip": "dd20c8b12db666cfcf82e36c836238f6",
    "snow.zip": "b03b94e36e07074a01fbb9d992420263",
}


# ---------------------------------------------------------------------------
# 输入源抽象：zip 或解压目录，统一产出 (kind, name, bytes_or_path)
# ---------------------------------------------------------------------------
class ZipSource:
    def __init__(self, path: Path):
        self.path = path
        self.zf = zipfile.ZipFile(path)

    def entries(self):
        """yield 规范化后的相对路径（去 ./ 前缀）。"""
        for name in self.zf.namelist():
            name = name.replace("\\", "/").lstrip("./")
            if not name or name.endswith("/"):
                continue
            yield name

    def read(self, name: str) -> bytes:
        return self.zf.read(name)

    def close(self):
        self.zf.close()


class DirSource:
    def __init__(self, root: Path):
        self.root = root

    def entries(self):
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                yield str(path.relative_to(self.root))

    def read(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def close(self):
        pass


def open_sources(zips, raw_root):
    sources = []
    if raw_root:
        sources.append(DirSource(Path(raw_root)))
    if zips:
        for zp in zips:
            sources.append(ZipSource(Path(zp)))
    if not sources:
        raise SystemExit("必须提供 --zips 或 --raw-root")
    return sources


def first_segment(name: str) -> str:
    return name.split("/", 1)[0]


def seg_after_first(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else ""


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
def cmd_scan(args) -> None:
    sources = open_sources(args.zips, args.raw_root)
    try:
        for src in sources:
            label = str(getattr(src, "path", getattr(src, "root", "?")))
            print(f"\n===== {label} =====")
            top = Counter(first_segment(n) for n in src.entries())
            for t, c in sorted(top.items()):
                print(f"  [dir] {t}/  ({c} files)")
            # 找退化根与 original 下的类别目录
            for topdir in [d for d in top if d not in ("original",)]:
                cat_count = Counter()
                samples = defaultdict(list)
                for name in src.entries():
                    if first_segment(name) != topdir:
                        continue
                    rest = seg_after_first(name)
                    cat = rest.split("/", 1)[0] if "/" in rest else rest
                    cat_count[cat] += 1
                    if len(samples[cat]) < 3:
                        samples[cat].append(name)
                print(f"\n  -- {topdir}/ 下的类别 --")
                for cat, c in sorted(cat_count.items()):
                    print(f"    {cat}/  {c} files   e.g. {samples[cat][0]}")
                    # 类别内部的第二层（子类别/严重度/平铺）
                    if cat in (args.category,):
                        sub = Counter()
                        for name in src.entries():
                            if first_segment(name) != topdir or seg_after_first(name).split("/", 1)[0] != cat:
                                continue
                            rest = seg_after_first(seg_after_first(name))
                            sub["(flat)" if "/" not in rest else rest.split("/", 1)[0]] += 1
                        for s2, c2 in sorted(sub.items()):
                            print(f"      └─ {s2}: {c2} files")
            # original 侧是否也有对应类别
            orig_cats = Counter()
            for name in src.entries():
                if first_segment(name) == ORIG_TOP:
                    rest = seg_after_first(name)
                    orig_cats[rest.split("/", 1)[0] if "/" in rest else rest] += 1
            print(f"\n  -- {ORIG_TOP}/ 下的类别 --")
            for cat, c in sorted(orig_cats.items()):
                print(f"    {cat}/  {c} files")
    finally:
        for src in sources:
            src.close()
    print("\n扫描完成。确认 vari-grip 的路径形态后，用 --extract 正式提取。")


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
def detect_subcategory(rel_path: str, subcat_map: dict) -> str:
    """从类别目录下的相对路径推断子类别（good/rust/bird-nest）。"""
    parts = rel_path.split("/")
    if parts[0] in subcat_map:  # 目录名显式映射
        return subcat_map[parts[0]]
    if parts[0] in ALLOWED_SUBCATEGORIES:
        return parts[0]
    stem = Path(rel_path).stem.lower()
    for key, sub in (("rust", "rust"), ("corr", "rust"), ("nest", "bird-nest"),
                     ("bird", "bird-nest"), ("good", "good"), ("normal", "good"), ("intact", "good")):
        if key in stem:
            return sub
    return "good"


def collect_candidates(sources, category: str, exclude: tuple, subcat_map: dict):
    """按退化收集候选路径。返回 {degradation: {rel_under_category: subcategory}}。"""
    by_deg = {d: {} for d in DEGRADATIONS}
    clean_candidates = {}
    for src in sources:
        for name in src.entries():
            top = first_segment(name)
            if top == ORIG_TOP:
                rest = seg_after_first(name)
                if rest.split("/", 1)[0] == category:
                    clean_candidates[seg_after_first(rest)] = None
                continue
            if top not in DEGRADATIONS or not seg_after_first(name):
                continue
            rest = seg_after_first(name)
            first = rest.split("/", 1)[0]
            if first != category:
                continue
            rel = seg_after_first(rest)
            if not rel:
                continue
            if any(rel.startswith(x + "/") or rel == x for x in exclude):
                continue
            by_deg[top][rel] = detect_subcategory(rel, subcat_map)
    return by_deg, clean_candidates


def build_sources(by_deg, clean_candidates, category: str):
    """求并集：每个 source 需要 7 退化 + clean 全齐。返回 (ok, skipped_reasons)。"""
    union = set(by_deg[DEGRADATIONS[0]])
    for d in DEGRADATIONS[1:]:
        union |= set(by_deg[d])
    sources, skipped = [], Counter()
    for rel in sorted(union):
        subcats = {by_deg[d].get(rel) for d in DEGRADATIONS}
        subcats.discard(None)
        missing = [d for d in DEGRADATIONS if rel not in by_deg[d]]
        if missing:
            skipped[f"missing_degradations:{','.join(missing)}"] += 1
            continue
        if rel not in clean_candidates:
            skipped["missing_clean"] += 1
            continue
        sub = next(iter(subcats)) if len(subcats) == 1 else sorted(subcats)[0]
        sources.append({"rel": rel, "subcategory": sub,
                        "filename": Path(rel).name, "stem": Path(rel).stem})
    return sources, skipped


def stratify_cap(sources, max_sources: int, rng: random.Random):
    """按子类别分层封顶（保类别平衡）。"""
    if not max_sources or len(sources) <= max_sources:
        return sources
    by_sub = defaultdict(list)
    for s in sources:
        by_sub[s["subcategory"]].append(s)
    per = {sub: max(1, round(len(v) * max_sources / len(sources))) for sub, v in by_sub.items()}
    picked = []
    for sub, v in by_sub.items():
        rng.shuffle(v)
        picked.extend(v[: per[sub]])
    return picked


def stratified_split(sources, ratios, rng: random.Random):
    """source 级分层划分（按子类别），返回 {split: [source]}。"""
    by_sub = defaultdict(list)
    for s in sources:
        by_sub[s["subcategory"]].append(s)
    out = {k: [] for k in ("train", "val", "test")}
    for sub, v in by_sub.items():
        v = sorted(v, key=lambda s: s["rel"])
        rng.shuffle(v)
        n_train = round(len(v) * ratios[0])
        n_val = round(len(v) * ratios[1])
        if n_val == 0 and len(v) >= 3:  # 子类别样本少时保证 val/test 至少 1 个
            n_val = 1
        out["train"].extend(v[:n_train])
        out["val"].extend(v[n_train:n_train + n_val])
        out["test"].extend(v[n_train + n_val:])
    for k in out:
        out[k].sort(key=lambda s: s["rel"])
    return out


def extract_and_verify(src, name, out_path: Path, check_size: bool):
    """读取一个条目并写出；返回 (size, ok, error)。"""
    try:
        data = src.read(name)
        size = None
        if check_size:
            try:
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    size = im.size
            except Exception as exc:
                return None, False, f"decode_error:{exc}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return size, True, None
    except Exception as exc:
        return None, False, f"read_error:{exc}"


def cmd_extract(args) -> None:
    sources = open_sources(args.zips, args.raw_root)
    outdir = Path(args.outdir)
    images_dir = outdir / "images"
    meta_dir = outdir / "metadata"
    if images_dir.exists() or meta_dir.exists():
        raise SystemExit(f"输出目录已存在：{outdir}（先移走再重跑，避免混入旧数据）")
    report = {"parameters": {"category": args.category, "exclude_subtrees": list(DEFAULT_EXCLUDE),
                             "degradations": DEGRADATIONS, "split_ratios": args.split_ratios,
                             "split_seed": args.split_seed, "max_sources": args.max_sources},
              "inputs": [], "checks": [], "skipped": {}, "errors": []}
    rng = random.Random(args.split_seed)
    try:
        by_deg, clean_candidates = collect_candidates(sources, args.category, DEFAULT_EXCLUDE, args.subcat_map)
        for src in sources:
            report["inputs"].append(str(getattr(src, "path", getattr(src, "root", "?"))))
        for d in DEGRADATIONS:
            report.setdefault("per_degradation_candidates", {})[d] = len(by_deg[d])
        report["clean_candidates"] = len(clean_candidates)

        all_sources, skipped = build_sources(by_deg, clean_candidates, args.category)
        report["skipped"] = dict(skipped)
        all_sources = stratify_cap(all_sources, args.max_sources, rng)
        splits = stratified_split(all_sources, args.split_ratios, rng)
        report["candidate_sources"] = len(all_sources)
        report["split_sources_planned"] = {k: len(v) for k, v in splits.items()}
        report["class_sources_planned"] = dict(Counter(s["subcategory"] for s in all_sources))

        # 逐 source 写图 + 校验尺寸
        rows, groups, errors = [], [], []
        per_split = {}
        for split in ("train", "val", "test"):
            for s in splits[split]:
                key = s["stem"]
                sub = s["subcategory"]
                # 读 clean（同时算 source_key 内容哈希）
                clean_name = f"{ORIG_TOP}/{args.category}/{s['rel']}"
                clean_data = None
                for src in sources:
                    try:
                        clean_data = src.read(clean_name)
                        break
                    except (KeyError, FileNotFoundError):
                        continue
                if clean_data is None:
                    errors.append(f"{s['rel']}: clean 读取失败")
                    continue
                source_key = hashlib.sha256(clean_data).hexdigest()[:16]
                source_id = f"{sub}|{s['stem']}"
                if any(g["source_id"] == source_id for g in groups):
                    source_id = f"{sub}|{s['stem']}-{source_key[:8]}"
                sizes = {}
                ok = True
                # clean
                clean_out = images_dir / "clean" / sub / f"{source_key}.jpg"
                size, ok_c, err = extract_and_verify(sources[0], clean_name, clean_out, args.check_size)
                if not ok_c:
                    errors.append(f"{source_id}: clean {err}")
                    ok = False
                else:
                    sizes["clean"] = size
                # 7 退化
                for d in DEGRADATIONS:
                    deg_name = f"{d}/{args.category}/{s['rel']}"
                    deg_out = images_dir / "degraded" / d / sub / f"{source_key}.jpg"
                    size, ok_d, err = extract_and_verify(sources[0], deg_name, deg_out, args.check_size)
                    if not ok_d:
                        errors.append(f"{source_id}: {d} {err}")
                        ok = False
                    else:
                        sizes[d] = size
                if args.check_size:
                    uniq = set(sizes.values())
                    if len(uniq) != 1:
                        errors.append(f"{source_id}: 尺寸不一致 {sizes}")
                        ok = False
                if not ok:
                    # 回滚本 source 已写文件
                    for p in [clean_out, *(images_dir / "degraded" / d / sub / f"{source_key}.jpg" for d in DEGRADATIONS)]:
                        if p.exists():
                            p.unlink()
                    continue
                groups.append({"source_id": source_id, "source_key": source_key,
                               "subcategory": sub, "split": split,
                               "clean_path": str(clean_out.relative_to(outdir)),
                               "degraded": {d: str((images_dir / "degraded" / d / sub / f"{source_key}.jpg").relative_to(outdir)) for d in DEGRADATIONS}})
                for d in DEGRADATIONS:
                    rows.append({"source_id": source_id, "source_key": source_key, "subcategory": sub,
                                 "split": split, "degradation": d,
                                 "degraded_path": groups[-1]["degraded"][d], "clean_path": groups[-1]["clean_path"]})
                per_split.setdefault(split, []).append(source_id)

        groups.sort(key=lambda g: g["source_id"])
        rows.sort(key=lambda r: (r["source_id"], r["degradation"]))
        n_sources = len(groups)
        n_images = n_sources * 8

        # 写 metadata
        meta_dir.mkdir(parents=True, exist_ok=True)
        groups_path = meta_dir / "source_groups.json"
        groups_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_path = meta_dir / "pilot_index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        groups_sha = hashlib.sha256(groups_path.read_bytes()).hexdigest()
        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
        audit = {"rows": len(rows), "sources": n_sources, "unique_images": n_images,
                 "degradations": DEGRADATIONS,
                 "split_sources": dict(Counter(g["split"] for g in groups)),
                 "class_sources": dict(Counter(g["subcategory"] for g in groups)),
                 "formats": {"jpeg": n_images},
                 "index_sha256": index_sha, "groups_sha256": groups_sha,
                 "decode_errors": errors, "passed": False}
        (meta_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # 校验清单（spec 第 5 节）
        checks = [
            ("clean 子目录齐全", {sub for g in groups for sub in [g["subcategory"]]} == set(ALLOWED_SUBCATEGORIES)),
            ("文件计数 8×sources", len(rows) == 7 * n_sources and n_images == 8 * n_sources),
            ("每 source 7 退化齐全", all(set(g["degraded"]) == set(DEGRADATIONS) for g in groups)),
            ("source_id 唯一", len({g["source_id"] for g in groups}) == n_sources),
            ("source_key 唯一", len({g["source_key"] for g in groups}) == n_sources),
            ("split 无泄漏", len({g["source_id"] for g in groups}) == n_sources),
            ("尺寸一致性（已校验）", not errors),
            ("无解码错误", not errors),
        ]
        report["checks"] = [{"name": n, "passed": bool(p)} for n, p in checks]
        audit["passed"] = all(p for _, p in checks)
        (meta_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["audit_passed"] = audit["passed"]
        report["errors"] = errors

        # 打包
        tarball = outdir.with_suffix(".tar.gz")
        with tarfile.open(tarball, "w:gz") as tf:
            for root in (images_dir, meta_dir):
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        tf.add(p, arcname=f"{outdir.name}/{p.relative_to(outdir)}")
        report["tarball"] = str(tarball)
        report["tarball_bytes"] = tarball.stat().st_size

        report_path = outdir.with_name(outdir.name + "_extraction_report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n完成：{n_sources} sources / {len(rows)} 行索引 / {n_images} 图 / {tarball}")
    finally:
        for src in sources:
            src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PLAMD vari-grip 全量提取（7 退化含 noise）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zips", nargs="*", type=str, default=None, help="7 个退化 zip 的路径")
    common.add_argument("--raw-root", type=str, default=None, help="或：已解压的原始数据目录")
    common.add_argument("--category", type=str, default=DEFAULT_CATEGORY)
    p_scan = sub.add_parser("scan", aliases=["--scan"], parents=[common], help="扫描 zip/目录内部结构")
    p_scan.set_defaults(func=cmd_scan)
    p_ext = sub.add_parser("extract", aliases=["--extract"], parents=[common], help="提取 + 校验 + 打包")
    p_ext.add_argument("--outdir", type=str, required=True)
    p_ext.add_argument("--split-seed", type=int, default=2026)
    p_ext.add_argument("--split-ratios", type=str, default="0.70,0.15,0.15")
    p_ext.add_argument("--max-sources", type=int, default=0, help="封顶（按子类别分层），0=不限")
    p_ext.add_argument("--subcategory-map", type=str, default="", help="JSON：目录名→子类别，如 '{\"corrosion\":\"rust\"}'")
    p_ext.add_argument("--no-size-check", action="store_true", help="跳过尺寸一致性校验（快）")
    p_ext.set_defaults(func=cmd_extract)
    args = parser.parse_args()
    if hasattr(args, "split_ratios"):
        args.split_ratios = [float(x) for x in args.split_ratios.split(",")]
        assert len(args.split_ratios) == 3 and abs(sum(args.split_ratios) - 1.0) < 1e-6
        args.subcat_map = json.loads(args.subcategory_map) if args.subcategory_map else {}
        args.check_size = not args.no_size_check
    args.func(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--scan", "--extract"):
        sys.argv[1] = sys.argv[1].lstrip("-")
    main()
