"""Materialize the aligned PLAMD vari-grip pilot into a portable folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(stem: str) -> str:
    return hashlib.sha1(stem.encode("utf-8")).hexdigest()[:16]  # noqa: S324


def extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.index.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    archives: dict[str, zipfile.ZipFile] = {}
    portable_rows: list[dict[str, str]] = []
    clean_paths: dict[str, str] = {}
    try:
        for row in rows:
            source_key = safe_name(row["source_id"])
            degraded_ext = Path(row["degraded_member"]).suffix.lower()
            clean_ext = Path(row["clean_member"]).suffix.lower()
            degraded_relative = PurePosixPath(
                "images", "degraded", row["degradation"], row["subcategory"], source_key + degraded_ext
            )
            clean_relative = PurePosixPath(
                "images", "clean", row["subcategory"], source_key + clean_ext
            )

            for archive_name, member, relative in (
                (row["archive"], row["degraded_member"], degraded_relative),
                (row["clean_archive"], row["clean_member"], clean_relative),
            ):
                if archive_name not in archives:
                    archives[archive_name] = zipfile.ZipFile(args.dataset_root / archive_name)
                extract_member(archives[archive_name], member, args.output_root / Path(relative))

            clean_paths[row["source_id"]] = clean_relative.as_posix()
            portable_rows.append(
                {
                    "source_id": row["source_id"],
                    "source_key": source_key,
                    "subcategory": row["subcategory"],
                    "split": row["split"],
                    "degradation": row["degradation"],
                    "degraded_path": degraded_relative.as_posix(),
                    "clean_path": clean_relative.as_posix(),
                }
            )
    finally:
        for archive in archives.values():
            archive.close()

    metadata = args.output_root / "metadata"
    metadata.mkdir(exist_ok=True)
    index_path = metadata / "pilot_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(portable_rows[0]))
        writer.writeheader()
        writer.writerows(portable_rows)

    sources: dict[str, dict[str, object]] = {}
    for row in portable_rows:
        item = sources.setdefault(
            row["source_id"],
            {
                "source_id": row["source_id"],
                "source_key": row["source_key"],
                "subcategory": row["subcategory"],
                "split": row["split"],
                "clean_path": row["clean_path"],
                "degraded": {},
            },
        )
        item["degraded"][row["degradation"]] = row["degraded_path"]
    groups_path = metadata / "source_groups.json"
    groups_path.write_text(
        json.dumps(list(sources.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_paths = sorted((args.output_root / "images").rglob("*.*"))
    errors: list[str] = []
    formats: Counter[str] = Counter()
    bytes_total = 0
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.load()
                formats[(image.format or "unknown").lower()] += 1
            bytes_total += path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(args.output_root)}: {exc!r}")

    split_sources = Counter(item["split"] for item in sources.values())
    class_sources = Counter(item["subcategory"] for item in sources.values())
    audit = {
        "rows": len(portable_rows),
        "sources": len(sources),
        "unique_images": len(image_paths),
        "degradations": sorted({row["degradation"] for row in portable_rows}),
        "split_sources": dict(sorted(split_sources.items())),
        "class_sources": dict(sorted(class_sources.items())),
        "formats": dict(sorted(formats.items())),
        "image_bytes": bytes_total,
        "index_sha256": sha256(index_path),
        "groups_sha256": sha256(groups_path),
        "decode_errors": errors,
        "passed": (
            len(portable_rows) == 600
            and len(sources) == 100
            and len(image_paths) == 700
            and set(row["degradation"] for row in portable_rows) == set(DEGRADATIONS)
            and not errors
        ),
    }
    (metadata / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
