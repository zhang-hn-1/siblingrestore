from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEGRADATIONS = ("blur", "haze", "inpainting", "lowlight", "rain", "snow")
DEGRADATION_TO_ID = {name: index for index, name in enumerate(DEGRADATIONS)}
CLASSES = ("good", "rust", "bird-nest")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def _read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _reflect_pad(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, height, width = image.shape
    pad_h = max(target_h - height, 0)
    pad_w = max(target_w - width, 0)
    if pad_h == 0 and pad_w == 0:
        return image
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode)


def paired_train_crop(
    images: list[torch.Tensor], crop_size: int, rng: random.Random
) -> list[torch.Tensor]:
    heights = {int(image.shape[-2]) for image in images}
    widths = {int(image.shape[-1]) for image in images}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError(f"unaligned sibling shapes: heights={heights}, widths={widths}")
    height, width = heights.pop(), widths.pop()
    target_h, target_w = max(height, crop_size), max(width, crop_size)
    images = [_reflect_pad(image, target_h, target_w) for image in images]
    top = rng.randint(0, target_h - crop_size)
    left = rng.randint(0, target_w - crop_size)
    return [image[:, top : top + crop_size, left : left + crop_size] for image in images]


def paired_eval_pad(
    degraded: torch.Tensor, clean: torch.Tensor, multiple: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if degraded.shape != clean.shape:
        raise ValueError(f"pair shape mismatch: {degraded.shape} vs {clean.shape}")
    _, height, width = degraded.shape
    target_h = ((height + multiple - 1) // multiple) * multiple
    target_w = ((width + multiple - 1) // multiple) * multiple
    mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    mask[:, :height, :width] = 1.0
    return (
        _reflect_pad(degraded, target_h, target_w),
        _reflect_pad(clean, target_h, target_w),
        mask,
    )


class PairDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        crop_size: int | None,
        seed: int,
        max_items: int = 0,
    ) -> None:
        self.root = Path(root)
        with (self.root / "metadata" / "pilot_index.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if max_items:
            self.rows = self.rows[:max_items]
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        degraded = _read_rgb(self.root / row["degraded_path"])
        clean = _read_rgb(self.root / row["clean_path"])
        if self.crop_size is not None:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            degraded, clean = paired_train_crop([degraded, clean], self.crop_size, rng)
            mask = torch.ones((1, self.crop_size, self.crop_size), dtype=torch.float32)
        else:
            degraded, clean, mask = paired_eval_pad(degraded, clean)
        return {
            "degraded": degraded,
            "clean": clean,
            "mask": mask,
            "degradation_id": torch.tensor(DEGRADATION_TO_ID[row["degradation"]]),
            "degradation": row["degradation"],
            "class_id": torch.tensor(CLASS_TO_ID[row["subcategory"]]),
            "source_id": row["source_id"],
        }


class SiblingGroupDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        crop_size: int,
        sibling_count: int,
        seed: int,
        max_sources: int = 0,
    ) -> None:
        self.root = Path(root)
        groups = json.loads(
            (self.root / "metadata" / "source_groups.json").read_text(encoding="utf-8")
        )
        self.groups = [group for group in groups if group["split"] == split]
        if max_sources:
            self.groups = self.groups[:max_sources]
        self.crop_size = crop_size
        self.sibling_count = sibling_count
        self.seed = seed
        self.epoch = 0
        if not 2 <= sibling_count <= len(DEGRADATIONS):
            raise ValueError("sibling_count must be between 2 and 6")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, object]:
        group = self.groups[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        available = sorted(group["degraded"])
        selected = rng.sample(available, self.sibling_count)
        degraded = [_read_rgb(self.root / group["degraded"][name]) for name in selected]
        clean = _read_rgb(self.root / group["clean_path"])
        cropped = paired_train_crop([*degraded, clean], self.crop_size, rng)
        return {
            "degraded": torch.stack(cropped[:-1]),
            "clean": cropped[-1],
            "degradation_ids": torch.tensor(
                [DEGRADATION_TO_ID[name] for name in selected], dtype=torch.long
            ),
            "degradations": selected,
            "class_id": torch.tensor(CLASS_TO_ID[group["subcategory"]]),
            "source_id": group["source_id"],
        }


class VerifierImageDataset(Dataset):
    """Clean/degraded verifier images built from source-group metadata."""

    def __init__(self, root: Path, source_ids: list[str], crop_size: int | None = 256) -> None:
        self.root = Path(root)
        groups = json.loads((self.root / "metadata" / "source_groups.json").read_text(encoding="utf-8"))
        wanted = set(source_ids)
        self.source_ids = sorted(source_ids)
        self.source_to_id = {source_id: index for index, source_id in enumerate(self.source_ids)}
        self.rows: list[tuple[str, str]] = []
        for group in groups:
            if group["source_id"] not in wanted:
                continue
            self.rows.append((group["source_id"], group["clean_path"]))
            self.rows.extend((group["source_id"], group["degraded"][name]) for name in DEGRADATIONS)
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        source_id, relative_path = self.rows[index]
        image = _read_rgb(self.root / relative_path)
        if self.crop_size is not None:
            height, width = image.shape[-2:]
            target_h, target_w = max(int(height), self.crop_size), max(int(width), self.crop_size)
            image = _reflect_pad(image, target_h, target_w)
            top = max((target_h - self.crop_size) // 2, 0)
            left = max((target_w - self.crop_size) // 2, 0)
            image = image[:, top : top + self.crop_size, left : left + self.crop_size]
        return {
            "image": image,
            "source_id": source_id,
            "source_index": torch.tensor(self.source_to_id[source_id], dtype=torch.long),
        }
