# PLAMD 数据提取规格（供本地处理模型执行）

本规格描述如何从原始 PLAMD 数据提取一个服务器可直接使用的数据包。
请严格按以下字段、目录结构和命名约定输出，服务器端代码可以零改动对接。

## 0. 目标

- 从 PLAMD Zenodo 记录 (21321669) 中提取 **500 个 vari-grip 来源**（如可用来源不足，至少 300；超过 1000 则先提取 500）。
- 每个来源 = 1 张 clean 图 + 6 张退化图。
- 6 种退化：`blur, haze, inpainting, lowlight, rain, snow`。
- 输出目录结构、元数据格式与现有 pilot 完全一致，服务器可直接运行现有训练/验证/verifier 代码。

## 1. 目录结构

```
plamd_vari_grip_500/
├── images/
│   ├── clean/
│   │   ├── bird-nest/<source_key>.jpg
│   │   ├── good/<source_key>.jpg
│   │   └── rust/<source_key>.jpg
│   └── degraded/
│       ├── blur/<subcategory>/<source_key>.jpg
│       ├── haze/<subcategory>/<source_key>.jpg
│       ├── inpainting/<subcategory>/<source_key>.jpg
│       ├── lowlight/<subcategory>/<source_key>.jpg
│       ├── rain/<subcategory>/<source_key>.jpg
│       └── snow/<subcategory>/<source_key>.jpg
└── metadata/
    ├── source_groups.json
    ├── pilot_index.csv
    └── audit.json
```

- 所有图片为 JPEG，RGB 三通道。
- `<subcategory>` ∈ {`good`, `rust`, `bird-nest`}。
- `<source_key>`：来源的稳定短标识（例如原始文件名主体或内容哈希，如 `e5d5d2d6b0313c9e`）。同一来源的 clean 与 6 张退化图使用**相同** `<source_key>`。

## 2. 命名约定

- `source_id` 格式：`<subcategory>|<原始文件名主体>`。
  示例（来自现有 pilot）：`bird-nest|08-06-2021_DJI_0285_392_3`。
  若原始文件名不适合，可改为 `<subcategory>|<source_key>`，但必须全局唯一且稳定。
- `source_key`：与图片文件名一致的短标识。
- `subcategory`：来源类别，必须为三者之一。

## 3. 元数据格式

### 3.1 source_groups.json

顶层为 JSON 数组，每个元素对应一个来源：

```json
[
  {
    "source_id": "bird-nest|08-06-2021_DJI_0285_392_3",
    "source_key": "e5d5d2d6b0313c9e",
    "subcategory": "bird-nest",
    "split": "train",
    "clean_path": "images/clean/bird-nest/e5d5d2d6b0313c9e.jpg",
    "degraded": {
      "blur": "images/degraded/blur/bird-nest/e5d5d2d6b0313c9e.jpg",
      "haze": "images/degraded/haze/bird-nest/e5d5d2d6b0313c9e.jpg",
      "inpainting": "images/degraded/inpainting/bird-nest/e5d5d2d6b0313c9e.jpg",
      "lowlight": "images/degraded/lowlight/bird-nest/e5d5d2d6b0313c9e.jpg",
      "rain": "images/degraded/rain/bird-nest/e5d5d2d6b0313c9e.jpg",
      "snow": "images/degraded/snow/bird-nest/e5d5d2d6b0313c9e.jpg"
    }
  }
]
```

要求：
- 每个来源的 `degraded` 必须恰好包含上述 6 个键。
- `split` ∈ {`train`, `val`, `test`}，划分见第 4 节。
- 数组中元素按 `source_id` 字典序排序。

### 3.2 pilot_index.csv

CSV，表头：

```csv
source_id,source_key,subcategory,split,degradation,degraded_path,clean_path
```

- 每来源 6 行（每种退化一行）。
- `degradation` 顺序任意，但每个来源必须覆盖 6 种。
- 共 500 × 6 = 3000 行。

### 3.3 audit.json

```json
{
  "rows": 3000,
  "sources": 500,
  "unique_images": 3500,
  "degradations": ["blur", "haze", "inpainting", "lowlight", "rain", "snow"],
  "split_sources": {"train": 355, "val": 80, "test": 65},
  "class_sources": {"bird-nest": <实际数>, "good": <实际数>, "rust": <实际数>},
  "formats": {"jpeg": 3500},
  "index_sha256": "<pilot_index.csv 的 sha256>",
  "groups_sha256": "<source_groups.json 的 sha256>",
  "decode_errors": [],
  "passed": true
}
```

## 4. 划分规则（source 级互斥）

- **禁止按图片划分**；同一来源的 7 张图必须全部属于同一个 split。
- 按 `subcategory` 分层采样，比例约 71% train / 16% val / 13% test（500 个来源 → 355/80/65）。
- 每类内部随机打散后按比例切分，`split` 写入每个来源。
- 输出 `audit.json` 中的 `split_sources` 与 `class_sources` 必须与实际数据一致。

## 5. 校验清单（输出前必须全部通过）

1. 目录数：`images/clean` 下 3 个子目录，`images/degraded` 下 6 个退化目录 × 3 个子目录。
2. 文件数：clean 500 + degraded 3000 = 3500 张 JPEG；所有文件可被 Pillow/OpenCV 解码为 RGB。
3. 每来源 1 clean + 6 degraded，degraded 键集合 == {blur, haze, inpainting, lowlight, rain, snow}。
4. clean 与 6 张 degraded 的像素尺寸一致（同一来源内）。
5. `source_id` 全局唯一；`source_key` 全局唯一。
6. 每个来源的 7 张图 split 一致；无图片泄漏到其他 split。
7. `pilot_index.csv` 行数 = 6 × sources；`source_groups.json` 元素数 = sources。
8. 计算两个元数据文件的 sha256 并写入 audit.json。

## 6. 输出方式

- 打包为 `plamd_vari_grip_500.tar.gz`（含 images/ 与 metadata/）。
- 另附一个 `extraction_report.json`：来源计数、每类计数、每 split 计数、6 种退化覆盖检查结果、校验清单逐项通过/失败。

## 7. 备注

- 若原始数据某来源缺少某种退化，跳过该来源并在报告中记录（不要用其他来源补）。
- 若超过 500 来源可用，优先选择类别均衡的子集（good/rust/bird-nest 尽量按原有比例保留）。
- 不要修改、重命名或重新编码图片内容；仅重新组织目录与生成元数据。
