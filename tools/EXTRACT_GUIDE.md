# PLAMD 全量 vari-grip 数据包提取指南（正式论文数据集）

配套脚本：`tools/extract_plamd_vari_grip.py`（本目录）
输出契约：`PLAMD_EXTRACTION_SPEC.md`（零改动对接服务器代码），唯一扩展：**加入 noise，7 种退化**

---

## 1. 需要下载什么（总共 174.8 GB）

Zenodo 记录 21321669，7 个 zip **全都要下**（vari-grip 分布在每个 zip 里，zip 无法部分下载）：

| 文件 | 大小 | md5（下载后校验） |
|---|---|---|
| blur.zip | 29.9 GB | `ec427a9adeda277070b907110a9f8cc7` |
| haze.zip | 17.7 GB | `30454d5062e40a097ca59694e020d040` |
| inpainting.zip | 20.3 GB | `9f4365c0bb7cf8f8f6f8588d88dc3250` |
| lowlight.zip | 32.1 GB | `3aa578bebbb3bde332d1dfe7cb2bc757` |
| noise.zip | 27.8 GB | `368caf8eeccc653f01beb0f6fb8649b9` |
| rain.zip | 23.7 GB | `dd20c8b12db666cfcf82e36c836238f6` |
| snow.zip | 23.3 GB | `b03b94e36e07074a01fbb9d992420263` |

下载地址：`https://zenodo.org/records/21321669/files/<文件名>`（或 zenodo 页面按钮）。

**zip 内部结构**（每个 zip 都一样）：

```
blur.zip
├── blur/          # 退化图
│   └── vari-grip/  （及其他 17 个类别）
│       ├── good/ 或直接 *.jpg  ← 子目录结构以 --scan 实际输出为准
│       └── ...
└── original/      # clean 参考图（目录结构与 blur/ 一一对应）
    └── vari-grip/
```

## 2. 需要多少磁盘

| 项 | 大小 |
|---|---|
| 下载 7 个 zip | 174.8 GB（临时，可删） |
| **选择性解压工作区（只取 vari-grip）** | **~10 GB**（脚本直接从 zip 流式读取，无需全量解压） |
| **最终数据包**（tar.gz） | **~2 GB**（约 2000 sources × 8 图 = 16,000 张 JPEG） |

## 3. 提取步骤（在原始数据机器上跑）

```bash
# 0) 进入项目（或把脚本拷到数据机器上，用任意 python3 + Pillow）
cd siblingrestore-pilot-server

# 1) 先扫描：确认 zip 内部结构、vari-grip 下有没有 good/rust/bird-nest 子目录
.venv/bin/python tools/extract_plamd_vari_grip.py --scan \
    --zips /data/plamd/blur.zip /data/plamd/haze.zip /data/plamd/inpainting.zip \
           /data/plamd/lowlight.zip /data/plamd/noise.zip /data/plamd/rain.zip \
           /data/plamd/snow.zip

# 2) 正式提取（--max-sources 可不加；加了则封顶，按子类别分层保留）
.venv/bin/python tools/extract_plamd_vari_grip.py --extract \
    --zips /data/plamd/*.zip \
    --outdir /data/out/plamd_vari_grip_2000_v1 \
    --split-seed 2026 --max-sources 2000

# 3) 打包 + 自校验（脚本会自动做：解码检查/尺寸一致性/计数/sha256）
#    产物：plamd_vari_grip_2000_v1/ 目录 + plamd_vari_grip_2000_v1.tar.gz + extraction_report.json

# 4) 传到服务器
rsync -avP /data/out/plamd_vari_grip_2000_v1.tar.gz \
    zhanghangning@<服务器>:~/siblingrestore-pilot-server/data/
```

> 若原始数据已解压成目录（含 `blur/`、`original/` 等），用 `--raw-root /data/plamd_extracted` 代替 `--zips`，脚本同样支持。

## 4. 输出物（服务器零改动对接）

```
plamd_vari_grip_2000_v1/
├── images/
│   ├── clean/
│   │   └── {good,rust,bird-nest}/<source_key>.jpg
│   └── degraded/
│       └── {blur,haze,inpainting,lowlight,noise,rain,snow}/{good,rust,bird-nest}/<source_key>.jpg
└── metadata/
    ├── source_groups.json   # 2000 元素，按 source_id 排序
    ├── pilot_index.csv      # 2000 × 7 = 14,000 行
    └── audit.json           # 含两个 metadata 的 sha256，passed=true 才算成功
```

外加 `extraction_report.json`（来源计数、每类每 split 计数、7 退化覆盖、校验逐项结果）。

## 5. 数量与"全用吗"（决策记录）

- **预期 ~2000 sources**（38,865 图/退化 ÷ 20 类 ≈ 1,943，实际以 scan 为准）。
- **全用**：论文主实验就用全量 vari-grip——类别全量、数量可控（~2000）、统计够用（test ~300）。
- **封顶**：若实际远超 2000（如 3000+），用 `--max-sources` 封顶 2000~2500（分层采样保类别平衡），不必硬吃全部。
- **其他 19 个类别不用**（除非之后做跨类别 OOD 测试，届时各取几十张即可）。
- **注意 1K/2K 之分**：`component_mix_1K`（1K 部件图）与 `overallL_2K`（2K 全景）是两个子集；vari-grip 通常指 1K 部件图。scan 输出会显示 vari-grip 出现在哪个子树，若需要剔除 2K 全景里的同名目录，用 `--exclude-subtree overallL_2K`（默认已排除）。

## 6. 时间估算

| 步骤 | 耗时 |
|---|---|
| 下载 174.8GB | 看带宽（10MB/s ≈ 5h） |
| scan | ~1 分钟 |
| 提取（流式拷贝 ~10GB） | ~20-40 分钟 |
| 解码校验（16,000 图） | ~10-20 分钟 |
| 打包 | ~5 分钟 |

## 7. 常见问题

1. **vari-grip 下没有 good/rust/bird-nest 子目录**：脚本自动按文件名/目录名推断子类别（含 corrosion→rust、nest→bird-nest 映射），推断规则见 scan 输出；不准就用 `--subcategory-map '{"corrosion":"rust",...}'` 手动指定。
2. **某个 source 缺某退化**：按 spec 跳过并记录在 report（不要补）。
3. **同一 source 的 clean 与退化图尺寸不一致**：跳过并记录（spec 校验项 4）。
4. **md5 校验**：下载后 `md5sum *.zip` 与上表比对，防下载损坏。
