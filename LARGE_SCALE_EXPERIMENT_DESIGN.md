# 大规模实验与论文数据方案设计

- 日期：2026-08-21
- 状态：设计稿（待确认后实施）
- 关联：`PLAMD_EXTRACTION_SPEC.md`（数据提取契约）、`runs/baselines_500/COMPARISON_SUMMARY.md`（500 集汇总）

---

## 0. TL;DR（结论摘要）

1. **500 集够做开发/选型，但不足以直接支撑论文主结果**：test 只有 65 sources（390 queries），
   身份类指标（top1 / margin / AUC / EER）的置信区间太宽，方法间差异无法统计显著；
   训练集 355 sources（2,130 图）对从零训练所有方法也偏小。
2. **论文数据的"恰到好处"规模 = 全量 vari-grip（约 2,000 sources）**：
   - 全量 PLAMD 是 174.8GB / 26.3 万图 / 20 个部件类别 / 7 退化（Zenodo 21321669 实测元数据）；
   - **vari-grip 只是 20 类之一，全量约 2,000 sources ≈ 1.4 万图 ≈ 1.5-2GB**——"全量训练"根本不大，直接用；
   - 175GB 全库训练**不需要**：类别/分辨率混杂、成本爆炸、故事反而讲不清。
3. **真正该扩的是 test，不是训练**：test 65 → 300 sources，身份指标的 95% CI 减半以上，
   方法间 0.5~1 个百分点级别的差异才能站得住。
4. **管线三件套**（本设计的主体）：离线预处理（tile + anchor 缓存）→ 流式可断点评估 → campaign
   管理与自动汇总。全部落地后 2000 集 × 13 模型 × (val+test) 的评估 1~2 天出齐。

---

## 1. 背景与现状

| 项 | 现状 |
|---|---|
| 数据包 | `data/plamd_vari_grip_500`：500 sources（355 train / 80 val / 65 test），6 退化（缺 noise），450MB |
| 图像 | 400~700px 的小 patch（component_mix_1K 子集），已按 source 组织 |
| 模型 | 4 个 baseline（Restormer / PromptIR / PReNet / GRL）+ ~10 个 ours 变体，configs + runs 按 family 组织 |
| 评估 | `scripts/evaluate_frozen_verifier.py` per-view（统一 tile 512/overlap 32，独立冻结 verifier）——**可信口径** |
| 批量评估 | `scripts/evaluate_batch.py` —— 因变尺寸输入反复报错、结果与 per-view 严重不一致（22.07 vs 25.39），**弃用** |
| 汇总 | `runs/baselines_500/COMPARISON_SUMMARY.md`（手工维护，含 500 集主表 / 每退化表 / pilot 表） |
| 硬件 | 4× V100 32GB（基本空闲）；`/home` 19T 剩 3.1T（83% 已用，共享机器） |

---

## 2. 论文实验数据充分性分析（核心问题：够不够发表）

### 2.1 训练数据规模

- 现状：355 sources × 6 views = 2,130 训练图。
- 参考：图像复原论文常见训练规模——DIV2K 800、GoPro 2,103、SOTS 500 对、Rain100H 1,800。
  2,130 张在同量级**下限附近**，不算硬伤，但所有方法从零训练时偏紧。
- **sibling 任务特有需求**：身份目标依赖"同源多视图"，每个 source 贡献 C(6,2)=15 个 sibling 对。
  - 355 sources → 5,325 对；~1,400 train sources → **21,000 对**（×4）；
  - 冻结 verifier 也在 train 上训练（355 → ~1,400 sources），verifier 本身更强，开放集身份评估更有说服力。
- 判断：**训练扩到 ~1,400 sources 收益显著且成本可控**（每模型单卡 1~3 天，13 模型 4 卡并行约 1 周）。

### 2.2 验证/测试规模（当前最大短板）

test 65 sources × 6 views = 390 queries，身份指标置信区间（近似，1σ~95%）：

| 指标（当前水平） | 现在 test 390 views | 建议 test 1,800 views |
|---|---|---|
| top1 ≈ 0.89（95% CI） | ±0.031 | ±0.014 |
| EER ≈ 0.05（1σ） | ±0.011 | ±0.005 |
| ROC-AUC（负样本 ~25k vs ~115k） | ±0.005 | ±0.002 |

- 500 集上 ours_dim48_anchor 与 Restormer 的 top1 差 0.009、margin 差 0.0065——在 390 queries 上
  **远不显著**；test 扩到 300 sources 后这些量级的差异才可检验。
- 身份验证是本文的核心卖点，审稿人最容易攻击"65 个身份太少"。**test ≥ 200~300 sources 是硬要求**。
- val 同步扩到 ~300，用于选型/消融同样受益。

### 2.3 方法覆盖与对照

- baseline 已够：Restormer、PromptIR、PReNet、GRL（含一个身份相关对照，覆盖好）。
- 消融已有 v02~v05 体系（backbone 决策 / 退化条件 / identity 分支 / 冻结 anchor / PCGrad），
  需升级到新数据口径重跑。
- **缺口**：
  1. 多 seed：500 集目前基本只有 seed13；v03~v05 已有 13/37/73 三 seed 先例 → 关键模型至少 3 seeds，
     报告 mean±std + 显著性（bootstrap，v05 已有先例）。
  2. 按退化分解表已存在（6 退化），加 noise 后补成 7 行。
  3. （可选加分）跨类别 zero-shot：用全量中其他类别（yoke / insulator 等）评估泛化，
     成本只有一次评估，比全量训练更能打动审稿人。

### 2.4 结论：什么规模"够发表"

| 项 | 建议 | 理由 |
|---|---|---|
| 数据包 | **全量 vari-grip ~2,000 sources**（7 退化，含 noise） | 训练信号 ×4、评估统计显著、成本可控 |
| split | ~1,400 train / ~300 val / ~300 test（source 级互斥，规则不变） | test 是论文硬指标 |
| seed | 关键模型 3 seeds（13/37/73），其余 1 seed | 报告 mean±std + bootstrap CI |
| 训练 | 所有方法同数据同预算从零训练 | 公平对比，避免 GRL "低恢复强度假象"类问题 |
| 不做 | 175GB 全库（20 类）训练 | 收益递减、资源爆炸、domain 混杂伤故事 |

---

## 3. 数据包方案

### 3.1 提取（沿用 PLAMD_EXTRACTION_SPEC.md 契约，零改动对接）

- 提取脚本参数化：`--sources N --split-seed <固定> --degradations blur,haze,inpainting,lowlight,rain,snow,noise`
- 新增 `noise` 退化目录（全量库已有 noise.zip，27.8GB，提取成本≈0）。
- 每包一个版本号 + `audit.json`（含 pilot_index / source_groups 的 sha256）→ **包指纹**。
- 与旧包按 content hash 去重：新旧包 train/val/test 不得有 source 重叠（防污染）。
- 传输沿用现有模式：本地提取 → tar.gz → rsync 到服务器 → 解压工作副本（tar.gz 留作 immutable 存档）。
- 命名：`plamd_vari_grip_2000_v1/`（含 `images/` + `metadata/`，结构与 500 包一致）。

### 3.2 规模估算

- 2000 sources ≈ 14k 张图 ≈ 1.5~2GB 原图（按 500 集 450MB/3.5k 张推算）。
- 磁盘：tar.gz + 解压 + tile 缓存 ≈ 15~25GB，无压力。

---

## 4. 离线预处理（`scripts/preprocess.py`，新文件）

目标：**评估时不再碰原图**，一次离线把数据包转成评估直接可用的缓存。

### 4.1 tile 缓存

- 每张图（clean + degraded）切成 512×512 tiles（overlap 32，与现有 `restore_tiled` 口径一致），
  附带有效区域 mask；存 `.pt`（uint8/fp16）或无损 PNG。
- 目录：`cache/<pkg_hash>/tiles/<split>/<source>/<degradation|clean>/tile_<r>_<c>.{pt,png}`
- 收益：
  - 消除运行时 JPEG 解码 + padding/tiling（GRL 500 集评估 1h+ 的主要瓶颈 → 十几分钟级）；
  - `evaluate_batch.py` 那些变尺寸输入报错的**根因彻底消失**；
  - GPU 利用率拉满，评估可并行。

### 4.2 anchor 缓存

- clean anchor bank 只依赖 verifier 版本 + 数据包 → **每个 verifier × 数据包只算一次，13 个模型共享**。
- 命名：`cache/<pkg_hash>/anchors/<verifier_sha12>.pt`（含 source 顺序与 sha 记录）。
- 500 集时每次评估都重算 80 个 anchors；2000 集 × 13 模型将省 13 × 300 sources 的重复 embed。

### 4.3 校验

- 抽样（每 split ≥ 20 sources）对比：tile 拼接评估 vs 原图评估的 PSNR/SSIM 差异 < 容差（如 0.05 dB）；
- 记录预处理脚本版本 + 参数 hash 到 `cache/<pkg_hash>/preprocess.manifest.json`。

---

## 5. 评估管线改造（`evaluate_frozen_verifier.py` v2，原地升级）

### 5.1 流式化（2000 sources 必须）

- 现状：`restored_by_source` 全量保留所有恢复图（500 集 480 张 OK；2000+ 集 ~9,000 张 × 512×512×3×4B ≈ **30~90GB，必炸**）。
  → sibling L1 / cross-source L1 / 均值图改**在线流式计算**（逐 source 处理完即弃，只保留统计量）。
- pairs 矩阵 O(N²)：300 sources 时 pair 行 ~300×6×2×300 = 108 万行（可接受）；
  2000 全量 val+test ~600 sources 时 ~430 万行（CSV 数百 MB，可接受但非必要）。
  → 默认只落 aggregate + per-view 行 + top-k impostor（k=5）；全 pair 保留 `--save-pairs` 开关。

### 5.2 断点续跑

- 每 source 一行 JSONL 落盘（`<output>.sources.jsonl`），重跑时跳过已完成 source；
- 失败 source 单独记录，不阻塞整轮。

### 5.3 多 GPU

- 按 source shard 切分（或 split 切分）；沿用 v05 / `run_baseline_multiseed.sh` 的 per-GPU flock 模式，
  4 卡并行，screen 后台跑。

### 5.4 输出契约（保持兼容）

- 主报告：`results/campaigns/<campaign>/<model>.<seed>.<split>.json`
  - 结构沿用现有 report（status / split / sources / verifier_fingerprint / aggregate / notes）；
  - **新增**：`data_pkg_hash`、`config_hash`、`preprocess_version`、`tile_params`、`per_degradation`（aggregate 按退化分解，论文表 2 直接可用）。
- per-view：同名 `.csv`；pairs：`--save-pairs` 时 `<stem>_pairs.csv`。

---

## 6. Campaign 管理（`scripts/run_campaign.py` + `scripts/summary_campaign.py`，新文件）

### 6.1 manifest

`results/campaigns/<campaign>/jobs.jsonl`，每行一个 job：

```json
{
  "model": "ours_dim48_anchor", "seed": 13, "split": "val",
  "config_hash": "<config json sha12>", "verifier_sha": "<verifier ckpt sha12>",
  "data_pkg_hash": "<audit.json sha12>", "gpu": 2,
  "status": "done", "output": "results/campaigns/c2000_v1/ours_dim48_anchor.13.val.json",
  "started_utc": "...", "finished_utc": "...", "wall_seconds": 1234
}
```

### 6.2 幂等规则

- status=done 且 model/seed/split/config_hash/verifier_sha/data_pkg_hash 全一致 → **跳过**；
- 任一指纹变化 → 自动重跑（防"换了 verifier 结果混用"的旧教训，见第 9 节风险）。

### 6.3 训练侧（与评估共用 manifest）

- 训练 job 同样入 manifest（`phase: train`）：`run_baseline_multiseed.sh` 模式 + 统一训练预算
  （epoch / steps / crop / lr 全家族一致，避免 GRL 那种"训练配置漂移导致对比无效"）。

### 6.4 汇总

- `summary_campaign.py` 从 manifest + 各 job json 自动生成：
  - 表 1 主表：方法 × PSNR/SSIM/top1/margin/AUC/EER/参数 (M)；
  - 表 2 每退化 PSNR/SSIM；
  - 表 3 多 seed mean±std + bootstrap 95% CI（v05 已有先例代码）；
  - 输出 `runs/<campaign>/COMPARISON_SUMMARY.md`（保持现汇总文件格式，从手工改为自动）。

---

## 7. 论文实验矩阵（目标产出）

| 表/图 | 内容 | 数据要求 |
|---|---|---|
| 表 1 主表 | 13 方法 × 指标 | 2000 集 val+test，seed13（关键模型 3 seeds） |
| 表 2 | 每退化分解（7 退化） | 同主表 |
| 表 3 消融 | v02~v05 体系升级重跑 | 2000 集口径 |
| 表 4 | 多 seed mean±std + bootstrap CI | 关键模型 3 seeds |
| 表 5（可选） | 跨类别 zero-shot（yoke/insulator…） | 全量其他类别一次评估 |
| 图 | 定性对比 + CAM / t-SNE | `pytorch_model-master` 已有 visualize 工具 |

---

## 8. 资源与时间估算

| 阶段 | 耗时/占用 |
|---|---|
| 提取 2000 集数据包 + 传输 | ~1~2 天（含审核） |
| tile + anchor 预处理 | ~1~2 小时 + 10~20GB 缓存 |
| 训练 13 模型 × ~1,400 sources | 每模型单卡 1~3 天；4 卡并行 ~1 周 |
| 评估全模型 (val+test) | 预处理后 ~1~2 天（4 卡并行） |
| 汇总 + 统计检验 | ~1 天 |

磁盘：数据 2GB + 缓存 20GB + 训练 ckpt/日志 ~50GB → 全程 <100GB，当前剩 3.1T 充裕；
注意共享机器（其他用户也在跑大实验），GPU 调度用锁 + 分时，避免争抢。

---

## 9. 风险与注意事项

1. **verifier 版本混用**（已发生）：Restormer 早期用了 v05 verifier 导致 top1 虚高 → 所有结果
   强制记录 `verifier_fingerprint`，汇总表只收同一指纹。
2. **批量管线不可信**：正式实验一律 per-view；`evaluate_batch.py` 不再用于产出。
3. **训练配置漂移**：GRL 因 crop 128 训练导致"低恢复强度假象"（PSNR 20.0、丢失率 2.7% 无对比价值）
   → 统一训练预算，所有方法同一 config 模板（仅 model 差异）。
4. **test sealed 纪律**：test 只在全部选型/消融定稿后开一次，防"用 test 调参"。
5. **共享机器**：与同机用户错峰；4 GPU 锁 + screen。
6. **类目估算误差**：vari-grip 全量 ~2,000 sources 是按 38,865/20 类均匀估算，
   **以实际提取结果为准**（≥1,500 即可按方案执行；<1,500 需重新权衡）。

---

## 10. 执行顺序（里程碑）

- **M0** 确认规模与范围（本设计定稿）
- **M1** 参数化提取 2000 集数据包（含 noise），audit 校验通过，rsync 到服务器
- **M2** `preprocess.py`（tile + anchor 缓存）+ `evaluate_frozen_verifier.py` v2 改造（流式/断点/指纹）
- **M3** `run_campaign.py` + `summary_campaign.py`
- **M4** 2000 集全模型训练（13 模型 × seeds，分批推进）
- **M5** 全模型评估 + 自动汇总 + bootstrap 显著性
- **M6** 论文表格/图生成（含可选 OOD 表）

---

## 11. 待确认事项

1. 规模：全量 vari-grip ~2,000 sources（推荐）还是 1,500？
2. noise 退化：加入 7/7（推荐，成本≈0）？
3. seeds：关键模型 3 seeds（13/37/73）够不够，还是全模型 3 seeds？
4. OOD 跨类别表：做不做（加分项，成本 ~1 天评估）？
5. 训练预算统一方案：以哪个模型为准定 epoch/steps 上限？
