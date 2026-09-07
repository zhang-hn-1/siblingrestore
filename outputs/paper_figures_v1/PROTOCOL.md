# 评测协议、指纹与运行命令（paper_figures_v1）

## 1. 数据与评测口径（全部来自官方 v2 管线，未重新评测）

- 数据包：`data/plamd_sfr_v1`（contract）→ `data/PLAMD_SFR_v1_method_b`（图像本体）。
- 划分：test = 256 sources（895 train / 128 val / 256 test，source 级互斥，seed 2026）。
- 每 source：1 张 clean + 7 张退化图（blur/haze/inpainting/lowlight/noise/rain/snow）→ 1792 恢复视图。
- 评测脚本：`scripts/evaluate_frozen_verifier_v2.py`；数值即该脚本产物 `.test.csv/.json`。
- PSNR/SSIM：RGB float [0,1]，masked 口径（reflection-pad 不计）；SSIM 为全局 masked 实现（非 windowed）。
- LPIPS：AlexNet `alex_256x256`，双图 bilinear 到 256×256，输入 x*2-1。
- noise（Method B）：clean 用 PIL Lanczos resize 到 noise 原生尺寸后同源评估；恢复视图 bilinear 回 clean 尺寸。
- 推理：分块恢复 `restore_tiled(512, 32)` 重叠均值融合。
- 检索：恢复图 embedding（verifier tiled 512/32）与 256 个 test clean anchor 的余弦，候选含自身；
  并列按更小索引优先（与官方 argmax 一致）；`correct_source_rank` 定义与此严格一致。
- 冻结评测 verifier：`best.pt`，sha256 `d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6`。

## 2. 指纹表（method → checkpoint sha256 / config sha / cohort）

| method | cohort | checkpoint_sha256(前12) | config_sha(12) | 逐图CSV行数 |
|---|---|---|---|---|
| A0 | ablation_core | `819ed9b271e4` | `3e729e9f6330` | 1792 |
| B1 | ablation_core | `2924dc96b84d` | `2580c491ef9b` | 1792 |
| B2 | ablation_core | `4151b7f6bb79` | `52eb473a8dd9` | 1792 |
| A5 | ablation_core | `221250dc3332` | `5f2f096d42ac` | 1792 |
| A1 | ablation_core | `0abcf9591530` | `15308f983821` | 1792 |
| A2 | ablation_core | `e744accbe42e` | `bc2e56421c8c` | 1792 |
| A3 | ablation_core | `3600aea0c693` | `c7826a5719dc` | 1792 |
| C1 | ablation_core | `d0d66ae881cf` | `13cc4db3edad` | 1792 |
| C2 | ablation_core | `9f2bcb31f3bc` | `6306865bb48f` | 1792 |
| dehazeformer | c005 | `9d775373ef7e` | `450bafc2e953` | 1792 |
| restormer | c005 | `cdc94e587131` | `3b6efe79cfcc` | 1792 |
| promptir | c005 | `fce760bb5b3f` | `d9df3100581c` | 1792 |
| ours_dim48_anchor | c005 | `75f34474ef12` | `186813f72b56` | 1792 |
| ours_dim64_anchor | c005 | `e0f2184ae81a` | `8e30144b66ab` | 1792 |
| airnet | c005 | `386282930220` | `d80c75c84e35` | 1792 |
| clearair | c005 | `e6b40930cb85` | `6150a90e8c5c` | 1792 |
| dfpir | c005 | `dfa4d7c6cf4c` | `d6d6689955cc` | 1792 |
| ffanet | c005 | `cdbe92d4fc95` | `63dfb70727ac` | 1792 |
| grl | c005 | `4a5f17626df9` | `47986eca8d0a` | 1792 |
| prenet | c005 | `ce83f215be02` | `e3404f250dcf` | 1792 |
| r2r | c005 | `7f293730d0e0` | `832830207608` | 1792 |
| swinir | c005 | `e355ba1c50ee` | `a498a51b0370` | 1792 |
| transweather | c005 | `f752218e787d` | `28eb9dd0d5d7` | 1792 |
| uformer | c005 | `99b434a0782c` | `a016377c01b4` | 1792 |
| M3_alcrb | psnr_modules_12k | `34ab20a22eac` | `7fecea956994` | 1792 |
| m3_alcrb_safe_seed13 | psnr_safe_refine | `70fa82f52e39` | `e8de763ce911` | 1792 |

注：全 26 方法同一 verifier、同一 data_pkg 指纹（audit_index `a4b095234bc8…` / groups `990880d4c2a6…`）；config 各不相同。

## 3. 生成资产与命令

产出：
- `unified_per_image.csv`（26 方法 × 1792 行，schema 见 `unified_schema.md`）
- `tables/table{1,2,3}_*.csv/.tex`、`table_supplementary_ablation.*`、`efficiency_records.csv`
- `bootstrap/bootstrap_a5_vs_b1.*`
- `figures/fig{1,4,6}_*`（纯指标）与 `fig{2,3,5}_*`（样本恢复图）
- `cases/case_plan.json|csv`、`fig1_*_delta_counts.csv`、`fig4_stability_stats.csv`
- `consistency_report.md`、`ANALYSIS_REPORT.md`

离线再生成（无 GPU）：
```bash
bash analysis/paper_figures_v1/run_all.sh
```
含样本恢复图推理（GPU，需要显卡，默认 cuda:3）：
```bash
RUN_INFERENCE=1 DEVICE=cuda:3 bash analysis/paper_figures_v1/run_all.sh
```

对应官方命令（如需从 checkpoint 重新生成任意方法的 v2 结果）：
```bash
cd /home/zhanghangning/siblingrestore-pilot-server
.venv/bin/python scripts/evaluate_frozen_verifier_v2.py \
    --checkpoint runs/ablation_core/A5_no_sibling_seed13/best.pt \
    --verifier runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt \
    --data-root data/plamd_sfr_v1 --split test --device cuda \
    --method A5_no_sibling --seed 13 --lpips \
    --output results/ablation_core/A5_no_sibling.13.test.json
```
（其余方法改 `--checkpoint`/`--method` 与输出前缀即可；输出与现网一致时才可覆盖，否则写新前缀。）

## 4. 本轮规则与说明
- 未启动任何训练；未修改/覆盖任何历史结果；未改动数据划分与生成。
- 恢复图未在本项目落盘（restored_path 列为空=已知缺失）；图 2/3/5 的恢复图为补推理产物。
- 选图规则在拼图前锁定（`10_select_cases.py` 头注释），案例坐标与依据见 `case_plan.json`。
- 仅用 test 结果分析；未用测试集选择权重/检查点。
