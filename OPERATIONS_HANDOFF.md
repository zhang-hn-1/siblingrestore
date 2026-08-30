# 操作交接文档（给 Codex / 协作者）

生成时间：2026-08-22 13:40 CST
项目根：`/home/zhanghangning/siblingrestore-pilot-server`（python 环境：`.venv/`，torch 2.4.1 + lpips 0.1.4）

---

## 1. 一句话现状

**正式实验（campaign c001_sfr_v1）已启动并在 4 张 V100 上运行**：1 个 verifier + 11 个模型训练 + 22 个评估共 34 个 job，
预计全流程 ~24h。全部基础设施（数据适配 / 评估 v2 / 编排器 / 汇总）已交付并测试通过。

## 2. 正式数据集（已就绪）

| 项 | 值 |
|---|---|
| 数据包 | `data/PLAMD_SFR_v1_method_b/`（tar.gz 同名，2.4GB，1279 sources × 7 退化 + clean = 10232 图） |
| split | **895 train / 128 val / 256 test**（source 级互斥，seed 2026，70/10/20） |
| 退化 | blur, haze, inpainting, lowlight, **noise**, rain, snow（noise 为 Method B，与 clean 原生尺寸不一致，1235/1279 个 source） |
| 契约 metadata | `data/plamd_sfr_v1/metadata/`（旧契约 source_groups.json + pilot_index.csv + audit.json，由 `tools/build_contract_sfr_v1.py` 生成） |
| noise 协议 | 训练/评估时 clean 用 **Lanczos resize 到 noise 原生尺寸**后同源随机 crop（已实现在 `siblingrestore/data.py` 的 `align_pair_shape`） |

## 3. 正在运行的实验（campaign c001_sfr_v1）

- 定义：`scripts/campaigns/c001_sfr_v1.json`（11 模型 × seed13 × val+test，**统一预算 max_steps=12000, crop=256**）
- 模型：restormer, promptir, prenet, grl（baseline）+ ours_dim48_anchor, ours_dim48, ours_two_stage_v2,
  ours_dim32_msc, ours_dim48_two_stage_anchor, ours_dim64, ours_dim64_anchor（家族）
- manifest：`results/campaigns/c001_sfr_v1/jobs.jsonl`；日志：`results/campaigns/c001_sfr_v1/logs/`
- 检查状态：
  ```bash
  cd /home/zhanghangning/siblingrestore-pilot-server
  .venv/bin/python scripts/run_campaign.py status scripts/campaigns/c001_sfr_v1.json
  ```
- 后台进程：`run_campaign.py run`（重启安全：崩溃残留的 running 会被重置为 pending，幂等续跑）
- 训练产物：`runs/campaigns/c001_sfr_v1/<model>/<seed>/best.pt`；评估产物：`results/campaigns/c001_sfr_v1/<model>.<seed>.<split>.json`
- 全流程结束后生成汇总表：
  ```bash
  .venv/bin/python scripts/summary_campaign.py results/campaigns/c001_sfr_v1
  # → COMPARISON_SUMMARY.md + metrics.csv（含主表/按退化表/多 seed mean±std）
  ```

## 4. 基础设施清单（已交付，全部测试过）

| 文件 | 作用 | 备注 |
|---|---|---|
| `tools/extract_plamd_vari_grip.py` | 原始 zip → 数据包（scan/extract/打包） | 用户侧已用完；新包是 PLAMD_SFR_v1_method_b |
| `tools/build_contract_sfr_v1.py` | 新包 manifest → 旧契约 metadata | 已运行；换包后重跑 |
| `scripts/evaluate_frozen_verifier_v2.py` | 评估 v2：流式/断点/指纹/LPIPS/anchor 缓存/按退化聚合 | **正式评估唯一口径**；v1 与 `evaluate_batch.py` 不再用于产出 |
| `scripts/run_campaign.py` | 训练+评估编排（prepare/run/status，文件锁并发安全） | 已修复并发 bug（pick_job 跳过 running） |
| `scripts/summary_campaign.py` | 自动汇总主表/按退化表/mean±std | 输出 COMPARISON_SUMMARY.md |
| `scripts/campaigns/c001_sfr_v1.json` | campaign 定义（模型列表/预算/GPU） | 加模型/seed 改这里后重新 prepare |
| `siblingrestore/data.py` | noise Lanczos 对齐 + 7 退化 | 已改 |
| `siblingrestore/model.py` | degradation_count 默认 7 | 已改 |

## 5. 论文实验路线图（剩余工作，按优先级）

### 5.1 等 campaign 完成后（预计 8-23 中午）
1. 跑 `summary_campaign.py` 生成主表 → 与 500 集结论对比（预期：ours_dim48_anchor 身份领先、PSNR 接近 Restormer）
2. **R1（最高优先，身份指标排雷）**：双 verifier + 外部嵌入身份评估
   - 现状：训练用 verifier（runs/baselines_500/verifier_evaluator/best.pt，sha 55fb77d6）与评估同源 → 循环论证风险
   - 做法：在 eval v2 基础上支持 `--embedder`（ImageNet 预训练 ResNet50 / DINO 特征替代 verifier 做同一套
     retrieval/AUC/EER/margin），或独立脚本；评估不依赖训练 verifier 的嵌入
   - 报告两个嵌入的身份指标 → 论文写 "结论在两种独立嵌入下成立"
3. **R2**：Pareto 散点（PSNR × 身份增益，所有方法）证明 ours 在效率前沿；质量匹配对比
4. **R4**：inspection classifier 闭环 —— 用 train 集 clean 图训 3 类分类器（good/rust/bird-nest，数据包自带标签），
   测恢复图上的 defect flip 率（F_d2g / F_g2d）

### 5.2 中期（并行可做）
- **R3 消融**：单 pair（无 sibling）/ 等权 sibling / 对比正则（AirNet 式）三对照 → 证明 RSD 机制是真因
- **dim64 已纳入 c001**（ours_dim64, ours_dim64_anchor）→ 家族 scaling 曲线
- 多 seed：关键模型补 seed 37/73（改 spec `"seeds": [13, 37, 73]` 重新 prepare，幂等只会加新 job）
- LODO（leave-one-degradation-out）+ 跨类别 OOD（metadata/ood_candidates.json 有现成候选）

### 5.3 论文写作素材
- 500 集汇总：`runs/baselines_500/COMPARISON_SUMMARY.md`（旧口径，仅作参考）
- 设计文档：`LARGE_SCALE_EXPERIMENT_DESIGN.md`（含发表充分性分析、实验矩阵、风险对策）
- 用户 protocol：参考对话中 "Experimental Protocol v1.0"（RSD 方法描述、消融表设计）

## 6. 已知坑（务必遵守）

1. **评估只用 v2**：`evaluate_batch.py` 批量管线不可信（结果与 per-view 不一致）；v1 无指纹/断点
2. **统一训练预算**：所有方法 12000 步 / crop 256（GRL 之前 crop 128 导致"低恢复强度假象"，已通过 overrides 统一）
3. **verifier 口径**：所有评估报告带 `verifier_fingerprint.sha256`，汇总表只接受同一 sha（或显式 --min-verifier 过滤）
4. **test sealed**：test 只在全部选型/消融定稿后开一次
5. **共享机器**：4× V100 32GB 与其他用户共用（jiangshuo 在跑 TCGA）；campaign 用 4 卡，注意避开高峰
6. **数据包不可变**：`data/PLAMD_SFR_v1_method_b/` 视为只读；改数据 → 重跑 `build_contract_sfr_v1.py` + 重新 prepare（指纹自动失效）
7. **磁盘**：/home 剩 ~3.1T；campaign 产物（ckpt/日志/评估 json）约 50GB，够用但别复制数据包

## 7. 时间估算（实测）

- val 验证（128 sources）：~10 min/次 → 训练含 24 次验证
- 训练：ours 家族 ~6-7h/模型，restormer/promptir/prenet 类似，grl 最慢（可能 12h+）
- 评估：val ~19 min + test ~38 min / 模型（含 LPIPS）
- 全 campaign：~24h（4 卡并行）

## 8. 下一步交接点

campaign 全部 done 后：跑 summary → 把主表发我 → 我安排 R1/R2/R4 的具体实现。
期间任何 job failed：看 `results/campaigns/c001_sfr_v1/logs/<job>.log`，修好后重跑 `run_campaign.py run`（自动续传）。
