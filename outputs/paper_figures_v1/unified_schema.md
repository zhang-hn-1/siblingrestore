# 统一逐图结果表字段说明

## evaluation_protocol（所有行相同）

```
eval_v2 test(256src/1792views) verifier=d9b000aeb04d6e5dc64e6eff1fee7fc8c74e486935dd962fbcd1e26462451ae6 lpips=alex_256x256 psnr/ssim=masked_official noise=clean_lanczos_to_noise retrieval=cos_over_256_test_clean_anchors incl_self tie=first_index t512_o32
```

## 字段

| 字段 | 含义 | 来源 |
|---|---|---|
| method | 方法 ID（见表 catalog） | |
| cohort | 训练批次（ablation_core=20k sibling 采样；c005=12k independent 官方对比；psnr_modules_12k/psnr_safe_refine=12k 主线） | |
| source_id | 测试来源 ID（vari-grip/<class>/<stem>） | source_groups.json |
| sample_id | 0..6 对应 blur,haze,inpainting,lowlight,noise,rain,snow | |
| degradation | 退化类型 | |
| component_category | 数据包 subcategory：good/rust/bird-nest（设备部件状态类，非观测到的缺陷标注） | |
| input_path / clean_path | 退化图/对应 clean 图绝对路径 | metadata 解析 |
| restored_path | 恢复图路径：项目未落盘恢复图，恒为空（缺失项） | |
| checkpoint_fingerprint | best.pt sha256（全 64 位）或 config 指纹兜底 | 本清单 sha |
| verifier_fingerprint | 评测 frozen verifier sha256 | .test.json |
| psnr/ssim/lpips | 官方 v2 masked PSNR / masked SSIM / LPIPS alex_256x256 | .test.csv |
| own_clean_similarity | 恢复图 embedding 与自身 clean anchor 余弦 | sources scores |
| nearest_wrong_similarity | 恢复图与最近错误 anchor（排除自身）余弦 | sources scores |
| correct_source_rank | 正确来源在 256 全库中的检索排名（并列按更小 index 优先，与官方 argmax 一致） | 由全 score 向量重算 |
| top1_correct / top5_correct | 正确来源命中 Top1 / Top5（0/1） | 重算 |
| samecat_top1_correct / samecat_correct_source_rank | 同 subcategory 候选库内的 Top1/排名（候选>=5） | 重算 |
| retrieved_source_id | 检索返回的最高相似来源 | 重算 |
| restored_margin | own_sim − nearest_wrong | 重算（等于官方 margin） |
| input_* | 退化输入 embedding 的对应指标（表 2 退化输入对照行） | 重算 |