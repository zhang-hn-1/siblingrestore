# 一致性核对报告

- 每个方法 1792 行
- 每个方法 256 个来源
- 每个方法覆盖 7 类退化
- 每来源 7 视图
- 退化输入检索与方法无关
- 统一表均值与官方 json 最大偏差 7.11e-15 < 1e-6
- 宏平均 PSNR 与官方 aggregate 最大偏差 7.11e-15
- top1<=top5 全部满足
- rank==1 与 top1_correct 完全一致
- 同类别候选均>=5：{'bird-nest': 75, 'good': 119, 'rust': 62}
- input_path / clean_path 全部非空
- EXPECTED restored_path 全部为空（项目未落盘恢复图；图 2/3/5 已补推理到 sample_restored/）
- 图1 逐退化视图计数总和 = 1792（应为 1792）
- 图1 分类计数合计等于视图数
- 图1 每源聚合样本数=256
- 图4 完整七类来源数=256
- 图4 直方图计数合计=256
- fig2 来源 vari-grip/good/Fotos 22-10-202... 7 视图齐备
- fig2 来源 vari-grip/bird-nest/Fotos 25-1... 7 视图齐备
- fig3 案例 improvement 可在统一表找到
- fig3 案例 improvement 恢复图存在 (A5)
- fig3 案例 improvement 恢复图存在 (B1)
- fig3 案例 regression 可在统一表找到
- fig3 案例 regression 恢复图存在 (A5)
- fig3 案例 regression 恢复图存在 (B1)
- fig3 案例 unchanged 可在统一表找到
- fig3 案例 unchanged 恢复图存在 (A5)
- fig3 案例 unchanged 恢复图存在 (B1)
- fig5 crop 坐标已确定
- 除 restored_path 外无缺失值（NaN 分布：{}）

## 未通过项
- 无
