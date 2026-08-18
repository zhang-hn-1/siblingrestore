# SiblingRestore PLAMD 单类别潜力测试包

这是一个可直接上传 Linux 服务器的**小范围可行性实验**，目的不是立即产出论文表格，而是先回答：利用同一电力设备来源的多种退化图作为 sibling supervision，是否比普通独立成对恢复更能保持设备来源身份，同时不牺牲恢复质量。

## 包内内容

- `data/plamd_vari_grip_pilot/`：已物化的数据，不依赖原始大压缩包。
- 100 个 `vari-grip` 来源，按来源互斥划分为 train/val/test = 71/16/13。
- 每个来源含 1 张 clean 和 blur、haze、inpainting、lowlight、rain、snow 各 1 张，共 700 张 JPEG（约 87.6 MiB）。
- 三条公平训练路径、验证、来源保持诊断、单图推理和数据校验代码。

`metadata/audit.json` 已记录数据数量、解码检查及索引哈希。任何 train/val/test 划分都以 `source_id` 为单位，避免同一来源泄漏到不同集合。

数据来源为 [PLAMD Zenodo 记录](https://zenodo.org/records/21321669)。本包只用于将你已下载的数据搬到自己的研究服务器；使用或进一步分发前仍应遵守原数据页面列出的许可和引用要求。当前 PLAMD 证据属于受控/合成退化实验，不能直接表述为真实野外鲁棒性。

## 当前主模型与实验角色

主干是轻量 Restormer-style encoder-decoder（正式配置约 1.19M 参数）。推理时仍然只输入一张退化图。SiblingRestore 仅在训练时增加三种约束：

1. degraded sibling 到同一 clean source anchor 的来源对比学习；
2. 同类别、不同来源作为加权 hard negatives；
3. sibling 输出一致性和退化类型辅助分类。

三组实验使用相同主干、优化器、crop、5000 updates 和每步 8 张退化图：

| 实验 | 采样 | 每步退化图 | 额外 sibling loss |
|---|---:|---:|---:|
| independent | 8 个独立 pair | 8 | 无 |
| group control | 4 个来源 × 2 siblings | 8 | 无 |
| SiblingRestore | 4 个来源 × 2 siblings | 8 | 有 |

`group control` 用来排除“只是 grouped sampling 带来收益”的解释。

## 服务器安装与 smoke

建议 Python 3.10+。先按服务器 CUDA 版本安装 PyTorch，再安装其余依赖：

```bash
cd siblingrestore-pilot-server
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 按 https://pytorch.org/get-started/locally/ 选择服务器对应的 torch 命令
pip install -r requirements.txt
python check_bundle.py
bash run_smoke.sh
```

如果服务器没有 `python` 命令，请使用 `python3`，或显式指定已有的 PyTorch 环境：

```bash
cd siblingrestore-pilot-server
source /home/zhanghangning/.venv/bin/activate  # 本机已有 torch 的环境；按实际路径调整
python -m pip install -r requirements.txt
python check_bundle.py
PYTHON=/home/zhanghangning/.venv/bin/python bash run_smoke.sh
```

`run_smoke.sh` 和 `run_pilot.sh` 会自动定位项目目录，也支持通过 `PYTHON=/path/to/python` 指定解释器。

如果已有可用的 PyTorch 环境，直接安装 `numpy` 和 `Pillow` 即可。`check_bundle.py` 应显示 `status: passed`；三条 smoke 均应生成 `best.pt`。smoke 只跑 1 step，输出数值不能用于判断模型效果。

## 第一阶段正式潜力测试

单卡依次运行：

```bash
bash run_pilot.sh
```

也可以分开运行，方便中断或分配不同 GPU：

```bash
python train.py --config configs/pilot_independent.json
python train.py --config configs/pilot_group.json
python train.py --config configs/pilot_sibling.json
```

验证采用 512×512 重叠分块，降低 1080p 航拍图的显存风险。若仍 OOM，将三个 pilot 配置中的 `eval_tile_size` 改成 256；训练 OOM 则把 independent batch 从 8 改成 4、group/sibling batch 从 4 改成 2，并保持“每步退化图数量”一致。

训练结束后，对每个 best checkpoint 运行完整 16-source 验证诊断：

```bash
python evaluate_source_fidelity.py --checkpoint runs/pilot_sibling/best.pt \
  --data-root data/plamd_vari_grip_pilot --split val \
  --output runs/pilot_sibling/source_fidelity_val.json
```

结果包括 PSNR、快速诊断 SSIM、own-clean top-1、同类别内 own-clean top-1，以及六种 sibling 恢复结果之间的 L1。后两类是方向筛选指标，不是最终论文指标。

## 是否继续扩大的判断门槛

先看 SiblingRestore 相比两个 control 是否同时满足：

- clean-reference PSNR 不低于最佳 control 超过 0.3 dB；
- 同类别 own-clean top-1 提升至少 5 个百分点，或 sibling-output L1 降低至少 10%；
- 视觉上没有结构抹平、颜色漂移或把不同设备恢复成相似模板。

若达到门槛，再用 `--seed 13/37/73` 做三次重复，并扩到更多来源；若来源指标提升但 PSNR 明显下降，先调低 `source` / `output` loss 权重；若两类指标都没有改善，就停止扩大数据，优先重设来源表征与损失，而不是直接堆更多数据。

## 单图推理

```bash
python infer.py --checkpoint runs/pilot_sibling/best.pt \
  --input path/to/degraded.jpg --output restored.png
```

当前代码是“验证研究假设的 pilot 主模型”，不是最终顶会版本。只有通过上述门槛，才值得加入检测/分割 task metric、LPIPS/DISTS、强公开 baseline 和更大数据范围。
