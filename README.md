# SiblingRestore PLAMD 单类别潜力测试包

这是一个可直接上传 Linux 服务器的**小范围可行性实验**，目的不是立即产出论文表格，而是先回答：利用同一电力设备来源的多种退化图作为 sibling supervision，是否比普通独立成对恢复更能保持设备来源身份，同时不牺牲恢复质量。

## 包内内容

- `data/plamd_vari_grip_pilot/`：已物化的数据，不依赖原始大压缩包。
- 100 个 `vari-grip` 来源，按来源互斥划分为 train/val/test = 71/16/13。
- 每个来源含 1 张 clean 和 blur、haze、inpainting、lowlight、rain、snow 各 1 张，共 700 张 JPEG（约 87.6 MiB）。
- 三条公平训练路径、验证、来源保持诊断、单图推理和数据校验代码。

`metadata/audit.json` 已记录数据数量、解码检查及索引哈希。任何 train/val/test 划分都以 `source_id` 为单位，避免同一来源泄漏到不同集合。

数据来源为 [PLAMD Zenodo 记录](https://zenodo.org/records/21321669)。本包只用于将你已下载的数据搬到自己的研究服务器；使用或进一步分发前仍应遵守原数据页面列出的许可和引用要求。当前 PLAMD 证据属于受控/合成退化实验，不能直接表述为真实野外鲁棒性。

## 当前最佳模型（SOTA）

在当前 `plamd_sfr_v1` 数据集、7 种退化、`seed=13`、`sibling` 训练协议和统一 frozen-verifier `v2 + LPIPS` 评估口径下，**Safe-refine 是当前项目的 SOTA 模型**。它从 M3 ALCRB checkpoint 初始化，加入 identity-safe gated residual 和 frozen-verifier distillation。

这里的 SOTA 是本项目当前实验范围内的结果，不代表已在所有公开数据集或所有方法上取得领域 SOTA。当前结果仍是单 seed；多 seed 和独立 embedding 验证是后续稳健性工作。

| Split | Sources / views | PSNR | SSIM | Restored Top-1 | ROC-AUC | EER | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val | 128 / 896 | **28.2325** | **0.94557** | **0.7344** | **0.9726** | **0.07354** | **0.14565** |
| Test | 256 / 1792 | **28.8228** | **0.94536** | **0.6802** | **0.9729** | **0.08373** | **0.14420** |

与原始 M3 ALCRB 相比，Safe-refine 在 test 上 PSNR 提升 `+0.7750 dB`、Restored Top-1 提升 `+0.0485`、ROC-AUC 提升 `+0.0061`、EER 降低 `0.0102`、LPIPS 降低 `0.0234`。

### 发布权重

最新 Safe-refine 权重：

```text
weights/psnr_safe_refine/m3_alcrb_safe_seed13/best.pt
```

- 文件大小：约 31.4 MB
- SHA256：`70fa82f52e39234ecb113873688cba55eaf95ba68a2501b90169941719d6cd58`
- 配置：`configs/psnr_safe_refine_m3/seed13.json`
- 训练输出和完整日志保留在本地 `runs/`，不纳入仓库。

使用发布权重进行单图推理：

```bash
python infer.py \
  --checkpoint weights/psnr_safe_refine/m3_alcrb_safe_seed13/best.pt \
  --input path/to/degraded.jpg \
  --output restored.png
```

完整实验数据见 [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)。

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

## 早期 Pilot 说明

本节中的 `pilot` 命令和 100-source 小规模数据仅用于复现实验管线与早期假设验证；当前项目的正式最佳模型和完整结果请以“当前最佳模型（SOTA）”章节、`EXPERIMENT_RESULTS_ALL.md` 以及 `plamd_sfr_v1` 结果文件为准。

