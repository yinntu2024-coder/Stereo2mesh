# 论文复现完整方案（通用模板）

> 当前仓库未包含论文正文、题目、链接、方法细节或官方代码。本文档先提供“可直接执行”的完整复现框架。
> 当补充论文信息后，可将其中占位符替换为具体内容。

## 0. 输入材料清单（先补齐）

- 论文标题：`<Paper Title>`
- arXiv/会议链接：`<URL>`
- 官方代码仓库：`<URL>`
- 预训练权重（如有）：`<URL>`
- 数据集下载与许可证：`<URL>`
- 补充材料/附录：`<URL>`
- 目标硬件：GPU 型号、显存、CUDA 版本

## 1. 复现目标定义（成功标准）

1. **主结果复现**：达到论文主表 Top-1 / mIoU / PSNR 等指标的误差范围 ±1%（或论文常见波动范围）。
2. **关键消融复现**：至少复现 2~3 项核心 ablation，趋势一致。
3. **效率指标复现**：推理速度、参数量、FLOPs、显存峰值与论文声明相符。
4. **可复用产物**：训练脚本、评估脚本、日志、权重、环境锁定文件齐全。

## 2. 环境冻结（可复现的核心）

### 2.1 系统与依赖

- 操作系统：Ubuntu 22.04（建议）
- Python：3.10.x
- CUDA / cuDNN：按论文和官方仓库对齐
- 深度学习框架：PyTorch `<version>`

### 2.2 环境构建

```bash
# 1) 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2) 固定 pip
python -m pip install -U pip==24.2

# 3) 安装核心依赖（替换为论文真实版本）
pip install torch==<ver> torchvision==<ver> --index-url https://download.pytorch.org/whl/cu<cuda>
pip install -r requirements.txt

# 4) 冻结环境
pip freeze > requirements-lock.txt
python -V > .python-version.txt
nvidia-smi > .nvidia-smi.txt
```

### 2.3 随机性控制

- 固定随机种子：`seed=42`（并额外测 3 个 seed）
- 配置 cudnn deterministic / benchmark
- 记录所有影响随机性的开关

## 3. 数据集复现路径

1. 下载原始数据并校验哈希（MD5/SHA256）。
2. 记录数据划分来源（官方 split 或论文自定义 split）。
3. 数据预处理脚本版本化（resize、crop、normalize、tokenize 等）。
4. 保存可复用索引：`train.txt / val.txt / test.txt`。
5. 输出数据统计：类别分布、样本数量、异常样本列表。

## 4. 方法还原与实现映射

### 4.1 从论文到代码的逐段映射

建立“论文公式/模块 -> 代码实现”对照表：

- 模块 A（论文 §3.1, Eq.1-3） -> `model/module_a.py`
- 损失函数（§3.2） -> `losses/*.py`
- 训练策略（§4.1） -> `train.py + config/*.yaml`

### 4.2 易错点检查

- 默认超参数是否与论文一致
- BN/Dropout 在训练与评估阶段切换是否正确
- 输入归一化与数据增强顺序是否一致
- 推理时是否使用 EMA 权重 / TTA

## 5. 训练计划（分阶段）

### 阶段 A：冒烟测试（0.5 天）

- 用 1% 数据跑通 1~3 epoch
- 验证 loss 下降、无 NaN、评估脚本可运行

### 阶段 B：小规模对齐（1 天）

- 用 10% 数据进行参数敏感性检查
- 找到稳定学习率区间、batch size 上限

### 阶段 C：全量训练（2~5 天）

- 严格按论文超参数训练
- 每 N step 记录：train/val 指标、学习率、梯度范数、显存

### 阶段 D：重跑统计（1~2 天）

- 至少 3 个随机种子
- 报告均值 ± 标准差

## 6. 评估与对比

1. 完整复现主表：与论文逐项对比（绝对差值 + 相对差值）。
2. 复现关键图：PR 曲线、混淆矩阵、可视化案例。
3. 做误差归因：数据、实现、超参数、硬件差异。

## 7. 消融实验最小集合

- 去掉关键模块 1
- 去掉关键模块 2
- 改变损失权重
- 改变输入分辨率

每项至少跑 1 次，关键项跑 3 seed。

## 8. 实验管理与产出规范

- 日志：TensorBoard/W&B（记录 git commit hash）
- 配置：所有实验配置 YAML 化，不在代码硬编码
- 权重命名：`<paper>_<dataset>_<seed>_<date>.pt`
- 结果归档：
  - `artifacts/checkpoints/`
  - `artifacts/logs/`
  - `artifacts/plots/`
  - `artifacts/tables/`

## 9. 风险与备选策略

- 显存不足：梯度累积、混合精度、减小分辨率
- 训练不稳定：warmup、梯度裁剪、调低学习率
- 指标偏差大：先对齐预处理和评估脚本，再调参
- 论文细节缺失：优先以官方代码为准，并在报告中声明

## 10. 时间线（示例）

- Day 1：环境+数据+冒烟
- Day 2：小规模对齐
- Day 3~5：全量训练
- Day 6：消融+重跑
- Day 7：报告与可复现打包

## 11. 最终交付清单

- `README_repro.md`（一键运行说明）
- `requirements-lock.txt` / `environment.yml`
- 训练与评估脚本
- 复现结果表格（含与论文差值）
- 失败案例分析
- 可下载模型权重与日志

## 12. 你补充论文后我将立即给出“定制版”

请补充以下任一项：

1. 论文标题 / 链接（arXiv/会议）
2. 你当前打算复现的任务（检测/分割/3D重建/多模态等）
3. 你的硬件配置（GPU、显存）

收到后我会输出：

- 针对该论文的逐节技术解读
- 精确到命令级别的复现步骤
- 预期结果区间与常见报错修复手册
