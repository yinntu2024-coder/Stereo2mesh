# ShipTraj-R1 论文精读与完整复现方案

> 论文：**ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large Language Models via Group Relative Policy Optimization**（arXiv:2603.02939，提交日期 2026-03-03，PAKDD 2026 接收）。

## 1. 论文核心贡献（精读版）

## 1.1 问题重述
ShipTraj-R1 将“船舶轨迹预测”从传统的数值回归任务改写为 **text-to-text 生成任务**：
- 输入：目标船历史轨迹 + 冲突邻船轨迹（prompt 内）
- 输出：`<think>...</think><answer>{trajectory:[...]}</answer>`

这样做的关键收益：
1. 把“推理过程”显式化（可解释性更强）；
2. 让模型利用冲突场景上下文，学习“避碰逻辑”；
3. 用 GRPO + 规则奖励替代单纯监督回归。

## 1.2 方法结构（对应论文主线）
论文方法是三件事的组合：
1. **动态冲突提示（Prompt）**：输入里显式加入高冲突风险邻船轨迹；
2. **规则奖励（Rule-based reward）**：同时奖励“输出格式合法性”和“预测精度”；
3. **GRPO 强化微调（RFT）**：基于 group-relative advantage 更新策略。

## 1.3 奖励函数设计要点
奖励由两部分构成：
- **格式奖励**：是否严格输出 `<think>` + `<answer>`，且 `trajectory` 长度等于 `Tpred`；
- **精度奖励**：预测轨迹与真值轨迹的地理距离误差（文中提到使用 Vincenty 距离）。

这意味着：**它不是纯语言对齐 RL，而是“结构约束 + 几何误差”联合优化。**

## 1.4 关键实验设定与结果（复现目标）
论文实验关键参数：
- 数据：AIS 两个区域数据集 CSJP / CFDP；
- 采样：5 秒间隔；
- 历史长度：`Tobs ∈ [4, 8]`（20s / 40s）；
- 预测长度：`Tpred ∈ [1,2,3,4]`（5/10/15/20s）；
- Backbone：Qwen3-8B（也有 4B）；
- 训练：LR=1e-6, weight decay=1e-2, batch size=16；
- 指标：FDE / ADE（越低越好）。

建议将以下作为“主结果复现门槛”：
- ShipTraj-R1-8B: CSJP FDE 0.001297, ADE 1.1547e-05；CFDP FDE 0.000311, ADE 3.8912e-07。

## 2. 可复现性现状评估（截至 2026-04-12）

目前论文已在 arXiv 可获取，但未检索到明确的官方开源训练代码与数据下载脚本链接。
因此给出两条复现路径：

- **路径 A（严格论文复刻）**：等待官方实现/配置发布后一比一对齐；
- **路径 B（工程复现）**：用公开 Qwen + GRPO 训练框架自行搭建，最大化逼近论文。

下文给出可直接落地的路径 B 完整方案，同时保留路径 A 对齐点位。

## 3. 完整复现方案（路径 B，可执行）

## 3.1 环境准备

```bash
conda create -n shiptraj-r1 python=3.10 -y
conda activate shiptraj-r1

# 按你机器 CUDA 版本调整
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate peft trl deepspeed wandb pandas numpy scipy pyproj shapely

python -V
nvidia-smi
```

> 说明：若你使用 8×24GB（4090）或混合集群，可优先启用 deepspeed ZeRO + bf16。

## 3.2 数据复建（AIS -> 训练样本）

### Step 1: 原始数据整理
- 字段至少包括：MMSI、timestamp、lon、lat、SOG、COG（若有）。
- 过滤规则建议：
  - 删除经纬度越界点；
  - 删除速度异常（如 > 40 节，可按场景调）；
  - 按 MMSI 分轨迹段，长时间断点切段（如 > 10 分钟）。

### Step 2: 重采样
- 与论文对齐为 5s 间隔；
- 对非等间隔点执行三次样条插值（cubic spline）。

### Step 3: 样本切片
- `Tobs in {4,8}`，`Tpred in {1,2,3,4}` 组合采样；
- 形成 `(history, future)` 对。

### Step 4: 冲突邻船构造
- 每个目标船时刻，检索邻域（如半径 1~3 km）候选邻船；
- 依据 QSD / CPA(TCPA,DCPA) 指标挑高风险邻船 Top-K（建议 K=3~5）；
- 将邻船历史轨迹作为 prompt context。

## 3.3 Prompt 与输出协议（必须严格）

### 输入模板（建议）
```text
Context: target ship past trajectory: [...]
Adjacent conflict ships:
- ship1: [...]
- ship2: [...]
...
Question: predict next P coordinates.
Return in format:
<think>...</think>
<answer>{"trajectory": [(x1,y1), ...]}</answer>
```

### 解析规则
- 必须同时出现 `<think>` 和 `<answer>`；
- `trajectory` 长度必须等于当前 `Tpred`；
- 坐标必须在区域边界内（经纬度合法 + 区域 mask 内）。

## 3.4 训练流程（推荐三阶段）

### 阶段 A：SFT 冷启动
- 用 teacher-forcing 学到基本输出格式与粗精度。
- 关键：把目标格式学稳，否则 RL 早期 reward 全是 0。

### 阶段 B：GRPO 强化微调（核心）
- 每个输入采样 M 个 completion（建议 M=4/8）；
- 对每个 completion 计算：
  - `r_format`（0/1）
  - `r_acc`（负距离误差归一化）
  - `r_total = w1*r_format + w2*r_acc`
- 组内标准化 advantage：
  - `A_i = (r_i - mean(r_group)) / std(r_group)`
- GRPO 目标加入 KL 约束到 reference policy。

### 阶段 C：多 seed 稳定性复跑
- seeds: 42/43/44；
- 输出均值±方差，避免偶然跑高。

## 3.5 推荐超参起点（基于论文与工程经验）
- model: Qwen3-8B-Instruct（或 4B 先做小规模验证）
- lr: `1e-6`
- weight_decay: `1e-2`
- global_batch_size: `16`
- max_new_tokens: 256
- kl_coef: 从 `1e-4` 起扫（建议 `[1e-4, 5e-4, 1e-3]`）
- reward 权重：`w1=0.3`（格式），`w2=0.7`（精度）作为初始值

## 3.6 评估实现

### 主指标
- **ADE**: 全时刻平均位移误差
- **FDE**: 终点位移误差

### 评估注意事项
1. 必须使用球面地理距离（Vincenty/Geodesic），不要用简单欧氏替代；
2. 按 `Tobs/Tpred` 分组汇总，再给总平均；
3. 每组至少报 3 次 seed 均值；
4. 输出违规率（格式不合法比例）应单独统计。

## 3.7 消融实验清单（对齐论文）
必须做：
1. SFT vs RFT；
2. RFT with/without CoT；
3. Prompt with/without conflicting ships；
4. KL 系数扫描（至少 `1e-4, 5e-4, 1e-3, 5e-3`）。

## 4. 复现中最容易失败的点（重点避坑）

1. **奖励稀疏**：输出格式不稳导致 `r_format=0`，训练无信号。先 SFT 把格式打牢。
2. **坐标尺度不一致**：经纬度直接做 L2 会失真，必须统一为地理测距。
3. **邻船选择错误**：仅按最近距离而非冲突风险，可能削弱方法优势。
4. **KL 过大**：会把策略拉回参考模型，性能退化（论文消融也体现这一点）。
5. **数据泄漏**：按轨迹 ID 与时间段严格切分 train/val/test，不能随机点级切分。

## 5. 一周执行排期（可直接照抄）

- Day1：数据清洗 + 5s 重采样 + 切片器
- Day2：Prompt 构建 + 解析器 + SFT baseline
- Day3：GRPO reward 实现 + 小规模冒烟
- Day4：正式 RFT（4B）+ 超参粗扫
- Day5：8B 训练 + 主指标评估
- Day6：4 项消融
- Day7：多 seed 重跑 + 复现报告

## 6. 交付物模板

- `configs/*.yaml`：全实验配置
- `scripts/train_sft.sh` / `scripts/train_grpo.sh`
- `scripts/eval.sh`
- `artifacts/checkpoints/`
- `artifacts/logs/`
- `report/repro_shiptraj_r1.md`

## 7. 给你的“最小启动清单”

你现在只要补这 4 项，我就能继续给你“命令级可运行版本（逐条命令）”：
1. 你的 GPU 资源（卡型、张数、显存）
2. 你手头 AIS 数据格式示例（10 行即可）
3. 你偏好的训练框架（TRL / verl / open-r1）
4. 你优先目标（先跑通 or 先逼近论文数值）


## 8. 仓库内可直接使用的最小实现

已提供以下文件用于快速启动：
- `src/repro_shiptraj_r1/metrics.py`：ADE/FDE + 大圆距离实现
- `src/repro_shiptraj_r1/rewards.py`：格式奖励、精度奖励、组合奖励
- `src/repro_shiptraj_r1/prompting.py`：prompt 构造与 `<answer>` 解析
- `configs/shiptraj_r1_sft.yaml` / `configs/shiptraj_r1_grpo.yaml`：配置样例
- `scripts/train_sft.sh` / `scripts/train_grpo.sh` / `scripts/eval.sh`：执行入口

### 快速验证
```bash
PYTHONPATH=src bash scripts/eval.sh
```

## 9. 一键跑通（仓库内 Demo）

```bash
# 1) SFT 数据准备
bash scripts/train_sft.sh

# 2) GRPO 风格多候选奖励选择（轻量模拟）
bash scripts/train_grpo.sh

# 3) 指标冒烟
PYTHONPATH=src bash scripts/eval.sh
```

> 注意：`tools/train_grpo.py` 目前是轻量 GRPO 思路模拟器（多候选 -> 奖励选择），用于验证流程完整性，后续可替换为 TRL/verl 的真实策略优化。

## 10. 已落地的创新增强：COLREGs 奖励

当前仓库已实现：
- `src/repro_shiptraj_r1/colregs.py`：`tcpa_dcpa` 与 `colregs_risk_reward`
- `src/repro_shiptraj_r1/rewards.py`：`colregs_reward` 与三项组合奖励
- `configs/shiptraj_r1_grpo.yaml`：可配置 `w_colregs` 与 `safety_dcpa_m`

你可以直接通过调节 `w_colregs` 做消融：
- `w_colregs=0.0`（不启用规则风险）
- `w_colregs=0.2`（默认）
- `w_colregs=0.4`（强规则）

## 11. 已落地的二阶增强：多邻船聚合 + CVaR 尾部风险

新增能力：
- 多邻船 COLREGs 奖励聚合：`multi_colregs_reward` 支持 `min/mean/softmin`
- 尾部风险奖励：`cvar_accuracy_reward(alpha)`，对最差分位误差敏感
- 在 `combined_reward` 中可与 `w_cvar` 一起使用

配置建议（`configs/shiptraj_r1_grpo.yaml`）：
- `reward.colregs_mode: softmin`
- `reward.w_cvar: 0.05`
- `reward.cvar_alpha: 0.5`

## 12. 从“候选选择”到“真实参数更新”

新增 `tools/train_grpo_policy.py`：
- 使用 `GaussianVelocityPolicy(theta, sigma)`
- 使用 group-relative advantage 的 REINFORCE 更新 `theta`
- 可用于验证“奖励是否能驱动策略参数变好”

命令：
```bash
PYTHONPATH=src python tools/train_grpo_policy.py \
  --input data/sample_shiptraj.jsonl \
  --epochs 15 --num-samples 8 --lr 1e-2 --seed 42
```

## 13. 更强策略基线：线性特征策略（无外部依赖）

新增：
- `src/repro_shiptraj_r1/policy_linear.py`
- `tools/train_grpo_linear.py`

相比 `theta` 单参数策略，线性特征策略用 `W @ [1, vlon, vlat]` 同时建模经度/纬度增量，表示能力更强。

命令：
```bash
PYTHONPATH=src python tools/train_grpo_linear.py \
  --input data/sample_shiptraj.jsonl \
  --epochs 15 --num-samples 8 --lr 1e-3 --seed 42
```
