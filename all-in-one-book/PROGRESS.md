# 项目实现原理全解 — 写作进度

## 章节规划

| # | 章节标题 | 文件名 | 核心覆盖 | 特殊内容 | 状态 |
|---|---------|--------|---------|---------|------|
| 1 | 序章：全书地图 | ch01-overview.md | 项目定位 / 架构全景图 / 核心概念词典 / 代码库地图 / 一次实验的极简全流程 | | ✅ |
| 2 | 数据流全景：一次实验的完整旅程 | ch02-data-flow.md | 从 agent 发起实验到 val_bpb 产出的完整数据流，每步拆解数据形态变化 | | ✅ |
| 3 | 数据准备与加载：从原始文本到训练张量 | ch03-data-pipeline.md | 数据下载 / BPE 分词器训练 / best-fit packing 数据加载器 / BPB 评估指标 | | ✅ |
| 4 | GPT 模型架构：每一层在做什么 | ch04-model.md | GPTConfig / RoPE 旋转位置编码 / QK-Norm / Value Embedding / 滑动窗口注意力 / ReLU² MLP / 残差缩放 / logit softcap | 同类对比：nanochat/nanoGPT 架构选择 | ✅ |
| 5 | MuonAdamW 优化器：矩阵参数的特殊待遇 | ch05-optimizer.md | Muon 正交化（Polar Express）/ NorMuon 方差缩减 / AdamW 融合 / 分组学习率策略 / cautious weight decay | 同类对比：标准 AdamW vs Muon | ⏳ |
| 6 | 训练循环与调度策略 | ch06-training-loop.md | 时间预算控制 / 梯度累积 / LR warmup-cooldown / momentum 调度 / GC 管理 / 快速失败检查 | | ⏳ |
| 7 | Agent 协议与 Prompt 设计 | ch07-agent-protocol.md | program.md 作为 prompt 的完整分析 / 实验循环协议 / keep/discard 决策 / 分析笔记本 | prompt 分析 | ⏳ |
| 8 | 项目演进史：从初始提交到当前形态 | ch08-evolution.md | 35 个 commit 的里程碑式演进 / 设计决策的时间线 / 架构从简到繁的过程 | | ⏳ |
| 9 | 端到端追踪：跟着一次训练走完全程 | ch09-end-to-end.md | 从 `uv run train.py` 启动到 val_bpb 输出的逐行追踪，串联全书知识 | | ⏳ |

## 章节规划说明

认知路径设计：

1. **ch01 序章** → 读者知道"这是什么"、"整体长什么样"
2. **ch02 数据流全景** → 读者建立端到端的粗粒度理解，知道每个模块的角色
3. **ch03 数据准备** → 打开第一个黑盒：数据从哪来、怎么变成 token、怎么打包成 batch
4. **ch04 模型架构** → 打开核心黑盒：GPT 模型每一层做了什么
5. **ch05 优化器** → 理解了模型才能理解优化器为什么这样设计
6. **ch06 训练循环** → 有了模型和优化器，才能理解训练循环的调度策略
7. **ch07 Agent 协议** → 理解了技术栈，才能理解 agent 的 prompt 为什么这样写
8. **ch08 演进史** → 理解了当前形态，回看演进过程才有意义
9. **ch09 端到端追踪** → 最终验收：用一次完整实验串联全书

## 同类项目对比规划

| 对比点 | 本项目思路 | 对比项目 | 放在哪一章 |
|--------|----------|---------|----------|
| GPT 模型架构选择（Value Embedding、ReLU²、滑动窗口） | 从 nanochat 精简而来，采用多项最新技巧 | nanoGPT（经典简洁）、nanochat（完整版） | ch04 |
| 优化器设计（Muon vs AdamW 分治） | 矩阵参数用 Muon 正交化，其余用 AdamW | 标准 AdamW（PyTorch 默认）、SOAP | ch05 |

## 状态说明
- ✅ 已完成
- 🔄 进行中
- ⏳ 待开始

## 术语约定

### 行业标准术语
- BPE (Byte Pair Encoding) — 字节对编码，主流子词分词算法
- RoPE (Rotary Position Embedding) — 旋转位置编码
- MLP (Multi-Layer Perceptron) — 多层感知机
- BPB (Bits Per Byte) — 每字节比特数，语言模型评估指标
- MFU (Model FLOPs Utilization) — 模型算力利用率
- GQA (Grouped Query Attention) — 分组查询注意力
- RMS Norm — 均方根归一化

### 项目特有术语
- **Value Embedding (VE)** — 项目特有机制：为 attention 的 value 分支注入一个可学习的 token 级嵌入，来源于 ResFormer 论文。类似标准的 token embedding，但作用于 value 而非输入
- **Polar Express** — Muon 优化器中的正交化算法，用多项式近似代替 SVD 来计算矩阵的正交投影
- **NorMuon** — Muon 的方差缩减扩展，通过二阶矩估计稳定更新幅度
- **Window Pattern (SSSL)** — 项目定义的注意力窗口模式：S=半上下文滑动窗口，L=全上下文。按 layer 交替
- **autoresearch** — 项目名称，指"自主研究"范式：AI agent 自主修改代码、运行实验、迭代改进

## 下次续写指引
### 从哪里继续
从 ch05 开始写作

### 交接备忘
- 项目只有 3 个核心文件：prepare.py (389行)、train.py (630行)、program.md (114行)
- 这是 karpathy/autoresearch 的 fork，上游有 35 个 commit
- 项目不是 LLM 应用类项目（不调用 LLM API），但 program.md 是给 AI coding agent 的 prompt，prompt 分析章节适用
- train.py 的模型基于 nanochat 精简而来

### 待验证项
- Polar Express 系数的具体来源（论文引用）
- NorMuon 的原始论文引用
- Value Embedding 的 ResFormer 论文确切引用
