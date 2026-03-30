# 项目实现原理全解 — 写作进度

## 章节规划

| # | 章节标题 | 文件名 | 核心覆盖 | 特殊内容 | 状态 |
|---|---------|--------|---------|---------|------|
| 0 | 序言：先建立全局心智模型 | ch00-preface-map.md | 项目定位、架构全景图、核心概念词典、代码库地图、一次典型交互极简全流程 | 含全景 flowchart | ⏳ |
| 1 | 从一次 5 分钟实验看全链路数据流 | ch01-end-to-end-dataflow.md | 从 `uv run train.py` 到 `val_bpb` 输出的主干路径；输入/中间张量/输出的形态变化 | 数据流全景图；调用路径主线 | ⏳ |
| 2 | prepare.py 黑盒打开：数据与 tokenizer 如何被“固定下来” | ch02-prepare-data-tokenizer.md | 下载分片、训练 tokenizer、生成 token_bytes、运行时 dataloader/eval 工具边界 | 函数级流程图 | ⏳ |
| 3 | train.py 第一层：训练入口、配置折叠与时间预算调度 | ch03-train-bootstrap-and-schedule.md | 超参数如何映射到模型配置；编译/预热/计时策略；为什么固定 wall-clock 是核心约束 | 关键调度路径 | ⏳ |
| 4 | 模型主干实现：GPT 变体、窗口注意力与 Value Embedding | ch04-model-architecture-core.md | `GPTConfig`、Block 结构、rotary、window pattern、VE gate、残差混合参数 | 模块内部 flowchart | ⏳ |
| 5 | 优化器系统：Muon + AdamW 的组合机制与参数分组哲学 | ch05-optimizer-muon-adamw.md | 参数分组、fused step、动量与权重衰减调度、为何按参数类型分治优化 | 同类对比：nanochat/传统 AdamW-only | ⏳ |
| 6 | 训练循环逐拍拆解：每一步到底改了什么 | ch06-training-loop-step-by-step.md | micro-step 累积、loss 检查、lr/momentum/wd 更新、吞吐与 MFU 指标的形成 | sequenceDiagram + 数据形态注释 | ⏳ |
| 7 | LLM 项目核心：program.md 如何“编排”自治研究循环 | ch07-program-md-prompt-analysis.md | `program.md` 原文分段解析；setup 与 loop 约束如何塑造 agent 行为 | **prompt 分析（完整原文+逐段注解）** | ⏳ |
| 8 | 项目演进史：从最小可运行到当前形态（36 commits） | ch08-evolution-history.md | 按阶段还原架构演进：初版→稳定化→鲁棒性修补→文档与生态扩展 | 时间线阶段图；关键 commit 转折 | ⏳ |
| 9 | 端到端验收：把“代码机制”与“自治实验组织”串成闭环 | ch09-e2e-walkthrough-and-checklist.md | 选一个完整场景追踪：准备→训练→评估→记录→保留/回滚决策 | 端到端调用路径总复盘 | ⏳ |

## 章节规划说明

认知路径采用“先黑盒、后白盒、再回闭环”：

1. 先在序言建立“这个项目是什么、为什么要这样设计”的底图。  
2. 再用一条真实主干（5 分钟实验）建立读者的时间轴感和数据流感。  
3. 然后分别拆开两大技术支柱：`prepare.py`（数据与评估基准）与 `train.py`（模型+优化+训练循环）。  
4. 在读者已经理解代码路径后，再进入 `program.md`（LLM agent 行为编排）的 prompt 级实现逻辑。  
5. 最后通过演进史解释“为什么今天是这个样子”，并以端到端验收把全书合拢。

## 同类项目对比规划

### 对比点 A（放在第 5 章）
- 问题：在相同训练目标下，优化器如何组织参数更新策略。  
- 本项目：Muon（2D 矩阵参数）+ AdamW（其他参数）分治。  
- 对比项目：nanochat（同源工程背景，优化器组织可直接同域比较）、常见 GPT 训练脚本（AdamW-only）。  
- 满足前提说明：
  - 问题域一致：都是语言模型预训练优化；
  - 输入输出一致：输入梯度/参数，输出更新后参数；
  - 思路差异明确：分治优化 vs 单优化器统一更新。

### 对比点 B（放在第 7 章）
- 问题：自治实验流程如何被“文本规约”约束执行。  
- 本项目：`program.md` 用自然语言协议直接规定 setup/loop/日志格式。  
- 对比项目：OpenHands / SWE-agent 一类项目中的任务协议模板（同为 LLM agent 行为约束）。  
- 满足前提说明：
  - 问题域一致：都在约束 agent 执行多步任务；
  - 输入输出一致：输入任务上下文，输出可执行动作序列与记录；
  - 思路差异明确：本项目强调“单文件可改 + 永不停机实验循环”的研究组织约束。

> 注：仅在满足“三前提”时嵌入对比，不做泛化产品横评。

## 状态说明
- ✅ 已完成
- 🔄 进行中
- ⏳ 待开始

当前状态：已完成初次仓库扫描与历史梳理，等待开始 `ch00` 写作。

## 术语约定

### 行业标准术语（直接使用）
- tokenizer、BPE、dataloader、attention、rotary embedding、optimizer、warmup/warmdown、gradient accumulation、bits per byte (BPB)

### 项目特有术语（首次出现会做类比说明）
- autoresearch：把“研究流程”本身当作可迭代程序（类比 CI pipeline，但目标是模型效果改进）
- fixed 5-minute budget：固定 wall-clock 预算评测协议（类比固定 benchmark timebox）
- program.md：agent 的“组织操作手册”（类比 SOP + prompt policy）
- keep/discard branch advancement：实验结果驱动的提交保留策略（类比 A/B 试验中的胜者晋级）

## 下次续写指引

### 从哪里继续
从 `ch00-preface-map.md` 开始，先产出“序言 + 全景图 + 一次极简交互流程”。

### 交接备忘
- 已确认仓库当前不存在 `all-in-one-book/PROGRESS.md`，本文件为首次规划稿。  
- 已读取核心文件：`README.md`、`prepare.py`、`train.py`、`program.md`。  
- 已读取完整 commit 历史（36 commits，适合单章演进史）。  
- 当前环境缺少 `uv` 与 `pytest` 可执行能力，无法在本地跑测试基线；后续以文档改动为主，无运行时行为变更。

### 待验证项
- [需源码验证] 第 5 章 Muon 机制讲解时，是否需要补充与上游 nanochat 的具体实现差异点（当前只完成对比点预案）。  
- [需源码验证] 第 7 章对比 OpenHands / SWE-agent 时，需在写作当次抓取其最新 prompt/协议模板，避免过时描述。  
