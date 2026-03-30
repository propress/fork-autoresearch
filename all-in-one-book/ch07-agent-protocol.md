# 第七章 Agent 协议与 Prompt 设计

前六章我们拆解了 autoresearch 的全部技术栈——数据管道、模型、优化器、训练循环。但这些只是"被研究的对象"。autoresearch 的真正独特之处在于：**谁来做研究？**

答案是 AI coding agent（如 Claude、Codex）。而 agent 的行为完全由一个 114 行的 Markdown 文件定义——`program.md`。这个文件既是 agent 的"研究指南"，也是给 LLM 的 prompt。这一章我们完整分析这个 prompt 的设计。

---

## program.md 的定位

在 autoresearch 中，`program.md` 扮演了一个独特的角色：它不是代码，不会被执行，但它**控制了整个研究过程**。当用户把 AI agent 指向这个文件时，agent 会按照其中的指令自主运行实验循环。

从 prompt engineering 的角度看，`program.md` 是一个精心设计的**任务指令 prompt**——它定义了 agent 的角色、约束、流程和判断标准。

---

## Prompt 完整结构分析

`program.md` 分为 6 个段落，我们逐段分析其设计意图。

### 第一段：Setup（设置协议）

> **原文：**
>
> To set up a new experiment, work with the user to:
> 1. Agree on a run tag
> 2. Create the branch: `git checkout -b autoresearch/<tag>`
> 3. Read the in-scope files: README.md, prepare.py, train.py
> 4. Verify data exists
> 5. Initialize results.tsv with header row
> 6. Confirm and go

**设计意图**：这是一个**带人类确认的启动仪式**。几个关键设计点：

- **分支管理**：每次实验用独立分支。这确保了 agent 的修改是隔离的，人类可以随时回到 main 分支。
- **强制阅读源码**：step 3 要求 agent 阅读三个核心文件。这不只是"提供上下文"——它确保 agent 理解了代码结构后再开始修改，减少盲目实验。
- **数据检查**：step 4 预防了一个常见的失败模式——数据没准备好就跑训练。
- **人类确认**：setup 是唯一需要人类交互的阶段。确认完成后，agent 就进入完全自主模式。

### 第二段：Experimentation（实验规则）

> **原文：**
>
> **What you CAN do:**
> - Modify `train.py` — this is the only file you edit.
>
> **What you CANNOT do:**
> - Modify `prepare.py`.
> - Install new packages.
> - Modify the evaluation harness.
>
> **The goal is simple: get the lowest val_bpb.**

**设计意图**：这是 prompt 中最关键的**约束框架**。它用 CAN/CANNOT 的二元对立明确划定了 agent 的行动边界：

- **单文件约束**：agent 只能改 `train.py`。这个设计精妙之处在于——`prepare.py` 中的评估函数是"裁判"，如果 agent 能改裁判规则，实验就没有意义了。
- **目标单一化**："get the lowest val_bpb" 一句话定义了优化目标。没有歧义，没有多目标权衡（VRAM 只是"软约束"），agent 可以把所有注意力集中在一个数字上。
- **简洁性偏好**：prompt 还加入了"simplicity criterion"——同等效果下更简单的方案更好。这防止 agent 堆砌复杂但边际效果极小的改动。

### 第三段：Output format（输出格式）

> **原文**展示了训练脚本的输出格式和 grep 命令。

**设计意图**：告诉 agent 怎么"看"实验结果。`grep "^val_bpb:" run.log` 这行代码看似简单，但它定义了 agent 的**感知接口**——agent 不需要理解整个训练日志，只需要提取一个数字。这降低了 agent 的认知负担。

### 第四段：Logging results（日志格式）

> **原文**定义了 results.tsv 的格式：commit, val_bpb, memory_gb, status, description

**设计意图**：结构化的实验记录。几个设计细节：

- **TSV 而非 CSV**：prompt 明确说"tab-separated, NOT comma-separated — commas break in descriptions"——description 字段可能包含逗号，TSV 避免了解析错误。
- **status 三状态**：keep / discard / crash，完整覆盖了实验的三种结局。
- **不 commit results.tsv**：这个文件故意留在 git untracked 状态。为什么？因为它是**元数据**，不是代码。如果 agent reset 了一个失败的实验，results.tsv 不应该被回滚——它应该保留完整的实验历史。

### 第五段：The experiment loop（核心实验循环）

> **原文：**
>
> LOOP FOREVER:
> 1. Look at the git state
> 2. Tune `train.py` with an experimental idea
> 3. git commit
> 4. Run: `uv run train.py > run.log 2>&1`
> 5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
> 6. If grep empty → crashed, run `tail -n 50 run.log`
> 7. Record in tsv
> 8. If improved → keep commit
> 9. If not → git reset
>
> **NEVER STOP**: Do NOT pause to ask the human...

**设计意图**：这是整个 prompt 的灵魂——**完全自主的实验循环**。设计上有几个值得注意的点：

- **先 commit 再 run**：step 3 在 step 4 之前。这确保了每个实验都有对应的 commit，git history 就是完整的实验日志。即使 agent 进程中断，也能从 git log 中恢复状态。
- **输出重定向**：`> run.log 2>&1` 把所有输出写入文件而不是打印到终端。这避免了大量训练日志"淹没" agent 的上下文窗口——LLM 的上下文是有限的，每个 token 都很宝贵。
- **crash 处理**：step 6 给了 agent 处理崩溃的能力——通过 `tail -n 50` 读取错误信息，判断是简单的 bug（修复后重试）还是根本性问题（放弃跳过）。
- **NEVER STOP 指令**：全大写、加粗，这是 prompt 中最强烈的指令。它解决了 LLM agent 的一个常见问题——倾向于在完成几轮后停下来"请示"人类。这个指令明确告诉 agent：人类不在，你就是唯一的决策者。
- **超时处理**：10 分钟超限就 kill，防止死循环浪费时间。

### 第六段：隐含的 Prompt 技巧

虽然 `program.md` 没有使用显式的 prompt engineering 标记（如 "You are a..." 角色设定），但它运用了几种有效的技巧：

| 技巧 | 在 program.md 中的体现 |
|------|----------------------|
| **明确约束** | CAN/CANNOT 二元框架 |
| **具体示例** | results.tsv 的完整示例行，包含 keep/discard/crash 三种情况 |
| **输出格式约束** | 精确的 grep 命令和 TSV 列定义 |
| **永续指令** | "LOOP FOREVER"、"NEVER STOP"，使用全大写强化 |
| **异常处理指南** | crash 时的诊断步骤和决策规则 |
| **自主决策授权** | "Use your judgment"、"you are autonomous" |

### 动态部分

`program.md` 本身是一个**静态模板**——没有运行时注入的变量。所有的动态信息（实验结果、git 状态、run.log 内容）是 agent 在执行过程中自己获取的。这是一个纯指令性 prompt，不需要任何模板引擎。

---

## 分析笔记本：results.tsv 的可视化

`analysis.ipynb` 虽然不是 agent 使用的，但它完成了实验循环的"闭环"——让人类能看到 agent 的研究成果。

笔记本做了四件事：
1. **加载 results.tsv**，统计 keep/discard/crash 的分布
2. **绘制 val_bpb 随时间的变化图**，包括 running minimum（每一刻的历史最佳）
3. **标注每个 kept 实验**的描述文字
4. **排行榜**：按改进幅度排序所有 kept 实验

这个可视化回答了人类最关心的问题：agent 做了多少实验？成功率多高？最大的改进来自什么？

---

## program.md 的设计哲学

回顾整个 prompt，它的设计哲学可以概括为三点：

1. **最小权限原则**：agent 只能改一个文件，不能改评估标准，不能装新包——把破坏的可能性控制到最小。
2. **最大自主原则**：一旦 setup 完成，agent 完全自主决策——改什么、怎么判断、怎么恢复，都由 agent 自己决定。
3. **可恢复性原则**：git commit + results.tsv 的设计让实验过程完全可追溯、可恢复。即使 agent 崩溃了，人类可以从 git log 和 results.tsv 中完整还原实验历史。

---

## 本章小结

这一章我们从 prompt 设计的角度分析了 `program.md`：

- **Setup 协议**：带人类确认的启动仪式，之后 agent 完全自主
- **约束框架**：CAN/CANNOT 明确边界，单一优化目标
- **实验循环**：先 commit 再 run，输出重定向保护上下文，crash 诊断和恢复
- **永续指令**：NEVER STOP，解决 LLM 的"请示"倾向
- **分析闭环**：analysis.ipynb 让人类可视化 agent 的研究进展

到这里，我们理解了 autoresearch 的所有组成部分。下一章我们回过头来，通过 commit 历史看这个项目是怎么一步步"长"成现在这个样子的。

---

### 质检报告

**讲解节奏**
- [x] 先讲 program.md 的定位，再逐段分析

**Prompt 分析**
- [x] prompt 结构完整展示（通过引用原文关键段落）
- [x] 逐段分析了设计意图
- [x] 动态变量来源已说明（无动态注入，agent 自行获取）
- [x] 识别了使用的 prompt 技巧（约束框架、示例、永续指令等）

**周边知识**
- [x] 解释了为什么用 TSV 而非 CSV
- [x] 解释了输出重定向保护上下文窗口的原因
- [x] 解释了 NEVER STOP 解决的 LLM 行为倾向

**讲透了吗**
- [x] 6 个段落逐段分析
- [x] 设计哲学总结

**代码纪律**
- [x] 全章代码片段 0 处
- [x] prompt 原文通过引用块展示（符合 prompt 分析例外规则）

**流程图准确性**
- [x] 本章无自绘流程图（分析性章节）

**过渡自然吗**
- [x] 章头承接前六章（技术栈已完整，转向 agent 协议）
- [x] 章尾引出下一章（项目演进史）
- [x] 章内按 prompt 段落顺序衔接

**准确吗**
- [x] program.md 引用与原文一致
- [x] results.tsv 格式描述正确
- [x] 实验循环步骤与原文一致

**读得下去吗**
- [x] 每段分析都有"设计意图"解读
- [x] 用表格总结 prompt 技巧
