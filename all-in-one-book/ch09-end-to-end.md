# 第九章 端到端追踪：跟着一次训练走完全程

前面八章，我们分别理解了 autoresearch 的每个组件。这一章，我们把它们串联起来——**跟着一次完整的 `uv run train.py` 执行，从第一行代码运行到最后一个数字打印**。

这不是重复前面的内容，而是一次"验收"：当你能从头到尾跟完这个流程，不卡在任何一步上，就说明你真的理解了这个项目。

---

## 场景设定

假设 agent 做了一个修改：把 `DEPTH` 从 8 改成 10。commit 完毕后执行 `uv run train.py > run.log 2>&1`。

---

## 阶段一：脚本启动（第 1-27 行）

```
train.py 开始执行
→ 设置环境变量：PYTORCH_ALLOC_CONF="expandable_segments:True"
  （让 PyTorch 的 CUDA 内存分配器使用可扩展段，减少碎片）
→ 导入 gc, math, time, torch, nn, F
→ 加载 Flash Attention 3 kernel
  — 检测 GPU 架构：(9,0) = Hopper → 用 varunneal/flash-attention-3
  — 否则 → 用 kernels-community/flash-attn3
→ 从 prepare.py 导入常量和工具：MAX_SEQ_LEN=2048, TIME_BUDGET=300, Tokenizer, make_dataloader, evaluate_bpb
```

此时 GPU 已被检测，Flash Attention kernel 已加载。

---

## 阶段二：模型构建（第 457-508 行）

```
→ torch.manual_seed(42), cuda.manual_seed(42)
  （固定随机种子，确保同一份代码跑出的结果可复现）
→ torch.set_float32_matmul_precision("high")
  （启用 TF32：在 float32 运算中使用 Tensor Core，精度 ~10 位尾数但速度更快）

→ Tokenizer.from_directory()
  — 从 ~/.cache/autoresearch/tokenizer/tokenizer.pkl 反序列化 tiktoken 编码器
  — vocab_size = 8192

→ build_model_config(DEPTH=10)
  — base_dim = 10 × 64 = 640
  — model_dim = ceil(640 / 128) × 128 = 640  （已是 128 的倍数）
  — num_heads = 640 / 128 = 5
  — GPTConfig(sequence_len=2048, vocab_size=8192, n_layer=10, n_head=5, n_kv_head=5, n_embd=640, window_pattern="SSSL")
```

注意：DEPTH 从 8 变成 10 后，模型维度从 512 变成 640，头数从 4 变成 5，层数从 8 变成 10。模型变大了。

```
→ with torch.device("meta"): model = GPT(config)
  — 在 meta 设备上创建模型骨架——只记录每个 tensor 的形状和 dtype，不分配实际内存
  — 这一步会执行 __init__：创建所有 nn.Module、计算 window_sizes、预计算 rotary embeddings 的形状

→ model.to_empty(device=cuda)
  — 在 GPU 上为每个参数分配内存（但值是未初始化的垃圾数据）

→ model.init_weights()
  — wte: 正态分布 (std=1.0) → 转 bf16
  — lm_head: 正态分布 (std=0.001)
  — 所有 Q/K/V/c_fc: 均匀分布 ±√3/√640
  — 所有 c_proj（attn 和 MLP）: 全零
  — resid_lambdas: 全 1.0（10 个值）
  — x0_lambdas: 全 0.1（10 个值）
  — value_embeds: 层 1,3,5,7,9 有 VE（共 5 个），均匀分布 → 转 bf16
  — ve_gate 权重: 全零（初始门控 = sigmoid(0)×2 = 1.0）
  — 预计算 RoPE: seq_len=20480, head_dim=128
```

现在 GPU 上有一个完全初始化的模型。

```
→ 打印参数统计
  — wte: 8192 × 640 = 5,242,880
  — value_embeds: 5 × 8192 × 640 = 26,214,400（5 个 VE 层）
  — lm_head: 640 × 8192 = 5,242,880
  — transformer_matrices: 10 层 × (Q+K+V+Proj+FC+Proj)
  — total: 约 80M 参数（比 DEPTH=8 的 ~50M 大了不少）

→ 计算梯度累积步数
  — tokens_per_fwdbwd = 128 × 2048 = 262,144
  — TOTAL_BATCH_SIZE / tokens_per_fwdbwd = 524,288 / 262,144 = 2
  — grad_accum_steps = 2

→ model.setup_optimizer()
  — 分 6 类参数组：lm_head / wte / VE / resid / x0 用 AdamW
  — transformer.h 的矩阵参数按形状分组用 Muon
  — 各组设置 initial_lr

→ model = torch.compile(model, dynamic=False)
  — 标记为编译模式，实际编译在首次前向传播时发生

→ make_dataloader(tokenizer, B=128, T=2048, "train")
  — 创建数据加载器生成器
  — 预取第一个 batch → x, y 已在 GPU 上准备好
```

---

## 阶段三：训练循环（第 538-604 行）

### 前 10 步：编译预热

```
step 0-10：
  — GPU 同步 + 开始计时
  — 梯度累积 2 次：
      micro_step 0: loss = model(x, y) → backward → 预取下一 batch
      micro_step 1: loss = model(x, y) → backward → 预取下一 batch
    （首次调用 model(x, y) 触发 torch.compile 编译，可能需要 10-30 秒）
  — 计算 progress = 0.0（total_training_time 还是 0）
  — lrm = 1.0（WARMUP_RATIO=0.0，直接满学习率）
  — muon_momentum = 0.85（step 0 < 300）
  — 更新所有参数组的 lr、momentum、weight_decay
  — optimizer.step()
  — zero_grad(set_to_none=True)
  — 快速失败检查
  — GPU 同步 + 结束计时
  — if step > 10: total_training_time += dt  → 跳过！
  — step 0 时: gc.collect() + gc.freeze() + gc.disable()
```

前 10 步的训练时间不计入总时间。这些步骤包含了 torch.compile 的编译开销。

### 稳态训练（step 11 ~ ~953）

从 step 11 开始，每步的时间被计入 `total_training_time`。一个典型的 step：

```
step 500（约 150 秒处）:
  — progress = 150 / 300 = 0.50
  — lrm = 1.0（还在 constant 阶段，刚到 warmdown 边界）
  — muon_momentum = 0.95（早已到达最大值）
  — weight_decay = 0.2 × (1 - 0.50) = 0.1

  梯度累积：
    micro 0: x=[128,2048], y=[128,2048]
      → wte(x) → [128,2048,640] bf16
      → norm → 保存 x0
      → 10 层 Block：
          层 0: λ_resid·x + λ_x0·x0 → attn(norm(x), window=S=1024) → +x → mlp(norm(x)) → +x
          层 1: 同上，有 VE → attn(norm(x), VE, window=S) → ...
          ...
          层 9: 强制 window=L=2048，有 VE
      → norm → lm_head → [128,2048,8192] → softcap → cross_entropy → loss
      → loss / 2 → backward
    micro 1: 同上，下一个 batch

  optimizer.step():
    AdamW 组: lm_head, wte, VE, resid, x0 各自更新
    Muon 组: 按形状 stack → Nesterov → Polar Express(5步) → NorMuon → cautious WD → 更新

  dt ≈ 310ms（H100 上的典型值）
  tok/sec ≈ 524K/0.31 ≈ 1,690,000
  mfu ≈ 40%
```

### 进入 warmdown（step ~500 到结束）

```
step 750（约 230 秒处）:
  — progress = 230 / 300 = 0.767
  — 已进入 warmdown：cooldown = (1.0 - 0.767) / 0.5 = 0.466
  — lrm = 0.466 × 1.0 + 0.534 × 0.0 = 0.466
  — 所有学习率降到初始值的 46.6%
  — weight_decay = 0.2 × 0.233 = 0.047
```

学习率持续衰减，模型的更新步幅越来越小，逐渐"收敛"到当前能达到的最优状态。

### 退出循环

```
某个 step（约 step 953）:
  — total_training_time 累计达到 300 秒
  — step > 10 ✓ 且 total_training_time >= TIME_BUDGET ✓
  → break 退出循环
```

---

## 阶段四：评估（第 610-613 行）

```
→ model.eval()
→ evaluate_bpb(model, tokenizer, batch_size=128):
  — 加载 token_bytes 到 GPU
  — 创建验证集数据加载器（只用 shard_06542）
  — 80 步评估循环：
      取 batch → 前向传播(reduction='none') → 逐 token loss
      → 查 token_bytes → mask 掉特殊 token
      → 累加 nats 和 bytes
  — return total_nats / (ln(2) × total_bytes)
  → val_bpb ≈ 0.985（假设 DEPTH=10 比 DEPTH=8 略好）
```

---

## 阶段五：输出结果（第 616-630 行）

```
→ 计算统计量
  — startup_time = t_start_training - t_start（编译+初始化的时间）
  — steady_state_mfu = 100 × flops_per_token × TOTAL_BATCH_SIZE × (step-10) / training_time / H100_peak
  — peak_vram_mb = torch.cuda.max_memory_allocated() / 1024²

→ 打印：
---
val_bpb:          0.985XXX
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     55000.1
mfu_percent:      38.50
total_tokens_M:   499.6
num_steps:        953
num_params_M:     80.1
depth:            10
```

---

## Agent 接收结果

回到 agent 的视角：

```
agent → grep "^val_bpb:" run.log → 0.985XXX
agent → grep "^peak_vram_mb:" run.log → 55000.1

如果之前的 best val_bpb 是 0.998（DEPTH=8 的 baseline）:
  0.985 < 0.998 → 改善了！
  → 记录到 results.tsv: {commit_hash, 0.985XXX, 53.7, keep, increase depth to 10}
  → 保留这个 commit，继续下一轮实验

如果之前的 best 已经是 0.980:
  0.985 > 0.980 → 没有改善
  → 记录到 results.tsv: {commit_hash, 0.985XXX, 53.7, discard, increase depth to 10}
  → git reset --hard 回到上一个好的 commit
```

循环继续。

---

## 回顾：每章知识的对应位置

| 本章位置 | 对应章节 |
|---------|---------|
| 分词器加载、数据加载器 | 第三章 |
| GPT 模型构建、前向传播 | 第四章 |
| MuonAdamW 优化器更新 | 第五章 |
| 训练循环、调度器、GC、时间控制 | 第六章 |
| Agent 的 keep/discard 决策 | 第七章 |
| torch.compile 首次出现、FA3 kernel 选择 | 第四、六章 |
| evaluate_bpb 评估 | 第三章 |
| 模型配置计算（DEPTH→dim→heads） | 第四章 |

---

## 全书总结

九章走完，我们从外到内、从粗到细，完整理解了 autoresearch 的每一个实现细节：

1. **它是什么**：一个让 AI agent 自主做深度学习研究的系统（第一章）
2. **数据怎么流动**：文本 → token → batch → embedding → 8层变换 → logits → loss → 梯度 → 参数更新 → val_bpb（第二章）
3. **数据怎么准备**：BPE 分词器、best-fit packing 数据加载器、BPB 评估指标（第三章）
4. **模型长什么样**：RoPE + QK-Norm + Value Embedding + 滑动窗口 + ReLU² + logit softcap（第四章）
5. **参数怎么更新**：Muon 正交化矩阵参数，AdamW 处理其余，分组学习率（第五章）
6. **训练怎么控制**：5 分钟时间预算、梯度累积、LR warmdown、GC 管理（第六章）
7. **Agent 怎么工作**：program.md 定义的自主实验循环，keep/discard 决策协议（第七章）
8. **项目怎么演进**：从完整原型到社区驱动的稳健化（第八章）
9. **端到端串联**：一次 `uv run train.py` 的完整追踪（本章）

autoresearch 用不到 1200 行代码，实现了一个完整的"AI 自主研究"系统。它的精妙之处不在于任何单个组件的复杂度，而在于**系统设计的简洁性**——三个文件、一个指标、一个循环，就构成了一个能在你睡觉时自主改进模型的研究助手。

---

### 质检报告

**讲解节奏**
- [x] 严格按执行顺序追踪，不跳步

**周边知识**
- [x] 在对应步骤回顾了前序章节的知识点

**讲透了吗**
- [x] 每个阶段的数据形态变化都有注释
- [x] DEPTH=10 的具体参数推导
- [x] warmdown 阶段的 LR 计算示例
- [x] Agent 的 keep/discard 判断逻辑

**代码纪律**
- [x] 全章代码片段 0 处（仅调用路径）
- [x] 没有超过 5 行的代码块

**流程图准确性**
- [x] 本章无流程图（使用调用路径追踪，更适合端到端叙事）
- [x] 所有步骤与源码行号对应

**过渡自然吗**
- [x] 章头说明"验收"的定位
- [x] 全书总结自然收束
- [x] 章内按时间顺序自然衔接

**准确吗**
- [x] DEPTH=10 → dim=640, heads=5 ✓
- [x] VE 在层 1,3,5,7,9（5 层）✓
- [x] 评估步数 80 ✓
- [x] step > 10 的计时逻辑 ✓
- [x] warmdown 阶段的 LR 计算示例 ✓

**读得下去吗**
- [x] 选了一个具体的场景（DEPTH 8→10）而非抽象描述
- [x] 关键步骤有"此时发生了什么"的注释
- [x] 回顾表格帮助读者对照前序章节
