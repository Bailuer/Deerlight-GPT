# Deerlight GPT — 学习档案

最后更新：2026-09-11

## 先看这里：我们现在在哪

- 长期方向：从理解、实现小型 GPT，逐步发展到能提出问题、设计实验和解释结果的 LLM Research 能力。
- 当前阶段：Byte-level BPE 已实现、分模块讲解，并接入独立的小 GPT Training Pipeline。
- 最近完成：3,000 Steps BPE 实验；定期 Training/Validation Evaluation、Best/Final Checkpoint、Loss History 和 Generation。
- 当前限制：生成仍有假词、不连贯，不能正常对话；未完成 Character/BPE 的公平质量比较。
- 下一步建议（尚未执行）：先理解 BPB 的实际评估方式，再设计 Character/BPE 对照。不要直接比较两个 Tokenizer 的 per-token Loss。
- 本档案是学习导航，不是能力认证。代码跑通 ≠ 学习者能独立实现；聊天中回答正确 ≠ 长期掌握。

## 怎么记录掌握程度

用证据描述，不给主观百分比或“研究员等级”：

- **接触/讲解过**：已学习概念，但未验证独立复述。
- **有解释证据**：曾用自己的话回答相关问题，不代表整个模块全部掌握。
- **有实现证据**：注明是自己写、辅助修改，还是由助手实现后阅读。
- **有实验记录**：必须写配置、结果及结论边界。
- **待复核**：以后用一个具体问题或小任务检查，不需要每次全部重考。

## 知识地图

早期内容依据聊天记录回顾，未对全部历史文件重新审计；不虚构学习日期。

| 模块 | 已有证据 | 待复核/边界 |
|---|---|---|
| Gradient Descent、Chain Rule、MSE | 手算单 Sample 的 Gradient 和 Update；随后自行提交 Batch 练习及训练代码 | 任意 Batch Size 的 `2/N` 系数；同时用旧 Parameters 计算 Gradients |
| NumPy Vectorization | 自行写出整 Batch Predictions、Errors、Gradients；比较多个 Learning Rates | 区分 Gradient 与已经乘 Learning Rate 的 Update |
| PyTorch Autograd | 学习 Computation Graph、backward、no_grad、Gradient 清零和 Leaf Tensor | 不看代码解释为什么 Update 不记录 Graph、为什么 Gradients 会累加 |
| Multiple Features、MLP | 练过 Matrix Shapes、ReLU/GELU、训练 MLP 拟合绝对值和三分类 | 非线性与表达能力；MLP 不是每层必须使用 ReLU |
| Token Embedding、Shifted Batches | 完成 Character Tokenizer/Batch 验证；多次回答数据流问题 | Tokens 通常先 Encode 再抽 Batch；区分 Token ID、Position 和 Feature |
| Transformer 本体 | 学过 Q/K/V、Causal Mask、Multi-Head、Projection、Residual、Normalization、FFN；有代码及复盘问答 | 不看实现独立串起 `(B,T,C)` 到 `(B,T,V)`；Heads 数量与 Head Dimension 不混用 |
| Training/Generation | 解释过 Loss、Sampling、Temperature、Top-k、Eval、Checkpoint | Training Loss 不保证每步下降；eval 与 no_grad 的职责不同 |
| KV Cache、GQA/MQA | 有 Shape 问答、实现与测试记录 | Training 并行 Teacher Forcing vs Generation 追加 Token；旧 Prefix 不变才能复用 K/V |
| RoPE | 多轮数值/Pair/频率解释后表示理解 | 用一个二维 Pair 解释 Rotation 与 Relative Position；数值内容仍影响 Dot Product |
| RMSNorm、SwiGLU | 有公式和 Gate 问答；架构中已使用 | RMSNorm 不减均值，也不保证只缩小数值；Gate 是连续调制 |
| SDPA/FlashAttention | 已讲 IO、HBM、分块和 Online Softmax；有 SDPA 实现 | 使用现成 Fused Backend 不等于手写 Kernel；不能把所有 Backend 统称 FlashAttention-2 |
| MoE | 学过 Router、Top-k、Shared Experts、Load Balancing、Capacity、Dispatch/Combine；有辅助实现及实验 | 一个 Token 的完整 Representation 送入多个 Experts，Outputs 加权相加，不是切成两半或拼接 |
| Scaling Laws | 做过固定近似 Compute 的配置比较；能指出变量和数据量限制 | 一次小实验不能得到普适 Scaling Law；重复数据不等于新知识 |
| Byte-level BPE | 概念问答后，由助手完整实现并逐模块讲解；Round-trip/Save/Load 测试及真实 Dataset 实验 | 尚未验证独立实现；理解 Rank 而非 Encoding 时重新选择最高频 Pair |
| Evaluation | 已讲 Uniform Baseline、per-token CE、BPB、Best Checkpoint | BPB 尚未实现；评估 Context 与原文范围也需统一 |

## 常见卡点：以后复习这些，而不是从头重学

- **Computation Graph**：记录运算依赖；Backward 沿依赖累计对 Parameters 的贡献。
- **ReLU / MLP**：加入 Nonlinearity，避免多个 Linear 仅等价于一个 Linear。
- **Output Projection**：可训练地整合 Head Outputs，不是固定平均。
- **RoPE**：Pair 使用不同频率；Relative Position 进入 Q/K Dot Product，但不是抛弃 Q/K 内容。
- **MoE Capacity**：限制一次分配的容量，影响 Overflow/Buffer；不等同于解决 Router Collapse。
- **Tokenizer 替换**：相同 Vocabulary Size 只保证 Shape，不保证每个 ID 对应的 Bytes 一致。
- **BPE Generation**：若一个 Token 表示 hello，输出该 ID 就一次产生 hello；并非必须逐 Character。
- **Uniform Prediction**：所有候选 Probability 相同，不等于随机初始化。512 类时 CE 为 ln(512) ≈ 6.2383。
- **Loss 比较**：不同 Tokenizer 的平均 per-token CE 不是同一个单位；BPB 用累计 NLL 除以原文 Bytes 和 ln(2)。
- **Teacher Forcing**：训练使用 Dataset 中真实 Prefix，并行监督多个位置；Generation 逐步追加自己生成的 Token。
- **多位置监督**：一个 Batch 的各位置贡献汇总成 Loss，再一次 Backward/Update，不是每个位置分别 Update。

## 实验索引与结论边界

### 早期实验（历史回顾）

- Learning Rate 比较：0.01 较慢，0.1 收敛，0.28 在该二次问题上效果好，0.3 发散。这不是其他 Model 的推荐值。
- Character Medium：历史记录的最佳 Validation Loss 约 1.561，后续长训出现 Overfitting；不是当前小 BPE 配置的直接对照。
- Dense/MoE Pilot：见 `experiments/dense-vs-moe-pilot.md`。小模型上 MoE 不自动更快；Active Compute 只是近似匹配。
- Routing Bias Ablation：见 `experiments/moe-routing-bias-ablation.md`。单 Seed 观察到负载平衡改善，不代表统计显著的普遍结论。
- IsoFLOP Pilot：见 `experiments/isoflop-scaling-pilot.md`。5 个候选中约 55.8K Parameters 的配置最好；未证明全局最优或拟合 Scaling Exponents。

### 2026-09-11：真实文本 BPE

实现：`byte_bpe.py`；测试：`test_byte_bpe.py`；实验：`experiments/evaluate_bpe.py`。

- 原文先按 Character Index 做 90/10 Split，仅 Training Text 用于 Tokenizer Training。
- Vocabulary 512 = 256 Byte Tokens + 3 Special Tokens + 253 Merge Tokens。
- Training：1,003,854 Bytes → 512,353 Tokens，1.9593 Bytes/Token。
- Validation：111,540 Bytes → 57,641 Tokens，1.9351 Bytes/Token。
- 两个 Split 完整 Round-trip 通过；Save/Load 后映射一致。
- 结论：此英文 Dataset 的 Token 数约减半；不能由此推断生成质量必然提高。
- 本地输出：`runs/bpe-512-20260911-020342/`。

### 2026-09-11：BPE 小模型 Training

实现：`train_bpe.py`；配置：C=64，2 Layers，4 Query Heads，2 KV Heads，FFN=176，B=8，T=64，FP32，AdamW lr=3e-4，Seed=1337。

30/1,000/3,000 Steps 是同配置从头运行，不是从旧 Checkpoint Resume。Training Batch RNG 独立；Validation 使用相同 5 个固定 Batches。

| Step | Training Eval Loss | Validation Loss |
|---|---:|---:|
| 0 | 6.3838 | 6.4024 |
| 30（早期 Pilot） | 未做同口径 Training Eval | 6.2753 |
| 1,000 | 3.9920 | 4.0329 |
| 2,000 | 3.6268 | 3.7602 |
| 3,000 | 3.4726 | 3.6100 |

- 每 250 Steps 评估，保存 `best.pt`、`final.pt`、`history.json`、`sample.txt`。
- 当前 Best 是 Step 3,000，只在已评估 Checkpoints 中最优，不表示训练已达最优。
- Latest Run：`runs/bpe-model-20260911-035646/`；本档案已核对其中的 History。
- 检查：Shifted Batch、Forward/Backward、Final Logits 重载一致、Best Loss 重载一致、Generation。
- 结论：两条 Loss 持续下降，超过 Uniform；这段记录没有 Validation 回升。不能证明已学会长距离推理。
- 局限：单 Seed、5 个固定评估 Batches、没有独立 Test Set；尚未测 BPB。生成仍有假词、不能对话。
- 保存的是 Tokenizer Rules + Model Config/Weights；旧 Character Checkpoints 不被替换。

`runs/` 和 Checkpoints 被 Git 忽略：上传本档案不会上传 Weights/完整运行结果。上表保留关键摘要，重要本地产物另行备份。

## 下一次怎么接上

1. 先读“我们现在在哪”，不从 Transformer 基础重复开始。
2. 回顾一个问题：为什么当前 BPE 的 3.6100 不能直接和 Character Model 的 1.561 比？
3. 实现 BPB Evaluation 前先约定同一原文、预测范围、Context Policy，避免无意改变比较口径。
4. 若要做 Character/BPE 对照，先确定预算/架构控制；当前两个现成 Model 并非公平对照。
5. 暂不默认扩模型、重训 Tokenizer、长时间训练、删除 Checkpoint 或 Commit/Push；按当次请求执行。

## 维护约定

- 每完成一个模块、实验，或发现重要误解时更新；不必把每条聊天都抄进来。
- 新结果注明日期、配置、证据路径和局限；纠正错误时保留更正说明。
- 用户表示“懂了”记为交流进展；独立解释/实现需有独立证据。
- 代码由助手编写就明确记录，不作为学习者独立编码能力的证明。
- 每次结束只留下一个清晰的下一步；新兴趣可以列为支线，不自动切换主线。
- 支线兴趣：Agent Harness。曾讨论，尚未进入系统课程；与当前 BPE/GPT 主线区分。

### 更新记录

- 2026-09-11：建立第一版；整理历史学习路径，核对最新 BPE History，列明掌握证据和待复核项。
