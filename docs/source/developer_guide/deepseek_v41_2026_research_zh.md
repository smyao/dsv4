# DeepSeek V4.1 Flash：2026 年论文与 GitHub 推理优化评估

研究截止：**2026-10-24**。源码基线：`releases/v0.31.0rc`，`e099a085329b57cb62b8eee4d381c70c3db2566b`。适用对象：本地 vLLM Ascend 的 DSV4.1 文本推理，输入 65,536 / 131,072 tokens。前置证据为[源码与仿真报告](deepseek_v41_long_context_analysis_zh.md)和[六项技术特性](deepseek_v41_long_context_features_zh.md)。本文新增文献筛选、公开实现对照、数值契约反例及工程决策，不修改生产推理代码。

**结论：第一批开发应保持当前模型的索引集合、缓存值和验证规则，优化候选消费者 QK、C2 有效行处理、缓存布局及 DSpark 调度；第二批研究精确 Top-K 和容量受限时的 HiSparse。HISA、MISA、PIVOT、SPIN、额外跨层共享与 KV 量化不能以“精度不变”直接上线。** 本文给出每项成立条件及拒绝条件；没有将论文的 GPU 倍率写成本地 Ascend 已测吞吐。

## 1 证据范围与评估口径

### 1.1 检索覆盖

检索覆盖 2026 年 1 月至 10 月 24 日公开材料，并追溯必要的 2024–2025 年基础工作。来源包括 arXiv 正文/摘要、ICLR/ICML/ACL/MLSys/AAAI 官方论文页面、作者机构页面、作者 GitHub、DeepSeek 官方仓库及 vLLM/SGLang 上游实现。10 月新增的 SPIN、LatentIndex、Prism Attention、FlashMask-3 已纳入。11–12 月尚未公开的研究不在范围内。

第 7 节逐项评估 **44 项论文/研究**；其中 40 项为 2026 年首发或 2026 年会议发表，4 项为基础工作。第 8 节另列工程仓库。通过六个方向交叉检索：索引与选择、稀疏内核、缓存与传输、推测解码、调度与并行、MoE 通信。以直接命中 DSA/DSV4.1 的工作为深读重点，其余用作者摘要与实现范围作适用性筛选。**这是有范围的广泛检索，不是对全球全年所有论文的穷尽证明。** 检索词、读取失败及外部来源索引见[检索记录](../../../benchmarks/deepseek_v41_long_context/results/research_sources.json)。

没有把同名视频扩散 DSA、GUI-agent HiSA 当作 DeepSeek Sparse Attention 文献；没有将第三方宣传、“near-lossless”字样或 GitHub issue 标题当作数值等价证明。会议年份和首发年份分别记录，例如 TurboQuant 首发 2025、ICLR 2026；LongSpec 首发 2025、ACL 2026。

### 1.2 四类精度契约

| 类型 | 本报告含义 | 可接受证据 |
| --- | --- | --- |
| E0：数据不变 | KV 原始位模式、位置映射、索引集合不变，仅移动/排布/调度 | bytewise / indexwise 差分及完整生命周期验证 |
| E1：数学相同 | 相同运算及选择目标，但融合、分块或归约顺序改变 | 算子差分、Top-K 边界差分、完整模型评测；不能自动声称 bitwise 相同 |
| E2：验证保分布 | 草稿可近似，输出由正确的 target 验证与补偿采样决定 | 验证公式、停止决策因果性、rollback 状态及采样分布测试 |
| A：算法近似 | 改索引、缓存数值、窗口或模型结构 | 实际 DSV4.1 长输入、多轮、长输出评测；其他模型均分持平不够 |

“不掉精度”的基准必须固定：同一 checkpoint、同一量化权重、同一 tokenizer/模板、同一目标模型执行模式。W8A8 target 的等价优化，不代表恢复到原始 BF16 权重的精度。A2/A3 和 A5 的缓存、算子及数值链不同，分别验收。

本机没有 torch/torch_npu、Ascend 设备、实际权重或 serving trace。已执行的是原源码局部控制流及 CPU 语义/工作量实验；没有执行真实模型质量或 NPU 吞吐测试。这是明确的证据边界，不以“预计提升”填补。

## 2 本次核对后必须补充的事实

### 2.1 公开模型配置与上一轮实验拓扑一致

读取了 [DeepSeek 官方 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)：40 层；KV 来源 2/8/14/20；索引来源 2/8/14/20/24/28/32/36；32 个索引头、维度 128；Top-K=512；候选 2,048 块、每块 8；SWA=128。上述字段与本地实验 fixture 一致，32-head 工作量分支与该公开配置吻合。**这核实了公开原版 checkpoint 的结构，仍不能代替用户部署权重/config 的现场记录；1017 更新版也不与原版混写。**

### 2.2 bounded replay 是近似执行模式

本地 [prepare](../../../vllm_ascend/models/deepseek_v41/decoder_replay_state.py) 将可裁剪请求保留为 `min(query_len, window)`；[build_replay_swa_visible_lens](../../../vllm_ascend/attention/dsa_v41.py) 将可见起点截到 `replay_start`。DeepSeek 官方 [部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) 同样将重建结果定义为 approximate SWA KV cache。

不能以“单层窗口只有 128”证明多层保留最后 128 行就等价。可复现实验：两个 window=128 的均匀因果注意力层，256 个输入位置，仅位置 127 的值为 1；完整执行最后位置输出 **0.00775146484375**，仅从位置 128 开始重放输出 **0**。它证明“通用严格等价”命题不成立，不测量 DSV4.1 实际质量损失。

因此上一份 F02 应解释为“将既有近似模式扩展到 PCP”；它不属于精度优先路线的无损 P0。原报告中 46.7578% 层行数和约 60.5% 索引对数下降仍成立，**收益必须同时注明 replay 精度基线**。本地切点为层 20 后、共 19 个尾层；不直接套用外部文档的层数。

### 2.3 当前已经具备的优化不能重复计入

- 40 层只运行 8 个索引器，只有 4 个 KV 来源；IndexCache/CLSA 的“从每层索引开始省掉大部分索引”不是当前增量收益。
- 主注意力已为 SWA + sparse MLA；用稀疏方法相对 dense FlashAttention 的加速不能直接迁移。
- DSpark、动态验证调度、MC2/MegaMoE、EPLB 接口、chunked prefill、PCP、PD 传输均已有对应代码；需要核对实际分派及缺口。
- 本地 `rg HiSparse` 仅命中旧模型补丁中的 GPU 功能说明，V41 cache/model/runner 没有完整 HiSparse 接线。上游 [Ascend RFC #19293](https://github.com/vllm-project/vllm-ascend/issues/19293) 是设计讨论，不能据此声称当前支持。

## 3 优先级最高的源码落地方向

### 3.1 候选消费者在 QK 前裁剪：保留 F01 为 A2/A3 P0

**可借鉴对象：DeepInfer 的真实 sparse-MQA 实现；HISA/LSA 的粗筛后细算组织方式。**

DeepInfer 固定提交 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 的 [index_attn.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/index_attn.py) 明确构造 candidate compact lengths、缓存 sparse schedule，并调用 `fp8_fp4_sparse_mqa_logits` / paged sparse 路径，最后将局部 Top-K 还原为原位置。它提供了实际工程参照。

本地 [indexer.py](../../../vllm_ascend/models/deepseek_v41/indexer.py) 的 A2/A3 consumer 进入 mode=2；[Cube 主循环](../../../csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h) 仍遍历可见历史，candidate mask 到 Vector 阶段才施加。上一轮已编译执行这两个调度函数，确认候选不控制 QK tile 跳过。

**设计决策：复用现有层 20 产生的候选，不引入新近似粗筛。** 在层 24/28/32/36 的 Cube 加载前映射候选页，使用当前 INT8 QK、FP16 scale/中间舍入和 FP32 归约，再还原压缩坐标。不要直接用 DeepInfer 的 FP4/BF16 链替换本地数值链。

E0/E1 验收条件包括：候选不足时保留原有限负数填充行为、同分选择一致、最后可见块保留、C2 尾部因果一致、分页和 graph 稳定地址一致。不满足快速路径前提时走原内核。本地已经证明“只有 16 个候选位置时仍需填满 512 个有效结果”的边界，不能一律把候选外位置删除。

收益适用于 decode 和关闭 replay 的 prefill；开启 replay 后尾层已只算少量行，本项冷 prefill 优先级下降。**不能再乘一次 HISA 的 4× 或 DeepInfer 的任何系统倍率。**

### 3.2 精确 Top-K：取算法原则，不直接拆开本地融合算子

[HPC-Ops 论文](https://arxiv.org/html/2609.08450v1) 先用样本提出边界，再完整扫描认证，候选不足时恢复，最后精确 FP32 refinement。它仍必须读完整 score 行，减少的是选择器工作，不是生成 score 的 QK 工作。其 [接口源码](https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/hpc/topk.py) 支持 CUDA FP32、K=512/2048，允许边界同分时返回任意对应索引。

[DeepSelect](https://github.com/deepseek-ai/DeepSelect) 已有 Ascend 后端，重点优化 K=512；但当前 Ascend 接口只支持 BF16 输入。其“相对 torch.topk 2–20×”基线不是本地 QLI 的 Vector 内选择器。算法值得参考，不能把 BF16 cast 当作无损适配。

本次执行的两个反例：

1. FP32 `[1.001, 1.002]` 转 BF16 都变成 `1.0`，稳定 tie-break 选第一个而原 FP32 最大值在第二个位置。
2. 两个位置 indexer score 均为 1，但对应主注意力 V 分别为 0 和 10，K=1 时选择结果改变主注意力输出。**“Top-K 值精确”不推出“模型输出等价”。**

工程上先 profile Vector 选择器占比，再将阈值过滤/小集合精排嵌入原融合路径，保留 FP32 score 及位置 tie-break。若改成独立选择器，chunk=8192、N=128K 的一个 C1 来源层就要物化 **4 GiB FP32 score**；candidate 消费者仍需 **512 MiB**。这会引入当前融合路径没有的流量，不能仅凭 Top-K 微基准判定收益。

CPU 认证参考在 64K/128K 上覆盖随机、递增、全同分和采样漏掉高值四种分布，共 8 例，与完整排序的确定性索引一致；它只验证“认证+恢复”原则，不是 HPC-Ops 的 CUDA 复现。

### 3.3 DSpark：补全基于耗时的预算，不能重复开发已有草稿器

[DSpark 论文](https://arxiv.org/abs/2607.05147) 的核心还包括依据存活概率和引擎性能曲线分配验证预算。本地 [DynamicSpecScheduler](../../../vllm_ascend/spec_decode/utils.py) 已执行置信度、cumprod、全局 topk 分配；但 `compute_verify_budget` 用阈值 0.3 计数、默认每 16 步更新、将预算调整为奇数，没有按当前 DSV4.1 的批量/上下文耗时曲线求最优。

本次直接提取并执行原 `compute_verify_budget`，32 次调用记录 **2 次 `.item()`**。这是设备标量读取次数的源码事实；没有测量 NPU 阻塞时长。该路径与上一报告核对的 CPU metadata `.item()` 是不同对象。

DeepInfer 的 [mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp) 则在离散 graph size 上比较 `expected_output / predicted_time`，纳入 sampler 历史耗时和 forward 修正；还能用于确定本地优化的具体缺口。官方 Q&A 的固定 5 对动态长度数据包括低并发负收益，不能只摘取高并发正收益。

建议以 `(batch, context_bucket, verification_width, backend, graph_mode)` 建立实测表；允许 K=0 回退普通 decode；使用上一轮已完成事件中的统计更新预算，避免关键路径同步；设备侧决定本轮每请求长度。保留严格验证，`enable_entropy_verify=false`，禁止 synthetic acceptance。本地 [rejection_sampler.py](../../../vllm_ascend/sample/rejection_sampler.py) 中 entropy 模式会缩放接受判据，不能用于证明 target 分布不变；两个相关开关默认均为 false。

E2 还要求验证长度决策满足因果条件。外部实现专门记录相邻两轮 confidence，避免后续草稿的置信度影响当前 token 是否验证。简单反例：p=q=Bernoulli(0.5)，只在草稿为 A 时验证，否则重新从 target 采样，输出 A 概率变成 0.75。**本报告没有据此断言本地 DSpark 已有采样偏置；它证明只检查拒绝采样公式不足以完成动态调度的无损验收。** Greedy 必须逐 token 比对；随机采样验证分布，不能要求同 seed 路径一定生成同串。

### 3.4 HiSparse：只在缓存容量限制并发时进入 P1

[HiSparse](https://arxiv.org/html/2608.07009v1) 把完整主 KV 放 host、设备保留热集，按既定索引精确拉取。它改变位置，不改模型选取的 token，属于 E0 路线。作者实现已进入 SGLang；上游 Ascend RFC 明确区分 Source/Resident/Hot，index K 留在设备。

对本地 A2/A3 的 128K 单请求，四个来源的主 KV payload 为 **320 MiB**，index K+scale 为 **40.625 MiB**。indexer 仍扫描完整 index K，不能一并无条件卸载。若每个来源只需一个 512 行热集，主 KV 是 2 MiB；但层 20 的同一 KV 来源服务多个独立索引器，热集须按这些选择的并集计算，8 个索引组无重叠时主 KV 热集为 4 MiB。DSpark 多 query 还会扩大并集。

这些是 payload 算术，**不是当前 allocator 实际可以释放的显存**。长 KV 与 SWA/ring 同池的 128 KiB 下限会阻碍小热集兑现；先实现 F03 或新的独立 pool 契约。保留 SWA、压缩状态、页引用计数、异步 D2H 完成、prefix/PD、rollback 和 CoW。共享索引的后续层可以准确预取；新的独立索引层必须等待其真实选择。

吞吐成立条件：释放显存允许批量从 B 增加到 B′，且 `B′ / T_step(B′, IO) > B / T_step(B, resident)`。已有足够显存且并发不变时，额外 IO 不构成必然加速。先测命中率和传输重叠，不能引用论文最高 4.7× 作为 DSV4.1 收益。

### 3.5 无损 PD 压缩与双路径加载：场景不同，分别验收

[SplitZip](https://arxiv.org/abs/2605.01708v3) 做位级无损浮点指数编码；可在 A2/A3 的 BF16 主 KV 传输段评估。INT8 index、A5 FP4 pack 不满足同一 BF16 冗余假设。先去除无需传输的 padding，再测压缩；不可重复计算两次节省。

其 BF16 编码每元素保留 8 bit 符号/尾数和 4 bit 指数码，外加逃逸记录；因此该编码的理想压缩率上限是 **4/3**，最多节省 25% payload。它不是数量级的容量优化，不能把其他有损量化的压缩倍率接到这一无损路线。

设原始字节 S、压缩率 r、链路有效带宽 b。串行路径收益成立当且仅当 `T_encode + T_decode < S/b × (1−1/r)`，还须包括启动/元数据开销。实际流水按关键路径重算。高速同机互联上编解码成本可以超过传输节省；压缩率不足时直接传原数据。

[DualPath](https://arxiv.org/html/2602.21548v1) 针对远端缓存恢复的存储 NIC 瓶颈，使用 decode 侧空闲存储链路及计算网络搬运。它不会降低首次冷 128K 的索引计算量。只有远端命中/agent 多轮负载、现有 storage→prefill 链路饱和且有另一条空闲路径时才验证。实现接在 PD/缓存 connector 与路由层，不在 `DeepseekV41Attention.forward` 中塞网络逻辑。

### 3.6 调度、PCP 和 MoE：先量化代价，不替换模型语义

[P-PAS](https://arxiv.org/abs/2608.15171)、[SLOWeave](https://arxiv.org/abs/2609.07883)、[Vertumnus](https://arxiv.org/abs/2609.04774) 的可用思想是按上下文、缓存命中、队列和 deadline 选择 chunk/CP。对于本地全扫描索引，单来源新 chunk 的精确逻辑成本是 `Σ floor(t/r)`，约为 `(c·q + q²/2)/r`；只按本轮 q 预测会漏掉历史 c。

从 PCP=1 增至 PCP=4，单请求 TTFT 改善不等于整组吞吐增加：若占卡增加 4 倍且耗时只缩短 3 倍，tokens/s/卡反而下降。必须在固定总卡数比较。当前 PCP 还关闭 bounded replay；不同模式的质量和工作量基线不能混合。新调度不要照抄过时 `schedule()`：本地 [platform.py](../../../vllm_ascend/platform.py) 已因四种 scheduler fork 不支持 rewind 而禁用 replay。

[VPP](https://arxiv.org/abs/2608.26523) 是直接基于 vLLM Ascend、16 张 910C 的 2026 研究，但其已报告收益主要对应长序列 pipeline 场景，不能当成本分支 V41 已支持。当前 PP>1 会禁用 decoder replay，Engram 也有 PP 限制；默认路线不选 VPP，先验证无 Engram 的 PP 适配与同精度基线。

[Weave](https://arxiv.org/abs/2609.21483)、[Analytical Resource Management](https://arxiv.org/abs/2609.07536) 和 [MonoMoE](https://arxiv.org/abs/2609.04244) 值得借鉴的部分是按实际 expert tile 负载调节通信与计算重叠，以及小批量持久化融合。CUDA SM/CTA/NVLink 的资源模型不能直接套到 Ascend Cube/Vector/HCCL。当前 [moe_comm_method.py](../../../vllm_ascend/ops/fused_moe/moe_comm_method.py) 已有 CANN MegaMoE 路径；新增收益必须对比该路径而非未融合 GEMM。该文件明确警告 P/混合节点接收容量不足会损害精度，不能靠截断 expert token 获得性能。严格保留 router、专家 ID、路由权重及完整 token 数。

## 4 工作量如何转换为可审计的吞吐判断

### 4.1 已有候选集紧凑化的工作量

下面来自上一轮原源码调度实验。候选后工作量按每 query 最多 16,384 个候选位置计算；真实部分尾块更少。均为一次完整请求、单索引副本；不是设备耗时。

| 输入/模式 | 原逻辑 QK 对 | 候选容量模型 QK 对 | 减少比例 |
| --- | ---: | ---: | ---: |
| 64K 冷 prefill，replay 关 | 13,958,807,552 | 9,126,871,040 | 34.6157% |
| 128K 冷 prefill，replay 关 | 55,834,902,528 | 29,527,998,464 | 47.1155% |
| 64K 冷 prefill，replay 开，chunk=8192 | 5,519,476,736 | 5,431,591,424 | 1.5923% |
| 128K 冷 prefill，replay 开，chunk=8192 | 22,044,807,168 | 21,604,860,416 | 1.9957% |
| 64K 单 query decode | 425,984 | 229,376 | 46.1538% |
| 128K 单 query decode | 851,968 | 393,216 | 53.8462% |

### 4.2 Amdahl 门槛及负收益示例

令 f 为原总耗时中被优化 QK 的占比，r 为该部分时间减少比例，h 为新增开销/原总耗时；固定批量且其他成本不变时，`speedup = 1/(1−f·r+h)`，正收益条件为 **h < f·r**。把 pair 减少比例当作 r，额外假设了 QK 时间与 pair 数线性；真实稀疏 gather、tile 填充和重叠会改变它。

| 条件场景：假设 f=50% | h=0 的模型倍率 | h=1% 的模型倍率 |
| --- | ---: | ---: |
| 64K prefill，replay 关 | 1.2093 | 1.1949 |
| 128K prefill，replay 关 | 1.3082 | 1.2913 |
| 64K prefill，replay 开 | 1.0080 | **0.9980** |
| 128K prefill，replay 开 | 1.0101 | **0.99998** |

这张表是敏感性仿真，**f=50%、h=1% 均为人为假设**。它证明 replay 场景增加少量 gather/启动开销即可抵消 pair 节省。不得据此承诺 30.82% 本地吞吐提升。

### 4.3 动态推测预算的目标函数

每请求条件接受率为 aᵢ，存活率 sᵢ=∏aⱼ；验证 K 个草稿并产生一个 bonus/correction token 时，期望输出为 `E(K)=1+Σsᵢ`。固定批量的优化目标是 `E(K)/(T_draft+T_verify+T_sample+T_schedule)`，不是最大化 E(K)。跨批量还要计入容量和调度时间。

CPU 构造 aᵢ=0.9，K=0..5 的相对 step 成本为 `[1,1.1,1.2,1.3,2.3,3.3]`。原阈值预算选择 5；按目标函数选择 **3**。这是可执行反例，证明阈值策略不保证耗时最优；成本数组不来自 NPU，不构成实测收益。

### 4.4 多方案组合规则

- F01 缩短 indexer 后，Top-K/通信占比会重新变化；收益按新 trace 重算，不能把独立倍率相乘。
- F03 降低 pool 填充与 HiSparse offload 都影响容量；按最终实际分配字节计算一次。
- DSpark 增大 query 数，改变 indexer、热集并集、MoE 批量和 PCP 负载；用接受后输出 tokens/s 计量，草稿和被拒绝 token 不计产出。
- PD 提高 serving goodput 可以不改变孤立请求 kernel latency；分别报告 TTFT、TPOT、requests/s、tokens/s/卡及 SLO 达标率。

## 5 不进入无损默认路线的论文方案

HISA、MISA、PIVOT、SPIN 针对剩余全扫描来源层有实际计算量空间，但筛掉 token/头、预测候选或共享 query 会改变当前索引函数。HISA 的“保持 token-level Top-K 模式”指输出形式，论文 IoU>99% 也不是索引 100% 一致；MISA 的 92% 以上召回同样不足以证明等价。

对这些方案，额外重排候选只保证“候选内部精确”，不保证被粗筛删除的位置不在真实 Top-K。若想进入 E0/E1 路线，必须给每个被省略块建立对**当前完整分数链**的可证明上界，只有上界低于当前第 K 大值时才跳过，并处理 FP16 舍入、scale、负权重、tie 和溢出。拿不到有效上界就回退完整扫描。此改造是本项目新研究方向，不是上述论文已经替本地证明的性质。

IndexCache、CLSA、LatentIndex 和 Prism 对当前 8 个来源再减少数量，也属于改变模型。LSA/MSA/NSA 的原生训练收益不能通过替换一次 forward 获得。YOCO 原始架构的提前退出结论不能直接套用到保留每层 SWA 的 DSV4.1。

TurboQuant、UltraQuant、AATC、JoLT、KeepKV 等压缩方案改变目标 KV。A2/A3 的共享 512 维 latent 并非普通多头独立 K/V；误差还会进入 index K 投影和离散 Top-K。A5 已有 FP4/MXFP4 pack，不能把相对 BF16 的 4× 压缩再乘一遍。首选只用于草稿，target 保持原缓存并严格验证；改 target 需另立质量实验，默认关闭。

VeriCache/FAFO 将近似生成放入精确验证框架，方向与 E2 一致，但本地已有专用 DSpark，target 本身也已稀疏、共享 KV。另建压缩 target 草稿必须覆盖额外验证、缓存 swap、attention mask、压缩状态回滚成本。没有“比已有 DSpark 更省”的证据，排为 P2。

## 6 从论文转成工程技术特性

P0/P1/P2 是开发顺序；只有完成验收才能宣称支持或性能收益。

| 特性 | 实施内容及本地接口 | 精度契约 | 进入条件 / 停止条件 |
| --- | --- | --- | --- |
| R01 / F01，P0 | A2/A3 consumer 稀疏 QK；native tiling/Cube、indexer dispatch、候选位置恢复 | E0/E1；保持分数链、tie、filler | 原 kernel 差分通过，decode/关闭 replay prefill 的端到端收益 95% CI 下界>1；否则不启用 |
| R02，P1 | 精确选择过滤嵌入 Vector，FP32 refinement 和确定性边界处理；参考 DeepSelect/HPC-Ops | E0/E1 | trace 证明选择器有成本；不新增完整 score 物化；同分/NaN/短行契约全部覆盖 |
| R03，P0 | DSpark 预算曲线、异步反馈、K=0；`spec_decode/utils.py`、proposer、runner scheduler | E2 | 对固定 5 和普通 decode 均比较；置信度分桶校准；随机采样因果审查通过；高并发负收益回退 |
| R04 / F03，P1 | 拆 long KV/index 与 SWA/ring pool，之后接 HiSparse Source/Resident/Hot | E0 | 实际 HBM 释放与并发提升；index K 常驻；跨层并集/rollback/PD/CoW 无错；IO 不抵消吞吐 |
| R05，P1 | deadline/cost 驱动 chunk，混合 PCP 度的实例路由；先离线配置选择，再动态调度 | E0/E1 | 固定总卡数、精度模式相同；使用真实 context 成本；保留 rewind/replay 契约 |
| R06，P1 | MoE tile 负载驱动的 MC2/MegaMoE 重叠和容量校验 | E1 | trace 表明通信/负载不均为瓶颈；相同 router 与量化；不丢 token；不是仅换未融合基线 |
| R07，P2 | BF16 KV 无损 PD 编码、DualPath 远端缓存恢复 | E0 | 分别满足 codec 不等式和远端 IO 饱和条件；不与已去除 padding 重复记收益 |
| R08 / F04、F05，P1/P2 | 完成 C2 后只处理有效行；候选直接写稳定 buffer | E0/E1 | 保留 C2 两个输入的 FP32 WKV/Wgate；多请求奇偶边界、graph 地址、output stride 均正确 |
| R09，研究分支 | HISA/MISA/PIVOT/SPIN/额外共享/target KV 量化 | A | 首先测 DSV4.1 各任务退化；没有真实模型结果不得提升到无损默认路线 |
| F02 修订，质量模式专项 | PCP decoder bounded replay | A，相对完整 prefill；相对既有 replay 可做等价移植 | 先分别建立完整与 replay 质量基线；不能因“官方使用”免除测试 |

R03 不需要再增加一组环境变量，优先扩展已有 `dynamic_spec_config`；R01/R02 用硬件能力及明确算子契约分派。每项实现遵循仓库 patch/model runner 审查要求。本文是设计，不创建未经测试的启动开关。

## 7 论文逐项评估清单

“摘要筛选”表示已读取作者摘要/官方页面，结论限于方法与适用范围；“正文核对”表示读取了相应方法/实验段落，仍不代表复现论文。以下论文倍率均属于其作者报告的对象，不能作为本地测试结果。

### 7.1 索引、注意力与架构：14 项

| 编号 / 时间 | 一手来源及阅读深度 | 对当前 DSV4.1 的结论 |
| --- | --- | --- |
| P01 / 2026-03 | [IndexCache](https://arxiv.org/html/2603.12201v1)，正文；作者 GitHub | 当前已有 8 源/40 层共享。原文 30B DSA 上去掉 75% 索引的收益不重复计入；继续共享为 A。 |
| P02 / 2026-03 | [HISA](https://arxiv.org/html/2603.28458v1)，正文；TransArch/TileLang | 原实验 block=128、64 候选块、Top-K=2048，与本地 8/2048/512 不同。借组织方式实现 R01；新粗筛为 A。 |
| P03 / 2026-05 | [MISA](https://arxiv.org/html/2605.07363v1)，正文 | 选择部分索引头再 rerank；原 DeepSeek-V3.2/GLM-5 的头数不同，本地仅 32 头。不能照搬“8× 更少头”；A。 |
| P04 / 2026-07 | [PIVOT](https://arxiv.org/html/2607.24593v1)，正文 | query 分组代理扫描/候选内重排可减少剩余 source 工作；不能跨 request、C2 边界或未来 query 泄漏；A。 |
| P05 / 2026-08 | [LongCat Sparse Attention](https://arxiv.org/html/2608.01662v2)，正文 | streaming/shared/hierarchical 三项共同训练设计；复用布局思想，不改本地 128 SWA、512 sparse 或已训练共享拓扑；完整替换需训练。 |
| P06 / 2026-06 | [You Only Index Once / CLSA](https://arxiv.org/abs/2606.06467)，摘要筛选 | 原生 KV/routing 共享架构，128K 的总体倍率相对其基线。本地已共享，不能解释为可直接再提速。 |
| P07 / 2026-10 | [LatentIndex](https://arxiv.org/abs/2610.04635)，摘要筛选 | 改共享 latent 与层特定选择，需要校准/训练；本地 index K 已从共享 compressed KV 产生；非直接无损替换。 |
| P08 / 2026-10 | [SPIN](https://arxiv.org/abs/2610.09025v2)，摘要筛选 | 历史预测跳过块，作者 vLLM 最高输出吞吐 +14.9%；目标模式与本地不同，加入 A 类 decode 实验，不作为保证。 |
| P09 / 2026-10 | [Prism Attention](https://arxiv.org/abs/2610.22985)，摘要筛选 | 跨层 proposal/refinement；3.4× 对比 FlashAttention，不是对本地 sparse MLA；A。 |
| P10 / 2026-09 | [Sample-Guided Exact Top-K](https://arxiv.org/html/2609.08450v1)，正文+源码 | 唯一直接围绕精确选择认证的重点项；HPC-Ops 容许任意边界同分索引，本项目须收紧契约；R02。 |
| P11 / 2026-03 | [FlashAttention-4](https://arxiv.org/abs/2603.05451)，摘要+作者实现范围 | B200/CuTeDSL 异步流水；移植思想需要 Ascend 重写，不能安装 CUDA wheel 获得收益；E1。 |
| P12 / 2026-10 | [FlashMask-3](https://arxiv.org/abs/2610.17901)，摘要+作者 GitHub | 主要为长上下文训练/结构 mask 的 CP；借 mask 本地化和负载均衡，不能认为 ragged 512-token 稀疏索引可直接套其区间表示。 |
| P13 / 2026-06 | [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)，摘要+作者 GitHub | GQA group-specific block 选择，非共享 MLA 的 drop-in；原模型训练及 H800 内核倍率不迁移。 |
| P14 / 2026-01 | [S³-Attention](https://arxiv.org/abs/2601.17702)，摘要筛选 | SAE/倒排检索并丢完整 KV，改变模型输入与检索；作者说明原型 wall-clock 更慢；不进入本地吞吐路线。 |

### 7.2 草稿与验证：6 项

| 编号 / 时间 | 一手来源及阅读深度 | 对当前实现的结论 |
| --- | --- | --- |
| P15 / 2026-07 | [DSpark](https://arxiv.org/abs/2607.05147)，正文+DeepInfer/DeepSpec 源码 | 当前已有专用 proposer；新增工作为成本驱动预算、同步移除、验证因果性，优先 R03。 |
| P16 / 2026-02，ICML 2026 | [DFlash](https://proceedings.mlr.press/v306/chen26bv.html)，官方论文摘要+作者 GitHub | 本地已有 proposer；新 DSV4.1 草稿还需对应训练，不将 Qwen 结果当成可用 checkpoint；DSpark 优先。 |
| P17 / 2025 首发，ACL 2026 | [LongSpec](https://aclanthology.org/2026.acl-long.83/)，官方论文摘要+作者 GitHub | 有价值的是长上下文草稿常数 KV 与位置泛化；现成权重不对应 V41，树验证不是免费替换。 |
| P18 / MLSys 2026 | [Speculative Decoding: Performance or Illusion?](https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html)，官方摘要 | 实际 vLLM、多模型/批量研究；支持必须评估 target verification 和接受长度分布，不能只报 batch=1。 |
| P19 / 2026-05 | [VeriCache](https://arxiv.org/abs/2605.17613)，摘要筛选 | 压缩缓存草稿+完整缓存验证，E2 方向；额外 host swap 对共享 KV+DSpark 的净收益未建立，P2。 |
| P20 / ICML 2026 | [FAFO](https://proceedings.mlr.press/v306/le26g.html)，官方摘要/方法附件 | 单模型同次 forward 混合近似草稿与精确验证；本地 sparse/SWA mask 需新适配，不能与 DSpark 直接叠乘，P2。 |

### 7.3 缓存、量化与传输：8 项

| 编号 / 时间 | 一手来源及阅读深度 | 对当前实现的结论 |
| --- | --- | --- |
| P21 / 2026-08 | [HiSparse](https://arxiv.org/html/2608.07009v1)，正文+SGLang 源码/RFC | E0 放置优化；保留 index K、按共享 KV 的跨索引并集预取；R04，先解决 pool。 |
| P22 / 2026-02 | [DualPath](https://arxiv.org/html/2602.21548v1)，正文 | 多轮远端缓存恢复的双路 IO；与 cold prefill 计算优化是不同场景；R07。 |
| P23 / 2026-05/06 | [SplitZip](https://arxiv.org/html/2605.01708v3)，正文 | 位级无损传输编码；BF16 主 KV 可测，INT8/FP4 不能套同压缩率；R07。 |
| P24 / 2025 首发，ICLR 2026 | [TurboQuant](https://arxiv.org/abs/2504.19874)，论文/作者说明及 vLLM 实现评估 | 有损低位量化，有失真界不等于逐输出不变；本地 cache ABI/latent 需重写，A。 |
| P25 / 2026-06/09 | [UltraQuant](https://arxiv.org/abs/2606.20474v3)，摘要筛选 | AMD 原生 FP4 与多轮质量/吞吐联合评估很有参考价值；硬件 MFMA 路径和目标模型不同，A。 |
| P26 / 2026-08 | [Attention-Aware Transform Coding](https://arxiv.org/abs/2608.14191)，摘要筛选 | 根据 attention 失真分配位数；Llama/Qwen 的近无损结果不能证明共享 latent 无损，A。 |
| P27 / 2026-07 | [JoLT](https://arxiv.org/abs/2607.12550)，摘要筛选 | token/feature Tucker 分解+残差，重构非零误差且有在线成本；低于 pool/选择器优先级，A。 |
| P28 / AAAI 2026 | [KeepKV](https://ojs.aaai.org/index.php/AAAI/article/view/40611)，官方摘要 | “单步 lossless / 多步 error bound”不是完整生成逐步等价；需改注意力补偿与缓存，A。 |

### 7.4 调度、并行与 MoE：9 项

| 编号 / 时间 | 一手来源及阅读深度 | 对当前实现的结论 |
| --- | --- | --- |
| P29 / 2026-01 | [LAPS](https://arxiv.org/abs/2601.11589)，摘要筛选 | 长短 prefill 分队列，适用混合负载；单个冷 128K 无直接算法减算，R05。 |
| P30 / 2026-02 | [FlowPrefill](https://arxiv.org/abs/2602.16603)，摘要筛选 | 算子边界抢占；V41 状态 scatter、ring 更新和 replay 有依赖，不能任意中断恢复；P2。 |
| P31 / 2026-08 | [P-PAS](https://arxiv.org/abs/2608.15171)，摘要+作者 GitHub | vLLM 动态 MBT 控制思路可接，需增加 context/replay/EP 成本特征；R05。 |
| P32 / 2026-09 | [SLOWeave](https://arxiv.org/abs/2609.07883)，摘要筛选 | 选择满足 decode deadline 的最大 chunk；作者证据含事件仿真与迭代 GPU runtime，不等于完整 V41 serving；R05。 |
| P33 / 2026-09 | [Vertumnus](https://arxiv.org/html/2609.04774v1)，正文 | CP 度路由、实例重组和缓存管理；先做静态多池路由，动态重组需处理 replicated cache/HCCL/graph；R05。 |
| P34 / 2026-08 | [VPP](https://arxiv.org/html/2608.26523v1)，正文 | 16×910C 的 Ascend 实证；本地 V41 PP/Engram/replay 约束未解除，不作为立即启用项。 |
| P35 / 2026-09 | [Weave](https://arxiv.org/abs/2609.21483v3)，摘要筛选 | H100 持久 megakernel 的动态资源划分；移植负载模型，比较已有 CANN MegaMoE，R06。 |
| P36 / 2026-09 | [Analytical Resource Management](https://arxiv.org/abs/2609.07536)，摘要筛选 | COMET/A100 的 wave/CTA 模型；重建 Ascend 资源约束后评估，R06。 |
| P37 / 2026-08 首发 | [MonoMoE](https://arxiv.org/abs/2609.04244)，摘要+作者代码位置 | H200 小批量 weight-major 融合，适合借鉴布局；本地 W8A8/MegaMoE 不同，不能引用原端到端 18.7% 为收益。 |

### 7.5 基础工作：4 项

| 编号 / 年份 | 一手来源 | 当前用途 |
| --- | --- | --- |
| P38 / 2024 | [YOCO](https://arxiv.org/abs/2405.05254) | 理解共享 KV 与 prefill 退出的结构前提；DSV4.1 每层 SWA 不满足原版直接退出证明。 |
| P39 / 2025 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | 理解压缩/选择/局部注意力和硬件协同；重训架构，不直接换当前模型。 |
| P40 / 2024 | [Quest](https://arxiv.org/abs/2406.10774) | 借块统计做筛选；要无损需为当前分数链证明上界，固定预算启发式仍为近似。 |
| P41 / 2024，FAST 2025 | [Mooncake](https://arxiv.org/abs/2407.00079) | PD 与分层缓存基础；已有 Transfer Engine 不等于已有 DualPath 全局调度。 |

### 7.6 引用交叉检索补充：3 项

| 编号 / 时间 | 一手来源及阅读深度 | 对当前实现的结论 |
| --- | --- | --- |
| P42 / 2026-06 | [SparDA](https://arxiv.org/abs/2606.04511)，摘要+作者 NVlabs 仓库 | 训练 Forecast 投影预测下一层选择，两个 8B 模型的结果不是 V41 证据；改变选择为 A。若仅提前搬运预测页，仍等待原索引并补齐 miss，则可保持 E0，但不省原 QK，需测预取浪费。 |
| P43 / 2026-06，07 修订 | [FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079)，摘要+作者技术报告算法段 | 作者实现每 64 decode 步用训练的检索器固定 resident set，后续 native Top-K 只能从该集合选；这是 A，与 HiSparse 按真实选择精确拉取不同。原 V4 为 43 层/21 个 C4 来源，不能直接迁入 V41 的 40 层/4 KV 源。 |
| P44 / 2026-08 首发，10 修订 | [Self-Indexing Attention](https://arxiv.org/abs/2609.13205v2)，摘要筛选 | 变换域 1-bit 符号索引和分组检索改变选取依据；作者报告 V4-Flash 兼容性，不等于 V4.1 数值等价。作为 A 类与量化联合评估，算子倍率不等于端到端吞吐。 |

FlashMemory 的 Lookahead Sparse Attention 与 P05 的 LongCat Sparse Attention 都简称 LSA，本文分别处理。对于保持精度的路线，预测只用于预取；不允许预测器永久屏蔽原索引会选中的位置。

另外筛查后未进入优先清单：FCP（训练 CP，非 serving 实证）、FLYING SERVING（运行时 DP/TP 切换，V41 Engram/共享缓存/graph 迁移复杂）、ExactMoE（“Exact”指专家不删，不指 W4 数值等价）、OmniKVQuant（本轮纯文本范围外）、S³-Attention 同名旧论文、视频扩散 DSA。相关原始链接保存在检索索引。排除原因不是论文无价值，而是不能支撑本任务的直接收益结论。

## 8 GitHub 实现核对与移植成本

| 官方实现 | 实际核对内容 | 本地集成判断 |
| --- | --- | --- |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | README、部署 Q&A、固定提交 index_attn.py、mtp_varlen.hpp；支持 SM100 与 Ascend 950 | 候选 compact、跨层 schedule 复用、耗时预算值得直接对照；公开 release 不含完整 KV 系统/跨 DP 负载均衡，不是整套无缝替换 |
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect) | interface.py、算法说明、Ascend BF16 限制、output_idx 与对齐要求 | 借内核结构；A2/A3 FP32 分数不能直接 BF16 cast；输出 stride 和 index 顺序须适配 |
| [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops) | 固定提交 `2a2e2656…` 的 topk.py、coarse boundary C++ | CUDA FP32 精确认证，直接运行不支持 Ascend；保留有用算法并嵌入本地融合流水 |
| [SGLang HiSparse](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/hisparse_coordinator.py) | coordinator、compressed-position 备份、异步完成、选页拉取及作者正文 | 需要 scheduler/allocator/worker 三层适配；压缩坐标、草稿 rollback 和共享源不能只靠 adapter 改名 |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | 训练/评估流程、checkpoint 表、对应 target 的限制 | 可用于未来域内 DSpark confidence 校准；Qwen/Gemma 草稿不能直接配 V41 |
| [z-lab/DFlash](https://github.com/z-lab/dflash)、[LongSpec](https://github.com/sail-sg/LongSpec) | 作者 README、目标权重与训练说明 | 作为替代草稿实验来源；当前优先已有 DSpark |
| [THUDM/IndexCache](https://github.com/THUDM/IndexCache)、[TransArch](https://github.com/MuLabPKU/TransArch) | 作者模型/patch/算法范围 | V3.2/GLM 插件不能覆盖 V41 layer plan；只在研究分支做额外共享/粗筛 |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 官方内核范围及 DeepInfer 调用关系 | 用作算法和性能对照；不得把 CUDA 库替换 torch_npu 路径视为完成移植 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[FlashMask](https://github.com/PaddlePaddle/FlashMask)、[MiniMax MSA](https://github.com/MiniMax-AI/MSA) | 论文及官方仓库对应关系 | FA4/FlashMask 硬件流水与 MSA 的 GQA 拓扑分别处理，不混成通用 sparse 加速包 |
| [P-PAS](https://github.com/TimoSaemann/ppas-vllm)、[Mooncake](https://github.com/kvcache-ai/Mooncake) | 作者发布的控制器/传输组件及接口范围 | 调度和 IO 实验入口；迁移时保留本地 replay/PD 状态契约 |
| [SparDA](https://github.com/NVlabs/SparDA)、[FlashMemory](https://github.com/libertywing/FlashMemory-Deepseek-V4/blob/main/TECHNICAL_REPORT_en.md) | 作者仓库对应关系；FlashMemory 的 resident set / 两级检索算法段 | 区分仅预取与修改选择；不将“关掉功能可回退”当成打开后无损证明 |
| [VPP 作者分支](https://github.com/RookieCoder-Camera/vllm-ascend/tree/vpp-dev) | 论文提供的 Ascend fork 及分支存在性；未执行其内核 | 供后续差分阅读，当前报告不声称完成 V41 移植或复现 |

固定提交/文件 SHA-256 见检索证据 JSON。只有实际下载成功的文件记录 hash；`main` 链接是检索日快照，不能假定后续内容不变。外部代码仅下载阅读，没有安装或执行。

## 9 本地验证成果与下一阶段验收规范

### 9.1 本次已执行

新增 [research_checks.py](../../../benchmarks/deepseek_v41_long_context/research_checks.py)，结果为 [research_checks.json](../../../benchmarks/deepseek_v41_long_context/results/research_checks.json)：

- 原源码预算函数执行 32 次，记录 2 次标量读取；构造了阈值预算与耗时最优不一致的反例。
- 8 个 64K/128K 精确选择认证案例通过；含采样不足回退和大量同分。
- BF16 转换改变 Top-K 位置的反例、同分索引影响下游输出的反例通过。
- 两层 SWA 证明固定 128 行 replay 不是通用数值等价执行。
- 从上一轮结果读取 4 个冷 prefill 基线，生成 24 个明确标注参数假设的 Amdahl 场景。
- 计算 HiSparse payload/union、score 物化内存和 codec 收益条件；不假设实测带宽。

运行命令：

```bash
/Users/loki.yao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  benchmarks/deepseek_v41_long_context/research_checks.py
```

这些实验只支持上述语义/控制流/成本关系，不宣称任何论文已在 NPU 复现。

[最终校验记录](../../../benchmarks/deepseek_v41_long_context/results/research_validation.json)包含源码及外部下载 SHA-256、结果一致性和格式检查。已核对上一轮 19 个源码哈希及本轮 7 个源码哈希；Ruff 与四份 Markdown 的 markdownlint 均通过。仓库全量 `format.sh ci` 上一轮因未安装 pre-commit 未能执行；不将局部检查表述为全量 CI 通过。

### 9.2 精度与吞吐必须联合验收

固定原版/1017 checkpoint hash、vLLM revision、Ascend revision、CANN、NPU 型号、权重量化、KV dtype、模板与 reasoning 设置。严格模式关闭 bounded replay、entropy/synthetic acceptance；部署模式另测现有 replay，两个结果表分开。

测试矩阵：输入 65,536 / 131,072；输出 128 / 2,048 / 8,192；batch 1/8/32/64 或设备实际容量上限；cold / 50% / 90% 对齐 prefix hit；连续多轮 append/rollback；PCP 1/2/4；A2/A3 与 A5 独立。最大长度设为输入+输出以上，不能用 `max_model_len=131072` 测 131072 输入再生成 128。

E0/E1 首先做 source、consumer 的 score/Top-K/sparse attention 差分，覆盖相等分数、近边界、短行、非连续页、C2 奇偶、chunk 边界、prefix restore、DSpark 拒绝回滚。E0 索引和缓存位模式要求完全一致；E1 的数值容差必须在知道 dtype、算子链和基线波动后预先固定，不能测完再放宽。

模型级使用 RULER、LongBench v2/HELMET 的适配任务、长仓库代码定位与生成、多跳检索、跨块引用、多轮工具调用和长输出退化检查。对近似方案报告逐任务差值与 paired bootstrap 95% CI，不能只报平均分。若要求零可见退化，任何任务出现确定下降即不进入默认路线；“统计不显著”不等于已证明相同。未经用户确定损失预算，不自行接受 1% 或 0.5 个百分点的退化。

性能测量先 warmup 到稳定，固定请求集、输出预算和总卡数，交错跑 baseline/candidate 多轮；报告 p50/p95/p99 TTFT/TPOT、达标 goodput、总接受输出 tokens/s/卡、HBM 峰值、H2D/D2H/RDMA 字节、索引 QK/Top-K 时间、MoE dispatch/GEMM/combine 时间、host 同步、DSpark 存活率和每 step 验证宽度。成功门槛为同精度合同下端到端倍率 CI 下界>1，且既定 SLO 不退化；未过则保留原分派。

## 10 最终开发顺序

1. **先补 F06 的精度模式/计时证据，再做 R01/F04/F05。** 这些直接作用于已发现的重复工作，保持原选择语义。
2. **评估 R03。** 优化已有 DSpark，按 64K/128K 和并发选预算，避免新增同步；严格验证为硬约束。
3. **由 trace 选择 R02 或 R06；由容量证明选择 F03+R04。** 不因为论文倍率大就抢先替换融合内核或卸载 KV。
4. **R05/R07 只在真实混合/多轮/PD 工作负载上验收。** 单请求冷 prefill 的减算结果不能替代 serving goodput。
5. **F02、HISA/MISA/PIVOT/SPIN、额外共享及 target KV 量化留在明确的近似模式。** 实际模型精度数据到位前，不写“无损”“精度不掉”或确定的本地吞吐百分比。
