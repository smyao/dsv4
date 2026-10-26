# DeepSeek V4.1 Flash 长输入优化技术特性报告

设计基线为本地 `releases/v0.31.0rc`，提交 `e099a085329b57cb62b8eee4d381c70c3db2566b`。本文把[源码与仿真分析](/Users/loki.yao/vllm-ascend/docs/source/developer_guide/deepseek_v41_long_context_analysis_zh.md)中已经确认的问题转化为六项工程特性。所有收益数字均明确限定为逻辑计算量、处理行数或布局字节数；当前没有任何一项 NPU 时延收益被测量。

**本次已完成的是可运行的 CPU 实验、证据和设计。以下 F01–F06 不代表生产代码已实现。** 已有 bounded replay、共享 KV、共享 TopK、metadata 复用是设计基线，不作为新特性重复申报。

**2026-10-24 精度契约补充：** [论文与公开实现复核](deepseek_v41_2026_research_zh.md)确认 bounded replay 对完整 prefill 是近似 SWA 重建。F02 的工作量数字不变，但不能按“精度不变”列入默认 P0；它属于需单独验证真实模型质量的专项。F01/F03/F04/F05 则以保持原模型语义为验收目标，尚未完成 NPU 数值等价证明。

## 1 特性优先级与适用路径

| 编号 | 技术特性 | 优先级 | 主要适用路径 | 已确认的问题 |
| --- | --- | --- | --- | --- |
| F01 | A2/A3 候选消费者在 QK 之前收缩 K 集合 | P0 | Decode；关闭 replay 的 prefill | mode=2 仍扫描完整历史 K |
| F02 | MRV2 PCP 支持 decoder bounded replay | 质量模式专项 | 长输入 PCP prefill | PCP>1 时关闭 replay；replay 相对完整 prefill 为近似 |
| F03 | 长 KV 与 SWA/ring 分离物理 pool | P1 | 长上下文并发容量 | 长上下文页容量的 31.7321% 是当前布局填充 |
| F04 | C2 完成组后处理紧凑化 | P1 | C2 来源层 prefill | 完成组之后的处理仍覆盖全部 token 行 |
| F05 | 候选结果直接写入稳定输出 buffer | P2 | prefill；ACL Graph 兼容路径 | 层 20 额外复制整个候选输出 |
| F06 | 固定版本的长输入回归与运行路径诊断 | P0 | 所有发布及性能比较 | 文档、配置和真实运行分支不能相互替代 |

P0 表示应首先安排验证和开发，不表示代码已能上线。F01 的 cold prefill 收益必须按 replay 状态分别报告：开启 replay 时，四个候选消费者已只处理尾部窗口，不能把关闭 replay 的工作量减幅移用过来。

## 2 F01 候选消费者在 QK 之前收缩 K 集合

### 2.1 问题及执行证据

当前 `candidate_mode=2` 在 `ProcessVec1` 中、QK 与 K scale 运算完成后才生成候选 mask。原 C++ 调度体执行证明，候选集合不参与 `GetS2BaseBlockNumOnMask` 和 `ProcessBaseBlock` 的跳过判断。

消费者是层 24、28、32、36；来源层 20 仍必须生成候选。层 2、8、14 的 C2 普通索引不使用该候选集合。[native 调用](/Users/loki.yao/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:277)、[Vector 降分](/Users/loki.yao/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h:846)

### 2.2 设计与接口

为 A2/A3 新增独立 candidate consumer 内核或明确的 sparse consumer dispatch。输入继续使用分页 index K、FP16 K scales、INT8 query、FP16 query scales 与 weights，增加显式候选有效块数契约。输出保持原始压缩坐标下的 INT32 position TopK。

执行步骤：

1. 在 Cube 计算前读取候选块 ID，过滤 `-1`，保持请求及 query 行隔离。
2. 将候选块中的有效位置映射为 `(logical_block, page_id, row_offset)`，读取 K/scale。使用真实 block stride，不假定 C2/C1 页面紧密连续。
3. 对候选位置执行原量化 QK 和头归约。来源层产生候选的 score 不能替代消费者自身 query 的 score。
4. 在候选集合内执行 position TopK，将局部索引还原为原始压缩位置，再执行已有因果过滤与时间顺序排序。
5. 处理候选不足、重复、尾块不完整及同分的兼容路径。无法满足已证明的快速路径前提时，保留原全扫描路径。

不得为每条 query gather 完整历史 K，也不得构造 `[chunk, context]` 的全尺寸候选 mask。候选临时存储应受 `candidate_topk_blocks×candidate_block_size` 限制。

### 2.3 必须保持的语义

- 来源层最后一个可见块必须被保留。
- C2 的可见长度为 `floor((position+1)/2)`，奇数残余不能提前暴露。
- `-1`、请求边界、页号及实际 stride 的语义不变。
- 当前候选外 score 为有限负数 `NEG_HUGE_F32=0xF14A3E31`，不是负无穷。候选不足时会选入候选外可见位置。
- 原数值链的 FP16 舍入和 FP32 归约必须保留；NaN、Inf、极端权重与同分边界必须单独定义并验证。
- “数值接近”不替代 TopK 位置集合一致；TopK 边界微小差异会改变后续 attention。

本次 CPU 实验给出了一个反例：候选只有 16 个位置而 TopK=512 时，原语义还返回 496 个候选外位置。仅返回候选 TopK 的实现不满足兼容要求。若快速路径仅接受足量、有效、有限值候选，必须通过设备侧判断或已有 metadata 分派，不能引入 host `.item()` 同步。

### 2.4 可量化目标

以下是相同拓扑下的候选容量上界推导；不是优化后设备测量结果。

| 工作负载 | 当前逻辑 QK 对数 | 紧凑化工作量上界 | 对数减少 |
| --- | ---: | ---: | ---: |
| 64K 冷 prefill，无 replay | 13,958,807,552 | 9,126,871,040 | 34.6157% |
| 128K 冷 prefill，无 replay | 55,834,902,528 | 29,527,998,464 | 47.1155% |
| 64K 冷 prefill，replay，chunk=8192 | 5,519,476,736 | 5,431,591,424 | 1.5923% |
| 128K 冷 prefill，replay，chunk=8192 | 22,044,807,168 | 21,604,860,416 | 1.9957% |
| 64K 单 query decode | 425,984 | 229,376 | 46.1538% |
| 128K 单 query decode | 851,968 | 393,216 | 53.8462% |

四个非消费者来源层的全扫描仍存在。F01 不把整个模型的 prefill 复杂度变成线性，也不把 decode 的整个模型成本变成常数。

### 2.5 变更面与验收

变更位于 indexer dispatch、Torch adapter、native tiling、Cube K 加载、TopK 输出处理及 metadata。A5 已有独立 QSLI，不通过修改公共分派把 A5 切回 A2/A3 路径。不新增环境变量；优先用已有硬件能力和明确 operator contract 选择后端。

验收必须同时满足：

- C1/C2，32/64 heads，TopK 和候选块合法边界，空输入、混合请求、非连续物理页、最后部分块全部通过。
- 64K/128K 测试记录实际 K 访问位置与 QK tile；快速路径不再遍历候选外完整历史。
- 本次五个 NumPy 案例位置集合一致；新增候选不足及同分用例通过原内核差分；差分必须在 A2/A3 执行。
- eager、FULL_DECODE_ONLY、chunked prefill、prefix replay 均保持 graph buffer 地址和同步关系。
- 完整模型分别记录固定检索用例、质量数据集、TTFT、TPOT、吞吐和峰值显存。未取得这些数据之前，发布说明只写工作量改变，不写百分比加速。

## 3 F02 MRV2 PCP 支持 decoder bounded replay

### 3.1 问题及现有边界

`make_decoder_replay` 在 `prefill_context_parallel_size>1` 时直接关闭 replay。另一方面，MRV2 不允许使用旧 `enable_dsa_cp` 替代 PCP。把旧 DSA CP 开关写进 MRV2 启动命令不能解决这个问题，会触发配置异常。

当前层 0–20 产生全部长 KV；在既有近似 replay 模式下，层 21–39 仅执行规则允许的尾部窗口，不能据此证明与完整 prefill 等价。单请求、chunk=8192、window=128 的现有非 PCP replay，使两个输入长度的总层 token 行数均减少 46.7578%，逻辑索引对数分别减少 60.4588%、60.5179%。这些是已经执行的非 PCP 路径工作量对比，不是 PCP 功能实现后的性能或模型质量测量。

### 3.2 新执行方案

1. 层 0–20 保持当前 PCP 全局缓存更新、本地 query 计算的流程。
2. 在层 20 结束后，根据每个请求的全局 query 结束位置、是否 prefill、是否要求 prompt logprobs，生成保留行集合。
3. 将保留行映射回 PCP 的本地 query 编排；有些 rank 的某些请求没有保留行，仍须参加所需 collective。
4. 同步重排 hidden states、positions、input IDs、pre-mix、layer-20 TopK、candidate IDs、candidate lengths 和 DSpark 辅助状态。
5. 为尾部 decoder 生成独立、稳定的 query metadata，保留完整长 KV block table；按现有 replay 规则处理 SWA 起点和缺失前缀。
6. 完成后将需要的 hidden state 或 logits 结果映射回 runner 期望的位置；不能把零填充结果交给要求完整 prompt logprobs 的请求。

接口主要落在 `DecoderReplayState`、`AscendDSAV41PCPMetadataBuilder`、`AscendDecoderReplayLayers` 和 MRV2 model state。现有普通 replay 路径保留；只在新增 PCP layout 通过校验后放开构造门槛。删除一个 `PCP>1` 条件而不实现坐标重排不构成该特性。

### 3.3 正确性与并发约束

- Global cache 坐标不因 query 裁剪而改变。
- 后 19 层的局部 query、稀疏 indices 和 candidate 第一维严格一致。
- Prefix hit、replay rewind 和 chunk 余数使用同一组原始 positions。
- DP 各 rank 对执行模式和总保留行数形成一致决定。
- 空 query rank 不能跳过 hidden、weight 或结果 collective。
- Decode、prompt logprobs、dummy/profile/capture 保留现有行为。
- 不放开 Engram 的 PP/DCP 限制，不把 PCP KV 复制模式当作分片模式。

### 3.4 验收指标

在 64K/128K、chunk=8192、单请求、无 logprobs 的规则场景下，所有 rank 合计的层 query 行数应分别对应非 PCP replay 的 1,395,712 和 2,791,424，另单独统计 padding。不能把 collective 传输行数混入 query 执行行数。

测试矩阵至少包括 PCP2/4、TP1/2/4、混合 prefill/decode、不同请求长度、空 rank、prefix0/prefix90、DSpark 开关、prompt logprobs。对照当前 PCP 实现采集相同总设备数的端到端结果；未实测前不声明 PCP 与 replay 组合的加速倍数。

## 4 F03 长 KV 与 SWA ring 分离物理 pool

### 4.1 目标及布局证据

当前四个 slot 的总 block stride 为 540,928 bytes，而有效长 KV 与 index 数据只需 369,280 bytes。每个逻辑 block 的 171,648-byte 差值来自三个 C2 slot 受 SWA/ring 页尺寸限制。[分配逻辑](/Users/loki.yao/vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py:153)

目标是解除长上下文页与 SWA/ring 页的 stride 耦合，而不是修改模型的 KV 数值精度。

### 4.2 设计

将物理资源分为长 KV 与 index pool、SWA pool、C2 ring pool，并用类型明确的资源句柄管理。长缓存保持 C1/C2 对应的压缩页大小；SWA 使用原窗口缓存页；ring 保持每请求独立的 FP32 32 行状态。

KV scheduler 必须分别申请、释放及保留不同 pool 的块，prefix cache 需要记录资源类型。worker view、metadata、block table 和 PD connector 必须共同升级，不能只修改 allocator 的 `page_size`。

以下语义保持不变：

- C2 三组和 C1 一组的来源所有权。
- 原始 token block 与压缩 storage block 的映射。
- SWA 回收与 replay 规则。
- Ring 不能被不同活跃请求共享。
- Prefix eviction、取消、抢占、恢复、失败清理各释放正确资源。
- P/D 两端传输的 view、stride 与资源类型一致。

仅移走 ring 没有收益保证：SWA 本身同样要求 131,072-byte 页；必须处理两者与长缓存的共池关系。

### 4.3 容量目标及验收

| 长上下文部分，单请求单副本 | 64K | 128K |
| --- | ---: | ---: |
| 当前容量 | 264.125 MiB | 528.25 MiB |
| 有效 payload | 180.3125 MiB | 360.625 MiB |
| 当前填充量 | 83.8125 MiB | 167.625 MiB |

该特性的长期上下文页目标是逼近有效 payload；净进程显存收益必须扣除新增 pool 的独立预留、碎片和 metadata。验收应报告四项：有效 payload、物理 pool bytes、活跃块占用、设备峰值显存。不能仅用“块数增加”代替内存收益证明。

需增加任意物理页排列、别名边界、混合请求、prefix 共享与淘汰、PD、取消和抢占回归。A5 使用不同 packed ABI，应独立设计，不能套用本节 A2/A3 页字节数。

## 5 F04 C2 完成组后处理紧凑化

### 5.1 问题

当前 C2 输出是 token-aligned：先清零 T 行，再仅在完成配对的位置写 pooled 值，后续 norm、index projection、量化与 RoPE 仍处理 T 行。对于单请求偶数完整输入，只有 T/2 行最终写入长缓存。

三个来源层的当前后处理行数为 `3N`，有效行数为 `3N/2`。64K 有 98,304 行、128K 有 196,608 行后处理不产生有效长缓存记录。单就 512→128 的 index K 投影，删除这些行分别减少 12,884,901,888 和 25,769,803,776 次逻辑 multiply/add 运算。这里不包含 hidden→KV/gate 投影。

### 5.2 设计与边界

`compressor_from_projected` 输出紧凑的已完成组，并同时生成原始 completion position 与长 KV/index slot。之后的 RMSNorm、index K 投影、量化、RoPE、scatter 只消费这些组。

每个请求的完成组数为：

```text
completed(start, count) = floor((start + count)/2) - floor(start/2)
```

**不能给整个混合 batch 固定分配 `ceil(total_tokens/2)` 容量。** 多个请求各带一个恰好完成 C2 的 token 时，有效完成组数等于总 token 数。图模式下可以保留 max_tokens 容量，用设备侧有效范围或 request metadata 处理；单请求长 prefill 再采用更紧凑的专用分派。

维持 FP32 WKV、Wgate 和 ring 状态；pooling 的 BF16 舍入及 RMSNorm 顺序保持不变。奇数 chunk 依赖历史 ring 中的上一行，不能跳过对该历史的读取。当前 kernel 要求“先 pooling 再更新 ring”的顺序不变。

### 5.3 验收

- 完成组计数和位置映射通过现有 72 组参考检查，并补充多请求全部单 token 完成组的容量反例。
- 未完成组、首个奇数位置、32 行 ring 回绕、取消后复用、图 padding 不发生错误写入。
- 对照原 FP32→BF16 数值链执行 NPU 差分；同一输入有效长 KV 和 index K 保持约定精度与坐标。
- 记录后处理有效行数、实际投影 M 维度、kernel 次数、buffer bytes 和时间。只减少已完成组之后的工作，不宣称 compressor 全部计算减半。

## 6 F05 候选结果直接写入稳定输出 buffer

### 6.1 问题与目标

层 20 先获得 native operator 新输出，再执行 `shared.candidates.copy_(candidate_out)`。一次冷输入累计 candidate payload 为 `N×2048×4` bytes：64K 是 512 MiB，128K 是 1 GiB。源与目标为不同 buffer；本次形状执行记录确认这次发布复制。

目标是提供可选 candidate output 参数，让 operator 直接写入 graph-stable `shared.candidates`，删除后续发布复制。

### 6.2 设计

扩展 Torch schema、Meta 实现与 adapter，验证输出的 dtype、shape、设备、stride、capacity；对旧调用保留分配输出的兼容行为。Source mode 写 candidates，consumer mode 只读；padding 行、空上下文及空 query 的输出契约不变。

输出 buffer 生命周期覆盖一个完整前向，下一步 source 覆盖活跃行前，所有当前消费者必须结束。多流 query quantization、metadata wait 和图 replay 使用同样的依赖关系，不能靠 Python 引用存在来代替设备事件。

TopK 已有 `indices_output`，不重复实现。源码中出现的同存储 TopK `copy_` 只能通过 profiler 判断是否有真实设备开销，不将其与本特性的不同 buffer 复制合并计算收益。

### 6.3 验收

- 原 source candidate 内容与长度、所有 consumer position TopK 保持一致。
- profiler 中不再存在从临时 native candidate 到稳定候选 buffer 的完整发布复制。
- eager、FULL_DECODE_ONLY、空批次、padding 和反复 graph replay 的地址稳定。
- 分别报告删除的逻辑 copy bytes 和测得的设备流量/时延，不把前者直接称为后者。

## 7 F06 长输入回归与运行路径诊断

### 7.1 已完成的本地部分

本次交付的 `simulate.py` 提供：

- 40 层真实 topology 构造、真实 indexer 分发及源码行 trace。
- 577 组原 C++ 调度体探针。
- 64K/128K、五档 chunk、replay 开关、prefix0/约90% 的 40 组工作量。
- 五组候选数值对比与候选不足反例。
- 72 组压缩及坐标边界参考。
- 实际 allocator 页元组、buffer 字节数、replay 条件和保留行。
- commit、输入 profile 与消费源码的 SHA256。

它提供回归证据的 CPU 基线，不装作 NPU benchmark。

### 7.2 需增加的服务诊断

在服务启动和 benchmark 结果中记录下列字段，并将缺失关键字段的结果排除出性能比较：

| 类别 | 必须记录 |
| --- | --- |
| 版本 | Ascend commit、上游 vLLM commit/tag、CANN、torch_npu、算子包版本 |
| 模型 | checkpoint 路径及 config 指纹、量化模式、完整 topology |
| 硬件 | 具体 NPU 型号、每进程设备、TP/DP/PCP/DCP/EP |
| Attention | A2/A3 QLI 或 A5 DSL、head 数、TopK、候选块数、cache dtype |
| 调度 | 输入总长度、新计算 token、prefix hit、replay rows、chunk、batch |
| 分支 | runner 版本、replay 启用状态及关闭原因、graph runtime mode |
| 功能 | Engram、DSpark、prompt logprobs、PD、prefix caching |
| 结果 | TTFT、TPOT、吞吐、峰值显存、失败请求、输出 token 数 |

把 `make_decoder_replay` 关闭原因作为明确状态输出，防止“配置了 replay 开关”被误认为“本次执行发生了裁剪”。将教程中 MRV1-only 和统一 FP4 字节数的表述拆成按实际硬件/runner 的说明。

### 7.3 设备验证矩阵

对每个变更，在相同硬件、权重、软件和输入下比较 baseline 与 candidate。冷输入使用不共享的 token 序列并检查实际 hit 数为零；热前缀单独预热并记录命中量。

| 维度 | 必测值 |
| --- | --- |
| 输入 | 65536、131072；邻界 65535/65537、131071/131073 |
| 输出 | 1 token 用于 TTFT 隔离；128 token 用于稳态 decode |
| 并发 | 1、8、32；记录实际完成数量 |
| chunk | 2048、4096、8192、16384 |
| prefix | 0%；约90%，记录精确 hit token |
| replay | 实际开启；实际关闭 |
| runner/parallel | MRV1 旧 DSA CP；MRV2 PCP1；MRV2 PCP2/4 |
| graph | eager；FULL_DECODE_ONLY |
| 功能 | 基础目标模型；Engram；DSpark；PD 分别叠加 |

`max_model_len` 必须容纳输入和输出；131,072 输入加 128 输出不能用 131,072 的总长度上限进行该项测量。测试配置的名字包含“128K”并不证明输入真的有 131,072 token，必须使用 tokenizer 后长度或显式 token IDs。

每个固定配置预热后执行至少 30 个完成请求，报告样本数、p50/p95 及失败数；此处 30 是本设计的最低测量样本要求，不是本次已经取得的样本。跨并发比较同时保留逐请求与系统吞吐指标。质量评测单独运行，不能用性能请求的随机文本来证明模型能力。

### 7.4 分特性验收契约

| 特性 | 结构性验收条件 | 设备性能验收内容 |
| --- | --- | --- |
| F01 | 消费者 Cube 不扫描候选外完整历史；输出契约一致 | indexer kernel 时间、K 流量、TTFT/TPOT |
| F02 | PCP 后 19 层按正确全局尾部行集合执行 | query 工作量、collective 时间、TTFT |
| F03 | 长 pool stride 不再受 SWA/ring 最大页约束 | 活跃容量、峰值显存、最大成功并发 |
| F04 | 完成组后处理 M 维只含有效组 | 投影与 postprocess 时间、workspace |
| F05 | 删除 candidate 发布复制 | copy 事件、实际流量和时延 |
| F06 | 每份结果能还原版本、分支与输入 | 对照实验可复现，失效项显式报错 |

设备性能提升的验收必须使用实测数据，不能以本报告的逻辑运算减少率替代。所有行为变更还需遵守仓库测试要求、架构 review 和用户可见变更 release note 要求。

## 8 实施顺序

第一阶段先把 F06 的运行路径字段、冷/热输入区分和源码仿真纳入验证流程；并以现有功能建立 MRV1 DSA CP、MRV2 PCP1 replay、MRV2 PCP4 三组真实基线。配置切换不能在缺少测量时被称作优化完成。

第二阶段开发 F01 与 F02。F01 优先覆盖长上下文 decode 和当前未裁剪的消费者；F02解决 PCP 与已有 decoder 裁剪不能同时工作的结构性限制。两者最终叠加时重新统计收益，不能直接相加百分比。

第三阶段开发 F03 与 F04。前者涉及 scheduler、allocator 和 connector，变更范围最大；后者聚焦 C2 后处理，保持原投影精度。F05 在输出 ABI 和 graph 测试完整后纳入。

保留三个全扫描 C2 来源和层 20 的精确索引意味着仍有二次 prefill 项。进一步改变这些来源的候选生成策略会改变模型的索引算法；本次没有模型质量证据支持该方向，不把它列入上述保持原语义的六项特性。
