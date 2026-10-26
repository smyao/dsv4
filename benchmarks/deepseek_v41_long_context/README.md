# DeepSeek V4.1 long context source experiments

This is a CPU analysis harness, not an NPU benchmark or a full-model emulator.
It uses Python 3.10+, NumPy, and clang++ or g++. It does not import torch,
torch_npu, vLLM, CANN, or model weights.

```bash
python3 benchmarks/deepseek_v41_long_context/simulate.py
```

On the authoring Mac, use the bundled NumPy runtime and the installed Xcode:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  /Users/loki.yao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  benchmarks/deepseek_v41_long_context/simulate.py
```

The harness executes selected, unmodified Python function bodies from the
checkout through AST extraction. Device tensors and native calls in the dispatch
experiment are shape/call-recording stubs. The C++ harness compiles two actual
kernel scheduling bodies, with device operations replaced by counters.
Numerical references are separate NumPy implementations based on the checked-in
INT8 golden. They do not establish Ascend numerical equivalence or model quality.

The fixture combines the 40-layer topology in
`tests/ut/models/test_deepseek_v41_layer_plan.py` with the 512/128 head dimensions,
128-token window and block geometry in `tests/deepseek_v41_utils.py`.
It is not a downloaded checkpoint configuration. Index-head operation counts
are provided for both supported 32- and 64-head contracts.

Results include source hashes, the commit, exact input profile, cold and
controlled prefix-hit workloads, replay gates, dispatch traces, cache byte
counts, native schedule counters, numerical comparisons and boundary checks.
The candidate-compaction counts are capacity upper bounds for a proposed path;
the production kernel is unchanged. No NPU latency is inferred from CPU time.

For an interactive breakpoint, add `--pdb`. At the initial prompt:

```text
p (profile["num_hidden_layers"], topology.index_source_layer_ids)
b /Users/loki.yao/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:278
c
p (mode, self.compress_ratio, query.shape)
```

Use your checkout's absolute path on another machine. The native indexer is a
stub at this breakpoint, while the enclosing Python control flow is the actual
source function. `results/breakpoints.txt` records the authoring run.

The generated `results/source_probe.cpp` can also be compiled independently:

```bash
clang++ -std=c++17 -O2 \
  benchmarks/deepseek_v41_long_context/results/source_probe.cpp \
  -o /tmp/dsv41-source-probe
```

Feed whitespace-separated `query_count context_end compression_ratio mode`
rows on stdin. Example `8192 131072 1 2` probes the final 8K chunk of a 128K
C1 candidate consumer. Results are Cube/Vector dispatch counts and unpadded
query-group pair counts, not accelerator cycles.

The detailed Chinese reports are
[analysis](../../docs/source/developer_guide/deepseek_v41_long_context_analysis_zh.md)
and [technical features](../../docs/source/developer_guide/deepseek_v41_long_context_features_zh.md).

The [2026 literature assessment](../../docs/source/developer_guide/deepseek_v41_2026_research_zh.md)
adds paper-to-source decisions and corrects the accuracy classification of
bounded replay: it is an approximate SWA reconstruction, not a general proof of
equivalence to full prefill. The public original checkpoint's key topology was
subsequently verified against the fixture; deployed weights remain untested.

Run the additional CPU decision checks with the same NumPy Python runtime:

```bash
python3 benchmarks/deepseek_v41_long_context/research_checks.py
```

This executes the local budget-update function with scalar-read counters,
checks exact-selection recovery, constructs numerical counterexamples, and
computes explicitly assumed cost scenarios. It is not a reproduction of any
paper's accelerator performance. See `results/research_checks.json` and
`results/research_sources.json` for outputs and research provenance.
