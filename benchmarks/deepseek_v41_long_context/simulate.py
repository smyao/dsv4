# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Source-linked CPU experiments; never estimates NPU time or full-model accuracy.

Requires Python 3.10+, NumPy and clang++/g++. No torch, vLLM, CANN or weights.
Actual Python functions are AST-extracted without rewriting their bodies.
Actual C++ scheduling bodies are compiled with instrumented device-operation stubs.
The numerical experiments are separate NumPy references, not native NPU kernels.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "vllm_ascend/models/deepseek_v41/model.py"
CACHE = "vllm_ascend/models/deepseek_v41/cache_config.py"
REPLAY = "vllm_ascend/models/deepseek_v41/decoder_replay_layers.py"
REPLAY_STATE = "vllm_ascend/models/deepseek_v41/decoder_replay_state.py"
KERNEL = "csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h"
VECTOR = "csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h"
LENGTHS = (65536, 131072)
CHUNKS = (2048, 4096, 8192, 16384)
SEED = 41
BLOCK_SIZE = 128
BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
RING_ROWS = 32
NEG_HUGE_BITS = 0xF14A3E31  # quant_lightning_indexer_v2_vector.h:NEG_HUGE_F32


def load_symbols(path: str, names: tuple[str, ...], namespace: dict) -> dict:
    """Run selected source definitions, preserving original code and line numbers."""
    tree = ast.parse((ROOT / path).read_text())
    selected = []
    for name in names:
        if "." in name:
            parent, child = name.split(".")
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
            node = next(n for n in cls.body if getattr(n, "name", None) == child)
        else:
            node = next(n for n in tree.body if getattr(n, "name", None) == name)
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        selected.append(node)
    prefix = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=prefix + selected, type_ignores=[]))
    env = {"__name__": __name__, "dataclass": dataclasses.dataclass, "Any": Any, **namespace}
    exec(compile(module, str(ROOT / path), "exec"), env)
    return env


def cpp_body(text: str, name: str) -> str:
    start = text.index("QLIV2Preload<QLIV2T>::" + name)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[opening:end]


def profile_and_topology():
    fixture = "tests/ut/models/test_deepseek_v41_layer_plan.py"
    env = load_symbols(fixture, ("text_config",), {"DeepseekV41Config": lambda **kw: kw["text_config"]})
    profile = env["text_config"]()
    # Geometry explicitly supplied by tests/deepseek_v41_utils.py:make_cache_config.
    profile.update(sliding_window=128, head_dim=512, index_head_dim=128, block_size=BLOCK_SIZE)
    env = load_symbols(
        MODEL,
        ("DeepseekV41LayerRole", "DeepseekV41Topology", "_latest_source", "build_layer_plan"),
        {},
    )
    return profile, env["build_layer_plan"](SimpleNamespace(**profile))


def replay_experiments(profile):
    modes = SimpleNamespace(NONE="NONE", FULL="FULL", FULL_DECODE_ONLY="FULL_DECODE_ONLY")
    log = SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None)
    env = load_symbols(
        REPLAY,
        ("make_decoder_replay",),
        {
            "CUDAGraphMode": modes,
            "logger": log,
            "enable_dsa_cp": lambda: False,
            "AscendDecoderReplayLayers": lambda *a: SimpleNamespace(window=a[0]),
        },
    )
    observations = []
    for label, pcp, pp, graph, enabled, late_engram in (
        ("eligible", 1, 1, modes.NONE, True, False),
        ("pcp4", 4, 1, modes.NONE, True, False),
        ("pp2", 1, 2, modes.NONE, True, False),
        ("full_graph", 1, 1, modes.FULL, True, False),
        ("flag_off", 1, 1, modes.NONE, False, False),
        ("late_engram", 1, 1, modes.NONE, True, True),
    ):
        cfg = SimpleNamespace(**profile)
        if late_engram:
            cfg.engram_layer_ids = [1, 25]
        model = SimpleNamespace(
            config=cfg,
            end_layer=40,
            use_sequence_parallel=False,
            _run_replay_layers=None,
            topk_indices_buffer=None,
            candidate_indices_buffer=None,
            candidate_lengths_buffer=None,
            topk_lengths_buffer=None,
        )
        runtime = SimpleNamespace(
            cache_config=SimpleNamespace(swa_bounded_replay=enabled),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=pp,
                prefill_context_parallel_size=pcp,
                decode_context_parallel_size=1,
                use_ubatching=False,
            ),
            speculative_config=None,
            lora_config=None,
            compilation_config=SimpleNamespace(cudagraph_mode=graph),
        )
        result = env["make_decoder_replay"](model, runtime)
        observations.append(
            {
                "case": label,
                "enabled": result is not None,
                "first_replay_layer": getattr(model, "decoder_replay_start", None),
            }
        )
    assert [r["enabled"] for r in observations] == [True, False, False, False, False, False]

    prepare = load_symbols(
        REPLAY_STATE,
        ("DecoderReplayState.prepare",),
        {
            "np": np,
            "CUDAGraphMode": modes,
            "get_mrv2_in_profile_run": lambda: False,
            "enable_dsa_cp": lambda: False,
        },
    )["prepare"]
    row_tests = []
    for queries, prefills, keep in (
        ([8192], [True], [False]),
        ([8192, 1, 127], [True, False, True], [False, False, False]),
        ([8192], [True], [True]),
    ):
        batch = SimpleNamespace(
            num_reqs=len(queries),
            query_start_loc_np=np.cumsum([0, *queries]),
            is_prefilling_np=np.array(prefills),
            idx_mapping_np=np.arange(len(queries)),
            is_dummy=False,
        )
        state = SimpleNamespace(
            slots=object(),
            layers=SimpleNamespace(window=profile["sliding_window"]),
            config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)),
            keeps_rows=np.array(keep),
            dropped_before=SimpleNamespace(np=np.zeros(len(queries) + 1, np.int32), copy_to_gpu=lambda n: None),
            rows=np.arange(sum(queries)),
            _gather=lambda *a: None,
            _build_metadata=lambda *a: {},
            forward_context=None,
        )
        prepare(state, SimpleNamespace(batch=batch), modes.NONE)
        expected = sum(
            min(q, profile["sliding_window"]) if p and not k else q for q, p, k in zip(queries, prefills, keep)
        )
        assert state.num_actual_tokens == expected
        row_tests.append({"query_lens": queries, "prompt_logprobs": keep, "retained": expected})
    return {"gates": observations, "actual_prepare": row_tests}


@dataclasses.dataclass(frozen=True)
class PageSpec:
    unpadded_page_size_bytes: int


@dataclasses.dataclass(frozen=True)
class MlaSpec(PageSpec):
    pass


@dataclasses.dataclass(frozen=True)
class SwaSpec(PageSpec):
    pass


@dataclasses.dataclass(frozen=True)
class RingSpec(PageSpec):
    pass


def cache_experiment(profile, topology):
    env = load_symbols(
        CACHE,
        ("_layer_number", "_draft_layer_number", "get_layer_tuples"),
        {
            "AscendMLAAttentionSpec": MlaSpec,
            "AscendSlidingWindowMLASpec": SwaSpec,
            "CircularBufferSpec": RingSpec,
        },
    )
    specs = {}
    for role in topology.layers:
        prefix = f"model.layers.{role.layer_idx}.self_attn"
        specs[prefix + ".swa_cache"] = SwaSpec(BLOCK_SIZE * profile["head_dim"] * BF16_BYTES)
        if role.is_kv_source:
            rows = BLOCK_SIZE // role.compress_ratio
            specs[prefix + ".long_kv_cache"] = MlaSpec(rows * profile["head_dim"] * BF16_BYTES)
            specs[prefix + ".indexer.k_cache"] = MlaSpec(rows * (profile["index_head_dim"] + BF16_BYTES))
            if role.compress_ratio == 2:
                specs[prefix + ".compressor.state_cache"] = RingSpec(RING_ROWS * 2 * profile["head_dim"] * FP32_BYTES)
    sizes, tuples = env["get_layer_tuples"](specs)
    raw = sum(s.unpadded_page_size_bytes for s in specs.values() if isinstance(s, MlaSpec))
    assert sizes == [131072, 131072, 131072, 147712]
    return {
        "slot_page_bytes": sizes,
        "slot_layers": tuples,
        "pool_bytes_per_block": sum(sizes),
        "long_kv_and_index_payload_per_block": raw,
        "padding_bytes_per_block": sum(sizes) - raw,
        "cases": [
            {
                "tokens": n,
                "long_context_pool_bytes": n // BLOCK_SIZE * sum(sizes),
                "long_context_payload_bytes": n // BLOCK_SIZE * raw,
                "padding_bytes": n // BLOCK_SIZE * (sum(sizes) - raw),
            }
            for n in LENGTHS
        ],
    }


def build_probe(output: Path) -> Path:
    text = (ROOT / KERNEL).read_text()
    mask = cpp_body(text, "GetS2BaseBlockNumOnMask")
    block = cpp_body(text, "ProcessBaseBlock")
    assert "candidate" not in mask.lower() and "candidate" not in block.lower()
    assert "matmulService.ComputeMm1(curRunInfo)" in block
    harness = (Path(__file__).parent / "probe_harness.cpp.in").read_text()
    harness = harness.replace("@@MASK_BODY@@", mask).replace("@@BLOCK_BODY@@", block)
    cpp = output / "source_probe.cpp"
    cpp.write_text(harness)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("clang++ or g++ is required; no probe result will be fabricated")
    binary = output / "source_probe"
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
    return binary


def arithmetic_and_probe(profile, topology, binary):
    cases, requests = [], {}
    capacity = profile["candidate_topk_blocks"] * profile["candidate_block_size"]
    for length in LENGTHS:
        for chunk in (*CHUNKS, length):
            for replay in (False, True):
                for prefix_fraction in (0.0, 0.9):
                    # Controlled synthetic prefix; block-aligned and excludes scheduler rewind.
                    prefix = int(length * prefix_fraction) // BLOCK_SIZE * BLOCK_SIZE
                    pairs = compact_pairs = layer_rows = candidate_copy = 0
                    per_layer, keys = {}, []
                    for begin in range(prefix, length, chunk):
                        end = min(begin + chunk, length)
                        for role in topology.layers:
                            q = (
                                min(end - begin, profile["sliding_window"])
                                if replay and role.layer_idx > 20
                                else end - begin
                            )
                            layer_rows += q
                            if not role.is_index_source:
                                continue
                            first = end - q + 1
                            visible = np.arange(first, end + 1, dtype=np.int64) // role.compress_ratio
                            work = int(visible.sum())
                            optimized = int(np.minimum(visible, capacity).sum()) if role.uses_candidate_filter else work
                            pairs += work
                            compact_pairs += optimized
                            per_layer[role.layer_idx] = per_layer.get(role.layer_idx, 0) + work
                            mode = 1 if role.is_candidate_source else 2 if role.uses_candidate_filter else 3
                            key = (q, end, role.compress_ratio, mode)
                            requests[key] = None
                            keys.append(key)
                            if role.is_candidate_source:
                                candidate_copy += q * profile["candidate_topk_blocks"] * INT32_BYTES
                    cases.append(
                        {
                            "tokens": length,
                            "chunk": chunk,
                            "replay": replay,
                            "prefix_tokens": prefix,
                            "scheduled_tokens": length - prefix,
                            "steps": (length - prefix + chunk - 1) // chunk,
                            "logical_qk_pairs": pairs,
                            "candidate_compacted_pair_upper_bound": compact_pairs,
                            "candidate_pair_reduction": 1 - compact_pairs / pairs,
                            "layer_token_rows": layer_rows,
                            "per_index_layer_pairs": per_layer,
                            "candidate_copy_payload_bytes": candidate_copy,
                            "keys": keys,
                        }
                    )
    for end in (1, 127, 128, 129, 16383, 16384, 16385, 65535, 65536, 65537, 131071, 131072, 131073):
        for ratio in (1, 2):
            for mode in (1, 2, 3):
                requests[(min(end, 127), end, ratio, mode)] = None
    payload = "".join(" ".join(map(str, key)) + "\n" for key in requests)
    result = subprocess.run([str(binary)], input=payload, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(requests)
    probes = {key: json.loads(line) for key, line in zip(requests, lines)}
    for key, value in probes.items():
        q, end, ratio, mode = key
        expected = sum((min((end - q + row + 4) // ratio, end // ratio) + 2047) // 2048 for row in range(0, q, 4))
        assert value["cube_calls"] == expected
        assert value["vector_calls"] == expected
        assert value["mode"] == mode
    for case in cases:
        keys = case.pop("keys")
        case["native_schedule_cube_calls"] = sum(probes[k]["cube_calls"] for k in keys)
        case["native_group_qk_pairs"] = sum(probes[k]["group_pairs"] for k in keys)
        case["qk_integer_ops_heads32_dim128"] = case["logical_qk_pairs"] * 2 * 32 * 128
        case["qk_integer_ops_heads64_dim128"] = case["logical_qk_pairs"] * 2 * 64 * 128
    # Decode includes the current token: positions 65535 and 131071 respectively.
    decode = []
    for n in LENGTHS:
        pairs = sum(n // role.compress_ratio for role in topology.layers if role.is_index_source)
        compact = sum(
            min(n // role.compress_ratio, capacity) if role.uses_candidate_filter else n // role.compress_ratio
            for role in topology.layers
            if role.is_index_source
        )
        decode.append(
            {
                "context_including_current_token": n,
                "qk_pairs_per_query": pairs,
                "compacted_pairs_per_query": compact,
                "reduction": 1 - compact / pairs,
            }
        )
    return {
        "prefill": cases,
        "decode": decode,
        "probe_cases": len(probes),
        "probe_samples": [
            {"input": list(k), **v} for k, v in probes.items() if k[0] in (128, 8192) and k[1] in LENGTHS
        ],
    }


def ordered_topk(scores, k):
    """Explicit CPU tie policy; does not claim to reproduce Ascend SortAll ties."""
    return np.lexsort((np.arange(scores.size), -scores))[:k]


def index_scores(query, keys, weights, qscale, kscale):
    # Small integer operands ensure FP32 matmul equals the INT32 dot exactly.
    qk = query.astype(np.float32) @ keys.astype(np.float32).T
    qk = np.maximum((qk / 1024).astype(np.float16), np.float16(0))
    head_weight = (weights.astype(np.float16) * qscale.astype(np.float16)).astype(np.float16)
    weighted = qk.astype(np.float32) * head_weight[:, None].astype(np.float32)
    # Fix head accumulation order for dense and compact references.
    reduced = np.zeros(keys.shape[0], dtype=np.float32)
    for head in weighted:
        reduced += head.astype(np.float32)
    return reduced * kscale.astype(np.float32)


def numerical_candidates(profile):
    rng = np.random.default_rng(SEED)
    results = []
    heads, width = 32, profile["index_head_dim"]
    for n in (16383, 16384, 16385, *LENGTHS):
        keys = rng.integers(-7, 8, (n, width), dtype=np.int8)
        q = rng.integers(-7, 8, (2, heads, width), dtype=np.int8)
        weights = rng.uniform(-0.9, 1.1, (2, heads)).astype(np.float16)
        qs = np.full(heads, 0.25, dtype=np.float16)
        ks = rng.choice(np.array([0.125, 0.25, 0.5, 1], np.float16), n)
        integer = q[0].astype(np.int32) @ keys[:128].astype(np.int32).T
        floating = q[0].astype(np.float32) @ keys[:128].astype(np.float32).T
        assert np.array_equal(integer, floating)
        source = index_scores(q[0], keys, weights[0], qs, ks)
        dense = index_scores(q[1], keys, weights[1], qs, ks)
        block_size = profile["candidate_block_size"]
        block_scores = np.pad(source, (0, -n % block_size), constant_values=-np.inf).reshape(-1, block_size).max(1)
        block_scores[-1] = np.inf
        blocks = ordered_topk(block_scores, min(profile["candidate_topk_blocks"], len(block_scores)))
        selected = np.sort((blocks[:, None] * block_size + np.arange(block_size)).ravel())
        selected = selected[selected < n]
        mask = np.ones(n, dtype=bool)
        mask[selected] = False
        # Native finite penalty chain: s + (NEG_HUGE - s) * isOut.
        neg_huge = np.array([NEG_HUGE_BITS], dtype=np.uint32).view(np.float32)[0]
        penalized = dense + (neg_huge - dense) * mask.astype(np.float32)
        compact = index_scores(q[1], keys[selected], weights[1], qs, ks[selected])
        assert np.array_equal(compact, dense[selected])
        baseline_ids = np.sort(ordered_topk(penalized, profile["index_topk"]))
        compact_ids = np.sort(selected[ordered_topk(compact, profile["index_topk"])])
        assert np.array_equal(baseline_ids, compact_ids)
        sorted_scores = np.sort(compact)[::-1]
        results.append(
            {
                "visible_keys": n,
                "candidate_keys": len(selected),
                "topk_equal": True,
                "candidate_score_max_abs_error": float(np.max(np.abs(compact - dense[selected]))),
                "boundary_score_tie": bool(sorted_scores[511] == sorted_scores[512]),
                "last_block_pinned": bool(len(block_scores) - 1 in blocks),
            }
        )
    # Required counterexample: a gather-only implementation drops native filler indices.
    scores = np.arange(1024, dtype=np.float32)
    eligible = np.arange(16)
    penalized = np.full(1024, neg_huge)
    penalized[eligible] = scores[eligible]
    dense_result = ordered_topk(penalized, 512)
    assert len(dense_result) == 512 and np.count_nonzero(dense_result >= 16) == 496
    return {
        "seed": SEED,
        "heads": heads,
        "width": width,
        "cases": results,
        "underfilled_counterexample": {
            "candidate_positions": 16,
            "requested_topk": 512,
            "native_semantic_valid_fillers": 496,
            "naive_compact_output_length": 16,
        },
    }


class ShapeTensor:
    """Only shapes/alias identities; never a numerical tensor or NPU emulation."""

    def __init__(self, shape, name, events):
        self.shape, self.name, self.events = tuple(shape), name, events
        self.device = "cpu_stub"

    def __getitem__(self, key):
        assert isinstance(key, slice)
        start, stop, step = key.indices(self.shape[0])
        return ShapeTensor((len(range(start, stop, step)), *self.shape[1:]), self.name, self.events)

    def squeeze(self, dim):
        shape = list(self.shape)
        if shape[dim] == 1:
            shape.pop(dim)
        return ShapeTensor(shape, self.name, self.events)

    def to(self, *args):
        return self

    def copy_(self, source):
        assert self.shape == source.shape
        self.events.append(
            {
                "event": "copy_call",
                "source": source.name,
                "destination": self.name,
                "shape": self.shape,
                "same_storage": source.name == self.name,
            }
        )
        return self


def dispatch_trace(topology):
    events, executed_lines = [], set()
    tokens = 8192

    def tensor(shape, name):
        return ShapeTensor(shape, name, events)

    def native_stub(q, k, w, qs, ks, topk, quant, **kw):
        events.append(
            {
                "event": "native_indexer_call",
                "candidate_mode": kw["candidate_mode"],
                "ratio": kw["cmp_ratio"],
                "query_shape": q.shape,
                "candidate_input": kw["candidate_topk_index"] is not None,
            }
        )
        return (
            tensor((tokens, 1, topk), "native_topk"),
            None,
            tensor((tokens, 1, topology.candidate_topk_blocks), "native_candidates"),
        )

    index_fn = load_symbols(
        "vllm_ascend/models/deepseek_v41/indexer.py",
        ("DeepseekV41Indexer.select_projected",),
        {
            "torch": SimpleNamespace(
                float16="float16",
                ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_quant_lightning_indexer_v3=native_stub)),
            ),
            "wait_for_device_metadata": lambda *a: None,
            "DeviceMetadataStage": SimpleNamespace(INDEXER=1),
            "prepare_indexer_indices": lambda *a, **kw: kw["indices_output"],
        },
    )["select_projected"]
    cache = (tensor((1024, 128, 1, 128), "index_cache"), tensor((1024, 128, 1, 1), "index_scales"))
    context = SimpleNamespace(no_compile_layers={"source": SimpleNamespace(kv_cache=[cache])})
    layer_fn = load_symbols(
        "vllm_ascend/attention/dsa_v41.py",
        ("AscendDSAV41Impl._select_sparse_indices",),
        {
            "get_forward_context": lambda: context,
        },
    )["_select_sparse_indices"]
    shared = SimpleNamespace(
        topk_indices=tensor((tokens, 512), "shared_topk"),
        candidates=tensor((tokens, 1, 2048), "shared_candidates"),
        candidate_lengths=None,
        topk_lengths=None,
    )
    metadata = SimpleNamespace(
        indexer=SimpleNamespace(
            cache=SimpleNamespace(
                max_cache_seq_len=131072,
                query_start_loc=None,
                cache_seq_lens=None,
                cmp_residual=None,
                max_query_len=tokens,
                qli_metadata=object(),
                block_table=None,
            )
        )
    )
    prepared = SimpleNamespace(
        query=tensor((tokens, 32, 128), "query"),
        weights=tensor((tokens, 32), "weights"),
        quantized_query=None,
        query_scale=None,
        quantize_done=None,
    )
    input_tensor = tensor((tokens, 1), "hidden")

    def tracer(frame, event, arg):
        if old_trace is not None:
            old_trace(frame, event, arg)
        if event == "line" and frame.f_code.co_filename in (
            index_fn.__code__.co_filename,
            layer_fn.__code__.co_filename,
        ):
            executed_lines.add((str(Path(frame.f_code.co_filename).relative_to(ROOT)), frame.f_lineno))
        return tracer

    old_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        for role in topology.layers:
            if role.has_long_context:
                rows = BLOCK_SIZE // role.compress_ratio
                context.no_compile_layers["source"].kv_cache[0] = (
                    tensor((1024, rows, 1, 128), "index_cache"),
                    tensor((1024, rows, 1, 1), "index_scales"),
                )
                metadata.indexer.cache.max_cache_seq_len = 131072 // role.compress_ratio
            indexer = SimpleNamespace(
                packed_cache_ops=None,
                index_topk=512,
                compress_ratio=role.compress_ratio,
                quantize_query=lambda query: (query, tensor((tokens, 32), "qscale")),
            )
            indexer.select_projected = lambda *a, owner=indexer, **kw: index_fn(owner, *a, **kw)
            impl = SimpleNamespace(
                role=role, topology=topology, index_k_source_prefix="source", _indexer_hidden_states=lambda h, m: h
            )
            attn = SimpleNamespace(shared_state=shared, indexer=indexer, uses_a5_packed_cache=False)
            before = len(events)
            result = layer_fn(impl, attn, input_tensor, None, None, None, None, metadata, prepared)
            events.append(
                {
                    "event": "layer_return",
                    "layer": role.layer_idx,
                    "indexer_called": any(e["event"] == "native_indexer_call" for e in events[before:]),
                    "result_is_none": result is None,
                }
            )
    finally:
        sys.settrace(old_trace)
    calls = [e for e in events if e["event"] == "native_indexer_call"]
    assert [e["candidate_mode"] for e in calls] == [3, 3, 3, 1, 2, 2, 2, 2]
    return {
        "native_calls": len(calls),
        "events": events,
        "executed_source_lines": [{"file": p, "line": n} for p, n in sorted(executed_lines)],
    }


def boundary_tests(profile):
    compressed_slots = load_symbols(
        "vllm_ascend/attention/dsa_v41.py",
        ("compressed_slot_mapping",),
        {"torch": SimpleNamespace(where=np.where)},
    )["compressed_slot_mapping"]
    checks = 0
    for n in (127, 128, 129, 16383, 16384, 16385, 65535, 65536, 65537, 131071, 131072, 131073):
        for chunk in (1, 127, 128, 129, 2047, 8192):
            completions = []
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                completions.extend((p + 1) // 2 - 1 for p in range(begin, end) if (p + 1) % 2 == 0)
            assert completions == list(range(n // 2))
            positions = np.arange(n, dtype=np.int64)
            for ratio in (1, 2):
                visible = (positions + 1) // ratio
                slots = (positions // BLOCK_SIZE) * (BLOCK_SIZE // ratio) + positions % BLOCK_SIZE // ratio
                assert np.array_equal(slots, positions // ratio)
                assert visible[-1] == n // ratio
                published = compressed_slots(positions, ratio)
                actual = published[published >= 0]
                assert np.array_equal(actual, np.arange(n // ratio))
            checks += 1
    # Chronological-sort FP32 bit encoding from prepare_indexer_indices.py.
    selected = np.array([131071, -1, 65535, 131072, 0, 127], np.int32)
    visible = 131072
    sentinel = np.iinfo(np.int32).max
    valid = np.where((selected >= 0) & (selected < visible), selected, sentinel).astype(np.int32)
    shift, negative = 1 << 23, 0x81800000 - (1 << 32)
    bits = np.where(valid < 2 * shift, negative - valid.astype(np.int64), valid.astype(np.int64) - shift).astype(
        np.int32
    )
    sorted_bits = np.sort(bits.view(np.float32)).view(np.int32)
    restored = np.where(sorted_bits < 0, negative - sorted_bits.astype(np.int64), sorted_bits.astype(np.int64) + shift)
    restored[restored == sentinel] = -1
    assert restored.tolist() == [0, 127, 65535, 131071, -1, -1]
    assert compressed_slots(np.array([-1, 0, 1, 127, 128, 129]), 2).tolist() == [-1, -1, 0, 63, -1, 64]
    return {
        "chunk_and_compression_cases": checks,
        "index_sort_128k": restored.tolist(),
        "actual_compressed_slot_mapping_executed": True,
    }


def source_guards():
    vector = (ROOT / VECTOR).read_text()
    indexer = (ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text()
    assert "BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx)" in vector
    assert vector.index("AscendC::Mul(mmInUb, mmInUb, kScaleUb") < vector.index("BuildCandidateMask(info, cuS1Idx")
    assert "npu_quant_lightning_indexer_v3" in indexer
    assert "ReplicatedLinear" in indexer
    assert (
        "source_cache[2].squeeze(2)" in (ROOT / "vllm_ascend/models/deepseek_v41/mixed_quant_attention.py").read_text()
    )
    return {"a2_a3_candidate_mask_after_qk": True, "index_heads_replicated": True, "a5_uses_separate_folded_qsli": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--pdb", action="store_true", help="Break after profile and real topology creation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile, topology = profile_and_topology()
    if args.pdb:
        breakpoint()
    guards = source_guards()
    with tempfile.TemporaryDirectory(prefix="dsv41-probe-") as temporary:
        temp = Path(temporary)
        binary = build_probe(temp)
        work = arithmetic_and_probe(profile, topology, binary)
        shutil.copyfile(temp / "source_probe.cpp", args.output / "source_probe.cpp")
    sources = (
        MODEL,
        CACHE,
        REPLAY,
        REPLAY_STATE,
        KERNEL,
        VECTOR,
        "vllm_ascend/models/deepseek_v41/indexer.py",
        "vllm_ascend/attention/dsa_v41.py",
        "vllm_ascend/attention/context_parallel/dsa_v41_cp.py",
        "vllm_ascend/models/deepseek_v41/mixed_quant_attention.py",
        "vllm_ascend/core/kv_cache_interface.py",
        "vllm_ascend/ops/triton/prepare_indexer_indices.py",
        "tests/ut/models/test_deepseek_v41_layer_plan.py",
        "tests/deepseek_v41_utils.py",
        "csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_vector.h",
        "csrc/attention/quant_lightning_indexer_v2/tests/pytest/quant_lightning_indexer_v2_golden.py",
        "vllm_ascend/platform.py",
        "vllm_ascend/device/hardware_profile.py",
        "vllm_ascend/ops/triton/compressor/compressor_triton.py",
    )
    manifest = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "npu_executed": False,
            "full_model_executed": False,
        },
        "profile": profile,
        "topology": [dataclasses.asdict(role) for role in topology.layers],
        "source_guards": guards,
        "source_sha256": manifest,
        "replay": replay_experiments(profile),
        "cache": cache_experiment(profile, topology),
        "work": work,
        "numerical": numerical_candidates(profile),
        "boundaries": boundary_tests(profile),
        "dispatch_trace": dispatch_trace(topology),
        "c2_postprocess": [
            {
                "tokens": n,
                "current_rows": 3 * n,
                "valid_rows": 3 * (n // 2),
                "removable_index_projection_ops": 3 * (n // 2) * 2 * profile["head_dim"] * profile["index_head_dim"],
            }
            for n in LENGTHS
        ],
        "buffer_bytes_by_chunk": [
            {"chunk": c, "candidate": c * 2048 * 4, "topk": c * 512 * 4, "three_ring_pooled_buffers": 3 * c * 512 * 2}
            for c in (*CHUNKS, *LENGTHS)
        ],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "result": "PASS",
                "probe_cases": work["probe_cases"],
                "prefill_cases": len(work["prefill"]),
                "numeric_cases": len(report["numerical"]["cases"]),
                "boundary_cases": report["boundaries"]["chunk_and_compression_cases"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
