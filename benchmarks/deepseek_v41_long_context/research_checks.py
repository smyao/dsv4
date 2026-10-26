"""Reproducible decision checks for the 2026 literature report; no device timing."""

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def execute_budget_source(root: Path) -> dict:
    """Execute the checkout's budget method with a counted scalar-read stub."""
    path = root / "vllm_ascend/spec_decode/utils.py"
    tree = ast.parse(path.read_text())
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamicSpecScheduler")
    method = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == "compute_verify_budget")
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {"math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    reads = []

    class CountedTensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def __ge__(self, other):
            return CountedTensor(self.value >= other)

        def sum(self):
            return CountedTensor(self.value.sum())

        def item(self):
            reads.append(True)
            return self.value.item()

    state = SimpleNamespace(
        _steps_since_budget_update=0,
        budget_update_interval=16,
        budget_threshold=0.3,
        min_k=1,
        num_speculative_tokens=5,
        budget_k=5,
    )
    survival = np.power(0.9, np.arange(1, 6))[None, :]
    for _ in range(32):
        namespace["compute_verify_budget"](state, CountedTensor(survival))
    assert len(reads) == 2 and state.budget_k == 5
    # These dimensionless costs are deliberately synthetic, not hardware data.
    relative_step_cost = [1.0, 1.1, 1.2, 1.3, 2.3, 3.3]
    choices = []
    for draft_len, cost in enumerate(relative_step_cost):
        expected_output = 1.0 + float(survival[0, :draft_len].sum())
        choices.append(
            {
                "draft_len": draft_len,
                "relative_cost": cost,
                "expected_output": expected_output,
                "normalized_throughput": expected_output / cost,
            }
        )
    best = max(choices, key=lambda item: item["normalized_throughput"])
    assert best["draft_len"] == 3
    return {
        "source": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "calls": 32,
        "scalar_reads": len(reads),
        "heuristic_budget": state.budget_k,
        "synthetic_cost_choices": choices,
        "synthetic_optimum": best,
        "note": "Source execution with CPU stubs proves call count, not NPU synchronization duration.",
    }


def exact_selection_checks() -> dict:
    """Certify sampled thresholds, including deliberate underfilled samples."""
    rng = np.random.default_rng(41)
    checks = []
    for length in (65536, 131072):
        for kind in ("random", "ascending", "ties", "sample_misses_high_tail"):
            scores = rng.normal(size=length).astype(np.float32)
            if kind == "ascending":
                scores = np.arange(length, dtype=np.float32)
            elif kind == "ties":
                scores.fill(1)
            elif kind == "sample_misses_high_tail":
                scores.fill(0)
                scores[1::64] = 10
            # A high sample threshold intentionally stresses the recovery path.
            threshold = np.max(scores[::64])
            admitted = np.flatnonzero(scores >= threshold)
            recovery = len(admitted) < 512
            if recovery:
                admitted = np.arange(length)
            order = np.lexsort((admitted, -scores[admitted]))
            selected = admitted[order[:512]]
            reference = np.lexsort((np.arange(length), -scores))[:512]
            assert np.array_equal(selected, reference)
            checks.append(
                {
                    "length": length,
                    "kind": kind,
                    "recovery": recovery,
                    "selected_indices_equal": True,
                    "admitted_count": len(admitted),
                }
            )
    # Equal indexer scores do not imply interchangeable downstream KV values.
    tied_values = np.array([0.0, 10.0])
    # BF16 round-to-nearest-even; two distinct FP32 scores become tied.
    scores = np.array([1.001, 1.002], dtype=np.float32)
    bits = scores.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    assert rounded[0] == rounded[1] and scores[1] > scores[0]
    return {
        "certification_cases": checks,
        "tie_counterexample": {
            "equal_indexer_scores": [1, 1],
            "selected_k": 1,
            "downstream_outputs": tied_values.tolist(),
        },
        "bf16_counterexample": {
            "fp32": scores.tolist(),
            "bf16": rounded.tolist(),
            "fp32_argmax": int(scores.argmax()),
            "stable_bf16_argmax": int(rounded.argmax()),
        },
        "note": "Independent reference of certify-and-recover principle, not a port of HPC-Ops or DeepSelect.",
    }


def replay_counterexample() -> dict:
    """Two causal uniform-attention layers disprove generic exact suffix replay."""
    window = 128
    values = np.zeros(256, dtype=np.float64)
    values[127] = 1.0

    def forward(source, start):
        output = np.zeros_like(source)
        for index in range(start, len(source)):
            output[index] = source[max(start, index - window + 1) : index + 1].mean()
        return output

    complete = forward(forward(values, 0), 0)
    truncated = forward(forward(values, 128), 128)
    assert complete[-1] > 0 and truncated[-1] == 0
    return {
        "window": window,
        "layers": 2,
        "last_full": float(complete[-1]),
        "last_replayed": float(truncated[-1]),
        "note": "Structural counterexample with uniform SWA, not a DSV4.1 checkpoint accuracy measurement.",
    }


def sampling_causality_counterexample() -> dict:
    """Enumerate exact probabilities for a draft-dependent stopping rule."""
    target = {"A": 0.5, "B": 0.5}
    draft = target.copy()
    baseline = {token: 0.0 for token in target}
    selective = baseline.copy()
    for token, probability in draft.items():
        # Identical p and q accept every proposed token under strict verification.
        baseline[token] += probability
        if token == "A":
            selective[token] += probability
        else:
            for replacement, target_probability in target.items():
                selective[replacement] += probability * target_probability
    assert baseline == target and selective == {"A": 0.75, "B": 0.25}
    return {
        "target": target,
        "draft": draft,
        "always_verify_output": baseline,
        "verify_only_if_draft_is_A_otherwise_sample_target": selective,
        "note": "Enumerated toy counterexample; not a finding of bias in the local DSpark path.",
    }


def workload_bounds(previous: dict) -> dict:
    bounds = []
    for case in previous["work"]["prefill"]:
        if case["chunk"] != 8192 or case["prefix_tokens"]:
            continue
        removed = case["candidate_pair_reduction"]
        bounds.append(
            {
                "tokens": case["tokens"],
                "replay": case["replay"],
                "pair_fraction_removed": removed,
                "idealized_speedup_if_qk_is_all_time": 1 / (1 - removed),
                "scenarios": [
                    {
                        "qk_time_fraction_assumed": f,
                        "added_time_fraction_assumed": overhead,
                        "speedup": 1 / (1 - f * removed + overhead),
                    }
                    for f in (0.25, 0.5, 0.75)
                    for overhead in (0, 0.01)
                ],
            }
        )
    # At 128K, long payload = 320 MiB KV + 40.625 MiB resident index keys.
    index_bytes = int(131072 * 2.5 * 130)
    kv_bytes = int(131072 * 2.5 * 1024)
    return {
        "candidate_compaction": bounds,
        "hisparse_payload_only": {
            "context": 131072,
            "index_bytes_stay_resident": index_bytes,
            "main_kv_bytes_full": kv_bytes,
            "main_kv_bytes_four_512_row_hot_sets": 4 * 512 * 1024,
            "main_kv_union_worst_rows_8_distinct_selections": 8 * 512,
            "max_single_query_selected_bytes": 8 * 512 * 1024,
            "note": "No SWA, ring, staging, alignment, allocator or multi-token union included.",
        },
        "materialized_fp32_scores_per_source_chunk": {
            "chunk": 8192,
            "context": 131072,
            "full_c1_bytes": 8192 * 131072 * 4,
            "consumer_16384_bytes": 8192 * 16384 * 4,
        },
        "codec_break_even": "encode_time + decode_time < bytes / link_bandwidth * (1 - 1/compression_ratio)",
        "splitzip_bf16_no_escape_payload_compression_ceiling": 16 / (8 + 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = Path(__file__).resolve().parent / "results"
    previous_path = folder / "results.json"
    previous = json.loads(previous_path.read_text())
    result = {
        "base_commit": previous["commit"],
        "base_results_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "npu_executed": False,
        "model_accuracy_measured": False,
        "budget_source_execution": execute_budget_source(root),
        "selection_contracts": exact_selection_checks(),
        "replay_counterexample": replay_counterexample(),
        "bounds": workload_bounds(previous),
        "sampling_causality_counterexample": sampling_causality_counterexample(),
    }
    target = folder / "research_checks.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(target),
                "selection_cases": 8,
                "source_budget_calls": 32,
                "scalar_reads": 2,
                "npu_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
