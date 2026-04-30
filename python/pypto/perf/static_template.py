# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Metrics helpers for static-template performance benchmarks."""

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

_THREAD_RE = re.compile(r"Thread (\d+):")
_SCHED_START_RE = re.compile(r"sched_start=(\d+)")
_SCHED_END_RE = re.compile(r"sched_end(?:[^=]*)=(\d+)")
_ORCH_START_RE = re.compile(r"orch_start=(\d+)")
_ORCH_END_RE = re.compile(r"orch_end=(\d+)")
_ORCH_STAGE_END_RE = re.compile(r"orch_stage_end=(\d+)")


@dataclass(frozen=True)
class RuntimeTimingMetrics:
    """Parsed per-run timing metrics from PTO2 device logs."""

    rounds: int
    elapsed_avg_us: float
    elapsed_trimmed_avg_us: float
    sched_avg_us: float | None
    sched_trimmed_avg_us: float | None
    orch_avg_us: float | None
    orch_trimmed_avg_us: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return metrics as a JSON-serialisable dict."""
        return asdict(self)


@dataclass
class _RoundState:
    min_start: int = 0
    max_end: int = 0
    min_sched_start: int = 0
    max_sched_end: int = 0
    min_orch_start: int = 0
    max_orch_end: int = 0


def trimmed_mean(values: list[float], *, trim: int = 10) -> float:
    """Return the mean after dropping a fixed number of low/high outliers."""
    if not values:
        raise ValueError("trimmed_mean requires at least one value")
    ordered = sorted(values)
    if len(ordered) > 2 * trim:
        ordered = ordered[trim:-trim]
    return sum(ordered) / len(ordered)


def compile_metrics_from_profile(profile: dict[str, Any]) -> dict[str, float]:
    """Extract static-template compile metrics from CompileProfiler JSON."""
    stage_seconds = _flatten_stage_seconds(profile.get("stages", []))
    identify_us = stage_seconds.get("IdentifyStableRegions", 0.0) * 1_000_000
    lower_us = stage_seconds.get("LowerStableRegionsToManualScope", 0.0) * 1_000_000
    orchestration_us = stage_seconds.get("orchestration_codegen", 0.0) * 1_000_000
    passes_us = stage_seconds.get("passes", 0.0) * 1_000_000
    codegen_us = stage_seconds.get("codegen", 0.0) * 1_000_000
    ptoas_us = _sum_named_stage_seconds(profile.get("stages", []), "ptoas") * 1_000_000
    return {
        "passes_us": round(passes_us, 3),
        "codegen_us": round(codegen_us, 3),
        "ptoas_us": round(ptoas_us, 3),
        "identify_stable_regions_us": round(identify_us, 3),
        "lower_stable_regions_us": round(lower_us, 3),
        "orchestration_codegen_us": round(orchestration_us, 3),
        "compile_frontend_sched_us": round(identify_us + lower_us + orchestration_us, 3),
    }


def parse_device_timing_log(text: str, *, freq_mhz: int) -> RuntimeTimingMetrics:
    """Parse PTO2 device timing logs using the runtime benchmark round model."""
    if freq_mhz <= 0:
        raise ValueError(f"freq_mhz must be positive, got {freq_mhz}")

    elapsed_results: list[float] = []
    sched_results: list[float] = []
    orch_results: list[float] = []
    state = _RoundState()
    sched_seen: set[int] = set()
    orch_seen: set[int] = set()
    saw_any_timing = False

    def flush_round() -> None:
        nonlocal state
        if state.max_end > 0 and state.min_start > 0:
            elapsed_results.append((state.max_end - state.min_start) / freq_mhz)
            if state.max_sched_end > 0 and state.min_sched_start > 0:
                sched_results.append((state.max_sched_end - state.min_sched_start) / freq_mhz)
            if state.max_orch_end > 0 and state.min_orch_start > 0:
                orch_results.append((state.max_orch_end - state.min_orch_start) / freq_mhz)

    def new_round() -> None:
        nonlocal state, sched_seen, orch_seen
        flush_round()
        state = _RoundState()
        sched_seen = set()
        orch_seen = set()

    for line in text.splitlines():
        thread_match = _THREAD_RE.search(line)
        tid = int(thread_match.group(1)) if thread_match else -1

        if sched_match := _SCHED_START_RE.search(line):
            if tid in sched_seen:
                new_round()
            sched_seen.add(tid)
            saw_any_timing = True
            value = int(sched_match.group(1))
            state.min_sched_start = _min_nonzero(state.min_sched_start, value)
            state.min_start = _min_nonzero(state.min_start, value)

        if orch_match := _ORCH_START_RE.search(line):
            if tid in orch_seen:
                new_round()
            orch_seen.add(tid)
            saw_any_timing = True
            value = int(orch_match.group(1))
            state.min_orch_start = _min_nonzero(state.min_orch_start, value)
            state.min_start = _min_nonzero(state.min_start, value)

        if sched_end_match := _SCHED_END_RE.search(line):
            value = int(sched_end_match.group(1))
            state.max_sched_end = max(state.max_sched_end, value)
            state.max_end = max(state.max_end, value)

        if orch_end_match := _ORCH_END_RE.search(line):
            value = int(orch_end_match.group(1))
            state.max_orch_end = max(state.max_orch_end, value)
            state.max_end = max(state.max_end, value)

        if orch_stage_match := _ORCH_STAGE_END_RE.search(line):
            value = int(orch_stage_match.group(1))
            state.max_end = max(state.max_end, value)

    flush_round()
    if not saw_any_timing or not elapsed_results:
        raise ValueError("no PTO2 timing rounds found in device log")

    return RuntimeTimingMetrics(
        rounds=len(elapsed_results),
        elapsed_avg_us=_mean(elapsed_results),
        elapsed_trimmed_avg_us=trimmed_mean(elapsed_results),
        sched_avg_us=_mean(sched_results) if sched_results else None,
        sched_trimmed_avg_us=trimmed_mean(sched_results) if sched_results else None,
        orch_avg_us=_mean(orch_results) if orch_results else None,
        orch_trimmed_avg_us=trimmed_mean(orch_results) if orch_results else None,
    )


def compare_static_template_variants(
    *,
    static_hit: dict[str, Any],
    baseline: dict[str, Any],
    rounds: int,
) -> dict[str, float | int | None]:
    """Compare static-hit metrics against traditional AUTO-scope baseline."""
    static_compile = _nested_float(static_hit, "compile", "compile_frontend_sched_us")
    baseline_compile = _nested_float(baseline, "compile", "compile_frontend_sched_us")
    static_orch = _nested_float(static_hit, "runtime", "orch_trimmed_avg_us")
    baseline_orch = _nested_float(baseline, "runtime", "orch_trimmed_avg_us")

    compile_delta = static_compile - baseline_compile
    saved_per_round = baseline_orch - static_orch
    break_even: int | None
    if saved_per_round > 0:
        break_even = max(0, math.ceil(compile_delta / saved_per_round))
    else:
        break_even = None
    return {
        "compile_delta_us": round(compile_delta, 3),
        "device_orch_saved_us_per_round": round(saved_per_round, 3),
        "net_gain_after_rounds_us": round(rounds * saved_per_round - compile_delta, 3),
        "break_even_runs": break_even,
    }


def _flatten_stage_seconds(stages: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for stage in stages if isinstance(stages, list) else []:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        seconds = stage.get("seconds", 0.0)
        if isinstance(name, str) and isinstance(seconds, int | float):
            result[name] = result.get(name, 0.0) + float(seconds)
        child_result = _flatten_stage_seconds(stage.get("children", []))
        for child_name, child_seconds in child_result.items():
            result[child_name] = result.get(child_name, 0.0) + child_seconds
    return result


def _sum_named_stage_seconds(stages: Any, target_name: str) -> float:
    total = 0.0
    for stage in stages if isinstance(stages, list) else []:
        if not isinstance(stage, dict):
            continue
        if stage.get("name") == target_name:
            seconds = stage.get("seconds", 0.0)
            if isinstance(seconds, int | float):
                total += float(seconds)
        total += _sum_named_stage_seconds(stage.get("children", []), target_name)
    return total


def _min_nonzero(current: int, value: int) -> int:
    if current == 0 or value < current:
        return value
    return current


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _nested_float(data: dict[str, Any], section: str, key: str) -> float:
    section_value = data.get(section, {})
    if not isinstance(section_value, dict):
        raise ValueError(f"{section!r} must be a mapping")
    value = section_value.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{section}.{key} must be numeric, got {value!r}")
    return float(value)
