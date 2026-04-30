# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for static-template performance metric helpers."""

import pytest

from pypto.perf.static_template import (
    compare_static_template_variants,
    compile_metrics_from_profile,
    parse_device_timing_log,
    trimmed_mean,
)


def test_trimmed_mean_drops_fixed_tails_when_enough_rounds():
    values = [1000.0] + [10.0] * 30 + [2000.0]
    assert trimmed_mean(values, trim=10) == 10.0


def test_compile_metrics_extracts_static_template_stages():
    profile = {
        "stages": [
            {
                "name": "passes",
                "seconds": 0.02,
                "children": [
                    {"name": "IdentifyStableRegions", "seconds": 0.003, "children": []},
                    {"name": "LowerStableRegionsToManualScope", "seconds": 0.002, "children": []},
                ],
            },
            {
                "name": "codegen",
                "seconds": 0.03,
                "children": [
                    {"name": "orchestration_codegen", "seconds": 0.004, "children": []},
                ],
            },
        ],
    }
    metrics = compile_metrics_from_profile(profile)
    assert metrics["identify_stable_regions_us"] == 3000.0
    assert metrics["lower_stable_regions_us"] == 2000.0
    assert metrics["orchestration_codegen_us"] == 4000.0
    assert metrics["compile_frontend_sched_us"] == 9000.0


def test_parse_device_timing_log_uses_repeated_start_as_round_boundary():
    log = "\n".join(
        [
            "Thread 2: orch_start=1000 orch_end=2000 orch_cost=20.000us",
            "Thread 0: sched_start=900 sched_end=2100 sched_cost=24.000us",
            "Thread 2: orch_start=3000 orch_end=3900 orch_cost=18.000us",
            "Thread 0: sched_start=2900 sched_end=4000 sched_cost=22.000us",
        ]
    )
    metrics = parse_device_timing_log(log, freq_mhz=50)
    assert metrics.rounds == 2
    assert metrics.elapsed_avg_us == 23.0
    assert metrics.orch_avg_us == 19.0
    assert metrics.sched_avg_us == 23.0


def test_compare_static_template_variants_computes_break_even():
    result = compare_static_template_variants(
        static_hit={
            "compile": {"compile_frontend_sched_us": 30.0},
            "runtime": {"orch_trimmed_avg_us": 80.0},
        },
        baseline={
            "compile": {"compile_frontend_sched_us": 10.0},
            "runtime": {"orch_trimmed_avg_us": 100.0},
        },
        rounds=100,
    )
    assert result["compile_delta_us"] == 20.0
    assert result["device_orch_saved_us_per_round"] == 20.0
    assert result["break_even_runs"] == 1
    assert result["net_gain_after_rounds_us"] == 1980.0


def test_trimmed_mean_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one value"):
        trimmed_mean([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
