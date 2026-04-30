# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Benchmark static-template manual-scope hit against the AUTO-scope baseline."""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_ST_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _ST_DIR.parent.parent
for _path in (_PROJECT_ROOT, _ST_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from harness.core.harness import platform_to_backend  # noqa: E402
from pypto import ir  # noqa: E402
from pypto.perf.static_template import (  # noqa: E402
    compare_static_template_variants,
    compile_metrics_from_profile,
    parse_device_timing_log,
)
from pypto.runtime.runner import RunConfig  # noqa: E402
from test_manual_scope_template import (  # noqa: E402
    BgemmManualScopeProgram,
    make_bgemm_manual_scope_inputs,
)

_VARIANTS = {
    "static_hit": ir.OptimizationStrategy.Default,
    "traditional_baseline": ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates,
}
_FREQ_MHZ = {"a2a3": 50, "a5": 1000}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--rounds", default=100, type=int)
    parser.add_argument("--output", default="build_output/static_template_perf")
    parser.add_argument("--profiling-detail", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    freq_mhz = _FREQ_MHZ[args.platform]

    variant_results: dict[str, dict[str, Any]] = {}
    with _profiling_env(args.profiling_detail):
        for variant, strategy in _VARIANTS.items():
            variant_results[variant] = _run_variant(
                variant=variant,
                strategy=strategy,
                root=root,
                platform=args.platform,
                device=args.device,
                rounds=args.rounds,
                freq_mhz=freq_mhz,
            )

    comparison = compare_static_template_variants(
        static_hit=variant_results["static_hit"],
        baseline=variant_results["traditional_baseline"],
        rounds=args.rounds,
    )
    summary = {
        "platform": args.platform,
        "device": args.device,
        "rounds": args.rounds,
        "variants": variant_results,
        "comparison": comparison,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "summary.md").write_text(_format_summary(summary), encoding="utf-8")
    print(f"summary written to {root / 'summary.md'}")
    return 0


def _run_variant(
    *,
    variant: str,
    strategy: ir.OptimizationStrategy,
    root: Path,
    platform: str,
    device: int,
    rounds: int,
    freq_mhz: int,
) -> dict[str, Any]:
    variant_dir = root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    compiled = ir.compile(
        BgemmManualScopeProgram,
        output_dir=str(variant_dir),
        strategy=strategy,
        backend_type=platform_to_backend(platform),
        platform=platform,
        profiling=True,
    )
    orchestration_cpp = _read_orchestration_cpp(Path(compiled.output_dir))
    manual_scope = "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in orchestration_cpp

    profile_path = Path(compiled.output_dir) / "report" / "pipeline_profile.json"
    compile_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    compile_metrics = compile_metrics_from_profile(compile_profile)

    log_dir = _device_log_dir(device)
    before_logs = _snapshot_logs(log_dir)
    config = RunConfig(platform=platform, device_id=device)
    for _ in range(rounds):
        tensors = make_bgemm_manual_scope_inputs()
        compiled(*tensors, config=config)

    device_log_text = _collect_new_log_text(log_dir, before_logs)
    runtime_metrics = parse_device_timing_log(device_log_text, freq_mhz=freq_mhz)
    (variant_dir / "device_timing.log").write_text(device_log_text, encoding="utf-8", errors="ignore")
    return {
        "manual_scope": manual_scope,
        "compile": compile_metrics,
        "runtime": runtime_metrics.to_dict(),
        "output_dir": str(Path(compiled.output_dir)),
    }


def _read_orchestration_cpp(output_dir: Path) -> str:
    cpp_files = sorted((output_dir / "orchestration").glob("*.cpp"))
    if not cpp_files:
        raise FileNotFoundError(f"No orchestration C++ files found under {output_dir / 'orchestration'}")
    return "\n".join(path.read_text(encoding="utf-8") for path in cpp_files)


def _device_log_dir(device: int) -> Path:
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        candidate = Path(ascend_work_path).expanduser() / "log" / "debug" / f"device-{device}"
        if candidate.exists():
            return candidate
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device}"


def _snapshot_logs(log_dir: Path) -> set[Path]:
    if not log_dir.exists():
        return set()
    return set(log_dir.glob("*.log"))


def _collect_new_log_text(log_dir: Path, before_logs: set[Path], timeout_s: float = 15.0) -> str:
    deadline = time.monotonic() + timeout_s
    new_logs: set[Path] = set()
    while time.monotonic() < deadline:
        if log_dir.exists():
            new_logs = set(log_dir.glob("*.log")) - before_logs
            if new_logs:
                break
        time.sleep(0.5)
    if not new_logs and log_dir.exists():
        newest = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        new_logs = set(newest[:1])
    if not new_logs:
        raise FileNotFoundError(f"No device log found under {log_dir}")
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in sorted(new_logs))


@contextmanager
def _profiling_env(enable_detail: bool):
    updates = {"PTO2_PROFILING": "1"}
    if enable_detail:
        updates.update(
            {
                "PTO2_ORCH_PROFILING": "1",
                "PTO2_TENSORMAP_PROFILING": "1",
            }
        )
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Static Template Performance Summary",
        "",
        f"- Platform: `{summary['platform']}`",
        f"- Device: `{summary['device']}`",
        f"- Rounds: `{summary['rounds']}`",
        "",
        "| Variant | Manual scope | Compile frontend sched (us) | Orch trimmed avg (us) | Sched trimmed avg (us) | Elapsed trimmed avg (us) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in summary["variants"].items():
        compile_data = data["compile"]
        runtime_data = data["runtime"]
        lines.append(
            "| "
            f"{name} | "
            f"{'yes' if data['manual_scope'] else 'no'} | "
            f"{compile_data['compile_frontend_sched_us']:.3f} | "
            f"{_fmt_optional(runtime_data['orch_trimmed_avg_us'])} | "
            f"{_fmt_optional(runtime_data['sched_trimmed_avg_us'])} | "
            f"{runtime_data['elapsed_trimmed_avg_us']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Value |",
            "| --- | ---: |",
        ]
    )
    comparison = summary["comparison"]
    lines.extend(
        [
            f"| Compile delta (us) | {comparison['compile_delta_us']:.3f} |",
            f"| Device orch saved per round (us) | {comparison['device_orch_saved_us_per_round']:.3f} |",
            f"| Net gain after measured rounds (us) | {comparison['net_gain_after_rounds_us']:.3f} |",
            f"| Break-even runs | {_fmt_optional(comparison['break_even_runs'])} |",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
