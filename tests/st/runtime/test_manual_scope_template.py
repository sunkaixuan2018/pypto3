# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end runtime coverage for stable-region manual-scope templates."""

from datetime import datetime
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from harness.core.harness import platform_to_backend
from pypto import ir

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output" / "test_manual_scope_template"
_SIZE = 64


@pl.program
class BgemmManualScopeProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_bgemm_stage0(
        self,
        lhs: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        rhs: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        dst: pl.Out[pl.Tensor[[_SIZE, _SIZE], pl.FP32]],
    ) -> pl.Tensor[[_SIZE, _SIZE], pl.FP32]:
        lhs_l1 = pl.load(lhs, [0, 0], [_SIZE, _SIZE], target_memory=pl.MemorySpace.Mat)
        rhs_l1 = pl.load(rhs, [0, 0], [_SIZE, _SIZE], target_memory=pl.MemorySpace.Mat)
        lhs_l0a = pl.move(lhs_l1, target_memory=pl.MemorySpace.Left)
        rhs_l0b = pl.move(rhs_l1, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(lhs_l0a, rhs_l0b)
        ret = pl.store(acc, [0, 0], dst)
        return ret

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_tile_add_stage0(
        self,
        src: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        bias: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        dst: pl.Out[pl.Tensor[[_SIZE, _SIZE], pl.FP32]],
    ) -> pl.Tensor[[_SIZE, _SIZE], pl.FP32]:
        src_tile = pl.load(src, [0, 0], [_SIZE, _SIZE])
        bias_tile = pl.load(bias, [0, 0], [_SIZE, _SIZE])
        out_tile = pl.add(src_tile, bias_tile)
        ret = pl.store(out_tile, [0, 0], dst)
        return ret

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_bgemm_stage1(
        self,
        lhs: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        rhs: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        dst: pl.Out[pl.Tensor[[_SIZE, _SIZE], pl.FP32]],
    ) -> pl.Tensor[[_SIZE, _SIZE], pl.FP32]:
        lhs_l1 = pl.load(lhs, [0, 0], [_SIZE, _SIZE], target_memory=pl.MemorySpace.Mat)
        rhs_l1 = pl.load(rhs, [0, 0], [_SIZE, _SIZE], target_memory=pl.MemorySpace.Mat)
        lhs_l0a = pl.move(lhs_l1, target_memory=pl.MemorySpace.Left)
        rhs_l0b = pl.move(rhs_l1, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(lhs_l0a, rhs_l0b)
        ret = pl.store(acc, [0, 0], dst)
        return ret

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_tile_add_stage1(
        self,
        src: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        bias: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        dst: pl.Out[pl.Tensor[[_SIZE, _SIZE], pl.FP32]],
    ) -> pl.Tensor[[_SIZE, _SIZE], pl.FP32]:
        src_tile = pl.load(src, [0, 0], [_SIZE, _SIZE])
        bias_tile = pl.load(bias, [0, 0], [_SIZE, _SIZE])
        out_tile = pl.add(src_tile, bias_tile)
        ret = pl.store(out_tile, [0, 0], dst)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        lhs: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        rhs0: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        bias0: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        rhs1: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        bias1: pl.Tensor[[_SIZE, _SIZE], pl.FP32],
        dst: pl.Out[pl.Tensor[[_SIZE, _SIZE], pl.FP32]],
    ) -> pl.Tensor[[_SIZE, _SIZE], pl.FP32]:
        gemm0_buf: pl.Tensor[[_SIZE, _SIZE], pl.FP32] = pl.create_tensor([_SIZE, _SIZE], dtype=pl.FP32)
        gemm0_buf = self.kernel_bgemm_stage0(lhs, rhs0, gemm0_buf)

        add0_buf: pl.Tensor[[_SIZE, _SIZE], pl.FP32] = pl.create_tensor([_SIZE, _SIZE], dtype=pl.FP32)
        add0_buf = self.kernel_tile_add_stage0(gemm0_buf, bias0, add0_buf)

        gemm1_buf: pl.Tensor[[_SIZE, _SIZE], pl.FP32] = pl.create_tensor([_SIZE, _SIZE], dtype=pl.FP32)
        gemm1_buf = self.kernel_bgemm_stage1(add0_buf, rhs1, gemm1_buf)

        dst = self.kernel_tile_add_stage1(gemm1_buf, bias1, dst)
        return dst


@pytest.fixture(scope="session")
def output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = _BUILD_OUTPUT_DIR / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_orchestration_cpp(output_dir: Path) -> str:
    cpp_files = sorted((output_dir / "orchestration").glob("*.cpp"))
    assert cpp_files, f"No orchestration C++ files found under {output_dir / 'orchestration'}"
    return "\n".join(path.read_text() for path in cpp_files)


def test_bgemm_manual_scope_template_executes_end_to_end(output_root, test_config):
    compiled = ir.compile(
        BgemmManualScopeProgram,
        output_dir=str(output_root / "bgemm_manual_scope"),
        backend_type=platform_to_backend(test_config.platform),
        platform=test_config.platform,
    )

    orchestration_cpp = _read_orchestration_cpp(compiled.output_dir)
    assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in orchestration_cpp
    assert "auto task_result_0 = pto2_rt_submit_aic_task" in orchestration_cpp
    assert "auto task_result_1 = pto2_rt_submit_aiv_task" in orchestration_cpp
    assert "auto task_result_2 = pto2_rt_submit_aic_task" in orchestration_cpp
    assert "auto task_result_3 = pto2_rt_submit_aiv_task" in orchestration_cpp
    assert "params_t1.add_dep(task_result_0.task_id())" in orchestration_cpp
    assert "params_t2.add_dep(task_result_1.task_id())" in orchestration_cpp
    assert "params_t3.add_dep(task_result_2.task_id())" in orchestration_cpp

    lhs = torch.eye(_SIZE, dtype=torch.float32)
    rhs0 = torch.full((_SIZE, _SIZE), 1.25, dtype=torch.float32)
    bias0 = torch.full((_SIZE, _SIZE), 2.0, dtype=torch.float32)
    rhs1 = torch.eye(_SIZE, dtype=torch.float32)
    bias1 = torch.full((_SIZE, _SIZE), 3.0, dtype=torch.float32)
    dst = torch.zeros((_SIZE, _SIZE), dtype=torch.float32)

    compiled(lhs, rhs0, bias0, rhs1, bias1, dst, config=test_config)

    expected = torch.matmul(torch.matmul(lhs, rhs0) + bias0, rhs1) + bias1
    assert torch.allclose(dst, expected, rtol=1e-5, atol=1e-5), (
        f"Manual-scope template execution failed: max diff = {(dst - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
