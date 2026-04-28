# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
# You may not use this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS"
# BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for stable orchestration region identification and manual-scope lowering."""

import pypto.language as pl
import pytest
from pypto import passes
from pypto.pypto_core import ir


def _build_paged_attention_like_program():
    @pl.program
    class StableRegionProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_qk_matmul(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_softmax_prepare(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_pv_matmul(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_online_update(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            qk_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            qk_buf = self.kernel_qk_matmul(src, qk_buf)

            softmax_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            softmax_buf = self.kernel_softmax_prepare(qk_buf, softmax_buf)

            pv_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            pv_buf = self.kernel_pv_matmul(softmax_buf, pv_buf)

            updated = self.kernel_online_update(pv_buf, dst)
            return updated

    return StableRegionProgram


def _build_bgemm_like_program():
    @pl.program
    class BgemmTemplateProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_bgemm_stage0(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            lhs_l1 = pl.load(lhs, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            rhs_l1 = pl.load(rhs, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            lhs_l0a = pl.move(lhs_l1, target_memory=pl.MemorySpace.Left)
            rhs_l0b = pl.move(rhs_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(lhs_l0a, rhs_l0b)
            ret = pl.store(acc, [0, 0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_tile_add_stage0(
            self,
            src: pl.Tensor[[64, 64], pl.FP32],
            bias: pl.Tensor[[64, 64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src_tile = pl.load(src, [0, 0], [64, 64])
            bias_tile = pl.load(bias, [0, 0], [64, 64])
            out_tile = pl.add(src_tile, bias_tile)
            ret = pl.store(out_tile, [0, 0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_bgemm_stage1(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs: pl.Tensor[[64, 64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            lhs_l1 = pl.load(lhs, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            rhs_l1 = pl.load(rhs, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            lhs_l0a = pl.move(lhs_l1, target_memory=pl.MemorySpace.Left)
            rhs_l0b = pl.move(rhs_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(lhs_l0a, rhs_l0b)
            ret = pl.store(acc, [0, 0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_tile_add_stage1(
            self,
            src: pl.Tensor[[64, 64], pl.FP32],
            bias: pl.Tensor[[64, 64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src_tile = pl.load(src, [0, 0], [64, 64])
            bias_tile = pl.load(bias, [0, 0], [64, 64])
            out_tile = pl.add(src_tile, bias_tile)
            ret = pl.store(out_tile, [0, 0], dst)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            lhs: pl.Tensor[[64, 64], pl.FP32],
            rhs0: pl.Tensor[[64, 64], pl.FP32],
            bias0: pl.Tensor[[64, 64], pl.FP32],
            rhs1: pl.Tensor[[64, 64], pl.FP32],
            bias1: pl.Tensor[[64, 64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            gemm0_buf: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
            gemm0_buf = self.kernel_bgemm_stage0(lhs, rhs0, gemm0_buf)

            add0_buf: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
            add0_buf = self.kernel_tile_add_stage0(gemm0_buf, bias0, add0_buf)

            gemm1_buf: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
            gemm1_buf = self.kernel_bgemm_stage1(add0_buf, rhs1, gemm1_buf)

            dst = self.kernel_tile_add_stage1(gemm1_buf, bias1, dst)
            return dst

    return BgemmTemplateProgram


def _collect_user_calls(program: ir.Program) -> list[ir.Call]:
    class _Collector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            if not op.op.name.startswith("tensor."):
                self.calls.append(op)
            super().visit_call(op)

    collector = _Collector()
    collector.visit_program(program)
    return collector.calls


def _collect_manual_scopes(program: ir.Program) -> list[ir.ManualScopeStmt]:
    class _Collector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.scopes: list[ir.ManualScopeStmt] = []

        def visit_manual_scope_stmt(self, op: ir.ManualScopeStmt) -> None:
            self.scopes.append(op)
            super().visit_manual_scope_stmt(op)

    collector = _Collector()
    collector.visit_program(program)
    return collector.scopes


def _collect_return_stmts(program: ir.Program) -> list[ir.ReturnStmt]:
    class _Collector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.returns: list[ir.ReturnStmt] = []

        def visit_return_stmt(self, op: ir.ReturnStmt) -> None:
            self.returns.append(op)
            super().visit_return_stmt(op)

    collector = _Collector()
    collector.visit_program(program)
    return collector.returns


def test_identify_stable_regions_marks_paged_attention_calls():
    program = passes.derive_call_directions()(_build_paged_attention_like_program())
    program = passes.simplify()(program)

    identified = passes.identify_stable_regions()(program)

    calls = [
        call
        for call in _collect_user_calls(identified)
        if call.op.name
        in {
            "kernel_qk_matmul",
            "kernel_softmax_prepare",
            "kernel_pv_matmul",
            "kernel_online_update",
        }
    ]
    assert [call.attrs["stable_region_template_key"] for call in calls] == [
        "paged_attention_qk_softmax_pv_update",
        "paged_attention_qk_softmax_pv_update",
        "paged_attention_qk_softmax_pv_update",
        "paged_attention_qk_softmax_pv_update",
    ]
    assert [call.attrs["manual_task_index"] for call in calls] == [0, 1, 2, 3]
    assert [call.attrs["manual_dep_indices"] for call in calls] == [[], [0], [1], [2]]


def test_identify_stable_regions_marks_bgemm_template_calls():
    program = passes.derive_call_directions()(_build_bgemm_like_program())
    program = passes.simplify()(program)

    identified = passes.identify_stable_regions()(program)

    expected_names = [
        "kernel_bgemm_stage0",
        "kernel_tile_add_stage0",
        "kernel_bgemm_stage1",
        "kernel_tile_add_stage1",
    ]
    calls = [call for call in _collect_user_calls(identified) if call.op.name in set(expected_names)]

    assert [call.op.name for call in calls] == expected_names
    assert [call.attrs["stable_region_template_key"] for call in calls] == [
        "bgemm_tile_add_bgemm_tile_add",
        "bgemm_tile_add_bgemm_tile_add",
        "bgemm_tile_add_bgemm_tile_add",
        "bgemm_tile_add_bgemm_tile_add",
    ]
    assert [call.attrs["manual_task_index"] for call in calls] == [0, 1, 2, 3]
    assert [call.attrs["manual_dep_indices"] for call in calls] == [[], [0], [1], [2]]


def test_lower_stable_regions_wraps_marked_calls_in_manual_scope():
    program = passes.derive_call_directions()(_build_paged_attention_like_program())
    program = passes.simplify()(program)
    identified = passes.identify_stable_regions()(program)

    lowered = passes.lower_stable_regions_to_manual_scope()(identified)

    manual_scopes = _collect_manual_scopes(lowered)
    assert len(manual_scopes) == 1
    assert manual_scopes[0].template_key == "paged_attention_qk_softmax_pv_update"
    assert len(_collect_return_stmts(lowered)) == 1


def test_lower_stable_regions_wraps_bgemm_template_in_manual_scope():
    program = passes.derive_call_directions()(_build_bgemm_like_program())
    program = passes.simplify()(program)
    identified = passes.identify_stable_regions()(program)

    lowered = passes.lower_stable_regions_to_manual_scope()(identified)

    manual_scopes = _collect_manual_scopes(lowered)
    assert len(manual_scopes) == 1
    assert manual_scopes[0].template_key == "bgemm_tile_add_bgemm_tile_add"
    assert len(_collect_return_stmts(lowered)) == 1


def test_identify_stable_regions_rejects_open_output_boundary():
    @pl.program
    class OpenOutputProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_qk_matmul(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_softmax_prepare(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_pv_matmul(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_online_update(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_post_consumer(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            tile = pl.load(src, [0], [16])
            ret = pl.store(tile, [0], dst)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            src: pl.Tensor[[16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16], pl.FP32]],
        ) -> pl.Tensor[[16], pl.FP32]:
            qk_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            qk_buf = self.kernel_qk_matmul(src, qk_buf)
            softmax_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            softmax_buf = self.kernel_softmax_prepare(qk_buf, softmax_buf)
            pv_buf: pl.Tensor[[16], pl.FP32] = pl.create_tensor([16], dtype=pl.FP32)
            pv_buf = self.kernel_pv_matmul(softmax_buf, pv_buf)
            updated = self.kernel_online_update(pv_buf, dst)
            post = self.kernel_post_consumer(updated, dst)
            return post

    program = passes.derive_call_directions()(OpenOutputProgram)
    program = passes.simplify()(program)

    identified = passes.identify_stable_regions()(program)

    assert all("stable_region_template_key" not in call.attrs for call in _collect_user_calls(identified))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
