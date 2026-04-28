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


def test_lower_stable_regions_wraps_marked_calls_in_manual_scope():
    program = passes.derive_call_directions()(_build_paged_attention_like_program())
    program = passes.simplify()(program)
    identified = passes.identify_stable_regions()(program)

    lowered = passes.lower_stable_regions_to_manual_scope()(identified)

    manual_scopes = _collect_manual_scopes(lowered)
    assert len(manual_scopes) == 1
    assert manual_scopes[0].template_key == "paged_attention_qk_softmax_pv_update"
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
