# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeriveManualScopeDeps pass."""

from collections.abc import Iterable

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Skip the global print -> parse -> assert_structural_equal roundtrip.

    The python_printer does not surface ``Call.attrs['manual_dep_edges']`` (an
    internal post-pass attr), so the roundtrip would always fail after this
    pass. Property verification still runs.
    """

    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


class TestDeriveManualScopeDeps:
    def test_no_manual_scope_is_noop(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)
        assert ddep.same_as(ddir)

    def test_tensor_assemble_alias_forwards_task_id(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def prod(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile: pl.Tile[[1, 64], pl.FP32] = pl.load(x, [row, 0], [1, 64])
                ret: pl.Tensor[[1, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile: pl.Tile[[1, 64], pl.FP32] = pl.load(x, [row, 0], [1, 64])
                ret: pl.Tensor[[1, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.manual_scope():
                    row: pl.Scalar[pl.INDEX] = 0
                    scratch: pl.Tensor[[1, 64], pl.FP32] = pl.slice(out, [1, 64], [row, 0])
                    produced: pl.Tensor[[1, 64], pl.FP32] = self.prod(x, row, scratch)
                    carried: pl.Tensor[[64, 64], pl.FP32] = pl.assemble(out, produced, [row, 0])
                    scratch2: pl.Tensor[[1, 64], pl.FP32] = pl.slice(carried, [1, 64], [row, 0])
                    consumed: pl.Tensor[[1, 64], pl.FP32] = self.consume(x, row, scratch2, deps=[carried])
                return carried

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)
        fn = ddep.get_function("main")
        assert fn is not None

        calls: list[ir.Call] = []

        def walk(stmts: Iterable[object]) -> None:
            for stmt in stmts:
                if isinstance(stmt, ir.SeqStmts):
                    walk(stmt.stmts)
                elif isinstance(stmt, ir.RuntimeScopeStmt):
                    walk([stmt.body])
                elif isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                    calls.append(stmt.value)

        walk([fn.body])
        consume_call = next(call for call in calls if call.op.name == "consume")
        deps = consume_call.attrs.get("manual_dep_edges", [])
        assert len(deps) == 1
        assert deps[0].name_hint.endswith("__tid")

    def test_tensor_slice_alias_forwards_task_id(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                with pl.manual_scope():
                    row: pl.Scalar[pl.INDEX] = 64
                    out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(
                        data, row, 1.0, out, deps=[out]
                    )
                return out_next

        ssa = passes.convert_to_ssa()(Prog)
        optimized = passes.optimize_orch_tensors()(ssa)
        ddir = passes.derive_call_directions()(optimized)
        ddep = passes.derive_manual_scope_deps()(ddir)
        fn = ddep.get_function("main")
        assert fn is not None

        scope = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ir.RuntimeScopeStmt))
        slice_assign = next(
            stmt
            for stmt in scope.body.stmts
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "tensor.slice"
        )
        kernel_call_assign = next(
            stmt
            for stmt in scope.body.stmts
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "kernel_stripe__windowed"
        )
        slice_tid_assign = next(
            stmt
            for stmt in scope.body.stmts
            if isinstance(stmt, ir.AssignStmt)
            and stmt.var.name_hint == f"{slice_assign.var.name_hint}__tid"
        )

        deps = kernel_call_assign.value.attrs.get("manual_dep_edges", [])
        assert len(deps) == 1
        assert deps[0].same_as(slice_tid_assign.var)
        assert isinstance(slice_tid_assign.value, ir.Var)
        assert slice_tid_assign.value.name_hint.endswith("__tid")


class TestManualScopeNesting:
    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_nested_manual_scope_rejected(self):
        """The runtime forbids MANUAL inside MANUAL; reject at parse time."""

        with pytest.raises(Exception, match="manual_scope"):  # noqa: B017

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        with pl.manual_scope():
                            b = self.k1(a)
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
