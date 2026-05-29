# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #1503 fan-in repro for flat-view writes and base-view scalar reads.

The four variants keep the same route-table producer and checksum consumer:

* ``plain`` proves a flat-view write can feed base-view scalar reads by itself.
* ``side`` proves a simple second producer plus two fan-in edges is not enough.
* ``chunk_amax`` is the minimal failing structure reported in #1503.
* ``scale`` keeps the fuller row-amax loop that fails the same way.
"""

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

T = 128
TOPK = 6
D = 4096
N_ROUTE = T * TOPK
QUANT_CHUNK = 32
INT8_AMAX_EPS = 1.0e-12
INT8_SCALE_MAX = 127.0


def _init_route_src() -> torch.Tensor:
    flat = torch.arange(N_ROUTE, dtype=torch.int32)
    values = (flat * 7 + flat // TOPK * 3 + 1) % 16
    return values.reshape(T, TOPK).to(torch.int32)


def _init_x_norm() -> torch.Tensor:
    torch.manual_seed(20260525)
    return torch.randn(T, D, dtype=torch.float32).to(torch.bfloat16)


def _init_side_src() -> torch.Tensor:
    flat = torch.arange(T, dtype=torch.float32)
    return (flat.reshape(T, 1) + 1.0) / 256.0


@pl.jit.inline
def write_route_outputs(route_src, indices):
    route_src_flat = pl.reshape(route_src, [N_ROUTE])
    indices_flat = pl.reshape(indices, [N_ROUTE])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="write_route_outputs"):
        for t in pl.unroll(T):
            base = t * TOPK
            for k in pl.unroll(TOPK):
                indices_flat = pl.write(
                    indices_flat,
                    [base + k],
                    pl.read(route_src_flat, [base + k]),
                )
    return pl.reshape(indices_flat, [T, TOPK])


@pl.jit.inline
def copy_side_buffer(side_src, side_buf):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_side_buffer"):
        side_buf[:, :] = side_src[:, :]
    return side_buf


@pl.jit.inline
def xnorm_first_chunk_amax(x_norm, x_norm_scale_dq_buf):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="xnorm_first_chunk_amax"):
        xn_a_f32 = pl.cast(x_norm[:, 0:QUANT_CHUNK], target_type=pl.FP32)
        xn_a_abs = pl.maximum(xn_a_f32, pl.neg(xn_a_f32))
        x_norm_scale_dq_buf[:, :] = pl.reshape(pl.row_max(xn_a_abs), [T, 1])
    return x_norm_scale_dq_buf


@pl.jit.inline
def xnorm_scale_only(x_norm, x_norm_scale_dq_buf):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_norm_q"):
        xn_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, D, QUANT_CHUNK):
            xn_a_f32 = pl.cast(x_norm[:, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
            xn_a_abs = pl.maximum(xn_a_f32, pl.neg(xn_a_f32))
            xn_a_max = pl.reshape(pl.row_max(xn_a_abs), [1, T])
            xn_amax = pl.maximum(xn_amax, xn_a_max)
        xn_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), xn_amax)
        x_norm_scale_dq_buf[:, 0:1] = pl.reshape(pl.recip(xn_sq_row), [T, 1])
    return x_norm_scale_dq_buf


@pl.jit.inline
def route_stats_plain(indices, route_stats):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_stats_plain"):
        sum_e = pl.cast(0, pl.INT32)
        sum_pos_e = pl.cast(0, pl.INT32)
        sum_e2 = pl.cast(0, pl.INT32)
        n_seen = pl.cast(0, pl.INT32)
        for t in pl.range(T):
            for k in pl.unroll(TOPK):
                e = pl.read(indices, [t, k])
                pos = pl.cast(t * TOPK + k + 1, pl.INT32)
                sum_e = pl.cast(sum_e + e, pl.INT32)
                sum_pos_e = pl.cast(sum_pos_e + pos * e, pl.INT32)
                sum_e2 = pl.cast(sum_e2 + e * e, pl.INT32)
                n_seen = pl.cast(n_seen + 1, pl.INT32)
        pl.write(route_stats, [0], sum_e)
        pl.write(route_stats, [1], sum_pos_e)
        pl.write(route_stats, [2], sum_e2)
        pl.write(route_stats, [3], n_seen)
    return route_stats


@pl.jit.inline
def route_stats_xnorm_scale(indices, x_norm_scale_dq_buf, route_stats):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_stats_xnorm_scale"):
        sum_e = pl.cast(0, pl.INT32)
        sum_pos_e = pl.cast(0, pl.INT32)
        sum_e2 = pl.cast(0, pl.INT32)
        n_seen = pl.cast(0, pl.INT32)
        for t in pl.range(T):
            scale_guard = pl.cast(pl.read(x_norm_scale_dq_buf, [t, 0]) * 0.0, pl.INT32)
            for k in pl.unroll(TOPK):
                e = pl.cast(pl.read(indices, [t, k]) + scale_guard, pl.INT32)
                pos = pl.cast(t * TOPK + k + 1, pl.INT32)
                sum_e = pl.cast(sum_e + e, pl.INT32)
                sum_pos_e = pl.cast(sum_pos_e + pos * e, pl.INT32)
                sum_e2 = pl.cast(sum_e2 + e * e, pl.INT32)
                n_seen = pl.cast(n_seen + 1, pl.INT32)
        pl.write(route_stats, [0], sum_e)
        pl.write(route_stats, [1], sum_pos_e)
        pl.write(route_stats, [2], sum_e2)
        pl.write(route_stats, [3], n_seen)
    return route_stats


@pl.program
class RouteXnormFaninPlainProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        route_src: pl.Tensor[[T, TOPK], pl.INT32],
        route_stats: pl.Out[pl.Tensor[[4], pl.INT32]],
        indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    ) -> tuple[pl.Tensor[[4], pl.INT32], pl.Tensor[[T, TOPK], pl.INT32]]:
        indices = write_route_outputs(route_src, indices)
        route_stats = route_stats_plain(indices, route_stats)
        return route_stats, indices


@pl.program
class RouteXnormFaninSideProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        route_src: pl.Tensor[[T, TOPK], pl.INT32],
        side_src: pl.Tensor[[T, 1], pl.FP32],
        route_stats: pl.Out[pl.Tensor[[4], pl.INT32]],
        indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    ) -> tuple[pl.Tensor[[4], pl.INT32], pl.Tensor[[T, TOPK], pl.INT32]]:
        side_buf = pl.create_tensor([T, 1], dtype=pl.FP32)
        indices = write_route_outputs(route_src, indices)
        side_buf = copy_side_buffer(side_src, side_buf)
        route_stats = route_stats_xnorm_scale(indices, side_buf, route_stats)
        return route_stats, indices


@pl.program
class RouteXnormFaninChunkAmaxProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        route_src: pl.Tensor[[T, TOPK], pl.INT32],
        x_norm: pl.Tensor[[T, D], pl.BF16],
        route_stats: pl.Out[pl.Tensor[[4], pl.INT32]],
        indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    ) -> tuple[pl.Tensor[[4], pl.INT32], pl.Tensor[[T, TOPK], pl.INT32]]:
        x_norm_scale_dq_buf = pl.create_tensor([T, 1], dtype=pl.FP32)
        indices = write_route_outputs(route_src, indices)
        x_norm_scale_dq_buf = xnorm_first_chunk_amax(x_norm, x_norm_scale_dq_buf)
        route_stats = route_stats_xnorm_scale(indices, x_norm_scale_dq_buf, route_stats)
        return route_stats, indices


@pl.program
class RouteXnormFaninScaleProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        route_src: pl.Tensor[[T, TOPK], pl.INT32],
        x_norm: pl.Tensor[[T, D], pl.BF16],
        route_stats: pl.Out[pl.Tensor[[4], pl.INT32]],
        indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    ) -> tuple[pl.Tensor[[4], pl.INT32], pl.Tensor[[T, TOPK], pl.INT32]]:
        x_norm_scale_dq_buf = pl.create_tensor([T, 1], dtype=pl.FP32)
        indices = write_route_outputs(route_src, indices)
        x_norm_scale_dq_buf = xnorm_scale_only(x_norm, x_norm_scale_dq_buf)
        route_stats = route_stats_xnorm_scale(indices, x_norm_scale_dq_buf, route_stats)
        return route_stats, indices


def _fill_route_golden(tensors: dict[str, torch.Tensor]) -> None:
    route_src = tensors["route_src"].to(torch.int32)
    tensors["indices"][:] = route_src
    flat = route_src.reshape(-1)
    pos = torch.arange(1, flat.numel() + 1, dtype=torch.int32)
    tensors["route_stats"][:] = torch.tensor(
        [
            int(flat.sum().item()),
            int((flat * pos).sum().item()),
            int((flat * flat).sum().item()),
            int(flat.numel()),
        ],
        dtype=torch.int32,
    )


class _RouteXnormFaninBase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _fill_route_golden(tensors)


class RouteXnormFaninPlainTestCase(_RouteXnormFaninBase):
    def get_name(self) -> str:
        return "route_xnorm_fanin_plain_1503"

    def get_program(self):
        return RouteXnormFaninPlainProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("route_src", [T, TOPK], DataType.INT32, init_value=_init_route_src),
            TensorSpec("route_stats", [4], DataType.INT32, is_output=True),
            TensorSpec("indices", [T, TOPK], DataType.INT32, is_output=True),
        ]


class RouteXnormFaninSideTestCase(_RouteXnormFaninBase):
    def get_name(self) -> str:
        return "route_xnorm_fanin_side_1503"

    def get_program(self):
        return RouteXnormFaninSideProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("route_src", [T, TOPK], DataType.INT32, init_value=_init_route_src),
            TensorSpec("side_src", [T, 1], DataType.FP32, init_value=_init_side_src),
            TensorSpec("route_stats", [4], DataType.INT32, is_output=True),
            TensorSpec("indices", [T, TOPK], DataType.INT32, is_output=True),
        ]


class RouteXnormFaninChunkAmaxTestCase(_RouteXnormFaninBase):
    def get_name(self) -> str:
        return "route_xnorm_fanin_chunk_amax_1503"

    def get_program(self):
        return RouteXnormFaninChunkAmaxProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("route_src", [T, TOPK], DataType.INT32, init_value=_init_route_src),
            TensorSpec("x_norm", [T, D], DataType.BF16, init_value=_init_x_norm),
            TensorSpec("route_stats", [4], DataType.INT32, is_output=True),
            TensorSpec("indices", [T, TOPK], DataType.INT32, is_output=True),
        ]


class RouteXnormFaninScaleTestCase(RouteXnormFaninChunkAmaxTestCase):
    def get_name(self) -> str:
        return "route_xnorm_fanin_scale_1503"

    def get_program(self):
        return RouteXnormFaninScaleProgram


class TestRouteXnormFanin:
    @pytest.mark.platforms("a2a3")
    def test_plain(self, test_runner):
        result = test_runner.run(RouteXnormFaninPlainTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_side(self, test_runner):
        result = test_runner.run(RouteXnormFaninSideTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_chunk_amax(self, test_runner):
        result = test_runner.run(RouteXnormFaninChunkAmaxTestCase())
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_scale(self, test_runner):
        result = test_runner.run(RouteXnormFaninScaleTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
