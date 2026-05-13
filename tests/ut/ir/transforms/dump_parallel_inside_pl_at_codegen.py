# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dump a minimal 'parallel inside pl.at' case after outline/codegen.

Usage:
    python tests/ut/ir/transforms/dump_parallel_inside_pl_at_codegen.py --output-dir /tmp/parallel_inside_pl_at
"""

import argparse
from pathlib import Path

import pypto.language as pl
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate
from pypto.ir.printer import python_print


@pl.program
class ParallelInsidePlAtProgram:
    @pl.function
    def main(
        self,
        normed_tile: pl.Tensor[[16, 512], pl.BF16],
        wq: pl.Tensor[[512, 256], pl.BF16],
        q_proj: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="q_proj"):
            for ob in pl.parallel(0, 4, 1, chunk=4, chunk_policy="leading_full"):
                q0: pl.Scalar[pl.INDEX] = ob * 64
                tile_a: pl.Tensor[[16, 128], pl.BF16] = pl.slice(normed_tile, [16, 128], [0, 0])
                tile_b: pl.Tensor[[128, 64], pl.BF16] = pl.slice(wq, [128, 64], [0, q0])
                q_acc: pl.Tensor[[16, 64], pl.FP32] = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                for kb in pl.range(1, 4):
                    k0: pl.Scalar[pl.INDEX] = kb * 128
                    tile_a_i: pl.Tensor[[16, 128], pl.BF16] = pl.slice(normed_tile, [16, 128], [0, k0])
                    tile_b_i: pl.Tensor[[128, 64], pl.BF16] = pl.slice(wq, [128, 64], [k0, q0])
                    q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                q_proj = pl.assemble(q_proj, q_acc, [0, q0])
        return q_proj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    program = passes.unroll_loops()(ParallelInsidePlAtProgram)
    program = passes.convert_to_ssa()(program)
    program = passes.flatten_call_expr()(program)
    program = passes.split_chunked_loops()(program)
    (out_dir / "after_split.py").write_text(python_print(program), encoding="utf-8")

    program = passes.interchange_chunk_loops()(program)
    program = passes.outline_incore_scopes()(program)
    (out_dir / "after_outline.py").write_text(python_print(program), encoding="utf-8")

    q_proj_func = next(func for func in program.functions.values() if func.name == "q_proj")
    (out_dir / "outlined_q_proj.py").write_text(python_print(q_proj_func), encoding="utf-8")

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    files = generate(program, str(out_dir / "backend_generate"), skip_ptoas=True)

    for rel_path, content in files.items():
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Wrote artifacts to: {out_dir}")
    print(f"  {out_dir / 'after_split.py'}")
    print(f"  {out_dir / 'after_outline.py'}")
    print(f"  {out_dir / 'outlined_q_proj.py'}")
    print(f"  {out_dir / 'orchestration' / 'main.cpp'}")
    print(f"  {out_dir / 'kernels' / 'aic' / 'q_proj.pto'}")


if __name__ == "__main__":
    main()
