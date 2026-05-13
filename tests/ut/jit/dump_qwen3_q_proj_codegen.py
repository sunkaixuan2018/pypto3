# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dump Qwen3 decode orchestration and q_proj kernel codegen artifacts.

This is a small developer helper for validating that q_projection remains one
outlined kernel after the standard pass/codegen pipeline.

Usage:
    python tests/ut/jit/dump_qwen3_q_proj_codegen.py --output-dir /tmp/qwen3_q_proj_codegen
"""

import argparse
from pathlib import Path

import torch

from pypto import backend
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate

from examples.models.qwen3_jit.config import (
    BATCH,
    CACHE_ROWS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HIDDEN,
    MAX_SEQ,
)
from examples.models.qwen3_jit.qwen3_decode import qwen3_decode


def _make_args() -> list[torch.Tensor]:
    def randn(shape, dtype):
        return torch.empty(shape, dtype=dtype).normal_()

    return [
        randn([BATCH, HIDDEN], torch.bfloat16),
        randn([1, HIDDEN], torch.float32),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([1, HIDDEN], torch.float32),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([INTERMEDIATE, HIDDEN], torch.bfloat16),
        torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write dumped artifacts into")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    post_pass = qwen3_decode.compile_for_test(*_make_args())
    files = generate(post_pass, str(out_dir / "backend_generate"), skip_ptoas=True)

    q_proj_funcs = sorted(name for name in post_pass.functions if name.startswith("q_proj"))
    (out_dir / "post_pass_function_names.txt").write_text(
        "\n".join(sorted(post_pass.functions.keys())) + "\n", encoding="utf-8"
    )
    (out_dir / "q_proj_functions.txt").write_text("\n".join(q_proj_funcs) + "\n", encoding="utf-8")

    orch_code = files["orchestration/qwen3_decode.cpp"]
    q_proj_code = files["kernels/aic/q_proj.pto"]

    (out_dir / "qwen3_decode_orch.cpp").write_text(orch_code, encoding="utf-8")
    (out_dir / "q_proj.pto").write_text(q_proj_code, encoding="utf-8")

    print(f"Wrote artifacts to: {out_dir}")
    print(f"Outlined q_proj-like functions: {q_proj_funcs}")
    print("Generated files:")
    print(f"  {out_dir / 'post_pass_function_names.txt'}")
    print(f"  {out_dir / 'q_proj_functions.txt'}")
    print(f"  {out_dir / 'qwen3_decode_orch.cpp'}")
    print(f"  {out_dir / 'q_proj.pto'}")


if __name__ == "__main__":
    main()
