# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression guards binding ``pto_rebuild`` to ``pto_backend`` invariants.

The rebuild path in ``pypto.runtime.debug.pto_rebuild`` deliberately holds
local copies of three pto_backend internals (sentinel literals, the body
preprocess pipeline, and the base ptoas flag list) so the debug entry point
does not pull in the full codegen stack. These tests catch silent drift if
either side evolves.
"""

from __future__ import annotations

import inspect

import pytest
from pypto.backend import pto_backend
from pypto.runtime.debug import pto_rebuild


def _wrapper_src() -> str:
    return inspect.getsource(pto_backend._generate_kernel_wrapper)


def test_begin_sentinel_appears_in_pto_backend() -> None:
    assert pto_rebuild.PTOAS_BODY_BEGIN in _wrapper_src(), (
        f"BEGIN sentinel {pto_rebuild.PTOAS_BODY_BEGIN!r} no longer found in "
        "pto_backend._generate_kernel_wrapper — splice will fail to locate the body."
    )


def test_end_sentinel_appears_in_pto_backend() -> None:
    assert pto_rebuild.PTOAS_BODY_END in _wrapper_src(), (
        f"END sentinel {pto_rebuild.PTOAS_BODY_END!r} no longer found in "
        "pto_backend._generate_kernel_wrapper — splice will fail to locate the body."
    )


_PTOAS_SAMPLE = """#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
using namespace pto;

__global__ AICORE void main_kernel(__gm__ int64_t* args) {
    AICORE void helper();
    helper();
}

AICORE void helper() {}
"""


def test_preprocess_matches_pto_backend() -> None:
    """``pto_rebuild._preprocess_ptoas_body`` must produce byte-identical
    output to ``pto_backend._preprocess_ptoas_output`` on the same input —
    otherwise a spliced kernel cpp will diverge from a fresh ``ir.compile()``
    output and miscompile silently.
    """
    expected = pto_backend._preprocess_ptoas_output(_PTOAS_SAMPLE)
    actual = pto_rebuild._preprocess_ptoas_body(_PTOAS_SAMPLE)
    assert actual == expected, (
        "pto_rebuild._preprocess_ptoas_body drifted from pto_backend._preprocess_ptoas_output."
    )


def test_preprocess_inserts_barrier_after_vector_ops() -> None:
    ptoas = """AICORE void build_w(__gm__ int64_t* args) {
    TROWEXPANDMUL(v38, v22, v37);
    Tile<TileType::Vec, uint8_t, 64, 32> v40;
    TASSIGN(v40, v41);
    TCMP(v40, v28, v38, v3);
    TSEL(v48, v47, v45, v46, v44);
    TCVT(v50, v48, RoundMode::CAST_ROUND);
    TSTORE(gm, v50);
}
"""

    actual = pto_backend._preprocess_ptoas_output(ptoas)
    lines = actual.splitlines()
    assert lines[lines.index("    TROWEXPANDMUL(v38, v22, v37);") + 1] == "    pipe_barrier(PIPE_V);"
    assert lines[lines.index("    TCMP(v40, v28, v38, v3);") + 1] == "    pipe_barrier(PIPE_V);"
    assert lines[lines.index("    TSEL(v48, v47, v45, v46, v44);") + 1] == "    pipe_barrier(PIPE_V);"
    assert lines[lines.index("    TCVT(v50, v48, RoundMode::CAST_ROUND);") + 1] == "    pipe_barrier(PIPE_V);"


def test_preprocess_does_not_duplicate_existing_vector_barrier() -> None:
    ptoas = """AICORE void build_w(__gm__ int64_t* args) {
    TSEL(v48, v47, v45, v46, v44);
    pipe_barrier(PIPE_V);
}
"""

    actual = pto_backend._preprocess_ptoas_output(ptoas)
    assert actual.count("pipe_barrier(PIPE_V);") == 1


def test_base_ptoas_flags_subset_of_backend_flags() -> None:
    """The rebuild path uses base flags only (no backend-handler extras).
    Each base flag must still appear in ``_get_ptoas_flags``'s source so the
    rebuilt cpp is assembled with the same baseline as a fresh compile.
    Source-level check avoids needing a configured backend in this test.
    """
    src = inspect.getsource(pto_backend._get_ptoas_flags)
    missing = [f for f in pto_rebuild._ptoas_flags() if repr(f) not in src and f not in src]
    assert not missing, (
        f"pto_rebuild base flags {missing!r} no longer found in pto_backend._get_ptoas_flags source."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
