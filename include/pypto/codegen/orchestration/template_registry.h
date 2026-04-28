/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_CODEGEN_ORCHESTRATION_TEMPLATE_REGISTRY_H_
#define PYPTO_CODEGEN_ORCHESTRATION_TEMPLATE_REGISTRY_H_

#include <string>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto {
namespace codegen {
namespace orchestration {

struct StableRegionTemplate {
  std::string template_key;
  std::vector<std::string> kernel_name_tokens;
  std::vector<std::vector<ir::ArgDirection>> arg_direction_patterns;
};

const std::vector<StableRegionTemplate>& GetStableRegionTemplates();

bool KernelNameMatchesToken(const std::string& kernel_name, const std::string& token);

}  // namespace orchestration
}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_TEMPLATE_REGISTRY_H_
