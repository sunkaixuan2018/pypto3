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

#include "pypto/codegen/orchestration/template_registry.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace pypto {
namespace codegen {
namespace orchestration {

namespace {

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

}  // namespace

const std::vector<StableRegionTemplate>& GetStableRegionTemplates() {
  using ir::ArgDirection;
  static const std::vector<StableRegionTemplate> templates = {
      {"paged_attention_qk_softmax_pv_update", {"qk", "softmax", "pv", "update"}, {}},
      {"bgemm_tile_add_bgemm_tile_add",
       {"bgemm", "tile_add", "bgemm", "tile_add"},
       {
           {ArgDirection::Input, ArgDirection::Input, ArgDirection::OutputExisting},
           {ArgDirection::Input, ArgDirection::Input, ArgDirection::OutputExisting},
           {ArgDirection::Input, ArgDirection::Input, ArgDirection::OutputExisting},
           {ArgDirection::Input, ArgDirection::Input, ArgDirection::OutputExisting},
       }},
  };
  return templates;
}

bool KernelNameMatchesToken(const std::string& kernel_name, const std::string& token) {
  return LowerAscii(kernel_name).find(LowerAscii(token)) != std::string::npos;
}

}  // namespace orchestration
}  // namespace codegen
}  // namespace pypto
