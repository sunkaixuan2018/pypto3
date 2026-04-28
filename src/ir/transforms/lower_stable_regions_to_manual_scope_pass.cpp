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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kStableRegionTemplateKey = "stable_region_template_key";
constexpr const char* kManualTaskIndex = "manual_task_index";

using ::pypto::codegen::IsBuiltinOp;

CallPtr GetDirectCall(const StmtPtr& stmt) {
  if (auto assign = As<AssignStmt>(stmt)) {
    return As<Call>(assign->value_);
  }
  if (auto eval = As<EvalStmt>(stmt)) {
    return As<Call>(eval->expr_);
  }
  return nullptr;
}

std::optional<std::string> GetStableTemplateKey(const StmtPtr& stmt) {
  auto call = GetDirectCall(stmt);
  if (!call || IsBuiltinOp(call->op_->name_) || !call->HasAttr(kStableRegionTemplateKey)) {
    return std::nullopt;
  }
  return call->GetAttr<std::string>(kStableRegionTemplateKey);
}

int GetManualTaskIndex(const StmtPtr& stmt) {
  auto call = GetDirectCall(stmt);
  if (!call || !call->HasAttr(kManualTaskIndex)) {
    return -1;
  }
  return call->GetAttr<int>(kManualTaskIndex, -1);
}

bool ContainsManualUnsafeStmt(const std::vector<StmtPtr>& stmts, size_t start, size_t end) {
  for (size_t i = start; i <= end; ++i) {
    if (As<ForStmt>(stmts[i]) || As<WhileStmt>(stmts[i]) || As<IfStmt>(stmts[i])) {
      return true;
    }
    if (As<ScopeStmt>(stmts[i])) {
      return true;
    }
  }
  return false;
}

class StableRegionLowerer : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& seq) override {
    std::vector<StmtPtr> visited;
    bool child_changed = false;
    visited.reserve(seq->stmts_.size());
    for (const auto& stmt : seq->stmts_) {
      auto new_stmt = IRMutator::VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) {
        child_changed = true;
      }
      visited.push_back(std::move(new_stmt));
    }

    std::vector<StmtPtr> lowered;
    bool wrapped = false;
    for (size_t i = 0; i < visited.size();) {
      auto key = GetStableTemplateKey(visited[i]);
      if (!key.has_value() || GetManualTaskIndex(visited[i]) != 0) {
        lowered.push_back(visited[i]);
        ++i;
        continue;
      }

      size_t j = i;
      size_t last_marked = i;
      bool saw_marked = false;
      while (j < visited.size()) {
        auto current_key = GetStableTemplateKey(visited[j]);
        if (current_key.has_value()) {
          if (*current_key != *key) {
            break;
          }
          int task_index = GetManualTaskIndex(visited[j]);
          if (task_index == 0 && saw_marked) {
            break;
          }
          saw_marked = true;
          last_marked = j;
        }
        ++j;
      }

      if (!saw_marked || ContainsManualUnsafeStmt(visited, i, last_marked)) {
        lowered.push_back(visited[i]);
        ++i;
        continue;
      }

      std::vector<StmtPtr> body_stmts;
      body_stmts.reserve(last_marked - i + 1);
      for (size_t k = i; k <= last_marked; ++k) {
        body_stmts.push_back(visited[k]);
      }
      auto body = SeqStmts::Flatten(std::move(body_stmts), seq->span_);
      lowered.push_back(std::make_shared<ManualScopeStmt>(*key, std::nullopt, "", body, seq->span_));
      wrapped = true;
      i = last_marked + 1;
    }

    if (wrapped || child_changed) {
      return SeqStmts::Flatten(std::move(lowered), seq->span_);
    }
    return seq;
  }
};

}  // namespace

namespace pass {

Pass LowerStableRegionsToManualScope() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    auto new_functions = program->functions_;
    bool changed = false;

    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_ || func->func_type_ != FunctionType::Orchestration) {
        continue;
      }
      StableRegionLowerer lowerer;
      auto new_body = lowerer.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) {
        continue;
      }
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
      changed = true;
    }

    if (!changed) {
      return program;
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "LowerStableRegionsToManualScope",
                           kLowerStableRegionsToManualScopeProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
