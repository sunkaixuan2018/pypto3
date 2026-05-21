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

#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

enum class AccessKind { Read, Write, ReadWrite };

enum class RegionKind { Unknown, Full, Box };

struct AccessRegion {
  RegionKind kind = RegionKind::Unknown;
  std::vector<int64_t> offsets;
  std::vector<int64_t> shape;
};

struct StorageLocation {
  const Var* root = nullptr;
  AccessRegion region;
};

struct StorageAccess {
  StorageLocation location;
  AccessKind kind = AccessKind::Read;
  VarPtr task_id_var;
};

bool IsTensorType(const TypePtr& type) { return As<TensorType>(type) != nullptr; }

AccessRegion UnknownRegion() { return AccessRegion{RegionKind::Unknown, {}, {}}; }

AccessRegion FullRegion() { return AccessRegion{RegionKind::Full, {}, {}}; }

std::optional<std::vector<int64_t>> ConstIntTupleValues(const ExprPtr& expr) {
  auto tuple = As<MakeTuple>(expr);
  if (!tuple) return std::nullopt;

  std::vector<int64_t> values;
  values.reserve(tuple->elements_.size());
  for (const auto& element : tuple->elements_) {
    auto value = As<ConstInt>(element);
    if (!value) return std::nullopt;
    values.push_back(value->value_);
  }
  return values;
}

std::optional<int64_t> AddInt64(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return std::nullopt;
  }
  return lhs + rhs;
}

AccessRegion SliceRegion(const AccessRegion& parent, const ExprPtr& shape_expr, const ExprPtr& offset_expr) {
  auto shape = ConstIntTupleValues(shape_expr);
  auto offsets = ConstIntTupleValues(offset_expr);
  if (!shape.has_value() || !offsets.has_value() || shape->size() != offsets->size()) {
    return UnknownRegion();
  }

  if (parent.kind == RegionKind::Unknown) {
    return UnknownRegion();
  }

  std::vector<int64_t> absolute_offsets;
  absolute_offsets.reserve(offsets->size());
  if (parent.kind == RegionKind::Full) {
    absolute_offsets = *offsets;
  } else {
    if (parent.offsets.size() != offsets->size()) return UnknownRegion();
    for (size_t i = 0; i < offsets->size(); ++i) {
      auto absolute = AddInt64(parent.offsets[i], (*offsets)[i]);
      if (!absolute.has_value()) return UnknownRegion();
      absolute_offsets.push_back(*absolute);
    }
  }

  return AccessRegion{RegionKind::Box, std::move(absolute_offsets), std::move(*shape)};
}

bool RegionsMayOverlap(const AccessRegion& lhs, const AccessRegion& rhs) {
  if (lhs.kind == RegionKind::Unknown || rhs.kind == RegionKind::Unknown) return true;
  if (lhs.kind == RegionKind::Full || rhs.kind == RegionKind::Full) return true;
  if (lhs.offsets.size() != rhs.offsets.size() || lhs.shape.size() != rhs.shape.size() ||
      lhs.offsets.size() != lhs.shape.size()) {
    return true;
  }

  for (size_t i = 0; i < lhs.offsets.size(); ++i) {
    auto lhs_end = AddInt64(lhs.offsets[i], lhs.shape[i]);
    auto rhs_end = AddInt64(rhs.offsets[i], rhs.shape[i]);
    if (!lhs_end.has_value() || !rhs_end.has_value()) return true;
    if (*lhs_end <= rhs.offsets[i] || *rhs_end <= lhs.offsets[i]) return false;
  }
  return true;
}

bool SameRegion(const AccessRegion& lhs, const AccessRegion& rhs) {
  return lhs.kind == rhs.kind && lhs.offsets == rhs.offsets && lhs.shape == rhs.shape;
}

bool HasTaskIdTail(const CallPtr& call) {
  auto tuple_ty = As<TupleType>(call ? call->GetType() : TypePtr{});
  if (!tuple_ty || tuple_ty->types_.empty()) return false;
  auto scalar_ty = As<ScalarType>(tuple_ty->types_.back());
  return scalar_ty && scalar_ty->dtype_ == DataType::TASK_ID;
}

std::vector<ParamDirection> ResolveCalleeDirections(const ProgramPtr& program, const CallPtr& call,
                                                    const FunctionPtr& callee) {
  if (!callee) return {};
  if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
    return ComputeGroupEffectiveDirections(callee, program);
  }
  return callee->param_directions_;
}

std::vector<VarPtr> GetDepAttr(const CallPtr& call, const char* key) {
  if (!call) return {};
  for (const auto& [k, v] : call->attrs_) {
    if (k != key) continue;
    if (const auto* edges = std::any_cast<std::vector<VarPtr>>(&v)) {
      return *edges;
    }
    return {};
  }
  return {};
}

bool ContainsVar(const std::vector<VarPtr>& vars, const VarPtr& candidate) {
  if (!candidate) return false;
  for (const auto& var : vars) {
    if (var && var->UniqueId() == candidate->UniqueId()) return true;
  }
  return false;
}

void AppendUnique(std::vector<VarPtr>* vars, const VarPtr& candidate) {
  if (!vars || !candidate || ContainsVar(*vars, candidate)) return;
  vars->push_back(candidate);
}

bool HasHazard(AccessKind current, AccessKind prior) {
  const bool current_writes = current == AccessKind::Write || current == AccessKind::ReadWrite;
  const bool current_reads = current == AccessKind::Read || current == AccessKind::ReadWrite;
  const bool prior_writes = prior == AccessKind::Write || prior == AccessKind::ReadWrite;
  const bool prior_reads = prior == AccessKind::Read || prior == AccessKind::ReadWrite;
  return (current_reads && prior_writes) || (current_writes && prior_reads) ||
         (current_writes && prior_writes);
}

class StorageRootAnalysis : public IRVisitor {
 public:
  explicit StorageRootAnalysis(ProgramPtr program) : program_(std::move(program)) {}

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      if (param && IsTensorType(param->GetType())) {
        RegisterVarLocation(param, StorageLocation{param.get(), FullRegion()});
      }
    }
  }

  StorageLocation ResolveExpr(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return {};
    auto it = locations_.find(var.get());
    return it != locations_.end() ? it->second : StorageLocation{};
  }

  bool MayAlias(const Var* lhs, const Var* rhs) const {
    if (!lhs || !rhs) return false;
    if (lhs == rhs) return true;
    auto lhs_it = root_memrefs_.find(lhs);
    auto rhs_it = root_memrefs_.find(rhs);
    if (lhs_it == root_memrefs_.end() || rhs_it == root_memrefs_.end()) return false;
    return MemRef::MayAlias(lhs_it->second, rhs_it->second);
  }

 protected:
  void VisitStmt_(const IfStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    if (!op || op->return_vars_.empty() || !op->else_body_.has_value()) return;

    auto then_yield = GetTrailingYield(op->then_body_);
    auto else_yield = GetTrailingYield(op->else_body_.value());
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) break;
      auto then_location = ResolveExpr(then_yield->value_[i]);
      auto else_location = ResolveExpr(else_yield->value_[i]);
      if (!then_location.root || then_location.root != else_location.root) continue;
      if (!SameRegion(then_location.region, else_location.region)) {
        then_location.region = UnknownRegion();
      }
      RegisterVarLocation(op->return_vars_[i], std::move(then_location));
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto location = ResolveExpr(op->iter_args_[i]->initValue_);
      if (!location.root) continue;
      RegisterVarLocation(op->iter_args_[i], location);
      if (i < op->return_vars_.size()) {
        location.region = UnknownRegion();
        RegisterVarLocation(op->return_vars_[i], std::move(location));
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto location = ResolveExpr(op->iter_args_[i]->initValue_);
      if (!location.root) continue;
      RegisterVarLocation(op->iter_args_[i], location);
      if (i < op->return_vars_.size()) {
        location.region = UnknownRegion();
        RegisterVarLocation(op->return_vars_[i], std::move(location));
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      if (As<TupleType>(call->GetType()) && !IsBuiltinOp(call->op_->name_)) {
        tuple_locations_[op->var_.get()] = CollectCallOutputLocations(call);
        IRVisitor::VisitStmt_(op);
        return;
      }
    }

    if (!op->var_ || !IsTensorType(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    if (auto call = As<Call>(op->value_)) {
      const std::string& op_name = call->op_->name_;
      if (op_name == "tensor.create") {
        RegisterVarLocation(op->var_, StorageLocation{op->var_.get(), FullRegion()});
      } else if (op_name == "tensor.slice") {
        if (call->args_.size() >= 3) {
          auto parent = ResolveExpr(call->args_[0]);
          if (parent.root) {
            RegisterVarLocation(
                op->var_,
                StorageLocation{parent.root, SliceRegion(parent.region, call->args_[1], call->args_[2])});
          }
        }
      } else if (op_name == "tensor.assemble") {
        if (!call->args_.empty()) {
          auto base = ResolveExpr(call->args_[0]);
          if (base.root) {
            RegisterVarLocation(op->var_, std::move(base));
          }
        }
      } else if (!IsBuiltinOp(op_name)) {
        auto out_locations = CollectCallOutputLocations(call);
        if (As<TupleType>(call->GetType())) {
          tuple_locations_[op->var_.get()] = std::move(out_locations);
        } else if (!out_locations.empty() && out_locations[0].root) {
          RegisterVarLocation(op->var_, std::move(out_locations[0]));
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        auto it = tuple_locations_.find(tuple_var.get());
        if (it != tuple_locations_.end() && tuple_get->index_ >= 0 &&
            tuple_get->index_ < static_cast<int>(it->second.size()) && it->second[tuple_get->index_].root) {
          RegisterVarLocation(op->var_, it->second[tuple_get->index_]);
        }
      }
    } else {
      auto location = ResolveExpr(op->value_);
      if (location.root) {
        RegisterVarLocation(op->var_, std::move(location));
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  static YieldStmtPtr GetTrailingYield(const StmtPtr& stmt) {
    if (auto yield = As<YieldStmt>(stmt)) return yield;
    auto seq = As<SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) return nullptr;
    return As<YieldStmt>(seq->stmts_.back());
  }

  static MemRefPtr GetShapedMemRef(const TypePtr& type) {
    auto shaped = As<ShapedType>(type);
    if (!shaped || !shaped->memref_.has_value()) return nullptr;
    return shaped->memref_.value();
  }

  void RegisterVarLocation(const VarPtr& var, StorageLocation location) {
    if (!var || !location.root) return;
    locations_[var.get()] = location;
    if (const auto memref = GetShapedMemRef(var->GetType())) {
      root_memrefs_.try_emplace(location.root, memref);
    }
  }

  std::vector<StorageLocation> CollectCallOutputLocations(const CallPtr& call) const {
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return {};
    auto dirs = ResolveCalleeDirections(program_, call, callee);
    std::vector<StorageLocation> locations;
    for (size_t i = 0; i < dirs.size() && i < call->args_.size(); ++i) {
      if (dirs[i] != ParamDirection::Out && dirs[i] != ParamDirection::InOut) continue;
      locations.push_back(ResolveExpr(call->args_[i]));
    }
    return locations;
  }

  ProgramPtr program_;
  std::unordered_map<const Var*, StorageLocation> locations_;
  std::unordered_map<const Var*, MemRefPtr> root_memrefs_;
  std::unordered_map<const Var*, std::vector<StorageLocation>> tuple_locations_;
};

class SubmitTaskIdCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        tuple_get_by_tuple_[tuple_var.get()][tuple_get->index_] = op->var_;
        auto call_it = call_by_tuple_.find(tuple_var.get());
        if (call_it != call_by_tuple_.end()) {
          auto tuple_ty = As<TupleType>(call_it->second->GetType());
          const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
          if (tuple_get->index_ == task_id_index) {
            task_id_by_call_[call_it->second.get()] = op->var_;
          }
        }
      }
    }

    if (auto call = As<Call>(op->value_)) {
      if (HasTaskIdTail(call)) {
        call_by_tuple_[op->var_.get()] = call;
        auto tuple_ty = As<TupleType>(call->GetType());
        const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
        auto it = tuple_get_by_tuple_.find(op->var_.get());
        if (it != tuple_get_by_tuple_.end()) {
          auto elem_it = it->second.find(task_id_index);
          if (elem_it != it->second.end()) {
            task_id_by_call_[call.get()] = elem_it->second;
          }
        }
      }
    }

    IRVisitor::VisitStmt_(op);
  }

  const std::unordered_map<const Call*, VarPtr>& task_id_by_call() const { return task_id_by_call_; }

 private:
  std::unordered_map<const Var*, CallPtr> call_by_tuple_;
  std::unordered_map<const Var*, std::unordered_map<int, VarPtr>> tuple_get_by_tuple_;
  std::unordered_map<const Call*, VarPtr> task_id_by_call_;
};

class AutoDepMutator : public IRMutator {
 public:
  AutoDepMutator(ProgramPtr program, const StorageRootAnalysis* storage,
                 const std::unordered_map<const Call*, VarPtr>* task_id_by_call)
      : program_(std::move(program)), storage_(storage), task_id_by_call_(task_id_by_call) {}

 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (!op->manual_) {
      return IRMutator::VisitStmt_(op);
    }

    prior_stack_.emplace_back();
    auto out = IRMutator::VisitStmt_(op);
    prior_stack_.pop_back();
    return out;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call || prior_stack_.empty()) return base;
    if (IsBuiltinOp(call->op_->name_)) return call;

    VarPtr task_id = LookupTaskId(op.get());
    auto accesses = SummarizeAccesses(call);
    if (accesses.empty()) return call;

    std::vector<VarPtr> compiler_edges;
    auto user_edges = GetDepAttr(call, kAttrManualDepEdges);
    for (const auto& access : accesses) {
      for (const auto& prior : prior_stack_.back()) {
        if (!storage_ || !storage_->MayAlias(access.location.root, prior.location.root)) continue;
        if (!RegionsMayOverlap(access.location.region, prior.location.region)) continue;
        if (!HasHazard(access.kind, prior.kind)) continue;
        CHECK(prior.task_id_var)
            << "manual_scope auto-deps requires a producer TaskId for a prior call that writes storage read "
            << "or written by call '" << call->op_->name_
            << "'. Use `out, tid = pl.submit(self.kernel, ...)` for the producer inside manual_scope.";
        if (ContainsVar(user_edges, prior.task_id_var)) continue;
        AppendUnique(&compiler_edges, prior.task_id_var);
      }
    }

    for (auto& access : accesses) {
      access.task_id_var = task_id;
      prior_stack_.back().push_back(std::move(access));
    }

    if (compiler_edges.empty()) {
      return call;
    }

    auto new_attrs = WithCompilerManualDepEdgesAttr(call->attrs_, std::move(compiler_edges));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
  VarPtr LookupTaskId(const Call* call) const {
    if (!task_id_by_call_) return nullptr;
    auto it = task_id_by_call_->find(call);
    return it != task_id_by_call_->end() ? it->second : nullptr;
  }

  std::vector<StorageAccess> SummarizeAccesses(const CallPtr& call) const {
    std::vector<StorageAccess> out;
    auto dirs = call->GetArgDirections();
    if (dirs.size() != call->args_.size()) return out;

    for (size_t i = 0; i < dirs.size(); ++i) {
      auto location = storage_ ? storage_->ResolveExpr(call->args_[i]) : StorageLocation{};
      if (!location.root) continue;
      std::optional<AccessKind> kind;
      switch (dirs[i]) {
        case ArgDirection::Input:
          kind = AccessKind::Read;
          break;
        case ArgDirection::Output:
        case ArgDirection::OutputExisting:
          kind = AccessKind::Write;
          break;
        case ArgDirection::InOut:
          kind = AccessKind::ReadWrite;
          break;
        case ArgDirection::NoDep:
        case ArgDirection::Scalar:
          break;
      }
      if (kind.has_value()) {
        out.push_back(StorageAccess{std::move(location), *kind, nullptr});
      }
    }
    return out;
  }

  ProgramPtr program_;
  const StorageRootAnalysis* storage_;
  const std::unordered_map<const Call*, VarPtr>* task_id_by_call_;
  std::vector<std::vector<StorageAccess>> prior_stack_;
};

}  // namespace

namespace pass {

Pass AutoDeriveTaskDependencies() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    auto new_functions = program->functions_;
    bool changed = false;

    for (auto& [gvar, func] : new_functions) {
      (void)gvar;
      if (!func || !func->body_) continue;

      StorageRootAnalysis storage(program);
      storage.Initialize(func->params_);
      storage.VisitStmt(func->body_);

      SubmitTaskIdCollector task_ids;
      task_ids.VisitStmt(func->body_);

      AutoDepMutator mutator(program, &storage, &task_ids.task_id_by_call());
      auto new_body = mutator.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;

      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "AutoDeriveTaskDependencies", kAutoDeriveTaskDependenciesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
