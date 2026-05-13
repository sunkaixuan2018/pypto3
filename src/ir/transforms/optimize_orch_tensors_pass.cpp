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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace {

// ============================================================================
// Shared helpers
// ============================================================================

/// Find a function by name in a program.
FunctionPtr FindFunction(const ProgramPtr& program, const std::string& name) {
  for (const auto& [gvar, func] : program->functions_) {
    if (func->name_ == name) return func;
  }
  return nullptr;
}

/// Get the GlobalVar name from a Call, or empty string.
std::string GetCallFuncName(const CallPtr& call) {
  auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
  return gvar ? gvar->name_ : "";
}

/// Compute row-major strides from a shape: [D1*D2*...*Dn, D2*...*Dn, ..., 1].
/// Returns empty vector if any dimension is not a ConstInt.
std::vector<ExprPtr> ComputeRowMajorStrides(const std::vector<ExprPtr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto& dim : shape) {
    auto ci = As<ConstInt>(dim);
    if (!ci) return {};
    dims.push_back(ci->value_);
  }

  size_t ndim = dims.size();
  std::vector<ExprPtr> strides(ndim);
  int64_t product = 1;
  for (size_t i = ndim; i > 0; --i) {
    strides[i - 1] = std::make_shared<ConstInt>(product, DataType::INDEX, Span::unknown());
    product *= dims[i - 1];
  }
  return strides;
}

/// Return the unique Out-param index for `func`, or nullopt if the function
/// has zero or multiple Out params.
std::optional<size_t> GetSingleOutParamIndex(const FunctionPtr& func) {
  if (!func) return std::nullopt;
  std::optional<size_t> out_index;
  for (size_t i = 0; i < func->param_directions_.size(); ++i) {
    if (func->param_directions_[i] != ParamDirection::Out) continue;
    if (out_index.has_value()) return std::nullopt;
    out_index = i;
  }
  return out_index;
}

/// Count Var/IterArg references to `target` inside a statement subtree.
size_t CountVarRefsInStmt(const StmtPtr& stmt, const Var* target) {
  class Counter : public IRVisitor {
   public:
    explicit Counter(const Var* target) : target_(target) {}

    size_t count() const { return count_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

   private:
    const Var* target_;
    size_t count_ = 0;
  };

  Counter counter(target);
  counter.VisitStmt(stmt);
  return counter.count();
}

/// Build a unique cloned-function name rooted at `base_name`.
std::string MakeUniqueFunctionName(const ProgramPtr& program, const std::string& base_name) {
  if (!program || !program->GetFunction(base_name)) return base_name;
  for (size_t suffix = 1;; ++suffix) {
    auto candidate = base_name + "_" + std::to_string(suffix);
    if (!program->GetFunction(candidate)) return candidate;
  }
}

bool ExprReferencesOnlyVarsIn(const ExprPtr& expr, const std::unordered_set<const Var*>& allowed) {
  class Checker : public IRVisitor {
   public:
    explicit Checker(const std::unordered_set<const Var*>& allowed) : allowed_(allowed) {}

    bool ok() const { return ok_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

   private:
    const std::unordered_set<const Var*>& allowed_;
    bool ok_ = true;
  };

  Checker checker(allowed);
  checker.VisitExpr(expr);
  return checker.ok();
}

bool IsAllZeroOffsets(const std::vector<ExprPtr>& offsets) {
  for (const auto& offset : offsets) {
    auto ci = As<ConstInt>(offset);
    if (!ci || ci->value_ != 0) return false;
  }
  return true;
}

/// Info about an InCore function's Out params and their return mappings.
struct OutParamReturnMapping {
  size_t param_index;   ///< Position in param list
  size_t return_index;  ///< Which return value stores to this Out param
  VarPtr param_var;     ///< The Out param variable
};

/// Build the mapping from Out params to return indices for an InCore function.
/// Scans tile.store calls before the ReturnStmt to find which Out param
/// each return value stores to.
std::vector<OutParamReturnMapping> BuildOutParamReturnMappings(const FunctionPtr& func) {
  // Collect Out param vars and their indices
  std::unordered_map<const Var*, size_t> out_var_to_param_idx;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::Out) {
      out_var_to_param_idx[func->params_[i].get()] = i;
    }
  }
  if (out_var_to_param_idx.empty()) return {};

  auto body_stmts = FlattenToStmts(func->body_);

  // Build var->assign map for quick lookup
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  for (const auto& stmt : body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      var_def[assign->var_.get()] = assign;
    }
  }

  // Find return statement
  ReturnStmtPtr return_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
      break;
    }
  }
  if (!return_stmt) return {};

  std::vector<OutParamReturnMapping> result;

  for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
    auto ret_var = As<Var>(return_stmt->value_[ret_i]);
    if (!ret_var) continue;

    auto def_it = var_def.find(ret_var.get());
    if (def_it == var_def.end()) continue;

    auto call = As<Call>(def_it->second->value_);
    if (!call || call->op_->name_ != "tile.store") continue;

    if (call->args_.size() < 3) continue;
    auto out_tensor = As<Var>(call->args_[2]);
    if (!out_tensor) continue;

    auto param_it = out_var_to_param_idx.find(out_tensor.get());
    if (param_it == out_var_to_param_idx.end()) continue;

    result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
  }

  return result;
}

// ============================================================================
// Pattern 1: IterArgReuseOptimizer
//
// Detects when a tensor.create for an InCore Out param is inside a
// ForStmt/WhileStmt loop where the InCore result feeds back as an iter-arg,
// and the corresponding In param receives the iter-arg value.
//
// Optimization: remove the tensor.create, remove the Out param from the InCore
// function, promote the In param to InOut, redirect tile.store to the In param.
// ============================================================================

class IterArgReuseOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto reuse_results = Analyze(program, incore_names);

    // Rewrite InCore functions
    std::unordered_map<std::string, FunctionPtr> rewritten_incores;
    for (auto& [fname, reuse] : reuse_results) {
      auto func = FindFunction(program, fname);
      if (!func) continue;
      rewritten_incores[fname] = RewriteIncore(func, reuse.merges);
    }

    // Build the new function list
    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (rewritten_incores.count(func->name_)) {
        new_functions.push_back(rewritten_incores[func->name_]);
      } else if (!incore_names.count(func->name_)) {
        // Orchestration function: rewrite call sites
        DeadCreateScanner scanner(reuse_results);
        scanner.VisitStmt(func->body_);

        CallSiteRewriter rewriter(reuse_results, rewritten_incores, scanner.dead_creates());
        auto new_body = rewriter.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          auto new_func = MutableCopy(func);
          new_func->body_ = new_body;
          new_functions.push_back(new_func);
        } else {
          new_functions.push_back(func);
        }
      } else {
        new_functions.push_back(func);
      }
    }

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// A single Out->In merge for an InCore function.
  struct OutToInMerge {
    size_t out_param_index;
    size_t in_param_index;
  };

  /// Per-InCore-function analysis result.
  struct AnalysisResult {
    std::string func_name;
    std::vector<OutToInMerge> merges;
  };

  // -- Analysis: IRVisitor that finds ForStmt/WhileStmt with iter-arg reuse --

  class LoopAnalyzer : public IRVisitor {
   public:
    LoopAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names,
                 const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings)
        : program_(program), incore_names_(incore_names), out_mappings_(out_mappings) {}

    const std::unordered_map<std::string, AnalysisResult>& results() const { return results_; }

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);  // Recurse into body first
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

   private:
    void AnalyzeLoop(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body) {
      auto loop_body_stmts = FlattenToStmts(body);

      // Collect tensor.create vars and InCore call assignments
      std::unordered_set<const Var*> tensor_create_vars;
      std::vector<AssignStmtPtr> incore_calls;

      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call) continue;

        if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
          tensor_create_vars.insert(assign->var_.get());
        } else {
          auto fname = GetCallFuncName(call);
          if (incore_names_.count(fname)) {
            incore_calls.push_back(assign);
          }
        }
      }

      // Find yield statement
      auto yield = transform_utils::FindYieldStmt(body);
      if (!yield) return;

      // Build tuple extract map: call_result_var -> {tuple_index -> dest_var}
      std::unordered_map<const Var*, std::unordered_map<size_t, const Var*>> tuple_extracts;
      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto tgi = As<TupleGetItemExpr>(assign->value_);
        if (!tgi) continue;
        if (auto src_var = As<Var>(tgi->tuple_)) {
          tuple_extracts[src_var.get()][static_cast<size_t>(tgi->index_)] = assign->var_.get();
        }
      }

      // Analyze each InCore call
      for (const auto& call_assign : incore_calls) {
        auto call = As<Call>(call_assign->value_);
        auto fname = GetCallFuncName(call);
        auto mapping_it = out_mappings_.find(fname);
        if (mapping_it == out_mappings_.end()) continue;
        const auto& out_param_mappings = mapping_it->second;

        auto incore_func = FindFunction(program_, fname);
        if (!incore_func) continue;

        AnalysisResult reuse_result;
        reuse_result.func_name = fname;

        for (const auto& opm : out_param_mappings) {
          if (opm.param_index >= call->args_.size()) continue;
          auto out_arg_var = As<Var>(call->args_[opm.param_index]);
          if (!out_arg_var || !tensor_create_vars.count(out_arg_var.get())) continue;

          // Trace return[ret_i] -> yield[y_i]
          size_t yield_index = SIZE_MAX;

          if (out_param_mappings.size() == 1 && incore_func->return_types_.size() == 1) {
            for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
              auto yv = As<Var>(yield->value_[yi]);
              if (yv && yv.get() == call_assign->var_.get()) {
                yield_index = yi;
                break;
              }
            }
          } else {
            auto extract_it = tuple_extracts.find(call_assign->var_.get());
            if (extract_it != tuple_extracts.end()) {
              auto ret_it = extract_it->second.find(opm.return_index);
              if (ret_it != extract_it->second.end()) {
                for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
                  auto yv = As<Var>(yield->value_[yi]);
                  if (yv && yv.get() == ret_it->second) {
                    yield_index = yi;
                    break;
                  }
                }
              }
            }
          }
          if (yield_index == SIZE_MAX) continue;

          // Check iter_arg[yield_index] maps to an In param of the same call
          if (yield_index >= iter_args.size()) continue;
          const auto* iter_arg_ptr = iter_args[yield_index].get();

          for (size_t arg_i = 0; arg_i < call->args_.size(); ++arg_i) {
            const Var* raw_ptr = nullptr;
            if (auto var = As<Var>(call->args_[arg_i])) {
              raw_ptr = var.get();
            } else if (auto ia = As<IterArg>(call->args_[arg_i])) {
              raw_ptr = ia.get();
            }
            if (raw_ptr != iter_arg_ptr) continue;

            if (arg_i < incore_func->param_directions_.size() &&
                incore_func->param_directions_[arg_i] == ParamDirection::In) {
              reuse_result.merges.push_back({opm.param_index, arg_i});
            }
            break;
          }
        }

        if (!reuse_result.merges.empty()) {
          if (results_.find(fname) == results_.end()) {
            results_[fname] = std::move(reuse_result);
          }
        }
      }
    }

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings_;
    std::unordered_map<std::string, AnalysisResult> results_;
  };

  // -- Analysis: InCore internal In↔Out param pairings ----------------------
  //
  // A callee's In param and Out param are aliasing-compatible when the callee
  // reads the In fully via `tile.load` and writes the Out fully via `tile.store`
  // of a value that flows through at least one loop iter_arg chain starting
  // from that tile.load. The loop hop is what signals the In/Out are meant to
  // share storage (an accumulator); a plain load→store pair does not.

  /// Check that a tile.load call reads the full tensor — all offsets zero and
  /// both `shapes` and `valid_shapes` match the tensor shape dimension-by-
  /// dimension. `valid_shapes` differs from `shapes` for masked/padded loads,
  /// which must NOT be treated as full loads.
  static bool IsFullTensorLoad(const CallPtr& load_call, const TensorTypePtr& tensor_type) {
    if (!load_call || load_call->args_.size() < 4 || !tensor_type) return false;
    auto offsets = As<MakeTuple>(load_call->args_[1]);
    auto load_shape = As<MakeTuple>(load_call->args_[2]);
    auto valid_shape = As<MakeTuple>(load_call->args_[3]);
    if (!offsets || !load_shape || !valid_shape) return false;
    const size_t ndim = tensor_type->shape_.size();
    if (offsets->elements_.size() != ndim || load_shape->elements_.size() != ndim ||
        valid_shape->elements_.size() != ndim) {
      return false;
    }
    for (size_t i = 0; i < ndim; ++i) {
      auto want = std::dynamic_pointer_cast<const ConstInt>(tensor_type->shape_[i]);
      auto got_load = std::dynamic_pointer_cast<const ConstInt>(load_shape->elements_[i]);
      auto got_valid = std::dynamic_pointer_cast<const ConstInt>(valid_shape->elements_[i]);
      if (!want || !got_load || !got_valid) return false;
      if (want->value_ != got_load->value_ || want->value_ != got_valid->value_) return false;
      if (!IsConstValue(offsets->elements_[i], 0)) return false;
    }
    return true;
  }

  /// Compare two TensorTypes for compatible constant shape + dtype.
  static bool TensorTypesMatch(const TypePtr& a, const TypePtr& b) {
    auto ta = As<TensorType>(a);
    auto tb = As<TensorType>(b);
    if (!ta || !tb || ta->dtype_ != tb->dtype_) return false;
    if (ta->shape_.size() != tb->shape_.size()) return false;
    for (size_t i = 0; i < ta->shape_.size(); ++i) {
      auto ca = std::dynamic_pointer_cast<const ConstInt>(ta->shape_[i]);
      auto cb = std::dynamic_pointer_cast<const ConstInt>(tb->shape_[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  /// Walk body collecting: AssignStmt var_def map, ForStmt/WhileStmt
  /// return_var → iter_arg init value map, and the top-level ReturnStmt.
  class IterChainCollector : public IRVisitor {
   public:
    std::unordered_map<const Var*, AssignStmtPtr> var_def;
    std::unordered_map<const Var*, ExprPtr> return_var_to_init;
    ReturnStmtPtr return_stmt;

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      var_def[op->var_.get()] = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ReturnStmtPtr& op) override {
      if (!return_stmt) return_stmt = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ForStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
  };

  /// For each Out param, trace `tile.store` source back through loop iter_arg
  /// chains to a `tile.load` of an In param. Returns (in_idx, out_idx) pairs
  /// where the chain exists, types match, and the load covers the full tensor.
  static std::vector<std::pair<size_t, size_t>> BuildInOutParamPairings(const FunctionPtr& func) {
    std::vector<std::pair<size_t, size_t>> pairings;

    std::unordered_map<const Var*, size_t> in_param_idx;
    for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
      if (func->param_directions_[i] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[i]->GetType())) continue;
      in_param_idx[func->params_[i].get()] = i;
    }
    auto out_mappings = BuildOutParamReturnMappings(func);
    if (in_param_idx.empty() || out_mappings.empty()) return pairings;

    IterChainCollector collector;
    collector.VisitStmt(func->body_);
    if (!collector.return_stmt) return pairings;

    std::unordered_set<size_t> used_in_indices;
    for (const auto& opm : out_mappings) {
      if (opm.return_index >= collector.return_stmt->value_.size()) continue;
      auto ret_var = As<Var>(collector.return_stmt->value_[opm.return_index]);
      if (!ret_var) continue;
      auto ret_def = collector.var_def.find(ret_var.get());
      if (ret_def == collector.var_def.end()) continue;
      auto store_call = As<Call>(ret_def->second->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.empty()) continue;

      // Trace backward through iter_arg chains. Require at least one loop hop:
      // a bare tile.load → tile.store without an accumulator has no semantic
      // indication that In and Out were intended to alias.
      const Var* current = nullptr;
      if (auto src_var = As<Var>(store_call->args_[0])) current = src_var.get();
      int loop_hops = 0;
      for (int hops = 0; hops < 16 && current; ++hops) {
        auto it = collector.return_var_to_init.find(current);
        if (it == collector.return_var_to_init.end()) break;
        auto init_var = As<Var>(it->second);
        if (!init_var) {
          current = nullptr;
          break;
        }
        current = init_var.get();
        ++loop_hops;
      }
      if (!current || loop_hops == 0) continue;

      auto load_def = collector.var_def.find(current);
      if (load_def == collector.var_def.end()) continue;
      auto load_call = As<Call>(load_def->second->value_);
      if (!load_call || load_call->op_->name_ != "tile.load" || load_call->args_.empty()) continue;
      auto load_src = As<Var>(load_call->args_[0]);
      if (!load_src) continue;
      auto in_it = in_param_idx.find(load_src.get());
      if (in_it == in_param_idx.end()) continue;

      auto tensor_type = As<TensorType>(func->params_[in_it->second]->GetType());
      if (!TensorTypesMatch(tensor_type, opm.param_var->GetType())) continue;
      if (!IsFullTensorLoad(load_call, tensor_type)) continue;

      if (!used_in_indices.insert(in_it->second).second) continue;
      pairings.emplace_back(in_it->second, opm.param_index);
    }
    return pairings;
  }

  // -- Analysis: standalone (non-looped) InCore calls whose In/Out can merge -

  /// One-shot visitor that collects everything the standalone analyzer needs
  /// from an orchestration function body: per-Var use counts, the set of Vars
  /// assigned by `tensor.create`, and the expression AST of each AssignStmt.
  /// Keeping it all in a single walk keeps per-function analysis O(N).
  ///
  /// Counts exclude definitional occurrences (AssignStmt LHS, loop_var,
  /// return_vars, iter_arg self-refs) so `use_count[v]` is the number of
  /// real reads of `v` in expressions.
  class FunctionBodyIndex : public IRVisitor {
   public:
    std::unordered_map<const Var*, size_t> use_count;
    std::unordered_set<const Var*> local_creates;

   protected:
    void VisitExpr_(const VarPtr& op) override { ++use_count[op.get()]; }
    void VisitExpr_(const IterArgPtr& op) override { ++use_count[op.get()]; }

    void VisitStmt_(const AssignStmtPtr& op) override {
      // Skip LHS (a def); visit only the RHS value.
      VisitExpr(op->value_);
      if (auto call = As<Call>(op->value_); call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) &&
                                            call->op_->name_ == "tensor.create") {
        local_creates.insert(op->var_.get());
      }
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      VisitExpr(op->start_);
      VisitExpr(op->stop_);
      VisitExpr(op->step_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      VisitExpr(op->condition_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }
  };

  /// Count references to `target` within a single expression tree.
  static size_t CountVarRefs(const ExprPtr& expr, const Var* target) {
    class Counter : public IRVisitor {
     public:
      const Var* target;
      size_t count = 0;
      void VisitExpr_(const VarPtr& op) override {
        if (op.get() == target) ++count;
      }
      void VisitExpr_(const IterArgPtr& op) override {
        if (op.get() == target) ++count;
      }
    } c;
    c.target = target;
    c.VisitExpr(expr);
    return c.count;
  }

  /// Record of a standalone InCore call site. Owns references to the call's
  /// orchestration-function context so that we can test each candidate merge
  /// against every call site of the same callee before recording it.
  struct StandaloneCallSite {
    const FunctionBodyIndex* body_index;
    AssignStmtPtr assign_stmt;
    CallPtr call;
  };

  /// Collects standalone InCore calls (those outside any iter-arg-carrying
  /// loop) in an orchestration function body, keyed by callee name.
  class StandaloneCallCollector : public IRVisitor {
   public:
    StandaloneCallCollector(const std::unordered_set<std::string>& incore_names,
                            const FunctionBodyIndex& body_index,
                            std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out)
        : incore_names_(incore_names), body_index_(body_index), out_(out) {}

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (!inside_iter_loop_) {
        if (auto call = As<Call>(op->value_)) {
          auto fname = GetCallFuncName(call);
          if (!fname.empty() && incore_names_.count(fname)) {
            out_[fname].push_back({&body_index_, op, call});
          }
        }
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    const std::unordered_set<std::string>& incore_names_;
    const FunctionBodyIndex& body_index_;
    std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out_;
    bool inside_iter_loop_ = false;
  };

  /// Check whether a (in_idx, out_idx) pairing is safe to apply at `site`:
  /// the Out arg is a locally-allocated `tensor.create`, and the In arg's
  /// sole use in the enclosing orch function is this call.
  static bool IsPairingSafeAtCallSite(const StandaloneCallSite& site, size_t in_idx, size_t out_idx) {
    const auto& call = site.call;
    if (out_idx >= call->args_.size() || in_idx >= call->args_.size()) return false;
    auto out_var = As<Var>(call->args_[out_idx]);
    auto in_var = As<Var>(call->args_[in_idx]);
    if (!out_var || !in_var) return false;
    if (!site.body_index->local_creates.count(out_var.get())) return false;

    auto use_it = site.body_index->use_count.find(in_var.get());
    size_t total_refs = use_it == site.body_index->use_count.end() ? 0 : use_it->second;
    size_t self_refs = CountVarRefs(site.assign_stmt->value_, in_var.get());
    return total_refs == self_refs;
  }

  /// Analyze orchestration functions for iter-arg reuse opportunities.
  std::unordered_map<std::string, AnalysisResult> Analyze(
      const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::unordered_map<std::string, std::vector<OutParamReturnMapping>> out_mappings;
    std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> in_out_pairings;
    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) continue;
      out_mappings[func->name_] = BuildOutParamReturnMappings(func);
      in_out_pairings[func->name_] = BuildInOutParamPairings(func);
    }

    LoopAnalyzer analyzer(program, incore_names, out_mappings);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    auto results = analyzer.results();

    // Collect all standalone call sites (preserving body_index references per
    // orchestration function).
    std::vector<FunctionBodyIndex> body_indices;
    body_indices.reserve(program->functions_.size());
    std::unordered_map<std::string, std::vector<StandaloneCallSite>> standalone_sites;
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      body_indices.emplace_back();
      auto& body_index = body_indices.back();
      body_index.VisitStmt(func->body_);
      StandaloneCallCollector collector(incore_names, body_index, standalone_sites);
      collector.VisitStmt(func->body_);
    }

    // For each callee with standalone call sites, only record a merge if
    // EVERY standalone call site satisfies the pairing's safety preconditions.
    // Caller-dependent safety cannot be cached per-callee otherwise: the
    // rewrite applies globally to every call of that function.
    for (const auto& [fname, sites] : standalone_sites) {
      if (results.count(fname)) continue;  // LoopAnalyzer already handled
      auto pair_it = in_out_pairings.find(fname);
      if (pair_it == in_out_pairings.end() || pair_it->second.empty()) continue;

      AnalysisResult partial;
      partial.func_name = fname;
      std::unordered_set<size_t> seen_out;
      std::unordered_set<size_t> seen_in;

      for (const auto& [in_idx, out_idx] : pair_it->second) {
        bool all_safe = true;
        for (const auto& site : sites) {
          if (!IsPairingSafeAtCallSite(site, in_idx, out_idx)) {
            all_safe = false;
            break;
          }
        }
        if (!all_safe) continue;
        if (!seen_out.insert(out_idx).second) continue;
        if (!seen_in.insert(in_idx).second) continue;
        partial.merges.push_back({out_idx, in_idx});
      }

      if (!partial.merges.empty()) results[fname] = std::move(partial);
    }
    return results;
  }

  // -- Pre-scan: IRVisitor that identifies dead tensor.create vars -----------

  class DeadCreateScanner : public IRVisitor {
   public:
    explicit DeadCreateScanner(const std::unordered_map<std::string, AnalysisResult>& reuse_results)
        : reuse_results_(reuse_results) {}

    const std::unordered_set<const Var*>& dead_creates() const { return dead_creates_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return;
      for (const auto& merge : reuse_it->second.merges) {
        if (merge.out_param_index < call->args_.size()) {
          if (auto create_var = As<Var>(call->args_[merge.out_param_index])) {
            dead_creates_.insert(create_var.get());
          }
        }
      }
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    std::unordered_set<const Var*> dead_creates_;
  };

  // -- Mutation: IRMutator that substitutes Var references --------------------

  class VarSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  /// Rewrite an InCore function to merge Out params into In params.
  FunctionPtr RewriteIncore(const FunctionPtr& func, const std::vector<OutToInMerge>& merges) {
    std::unordered_set<size_t> out_indices_to_remove;
    VarSubstitutionMutator mutator;
    std::unordered_set<size_t> in_indices_to_promote;

    for (const auto& merge : merges) {
      out_indices_to_remove.insert(merge.out_param_index);
      in_indices_to_promote.insert(merge.in_param_index);
      mutator.AddSubstitution(func->params_[merge.out_param_index].get(),
                              func->params_[merge.in_param_index]);
    }

    std::vector<VarPtr> new_params;
    std::vector<ParamDirection> new_directions;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (out_indices_to_remove.count(i)) continue;
      new_params.push_back(func->params_[i]);
      if (in_indices_to_promote.count(i)) {
        new_directions.push_back(ParamDirection::InOut);
      } else {
        new_directions.push_back(i < func->param_directions_.size() ? func->param_directions_[i]
                                                                    : ParamDirection::In);
      }
    }

    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, new_params, new_directions, func->return_types_, new_body,
                                      func->span_, func->func_type_, func->level_, func->role_, func->attrs_);
  }

  // -- Mutation: IRMutator that rewrites orch call sites ---------------------

  class CallSiteRewriter : public IRMutator {
   public:
    CallSiteRewriter(const std::unordered_map<std::string, AnalysisResult>& reuse_results,
                     const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs,
                     const std::unordered_set<const Var*>& dead_creates)
        : reuse_results_(reuse_results), rewritten_funcs_(rewritten_funcs), dead_creates_(dead_creates) {}

   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return IRMutator::VisitStmt_(op);

      // Remove dead tensor.create
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
        if (dead_creates_.count(op->var_.get())) {
          return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
        }
        return IRMutator::VisitStmt_(op);
      }

      // Rewrite calls to rewritten InCore functions
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return IRMutator::VisitStmt_(op);

      auto func_it = rewritten_funcs_.find(fname);
      if (func_it == rewritten_funcs_.end()) return IRMutator::VisitStmt_(op);

      const auto& merges = reuse_it->second.merges;
      const auto& new_func = func_it->second;

      std::unordered_set<size_t> remove_indices;
      for (const auto& merge : merges) {
        remove_indices.insert(merge.out_param_index);
      }

      std::vector<ExprPtr> new_args;
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (remove_indices.count(i)) continue;
        new_args.push_back(VisitExpr(call->args_[i]));
      }

      TypePtr new_return_type;
      if (new_func->return_types_.empty()) {
        new_return_type = nullptr;
      } else if (new_func->return_types_.size() == 1) {
        new_return_type = new_func->return_types_[0];
      } else {
        new_return_type = std::make_shared<TupleType>(new_func->return_types_);
      }

      std::shared_ptr<Call> new_call;
      if (new_return_type) {
        new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, new_return_type, call->span_);
      } else {
        new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->span_);
      }

      auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(op);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = var_remap_.find(op.get());
      if (it != var_remap_.end()) return it->second;
      return op;
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs_;
    std::unordered_set<const Var*> dead_creates_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };
};

// ============================================================================
// Pattern 2: AssembleParentStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.assemble(parent, incore_result, offset)
// where incore_result comes from an InCore call. Records the parent
// tensor's shape. Then updates the InCore function's Out param
// TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class AssembleParentStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto parent_shapes = Analyze(program, incore_names);
    if (parent_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, parent_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  using ParentShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  // -- Analysis: IRVisitor that tracks InCore call results and finds assemble patterns --

  class AssembleAnalyzer : public IRVisitor {
   public:
    AssembleAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const ParentShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) {
        // Check for TupleGetItem extracting from an InCore call result
        auto tgi = As<TupleGetItemExpr>(op->value_);
        if (tgi) {
          auto src_var = AsVarLike(tgi->tuple_);
          if (src_var) {
            auto it = var_to_incore_return_.find(src_var.get());
            if (it != var_to_incore_return_.end()) {
              var_to_incore_return_[op->var_.get()] = {it->second.func_name,
                                                       static_cast<size_t>(tgi->index_)};
            }
          }
        }
        return;
      }

      // Check if this is an InCore call
      auto fname = GetCallFuncName(call);
      if (incore_names_.count(fname)) {
        auto incore_func = FindFunction(program_, fname);
        if (incore_func && incore_func->return_types_.size() == 1) {
          var_to_incore_return_[op->var_.get()] = {fname, 0};
        } else if (incore_func && incore_func->return_types_.size() > 1) {
          var_to_incore_return_[op->var_.get()] = {fname, SIZE_MAX};
        }
        return;
      }

      // Check if this is a tensor.assemble(parent, source, offset)
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.assemble" &&
          call->args_.size() == 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        auto source_var = AsVarLike(call->args_[1]);
        if (!parent_var || !source_var) return;

        auto src_it = var_to_incore_return_.find(source_var.get());
        if (src_it == var_to_incore_return_.end()) return;
        if (src_it->second.return_index == SIZE_MAX) return;

        auto parent_tensor_type = As<TensorType>(parent_var->GetType());
        if (!parent_tensor_type) return;

        result_[src_it->second.func_name][src_it->second.return_index] = parent_tensor_type->shape_;
      }
    }

   private:
    struct IncoreReturnInfo {
      std::string func_name;
      size_t return_index;
    };

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, IncoreReturnInfo> var_to_incore_return_;
    ParentShapeMap result_;
  };

  /// Analyze orchestration functions for tensor.assemble patterns.
  ParentShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    AssembleAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  static std::vector<ExprPtr> ComputeStrides(const std::vector<ExprPtr>& shape) {
    return ComputeRowMajorStrides(shape);
  }

  // -- Mutation: IRMutator that propagates updated param types through tile.store --

  class ParamStrideUpdateMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) {
      subs_[old_ptr] = new_var;
      new_param_ptrs_.insert(new_var.get());
    }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      auto remap_it = var_remap_.find(op.get());
      if (remap_it != var_remap_.end()) return remap_it->second;
      return op;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto assign = As<AssignStmt>(visited);
      if (!assign) return visited;

      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tile.store" || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || !new_param_ptrs_.count(out_var.get())) return assign;

      auto out_type = out_var->GetType();
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, out_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, out_type, assign->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(assign);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
    std::unordered_set<const Var*> new_param_ptrs_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };

  /// Apply assemble parent strides to InCore functions.
  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const ParentShapeMap& parent_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto ps_it = parent_shapes.find(func->name_);
      if (ps_it == parent_shapes.end()) continue;
      const auto& return_idx_to_shape = ps_it->second;

      auto out_mappings = BuildOutParamReturnMappings(func);
      if (out_mappings.empty()) continue;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& opm : out_mappings) {
        auto shape_it = return_idx_to_shape.find(opm.return_index);
        if (shape_it == return_idx_to_shape.end()) continue;

        auto full_strides = ComputeStrides(shape_it->second);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[opm.param_index]->GetType());
        if (!tensor_type) continue;

        // Extract trailing strides matching the output tensor's rank.
        // For a 3D parent [B, M, N] with strides [M*N, N, 1] and a 2D output [M', N'],
        // we need the last 2 strides: [N, 1].
        size_t out_rank = tensor_type->shape_.size();
        if (out_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(out_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[opm.param_index]->name_hint_, new_type,
                                               func->params_[opm.param_index]->span_);

        changed = true;
        new_params[opm.param_index] = new_param;
      }

      if (!changed) continue;

      ParamStrideUpdateMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 3: AssembleLoopRewriter
//
// InCore-local optimization: when an InCore function body has a ForStmt
// that does tile.assemble accumulation yielding back to iter-arg, and the
// ForStmt result feeds the final tile.store -> return, rewrite the loop
// to use tile.store instead of tile.assemble, with the iter-arg initialized
// from the Out param.
// ============================================================================

class AssembleLoopRewriter {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) {
        new_functions.push_back(func);
        continue;
      }
      bool has_out = false;
      for (const auto& dir : func->param_directions_) {
        if (dir == ParamDirection::Out) {
          has_out = true;
          break;
        }
      }
      if (!has_out) {
        new_functions.push_back(func);
        continue;
      }
      auto rewritten = RewriteFunction(func);
      if (rewritten.get() != func.get()) changed = true;
      new_functions.push_back(rewritten);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// Check if a statement subtree uses a given Var (by raw pointer).
  /// Uses VarDefUseCollector which handles all statement/expression types.
  static bool StmtUsesVar(const StmtPtr& stmt, const Var* var) {
    if (!stmt || !var) return false;
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(stmt);
    return collector.var_uses.count(var) > 0;
  }

  // -- Pre-scan: IRVisitor that collects var definitions and return statement --

  class BodyScanner : public IRVisitor {
   public:
    const std::unordered_map<const Var*, AssignStmtPtr>& var_def() const { return var_def_; }
    const ReturnStmtPtr& return_stmt() const { return return_stmt_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override { var_def_[op->var_.get()] = op; }
    void VisitStmt_(const ReturnStmtPtr& op) override { return_stmt_ = op; }

   private:
    std::unordered_map<const Var*, AssignStmtPtr> var_def_;
    ReturnStmtPtr return_stmt_;
  };

  // -- Mutation: IRMutator that rewrites matching ForStmt patterns -----------

  class LoopRewriteMutator : public IRMutator {
   public:
    struct StoreReturnInfo {
      const Var* store_var;
      const Var* out_param;
      size_t return_index;
    };

    LoopRewriteMutator(const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                       const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                       const std::unordered_set<const Var*>& dead_create_vars,
                       const std::unordered_set<const Var*>& dead_store_vars,
                       const std::unordered_map<const Var*, VarPtr>& return_var_remap,
                       const FunctionPtr& func)
        : out_var_to_param_idx_(out_var_to_param_idx),
          for_return_to_store_(for_return_to_store),
          dead_create_vars_(dead_create_vars),
          dead_store_vars_(dead_store_vars),
          return_var_remap_(return_var_remap),
          func_(func) {}

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
      if (fret_it == for_return_to_store_.end()) {
        return IRMutator::VisitStmt_(op);
      }
      const auto& store_info = *fret_it->second;

      auto loop_body_stmts = FlattenToStmts(op->body_);
      auto yield = transform_utils::FindYieldStmt(op->body_);
      if (!yield || yield->value_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      const IterArg* iter_arg = op->iter_args_[0].get();

      // Find the tile.assemble call
      AssignStmtPtr assemble_assign;
      for (const auto& body_stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(body_stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call || call->op_->name_ != "tile.assemble") continue;
        if (call->args_.size() < 3) continue;
        const Var* arg0_raw = nullptr;
        if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
        if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
        if (arg0_raw != iter_arg) continue;
        assemble_assign = assign;
        break;
      }

      if (!assemble_assign) return IRMutator::VisitStmt_(op);

      // --- Rewrite: tile.assemble -> tile.store ---

      auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
      auto out_tensor_type = As<TensorType>(out_param_var->GetType());
      INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
          << "Internal error: Out param should be TensorType";

      auto new_iter_arg = std::make_shared<IterArg>(op->iter_args_[0]->name_hint_, out_tensor_type,
                                                    out_param_var, op->iter_args_[0]->span_);

      auto assemble_call = As<Call>(assemble_assign->value_);
      auto& op_registry = OpRegistry::GetInstance();
      auto store_call =
          op_registry.Create("tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg},
                             assemble_assign->value_->span_);
      auto store_result_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                                    assemble_assign->span_);

      std::vector<StmtPtr> new_loop_stmts;
      for (const auto& body_stmt : loop_body_stmts) {
        if (body_stmt.get() == assemble_assign.get()) {
          auto store_assign = MutableCopy(assemble_assign);
          store_assign->var_ = store_result_var;
          store_assign->value_ = store_call;
          new_loop_stmts.push_back(std::move(store_assign));
        } else if (auto y = As<YieldStmt>(body_stmt)) {
          auto new_yield = MutableCopy(y);
          new_yield->value_ = std::vector<ExprPtr>{store_result_var};
          new_loop_stmts.push_back(std::move(new_yield));
        } else {
          new_loop_stmts.push_back(body_stmt);
        }
      }

      auto new_loop_body = SeqStmts::Flatten(std::move(new_loop_stmts), op->body_->span_);
      auto new_return_var = return_var_remap_.at(store_info.store_var);

      auto result = MutableCopy(op);
      result->iter_args_ = std::vector<IterArgPtr>{new_iter_arg};
      result->body_ = new_loop_body;
      result->return_vars_ = std::vector<VarPtr>{new_return_var};
      return result;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      if (dead_create_vars_.count(op->var_.get()) || dead_store_vars_.count(op->var_.get())) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
      }
      return IRMutator::VisitStmt_(op);
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto seq = As<SeqStmts>(visited);
      if (!seq) return visited;
      // Filter out empty SeqStmts children (from deleted statements)
      std::vector<StmtPtr> filtered;
      for (const auto& s : seq->stmts_) {
        if (auto child_seq = As<SeqStmts>(s)) {
          if (child_seq->stmts_.empty()) continue;
        }
        filtered.push_back(s);
      }
      if (filtered.size() == seq->stmts_.size()) return seq;
      return SeqStmts::Flatten(std::move(filtered), seq->span_);
    }

    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      if (return_var_remap_.empty()) return op;
      std::vector<ExprPtr> new_ret_values;
      bool remapped = false;
      for (const auto& v : op->value_) {
        auto var = As<Var>(v);
        if (var) {
          auto remap_it = return_var_remap_.find(var.get());
          if (remap_it != return_var_remap_.end()) {
            new_ret_values.push_back(remap_it->second);
            remapped = true;
            continue;
          }
        }
        new_ret_values.push_back(v);
      }
      if (!remapped) return op;
      auto result = MutableCopy(op);
      result->value_ = std::move(new_ret_values);
      return result;
    }

   private:
    const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
    const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
    const std::unordered_set<const Var*>& dead_create_vars_;
    const std::unordered_set<const Var*>& dead_store_vars_;
    const std::unordered_map<const Var*, VarPtr>& return_var_remap_;
    const FunctionPtr& func_;
  };

  /// Rewrite assemble-loop pattern in an InCore function.
  FunctionPtr RewriteFunction(const FunctionPtr& func) {
    // Pre-scan: collect var definitions and return statement
    BodyScanner scanner;
    scanner.VisitStmt(func->body_);
    const auto& var_def = scanner.var_def();
    const auto& return_stmt = scanner.return_stmt();
    if (!return_stmt) return func;

    // Identify Out params
    std::unordered_map<const Var*, size_t> out_var_to_param_idx;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::Out) {
        out_var_to_param_idx[func->params_[i].get()] = i;
      }
    }
    if (out_var_to_param_idx.empty()) return func;

    // Map return values -> tile.store -> Out params
    using StoreReturnInfo = LoopRewriteMutator::StoreReturnInfo;
    std::vector<StoreReturnInfo> store_returns;

    for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
      auto ret_var = As<Var>(return_stmt->value_[ret_i]);
      if (!ret_var) continue;
      auto def_it = var_def.find(ret_var.get());
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      if (call->args_.size() < 3) continue;
      auto out_tensor = As<Var>(call->args_[2]);
      if (!out_tensor || !out_var_to_param_idx.count(out_tensor.get())) continue;
      store_returns.push_back({ret_var.get(), out_tensor.get(), ret_i});
    }
    if (store_returns.empty()) return func;

    // Map ForStmt return_var -> which store_return it feeds
    std::unordered_map<const Var*, const StoreReturnInfo*> for_return_to_store;
    for (const auto& sr : store_returns) {
      auto def_it = var_def.find(sr.store_var);
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      auto tile_data_var = As<Var>(call->args_[0]);
      if (!tile_data_var) continue;
      for_return_to_store[tile_data_var.get()] = &sr;
    }

    // Pre-compute dead sets by scanning ForStmts for pattern matches.
    // This must happen before the IRMutator pass because dead tile.create
    // statements may appear before the ForStmt they correspond to.
    std::unordered_set<const Var*> dead_create_vars;
    std::unordered_set<const Var*> dead_store_vars;
    std::unordered_map<const Var*, VarPtr> return_var_remap;

    class ForStmtMatchScanner : public IRVisitor {
     public:
      ForStmtMatchScanner(const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                          const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                          const FunctionPtr& func, std::unordered_set<const Var*>& dead_create_vars,
                          std::unordered_set<const Var*>& dead_store_vars,
                          std::unordered_map<const Var*, VarPtr>& return_var_remap)
          : for_return_to_store_(for_return_to_store),
            out_var_to_param_idx_(out_var_to_param_idx),
            func_(func),
            dead_create_vars_(dead_create_vars),
            dead_store_vars_(dead_store_vars),
            return_var_remap_(return_var_remap) {}

      [[nodiscard]] bool matched() const { return matched_; }

     protected:
      void VisitStmt_(const ForStmtPtr& op) override {
        IRVisitor::VisitStmt_(op);
        if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) return;

        auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
        if (fret_it == for_return_to_store_.end()) return;
        const auto& store_info = *fret_it->second;

        auto loop_body_stmts = FlattenToStmts(op->body_);
        auto yield = transform_utils::FindYieldStmt(op->body_);
        if (!yield || yield->value_.size() != 1) return;

        const IterArg* iter_arg = op->iter_args_[0].get();

        AssignStmtPtr assemble_assign;
        for (const auto& body_stmt : loop_body_stmts) {
          auto assign = As<AssignStmt>(body_stmt);
          if (!assign) continue;
          auto call = As<Call>(assign->value_);
          if (!call || call->op_->name_ != "tile.assemble") continue;
          if (call->args_.size() < 3) continue;
          const Var* arg0_raw = nullptr;
          if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
          if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
          if (arg0_raw != iter_arg) continue;
          assemble_assign = assign;
          break;
        }
        if (!assemble_assign) return;

        auto yield_var = As<Var>(yield->value_[0]);
        if (!yield_var || yield_var.get() != assemble_assign->var_.get()) return;

        bool iter_arg_used_elsewhere = false;
        for (const auto& body_stmt : loop_body_stmts) {
          if (body_stmt.get() == assemble_assign.get()) continue;
          if (As<YieldStmt>(body_stmt)) continue;
          if (StmtUsesVar(body_stmt, iter_arg)) {
            iter_arg_used_elsewhere = true;
            break;
          }
        }
        if (iter_arg_used_elsewhere) return;

        // Pattern matched — record dead sets
        matched_ = true;
        auto init_var = As<Var>(op->iter_args_[0]->initValue_);
        if (init_var) dead_create_vars_.insert(init_var.get());
        dead_store_vars_.insert(store_info.store_var);

        auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
        auto out_tensor_type = As<TensorType>(out_param_var->GetType());
        INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
            << "Internal error: Out param should be TensorType";
        auto new_return_var = std::make_shared<Var>(op->return_vars_[0]->name_hint_, out_tensor_type,
                                                    op->return_vars_[0]->span_);
        return_var_remap_[store_info.store_var] = new_return_var;
      }

     private:
      const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
      const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
      const FunctionPtr& func_;
      std::unordered_set<const Var*>& dead_create_vars_;
      std::unordered_set<const Var*>& dead_store_vars_;
      std::unordered_map<const Var*, VarPtr>& return_var_remap_;
      bool matched_ = false;
    };

    ForStmtMatchScanner match_scanner(for_return_to_store, out_var_to_param_idx, func, dead_create_vars,
                                      dead_store_vars, return_var_remap);
    match_scanner.VisitStmt(func->body_);
    if (!match_scanner.matched()) return func;

    // Apply the IRMutator using pre-computed dead sets
    LoopRewriteMutator mutator(out_var_to_param_idx, for_return_to_store, dead_create_vars, dead_store_vars,
                               return_var_remap, func);
    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  }
};

// ============================================================================
// Pattern 4: SliceInputStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.slice(parent, size, offset)
// where the slice result is passed as an In argument to an InCore call.
// Records the parent tensor's shape. Then updates the InCore function's
// In param TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class SliceInputStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto input_shapes = Analyze(program, incore_names);
    if (input_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, input_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  // func_name -> { param_index -> parent_shape }
  using InputShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  static bool ShapesMatch(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      auto ca = As<ConstInt>(a[i]);
      auto cb = As<ConstInt>(b[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  class SliceAnalyzer : public IRVisitor {
   public:
    SliceAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const InputShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;

      // Track tensor.slice(parent, size, offset) results
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.slice" &&
          call->args_.size() >= 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        if (parent_var) {
          auto parent_tensor_type = As<TensorType>(parent_var->GetType());
          if (parent_tensor_type) {
            var_to_parent_shape_[op->var_.get()] = parent_tensor_type->shape_;
          }
        }
        return;
      }

      // Check InCore calls: map sliced In arguments to parent shapes
      auto fname = GetCallFuncName(call);
      if (!incore_names_.count(fname)) return;

      auto incore_func = FindFunction(program_, fname);
      if (!incore_func) return;

      for (size_t i = 0; i < call->args_.size() && i < incore_func->param_directions_.size(); ++i) {
        if (incore_func->param_directions_[i] != ParamDirection::In) continue;

        auto arg_var = AsVarLike(call->args_[i]);
        if (!arg_var) continue;

        auto it = var_to_parent_shape_.find(arg_var.get());
        if (it == var_to_parent_shape_.end()) continue;

        auto conflict_key = fname + ":" + std::to_string(i);
        if (conflicted_.count(conflict_key)) continue;

        auto& entry = result_[fname][i];
        if (entry.empty()) {
          entry = it->second;
        } else if (!ShapesMatch(entry, it->second)) {
          conflicted_.insert(conflict_key);
          entry.clear();
        }
      }
    }

   private:
    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> var_to_parent_shape_;
    std::unordered_set<std::string> conflicted_;
    InputShapeMap result_;
  };

  InputShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    SliceAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  class InParamSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const InputShapeMap& input_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto is_it = input_shapes.find(func->name_);
      if (is_it == input_shapes.end()) continue;
      const auto& param_idx_to_shape = is_it->second;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& [param_idx, parent_shape] : param_idx_to_shape) {
        if (parent_shape.empty()) continue;  // conflicted or not from slice
        if (param_idx >= func->params_.size()) continue;

        auto full_strides = ComputeRowMajorStrides(parent_shape);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[param_idx]->GetType());
        if (!tensor_type) continue;

        // Skip params that already have explicit strides
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) continue;

        size_t in_rank = tensor_type->shape_.size();
        if (in_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(in_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[param_idx]->name_hint_, new_type,
                                               func->params_[param_idx]->span_);

        changed = true;
        new_params[param_idx] = new_param;
      }

      if (!changed) continue;

      InParamSubstitutionMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 5: OutWindowExternalizer
//
// Cross-function rewrite for Scenario A:
//   orch:   out = self.kernel(..., parent_out)
//
// becomes
//   - original kernel preserved
//   - cloned kernel kernel__windowed rewrites its final tile.store offset to [0, ...]
//     and narrows its single Out/return tensor to the written window shape
//   - orch call rewritten to:
//         out__window = slice(parent_out, window_shape, offsets)
//         out__window_res = self.kernel__windowed(..., out__window)
//         out = assemble(parent_out, out__window_res, offsets)
//
// The call itself now depends on the sliced window tensor rather than the
// whole parent buffer, while the trailing tensor.assemble preserves the
// original full-tensor SSA carry for later users.
// ============================================================================

class OutWindowExternalizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto analyses = Analyze(program, incore_names);
    if (analyses.empty()) return program;

    std::unordered_map<std::string, FunctionPtr> cloned_funcs;
    for (const auto& [func_name, analysis] : analyses) {
      auto callee = program->GetFunction(func_name);
      if (!callee) continue;
      auto cloned = RewriteIncore(program, callee, analysis);
      if (!cloned) continue;
      cloned_funcs.emplace(func_name, cloned);
    }
    if (cloned_funcs.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    new_functions.reserve(program->functions_.size() + cloned_funcs.size());
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
      auto clone_it = cloned_funcs.find(func->name_);
      if (clone_it != cloned_funcs.end()) {
        new_functions.push_back(clone_it->second);
      }
    }

    bool changed = false;
    for (auto& func : new_functions) {
      if (!func || func->func_type_ != FunctionType::Orchestration) continue;
      OrchRewriter rewriter(program, analyses, cloned_funcs);
      auto new_body = rewriter.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->comm_groups_, program->name_, program->span_);
  }

 private:
  struct IncoreRewriteAnalysis {
    enum class Kind {
      FinalStore,
      ChunkInnerLoop,
    };

    Kind kind = Kind::FinalStore;
    size_t out_param_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_store_offsets;
  };

  using AnalysisMap = std::unordered_map<std::string, IncoreRewriteAnalysis>;

  class StoreOffsetLocalizer : public IRMutator {
   public:
    StoreOffsetLocalizer(const Var* old_out, const VarPtr& new_out, const std::vector<ExprPtr>& old_offsets,
                         const std::vector<ExprPtr>& new_offsets, const TypePtr& new_store_type)
        : old_out_(old_out),
          new_out_(new_out),
          old_offsets_(old_offsets),
          new_offsets_(new_offsets),
          new_store_type_(new_store_type) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto remap_it = result_var_remap_.find(op.get());
      if (remap_it != result_var_remap_.end()) return remap_it->second;
      if (op.get() == old_out_) return new_out_;
      return IRMutator::VisitExpr_(op);
    }

    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == old_out_) return new_out_;
      return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited_value = VisitExpr(op->value_);
      auto assign = MutableCopy(op);
      assign->value_ = visited_value;
      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tile.store" || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || out_var.get() != new_out_.get()) return assign;

      auto offsets = As<MakeTuple>(call->args_[1]);
      if (!offsets || !AreExprVectorsEqual(offsets->elements_, old_offsets_)) return assign;

      auto new_offset_tuple = std::make_shared<MakeTuple>(new_offsets_, offsets->span_);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = new_offset_tuple;
      new_args[2] = new_out_;
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_store_type_, call->span_);

      auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_store_type_, assign->var_->span_);
      result_var_remap_[assign->var_.get()] = new_result_var;
      assign->var_ = new_result_var;
      assign->value_ = new_call;
      return assign;
    }

   private:
    const Var* old_out_;
    VarPtr new_out_;
    const std::vector<ExprPtr>& old_offsets_;
    const std::vector<ExprPtr>& new_offsets_;
    TypePtr new_store_type_;
    std::unordered_map<const Var*, VarPtr> result_var_remap_;
  };

  class OrchRewriter : public IRMutator {
   public:
    OrchRewriter(const ProgramPtr& program, const AnalysisMap& analyses,
                 const std::unordered_map<std::string, FunctionPtr>& cloned_funcs)
        : program_(program), analyses_(analyses), cloned_funcs_(cloned_funcs) {}

   protected:
    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size());
      bool changed = false;

      for (const auto& stmt : op->stmts_) {
        auto call_assign = As<AssignStmt>(stmt);
        auto bundle = call_assign ? TryRewriteDirectOutCall(call_assign) : std::nullopt;
        if (bundle.has_value()) {
          changed = true;
          new_stmts.push_back(VisitStmt(bundle->slice_stmt));
          new_stmts.push_back(VisitStmt(bundle->call_stmt));
          new_stmts.push_back(VisitStmt(bundle->assemble_stmt));
          continue;
        }

        auto visited = VisitStmt(stmt);
        changed = changed || visited.get() != stmt.get();
        new_stmts.push_back(visited);
      }

      if (!changed) return op;
      return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }

   private:
    struct RewriteBundle {
      StmtPtr slice_stmt;
      StmtPtr call_stmt;
      StmtPtr assemble_stmt;
    };

    std::optional<RewriteBundle> TryRewriteDirectOutCall(const AssignStmtPtr& call_assign) {
      auto call = As<Call>(call_assign->value_);
      if (!call) return std::nullopt;

      auto callee_name = GetCallFuncName(call);
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return std::nullopt;
      const auto& analysis = analysis_it->second;
      if (analysis.out_param_index >= call->args_.size()) return std::nullopt;

      auto out_arg = AsVarLike(call->args_[analysis.out_param_index]);
      if (!out_arg) return std::nullopt;
      auto clone_it = cloned_funcs_.find(callee_name);
      if (clone_it == cloned_funcs_.end()) return std::nullopt;
      auto cloned_func = clone_it->second;
      auto cloned_out_type = As<TensorType>(cloned_func->params_[analysis.out_param_index]->GetType());
      if (!cloned_out_type) return std::nullopt;

      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      auto original_func = program_ ? program_->GetFunction(callee_name) : nullptr;
      if (!original_func) return std::nullopt;
      callsite_subst.reserve(original_func->params_.size());
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }

      std::vector<ExprPtr> shape_exprs;
      shape_exprs.reserve(analysis.window_shape.size());
      for (const auto& dim : analysis.window_shape) {
        shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
      }
      auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

      std::vector<ExprPtr> offset_exprs;
      offset_exprs.reserve(analysis.callsite_offsets.size());
      for (const auto& offset : analysis.callsite_offsets) {
        offset_exprs.push_back(transform_utils::Substitute(offset, callsite_subst));
      }
      auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

      ExprPtr parent_expr = VisitExpr(call->args_[analysis.out_param_index]);
      auto slice_call = OpRegistry::GetInstance().Create("tensor.slice", {parent_expr, shape_tuple, offset_tuple},
                                                         call_assign->span_);
      auto slice_var = std::make_shared<Var>(out_arg->name_hint_ + "__window", slice_call->GetType(), out_arg->span_);
      auto slice_stmt = std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_);

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      for (size_t i = 0; i < call->args_.size(); ++i) {
        new_args.push_back(i == analysis.out_param_index ? ExprPtr(slice_var) : VisitExpr(call->args_[i]));
      }

      std::vector<VarPtr> rewritten_user_deps;
      bool needs_user_dep_rewrite = false;
      for (const auto& [k, v] : call->attrs_) {
        if (k != kAttrUserManualDepEdges) continue;
        const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
        if (!user_deps) break;
        rewritten_user_deps.reserve(user_deps->size());
        for (const auto& dep : *user_deps) {
          if (dep && dep.get() == out_arg.get()) {
            rewritten_user_deps.push_back(slice_var);
            needs_user_dep_rewrite = true;
          } else {
            rewritten_user_deps.push_back(dep);
          }
        }
        break;
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      TypePtr new_return_type = cloned_func->return_types_.size() == 1
                                    ? cloned_func->return_types_[0]
                                    : std::make_shared<TupleType>(cloned_func->return_types_);
      auto new_attrs = call->attrs_;
      if (needs_user_dep_rewrite) {
        new_attrs = WithUserManualDepEdgesAttr(std::move(new_attrs), std::move(rewritten_user_deps));
      }
      auto new_call =
          std::make_shared<Call>(cloned_gvar, new_args, call->kwargs_, new_attrs, new_return_type, call->span_);
      auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed", new_return_type,
                                                  call_assign->var_->span_);
      auto call_stmt = std::make_shared<AssignStmt>(tmp_result_var, new_call, call_assign->span_);

      std::vector<ExprPtr> assemble_args = {parent_expr, tmp_result_var, offset_tuple};
      auto new_assemble_call = OpRegistry::GetInstance().Create("tensor.assemble", assemble_args, call_assign->span_);
      auto assemble_stmt_new = std::make_shared<AssignStmt>(call_assign->var_, new_assemble_call, call_assign->span_);

      RewriteBundle bundle;
      bundle.slice_stmt = slice_stmt;
      bundle.call_stmt = call_stmt;
      bundle.assemble_stmt = assemble_stmt_new;
      return bundle;
    }

    ProgramPtr program_;
    const AnalysisMap& analyses_;
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs_;
  };

  struct FinalStoreInfo {
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
  };

  struct ChunkInnerLoopInfo {
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> local_offsets;
  };

  class ChunkLoopBodyLocalizer : public IRMutator {
   public:
    ChunkLoopBodyLocalizer(const IterArg* old_iter_arg, const IterArgPtr& new_iter_arg,
                           const std::vector<ExprPtr>& new_offsets, const TypePtr& new_store_type)
        : old_iter_arg_(old_iter_arg),
          new_iter_arg_(new_iter_arg),
          new_offsets_(new_offsets),
          new_store_type_(new_store_type) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto remap_it = result_var_remap_.find(op.get());
      if (remap_it != result_var_remap_.end()) return remap_it->second;
      return IRMutator::VisitExpr_(op);
    }

    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == old_iter_arg_) return new_iter_arg_;
      return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited_value = VisitExpr(op->value_);
      auto assign = MutableCopy(op);
      assign->value_ = visited_value;
      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tile.store" || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || out_var.get() != new_iter_arg_.get()) return assign;

      auto new_offset_tuple = std::make_shared<MakeTuple>(new_offsets_, call->span_);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = new_offset_tuple;
      new_args[2] = new_iter_arg_;
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_store_type_, call->span_);

      auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_store_type_, assign->var_->span_);
      result_var_remap_[assign->var_.get()] = new_result_var;
      assign->var_ = new_result_var;
      assign->value_ = new_call;
      return assign;
    }

   private:
    const IterArg* old_iter_arg_;
    IterArgPtr new_iter_arg_;
    const std::vector<ExprPtr>& new_offsets_;
    TypePtr new_store_type_;
    std::unordered_map<const Var*, VarPtr> result_var_remap_;
  };

  static std::optional<int64_t> GetConstIntValue(const ExprPtr& expr) {
    auto ci = As<ConstInt>(expr);
    if (!ci) return std::nullopt;
    return ci->value_;
  }

  static std::optional<ExprPtr> SimplifyWithLoopBound(const ExprPtr& expr, const VarPtr& loop_var, int64_t value) {
    if (!expr) return std::nullopt;
    arith::Analyzer analyzer;
    analyzer.Bind(loop_var, value, value + 1);
    return analyzer.Simplify(expr);
  }

  static std::optional<ExprPtr> ExpandLoopLocalExpr(
      const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
    if (!expr) return std::nullopt;
    return transform_utils::Substitute(expr, scalar_defs);
  }

  static std::optional<ChunkInnerLoopInfo> AnalyzeChunkInnerLoop(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    if (body_stmts.size() != 2) return std::nullopt;

    auto loop = As<ForStmt>(body_stmts[0]);
    auto ret_stmt = As<ReturnStmt>(body_stmts[1]);
    if (!loop || !ret_stmt || loop->iter_args_.size() != 1 || loop->return_vars_.size() != 1 ||
        ret_stmt->value_.size() != 1) {
      return std::nullopt;
    }
    if (loop->iter_args_[0]->initValue_.get() != func->params_[out_param_index].get()) return std::nullopt;
    if (loop->GetAttr<LoopOrigin>("loop_origin", LoopOrigin::Original) != LoopOrigin::ChunkInner) {
      return std::nullopt;
    }

    auto trip_count = GetConstIntValue(loop->stop_);
    auto start = GetConstIntValue(loop->start_);
    auto step = GetConstIntValue(loop->step_);
    if (!trip_count.has_value() || !start.has_value() || !step.has_value()) return std::nullopt;
    if (*start != 0 || *step != 1 || *trip_count <= 0) return std::nullopt;

    auto loop_body_stmts = FlattenToStmts(loop->body_);
    AssignStmtPtr store_assign;
    YieldStmtPtr yield_stmt;
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    for (const auto& stmt : loop_body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        auto call = As<Call>(assign->value_);
        if (call && call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
          auto out_arg = AsVarLike(call->args_[2]);
          if (!out_arg || out_arg.get() != loop->iter_args_[0].get()) return std::nullopt;
          if (store_assign) return std::nullopt;
          store_assign = assign;
          continue;
        }
        if (As<ScalarType>(assign->var_->GetType())) {
          scalar_defs[assign->var_.get()] = assign->value_;
        }
        continue;
      }
      if (auto yield = As<YieldStmt>(stmt)) {
        if (yield_stmt || yield->value_.size() != 1) return std::nullopt;
        yield_stmt = yield;
      }
    }

    if (!store_assign || !yield_stmt) return std::nullopt;
    auto yielded = AsVarLike(yield_stmt->value_[0]);
    auto returned = AsVarLike(ret_stmt->value_[0]);
    if (!yielded || !returned || yielded.get() != store_assign->var_.get() ||
        returned.get() != loop->return_vars_[0].get()) {
      return std::nullopt;
    }

    if (!As<TensorType>(loop->iter_args_[0]->GetType()) || !As<TensorType>(loop->return_vars_[0]->GetType())) {
      return std::nullopt;
    }

    size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[out_param_index].get());
    if (total_out_refs != 1) return std::nullopt;

    size_t total_iter_refs = CountVarRefsInStmt(loop->body_, loop->iter_args_[0].get());
    size_t store_iter_refs = CountVarRefsInStmt(store_assign, loop->iter_args_[0].get());
    if (total_iter_refs != store_iter_refs) return std::nullopt;

    auto store_call = As<Call>(store_assign->value_);
    auto offsets = As<MakeTuple>(store_call->args_[1]);
    auto tile_type = As<TileType>(store_call->args_[0]->GetType());
    auto out_tensor_type = As<TensorType>(func->params_[out_param_index]->GetType());
    if (!offsets || !tile_type || !out_tensor_type) return std::nullopt;
    if (offsets->elements_.size() != tile_type->shape_.size() ||
        offsets->elements_.size() != out_tensor_type->shape_.size()) {
      return std::nullopt;
    }

    std::unordered_set<const Var*> allowed;
    for (const auto& param : func->params_) allowed.insert(param.get());
    allowed.insert(loop->loop_var_.get());

    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> local_offsets;
    std::vector<ExprPtr> window_shape;
    base_offsets.reserve(offsets->elements_.size());
    local_offsets.reserve(offsets->elements_.size());
    window_shape.reserve(offsets->elements_.size());

    for (size_t i = 0; i < offsets->elements_.size(); ++i) {
      auto expanded = ExpandLoopLocalExpr(offsets->elements_[i], scalar_defs);
      if (!expanded.has_value()) return std::nullopt;
      if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return std::nullopt;

      auto min_offset = SimplifyWithLoopBound(*expanded, loop->loop_var_, 0);
      auto max_offset = SimplifyWithLoopBound(*expanded, loop->loop_var_, *trip_count - 1);
      if (!min_offset.has_value() || !max_offset.has_value()) return std::nullopt;

      auto span_expr = arith::Analyzer().Simplify(
          MakeAdd(MakeSub(*max_offset, *min_offset, func->span_), tile_type->shape_[i], func->span_));
      auto span_ci = As<ConstInt>(span_expr);
      if (!span_ci || span_ci->value_ <= 0) return std::nullopt;

      base_offsets.push_back(*min_offset);
      local_offsets.push_back(arith::Analyzer().Simplify(
          MakeSub(offsets->elements_[i], *min_offset, offsets->elements_[i]->span_)));
      window_shape.push_back(std::make_shared<ConstInt>(span_ci->value_, DataType::INDEX, func->span_));
    }

    if (AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
      return std::nullopt;
    }

    ChunkInnerLoopInfo info;
    info.window_shape = std::move(window_shape);
    info.base_offsets = std::move(base_offsets);
    info.local_offsets = std::move(local_offsets);
    return info;
  }

  static std::optional<FinalStoreInfo> AnalyzeFinalStore(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    std::unordered_map<const Var*, AssignStmtPtr> var_defs;
    for (const auto& stmt : body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) var_defs[assign->var_.get()] = assign;
    }

    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : body_stmts) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt || ret_stmt->value_.size() != 1) return std::nullopt;

    auto ret_var = AsVarLike(ret_stmt->value_[0]);
    if (!ret_var) return std::nullopt;
    auto def_it = var_defs.find(ret_var.get());
    if (def_it == var_defs.end()) return std::nullopt;
    auto store_call = As<Call>(def_it->second->value_);
    if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.size() < 3) return std::nullopt;

    auto out_target = AsVarLike(store_call->args_[2]);
    if (!out_target || out_target.get() != func->params_[out_param_index].get()) return std::nullopt;
    auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
    if (!offset_tuple) return std::nullopt;

    size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[out_param_index].get());
    size_t store_out_refs = CountVarRefsInStmt(def_it->second, func->params_[out_param_index].get());
    if (total_out_refs != store_out_refs) return std::nullopt;

    auto tile_type = As<TileType>(store_call->args_[0]->GetType());
    if (!tile_type) return std::nullopt;

    FinalStoreInfo info;
    info.window_shape = tile_type->shape_;
    info.offsets = offset_tuple->elements_;
    return info;
  }

  AnalysisMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    AnalysisMap analyses;
    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) continue;
      auto out_index = GetSingleOutParamIndex(func);
      if (!out_index.has_value()) continue;
      auto store_info = AnalyzeFinalStore(func, *out_index);
      if (!store_info.has_value()) {
        auto chunk_info = AnalyzeChunkInnerLoop(func, *out_index);
        if (!chunk_info.has_value()) continue;

        IncoreRewriteAnalysis analysis;
        analysis.kind = IncoreRewriteAnalysis::Kind::ChunkInnerLoop;
        analysis.out_param_index = *out_index;
        analysis.window_shape = chunk_info->window_shape;
        analysis.callsite_offsets = chunk_info->base_offsets;
        analysis.local_store_offsets = chunk_info->local_offsets;
        analyses.emplace(func->name_, std::move(analysis));
        continue;
      }
      auto out_tensor_type = As<TensorType>(func->params_[*out_index]->GetType());
      if (!out_tensor_type) continue;

      if (AreExprVectorsEqual(store_info->window_shape, out_tensor_type->shape_) &&
          IsAllZeroOffsets(store_info->offsets)) {
        continue;
      }

      std::unordered_set<const Var*> allowed_params;
      for (const auto& param : func->params_) allowed_params.insert(param.get());
      bool exprs_ok = true;
      for (const auto& expr : store_info->window_shape) {
        if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
          exprs_ok = false;
          break;
        }
      }
      if (!exprs_ok) continue;
      for (const auto& expr : store_info->offsets) {
        if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
          exprs_ok = false;
          break;
        }
      }
      if (!exprs_ok) continue;

      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(store_info->offsets.size());
      for (size_t i = 0; i < store_info->offsets.size(); ++i) {
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }

      IncoreRewriteAnalysis analysis;
      analysis.kind = IncoreRewriteAnalysis::Kind::FinalStore;
      analysis.out_param_index = *out_index;
      analysis.window_shape = store_info->window_shape;
      analysis.callsite_offsets = store_info->offsets;
      analysis.local_store_offsets = std::move(local_zero_offsets);
      analyses.emplace(func->name_, std::move(analysis));
    }
    return analyses;
  }

  FunctionPtr RewriteIncore(const ProgramPtr& program, const FunctionPtr& func, const IncoreRewriteAnalysis& analysis) {
    auto out_tensor_type = As<TensorType>(func->params_[analysis.out_param_index]->GetType());
    if (!out_tensor_type || func->return_types_.size() != 1) return nullptr;

    std::optional<TensorView> new_view = std::nullopt;
    if (out_tensor_type->tensor_view_.has_value()) {
      new_view = out_tensor_type->tensor_view_;
      if (new_view->stride.empty()) {
        if (new_view->layout == TensorLayout::NZ) return nullptr;
        new_view->stride =
            tensor_view_semantics::BuildLogicalStridesFromLayout(out_tensor_type->shape_, new_view->layout);
      }
      if (!new_view->valid_shape.empty()) new_view->valid_shape = analysis.window_shape;
    } else {
      new_view = tensor_view_semantics::CanonicalizeView(out_tensor_type->shape_, TensorLayout::ND);
    }

    auto new_out_type =
        std::make_shared<TensorType>(analysis.window_shape, out_tensor_type->dtype_, out_tensor_type->memref_, new_view);

    auto cloned_name = MakeUniqueFunctionName(program, func->name_ + "__windowed");

    std::vector<VarPtr> new_params;
    new_params.reserve(func->params_.size());

    std::unordered_map<const Var*, ExprPtr> seed;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      auto param_type = (i == analysis.out_param_index) ? TypePtr(new_out_type) : func->params_[i]->GetType();
      auto new_param =
          std::make_shared<Var>(func->params_[i]->name_hint_, param_type, func->params_[i]->span_);
      new_params.push_back(new_param);
      seed[func->params_[i].get()] = new_param;
    }

    auto cloned = DeepClone(func->body_, seed);
    StmtPtr new_body;
    if (analysis.kind == IncoreRewriteAnalysis::Kind::ChunkInnerLoop) {
      auto body_stmts = FlattenToStmts(cloned.cloned_body);
      if (body_stmts.size() != 2) return nullptr;
      auto loop = As<ForStmt>(body_stmts[0]);
      auto ret_stmt = As<ReturnStmt>(body_stmts[1]);
      if (!loop || !ret_stmt || loop->iter_args_.size() != 1 || loop->return_vars_.size() != 1) return nullptr;

      auto new_iter_arg = std::make_shared<IterArg>(loop->iter_args_[0]->name_hint_, new_out_type,
                                                    new_params[analysis.out_param_index], loop->iter_args_[0]->span_);
      auto new_return_var =
          std::make_shared<Var>(loop->return_vars_[0]->name_hint_, new_out_type, loop->return_vars_[0]->span_);

      std::unordered_map<const Var*, ExprPtr> local_offset_subst = seed;
      for (const auto& [old_var, new_var] : cloned.var_map) {
        local_offset_subst[old_var] = new_var;
      }
      std::vector<ExprPtr> cloned_local_offsets;
      cloned_local_offsets.reserve(analysis.local_store_offsets.size());
      for (const auto& offset : analysis.local_store_offsets) {
        cloned_local_offsets.push_back(transform_utils::Substitute(offset, local_offset_subst));
      }

      ChunkLoopBodyLocalizer localizer(loop->iter_args_[0].get(), new_iter_arg, cloned_local_offsets, new_out_type);
      auto new_loop_body = localizer.VisitStmt(loop->body_);
      auto new_loop = std::make_shared<ForStmt>(loop->loop_var_, loop->start_, loop->stop_, loop->step_,
                                                std::vector<IterArgPtr>{new_iter_arg}, new_loop_body,
                                                std::vector<VarPtr>{new_return_var}, loop->span_, loop->kind_,
                                                loop->chunk_config_, loop->attrs_);
      auto new_return = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{new_return_var}, ret_stmt->span_);
      new_body = SeqStmts::Flatten(std::vector<StmtPtr>{new_loop, new_return}, cloned.cloned_body->span_);
    } else {
      std::vector<ExprPtr> cloned_callsite_offsets;
      cloned_callsite_offsets.reserve(analysis.callsite_offsets.size());
      for (const auto& offset : analysis.callsite_offsets) {
        cloned_callsite_offsets.push_back(transform_utils::Substitute(offset, seed));
      }

      StoreOffsetLocalizer localizer(func->params_[analysis.out_param_index].get(),
                                     new_params[analysis.out_param_index],
                                     cloned_callsite_offsets, analysis.local_store_offsets, new_out_type);
      new_body = localizer.VisitStmt(cloned.cloned_body);
    }
    std::vector<TypePtr> new_return_types = {new_out_type};

    return std::make_shared<Function>(cloned_name, new_params, func->param_directions_, new_return_types, new_body,
                                      func->span_, func->func_type_, func->level_, func->role_, func->attrs_);
  }
};

}  // namespace

// ============================================================================
// Pass entry point
// ============================================================================

namespace pass {

Pass OptimizeOrchTensors() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    auto* ctx = PassContext::Current();
    const bool enable_out_window_rewrite = ctx == nullptr || ctx->GetEnableOutWindowRewrite();

    // Collect InCore function names
    std::unordered_set<std::string> incore_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        incore_names.insert(func->name_);
      }
    }
    if (incore_names.empty()) return program;

    // Pattern 1: Iter-arg reuse (may remove Out params)
    auto p1 = IterArgReuseOptimizer().Run(program, incore_names);

    // Pattern 2: Assemble parent strides (sees Pattern 1 results)
    auto p2 = AssembleParentStridesOptimizer().Run(p1, incore_names);

    // Pattern 3: Assemble-loop rewrite (sees Pattern 2 results)
    auto p3 = AssembleLoopRewriter().Run(p2, incore_names);

    // Pattern 4: Slice input strides (propagate parent strides to In params)
    auto p4 = SliceInputStridesOptimizer().Run(p3, incore_names);

    if (!enable_out_window_rewrite) return p4;

    // Pattern 5: Externalize Out windows through orch create+assemble wrappers
    return OutWindowExternalizer().Run(p4, incore_names);
  };

  return CreateProgramPass(pass_func, "OptimizeOrchTensors", kOptimizeOrchTensorsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
