#include "transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"


using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_LOOPSPLIT
#include "transforms/Passes.h.inc"

namespace {

struct LoopSplitPass
    : public impl::LoopSplitBase<LoopSplitPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

static bool isArithOrMath(Operation &op) {
  llvm::StringRef dialect = op.getName().getDialectNamespace();
  return (dialect == "arith" || dialect == "math");
}

static void mapResults(Operation *oldOp, Operation *newOp,
                       BlockAndValueMapping &mapping) {
  assert(oldOp->getNumResults() == newOp->getNumResults());

  for (auto [oldResult, newResult] :
       llvm::zip(oldOp->getResults(), newOp->getResults())) {
    mapping.map(oldResult, newResult);
  }
}

static SmallVector<AffineLoadOp, 4>
getLoadsUsedBy(Operation *op,
               ArrayRef<AffineLoadOp> inputLoads) {
  SmallVector<AffineLoadOp, 4> result;

  for (Value operand : op->getOperands()) {
    Operation *definingOp = operand.getDefiningOp();
    auto load = dyn_cast_or_null<AffineLoadOp>(definingOp);

    if (!load)
      continue;

    if (llvm::is_contained(inputLoads, load))
      result.push_back(load);
  }

  return result;
}

static bool sameAffineAccess(AffineLoadOp load, AffineStoreOp store) {
  if (load.getMemRef() != store.getMemref())
    return false;

  if (load.getAffineMap() != store.getAffineMap())
    return false;

  return llvm::equal(
      load.getMapOperands(),
      store.getMapOperands());
}

static bool dependsOn(Value value, Value target) {
  if (value == target)
    return true;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  for (Value operand : def->getOperands()) {
    if (dependsOn(operand, target))
      return true;
  }

  return false;
}

static bool isReadModifyWriteReduction(
    ArrayRef<AffineLoadOp> inputLoads,
    AffineStoreOp outputStore) {
  Value storedValue = outputStore.getValueToStore();

  for (AffineLoadOp load : inputLoads) {
    if (!sameAffineAccess(load, outputStore))
      continue;

    if (dependsOn(storedValue, load.getResult()))
      return true;
  }

  return false;
}

void splitLoop(AffineForOp loop) {
  SmallVector<AffineLoadOp, 4> inputLoads;
  AffineStoreOp outputStore;
  SmallVector<Operation *> computeOps;

  // Analyze the original body before mutating it. We expect the body to
  // contain exactly one input affine.load, one output affine.store, and one
  // or more arith/math operations that form a linear SSA chain.
  for (Operation &op : loop.getBody()->getOperations()) {
    if (isa<AffineYieldOp>(op))
      continue;

    if (auto load = dyn_cast<AffineLoadOp>(&op)) {
      inputLoads.push_back(load);
      continue;
    }

    if (auto store = dyn_cast<AffineStoreOp>(&op)) {
      if (outputStore) {
        loop.emitError(
            "splitLoop supports exactly one output affine.store");
        return;
      }

      outputStore = store;
      continue;
    }

    if (isArithOrMath(op)) {
      if (op.getNumResults() != 1) {
        op.emitError(
            "splitLoop supports only single-result arith/math operations");
        return;
      }

      computeOps.push_back(&op);
      continue;
    }

    return;
  }

  if (!outputStore) {
    loop.emitError("splitLoop could not find an output affine.store");
    return;
  }

  if (computeOps.empty()) {
    loop.emitError("splitLoop could not find arith/math operations");
    return;
  }

  // Avoid splitting reductions by rejecting any loop that performs a
  // read-modify-write of the same output element.
  if (isReadModifyWriteReduction(inputLoads, outputStore)) {
    return;
  }

  // Nothing to distribute. This preserves one-operation loops, including
  // reduction-like loops such as max = max(max, input[i]).
  if (computeOps.size() <= 1) {
    return;
  }

  // Require the final affine.store to consume the final computation result.
  if (outputStore.getValueToStore() != computeOps.back()->getResult(0)) {
    loop.emitError(
        "splitLoop requires the final affine.store to store the result "
        "of the final arith/math operation");
    return;
  }

  // New loops are inserted immediately before the original loop.
  OpBuilder outerBuilder(loop);
  outerBuilder.setInsertionPoint(loop);

  auto createEquivalentLoop = [&]() -> AffineForOp {
    SmallVector<Value, 4> lowerBoundOperands(
        loop.getLowerBoundOperands().begin(),
        loop.getLowerBoundOperands().end());

    SmallVector<Value, 4> upperBoundOperands(
        loop.getUpperBoundOperands().begin(),
        loop.getUpperBoundOperands().end());

    return outerBuilder.create<AffineForOp>(
        loop.getLoc(),
        lowerBoundOperands,
        loop.getLowerBoundMap(),
        upperBoundOperands,
        loop.getUpperBoundMap(),
        loop.getStep());
  };

  // Create one loop per arith/math operation.
  for (unsigned stage = 0; stage < computeOps.size(); ++stage) {
    Operation *oldComputeOp = computeOps[stage];
    AffineForOp newLoop = createEquivalentLoop();

    OpBuilder bodyBuilder(newLoop.getContext());
    bodyBuilder.setInsertionPoint(newLoop.getBody()->getTerminator());

    BlockAndValueMapping mapping;
    mapping.map(loop.getInductionVar(), newLoop.getInductionVar());

    Value oldPreviousValue;

    if (stage == 0) {
      // Stage 0 clones exactly the affine.loads used by the first
      // arithmetic/math operation.
      SmallVector<AffineLoadOp, 4> loads =
          getLoadsUsedBy(oldComputeOp, inputLoads);

      for (AffineLoadOp oldLoad : loads) {
        Operation *newLoad =
            bodyBuilder.clone(*oldLoad, mapping);

        mapResults(oldLoad, newLoad, mapping);
      }
    } else {
      // Later stages consume the result materialized by the previous stage.
      Operation *oldPreviousOp = computeOps[stage - 1];
      oldPreviousValue = oldPreviousOp->getResult(0);

      SmallVector<Value, 4> remappedIndices;
      for (Value operand : outputStore.getMapOperands()) {
        remappedIndices.push_back(
            mapping.lookupOrDefault(operand));
      }

      Value outputMemref =
          mapping.lookupOrDefault(outputStore.getMemref());

      auto reload = bodyBuilder.create<AffineLoadOp>(
          outputStore.getLoc(),
          outputMemref,
          outputStore.getAffineMap(),
          remappedIndices);

      mapping.map(oldPreviousValue, reload.getResult());

      // If the current operation has additional direct affine.load operands,
      // clone only those loads too.
      SmallVector<AffineLoadOp, 4> additionalLoads =
          getLoadsUsedBy(oldComputeOp, inputLoads);

      for (AffineLoadOp oldLoad : additionalLoads) {
        if (mapping.contains(oldLoad.getResult()))
          continue;

        Operation *newLoad =
            bodyBuilder.clone(*oldLoad, mapping);

        mapResults(oldLoad, newLoad, mapping);
      }
    }

    // Clone the current arithmetic/math operation.
    Operation *newComputeOp =
        bodyBuilder.clone(*oldComputeOp, mapping);

    mapResults(oldComputeOp, newComputeOp, mapping);

    // The original store consumes the result of the final original operation.
    // Redirect that operand to the current stage result.
    mapping.map(
        outputStore.getValueToStore(),
        newComputeOp->getResult(0));

    // Clone the original store. The stored value now refers to the current
    // stage's result, while the output memref and affine indices are preserved.
    Operation *newStoreOp =
        bodyBuilder.clone(*outputStore, mapping);

    auto newStore = cast<AffineStoreOp>(newStoreOp);

    if (newStore.getValueToStore() != newComputeOp->getResult(0)) {
      newStore.emitError(
          "internal error: stage result was not mapped to affine.store");
      return;
    }
  }

  // Every original operation was cloned into the new loop sequence.
  loop.erase();
}

void LoopSplitPass::runOnOperation() {
  auto M = getOperation();

  llvm::SmallVector<AffineForOp> worklist;
  M.walk([&](AffineForOp forOp) { worklist.push_back(forOp); });

  for (auto loop : worklist) {
    splitLoop(loop);
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopSplitPass() {
  return std::make_unique<LoopSplitPass>();
}
