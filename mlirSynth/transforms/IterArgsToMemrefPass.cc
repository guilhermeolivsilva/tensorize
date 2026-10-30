#include "transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"


using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_ITERARGSTOMEMREF
#include "transforms/Passes.h.inc"

namespace {

struct IterArgsToMemrefPass
    : public impl::IterArgsToMemrefBase<IterArgsToMemrefPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir


static void convertIterArgsToMemref(AffineForOp forOp) {
  if (forOp.getNumIterOperands() == 0)
    return;

  OpBuilder b(forOp);
  Location loc = forOp.getLoc();
  AffineMap zeroMap = b.getConstantAffineMap(0);

  // 1. One alloca per iter_arg, initialised with the init operand.
  SmallVector<Value> allocas;
  for (Value init : forOp.getIterOperands()) {
    auto mrTy = MemRefType::get({1}, init.getType());
    Value alloca = b.create<memref::AllocaOp>(loc, mrTy);
    b.create<AffineStoreOp>(loc, init, alloca, zeroMap, ValueRange{});
    allocas.push_back(alloca);
  }

  // 2. New loop with the same bounds and no iter_args.
  auto newFor = b.create<AffineForOp>(
      loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
      forOp.getStep());

  // 3. Body: load the carried values at the top...
  OpBuilder bb = OpBuilder::atBlockBegin(newFor.getBody());
  SmallVector<Value> loaded;
  for (Value alloca : allocas)
    loaded.push_back(
        bb.create<AffineLoadOp>(loc, alloca, zeroMap, ValueRange{}));

  // Redirect the induction variable and the iter_arg block arguments.
  forOp.getInductionVar().replaceAllUsesWith(newFor.getInductionVar());
  for (auto [arg, ld] : llvm::zip(forOp.getRegionIterArgs(), loaded))
    arg.replaceAllUsesWith(ld);

  // ...move the old body (minus its terminator) into the new one...
  auto *yield = forOp.getBody()->getTerminator();
  SmallVector<Value> yielded(yield->getOperands());
  Block *newBody = newFor.getBody();
  newBody->getOperations().splice(std::prev(newBody->end()), // before yield
                                  forOp.getBody()->getOperations(),
                                  forOp.getBody()->begin(),
                                  yield->getIterator());

  // ...and store the yielded values at the bottom.
  OpBuilder tail(newBody->getTerminator());
  for (auto [val, alloca] : llvm::zip(yielded, allocas))
    tail.create<AffineStoreOp>(loc, val, alloca, zeroMap, ValueRange{});

  // 4. Replace the old results with loads after the loop.
  b.setInsertionPointAfter(newFor);
  for (auto [res, alloca] : llvm::zip(forOp.getResults(), allocas))
    res.replaceAllUsesWith(
        b.create<AffineLoadOp>(loc, alloca, zeroMap, ValueRange{}));

  forOp.erase();
}

void IterArgsToMemrefPass::runOnOperation() {
  auto M = getOperation();

  SmallVector<AffineForOp> withIterArgs;
  M.walk([&](AffineForOp f) { if (f.getNumIterOperands()) withIterArgs.push_back(f); });
  for (auto f : llvm::reverse(withIterArgs))
    convertIterArgsToMemref(f);
}

std::unique_ptr<OperationPass<ModuleOp>> createIterArgsToMemrefPass() {
  return std::make_unique<IterArgsToMemrefPass>();
}
