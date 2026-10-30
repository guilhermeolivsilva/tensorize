#include "transforms/Passes.h"

#include "analysis/PolyhedralAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_LOOPOUTLINE
#include "transforms/Passes.h.inc"

namespace {

struct LoopOutlinePass : public impl::LoopOutlineBase<LoopOutlinePass> {
  void runOnOperation() override;
};

} // namespace
} // namespace mlir

static BlockAndValueMapping reverseMap(BlockAndValueMapping &mapper) {
  BlockAndValueMapping reverseMapper;
  for (auto &pair : mapper.getValueMap())
    reverseMapper.map(pair.second, pair.first);
  return reverseMapper;
}

// Helper to search for the last `store` to an `alloca` before the given `loop`.
static AffineStoreOp findInitializer(Value memref,
                                     AffineForOp loop) {
  AffineStoreOp initializer;

  Block *block = loop->getBlock();

  for (Operation &op : block->getOperations()) {
    if (&op == loop.getOperation())
      break;

    auto store = dyn_cast<AffineStoreOp>(&op);
    if (!store)
      continue;

    if (store.getMemref() == memref)
      initializer = store;
  }

  return initializer;
}

static void mapExternalOperands(
    Operation *op,
    Block &bodyBlock,
    BlockAndValueMapping &mapping,
    Location loc) {
  for (Value operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;

    // A function/block argument from the original function must become an
    // argument of the outlined function.
    if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      auto newArg = bodyBlock.addArgument(
          blockArg.getType(), loc);

      mapping.map(operand, newArg);
      continue;
    }

    // Constants or other values defined outside the outlined region should
    // normally have been handled by the existing undefined-value logic.
    if (operand.getDefiningOp()) {
      continue;
    }

    op->emitError(
        "cannot outline initializer: unmapped external operand");
  }
}

static Value cloneValueProducer(
    Value value,
    BlockAndValueMapping &mapping,
    Block &destination) {
  if (mapping.contains(value))
    return mapping.lookup(value);

  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    value.getDefiningOp()->emitError(
        "block argument was not mapped before cloning");
    return value;
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return value;

  for (Value operand : def->getOperands()) {
    if (!mapping.contains(operand) &&
        operand.getDefiningOp()) {
      cloneValueProducer(
          operand, mapping, destination);
    }
  }

  Operation *clone = def->clone(mapping);
  destination.push_back(clone);

  for (auto oldResult : def->getResults()) {
    unsigned index = oldResult.getResultNumber();
    mapping.map(oldResult, clone->getResult(index));
  }

  return mapping.lookup(value);
}

static llvm::SmallVector<Operation *> getLoopsToOutline(func::FuncOp func) {
  llvm::SmallVector<Operation *> loops;
  func.walk([&](AffineForOp forOp) {
    bool hasInner = false;
    forOp.getBody()->walk([&](AffineForOp) { hasInner = true; });
    if (!hasInner)
      loops.push_back(forOp);
  });
  return loops;
}

static llvm::SetVector<Value> getCapturedValues(Operation *loop) {
  llvm::SetVector<Value> captured;
  loop->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      // Defined inside the loop (op result or nested block arg)?
      if (Operation *def = operand.getDefiningOp()) {
        if (loop->isAncestor(def))
          continue;
      } else {
        auto ba = cast<BlockArgument>(operand);
        Operation *owner = ba.getOwner()->getParentOp();
        if (owner && loop->isAncestor(owner))
          continue;
      }
      captured.insert(operand);
    }
  });
  return captured;
}

static void outlineLoops(func::FuncOp &origFunc, unsigned &loopCounter) {
  auto unknownLoc = UnknownLoc::get(origFunc.getContext());

  bool debug = false;
  if (debug)
    origFunc.dump();

  auto module = origFunc->getParentOfType<ModuleOp>();
  auto topLoops = getLoopsToOutline(origFunc);

  BlockAndValueMapping fnResultMapper;
  Operation *lastFunc = nullptr;

  for (auto *topLoop : topLoops) {
    auto loop = cast<AffineForOp>(topLoop);

    // auto undefinedValues = getOutOfBlockDefValues(loop.getBody());
    auto undefinedValues = getCapturedValues(topLoop);
    auto loadedValues = getLoadedMemRefValues(topLoop);
    auto storedValues = getStoredMemRefValues(topLoop);
    auto allocaValues = getAllocaMemRefValues(topLoop);

    if (debug) {
      llvm::outs() << "-----------------\n";
      topLoop->dump();
      llvm::outs() << "Undefined values:\n";
      for (auto value : undefinedValues)
        value.dump();
      llvm::outs() << "Loaded:\n";
      for (auto value : loadedValues)
        value.dump();
      llvm::outs() << "Stored:\n";
      for (auto value : storedValues)
        value.dump();
      llvm::outs() << "Alloca:\n";
      for (auto value : allocaValues)
        value.dump();
    }

    for (auto value : allocaValues) {
      Operation *def = value.getDefiningOp();

      if (!def || topLoop->isAncestor(def))
        continue;

      // This alloca is defined outside the outlined loop and carries state.
      loadedValues.insert(value);
      storedValues.insert(value);
    }

    // Create a new function.
    // ---------------------------------------------
    OpBuilder builder(origFunc.getContext());
    auto func = builder.create<func::FuncOp>(
        unknownLoc, "fn_" + std::to_string(loopCounter++),
        builder.getFunctionType({}, {}));
    func->setAttr("irsynth.original", builder.getUnitAttr());
    auto &bodyBlock = *func.addEntryBlock();

    // Add arguments to function.
    BlockAndValueMapping argMapper;
    auto isDefinedOutside = [&](Value v) {
      if (Operation *def = v.getDefiningOp())
        return !topLoop->isAncestor(def);
      auto ba = cast<BlockArgument>(v);
      return !topLoop->isAncestor(ba.getOwner()->getParentOp());
    };

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      if (!isDefinedOutside(value) ||
          std::find(undefinedValues.begin(),
                    undefinedValues.end(),
                    value) != undefinedValues.end()) {
        continue;
      }

      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      argMapper.map(value, newArg);
    }

    const bool isTopLevel = topLoop->getParentOp() == origFunc.getOperation();

    // Ops whose clones now live inside the outlined function. Only these may be
    // removed from the caller after the loop is erased.
    llvm::SmallVector<memref::AllocaOp> sunkAllocas;
    llvm::SmallVector<AffineStoreOp> sunkInitStores;
    llvm::SmallSetVector<Operation *, 4> sunkInitProducers;

    // - Add undefined values as arguments or as local variables if they are
    // constants.
    for (auto value : undefinedValues) {
      if (argMapper.contains(value)) {
        continue;
      }

      // If the defining operation is a constant or a memref alloca, copy and
      // add it to the new function. Else, add it as an argument.
      auto *definingOp = value.getDefiningOp();
      // - Constant.
      if (definingOp && dyn_cast<arith::ConstantOp>(definingOp)) {
        auto constantOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        argMapper.map(value, newConstantOp.getResult());

        // - Memref alloca.
      } else if (auto allocaOp = isTopLevel
               ? dyn_cast_or_null<memref::AllocaOp>(definingOp)
               : memref::AllocaOp()) {
        Operation *newAllocaOp = allocaOp->clone();
        bodyBlock.push_back(newAllocaOp);
        argMapper.map(value, newAllocaOp->getResult(0));

        AffineStoreOp initStore = findInitializer(value, loop);
        if (!initStore) { /* error as before */ return; }

        Value initValue = initStore.getValueToStore();
        Operation *initDef = initValue.getDefiningOp();
        if (initDef) {
          mapExternalOperands(initDef, bodyBlock, argMapper, unknownLoc);
          cloneValueProducer(initValue, argMapper, bodyBlock);
        } else if (!argMapper.contains(initValue)) { /* error as before */ return; }

        bodyBlock.push_back(initStore->clone(argMapper));

        // Record only after the clone succeeded.
        sunkAllocas.push_back(allocaOp);
        sunkInitStores.push_back(initStore);
        if (initDef)
          sunkInitProducers.insert(initDef);
      }

        // Else, add as argument.
      else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        argMapper.map(value, newArg);
      }
    }
    auto reverseMapper = reverseMap(argMapper);

    // Add body.
    bodyBlock.push_back(topLoop->clone(argMapper));

    // Add the last stored memref value as result.
    llvm::SetVector<Value> storedValue;

    Value lastStoredMemref = nullptr;
    topLoop->walk([&](AffineStoreOp storeOp) {
      Value m = storeOp.getMemref();
      Operation *def = m.getDefiningOp();
      if (def && topLoop->isAncestor(def))   // allocated inside the loop: local
        return;
      lastStoredMemref = m;
    });
    assert(lastStoredMemref != nullptr && "No last stored memref found.");

    storedValue.insert(lastStoredMemref);

    Value originalStoredMemref = *storedValue.begin();

    // - Create return operation.
    llvm::SmallVector<Value> results;
    for (auto value : storedValue)
      results.push_back(argMapper.lookup(value));
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    builder.create<func::ReturnOp>(unknownLoc, results);

    // - Add the results to function type.
    llvm::SmallVector<Type> resultTypes;
    for (auto value : storedValue)
      resultTypes.push_back(value.getType());
    func.setFunctionType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), resultTypes));

    // - Add arg attributes of the original function.
    llvm::SmallVector<Attribute> argAttrs;
    if (auto origFuncArgAttrs = origFunc.getAllArgAttrs()) {
      for (auto arg : bodyBlock.getArguments()) {
        auto newArg = reverseMapper.lookup(arg);
        if (auto value = newArg.dyn_cast<BlockArgument>();
            value && value.getOwner() == &origFunc.front()) {
          argAttrs.push_back(origFuncArgAttrs[value.getArgNumber()]);
        } else {
          argAttrs.push_back(builder.getDictionaryAttr({}));
        }
      }
      func.setAllArgAttrs(argAttrs);
    }

    // Insert the new function and replace the loop with a call to it.
    // ---------------------------------------------
    if (!lastFunc)
      builder.setInsertionPointToStart(module.getBody());
    else
      builder.setInsertionPointAfter(lastFunc);

    builder.insert(func);
    lastFunc = func;

    // Create args for the call.
    llvm::SmallVector<Value> args;
    for (auto arg : func.getArguments()) {
      auto value = reverseMapper.lookupOrNull(arg);

      // If the arg value was already recomputed by an earlier call, use this
      // one.
      if (fnResultMapper.contains(value))
        args.push_back(fnResultMapper.lookup(value));
      else
        args.push_back(value);
    }

    // Create function call.
    builder.setInsertionPoint(topLoop);
    auto callOp = builder.create<func::CallOp>(unknownLoc, func.getSymName(),
                                               func.getResultTypes(), args);

    // Update the value mapping and rewrite later uses.
    for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
      Value oldValue = storedValue[i];
      Value newValue = callOp.getResult(i);

      fnResultMapper.map(oldValue, newValue);

      SmallVector<OpOperand *, 8> usesToReplace;

      for (OpOperand &use : oldValue.getUses()) {
        Operation *owner = use.getOwner();

        if (owner == callOp.getOperation())
          continue;

        Operation *anc = topLoop->getBlock()->findAncestorOpInBlock(*owner);
        if (!anc || anc == callOp.getOperation() || !topLoop->isBeforeInBlock(anc))
          continue;

        usesToReplace.push_back(&use);
      }

      for (OpOperand *use : usesToReplace)
        use->set(newValue);
    }

    // Remove the loop and any `memref` ops that have been cloned
    // and are no longer relevant.
    topLoop->erase();

    // 1. The init stores were cloned into the outlined function. The caller-side
    //    copies are dead and must go first (they use the producer and the alloca).
    //    No use_empty() check here: stores have no results, so it is vacuous.
    for (AffineStoreOp store : sunkInitStores)
      store.erase();

    // 2. Producers (e.g. an affine.load or constant), only if nothing else
    //    still uses them.
    for (Operation *producer : sunkInitProducers)
      if (producer->use_empty())
        producer->erase();

    // 3. Finally the caller-side allocas, only if no remaining uses.
    for (memref::AllocaOp alloca : sunkAllocas)
      if (alloca->use_empty())
        alloca.erase();
  }
}

void LoopOutlinePass::runOnOperation() {
  auto operation = getOperation();
  unsigned loopCounter = 0;
  SmallVector<func::FuncOp> funcs(operation.getOps<func::FuncOp>());
  for (auto func : funcs) outlineLoops(func, loopCounter);
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
