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

static void outlineLoops(func::FuncOp &origFunc, unsigned &loopCounter) {
  auto unknownLoc = UnknownLoc::get(origFunc.getContext());

  bool debug = false;
  if (debug)
    origFunc.dump();

  auto module = origFunc->getParentOfType<ModuleOp>();
  auto topLoops = getTopLevelLoops(origFunc);

  BlockAndValueMapping fnResultMapper;
  Operation *lastFunc = nullptr;

  for (auto *topLoop : topLoops) {
    auto loop = cast<AffineForOp>(topLoop);

    auto undefinedValues = getOutOfBlockDefValues(loop.getBody());
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

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      if (std::find(undefinedValues.begin(),
                    undefinedValues.end(),
                    value) != undefinedValues.end()) {
        continue;
      }

      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      argMapper.map(value, newArg);
    }
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
      } else if (auto allocaOp =
                    dyn_cast_or_null<memref::AllocaOp>(definingOp)) {
        // Clone the allocation inside the outlined function.
        Operation *newAllocaOp = allocaOp->clone();
        bodyBlock.push_back(newAllocaOp);

        argMapper.map(
            value,
            newAllocaOp->getResult(0));

        AffineStoreOp initStore =
            findInitializer(value, loop);

        if (!initStore) {
          allocaOp->emitError(
              "cannot outline alloca without an initialization store");
          return;
        }

        Value initValue = initStore.getValueToStore();
        Operation *initDef = initValue.getDefiningOp();

        if (initDef) {
          mapExternalOperands(
              initDef,
              bodyBlock,
              argMapper,
              unknownLoc);

          cloneValueProducer(
              initValue,
              argMapper,
              bodyBlock);
        } else if (!argMapper.contains(initValue)) {
          initStore.emitError(
              "initializer value is not mapped into outlined function");
          return;
        }

        Operation *newInitStore =
            initStore->clone(argMapper);

        bodyBlock.push_back(newInitStore);
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
    topLoop->walk(
        [&](AffineStoreOp storeOp) { lastStoredMemref = storeOp.getMemref(); });
    assert(lastStoredMemref != nullptr && "No last stored memref found.");

    storedValue.insert(lastStoredMemref);

    Value originalStoredMemref = *storedValue.begin();

    auto originalAlloca =
        dyn_cast_or_null<memref::AllocaOp>(
            originalStoredMemref.getDefiningOp());

    AffineStoreOp originalInitStore;
    Operation *originalInitProducer = nullptr;

    if (originalAlloca) {
      originalInitStore =
          findInitializer(originalStoredMemref, loop);

      if (originalInitStore) {
        originalInitProducer =
            originalInitStore.getValueToStore().getDefiningOp();
      }
    }

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
        if (auto value = newArg.dyn_cast<BlockArgument>()) {
          auto attribute = origFuncArgAttrs[value.getArgNumber()];
          argAttrs.push_back(attribute);
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

        if (owner->getBlock() != topLoop->getBlock())
          continue;

        if (!topLoop->isBeforeInBlock(owner))
          continue;

        usesToReplace.push_back(&use);
      }

      for (OpOperand *use : usesToReplace)
        use->set(newValue);
    }

    // Remove the loop and any `memref` ops that have been cloned
    // and are no longer relevant.
    topLoop->erase();

    // The original initialization is now dead because:
    //   - the original loop was erased;
    //   - later uses were redirected to the call result.
    if (originalInitStore &&
        originalInitStore->use_empty()) {
      originalInitStore.erase();
    }

    // The initialization producer is usually an affine.load. It becomes dead
    // after the initialization store is removed.
    if (originalInitProducer &&
        originalInitProducer->use_empty()) {
      originalInitProducer->erase();
    }

    // Finally remove the original caller-side allocation.
    if (originalAlloca &&
        originalAlloca->use_empty()) {
      originalAlloca.erase();
    }
  }
}

void LoopOutlinePass::runOnOperation() {
  auto operation = getOperation();
  unsigned loopCounter = 0;
  for (auto func : operation.getOps<func::FuncOp>())
    outlineLoops(func, loopCounter);
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
