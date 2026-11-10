#include "dataflow/Planner.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

using namespace mlir;
namespace json = llvm::json;
namespace dataflow {
namespace {
bool constant(Value v, double expected) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  if (!c)
    return false;
  auto f = dyn_cast<FloatAttr>(c.getValue());
  return f && f.getType().isF32() &&
         f.getValueAsDouble() ==
             static_cast<double>(static_cast<float>(expected));
}
Operation *binary(Value v, StringRef name) {
  auto *op = v.getDefiningOp();
  return op && op->getName().getStringRef() == name && op->getNumOperands() == 2
             ? op
             : nullptr;
}
bool exactBinary(Value v, StringRef name, Value a, Value b) {
  auto *op = binary(v, name);
  return op && op->getOperand(0) == a && op->getOperand(1) == b;
}
bool isGelu(linalg::GenericOp op) {
  auto &b = op.getRegion().front();
  if (b.getNumArguments() != 2 || std::distance(b.begin(), b.end()) != 14)
    return false;
  for (auto &operation : b) {
    auto name = operation.getName().getStringRef();
    if (name != "arith.constant" && name != "arith.mulf" &&
        name != "arith.addf" && name != "math.tanh" && name != "linalg.yield")
      return false;
  }
  Value x = b.getArgument(0);
  auto *mul = binary(b.getTerminator()->getOperand(0), "arith.mulf");
  if (!mul)
    return false;
  auto *half = binary(mul->getOperand(0), "arith.mulf");
  auto *one = binary(mul->getOperand(1), "arith.addf");
  if (!half || !one || !constant(half->getOperand(0), 0.5) ||
      half->getOperand(1) != x || !constant(one->getOperand(0), 1.0))
    return false;
  auto *tanh = one->getOperand(1).getDefiningOp();
  if (!tanh || tanh->getName().getStringRef() != "math.tanh" ||
      tanh->getNumOperands() != 1)
    return false;
  auto *scale = binary(tanh->getOperand(0), "arith.mulf");
  if (!scale || !constant(scale->getOperand(0), 0.7978845608028654))
    return false;
  auto *sum = binary(scale->getOperand(1), "arith.addf");
  if (!sum || sum->getOperand(0) != x)
    return false;
  auto *cubic = binary(sum->getOperand(1), "arith.mulf");
  if (!cubic || !constant(cubic->getOperand(0), 0.044715))
    return false;
  auto *cube = binary(cubic->getOperand(1), "arith.mulf");
  return cube && cube->getOperand(1) == x &&
         exactBinary(cube->getOperand(0), "arith.mulf", x, x);
}
bool tensorType(Type type, int rank) {
  auto t = dyn_cast<RankedTensorType>(type);
  return t && t.getRank() == rank && t.hasStaticShape() &&
         t.getElementType().isF32() &&
         llvm::all_of(t.getShape(),
                      [](int64_t d) { return d > 0 && d <= 4096; });
}
bool zeroInit(linalg::MatmulOp op) {
  auto init = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  return init && constant(init.getDpsInputs()[0], 0.0) &&
         init.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
}
bool mapsAndLoops(linalg::LinalgOp op, ArrayRef<AffineMap> expected,
                  ArrayRef<utils::IteratorType> loops) {
  return op.getIndexingMapsArray() == expected &&
         op.getIteratorTypesArray() == loops;
}
LogicalResult extract(ModuleOp module, Graph &g) {
  auto funcs = module.getOps<func::FuncOp>();
  if (std::distance(funcs.begin(), funcs.end()) != 1 ||
      std::distance(module.getBody()->begin(), module.getBody()->end()) != 1)
    return module.emitError("expected exactly one func.func in the module");
  auto f = *funcs.begin();
  auto fail = [&]() -> LogicalResult {
    return f.emitError(
        "supported graph: static f32 matmul(zero init) -> column bias -> "
        "canonical tanh GELU"
        " [-> matmul(zero init)]; see examples and docs/input-contract.md");
  };
  if (f.isExternal() || !llvm::hasSingleElement(f.getBody()) ||
      (f.getNumArguments() != 3 && f.getNumArguments() != 4) ||
      f.getNumResults() != 1)
    return fail();
  for (unsigned i = 0; i < f.getNumArguments(); ++i)
    if (!tensorType(f.getArgument(i).getType(), i == 2 ? 1 : 2))
      return fail();
  if (!tensorType(f.getResultTypes()[0], 2))
    return fail();
  auto a = cast<RankedTensorType>(f.getArgument(0).getType());
  auto b = cast<RankedTensorType>(f.getArgument(1).getType());
  auto biasType = cast<RankedTensorType>(f.getArgument(2).getType());
  g = {a.getDimSize(0), a.getDimSize(1), b.getDimSize(1), 0};
  if (b.getDimSize(0) != g.k || biasType.getDimSize(0) != g.n)
    return fail();
  if (f.getNumArguments() == 4) {
    auto w2 = cast<RankedTensorType>(f.getArgument(3).getType());
    if (w2.getDimSize(0) != g.n)
      return fail();
    g.p = w2.getDimSize(1);
  }
  SmallVector<linalg::MatmulOp> matmuls;
  SmallVector<linalg::GenericOp> generics;
  for (auto &op : f.getBody().front()) {
    if (auto mm = dyn_cast<linalg::MatmulOp>(op))
      matmuls.push_back(mm);
    else if (auto generic = dyn_cast<linalg::GenericOp>(op))
      generics.push_back(generic);
    else if (!isa<arith::ConstantOp, tensor::EmptyOp, linalg::FillOp,
                  func::ReturnOp>(op))
      return fail();
  }
  if (matmuls.size() != (g.p ? 2u : 1u) || generics.size() != 2)
    return fail();
  auto *ctx = f.getContext();
  auto d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx),
       d2 = getAffineDimExpr(2, ctx);
  auto id = AffineMap::getMultiDimIdentityMap(2, ctx);
  auto col = AffineMap::get(2, 0, {d1}, ctx);
  using IT = utils::IteratorType;
  for (auto mm : matmuls) {
    if (!zeroInit(mm) || !mm->getResult(0).hasOneUse() ||
        !mapsAndLoops(mm,
                      {AffineMap::get(3, 0, {d0, d2}, ctx),
                       AffineMap::get(3, 0, {d2, d1}, ctx),
                       AffineMap::get(3, 0, {d0, d1}, ctx)},
                      {IT::parallel, IT::parallel, IT::reduction}))
      return fail();
  }
  auto mm = matmuls[0];
  auto bias = generics[0];
  auto gelu = generics[1];
  if (mm.getDpsInputs()[0] != f.getArgument(0) ||
      mm.getDpsInputs()[1] != f.getArgument(1))
    return fail();
  if (bias.getDpsInputs().size() != 2 || bias.getDpsInits().size() != 1 ||
      gelu.getDpsInputs().size() != 1 || gelu.getDpsInits().size() != 1 ||
      !bias.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
      !gelu.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
      !mapsAndLoops(bias, {id, col, id}, {IT::parallel, IT::parallel}) ||
      !mapsAndLoops(gelu, {id, id}, {IT::parallel, IT::parallel}) ||
      bias.getDpsInputs()[0] != mm->getResult(0) ||
      bias.getDpsInputs()[1] != f.getArgument(2) ||
      gelu.getDpsInputs()[0] != bias->getResult(0) ||
      !bias->getResult(0).hasOneUse() || !gelu->getResult(0).hasOneUse())
    return fail();
  auto &bb = bias.getRegion().front();
  if (std::distance(bb.begin(), bb.end()) != 2 ||
      !exactBinary(bb.getTerminator()->getOperand(0), "arith.addf",
                   bb.getArgument(0), bb.getArgument(1)) ||
      !isGelu(gelu))
    return fail();
  Value final = gelu->getResult(0);
  if (g.p) {
    if (matmuls[1].getDpsInputs()[0] != final ||
        matmuls[1].getDpsInputs()[1] != f.getArgument(3))
      return fail();
    final = matmuls[1]->getResult(0);
  }
  auto ret = dyn_cast<func::ReturnOp>(f.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != final)
    return fail();
  return success();
}

struct PlanPass : PassWrapper<PlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanPass)
  PlanPass() = default;
  PlanPass(const PlanPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "dataflow-plan"; }
  StringRef getDescription() const final {
    return "Select a tiled accelerator execution plan";
  }
  Option<std::string> output{*this, "output",
                             llvm::cl::desc("Plan JSON output path"),
                             llvm::cl::init("plan.json")};
  Option<std::string> hardware{
      *this, "hardware",
      llvm::cl::desc("Hardware JSON path (defaults if omitted)"),
      llvm::cl::init("")};
  Option<std::string> strategy{*this, "strategy",
                               llvm::cl::desc("heuristic, split, or fuse"),
                               llvm::cl::init("heuristic")};
  Option<std::string> tiles{*this, "tiles",
                            llvm::cl::desc("Comma-separated MxNxK tiles"),
                            llvm::cl::init("8x8x8,16x16x16,16x32x16,32x16x16")};
  Option<std::string> placement{*this, "placement",
                                llvm::cl::desc("search, compact, or spread"),
                                llvm::cl::init("search")};
  Option<bool> emitAll{
      *this, "emit-all",
      llvm::cl::desc("Include every legal plan for exhaustive simulation"),
      llvm::cl::init(false)};
  void runOnOperation() final {
    Graph graph;
    auto module = getOperation();
    if (failed(extract(module, graph))) {
      signalPassFailure();
      return;
    }
    Hardware hw;
    std::string error;
    if (!hardware.empty() && !readHardware(hardware, hw, error)) {
      module.emitError(error);
      signalPassFailure();
      return;
    }
    json::Object result;
    bool ok =
        plan(graph, hw, {strategy, placement, tiles, emitAll}, result, error);
    if (!result.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream file(output, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        module.emitError("cannot write plan: ") << ec.message();
        signalPassFailure();
        return;
      }
      file << llvm::formatv("{0:2}\n", json::Value(std::move(result)));
      file.flush();
      if (file.has_error()) {
        module.emitError("failed to write plan");
        signalPassFailure();
        return;
      }
    }
    if (!ok) {
      module.emitError(error);
      signalPassFailure();
      return;
    }
    module->setAttr("dataflow.plan_file",
                    StringAttr::get(&getContext(), output));
  }
};
} // namespace
void registerPlanPass() { PassRegistration<PlanPass>(); }
} // namespace dataflow
