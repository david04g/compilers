#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
namespace json = llvm::json;
namespace {
bool matrix(Type t) {
  auto r = dyn_cast<RankedTensorType>(t);
  return r && r.hasStaticShape() && r.getRank() == 2 &&
         r.getElementType().isF32() && r.getDimSize(0) > 0 &&
         r.getDimSize(0) <= 65535 && r.getDimSize(1) > 0 &&
         r.getDimSize(1) <= 131072 && r.getNumElements() <= 16777216;
}
LogicalResult checkNorm(Operation *op, unsigned gammaIndex) {
  auto t = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto g = dyn_cast<RankedTensorType>(op->getOperand(gammaIndex).getType());
  auto e = op->getAttrOfType<FloatAttr>("epsilon");
  if (!matrix(op->getOperand(0).getType()) || !g || g.getRank() != 1 ||
      !g.hasStaticShape() || !g.getElementType().isF32() ||
      g.getDimSize(0) != t.getDimSize(1) || op->getResult(0).getType() != t ||
      !e || !e.getType().isF32() || !std::isfinite(e.getValueAsDouble()) ||
      e.getValueAsDouble() <= 0)
    return op->emitOpError("requires static f32 matrix, matching gamma vector, "
                           "same output shape, and finite positive f32 epsilon");
  return success();
}
class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult,
                        OpTrait::ZeroRegions, OpTrait::ZeroSuccessors> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddOp)
  using Op::Op;
  static StringRef getOperationName() { return "fusion.add"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  Value getResult() { return (*this)->getResult(0); }
  LogicalResult verify() {
    Type t = getOperand(0).getType();
    if (!matrix(t) || getOperand(1).getType() != t || getResult().getType() != t)
      return emitOpError("requires matching static f32 matrices within shape limits");
    return success();
  }
};
class NormOp : public Op<NormOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult,
                         OpTrait::ZeroRegions, OpTrait::ZeroSuccessors> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormOp)
  using Op::Op;
  static StringRef getOperationName() { return "fusion.rms_norm"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  Value getResult() { return (*this)->getResult(0); }
  LogicalResult verify() { return checkNorm(getOperation(), 1); }
};
class ExecuteOp : public Op<ExecuteOp, OpTrait::NOperands<3>::Impl,
                            OpTrait::OneResult, OpTrait::ZeroRegions,
                            OpTrait::ZeroSuccessors> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExecuteOp)
  using Op::Op;
  static StringRef getOperationName() { return "fusion.execute"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  LogicalResult verify() {
    if (failed(checkNorm(getOperation(), 2))) return failure();
    auto mode = (*this)->getAttrOfType<StringAttr>("mode");
    auto threads = (*this)->getAttrOfType<IntegerAttr>("threads");
    if (getOperand(1).getType() != getOperand(0).getType() || !mode || !threads ||
        (mode.getValue() != "cached" && mode.getValue() != "stream" &&
         mode.getValue() != "partial" && mode.getValue() != "split") ||
        threads.getInt() < 32 || threads.getInt() > 1024 ||
        (threads.getInt() & (threads.getInt() - 1)))
      return emitOpError("invalid execution schedule");
    return success();
  }
};
class FusionDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionDialect)
  explicit FusionDialect(MLIRContext *ctx)
      : Dialect("fusion", ctx, TypeID::get<FusionDialect>()) {
    addOperations<AddOp, NormOp, ExecuteOp>();
  }
  static StringRef getDialectNamespace() { return "fusion"; }
};

struct Candidate {
  std::string mode, reason;
  int threads;
  int64_t items, bytes, scratch;
  int regs, shared, kernels;
  double score;
  bool legal() const { return reason.empty(); }
  std::string id() const { return mode + "-t" + std::to_string(threads); }
  json::Object toJSON() const {
    return json::Object{{"id", id()}, {"mode", mode}, {"threads", threads},
            {"items_per_thread", items}, {"kernel_count", kernels},
            {"estimated_logical_bytes", bytes}, {"scratch_bytes", scratch},
            {"estimated_registers_per_thread", regs},
            {"estimated_shared_bytes_per_block", shared},
            {"predicted_us", score}, {"legal", legal()},
            {"rejection_reason", reason}};
  }
};

// Uncalibrated Pascal prior. These are estimates, not hardware counters.
Candidate estimate(StringRef mode, int threads, int64_t m, int64_t n,
                   int64_t scratchLimit, int registerLimit) {
  int64_t s = m * n, q = (n + threads - 1) / threads;
  bool cached = mode == "cached", stream = mode == "stream";
  bool partial = mode == "partial", split = mode == "split";
  int regs = cached ? int(24 + q) : 24;
  int shared = threads * 4;
  int kernels = split ? 3 : partial ? 2 : 1;
  int64_t bytes = cached ? 16 * s : stream ? 24 * s :
                    partial ? 28 * s + 4 * m : 32 * s + 4 * m;
  int64_t scratch = (partial || split) ? 4 * s + 4 * m : 0;
  Candidate c{mode.str(), "", threads, q, bytes, scratch, regs, shared, kernels, 0};
  if (scratch > scratchLimit) c.reason = "scratch capacity exceeded";
  else if (regs > registerLimit) c.reason = "estimated register budget exceeded";
  // Approximate resident blocks from register/thread/shared-memory limits.
  int allocatedRegs = ((regs + 7) / 8) * 8;
  if (c.legal() && allocatedRegs * threads > 65536)
    c.reason = "estimated registers per block exceed the SM register file";
  int blocks = std::max(1, std::min({32, 2048 / threads,
                                   65536 / (allocatedRegs * threads),
                                   98304 / shared}));
  double occupancy = std::min(1.0, blocks * threads / 1024.0);
  double rowParallelism = std::min(1.0, m * (threads / 32.0) / (20.0 * 8));
  double efficiency = std::max(0.025, occupancy * rowParallelism);
  double rowBytes = cached ? double(bytes) : stream ? double(bytes) :
                      partial ? 12.0 * s + 4 * m : 4.0 * s + 4 * m;
  double flatBytes = double(bytes) - rowBytes;
  // 180 GB/s assumed effective bandwidth, 3 us assumed device launch cost.
  c.score = 3.0 * kernels + rowBytes / (180000.0 * efficiency) +
            flatBytes / 180000.0 + 0.08 * std::log2(threads);
  return c;
}

struct PlanPass : PassWrapper<PlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanPass)
  PlanPass() = default;
  PlanPass(const PlanPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "fusion-plan"; }
  StringRef getDescription() const final { return "Plan residual-add/RMSNorm CUDA schedules"; }
  Option<std::string> output{*this, "output", llvm::cl::init("plan.json")};
  Option<std::string> threadList{*this, "threads", llvm::cl::init("64,128,256")};
  Option<int64_t> scratchLimit{*this, "scratch-limit", llvm::cl::init(134217728)};
  Option<int> registerLimit{*this, "register-limit", llvm::cl::init(255)};
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    auto fail = [&](StringRef s) { mod.emitError(s); signalPassFailure(); };
    if (scratchLimit < 0 || registerLimit < 1 || registerLimit > 255) {
      fail("invalid scratch/register budget"); return;
    }
    SmallVector<StringRef> parts;
    StringRef(threadList).split(parts, ',');
    std::set<int> threads;
    for (auto part : parts) {
      int t = 0;
      if (part.getAsInteger(10, t) || t < 32 || t > 1024 || (t & (t - 1))) {
        fail("threads must be powers of two between 32 and 1024"); return;
      }
      threads.insert(t);
    }
    if (!llvm::hasSingleElement(mod.getBody()->getOperations())) {
      fail("expected exactly one function"); return;
    }
    auto f = dyn_cast<func::FuncOp>(mod.getBody()->front());
    if (!f || f.isExternal() || !llvm::hasSingleElement(f.getBody()) ||
        f.getNumArguments() != 3 || f.getNumResults() != 1 ||
        std::distance(f.front().begin(), f.front().end()) != 3) {
      fail("expected one three-argument add -> rms_norm -> return function"); return;
    }
    auto it = f.front().begin();
    auto add = dyn_cast<AddOp>(&*it++);
    auto norm = dyn_cast<NormOp>(&*it++);
    auto ret = dyn_cast<func::ReturnOp>(&*it);
    if (!add || !norm || !ret || ret.getNumOperands() != 1 ||
        add.getOperand(0) != f.getArgument(0) || add.getOperand(1) != f.getArgument(1) ||
        norm.getOperand(0) != add.getResult() || norm.getOperand(1) != f.getArgument(2) ||
        ret.getOperand(0) != norm.getResult() || !add.getResult().hasOneUse() ||
        !norm.getResult().hasOneUse()) {
      fail("unsupported graph or SSA wiring; expected x + residual -> rms_norm(gamma)"); return;
    }
    auto type = cast<RankedTensorType>(add.getResult().getType());
    int64_t m = type.getDimSize(0), n = type.getDimSize(1);
    double eps = norm->getAttrOfType<FloatAttr>("epsilon").getValueAsDouble();
    std::vector<Candidate> candidates;
    for (auto mode : {"cached", "stream", "partial", "split"})
      for (int t : threads)
        candidates.push_back(estimate(mode, t, m, n, scratchLimit, registerLimit));
    const Candidate *best = nullptr;
    json::Array entries;
    for (auto &c : candidates) {
      entries.push_back(c.toJSON());
      if (c.legal() && (!best || c.score < best->score)) best = &c;
    }
    if (!best) { fail("no legal schedule under the requested budgets"); return; }
    json::Object plan{{"schema_version", 1}, {"workload", "residual_rms_norm"},
                      {"rows", m}, {"width", n}, {"epsilon", eps},
                      {"target_prior", "GTX 1080 / sm_61; uncalibrated analytical prior"},
                      {"selected", best->id()}, {"candidates", std::move(entries)}};
    std::error_code ec;
    llvm::raw_fd_ostream file(output, ec, llvm::sys::fs::OF_Text);
    if (ec) { fail("could not open plan output"); return; }
    file << llvm::formatv("{0:2}\n", json::Value(std::move(plan)));
    file.flush();
    if (file.has_error()) { fail("failed to write plan"); return; }
    OpBuilder builder(norm);
    OperationState state(norm.getLoc(), ExecuteOp::getOperationName());
    state.addOperands(f.getArguments());
    state.addTypes(norm.getResult().getType());
    state.addAttribute("epsilon", norm->getAttr("epsilon"));
    state.addAttribute("mode", builder.getStringAttr(best->mode));
    state.addAttribute("threads", builder.getI64IntegerAttr(best->threads));
    auto *exec = builder.create(state);
    norm.getResult().replaceAllUsesWith(exec->getResult(0));
    norm.erase();
    add.erase();
    llvm::errs() << "fusion: selected " << best->id() << ", estimated "
                 << best->bytes << " logical bytes, " << best->regs
                 << " registers/thread; benchmark before claiming a speedup\n";
  }
};
} // namespace
int main(int argc, char **argv) {
  if (argc == 2 && StringRef(argv[1]) == "--version") {
    llvm::outs() << "fusion-opt; LLVM " << LLVM_VERSION_STRING << "\n";
    return 0;
  }
  if (argc < 2 || StringRef(argv[1]) == "--help") {
    llvm::outs() << "Usage: fusion-opt input.mlir --fusion-plan=output=plan.json [-o planned.mlir]\n";
    return argc < 2 ? 1 : 0;
  }
  std::string options, out = "-";
  bool havePass = false;
  for (int i = 2; i < argc; ++i) {
    StringRef arg(argv[i]);
    if (arg.consume_front("--fusion-plan=")) { options = arg.str(); havePass = true; }
    else if (arg == "-o" && i + 1 < argc) out = argv[++i];
    else { llvm::errs() << "unknown argument: " << arg << "\n"; return 1; }
  }
  if (!havePass) { llvm::errs() << "--fusion-plan is required\n"; return 1; }
  PassRegistration<PlanPass>();
  DialectRegistry registry;
  registry.insert<FusionDialect, func::FuncDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) return 1;
  PassManager pm(&context);
  if (failed(parsePassPipeline("fusion-plan{" + options + "}", pm)) ||
      failed(pm.run(*module))) return 1;
  if (out == "-") module->print(llvm::outs());
  else {
    std::error_code ec;
    llvm::raw_fd_ostream file(out, ec, llvm::sys::fs::OF_Text);
    if (ec) { llvm::errs() << ec.message() << "\n"; return 1; }
    module->print(file);
    file.flush();
    if (file.has_error()) return 1;
  }
  return 0;
}
