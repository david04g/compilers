#include "CustomRA/Allocator.h"
#include "CustomRA/DebugEmitter.h"
#include "CustomRA/InterferenceGraph.h"
#include "CustomRA/Liveness.h"

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace customra;

#define DEBUG_TYPE "custom-ra"

namespace llvm {
#ifdef _WIN32
template <>
__attribute__((dllimport))
    MachinePassRegistry<RegisterRegAlloc::FunctionPassCtor>
        RegisterRegAllocBase<RegisterRegAlloc>::Registry;
#endif
void initializeCustomRegAllocPass(PassRegistry &);
}

static cl::opt<bool> CustomRADebug(
    "custom-ra-debug", cl::Hidden,
    cl::desc("Emit custom register allocator trace output"), cl::init(false));

static cl::opt<std::string>
    CustomRADot("custom-ra-dot", cl::Hidden, cl::value_desc("path"),
                cl::desc("Write custom register allocator interference graph"),
                cl::init(""));

static cl::opt<std::string>
    CustomRATrace("custom-ra-trace", cl::Hidden, cl::value_desc("path"),
                  cl::desc("Write custom register allocator allocation trace"),
                  cl::init(""));

static cl::opt<bool>
    CustomRAAnnotateMIR("custom-ra-annotate-mir", cl::Hidden,
                        cl::desc("Include MIR-oriented allocation annotations "
                                 "in the custom allocator trace"),
                        cl::init(false));

namespace {

class CustomRegAlloc final : public MachineFunctionPass {
public:
  static char ID;

  CustomRegAlloc() : MachineFunctionPass(ID) {
    initializeCustomRegAllocPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Custom Heuristic Register Allocator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesCFG();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveRegMatrixWrapperLegacy>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    VirtRegMap &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    LiveRegMatrix &Matrix =
        getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    MF.getRegInfo().freezeReservedRegs();

    AllocationOptions Options;
    Options.Debug = CustomRADebug;
    Options.AnnotateMIR = CustomRAAnnotateMIR;
    Options.DotPath = CustomRADot;
    Options.TracePath = CustomRATrace;

    LivenessAnalyzer Liveness;
    BlockLivenessMap BlockLive = Liveness.computeBlockLiveness(MF);
    SmallVector<VirtRegStats, 32> Stats =
        Liveness.collectVirtualRegisterStats(MF, LIS, MLI);
    InterferenceGraph Graph = buildInterferenceGraph(MF, LIS, Stats);

    HeuristicAllocator Allocator;
    AllocationResult Result =
        Allocator.allocate(MF, LIS, MLI, VRM, Matrix, Graph, Stats, Options);

    DebugEmitter Debug(Options);
    Debug.emitTrace(MF, BlockLive, Stats, Graph, Result.Plan);
    Debug.emitDot(MF, Graph, Result.Plan);

    if (Result.Failed) {
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      std::string Message;
      raw_string_ostream OS(Message);
      OS << "custom-ra: unable to allocate "
         << printReg(Result.FailedVirtReg, TRI) << " in " << MF.getName()
         << ": " << Result.FailureReason;
      OS.flush();
      report_fatal_error(StringRef(Message), false);
    }

    LLVM_DEBUG(dbgs() << "custom-ra: allocated " << Result.Plan.size()
                      << " virtual registers in " << MF.getName() << '\n');

    return !Result.Plan.empty();
  }
};

} // namespace

char CustomRegAlloc::ID = 0;

INITIALIZE_PASS_BEGIN(CustomRegAlloc, DEBUG_TYPE,
                      "Custom Heuristic Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_END(CustomRegAlloc, DEBUG_TYPE,
                    "Custom Heuristic Register Allocator", false, false)

namespace llvm {

FunctionPass *createCustomRegAllocPass() { return new CustomRegAlloc(); }

} // namespace llvm

static RegisterRegAlloc
    CustomRARegAlloc("custom-ra", "custom heuristic register allocator",
                     createCustomRegAllocPass);

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCustomRegAllocPass(PassRegistry &Registry) {
  initializeCustomRegAllocPass(Registry);
}
