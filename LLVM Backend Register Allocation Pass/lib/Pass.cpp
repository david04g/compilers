#include "CustomRA/Allocator.h"
#include "CustomRA/DebugEmitter.h"
#include "CustomRA/InterferenceGraph.h"
#include "CustomRA/Liveness.h"

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/Support/FileSystem.h"
#include <memory>

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
__attribute__((
    dllimport)) MachinePassRegistry<RegisterRegAlloc::FunctionPassCtor>
    RegisterRegAllocBase<RegisterRegAlloc>::Registry;
#endif
void initializeCustomRegAllocPass(PassRegistry &);
} // namespace llvm

static cl::opt<bool>
    CustomRADebug("custom-ra-debug", cl::Hidden,
                  cl::desc("Emit custom register allocator trace output"),
                  cl::init(false));

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
    AU.addRequired<LiveStacksWrapperLegacy>();
    AU.addPreserved<LiveStacksWrapperLegacy>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
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
    LiveRegMatrix &Matrix = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    MF.getRegInfo().freezeReservedRegs();

    AllocationOptions Options;
    Options.Debug = CustomRADebug;
    Options.AnnotateMIR = CustomRAAnnotateMIR;
    Options.DotPath = CustomRADot;
    Options.TracePath = CustomRATrace;

    auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
    auto &Stacks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
    auto &DomTree = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
    VirtRegAuxInfo Aux(MF, LIS, VRM, MLI, MBFI,
                       &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
    Aux.calculateSpillWeightsAndHints();
    std::unique_ptr<Spiller> Spill(
        createInlineSpiller({LIS, Stacks, DomTree, MBFI}, MF, VRM, Aux));
    SmallPtrSet<MachineInstr *, 32> DeadRemats;

    // Intentionally simple: allocate, spill one failed interval, then retry.
    // Rebuilding the graph avoids using stale lifetimes after spill insertion.
    unsigned Round = 0;
    unsigned SpillCount = 0;
    const unsigned MaxRounds = 10000;
    auto logEvent = [&](StringRef Message) {
      if (Options.Debug)
        errs() << Message << '\n';
      if (!Options.TracePath.empty()) {
        std::error_code EC;
        raw_fd_ostream OS(Options.TracePath, EC,
                          sys::fs::OF_Text | sys::fs::OF_Append);
        if (EC)
          report_fatal_error("custom-ra: cannot write spill trace");
        OS << Message << '\n';
      }
    };
    while (true) {
      if (++Round > MaxRounds)
        report_fatal_error("custom-ra: spill retry limit exceeded");
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
        Register Failed = Result.FailedVirtReg;
        if (!LIS.hasInterval(Failed) || !LIS.getInterval(Failed).isSpillable())
          report_fatal_error("custom-ra: cannot spill an unspillable interval");

        // Live intervals must be removed from the matrix before spill edits.
        for (const auto &Entry : Result.Plan)
          if (Entry.second.PhysReg)
            Matrix.unassign(LIS.getInterval(Entry.first));
        Matrix.invalidateVirtRegs();

        SmallVector<Register, 16> NewRegs;
        {
          LiveRangeEdit Edit(&LIS.getInterval(Failed), NewRegs, MF, LIS, &VRM,
                             nullptr, &DeadRemats);
          Spill->spill(Edit);
        }
        ++SpillCount;
        const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
        std::string Message;
        raw_string_ostream OS(Message);
        OS << "spill: round=" << Round << " value=" << printReg(Failed, TRI);
        int Slot = VRM.getStackSlot(Failed);
        if (Slot != VirtRegMap::NO_STACK_SLOT)
          OS << " stack-slot=fi#" << Slot;
        else
          OS << " stack-slot=none (rematerialized or eliminated)";
        OS << " new-temporaries=" << NewRegs.size();
        for (Register R : NewRegs)
          OS << ' ' << printReg(R, TRI);
        OS.flush();
        logEvent(Message);
        continue;
      }

      Spill->postOptimization();
      for (MachineInstr *MI : DeadRemats) {
        LIS.RemoveMachineInstrFromMaps(*MI);
        MI->eraseFromParent();
      }
      logEvent("allocation-complete: rounds=" + std::to_string(Round) +
               " spill-events=" + std::to_string(SpillCount));

      LLVM_DEBUG(dbgs() << "custom-ra: allocated " << Result.Plan.size()
                        << " virtual registers in " << MF.getName() << '\n');

      return SpillCount != 0 || !Result.Plan.empty();
    }
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
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(CustomRegAlloc, DEBUG_TYPE,
                    "Custom Heuristic Register Allocator", false, false)

namespace llvm {

FunctionPass *createCustomRegAllocPass() { return new CustomRegAlloc(); }

} // namespace llvm

static RegisterRegAlloc CustomRARegAlloc("custom-ra",
                                         "custom heuristic register allocator",
                                         createCustomRegAllocPass);

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCustomRegAllocPass(PassRegistry &Registry) {
  initializeCustomRegAllocPass(Registry);
}
