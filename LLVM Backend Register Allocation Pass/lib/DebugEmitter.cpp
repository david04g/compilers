#include "CustomRA/DebugEmitter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace llvm;

namespace customra {

void DebugEmitter::emitTrace(MachineFunction &MF,
                             const BlockLivenessMap &LiveBlocks,
                             ArrayRef<VirtRegStats> Stats,
                             const InterferenceGraph &Graph,
                             const AllocationPlan &Plan) const {
  if (!Options.Debug && Options.TracePath.empty() && !Options.AnnotateMIR)
    return;

  std::error_code EC;
  std::unique_ptr<raw_fd_ostream> FileOS;
  raw_ostream *OS = &errs();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  if (!Options.TracePath.empty()) {
    FileOS = std::make_unique<raw_fd_ostream>(Options.TracePath, EC,
                                              sys::fs::OF_Text |
                                                  sys::fs::OF_Append);
    if (!EC)
      OS = FileOS.get();
    else
      errs() << "custom-ra: failed to open trace file " << Options.TracePath
             << ": " << EC.message() << '\n';
  }

  *OS << "custom-ra trace for " << MF.getName() << '\n';
  *OS << "blocks:\n";
  for (MachineBasicBlock &MBB : MF) {
    const BlockLiveness &BL = LiveBlocks.lookup(&MBB);
    *OS << "  bb." << MBB.getNumber() << " live-in={";
    interleaveComma(BL.LiveIn, *OS,
                    [&](Register Reg) { *OS << printReg(Reg, TRI); });
    *OS << "} live-out={";
    interleaveComma(BL.LiveOut, *OS,
                    [&](Register Reg) { *OS << printReg(Reg, TRI); });
    *OS << "}\n";
  }

  *OS << "virtual-registers:\n";
  for (const VirtRegStats &S : Stats) {
    *OS << "  " << printReg(S.VirtReg, TRI) << " uses=" << S.Uses
        << " defs=" << S.Defs << " weighted-uses=" << S.LoopWeightedUses
        << " segments=" << S.SegmentCount
        << " approx-length=" << S.ApproxLength
        << " spill-cost=" << S.SpillCost
        << " split-candidate=" << (S.SplitCandidate ? "yes" : "no") << '\n';
  }

  *OS << "interference: nodes=" << Graph.nodes().size()
      << " edges=" << Graph.edgeCount() << '\n';

  *OS << "allocation:\n";
  SmallVector<AllocationDecision, 32> Decisions;
  for (const auto &Entry : Plan)
    Decisions.push_back(Entry.second);
  llvm::sort(Decisions, [](const AllocationDecision &A,
                           const AllocationDecision &B) {
    if (A.Priority != B.Priority)
      return A.Priority > B.Priority;
    return A.VirtReg.id() < B.VirtReg.id();
  });

  for (const AllocationDecision &D : Decisions) {
    *OS << "  " << printReg(D.VirtReg, TRI) << " priority=" << D.Priority;
    if (D.Spilled)
      *OS << " spill-required";
    else
      *OS << " phys=" << printReg(D.PhysReg, TRI);
    if (D.SplitCandidate)
      *OS << " split-candidate";
    if (D.SplitApplied)
      *OS << " split-applied";
    if (!D.FailureReason.empty())
      *OS << " failure=\"" << D.FailureReason << "\"";
    *OS << '\n';
  }

  if (Options.AnnotateMIR)
    *OS << "annotation: decisions are available in this trace; MIR text "
           "is left unmodified so downstream verifier state remains stable\n";
}

void DebugEmitter::emitDot(MachineFunction &MF, const InterferenceGraph &Graph,
                           const AllocationPlan &Plan) const {
  if (Options.DotPath.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(Options.DotPath, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "custom-ra: failed to open dot file " << Options.DotPath << ": "
           << EC.message() << '\n';
    return;
  }

  Graph.writeDot(OS, MF.getSubtarget().getRegisterInfo(), &Plan);
}

} // namespace customra
