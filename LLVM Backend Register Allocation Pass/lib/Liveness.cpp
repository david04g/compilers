#include "CustomRA/Liveness.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#include <algorithm>

using namespace llvm;

namespace customra {

static bool isTrackedVirtualReg(const MachineOperand &MO) {
  if (!MO.isReg() || !MO.getReg())
    return false;
  return MO.getReg().isVirtual() && !MO.isUndef();
}

static bool sameSet(const SmallSetVector<Register, 16> &A,
                    const SmallSetVector<Register, 16> &B) {
  if (A.size() != B.size())
    return false;
  for (Register Reg : A)
    if (!B.count(Reg))
      return false;
  return true;
}

void LivenessAnalyzer::collectLocalSets(
    MachineFunction &MF,
    DenseMap<const MachineBasicBlock *, SmallSetVector<Register, 16>> &Use,
    DenseMap<const MachineBasicBlock *, SmallSetVector<Register, 16>> &Def)
    const {
  for (MachineBasicBlock &MBB : MF) {
    auto &BlockUse = Use[&MBB];
    auto &BlockDef = Def[&MBB];

    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!isTrackedVirtualReg(MO))
          continue;

        Register Reg = MO.getReg();
        if (MO.isUse() && !BlockDef.count(Reg))
          BlockUse.insert(Reg);
        if (MO.isDef())
          BlockDef.insert(Reg);
      }
    }
  }
}

BlockLivenessMap
LivenessAnalyzer::computeBlockLiveness(MachineFunction &MF) const {
  DenseMap<const MachineBasicBlock *, SmallSetVector<Register, 16>> Use;
  DenseMap<const MachineBasicBlock *, SmallSetVector<Register, 16>> Def;
  collectLocalSets(MF, Use, Def);

  BlockLivenessMap Result;
  for (MachineBasicBlock &MBB : MF)
    Result[&MBB] = BlockLiveness();

  bool Changed = true;
  while (Changed) {
    Changed = false;

    for (MachineBasicBlock &MBB : reverse(MF)) {
      SmallSetVector<Register, 16> NewOut;
      for (MachineBasicBlock *Succ : MBB.successors()) {
        for (Register Reg : Result[Succ].LiveIn)
          NewOut.insert(Reg);
      }

      SmallSetVector<Register, 16> NewIn = Use[&MBB];
      for (Register Reg : NewOut) {
        if (!Def[&MBB].count(Reg))
          NewIn.insert(Reg);
      }

      if (!sameSet(NewOut, Result[&MBB].LiveOut) ||
          !sameSet(NewIn, Result[&MBB].LiveIn)) {
        Result[&MBB].LiveOut = NewOut;
        Result[&MBB].LiveIn = NewIn;
        Changed = true;
      }
    }
  }

  return Result;
}

SmallVector<VirtRegStats, 32> LivenessAnalyzer::collectVirtualRegisterStats(
    MachineFunction &MF, LiveIntervals &LIS, MachineLoopInfo &MLI) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DenseMap<Register, unsigned> IndexByReg;
  SmallVector<VirtRegStats, 32> Stats;

  auto ensureStats = [&](Register Reg) -> VirtRegStats & {
    auto It = IndexByReg.find(Reg);
    if (It != IndexByReg.end())
      return Stats[It->second];

    unsigned Index = Stats.size();
    IndexByReg[Reg] = Index;
    Stats.push_back(VirtRegStats());
    VirtRegStats &S = Stats.back();
    S.VirtReg = Reg;
    S.RegClass = MRI.getRegClass(Reg);
    return S;
  };

  for (MachineBasicBlock &MBB : MF) {
    unsigned LoopDepth = 0;
    if (MachineLoop *Loop = MLI.getLoopFor(&MBB))
      LoopDepth = Loop->getLoopDepth();
    unsigned LoopWeight = 1u << std::min(LoopDepth, 6u);

    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!isTrackedVirtualReg(MO))
          continue;

        VirtRegStats &S = ensureStats(MO.getReg());
        if (MO.isUse()) {
          ++S.Uses;
          S.LoopWeightedUses += LoopWeight;
        }
        if (MO.isDef())
          ++S.Defs;
      }
    }
  }

  for (VirtRegStats &S : Stats) {
    if (LIS.hasInterval(S.VirtReg)) {
      const LiveInterval &LI = LIS.getInterval(S.VirtReg);
      S.SegmentCount = LI.segments.size();
      S.ApproxLength = LI.getSize();
    }

    S.SpillCost =
        static_cast<double>(S.LoopWeightedUses + S.Defs + 1) /
        static_cast<double>(std::max(1u, S.ApproxLength));
    S.SplitCandidate = S.SegmentCount > 2 && S.ApproxLength > 16;
  }

  llvm::sort(Stats, [](const VirtRegStats &A, const VirtRegStats &B) {
    if (A.VirtReg != B.VirtReg)
      return A.VirtReg.id() < B.VirtReg.id();
    return false;
  });

  return Stats;
}

bool LivenessAnalyzer::intervalsOverlap(const LiveInterval &A,
                                        const LiveInterval &B) {
  return A.overlaps(B);
}

} // namespace customra
