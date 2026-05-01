#include "CustomRA/Allocator.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"

using namespace llvm;

namespace customra {

double computePriority(const VirtRegStats &Stats) {
  double UsePressure = static_cast<double>(Stats.LoopWeightedUses + Stats.Defs);
  double LengthPenalty = static_cast<double>(Stats.ApproxLength) * 0.05;
  double SplitBonus = Stats.SplitCandidate ? 2.0 : 0.0;
  return UsePressure + Stats.SpillCost + SplitBonus - LengthPenalty;
}

AllocationResult HeuristicAllocator::allocate(
    MachineFunction &MF, LiveIntervals &LIS, MachineLoopInfo &MLI,
    VirtRegMap &VRM, LiveRegMatrix &Matrix, const InterferenceGraph &Graph,
    ArrayRef<VirtRegStats> Stats, const AllocationOptions &Options) const {
  (void)MLI;
  (void)Options;

  SmallVector<const VirtRegStats *, 32> Worklist;
  for (const VirtRegStats &S : Stats)
    Worklist.push_back(&S);

  llvm::sort(Worklist, [](const VirtRegStats *A, const VirtRegStats *B) {
    double AP = computePriority(*A);
    double BP = computePriority(*B);
    if (AP != BP)
      return AP > BP;
    return A->VirtReg.id() < B->VirtReg.id();
  });

  AllocationResult Result;
  for (const VirtRegStats *S : Worklist) {
    AllocationDecision Decision;
    Decision.VirtReg = S->VirtReg;
    Decision.Priority = computePriority(*S);
    Decision.SplitCandidate = S->SplitCandidate;

    if (!LIS.hasInterval(S->VirtReg)) {
      Decision.FailureReason =
          "missing live interval for virtual register during allocation";
      Result.Failed = true;
      Result.FailedVirtReg = S->VirtReg;
      Result.FailureReason = Decision.FailureReason;
      Result.Plan[S->VirtReg] = Decision;
      break;
    }

    Matrix.invalidateVirtRegs();
    Decision.PhysReg =
        selectPhysicalRegister(MF, LIS, Matrix, Graph, Result.Plan, *S);
    if (Decision.PhysReg) {
      Matrix.assign(LIS.getInterval(S->VirtReg), Decision.PhysReg);
    } else {
      Decision.Spilled = true;
      Decision.FailureReason =
          "no legal physical register; spilling/splitting is not implemented "
          "in this correctness-first out-of-tree prototype";
      Result.Failed = true;
      Result.FailedVirtReg = S->VirtReg;
      Result.FailureReason = Decision.FailureReason;
      Result.Plan[S->VirtReg] = Decision;
      break;
    }

    Result.Plan[S->VirtReg] = Decision;
  }

  (void)VRM;
  return Result;
}

MCRegister HeuristicAllocator::selectPhysicalRegister(
    MachineFunction &MF, LiveIntervals &LIS, LiveRegMatrix &Matrix,
    const InterferenceGraph &Graph,
    const AllocationPlan &Plan, const VirtRegStats &Stats) const {
  if (!Stats.RegClass)
    return MCRegister();

  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  BitVector Reserved = TRI.getReservedRegs(MF);
  BitVector Allocatable = TRI.getAllocatableSet(MF, Stats.RegClass);
  const LiveInterval &LI = LIS.getInterval(Stats.VirtReg);

  for (int Reg = Allocatable.find_first(); Reg >= 0;
       Reg = Allocatable.find_next(Reg)) {
    if (Reserved.test(Reg))
      continue;

    MCRegister Candidate = MCRegister::from(Reg);
    if (!conflictsWithAssignedNeighbors(Candidate, Stats.VirtReg, TRI, Graph,
                                        Plan) &&
        Matrix.checkInterference(LI, Candidate) == LiveRegMatrix::IK_Free)
      return Candidate;
  }

  return MCRegister();
}

bool HeuristicAllocator::conflictsWithAssignedNeighbors(
    MCRegister Candidate, Register VirtReg, const TargetRegisterInfo &TRI,
    const InterferenceGraph &Graph, const AllocationPlan &Plan) const {
  for (Register Neighbor : Graph.neighbors(VirtReg)) {
    auto It = Plan.find(Neighbor);
    if (It == Plan.end() || It->second.Spilled || !It->second.PhysReg)
      continue;

    if (TRI.regsOverlap(Candidate, It->second.PhysReg))
      return true;
  }

  return false;
}
} // namespace customra
