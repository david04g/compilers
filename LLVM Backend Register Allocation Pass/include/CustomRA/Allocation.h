#ifndef CUSTOM_RA_ALLOCATION_H
#define CUSTOM_RA_ALLOCATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Pass.h"

#include <string>

namespace llvm {
class LiveInterval;
class LiveRegMatrix;
class MachineFunction;
class MachineLoopInfo;
class MachineRegisterInfo;
class TargetRegisterClass;
class TargetRegisterInfo;
class VirtRegMap;
} // namespace llvm

namespace customra {

struct AllocationOptions {
  bool Debug = false;
  bool AnnotateMIR = false;
  std::string DotPath;
  std::string TracePath;
};

struct VirtRegStats {
  llvm::Register VirtReg;
  const llvm::TargetRegisterClass *RegClass = nullptr;
  unsigned Uses = 0;
  unsigned Defs = 0;
  unsigned LoopWeightedUses = 0;
  unsigned SegmentCount = 0;
  unsigned ApproxLength = 0;
  double SpillCost = 0.0;
  bool SplitCandidate = false;
};

struct AllocationDecision {
  llvm::Register VirtReg;
  llvm::MCRegister PhysReg = llvm::MCRegister();
  bool Spilled = false;
  bool SplitCandidate = false;
  bool SplitApplied = false;
  double Priority = 0.0;
  std::string FailureReason;
};

using AllocationPlan = llvm::DenseMap<llvm::Register, AllocationDecision>;

struct AllocationResult {
  AllocationPlan Plan;
  bool Failed = false;
  llvm::Register FailedVirtReg = llvm::Register();
  std::string FailureReason;
};

double computePriority(const VirtRegStats &Stats);

} // namespace customra

namespace llvm {
FunctionPass *createCustomRegAllocPass();
}

#endif
