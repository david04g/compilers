#ifndef CUSTOM_RA_LIVENESS_H
#define CUSTOM_RA_LIVENESS_H

#include "CustomRA/Allocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {
class LiveIntervals;
class MachineBasicBlock;
class MachineFunction;
class MachineLoopInfo;
class MachineRegisterInfo;
}

namespace customra {

struct BlockLiveness {
  llvm::SmallSetVector<llvm::Register, 16> LiveIn;
  llvm::SmallSetVector<llvm::Register, 16> LiveOut;
};

using BlockLivenessMap =
    llvm::DenseMap<const llvm::MachineBasicBlock *, BlockLiveness>;

class LivenessAnalyzer {
public:
  BlockLivenessMap computeBlockLiveness(llvm::MachineFunction &MF) const;

  llvm::SmallVector<VirtRegStats, 32>
  collectVirtualRegisterStats(llvm::MachineFunction &MF,
                              llvm::LiveIntervals &LIS,
                              llvm::MachineLoopInfo &MLI) const;

  static bool intervalsOverlap(const llvm::LiveInterval &A,
                               const llvm::LiveInterval &B);

private:
  void collectLocalSets(llvm::MachineFunction &MF,
                        llvm::DenseMap<const llvm::MachineBasicBlock *,
                                       llvm::SmallSetVector<llvm::Register, 16>>
                            &Use,
                        llvm::DenseMap<const llvm::MachineBasicBlock *,
                                       llvm::SmallSetVector<llvm::Register, 16>>
                            &Def) const;
};

} // namespace customra

#endif
