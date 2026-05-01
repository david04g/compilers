#ifndef CUSTOM_RA_ALLOCATOR_H
#define CUSTOM_RA_ALLOCATOR_H

#include "CustomRA/Allocation.h"
#include "CustomRA/InterferenceGraph.h"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class LiveIntervals;
class LiveRegMatrix;
class MachineFunction;
class MachineLoopInfo;
class MachineRegisterInfo;
class TargetRegisterInfo;
class VirtRegMap;
}

namespace customra {

class HeuristicAllocator {
public:
  AllocationResult allocate(llvm::MachineFunction &MF, llvm::LiveIntervals &LIS,
                            llvm::MachineLoopInfo &MLI,
                            llvm::VirtRegMap &VRM,
                            llvm::LiveRegMatrix &Matrix,
                            const InterferenceGraph &Graph,
                            llvm::ArrayRef<VirtRegStats> Stats,
                            const AllocationOptions &Options) const;

private:
  llvm::MCRegister selectPhysicalRegister(llvm::MachineFunction &MF,
                                          llvm::LiveIntervals &LIS,
                                          llvm::LiveRegMatrix &Matrix,
                                          const InterferenceGraph &Graph,
                                          const AllocationPlan &Plan,
                                          const VirtRegStats &Stats) const;

  bool conflictsWithAssignedNeighbors(llvm::MCRegister Candidate,
                                      llvm::Register VirtReg,
                                      const llvm::TargetRegisterInfo &TRI,
                                      const InterferenceGraph &Graph,
                                      const AllocationPlan &Plan) const;
};

} // namespace customra

#endif
