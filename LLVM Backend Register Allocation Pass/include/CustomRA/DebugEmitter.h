#ifndef CUSTOM_RA_DEBUG_EMITTER_H
#define CUSTOM_RA_DEBUG_EMITTER_H

#include "CustomRA/Allocation.h"
#include "CustomRA/InterferenceGraph.h"
#include "CustomRA/Liveness.h"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class MachineFunction;
}

namespace customra {

class DebugEmitter {
public:
  explicit DebugEmitter(const AllocationOptions &Options) : Options(Options) {}

  void emitTrace(llvm::MachineFunction &MF, const BlockLivenessMap &LiveBlocks,
                 llvm::ArrayRef<VirtRegStats> Stats,
                 const InterferenceGraph &Graph,
                 const AllocationPlan &Plan) const;

  void emitDot(llvm::MachineFunction &MF, const InterferenceGraph &Graph,
               const AllocationPlan &Plan) const;

private:
  const AllocationOptions &Options;
};

} // namespace customra

#endif
