#ifndef CUSTOM_RA_INTERFERENCE_GRAPH_H
#define CUSTOM_RA_INTERFERENCE_GRAPH_H

#include "CustomRA/Allocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {
class LiveIntervals;
class MachineFunction;
class raw_ostream;
class TargetRegisterInfo;
}

namespace customra {

class InterferenceGraph {
public:
  void addNode(llvm::Register Reg);
  void addEdge(llvm::Register A, llvm::Register B);
  bool interferes(llvm::Register A, llvm::Register B) const;

  llvm::ArrayRef<llvm::Register> nodes() const { return Nodes; }
  const llvm::SmallSetVector<llvm::Register, 16> &
  neighbors(llvm::Register Reg) const;

  unsigned edgeCount() const { return Edges; }
  void writeDot(llvm::raw_ostream &OS, const llvm::TargetRegisterInfo *TRI,
                const AllocationPlan *Plan) const;

private:
  llvm::SmallVector<llvm::Register, 32> Nodes;
  llvm::DenseMap<llvm::Register, llvm::SmallSetVector<llvm::Register, 16>>
      Adj;
  unsigned Edges = 0;
};

InterferenceGraph buildInterferenceGraph(llvm::MachineFunction &MF,
                                         llvm::LiveIntervals &LIS,
                                         llvm::ArrayRef<VirtRegStats> Stats);

} // namespace customra

#endif
