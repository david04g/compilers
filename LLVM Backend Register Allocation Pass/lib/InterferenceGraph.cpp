#include "CustomRA/InterferenceGraph.h"

#include "CustomRA/Liveness.h"

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace customra {

void InterferenceGraph::addNode(Register Reg) {
  if (!Adj.count(Reg)) {
    Adj[Reg] = SmallSetVector<Register, 16>();
    Nodes.push_back(Reg);
  }
}

void InterferenceGraph::addEdge(Register A, Register B) {
  if (A == B)
    return;
  addNode(A);
  addNode(B);
  if (Adj[A].insert(B)) {
    Adj[B].insert(A);
    ++Edges;
  }
}

bool InterferenceGraph::interferes(Register A, Register B) const {
  auto It = Adj.find(A);
  return It != Adj.end() && It->second.count(B);
}

const SmallSetVector<Register, 16> &
InterferenceGraph::neighbors(Register Reg) const {
  static const SmallSetVector<Register, 16> Empty;
  auto It = Adj.find(Reg);
  return It == Adj.end() ? Empty : It->second;
}

void InterferenceGraph::writeDot(raw_ostream &OS, const TargetRegisterInfo *TRI,
                                 const AllocationPlan *Plan) const {
  OS << "graph custom_ra_interference {\n";
  OS << "  node [shape=circle];\n";

  for (Register Reg : Nodes) {
    OS << "  v" << Reg.id() << " [label=\"" << printReg(Reg, TRI);
    if (Plan) {
      auto It = Plan->find(Reg);
      if (It != Plan->end()) {
        if (It->second.Spilled)
          OS << "\\nspill required";
        else if (It->second.PhysReg)
          OS << "\\nphys " << printReg(It->second.PhysReg, TRI);
      }
    }
    OS << "\"];\n";
  }

  for (Register A : Nodes) {
    for (Register B : neighbors(A)) {
      if (A.id() < B.id())
        OS << "  v" << A.id() << " -- v" << B.id() << ";\n";
    }
  }

  OS << "}\n";
}

InterferenceGraph buildInterferenceGraph(MachineFunction &MF,
                                         LiveIntervals &LIS,
                                         ArrayRef<VirtRegStats> Stats) {
  (void)MF;
  InterferenceGraph Graph;

  for (const VirtRegStats &S : Stats)
    Graph.addNode(S.VirtReg);

  for (unsigned I = 0; I < Stats.size(); ++I) {
    if (!LIS.hasInterval(Stats[I].VirtReg))
      continue;

    const LiveInterval &LI = LIS.getInterval(Stats[I].VirtReg);
    for (unsigned J = I + 1; J < Stats.size(); ++J) {
      if (!LIS.hasInterval(Stats[J].VirtReg))
        continue;

      const LiveInterval &LJ = LIS.getInterval(Stats[J].VirtReg);
      if (LivenessAnalyzer::intervalsOverlap(LI, LJ))
        Graph.addEdge(Stats[I].VirtReg, Stats[J].VirtReg);
    }
  }

  return Graph;
}

} // namespace customra
