#include "DependencyGraph.hpp"
#include "Parser.hpp"

#include "TestHarness.hpp"

using namespace gpu_sched;

namespace {

bool hasEdge(const DependencyGraph &graph, int from, int to, DependencyType type,
             const std::string &resource = "") {
  for (const DependencyEdge &edge : graph.edges()) {
    if (edge.from == from && edge.to == to && edge.type == type &&
        (resource.empty() || edge.resource == resource)) {
      return true;
    }
  }
  return false;
}

} // namespace

int runDependencyTests() {
  int checks = 0;
  GpuConfig config = defaultConfig();

  KernelProgram raw = parseAssemblyText("ADD R1, R2, R3\nMUL R4, R1, R5\n", config);
  DependencyGraph raw_graph(raw.warps[0].instructions);
  requireTrue(hasEdge(raw_graph, 0, 1, DependencyType::RAW, "R1"),
              "RAW dependency detected");
  ++checks;

  KernelProgram waw_war =
      parseAssemblyText("ADD R4, R1, R2\nMUL R1, R5, R6\nADD R1, R7, R8\n",
                        config);
  DependencyGraph waw_war_graph(waw_war.warps[0].instructions);
  requireTrue(hasEdge(waw_war_graph, 0, 1, DependencyType::WAR, "R1"),
              "WAR dependency detected");
  ++checks;
  requireTrue(hasEdge(waw_war_graph, 1, 2, DependencyType::WAW, "R1"),
              "WAW dependency detected");
  ++checks;

  KernelProgram memory = parseAssemblyText("ST [R1], R2\nLD R3, [R1]\n", config);
  DependencyGraph memory_graph(memory.warps[0].instructions);
  requireTrue(hasEdge(memory_graph, 0, 1, DependencyType::MEMORY),
              "store-to-load memory dependency detected");
  ++checks;

  KernelProgram barrier = parseAssemblyText("ADD R1, R2, R3\nBAR\nMUL R4, R5, R6\n",
                                            config);
  DependencyGraph barrier_graph(barrier.warps[0].instructions);
  requireTrue(hasEdge(barrier_graph, 0, 1, DependencyType::BARRIER),
              "instruction before barrier cannot cross barrier");
  ++checks;
  requireTrue(hasEdge(barrier_graph, 1, 2, DependencyType::BARRIER),
              "instruction after barrier depends on barrier");
  ++checks;

  KernelProgram cp =
      parseAssemblyText("LD R1, [R2]\nADD R3, R1, R4\nMUL R5, R3, R6\n", config);
  DependencyGraph cp_graph(cp.warps[0].instructions);
  std::vector<int> critical_path = cp_graph.criticalPath();
  requireEqual(critical_path[2], 4, "sink critical path is its latency");
  ++checks;
  requireEqual(critical_path[1], 5, "middle critical path includes successor");
  ++checks;
  requireEqual(critical_path[0], 25, "load critical path includes long latency");
  ++checks;

  return checks;
}
