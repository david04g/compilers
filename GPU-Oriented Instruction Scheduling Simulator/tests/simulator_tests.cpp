#include "DependencyGraph.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Scheduler.hpp"
#include "Simulator.hpp"

#include "TestHarness.hpp"

using namespace gpu_sched;

int runSimulatorTests() {
  int checks = 0;
  GpuConfig config = defaultConfig();

  KernelProgram dependent =
      parseAssemblyText("LD R1, [R2]\nADD R3, R1, R4\n", config);
  SimulationMetrics dependent_metrics = simulateProgram(dependent, config);
  requireTrue(dependent_metrics.hazards.raw_stalls > 0,
              "dependent load/use produces RAW stalls");
  ++checks;
  requireEqual(dependent_metrics.issued_instructions, 2,
               "dependent program issues both instructions");
  ++checks;

  KernelProgram independent =
      parseAssemblyText("ADD R1, R2, R3\nMUL R4, R5, R6\n", config);
  SimulationMetrics independent_metrics = simulateProgram(independent, config);
  requireEqual(independent_metrics.hazards.raw_stalls, 0,
               "independent program has no RAW stalls");
  ++checks;

  DependencyGraph graph(dependent.warps[0].instructions);
  ComparisonMetrics comparison =
      compareMetrics(dependent_metrics, dependent_metrics, graph, 2);
  requireNear(comparison.throughput_percent, 0.0, 0.001,
              "same metrics have zero throughput improvement");
  ++checks;
  requireEqual(comparison.critical_path_length, 21,
               "metrics expose critical path length");
  ++checks;

  return checks;
}
