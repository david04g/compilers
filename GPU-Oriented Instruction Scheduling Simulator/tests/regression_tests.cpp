#include "DependencyGraph.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Scheduler.hpp"
#include "Simulator.hpp"

#include "TestHarness.hpp"

using namespace gpu_sched;

int runRegressionTests() {
  int checks = 0;
  GpuConfig config = defaultConfig();

  KernelProgram memory_latency = parseAssemblyText(
      "kernel memory_latency\n"
      "warp 0\n"
      "LD R1, [R10]\n"
      "ADD R2, R1, R3\n"
      "MUL R20, R21, R22\n"
      "FMA R23, R24, R25\n"
      "ADD R26, R27, R28\n"
      "MUL R29, R30, R31\n"
      "SFU R14, R15\n"
      "ADD R11, R2, R12\n"
      "END\n",
      config);
  KernelProgram scheduled =
      scheduleProgram(memory_latency, SchedulerKind::StallFill, config);
  DependencyGraph graph(memory_latency.warps[0].instructions);
  requireTrue(graph.validateSchedule(scheduled.warps[0].instructions),
              "memory latency scheduled output is valid");
  ++checks;
  ComparisonMetrics comparison =
      compareMetrics(simulateProgram(memory_latency, config),
                     simulateProgram(scheduled, config), graph,
                     static_cast<int>(scheduled.warps[0].instructions.size()));
  requireTrue(comparison.throughput_percent >= 20.0,
              "memory latency benchmark keeps roughly 20 percent gain");
  ++checks;
  requireTrue(comparison.scheduled.stall_cycles < comparison.baseline.stall_cycles,
              "memory latency scheduling reduces stalls");
  ++checks;

  KernelProgram barrier =
      parseAssemblyText("ADD R1, R2, R3\nMUL R4, R5, R6\nBAR\nLD R7, [R8]\n"
                        "ADD R9, R7, R10\n",
                        config);
  KernelProgram barrier_scheduled = scheduleProgram(barrier, SchedulerKind::StallFill,
                                                    config);
  DependencyGraph barrier_graph(barrier.warps[0].instructions);
  requireTrue(barrier_graph.validateSchedule(barrier_scheduled.warps[0].instructions),
              "barrier schedule remains valid");
  ++checks;

  KernelProgram chain =
      parseAssemblyText("ADD R1, R2, R3\nMUL R4, R1, R5\nFMA R6, R4, R7\n"
                        "ADD R8, R6, R9\n",
                        config);
  KernelProgram chain_scheduled = scheduleProgram(chain, SchedulerKind::StallFill,
                                                  config);
  DependencyGraph chain_graph(chain.warps[0].instructions);
  ComparisonMetrics chain_comparison =
      compareMetrics(simulateProgram(chain, config),
                     simulateProgram(chain_scheduled, config), chain_graph,
                     static_cast<int>(chain_scheduled.warps[0].instructions.size()));
  requireTrue(chain_comparison.throughput_percent < 10.0,
              "serial dependency chain has limited improvement");
  ++checks;

  return checks;
}
