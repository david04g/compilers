#include "DependencyGraph.hpp"
#include "Parser.hpp"
#include "Scheduler.hpp"

#include "TestHarness.hpp"

using namespace gpu_sched;

int runSchedulerTests() {
  int checks = 0;
  GpuConfig config = defaultConfig();
  KernelProgram program =
      parseAssemblyText("LD R1, [R2]\nADD R3, R1, R4\nMUL R5, R6, R7\n", config);
  DependencyGraph graph(program.warps[0].instructions);

  std::vector<Instruction> baseline =
      scheduleInstructions(program.warps[0].instructions, SchedulerKind::Baseline, config,
                           graph);
  requireEqual(toString(baseline[1].opcode), std::string("ADD"),
               "baseline preserves order");
  ++checks;

  std::vector<Instruction> list =
      scheduleInstructions(program.warps[0].instructions, SchedulerKind::List, config,
                           graph);
  requireTrue(graph.validateSchedule(list), "list schedule validates");
  ++checks;

  std::vector<Instruction> latency = scheduleInstructions(
      program.warps[0].instructions, SchedulerKind::LatencyAware, config, graph);
  requireTrue(graph.validateSchedule(latency), "latency-aware schedule validates");
  ++checks;
  requireEqual(toString(latency[0].opcode), std::string("LD"),
               "latency-aware prioritizes long load");
  ++checks;

  std::vector<Instruction> stall_fill = scheduleInstructions(
      program.warps[0].instructions, SchedulerKind::StallFill, config, graph);
  requireTrue(graph.validateSchedule(stall_fill), "stall-fill schedule validates");
  ++checks;
  requireEqual(formatInstruction(stall_fill[0]), std::string("LD R1, [R2]"),
               "stall-fill keeps load first");
  ++checks;
  requireEqual(formatInstruction(stall_fill[1]), std::string("MUL R5, R6, R7"),
               "stall-fill moves independent work into latency slot");
  ++checks;
  requireEqual(formatInstruction(stall_fill[2]), std::string("ADD R3, R1, R4"),
               "stall-fill keeps dependent use after independent work");
  ++checks;

  return checks;
}
