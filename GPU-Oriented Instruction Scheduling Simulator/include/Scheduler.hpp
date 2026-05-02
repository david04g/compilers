#pragma once

#include "Config.hpp"
#include "DependencyGraph.hpp"

#include <string>
#include <vector>

namespace gpu_sched {

enum class SchedulerKind {
  Baseline,
  List,
  LatencyAware,
  StallFill
};

SchedulerKind schedulerKindFromString(const std::string &text);
std::string toString(SchedulerKind kind);

std::vector<Instruction> scheduleInstructions(const std::vector<Instruction> &instructions,
                                              SchedulerKind kind,
                                              const GpuConfig &config,
                                              const DependencyGraph &graph);

KernelProgram scheduleProgram(const KernelProgram &program, SchedulerKind kind,
                              const GpuConfig &config);

} // namespace gpu_sched
