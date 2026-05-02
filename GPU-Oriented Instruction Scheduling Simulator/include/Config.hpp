#pragma once

#include "Instruction.hpp"

#include <string>
#include <unordered_map>

namespace gpu_sched {

struct GpuConfig {
  std::unordered_map<Opcode, int> latencies;
  std::unordered_map<ExecUnit, int> units;
  int warp_size = 32;
  int issue_width = 1;
  int max_warps = 8;
  int sms = 1;
};

GpuConfig defaultConfig();
GpuConfig loadConfig(const std::string &path);
void applyLatencies(KernelProgram &program, const GpuConfig &config);
int latencyFor(Opcode opcode, const GpuConfig &config);
int unitCapacity(ExecUnit unit, const GpuConfig &config);

} // namespace gpu_sched
