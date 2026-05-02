#pragma once

#include "Config.hpp"
#include "Instruction.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gpu_sched {

struct HazardBreakdown {
  int raw_stalls = 0;
  int structural_stalls = 0;
  int memory_stalls = 0;
  int barrier_stalls = 0;
};

struct UnitUsage {
  int int_issues = 0;
  int fp_issues = 0;
  int mem_issues = 0;
  int sfu_issues = 0;
  int branch_issues = 0;
};

struct SimulationMetrics {
  int total_cycles = 0;
  int issued_instructions = 0;
  int completed_instructions = 0;
  int stall_cycles = 0;
  double ipc = 0.0;
  double average_issue_utilization = 0.0;
  HazardBreakdown hazards;
  UnitUsage unit_usage;
};

struct WarpState {
  int warp_id = 0;
  int pc = 0;
  uint32_t active_mask = 0xffffffffu;
  std::vector<Instruction> instructions;
  std::unordered_map<Register, int> ready_cycle;
  int stall_cycles = 0;
  int issued_instructions = 0;
  int completed_instructions = 0;
  bool completed = false;
  int blocked_until = 0;
  int memory_blocked_until = 0;
  int last_memory_complete = 0;
};

SimulationMetrics simulateProgram(const KernelProgram &program, const GpuConfig &config);

} // namespace gpu_sched
