#include "Simulator.hpp"

#include <algorithm>

namespace gpu_sched {
namespace {

void addUnitIssue(UnitUsage &usage, ExecUnit unit) {
  switch (unit) {
  case ExecUnit::INT:
    ++usage.int_issues;
    break;
  case ExecUnit::FP:
    ++usage.fp_issues;
    break;
  case ExecUnit::MEM:
    ++usage.mem_issues;
    break;
  case ExecUnit::SFU:
    ++usage.sfu_issues;
    break;
  case ExecUnit::BRANCH:
    ++usage.branch_issues;
    break;
  case ExecUnit::NONE:
    break;
  }
}

bool allCompleted(const std::vector<WarpState> &warps) {
  return std::all_of(warps.begin(), warps.end(),
                     [](const WarpState &warp) { return warp.completed; });
}

bool operandsReady(const WarpState &warp, const Instruction &inst, int cycle) {
  for (const Register &src : inst.srcs) {
    auto it = warp.ready_cycle.find(src);
    if (it != warp.ready_cycle.end() && it->second > cycle) {
      return false;
    }
  }
  return true;
}

} // namespace

SimulationMetrics simulateProgram(const KernelProgram &program, const GpuConfig &config) {
  SimulationMetrics metrics;
  std::vector<WarpState> warps;
  int warp_count = std::min(static_cast<int>(program.warps.size()), config.max_warps);
  for (int i = 0; i < warp_count; ++i) {
    WarpState state;
    state.warp_id = program.warps[i].id;
    state.active_mask = program.warps[i].active_mask;
    state.instructions = program.warps[i].instructions;
    state.completed = state.instructions.empty();
    warps.push_back(std::move(state));
  }

  if (warps.empty()) {
    return metrics;
  }

  int cycle = 0;
  int next_warp = 0;
  while (!allCompleted(warps)) {
    std::unordered_map<ExecUnit, int> used_units;
    int issued_this_cycle = 0;

    for (int offset = 0; offset < static_cast<int>(warps.size()); ++offset) {
      int index = (next_warp + offset) % static_cast<int>(warps.size());
      WarpState &warp = warps[index];
      if (warp.completed) {
        continue;
      }
      if (warp.pc >= static_cast<int>(warp.instructions.size())) {
        warp.completed = true;
        continue;
      }

      const Instruction &inst = warp.instructions[warp.pc];
      bool stalled = false;
      if (warp.blocked_until > cycle) {
        ++metrics.hazards.barrier_stalls;
        stalled = true;
      } else if ((isStore(inst) && warp.last_memory_complete > cycle) ||
                 (isLoad(inst) && warp.memory_blocked_until > cycle)) {
        ++metrics.hazards.memory_stalls;
        stalled = true;
      } else if (!operandsReady(warp, inst, cycle)) {
        ++metrics.hazards.raw_stalls;
        stalled = true;
      } else if (issued_this_cycle >= config.issue_width ||
                 used_units[inst.unit] >= unitCapacity(inst.unit, config)) {
        ++metrics.hazards.structural_stalls;
        stalled = true;
      }

      if (stalled) {
        ++warp.stall_cycles;
        ++metrics.stall_cycles;
        continue;
      }

      ++used_units[inst.unit];
      ++issued_this_cycle;
      ++metrics.issued_instructions;
      ++metrics.completed_instructions;
      ++warp.issued_instructions;
      ++warp.completed_instructions;
      addUnitIssue(metrics.unit_usage, inst.unit);

      if (inst.dst) {
        warp.ready_cycle[*inst.dst] = cycle + inst.latency;
      }
      if (isMemory(inst)) {
        int complete = cycle + inst.latency;
        warp.last_memory_complete = std::max(warp.last_memory_complete, complete);
        if (isStore(inst)) {
          warp.memory_blocked_until = std::max(warp.memory_blocked_until, complete);
        }
      }
      if (isBarrier(inst)) {
        warp.blocked_until = cycle + inst.latency;
      }

      ++warp.pc;
      if (warp.pc >= static_cast<int>(warp.instructions.size())) {
        warp.completed = true;
      }
      if (issued_this_cycle >= config.issue_width) {
        break;
      }
    }

    next_warp = (next_warp + 1) % static_cast<int>(warps.size());
    ++cycle;
  }

  metrics.total_cycles = cycle;
  metrics.ipc = cycle == 0 ? 0.0
                           : static_cast<double>(metrics.issued_instructions) /
                                 static_cast<double>(cycle);
  metrics.average_issue_utilization =
      cycle == 0 ? 0.0
                 : static_cast<double>(metrics.issued_instructions) /
                       static_cast<double>(cycle * std::max(1, config.issue_width));
  return metrics;
}

} // namespace gpu_sched
