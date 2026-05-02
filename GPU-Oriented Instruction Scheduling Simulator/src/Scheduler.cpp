#include "Scheduler.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace gpu_sched {
namespace {

std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

std::vector<Instruction> topologicalSchedule(const std::vector<Instruction> &instructions,
                                             SchedulerKind kind,
                                             const DependencyGraph &graph) {
  std::vector<int> critical_path = graph.criticalPath();
  std::unordered_map<int, int> index_for_id;
  for (size_t i = 0; i < instructions.size(); ++i) {
    index_for_id[instructions[i].id] = static_cast<int>(i);
  }

  std::vector<int> remaining_preds(instructions.size(), 0);
  std::vector<int> ready;
  for (size_t i = 0; i < instructions.size(); ++i) {
    remaining_preds[i] = static_cast<int>(graph.predecessors(instructions[i].id).size());
    if (remaining_preds[i] == 0) {
      ready.push_back(instructions[i].id);
    }
  }

  std::unordered_map<Register, int> pseudo_ready_cycle;
  std::vector<Instruction> scheduled;
  scheduled.reserve(instructions.size());

  auto operandsReady = [&](const Instruction &inst, int cycle) {
    for (const Register &src : inst.srcs) {
      auto it = pseudo_ready_cycle.find(src);
      if (it != pseudo_ready_cycle.end() && it->second > cycle) {
        return false;
      }
    }
    return true;
  };

  while (!ready.empty()) {
    int pseudo_cycle = static_cast<int>(scheduled.size());
    auto best = ready.begin();
    for (auto it = ready.begin(); it != ready.end(); ++it) {
      const Instruction &candidate = instructions[index_for_id[*it]];
      const Instruction &current = instructions[index_for_id[*best]];
      int candidate_index = index_for_id[candidate.id];
      int current_index = index_for_id[current.id];

      bool candidate_better = false;
      if (kind == SchedulerKind::List) {
        candidate_better = candidate.original_index < current.original_index;
      } else if (kind == SchedulerKind::LatencyAware) {
        int candidate_score = critical_path[candidate_index] + candidate.latency +
                              static_cast<int>(graph.successors(candidate.id).size());
        int current_score = critical_path[current_index] + current.latency +
                            static_cast<int>(graph.successors(current.id).size());
        candidate_better =
            candidate_score > current_score ||
            (candidate_score == current_score &&
             candidate.original_index < current.original_index);
      } else if (kind == SchedulerKind::StallFill) {
        bool candidate_ready = operandsReady(candidate, pseudo_cycle);
        bool current_ready = operandsReady(current, pseudo_cycle);
        if (candidate_ready != current_ready) {
          candidate_better = candidate_ready;
        } else if (critical_path[candidate_index] != critical_path[current_index]) {
          candidate_better = critical_path[candidate_index] > critical_path[current_index];
        } else if (candidate.latency != current.latency) {
          candidate_better = candidate.latency > current.latency;
        } else if (graph.successors(candidate.id).size() !=
                   graph.successors(current.id).size()) {
          candidate_better = graph.successors(candidate.id).size() >
                             graph.successors(current.id).size();
        } else {
          candidate_better = candidate.original_index < current.original_index;
        }
      }

      if (candidate_better) {
        best = it;
      }
    }

    int chosen_id = *best;
    ready.erase(best);
    const Instruction &chosen = instructions[index_for_id[chosen_id]];
    scheduled.push_back(chosen);
    if (chosen.dst) {
      pseudo_ready_cycle[*chosen.dst] = pseudo_cycle + chosen.latency;
    }

    for (int succ : graph.successors(chosen_id)) {
      int succ_index = index_for_id[succ];
      if (--remaining_preds[succ_index] == 0) {
        ready.push_back(succ);
      }
    }
  }

  if (scheduled.size() != instructions.size()) {
    throw std::runtime_error("dependency graph contains a cycle");
  }
  return scheduled;
}

} // namespace

SchedulerKind schedulerKindFromString(const std::string &text) {
  std::string value = lower(text);
  if (value == "baseline")
    return SchedulerKind::Baseline;
  if (value == "list")
    return SchedulerKind::List;
  if (value == "latency-aware" || value == "latency_aware")
    return SchedulerKind::LatencyAware;
  if (value == "stall-fill" || value == "stall_fill")
    return SchedulerKind::StallFill;
  throw std::invalid_argument("unknown scheduler: " + text);
}

std::string toString(SchedulerKind kind) {
  switch (kind) {
  case SchedulerKind::Baseline:
    return "baseline";
  case SchedulerKind::List:
    return "list";
  case SchedulerKind::LatencyAware:
    return "latency-aware";
  case SchedulerKind::StallFill:
    return "stall-fill";
  }
  return "baseline";
}

std::vector<Instruction> scheduleInstructions(const std::vector<Instruction> &instructions,
                                              SchedulerKind kind,
                                              const GpuConfig &,
                                              const DependencyGraph &graph) {
  if (kind == SchedulerKind::Baseline) {
    return instructions;
  }
  std::vector<Instruction> scheduled = topologicalSchedule(instructions, kind, graph);
  std::string error;
  if (!graph.validateSchedule(scheduled, &error)) {
    throw std::runtime_error("invalid scheduled program: " + error);
  }
  return scheduled;
}

KernelProgram scheduleProgram(const KernelProgram &program, SchedulerKind kind,
                              const GpuConfig &config) {
  KernelProgram result = program;
  for (WarpProgram &warp : result.warps) {
    DependencyGraph graph(warp.instructions);
    warp.instructions = scheduleInstructions(warp.instructions, kind, config, graph);
  }
  return result;
}

} // namespace gpu_sched
