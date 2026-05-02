#pragma once

#include "DependencyGraph.hpp"
#include "Simulator.hpp"

#include <string>

namespace gpu_sched {

struct ComparisonMetrics {
  SimulationMetrics baseline;
  SimulationMetrics scheduled;
  double throughput_percent = 0.0;
  double stall_reduction_percent = 0.0;
  int critical_path_length = 0;
  int schedule_length = 0;
};

ComparisonMetrics compareMetrics(const SimulationMetrics &baseline,
                                 const SimulationMetrics &scheduled,
                                 const DependencyGraph &graph,
                                 int schedule_length);

std::string metricsToJson(const std::string &kernel, const std::string &scheduler,
                          const ComparisonMetrics &metrics);
std::string metricsToText(const std::string &kernel, const std::string &scheduler,
                          const ComparisonMetrics &metrics);

} // namespace gpu_sched
