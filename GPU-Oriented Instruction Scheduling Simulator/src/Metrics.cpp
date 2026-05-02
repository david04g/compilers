#include "Metrics.hpp"

#include "Json.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace gpu_sched {
namespace {

void appendSimulationJson(std::ostringstream &out, const SimulationMetrics &metrics,
                          int indent) {
  std::string pad(indent, ' ');
  out << pad << "\"cycles\": " << metrics.total_cycles << ",\n";
  out << pad << "\"ipc\": " << std::fixed << std::setprecision(4) << metrics.ipc
      << ",\n";
  out << pad << "\"stall_cycles\": " << metrics.stall_cycles << ",\n";
  out << pad << "\"raw_stalls\": " << metrics.hazards.raw_stalls << ",\n";
  out << pad << "\"structural_stalls\": " << metrics.hazards.structural_stalls
      << ",\n";
  out << pad << "\"memory_stalls\": " << metrics.hazards.memory_stalls << ",\n";
  out << pad << "\"barrier_stalls\": " << metrics.hazards.barrier_stalls << ",\n";
  out << pad << "\"issued_instructions\": " << metrics.issued_instructions
      << ",\n";
  out << pad << "\"average_issue_utilization\": " << std::fixed
      << std::setprecision(4) << metrics.average_issue_utilization << ",\n";
  out << pad << "\"unit_utilization\": {\n";
  out << pad << "  \"INT\": " << metrics.unit_usage.int_issues << ",\n";
  out << pad << "  \"FP\": " << metrics.unit_usage.fp_issues << ",\n";
  out << pad << "  \"MEM\": " << metrics.unit_usage.mem_issues << ",\n";
  out << pad << "  \"SFU\": " << metrics.unit_usage.sfu_issues << ",\n";
  out << pad << "  \"BRANCH\": " << metrics.unit_usage.branch_issues << "\n";
  out << pad << "}\n";
}

} // namespace

ComparisonMetrics compareMetrics(const SimulationMetrics &baseline,
                                 const SimulationMetrics &scheduled,
                                 const DependencyGraph &graph,
                                 int schedule_length) {
  ComparisonMetrics result;
  result.baseline = baseline;
  result.scheduled = scheduled;
  if (baseline.ipc > 0.0) {
    result.throughput_percent = ((scheduled.ipc - baseline.ipc) / baseline.ipc) * 100.0;
  }
  if (baseline.stall_cycles > 0) {
    result.stall_reduction_percent =
        (static_cast<double>(baseline.stall_cycles - scheduled.stall_cycles) /
         static_cast<double>(baseline.stall_cycles)) *
        100.0;
  }
  std::vector<int> cp = graph.criticalPath();
  result.critical_path_length = cp.empty() ? 0 : *std::max_element(cp.begin(), cp.end());
  result.schedule_length = schedule_length;
  return result;
}

std::string metricsToJson(const std::string &kernel, const std::string &scheduler,
                          const ComparisonMetrics &metrics) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"kernel\": \"" << escapeJson(kernel) << "\",\n";
  out << "  \"scheduler\": \"" << escapeJson(scheduler) << "\",\n";
  out << "  \"baseline\": {\n";
  appendSimulationJson(out, metrics.baseline, 4);
  out << "  },\n";
  out << "  \"scheduled\": {\n";
  appendSimulationJson(out, metrics.scheduled, 4);
  out << "  },\n";
  out << "  \"improvement\": {\n";
  out << "    \"throughput_percent\": " << std::fixed << std::setprecision(2)
      << metrics.throughput_percent << ",\n";
  out << "    \"stall_reduction_percent\": " << std::fixed << std::setprecision(2)
      << metrics.stall_reduction_percent << "\n";
  out << "  },\n";
  out << "  \"critical_path_length\": " << metrics.critical_path_length << ",\n";
  out << "  \"schedule_length\": " << metrics.schedule_length << "\n";
  out << "}\n";
  return out.str();
}

std::string metricsToText(const std::string &kernel, const std::string &scheduler,
                          const ComparisonMetrics &metrics) {
  std::ostringstream out;
  out << "Kernel: " << kernel << "\n";
  out << "Scheduler: " << scheduler << "\n\n";
  out << "Baseline cycles: " << metrics.baseline.total_cycles << "\n";
  out << "Scheduled cycles: " << metrics.scheduled.total_cycles << "\n\n";
  out << "Baseline IPC: " << std::fixed << std::setprecision(2)
      << metrics.baseline.ipc << "\n";
  out << "Scheduled IPC: " << std::fixed << std::setprecision(2)
      << metrics.scheduled.ipc << "\n\n";
  out << "RAW stalls before: " << metrics.baseline.hazards.raw_stalls << "\n";
  out << "RAW stalls after: " << metrics.scheduled.hazards.raw_stalls << "\n";
  out << "Structural stalls after: " << metrics.scheduled.hazards.structural_stalls
      << "\n";
  out << "Memory stalls after: " << metrics.scheduled.hazards.memory_stalls << "\n";
  out << "Barrier stalls after: " << metrics.scheduled.hazards.barrier_stalls
      << "\n\n";
  out << "Throughput improvement: " << std::fixed << std::setprecision(1)
      << metrics.throughput_percent << "%\n";
  out << "Stall reduction: " << std::fixed << std::setprecision(1)
      << metrics.stall_reduction_percent << "%\n";
  out << "Critical path length: " << metrics.critical_path_length << "\n";
  out << "Schedule length: " << metrics.schedule_length << "\n";
  return out.str();
}

} // namespace gpu_sched
