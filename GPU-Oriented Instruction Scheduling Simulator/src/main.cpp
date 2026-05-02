#include "DependencyGraph.hpp"
#include "Metrics.hpp"
#include "Parser.hpp"
#include "Scheduler.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace gpu_sched;

namespace {

struct CliOptions {
  std::string input;
  std::string config = "configs/default_gpu.json";
  SchedulerKind scheduler = SchedulerKind::Baseline;
  bool metrics = false;
  bool emit_schedule = false;
  bool emit_dot = false;
  std::string output = "results";
};

void printUsage(std::ostream &out) {
  out << "usage: gpu_sched_sim --input <file> [--config <file>] "
         "[--scheduler baseline|list|latency-aware|stall-fill] "
         "[--metrics] [--emit-schedule] [--emit-dot] [--output <dir>]\n";
}

CliOptions parseArgs(int argc, char **argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto needValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(name + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--input") {
      options.input = needValue(arg);
    } else if (arg == "--config") {
      options.config = needValue(arg);
    } else if (arg == "--scheduler") {
      options.scheduler = schedulerKindFromString(needValue(arg));
    } else if (arg == "--metrics") {
      options.metrics = true;
    } else if (arg == "--emit-schedule") {
      options.emit_schedule = true;
    } else if (arg == "--emit-dot") {
      options.emit_dot = true;
    } else if (arg == "--output") {
      options.output = needValue(arg);
    } else if (arg == "--help" || arg == "-h") {
      printUsage(std::cout);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (options.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  return options;
}

void writeTextFile(const fs::path &path, const std::string &text) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("failed to write " + path.string());
  }
  file << text;
}

} // namespace

int main(int argc, char **argv) {
  try {
    CliOptions options = parseArgs(argc, argv);
    GpuConfig config = loadConfig(options.config);
    KernelProgram original = parseInputFile(options.input, config);
    KernelProgram scheduled = scheduleProgram(original, options.scheduler, config);

    SimulationMetrics baseline_metrics = simulateProgram(original, config);
    SimulationMetrics scheduled_metrics = simulateProgram(scheduled, config);

    DependencyGraph graph(original.warps.front().instructions);
    int schedule_length = 0;
    for (const WarpProgram &warp : scheduled.warps) {
      schedule_length += static_cast<int>(warp.instructions.size());
    }

    ComparisonMetrics comparison =
        compareMetrics(baseline_metrics, scheduled_metrics, graph, schedule_length);

    fs::path output_dir(options.output);
    if (options.emit_schedule || options.emit_dot || options.metrics) {
      fs::create_directories(output_dir);
    }
    if (options.emit_schedule) {
      writeTextFile(output_dir / "schedule.asm", formatProgramAssembly(scheduled));
    }
    if (options.emit_dot) {
      writeTextFile(output_dir / "deps.dot", graph.toDot());
    }
    if (options.metrics) {
      std::string json = metricsToJson(original.kernel, toString(options.scheduler),
                                       comparison);
      writeTextFile(output_dir / "metrics.json", json);
      std::cout << metricsToText(original.kernel, toString(options.scheduler),
                                 comparison);
    }
    if (!options.metrics && !options.emit_schedule && !options.emit_dot) {
      std::cout << formatProgramAssembly(scheduled);
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    printUsage(std::cerr);
    return 1;
  }
}
