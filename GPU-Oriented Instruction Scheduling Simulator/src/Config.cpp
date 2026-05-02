#include "Config.hpp"

#include "Json.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpu_sched {

GpuConfig defaultConfig() {
  GpuConfig config;
  config.latencies = {{Opcode::ADD, 1}, {Opcode::MUL, 4}, {Opcode::FMA, 4},
                      {Opcode::LD, 20}, {Opcode::ST, 10}, {Opcode::SFU, 12},
                      {Opcode::BRA, 2}, {Opcode::BAR, 8}, {Opcode::NOP, 1}};
  config.units = {{ExecUnit::INT, 1}, {ExecUnit::FP, 1}, {ExecUnit::MEM, 1},
                  {ExecUnit::SFU, 1}, {ExecUnit::BRANCH, 1},
                  {ExecUnit::NONE, 1024}};
  return config;
}

GpuConfig loadConfig(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open config: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();

  GpuConfig config = defaultConfig();
  JsonValue root = parseJson(buffer.str());
  if (root.contains("latencies")) {
    for (const auto &entry : root["latencies"].asObject()) {
      config.latencies[opcodeFromString(entry.first)] = entry.second.asInt();
    }
  }
  if (root.contains("units")) {
    for (const auto &entry : root["units"].asObject()) {
      config.units[execUnitFromString(entry.first)] = entry.second.asInt();
    }
  }
  if (root.contains("warp_size")) {
    config.warp_size = root["warp_size"].asInt();
  }
  if (root.contains("issue_width")) {
    config.issue_width = root["issue_width"].asInt();
  }
  if (root.contains("max_warps")) {
    config.max_warps = root["max_warps"].asInt();
  }
  if (root.contains("sms")) {
    config.sms = root["sms"].asInt();
  }
  return config;
}

void applyLatencies(KernelProgram &program, const GpuConfig &config) {
  for (WarpProgram &warp : program.warps) {
    for (Instruction &inst : warp.instructions) {
      inst.latency = latencyFor(inst.opcode, config);
      inst.unit = inst.unit == ExecUnit::NONE && inst.opcode != Opcode::NOP
                      ? defaultUnitForOpcode(inst.opcode)
                      : inst.unit;
    }
  }
}

int latencyFor(Opcode opcode, const GpuConfig &config) {
  auto it = config.latencies.find(opcode);
  return it == config.latencies.end() ? 1 : it->second;
}

int unitCapacity(ExecUnit unit, const GpuConfig &config) {
  auto it = config.units.find(unit);
  return it == config.units.end() ? 1 : it->second;
}

} // namespace gpu_sched
