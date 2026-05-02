#pragma once

#include "Config.hpp"
#include "Instruction.hpp"

#include <stdexcept>
#include <string>

namespace gpu_sched {

class ParseError : public std::runtime_error {
public:
  explicit ParseError(const std::string &message) : std::runtime_error(message) {}
};

KernelProgram parseAssemblyText(const std::string &text, const GpuConfig &config);
KernelProgram parseJsonIrText(const std::string &text, const GpuConfig &config);
KernelProgram parseInputFile(const std::string &path, const GpuConfig &config);

} // namespace gpu_sched
