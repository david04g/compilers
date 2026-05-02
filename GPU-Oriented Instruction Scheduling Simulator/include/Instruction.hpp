#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gpu_sched {

using Register = std::string;

enum class Opcode {
  ADD,
  MUL,
  FMA,
  LD,
  ST,
  SFU,
  BRA,
  BAR,
  NOP
};

enum class ExecUnit {
  INT,
  FP,
  MEM,
  SFU,
  BRANCH,
  NONE
};

enum class DependencyType {
  RAW,
  WAR,
  WAW,
  MEMORY,
  BARRIER,
  CONTROL
};

struct MemoryOperand {
  Register base;
  std::string text;
};

struct Instruction {
  int id = 0;
  int original_index = 0;
  Opcode opcode = Opcode::NOP;
  std::optional<Register> dst;
  std::vector<Register> srcs;
  std::optional<MemoryOperand> memory;
  ExecUnit unit = ExecUnit::NONE;
  int latency = 1;
  std::string label;
  std::string branch_target;
  std::string original_text;
};

struct WarpProgram {
  int id = 0;
  uint32_t active_mask = 0xffffffffu;
  std::vector<Instruction> instructions;
};

struct KernelProgram {
  std::string kernel = "kernel";
  std::vector<WarpProgram> warps;
};

std::string toString(Opcode opcode);
std::string toString(ExecUnit unit);
std::string toString(DependencyType type);

Opcode opcodeFromString(const std::string &text);
ExecUnit execUnitFromString(const std::string &text);
ExecUnit defaultUnitForOpcode(Opcode opcode);

bool isLoad(const Instruction &inst);
bool isStore(const Instruction &inst);
bool isMemory(const Instruction &inst);
bool isBarrier(const Instruction &inst);
bool isBranch(const Instruction &inst);

std::string formatInstruction(const Instruction &inst);
std::string formatProgramAssembly(const KernelProgram &program);

} // namespace gpu_sched
