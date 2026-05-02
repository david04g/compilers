#include "Instruction.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace gpu_sched {
namespace {

std::string upper(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return text;
}

} // namespace

std::string toString(Opcode opcode) {
  switch (opcode) {
  case Opcode::ADD:
    return "ADD";
  case Opcode::MUL:
    return "MUL";
  case Opcode::FMA:
    return "FMA";
  case Opcode::LD:
    return "LD";
  case Opcode::ST:
    return "ST";
  case Opcode::SFU:
    return "SFU";
  case Opcode::BRA:
    return "BRA";
  case Opcode::BAR:
    return "BAR";
  case Opcode::NOP:
    return "NOP";
  }
  return "NOP";
}

std::string toString(ExecUnit unit) {
  switch (unit) {
  case ExecUnit::INT:
    return "INT";
  case ExecUnit::FP:
    return "FP";
  case ExecUnit::MEM:
    return "MEM";
  case ExecUnit::SFU:
    return "SFU";
  case ExecUnit::BRANCH:
    return "BRANCH";
  case ExecUnit::NONE:
    return "NONE";
  }
  return "NONE";
}

std::string toString(DependencyType type) {
  switch (type) {
  case DependencyType::RAW:
    return "RAW";
  case DependencyType::WAR:
    return "WAR";
  case DependencyType::WAW:
    return "WAW";
  case DependencyType::MEMORY:
    return "MEMORY";
  case DependencyType::BARRIER:
    return "BARRIER";
  case DependencyType::CONTROL:
    return "CONTROL";
  }
  return "RAW";
}

Opcode opcodeFromString(const std::string &text) {
  std::string op = upper(text);
  if (op == "ADD")
    return Opcode::ADD;
  if (op == "MUL")
    return Opcode::MUL;
  if (op == "FMA")
    return Opcode::FMA;
  if (op == "LD")
    return Opcode::LD;
  if (op == "ST")
    return Opcode::ST;
  if (op == "SFU")
    return Opcode::SFU;
  if (op == "BRA")
    return Opcode::BRA;
  if (op == "BAR")
    return Opcode::BAR;
  if (op == "NOP")
    return Opcode::NOP;
  throw std::invalid_argument("unknown opcode: " + text);
}

ExecUnit execUnitFromString(const std::string &text) {
  std::string unit = upper(text);
  if (unit == "INT")
    return ExecUnit::INT;
  if (unit == "FP")
    return ExecUnit::FP;
  if (unit == "MEM")
    return ExecUnit::MEM;
  if (unit == "SFU")
    return ExecUnit::SFU;
  if (unit == "BRANCH")
    return ExecUnit::BRANCH;
  if (unit == "NONE")
    return ExecUnit::NONE;
  throw std::invalid_argument("unknown execution unit: " + text);
}

ExecUnit defaultUnitForOpcode(Opcode opcode) {
  switch (opcode) {
  case Opcode::ADD:
  case Opcode::MUL:
    return ExecUnit::INT;
  case Opcode::FMA:
    return ExecUnit::FP;
  case Opcode::LD:
  case Opcode::ST:
    return ExecUnit::MEM;
  case Opcode::SFU:
    return ExecUnit::SFU;
  case Opcode::BRA:
  case Opcode::BAR:
    return ExecUnit::BRANCH;
  case Opcode::NOP:
    return ExecUnit::NONE;
  }
  return ExecUnit::NONE;
}

bool isLoad(const Instruction &inst) { return inst.opcode == Opcode::LD; }
bool isStore(const Instruction &inst) { return inst.opcode == Opcode::ST; }
bool isMemory(const Instruction &inst) { return isLoad(inst) || isStore(inst); }
bool isBarrier(const Instruction &inst) { return inst.opcode == Opcode::BAR; }
bool isBranch(const Instruction &inst) { return inst.opcode == Opcode::BRA; }

std::string formatInstruction(const Instruction &inst) {
  std::ostringstream out;
  if (!inst.label.empty()) {
    out << inst.label << ": ";
  }
  out << toString(inst.opcode);
  switch (inst.opcode) {
  case Opcode::LD:
    if (inst.dst && inst.memory) {
      out << " " << *inst.dst << ", " << inst.memory->text;
    }
    break;
  case Opcode::ST:
    if (inst.memory && !inst.srcs.empty()) {
      out << " " << inst.memory->text << ", " << inst.srcs.back();
    }
    break;
  case Opcode::ADD:
  case Opcode::MUL:
  case Opcode::FMA:
  case Opcode::SFU:
    if (inst.dst) {
      out << " " << *inst.dst;
      for (const Register &src : inst.srcs) {
        out << ", " << src;
      }
    }
    break;
  case Opcode::BRA:
    out << " " << inst.branch_target;
    break;
  case Opcode::BAR:
  case Opcode::NOP:
    break;
  }
  return out.str();
}

std::string formatProgramAssembly(const KernelProgram &program) {
  std::ostringstream out;
  out << "kernel " << program.kernel << "\n";
  for (const WarpProgram &warp : program.warps) {
    out << "warp " << warp.id << "\n\n";
    for (const Instruction &inst : warp.instructions) {
      out << formatInstruction(inst) << "\n";
    }
    out << "END\n";
  }
  return out.str();
}

} // namespace gpu_sched
