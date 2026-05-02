#include "Parser.hpp"

#include "Json.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace gpu_sched {
namespace {

std::string trim(const std::string &text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

std::string upper(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return text;
}

std::string stripComment(const std::string &line) {
  size_t semi = line.find(';');
  size_t hash = line.find('#');
  size_t cut = std::min(semi == std::string::npos ? line.size() : semi,
                        hash == std::string::npos ? line.size() : hash);
  return line.substr(0, cut);
}

std::vector<std::string> splitOperands(const std::string &text) {
  std::vector<std::string> operands;
  std::string current;
  int bracket_depth = 0;
  for (char c : text) {
    if (c == '[') {
      ++bracket_depth;
    } else if (c == ']') {
      --bracket_depth;
    }
    if (c == ',' && bracket_depth == 0) {
      operands.push_back(trim(current));
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!trim(current).empty()) {
    operands.push_back(trim(current));
  }
  return operands;
}

bool isRegisterToken(const std::string &token) {
  if (token.size() < 2 || (token[0] != 'R' && token[0] != 'r')) {
    return false;
  }
  for (size_t i = 1; i < token.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(token[i]))) {
      return false;
    }
  }
  return true;
}

Register parseRegister(const std::string &token, int line_number) {
  std::string reg = upper(trim(token));
  if (!isRegisterToken(reg)) {
    throw ParseError("line " + std::to_string(line_number) +
                     ": expected register, got '" + token + "'");
  }
  return reg;
}

MemoryOperand parseMemoryOperand(const std::string &token, int line_number) {
  std::string text = trim(token);
  if (text.size() < 3 || text.front() != '[' || text.back() != ']') {
    throw ParseError("line " + std::to_string(line_number) +
                     ": expected memory operand like [R1]");
  }
  std::string base = trim(text.substr(1, text.size() - 2));
  return MemoryOperand{parseRegister(base, line_number), "[" + upper(base) + "]"};
}

Instruction parseInstructionLine(const std::string &raw_line, int line_number,
                                 int id, const GpuConfig &config,
                                 const std::string &pending_label) {
  std::string line = trim(raw_line);
  std::istringstream words(line);
  std::string opcode_text;
  words >> opcode_text;
  if (opcode_text.empty()) {
    throw ParseError("line " + std::to_string(line_number) + ": missing opcode");
  }

  Instruction inst;
  inst.id = id;
  inst.original_index = id;
  inst.opcode = opcodeFromString(opcode_text);
  inst.unit = defaultUnitForOpcode(inst.opcode);
  inst.latency = latencyFor(inst.opcode, config);
  inst.label = pending_label;
  inst.original_text = line;

  std::string rest;
  std::getline(words, rest);
  std::vector<std::string> operands = splitOperands(rest);

  switch (inst.opcode) {
  case Opcode::LD:
    if (operands.size() != 2) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": LD expects dst and memory operands");
    }
    inst.dst = parseRegister(operands[0], line_number);
    inst.memory = parseMemoryOperand(operands[1], line_number);
    inst.srcs.push_back(inst.memory->base);
    break;
  case Opcode::ST:
    if (operands.size() != 2) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": ST expects memory and source operands");
    }
    inst.memory = parseMemoryOperand(operands[0], line_number);
    inst.srcs.push_back(inst.memory->base);
    inst.srcs.push_back(parseRegister(operands[1], line_number));
    break;
  case Opcode::ADD:
  case Opcode::MUL:
  case Opcode::FMA:
    if (operands.size() != 3) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": arithmetic instruction expects dst, src, src");
    }
    inst.dst = parseRegister(operands[0], line_number);
    inst.srcs.push_back(parseRegister(operands[1], line_number));
    inst.srcs.push_back(parseRegister(operands[2], line_number));
    break;
  case Opcode::SFU:
    if (operands.size() != 2) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": SFU expects dst and source operands");
    }
    inst.dst = parseRegister(operands[0], line_number);
    inst.srcs.push_back(parseRegister(operands[1], line_number));
    break;
  case Opcode::BRA:
    if (operands.size() != 1) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": BRA expects one label target");
    }
    inst.branch_target = trim(operands[0]);
    break;
  case Opcode::BAR:
  case Opcode::NOP:
    if (!operands.empty()) {
      throw ParseError("line " + std::to_string(line_number) +
                       ": instruction does not accept operands");
    }
    break;
  }
  return inst;
}

Instruction instructionFromJson(const JsonValue &value, int id,
                                const GpuConfig &config) {
  Instruction inst;
  inst.id = id;
  inst.original_index = id;
  inst.opcode = opcodeFromString(value["opcode"].asString());
  inst.unit = value.contains("unit") ? execUnitFromString(value["unit"].asString())
                                     : defaultUnitForOpcode(inst.opcode);
  inst.latency = latencyFor(inst.opcode, config);
  if (value.contains("dst") && !value["dst"].isNull()) {
    inst.dst = upper(value["dst"].asString());
  }
  if (value.contains("srcs")) {
    for (const JsonValue &src : value["srcs"].asArray()) {
      inst.srcs.push_back(upper(src.asString()));
    }
  }
  if (value.contains("memory")) {
    std::string mem = value["memory"].asString();
    if (!mem.empty() && mem.front() == '[') {
      inst.memory = parseMemoryOperand(mem, 0);
    } else {
      Register base = upper(mem);
      inst.memory = MemoryOperand{base, "[" + base + "]"};
    }
  } else if ((inst.opcode == Opcode::LD || inst.opcode == Opcode::ST) &&
             !inst.srcs.empty()) {
    inst.memory = MemoryOperand{inst.srcs.front(), "[" + inst.srcs.front() + "]"};
  }
  if (value.contains("label")) {
    inst.label = value["label"].asString();
  }
  if (value.contains("target")) {
    inst.branch_target = value["target"].asString();
  }
  return inst;
}

} // namespace

KernelProgram parseAssemblyText(const std::string &text, const GpuConfig &config) {
  KernelProgram program;
  WarpProgram current_warp;
  bool in_warp = false;
  std::string pending_label;
  int next_id = 0;

  std::istringstream input(text);
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    std::string clean = trim(stripComment(line));
    if (clean.empty()) {
      continue;
    }
    std::istringstream directive(clean);
    std::string first;
    directive >> first;
    std::string first_upper = upper(first);
    if (first_upper == "KERNEL") {
      directive >> program.kernel;
      if (program.kernel.empty()) {
        throw ParseError("line " + std::to_string(line_number) +
                         ": kernel name is required");
      }
      continue;
    }
    if (first_upper == "WARP") {
      if (in_warp) {
        program.warps.push_back(current_warp);
      }
      current_warp = WarpProgram{};
      directive >> current_warp.id;
      in_warp = true;
      next_id = 0;
      continue;
    }
    if (first_upper == "END") {
      if (in_warp) {
        program.warps.push_back(current_warp);
        current_warp = WarpProgram{};
        in_warp = false;
      }
      continue;
    }
    if (!in_warp) {
      current_warp = WarpProgram{};
      current_warp.id = 0;
      in_warp = true;
      next_id = 0;
    }

    size_t colon = clean.find(':');
    if (colon != std::string::npos) {
      pending_label = trim(clean.substr(0, colon));
      clean = trim(clean.substr(colon + 1));
      if (clean.empty()) {
        continue;
      }
    }

    current_warp.instructions.push_back(
        parseInstructionLine(clean, line_number, next_id++, config, pending_label));
    pending_label.clear();
  }
  if (in_warp) {
    program.warps.push_back(current_warp);
  }
  if (program.warps.empty()) {
    throw ParseError("input contained no warp instructions");
  }
  return program;
}

KernelProgram parseJsonIrText(const std::string &text, const GpuConfig &config) {
  JsonValue root = parseJson(text);
  KernelProgram program;
  if (root.contains("kernel")) {
    program.kernel = root["kernel"].asString();
  }
  for (const JsonValue &warp_value : root["warps"].asArray()) {
    WarpProgram warp;
    warp.id = warp_value.contains("id") ? warp_value["id"].asInt()
                                        : static_cast<int>(program.warps.size());
    int id = 0;
    for (const JsonValue &inst_value : warp_value["instructions"].asArray()) {
      warp.instructions.push_back(instructionFromJson(inst_value, id++, config));
    }
    program.warps.push_back(warp);
  }
  if (program.warps.empty()) {
    throw ParseError("JSON IR contained no warps");
  }
  return program;
}

KernelProgram parseInputFile(const std::string &path, const GpuConfig &config) {
  std::ifstream file(path);
  if (!file) {
    throw ParseError("failed to open input: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  std::string text = buffer.str();
  std::string trimmed = trim(text);
  if (!trimmed.empty() && trimmed.front() == '{') {
    return parseJsonIrText(text, config);
  }
  return parseAssemblyText(text, config);
}

} // namespace gpu_sched
