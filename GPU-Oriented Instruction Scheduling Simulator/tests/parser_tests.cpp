#include "Parser.hpp"

#include "TestHarness.hpp"

using namespace gpu_sched;

int runParserTests() {
  int checks = 0;
  GpuConfig config = defaultConfig();

  KernelProgram asm_program = parseAssemblyText(
      "kernel vector_add\n"
      "warp 0\n"
      "entry: LD R1, [R10]\n"
      "ADD R3, R1, R2\n"
      "ST [R12], R3\n"
      "END\n",
      config);
  requireEqual(asm_program.kernel, std::string("vector_add"), "kernel name parsed");
  ++checks;
  requireEqual(asm_program.warps.size(), static_cast<size_t>(1), "one warp parsed");
  ++checks;
  requireEqual(asm_program.warps[0].instructions.size(), static_cast<size_t>(3),
               "three instructions parsed");
  ++checks;
  requireEqual(toString(asm_program.warps[0].instructions[0].opcode),
               std::string("LD"), "LD parsed");
  ++checks;
  requireEqual(asm_program.warps[0].instructions[0].label, std::string("entry"),
               "label parsed");
  ++checks;
  requireEqual(asm_program.warps[0].instructions[2].memory->base, std::string("R12"),
               "store memory base parsed");
  ++checks;

  KernelProgram json_program = parseJsonIrText(
      "{"
      "\"kernel\":\"json_kernel\","
      "\"warps\":[{\"id\":2,\"instructions\":["
      "{\"opcode\":\"LD\",\"dst\":\"R1\",\"srcs\":[\"R10\"],\"unit\":\"MEM\"},"
      "{\"opcode\":\"ADD\",\"dst\":\"R2\",\"srcs\":[\"R1\",\"R3\"],\"unit\":\"INT\"}"
      "]}]}",
      config);
  requireEqual(json_program.kernel, std::string("json_kernel"), "JSON kernel parsed");
  ++checks;
  requireEqual(json_program.warps[0].id, 2, "JSON warp id parsed");
  ++checks;
  requireEqual(json_program.warps[0].instructions[1].srcs[0], std::string("R1"),
               "JSON source parsed");
  ++checks;

  bool failed = false;
  try {
    parseAssemblyText("ADD R1, R2\n", config);
  } catch (const ParseError &) {
    failed = true;
  }
  requireTrue(failed, "invalid assembly reports a parse error");
  ++checks;

  return checks;
}
