#pragma once

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <string>
#include <vector>

namespace dataflow {
struct Graph {
  int64_t m, k, n, p = 0; // p == 0: one matmul; otherwise a second projection.
};
struct Hardware {
  int64_t rows = 2, cols = 2, banks = 2, bankBytes = 16384;
  int64_t macs = 64, elements = 16, geluCycles = 8;
  int64_t dramBytes = 16, linkBytes = 32, startup = 8;
  llvm::json::Object json() const;
};
struct Options {
  std::string strategy = "heuristic", placement = "search";
  std::string tiles = "8x8x8,16x16x16,16x32x16,32x16x16";
  bool emitAll = false;
};
// Diagnostics are returned explicitly; invalid configurations never produce a
// plan.
bool readHardware(const std::string &path, Hardware &hw, std::string &error);
bool plan(const Graph &graph, const Hardware &hw, const Options &options,
          llvm::json::Object &result, std::string &error);
void registerPlanPass();
} // namespace dataflow
