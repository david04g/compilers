#pragma once

#include "Instruction.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_sched {

struct DependencyEdge {
  int from = 0;
  int to = 0;
  DependencyType type = DependencyType::RAW;
  int latency = 0;
  std::string resource;
};

class DependencyGraph {
public:
  explicit DependencyGraph(const std::vector<Instruction> &instructions);

  const std::vector<Instruction> &instructions() const { return instructions_; }
  const std::vector<DependencyEdge> &edges() const { return edges_; }
  const std::vector<int> &successors(int id) const { return successors_.at(id); }
  const std::vector<int> &predecessors(int id) const { return predecessors_.at(id); }

  bool validateSchedule(const std::vector<Instruction> &schedule,
                        std::string *error = nullptr) const;
  std::vector<int> criticalPath() const;
  std::string toDot() const;
  bool hasCycle() const;

private:
  void build();
  void addEdge(int from, int to, DependencyType type, int latency,
               const std::string &resource);

  std::vector<Instruction> instructions_;
  std::vector<DependencyEdge> edges_;
  std::vector<std::vector<int>> successors_;
  std::vector<std::vector<int>> predecessors_;
  std::unordered_map<int, int> id_to_index_;
};

} // namespace gpu_sched
