#include "DependencyGraph.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>

namespace gpu_sched {

DependencyGraph::DependencyGraph(const std::vector<Instruction> &instructions)
    : instructions_(instructions), successors_(instructions.size()),
      predecessors_(instructions.size()) {
  for (size_t i = 0; i < instructions_.size(); ++i) {
    id_to_index_[instructions_[i].id] = static_cast<int>(i);
  }
  build();
}

void DependencyGraph::addEdge(int from, int to, DependencyType type, int latency,
                              const std::string &resource) {
  if (from == to) {
    return;
  }
  for (const DependencyEdge &edge : edges_) {
    if (edge.from == from && edge.to == to && edge.type == type &&
        edge.resource == resource) {
      return;
    }
  }
  edges_.push_back(DependencyEdge{from, to, type, latency, resource});
  successors_[id_to_index_.at(from)].push_back(to);
  predecessors_[id_to_index_.at(to)].push_back(from);
}

void DependencyGraph::build() {
  std::unordered_map<Register, int> last_writer;
  std::unordered_map<Register, std::vector<int>> pending_readers;
  std::vector<int> memory_ops;
  int last_store = -1;
  int last_barrier = -1;
  int last_control = -1;

  for (size_t i = 0; i < instructions_.size(); ++i) {
    const Instruction &inst = instructions_[i];

    if (last_barrier >= 0) {
      addEdge(last_barrier, inst.id, DependencyType::BARRIER, 0, "BAR");
    }
    if (last_control >= 0) {
      addEdge(last_control, inst.id, DependencyType::CONTROL, 0, "BRA");
    }

    for (const Register &src : inst.srcs) {
      auto writer = last_writer.find(src);
      if (writer != last_writer.end()) {
        addEdge(writer->second, inst.id, DependencyType::RAW,
                instructions_[id_to_index_.at(writer->second)].latency, src);
      }
      pending_readers[src].push_back(inst.id);
    }

    if (inst.dst) {
      auto writer = last_writer.find(*inst.dst);
      if (writer != last_writer.end()) {
        addEdge(writer->second, inst.id, DependencyType::WAW, 0, *inst.dst);
      }
      auto readers = pending_readers.find(*inst.dst);
      if (readers != pending_readers.end()) {
        for (int reader : readers->second) {
          addEdge(reader, inst.id, DependencyType::WAR, 0, *inst.dst);
        }
        readers->second.clear();
      }
      last_writer[*inst.dst] = inst.id;
    }

    if (isMemory(inst)) {
      if (isStore(inst)) {
        for (int previous_memory : memory_ops) {
          addEdge(previous_memory, inst.id, DependencyType::MEMORY, 0, "mem");
        }
        last_store = inst.id;
      } else if (isLoad(inst) && last_store >= 0) {
        addEdge(last_store, inst.id, DependencyType::MEMORY, 0, "mem");
      }
      memory_ops.push_back(inst.id);
    }

    if (isBarrier(inst)) {
      for (size_t prior = 0; prior < i; ++prior) {
        addEdge(instructions_[prior].id, inst.id, DependencyType::BARRIER, 0,
                "BAR");
      }
      last_barrier = inst.id;
    }

    if (isBranch(inst)) {
      for (size_t prior = 0; prior < i; ++prior) {
        addEdge(instructions_[prior].id, inst.id, DependencyType::CONTROL, 0,
                "BRA");
      }
      last_control = inst.id;
    }
  }
}

bool DependencyGraph::validateSchedule(const std::vector<Instruction> &schedule,
                                       std::string *error) const {
  std::unordered_map<int, int> position;
  for (size_t i = 0; i < schedule.size(); ++i) {
    position[schedule[i].id] = static_cast<int>(i);
  }
  if (position.size() != instructions_.size()) {
    if (error) {
      *error = "scheduled instruction count does not match input";
    }
    return false;
  }
  for (const DependencyEdge &edge : edges_) {
    if (position.find(edge.from) == position.end() ||
        position.find(edge.to) == position.end()) {
      if (error) {
        *error = "schedule is missing dependency endpoint";
      }
      return false;
    }
    if (position[edge.from] >= position[edge.to]) {
      if (error) {
        *error = "dependency violation: " + std::to_string(edge.from) + " -> " +
                 std::to_string(edge.to) + " (" + toString(edge.type) + ")";
      }
      return false;
    }
  }
  return true;
}

std::vector<int> DependencyGraph::criticalPath() const {
  std::vector<int> memo(instructions_.size(), -1);
  std::function<int(int)> visit = [&](int id) {
    int index = id_to_index_.at(id);
    if (memo[index] >= 0) {
      return memo[index];
    }
    int best_successor = 0;
    for (int succ : successors_[index]) {
      best_successor = std::max(best_successor, visit(succ));
    }
    memo[index] = instructions_[index].latency + best_successor;
    return memo[index];
  };
  for (const Instruction &inst : instructions_) {
    visit(inst.id);
  }
  return memo;
}

bool DependencyGraph::hasCycle() const {
  std::vector<int> indegree(instructions_.size(), 0);
  for (size_t i = 0; i < instructions_.size(); ++i) {
    indegree[i] = static_cast<int>(predecessors_[i].size());
  }
  std::queue<int> ready;
  for (size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) {
      ready.push(static_cast<int>(i));
    }
  }
  int visited = 0;
  while (!ready.empty()) {
    int index = ready.front();
    ready.pop();
    ++visited;
    for (int succ : successors_[index]) {
      int succ_index = id_to_index_.at(succ);
      if (--indegree[succ_index] == 0) {
        ready.push(succ_index);
      }
    }
  }
  return visited != static_cast<int>(instructions_.size());
}

std::string DependencyGraph::toDot() const {
  std::ostringstream out;
  out << "digraph deps {\n";
  for (const Instruction &inst : instructions_) {
    out << "  " << inst.id << " [label=\"" << inst.id << ": "
        << formatInstruction(inst) << "\"];\n";
  }
  out << "\n";
  for (const DependencyEdge &edge : edges_) {
    out << "  " << edge.from << " -> " << edge.to << " [label=\""
        << toString(edge.type);
    if (!edge.resource.empty()) {
      out << " " << edge.resource;
    }
    out << "\"];\n";
  }
  out << "}\n";
  return out.str();
}

} // namespace gpu_sched
