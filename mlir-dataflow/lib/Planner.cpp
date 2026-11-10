#include "dataflow/Planner.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <utility>

using namespace llvm;
namespace dataflow {
namespace {
int64_t ceilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }
int64_t align(int64_t x) { return ceilDiv(x, 16) * 16; }
json::Array integers(const std::vector<int64_t> &xs) {
  json::Array a;
  for (auto x : xs)
    a.push_back(x);
  return a;
}
json::Array strings(const std::vector<std::string> &xs) {
  json::Array a;
  for (const auto &x : xs)
    a.push_back(x);
  return a;
}
struct Buffer {
  std::string name;
  int64_t cu, bank, bytes, first = -1, last = -1, offset = 0;
};
struct Event {
  std::string kind, tensor;
  int64_t cu = 0, cycles = 0, bytes = 0;
  std::vector<int64_t> deps, origin, shape;
  std::vector<std::string> reads, writes, resources;
};
struct Program {
  std::vector<Buffer> buffers;
  std::vector<Event> events;
  Buffer &buffer(const std::string &name) {
    auto it = std::find_if(buffers.begin(), buffers.end(),
                           [&](const Buffer &b) { return b.name == name; });
    assert(it != buffers.end());
    return *it;
  }
};
std::vector<std::string> route(int64_t from, int64_t to, const Hardware &h) {
  std::vector<std::string> links;
  auto step = [&](int64_t next) {
    links.push_back("link:" + std::to_string(from) + ">" +
                    std::to_string(next));
    from = next;
  };
  while (from % h.cols != to % h.cols)
    step(from + (from % h.cols < to % h.cols ? 1 : -1));
  while (from / h.cols != to / h.cols)
    step(from + (from / h.cols < to / h.cols ? h.cols : -h.cols));
  return links;
}
std::string bankKey(int64_t cu, int64_t bank) {
  return "bank:" + std::to_string(cu) + ":" + std::to_string(bank);
}

// Emit a bounded, single-output-tile program. Each K block accumulates into C.
// A region boundary always materializes its output in off-chip memory.
Program emitTile(const Graph &g, const Hardware &h,
                 const std::array<int64_t, 3> &t, int begin, int end,
                 const std::string &placement, int64_t row, int64_t col) {
  Program p;
  auto stageCU = [&](int stage) -> int64_t {
    return (placement == "compact" ? begin : stage) % (h.rows * h.cols);
  };
  int64_t m = std::min(t[0], g.m - row);
  int64_t n = std::min(t[1], (begin == 3 ? g.p : g.n) - col);
  auto addBuffer = [&](std::string name, int64_t cu, int64_t bank,
                       int64_t bytes) {
    p.buffers.push_back({name, cu, bank % h.banks, bytes});
    return name;
  };
  auto add = [&](Event e) -> int64_t {
    std::set<std::string> rs(e.resources.begin(), e.resources.end());
    for (const auto &name : e.reads) {
      const auto &b = p.buffer(name);
      rs.insert(bankKey(b.cu, b.bank));
    }
    for (const auto &name : e.writes) {
      const auto &b = p.buffer(name);
      rs.insert(bankKey(b.cu, b.bank));
    }
    e.resources.assign(rs.begin(), rs.end());
    p.events.push_back(std::move(e));
    return p.events.size() - 1;
  };
  auto dma = [&](bool load, const std::string &buf, const std::string &tensor,
                 std::vector<int64_t> origin, std::vector<int64_t> shape,
                 std::vector<int64_t> deps) {
    Event e;
    e.kind = load ? "load" : "store";
    e.tensor = tensor;
    e.origin = origin;
    e.shape = shape;
    e.deps = deps;
    e.cu = p.buffer(buf).cu;
    e.bytes = 4;
    for (auto s : shape)
      e.bytes *= s;
    e.resources = load ? route(0, e.cu, h) : route(e.cu, 0, h);
    auto bandwidth =
        e.resources.empty() ? h.dramBytes : std::min(h.dramBytes, h.linkBytes);
    e.cycles =
        h.startup * (1 + e.resources.size()) + ceilDiv(e.bytes, bandwidth);
    e.resources.push_back("dram");
    (load ? e.writes : e.reads).push_back(buf);
    return add(std::move(e));
  };
  auto compute = [&](const std::string &kind, int64_t cu, int64_t cycles,
                     std::vector<std::string> reads, std::string write,
                     std::vector<int64_t> shape, std::vector<int64_t> deps) {
    Event e;
    e.kind = kind;
    e.cu = cu;
    e.cycles = cycles;
    e.reads = reads;
    e.writes = {write};
    e.shape = shape;
    e.deps = deps;
    e.resources = {"cu:" + std::to_string(cu)};
    return add(std::move(e));
  };
  std::string current;
  int64_t previous = -1;
  for (int stage = begin; stage <= end; ++stage) {
    int64_t cu = stageCU(stage);
    std::string prefix = "s" + std::to_string(stage) + ".";
    if (stage == 0 || stage == 3) {
      int64_t reduction = stage == 0 ? g.k : g.n;
      auto a =
          addBuffer(prefix + "a", cu, 0, 4 * m * std::min(t[2], reduction));
      auto b =
          addBuffer(prefix + "b", cu, 1, 4 * n * std::min(t[2], reduction));
      current = addBuffer(prefix + "acc", cu, 0, 4 * m * n);
      previous = compute("zero", cu, ceilDiv(m * n, h.elements), {}, current,
                         {m, n}, {});
      for (int64_t k = 0; k < reduction; k += t[2]) {
        int64_t kk = std::min(t[2], reduction - k);
        auto la =
            dma(true, a, stage == 0 ? "A" : "H", {row, k}, {m, kk}, {previous});
        auto lb = dma(true, b, stage == 0 ? "B" : "W2", {k, col}, {kk, n},
                      {previous});
        previous = compute("matmul", cu, ceilDiv(m * n * kk, h.macs),
                           {a, b, current}, current, {m, n, kk}, {la, lb});
      }
      continue;
    }
    if (stage == begin) {
      current = addBuffer(prefix + "in", cu, 0, 4 * m * n);
      previous =
          dma(true, current, stage == 1 ? "I0" : "I1", {row, col}, {m, n}, {});
    } else if (p.buffer(current).cu != cu) {
      auto recv = addBuffer(prefix + "in", cu, 0, 4 * m * n);
      Event e;
      e.kind = "send";
      e.cu = cu;
      e.shape = {m, n};
      e.bytes = 4 * m * n;
      e.resources = route(p.buffer(current).cu, cu, h);
      e.cycles = h.startup * e.resources.size() + ceilDiv(e.bytes, h.linkBytes);
      e.reads = {current};
      e.writes = {recv};
      e.deps = {previous};
      previous = add(std::move(e));
      current = recv;
    }
    auto out = addBuffer(prefix + "out", cu, 0, 4 * m * n);
    if (stage == 1) {
      auto bias = addBuffer(prefix + "bias", cu, 1, 4 * n);
      auto lb = dma(true, bias, "bias", {col}, {n}, {previous});
      previous = compute("bias", cu, ceilDiv(m * n, h.elements),
                         {current, bias}, out, {m, n}, {previous, lb});
    } else {
      previous = compute("gelu", cu, h.geluCycles * ceilDiv(m * n, h.elements),
                         {current}, out, {m, n}, {previous});
    }
    current = out;
  }
  const std::string output = end == 0          ? "I0"
                             : end == 1        ? "I1"
                             : end == 2 && g.p ? "H"
                                               : "Y";
  dma(false, current, output, {row, col}, {m, n}, {previous});
  for (size_t i = 0; i < p.events.size(); ++i) {
    for (const auto &name : p.events[i].reads) {
      auto &b = p.buffer(name);
      if (b.first < 0)
        b.first = i;
      b.last = i;
    }
    for (const auto &name : p.events[i].writes) {
      auto &b = p.buffer(name);
      if (b.first < 0)
        b.first = i;
      b.last = i;
    }
  }
  return p;
}

// First-fit linear-scan arena allocation with inclusive use intervals. Freed
// addresses can be reused by later temporaries; ping-pong slots stay disjoint.
std::map<std::string, int64_t> allocate(Program &p) {
  std::stable_sort(
      p.buffers.begin(), p.buffers.end(),
      [](const Buffer &a, const Buffer &b) { return a.first < b.first; });
  std::map<std::string, int64_t> high;
  std::vector<Buffer *> active;
  for (auto &b : p.buffers) {
    std::vector<std::pair<int64_t, int64_t>> occupied;
    for (auto *a : active)
      if (a->cu == b.cu && a->bank == b.bank && a->last >= b.first)
        occupied.push_back({a->offset, a->offset + align(a->bytes)});
    std::sort(occupied.begin(), occupied.end());
    int64_t offset = 0;
    for (auto [lo, hi] : occupied) {
      if (offset + align(b.bytes) <= lo)
        break;
      offset = std::max(offset, hi);
    }
    b.offset = offset;
    auto key = bankKey(b.cu, b.bank);
    high[key] = std::max(high[key], offset + align(b.bytes));
    active.push_back(&b);
  }
  return high;
}

json::Object candidate(const Graph &g, const Hardware &h,
                       std::array<int64_t, 3> tile, int mode,
                       const std::string &placement) {
  const char *names[] = {"split", "matmul-bias", "bias-gelu", "fused"};
  std::string id = std::string(names[mode]) + "/" + placement + "/" +
                   std::to_string(tile[0]) + "x" + std::to_string(tile[1]) +
                   "x" + std::to_string(tile[2]);
  json::Object result{{"id", id},
                      {"mode", names[mode]},
                      {"placement", placement},
                      {"tile", integers({tile[0], tile[1], tile[2]})},
                      {"legal", true}};
  std::vector<std::pair<int, int>> boundaries;
  int begin = 0;
  for (int stage = 0; stage < 2; ++stage)
    if (!(mode & (1 << stage))) {
      boundaries.push_back({begin, stage});
      begin = stage + 1;
    }
  boundaries.push_back({begin, 2});
  if (g.p)
    boundaries.push_back({3, 3});
  json::Array events, buffers, regions;
  std::map<std::string, int64_t> bankPeaks;
  int64_t dramBytes = 0, peakSRAM = 0, estimate = 0;
  std::vector<int64_t> previousRegion;
  int64_t regionIndex = 0;
  for (auto [first, last] : boundaries) {
    auto prototype = emitTile(g, h, tile, first, last, placement, 0, 0);
    auto high = allocate(prototype);
    int64_t slots = first == last ? 1 : 2;
    int64_t regionSRAM = 0;
    for (auto &[bank, bytes] : high) {
      bytes *= slots;
      regionSRAM += bytes;
      bankPeaks[bank] = std::max(bankPeaks[bank], bytes);
      if (bytes > h.bankBytes) {
        result["legal"] = false;
        result["rejection"] =
            "region " + std::to_string(regionIndex) + " " + bank + " needs " +
            std::to_string(bytes) + " bytes (" + std::to_string(slots) +
            " streaming slot(s)); capacity " + std::to_string(h.bankBytes);
        return result;
      }
    }
    peakSRAM = std::max(peakSRAM, regionSRAM);
    std::string prefix = "r" + std::to_string(regionIndex) + ".";
    for (int64_t slot = 0; slot < slots; ++slot)
      for (const auto &b : prototype.buffers) {
        auto key = bankKey(b.cu, b.bank);
        buffers.push_back(
            json::Object{{"id", prefix + std::to_string(slot) + "." + b.name},
                         {"region", regionIndex},
                         {"slot", slot},
                         {"cu", b.cu},
                         {"bank", b.bank},
                         {"bytes", b.bytes},
                         {"offset", b.offset + slot * high[key] / slots},
                         {"first_use", b.first},
                         {"last_use", b.last}});
      }
    std::vector<int64_t> slotLast(slots, -1), regionLast;
    std::map<std::string, int64_t> loads;
    int64_t tileIndex = 0;
    int64_t singleTileSpan = 0, firstTileBottleneck = 0;
    auto startEvent = events.size();
    for (int64_t i = 0; i < g.m; i += tile[0])
      for (int64_t j = 0; j < (first == 3 ? g.p : g.n); j += tile[1]) {
        auto p = emitTile(g, h, tile, first, last, placement, i, j);
        int64_t slot = tileIndex % slots, base = events.size();
        std::string bp = prefix + std::to_string(slot) + ".";
        std::vector<int64_t> critical(p.events.size(), 0);
        for (size_t ei = 0; ei < p.events.size(); ++ei) {
          auto &e = p.events[ei];
          std::vector<int64_t> deps;
          int64_t ready = 0;
          for (auto d : e.deps) {
            deps.push_back(base + d);
            ready = std::max(ready, critical[d]);
          }
          critical[ei] = ready + e.cycles;
          if (e.deps.empty()) {
            if (slotLast[slot] >= 0)
              deps.push_back(slotLast[slot]);
            else
              deps = previousRegion;
          }
          for (auto &b : e.reads)
            b = bp + b;
          for (auto &b : e.writes)
            b = bp + b;
          for (const auto &resource : e.resources)
            loads[resource] += e.cycles;
          if (e.kind == "load" || e.kind == "store")
            dramBytes += e.bytes;
          events.push_back(
              json::Object{{"id", static_cast<int64_t>(events.size())},
                           {"region", regionIndex},
                           {"tile_index", tileIndex},
                           {"kind", e.kind},
                           {"cu", e.cu},
                           {"cycles", e.cycles},
                           {"bytes", e.bytes},
                           {"tensor", e.tensor},
                           {"origin", integers(e.origin)},
                           {"shape", integers(e.shape)},
                           {"deps", integers(deps)},
                           {"reads", strings(e.reads)},
                           {"writes", strings(e.writes)},
                           {"resources", strings(e.resources)}});
        }
        if (!tileIndex) {
          singleTileSpan = *std::max_element(critical.begin(), critical.end());
          for (const auto &[_, n] : loads)
            firstTileBottleneck = std::max(firstTileBottleneck, n);
        }
        slotLast[slot] = events.size() - 1;
        ++tileIndex;
      }
    int64_t bottleneck = 0;
    std::string bottleneckName;
    for (const auto &[name, n] : loads)
      if (n > bottleneck) {
        bottleneck = n;
        bottleneckName = name;
      }
    // Resource-load estimate plus a first-tile fill/drain correction. It does
    // not simulate dependencies, contention timing, or SRAM slot backpressure.
    int64_t regionEstimate =
        bottleneck + std::max<int64_t>(0, singleTileSpan - firstTileBottleneck);
    estimate += regionEstimate;
    for (auto x : slotLast)
      if (x >= 0)
        regionLast.push_back(x);
    previousRegion = regionLast;
    regions.push_back(json::Object{
        {"id", regionIndex},
        {"first_stage", first},
        {"last_stage", last},
        {"slots", slots},
        {"first_event", static_cast<int64_t>(startEvent)},
        {"event_count", static_cast<int64_t>(events.size() - startEvent)},
        {"estimated_cycles", regionEstimate},
        {"bottleneck", bottleneckName},
        {"reserved_sram_bytes", regionSRAM}});
    ++regionIndex;
  }
  json::Object peaks;
  for (const auto &[bank, bytes] : bankPeaks)
    peaks[bank] = bytes;
  result["estimated_cycles"] = estimate;
  result["offchip_bytes"] = dramBytes;
  result["peak_sram_bytes"] = peakSRAM;
  result["peak_bank_bytes"] = std::move(peaks);
  result["regions"] = std::move(regions);
  result["buffers"] = std::move(buffers);
  result["events"] = std::move(events);
  return result;
}
} // namespace

json::Object Hardware::json() const {
  return json::Object{{"rows", rows},
                      {"cols", cols},
                      {"banks_per_cu", banks},
                      {"bank_bytes", bankBytes},
                      {"macs_per_cycle", macs},
                      {"elements_per_cycle", elements},
                      {"gelu_cycles_per_vector", geluCycles},
                      {"dram_bytes_per_cycle", dramBytes},
                      {"link_bytes_per_cycle", linkBytes},
                      {"transfer_startup_cycles", startup}};
}
bool readHardware(const std::string &path, Hardware &h, std::string &error) {
  auto file = MemoryBuffer::getFile(path);
  if (!file) {
    error = "cannot read hardware: " + file.getError().message();
    return false;
  }
  auto value = json::parse((*file)->getBuffer());
  if (!value) {
    error = toString(value.takeError());
    return false;
  }
  auto *obj = value->getAsObject();
  if (!obj) {
    error = "hardware must be an object";
    return false;
  }
  std::map<std::string, int64_t *> fields = {
      {"rows", &h.rows},
      {"cols", &h.cols},
      {"banks_per_cu", &h.banks},
      {"bank_bytes", &h.bankBytes},
      {"macs_per_cycle", &h.macs},
      {"elements_per_cycle", &h.elements},
      {"gelu_cycles_per_vector", &h.geluCycles},
      {"dram_bytes_per_cycle", &h.dramBytes},
      {"link_bytes_per_cycle", &h.linkBytes},
      {"transfer_startup_cycles", &h.startup}};
  for (const auto &entry : *obj) {
    auto it = fields.find(entry.first.str());
    auto n = entry.second.getAsInteger();
    if (it == fields.end() || !n || *n < 1 || *n > (1LL << 30)) {
      error = "unknown or invalid positive integer hardware field: " +
              entry.first.str();
      return false;
    }
    *it->second = *n;
  }
  if (h.rows > 8 || h.cols > 8 || h.banks > 8) {
    error = "starter model supports at most an 8x8 grid and 8 banks per CU";
    return false;
  }
  return true;
}
bool plan(const Graph &g, const Hardware &h, const Options &o,
          json::Object &result, std::string &error) {
  if (o.strategy != "heuristic" && o.strategy != "split" &&
      o.strategy != "fuse") {
    error = "strategy must be heuristic, split, or fuse";
    return false;
  }
  if (o.placement != "search" && o.placement != "compact" &&
      o.placement != "spread") {
    error = "placement must be search, compact, or spread";
    return false;
  }
  SmallVector<StringRef> tileStrings;
  StringRef(o.tiles).split(tileStrings, ',');
  std::set<std::array<int64_t, 3>> tiles;
  for (auto spec : tileStrings) {
    SmallVector<StringRef> dims;
    spec.split(dims, 'x');
    std::array<int64_t, 3> t;
    if (dims.size() != 3) {
      error = "tile must be MxNxK";
      return false;
    }
    for (int i = 0; i < 3; ++i)
      if (dims[i].getAsInteger(10, t[i]) || t[i] < 1 || t[i] > 256) {
        error = "tile dimensions must be integers in [1,256]";
        return false;
      }
    // Bound event expansion; this prototype emits explicit events, not loops.
    auto events =
        ceilDiv(g.m, t[0]) * ceilDiv(g.n, t[1]) * (3 * ceilDiv(g.k, t[2]) + 12);
    if (g.p)
      events += ceilDiv(g.m, t[0]) * ceilDiv(g.p, t[1]) *
                (3 * ceilDiv(g.n, t[2]) + 2);
    if (events > 200000) {
      error = "tile expands beyond 200000 events; use larger tiles";
      return false;
    }
    tiles.insert(t);
  }
  if (tiles.empty() || tiles.size() > 16) {
    error = "provide 1 to 16 distinct tiles";
    return false;
  }
  json::Array candidates;
  json::Value selected(nullptr);
  int64_t best = std::numeric_limits<int64_t>::max();
  std::vector<std::string> placements =
      o.placement == "search" ? std::vector<std::string>{"compact", "spread"}
                              : std::vector<std::string>{o.placement};
  for (const auto &tile : tiles)
    for (int mode = 0; mode < 4; ++mode)
      for (const auto &placement : placements) {
        auto c = candidate(g, h, tile, mode, placement);
        bool eligible = o.strategy == "heuristic" ||
                        (o.strategy == "split" && mode == 0) ||
                        (o.strategy == "fuse" && mode == 3);
        if (eligible && *c.getBoolean("legal") &&
            *c.getInteger("estimated_cycles") < best) {
          best = *c.getInteger("estimated_cycles");
          selected = json::Value(json::Object(c));
        }
        if (!o.emitAll) {
          c.erase("events");
          c.erase("buffers");
        }
        candidates.push_back(std::move(c));
      }
  result = json::Object{
      {"schema_version", 1},
      {"graph", json::Object{{"m", g.m}, {"k", g.k}, {"n", g.n}, {"p", g.p}}},
      {"hardware", h.json()},
      {"strategy", o.strategy},
      {"selected_plan", std::move(selected)},
      {"candidates", std::move(candidates)}};
  if (best == std::numeric_limits<int64_t>::max()) {
    error = "no legal candidate for strategy " + o.strategy +
            "; inspect candidate rejection reasons";
    return false;
  }
  return true;
}
} // namespace dataflow
