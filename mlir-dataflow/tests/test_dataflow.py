import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from dataflow.experiments import compare, compile_graph
from dataflow.generate import generate
from dataflow.simulator import schedule, simulate

ROOT = Path(__file__).resolve().parents[1]
COMPILER = os.environ.get("DATAFLOW_OPT")


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.artifact = json.loads((ROOT / "examples/edge.split.plan.json").read_text())

    def test_checked_in_edge_plan_executes(self):
        result = simulate(self.artifact, seed=23)
        self.assertTrue(result["verified"])
        # Independent traffic equation for materializing two graph boundaries.
        m, k, n = 17, 19, 13
        expected = 4 * (m*k*2 + k*n*3 + n*3 + 5*m*n)
        self.assertEqual(result["offchip_bytes"], expected)

    def test_link_contention_serializes_only_shared_links(self):
        def event(i, link):
            return {"id": i, "kind": "send", "cycles": 10, "deps": [], "resources": [link]}
        shared = schedule([event(0, "link:0>1"), event(1, "link:0>1")])
        separate = schedule([event(0, "link:0>1"), event(1, "link:2>3")])
        self.assertEqual(shared["simulated_cycles"], 20)
        self.assertEqual(shared["resource_wait_cycles"], {"link:0>1": 10})
        self.assertEqual(separate["simulated_cycles"], 10)

    def test_cyclic_plan_rejected(self):
        events = [{"id": i, "kind": "send", "cycles": 1, "deps": [1-i], "resources": []} for i in range(2)]
        with self.assertRaisesRegex(ValueError, "cycle"):
            schedule(events)

    def test_buffer_overflow_rejected(self):
        self.artifact["selected_plan"]["buffers"][0]["offset"] = 16384
        with self.assertRaisesRegex(ValueError, "capacity"):
            simulate(self.artifact)

    def test_live_alias_rejected(self):
        plan = self.artifact["selected_plan"]
        a = next(b for b in plan["buffers"] if b["id"].endswith("s0.a"))
        accumulator = next(b for b in plan["buffers"] if b["id"].endswith("s0.acc"))
        a["offset"] = accumulator["offset"]
        with self.assertRaisesRegex(ValueError, "live SRAM alias"):
            simulate(self.artifact)

    def test_transfer_corruption_changes_the_result(self):
        plan = self.artifact["selected_plan"]
        # Drop a contribution without changing allocation or output coverage.
        mm = next(e for e in plan["events"] if e["kind"] == "matmul")
        mm["kind"] = "zero"
        mm["shape"] = mm["shape"][:2]
        mm["reads"] = []
        with self.assertRaises(AssertionError):
            simulate(self.artifact)

    def test_no_verify_is_reported(self):
        self.assertFalse(simulate(self.artifact, verify=False)["verified"])


@unittest.skipUnless(COMPILER, "set DATAFLOW_OPT to run native compiler integration tests")
class CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hw = json.loads((ROOT / "configs/default.json").read_text())

    def test_all_candidates_and_second_projection(self):
        for shape in ((17, 19, 13), (17, 19, 13, 11), (1, 3, 2), (8, 16, 8)):
            artifact = compile_graph(COMPILER, shape, self.hw)
            modes = set()
            for plan in artifact["candidates"]:
                if plan["legal"]:
                    with self.subTest(shape=shape, plan=plan["id"]):
                        self.assertTrue(simulate(artifact, plan, seed=11)["verified"])
                        modes.add(plan["mode"])
            self.assertEqual(modes, {"split", "matmul-bias", "bias-gelu", "fused"})

    def test_fusion_removes_exactly_two_intermediates(self):
        artifact = compile_graph(COMPILER, (17,19,13), self.hw, "8x8x8", placement="spread")
        split = next(p for p in artifact["candidates"] if p["mode"] == "split")
        fused = next(p for p in artifact["candidates"] if p["mode"] == "fused")
        self.assertEqual(split["offchip_bytes"] - fused["offchip_bytes"], 16 * 17 * 13)
        self.assertTrue(any(e["kind"] == "send" for e in fused["events"]))
        # Lifetime allocation really reuses addresses across nonoverlapping values.
        compact = compile_graph(COMPILER, (17,19,13), self.hw, "8x8x8", "fuse", "compact")["selected_plan"]
        ranges = [(b["region"],b["slot"],b["cu"],b["bank"],b["offset"]) for b in compact["buffers"]]
        self.assertLess(len(set(ranges)), len(ranges))

    def test_budget_switches_same_graph_to_split(self):
        generous = compile_graph(COMPILER, (17,19,13), self.hw)
        tight = compile_graph(COMPILER, (17,19,13), {**self.hw, "bank_bytes":512})
        self.assertEqual(generous["selected_plan"]["mode"], "fused")
        self.assertEqual(tight["selected_plan"]["mode"], "split")
        self.assertTrue(all(not c["legal"] for c in tight["candidates"] if c["mode"] == "fused"))
        self.assertIn("capacity 512", next(c["rejection"] for c in tight["candidates"] if not c["legal"]))

    def test_exhaustive_is_a_distinct_simulated_oracle(self):
        artifact = compile_graph(COMPILER, (33,31,29), self.hw)
        rows = {r["policy"]: r for r in compare(artifact)}
        self.assertGreater(rows["heuristic"]["simulated_cycles"], rows["exhaustive"]["simulated_cycles"])
        self.assertGreater(rows["heuristic"]["regret_percent"], 0)

    def test_no_legal_plan_has_diagnostics(self):
        artifact = compile_graph(COMPILER, (17,19,13), {**self.hw, "bank_bytes":16})
        self.assertIsNone(artifact["selected_plan"])
        self.assertTrue(all(not c["legal"] and c["rejection"] for c in artifact["candidates"]))

    def test_one_bank_and_single_cu(self):
        hw = {**self.hw, "rows":1, "cols":1, "banks_per_cu":1}
        artifact = compile_graph(COMPILER, (9,7,11,5), hw, "8x8x8")
        for plan in artifact["candidates"]:
            self.assertTrue(simulate(artifact, plan)["verified"])

    def test_rejects_semantically_different_mlir(self):
        source = generate()
        invalid = [source.replace("arith.addf %x, %b", "arith.mulf %x, %b"),
                   source.replace("0.044715", "0.05"),
                   source.replace("%zero = arith.constant 0.0", "%zero = arith.constant 1.0"),
                   source.replace("return %gelu", "return %mm")]
        for text in invalid:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "input.mlir").write_text(text, encoding="utf-8")
                process = subprocess.run([str(Path(COMPILER).resolve()), "input.mlir", "--dataflow-plan", "-o", "out.mlir"],
                                         cwd=root, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("supported graph", process.stderr)
                self.assertFalse((root / "plan.json").exists())

    def test_rejects_invalid_hardware_and_tiles(self):
        with self.assertRaisesRegex(RuntimeError, "hardware field"):
            compile_graph(COMPILER, (17,19,13), {**self.hw, "dram_bytes_per_cycle":0})
        with self.assertRaisesRegex(RuntimeError, "tile dimensions"):
            compile_graph(COMPILER, (17,19,13), self.hw, "8x0x8")


if __name__ == "__main__":
    unittest.main()
