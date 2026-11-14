import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from cuda_fusion.benchmark import buffers, make_inputs, reference, run_case, verify_candidate, write_report
from cuda_fusion.codegen import emit_cuda, validate_plan
from cuda_fusion.cuda import CUDA, CUDAUnavailable, NVRTC
from cuda_fusion.frontend import compile_graph, make_mlir

COMPILER = os.environ.get("FUSION_OPT")


@unittest.skipUnless(COMPILER, "set FUSION_OPT to run native compiler tests")
class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def compile(self, m=7, n=257, options=""):
        return compile_graph(COMPILER, m, n, self.directory, options=options)

    def reject_source(self, source):
        (self.directory / "bad.mlir").write_text(source, encoding="utf-8")
        result = subprocess.run([COMPILER, "bad.mlir", "--fusion-plan=output=bad.json"],
                                cwd=self.directory, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("error", result.stderr.lower())
        self.assertFalse((self.directory / "bad.json").exists())

    def test_plan_lowers_graph_and_accounts_for_intermediates(self):
        plan = self.compile()
        validate_plan(plan)
        lowered = (self.directory / "planned.mlir").read_text(encoding="utf-8")
        self.assertIn('"fusion.execute"', lowered)
        self.assertNotIn('"fusion.add"', lowered)
        self.assertNotIn('"fusion.rms_norm"', lowered)
        self.assertEqual(len(plan["candidates"]), 12)
        choices = {c["id"]: c for c in plan["candidates"]}
        s = 7 * 257
        self.assertEqual(choices["cached-t64"]["estimated_logical_bytes"], 4 * s * 4)
        self.assertEqual(choices["cached-t64"]["scratch_bytes"], 0)
        self.assertEqual(choices["split-t64"]["scratch_bytes"], 4 * (s + 7))
        self.assertEqual(choices["split-t64"]["kernel_count"], 3)
        self.assertEqual(plan, self.compile(), "planner must be deterministic")

    def test_rejects_invalid_semantics(self):
        source = make_mlir(7, 257)
        bad = [source.replace("tensor<257xf32>", "tensor<256xf32>"),
               source.replace("1.00000000e-05", "0.00000000e+00"),
               source.replace("1.00000000e-05", "-1.00000000e-05"),
               source.replace("f32", "f64"),
               source.replace('"fusion.rms_norm"(%sum', '"fusion.rms_norm"(%x'),
               source.replace("return %out", "return %sum"),
               make_mlir(65536, 1), make_mlir(1, 131073)]
        for text in bad:
            with self.subTest(source=text[:90]):
                self.reject_source(text)

    def test_resource_limits_reject_candidates_without_miscompiling(self):
        plan = self.compile(n=1003, options="scratch-limit=0 register-limit=24")
        legal = [c for c in plan["candidates"] if c["legal"]]
        self.assertTrue(legal)
        self.assertEqual({c["mode"] for c in legal}, {"stream"})
        self.assertTrue(all(c["rejection_reason"] for c in plan["candidates"] if not c["legal"]))
        with self.assertRaisesRegex(RuntimeError, "no legal schedule"):
            self.compile(options="register-limit=1")
        large = self.compile(m=1, n=131072, options="threads=1024")
        cached = next(c for c in large["candidates"] if c["mode"] == "cached")
        self.assertFalse(cached["legal"])
        self.assertIn("per block", cached["rejection_reason"])

    def test_rejects_invalid_schedule_options(self):
        for option in ["threads=48", "threads=0", "threads=2048", "scratch-limit=-1", "register-limit=256"]:
            with self.subTest(option=option), self.assertRaises(RuntimeError):
                self.compile(options=option)

    def test_codegen_checks_plan_boundary(self):
        plan = self.compile(options="scratch-limit=0")
        invalid = next(c for c in plan["candidates"] if not c["legal"])
        with self.assertRaises(ValueError):
            emit_cuda(plan, invalid)
        for mutation in [{"schema_version": 2}, {"rows": 0}, {"epsilon": float("nan")}, {"selected": "missing"}]:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_plan({**plan, **mutation})
        changed = copy.deepcopy(plan)
        changed["candidates"][0]["threads"] = 33
        with self.assertRaises(ValueError):
            validate_plan(changed)


@unittest.skipUnless(COMPILER, "set FUSION_OPT to run CUDA integration tests")
class GPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.nvrtc = NVRTC()
            with CUDA():
                pass
        except CUDAUnavailable as e:
            if os.environ.get("FUSION_REQUIRE_CUDA") == "1":
                raise
            raise unittest.SkipTest(str(e)) from e

    def test_all_schedules_match_reference_on_tails_and_zero_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for m, n in [(1, 1), (7, 257), (3, 1003), (128, 1024)]:
                plan = compile_graph(COMPILER, m, n, directory)
                with CUDA() as cuda:
                    sets = []
                    for seed, zero in [(11, False), (23, False), (0, True)]:
                        inputs = make_inputs(m, n, seed, zero)
                        pointers, output = buffers(cuda, inputs)
                        sets.append((pointers, output, reference(*inputs, plan["epsilon"])))
                    for c in plan["candidates"]:
                        if not c["legal"]:
                            continue
                        with self.subTest(shape=(m, n), candidate=c["id"]):
                            source, launches = emit_cuda(plan, c)
                            cubin, _, _, _ = self.nvrtc.compile(source, cuda.arch)
                            loaded, _ = cuda.load(cubin, launches)
                            for pointers, output, expected in sets:
                                verify_candidate(cuda, loaded, pointers, output, expected)

    def test_budgeted_selector_and_metrics_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = compile_graph(COMPILER, 5, 1003, directory, options="threads=64,128")
            case = run_case(plan, self.nvrtc, directory, seed=19, samples=3, repetitions=4, budget=2)
            by_id = {c["id"]: c for c in case["candidates"]}
            self.assertEqual(len(case["shortlist_ids"]), 2)
            self.assertIn(case["selected_id"], case["shortlist_ids"])
            expected = min(case["shortlist_ids"], key=lambda k: by_id[k]["tuning_median_us"])
            self.assertEqual(case["selected_id"], expected)
            self.assertGreater(case["selected_median_us"], 0)
            self.assertGreaterEqual(case["regret_pct"], -1e-9)
            self.assertAlmostEqual(case["speedup_vs_split"], case["split_median_us"] / case["selected_median_us"])
            for c in case["candidates"]:
                self.assertEqual(len(c["samples_us"]), 3)
                self.assertTrue(c["correct"])
                self.assertTrue(np.isfinite(c["median_us"]))
            report = {"method": "test run", "device": case["device"], "nvrtc_version": "12.9",
                      "compiler_version": "test", "cases": [case]}
            write_report(report, directory)
            saved = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(saved["cases"][0]["selected_id"], case["selected_id"])
            self.assertTrue((Path(directory) / "candidates.csv").is_file())
            self.assertIn("not DRAM counters", (Path(directory) / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
