import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestInferencePromptOptimizationEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = cls._load_module()

    @staticmethod
    def _load_module():
        class EvaluationResult:
            def __init__(self, metrics=None, artifacts=None):
                self.metrics = metrics or {}
                self.artifacts = artifacts or {}

        openevolve_pkg = types.ModuleType("openevolve")
        evaluation_result_mod = types.ModuleType("openevolve.evaluation_result")
        evaluation_result_mod.EvaluationResult = EvaluationResult

        evaluation3d_mod = types.ModuleType("evaluation3d")
        evaluation3d_mod.DEFAULT_GEOMETRY_EXPORT_SCRIPT_PATH = "geometry_export.py"
        evaluation3d_mod.DEFAULT_RENDER_WORKER_SCRIPT_PATH = "render_worker.py"
        evaluation3d_mod.evaluate_geometry_instance_3d = lambda **kwargs: ({}, 0, [])
        evaluation3d_mod._completed_proposal_items = lambda _: []
        evaluation3d_mod._infer_dataset_type = lambda *args, **kwargs: "test"
        evaluation3d_mod._load_json = lambda path: {}
        evaluation3d_mod._normalize_instance_info_paths = lambda info: info
        evaluation3d_mod._save_json = lambda path, payload: None

        answer_types_mod = types.ModuleType("tasksolver.answer_types")
        answer_types_mod.PythonExecutableAnswer = object

        common_mod = types.ModuleType("tasksolver.common")

        class ParsedAnswer:
            pass

        class Question(list):
            pass

        class TaskSpec:
            def __init__(self, *args, **kwargs):
                pass

            def add_background(self, question):
                return None

        common_mod.ParsedAnswer = ParsedAnswer
        common_mod.Question = Question
        common_mod.TaskSpec = TaskSpec

        keychain_mod = types.ModuleType("tasksolver.keychain")

        class KeyChain:
            def add_key(self, *args, **kwargs):
                return None

        keychain_mod.KeyChain = KeyChain

        utils_mod = types.ModuleType("tasksolver.utils")
        utils_mod.docs_for_GPT4 = lambda parser: "docs"

        agents_mod = types.ModuleType("agents")

        class GeneralAgent:
            def __init__(self, *args, **kwargs):
                pass

            def think(self, *args, **kwargs):
                return types.SimpleNamespace(data=0.0, raw="0.0")

        agents_mod.GeneralAgent = GeneralAgent

        stub_modules = {
            "openevolve": openevolve_pkg,
            "openevolve.evaluation_result": evaluation_result_mod,
            "evaluation3d": evaluation3d_mod,
            "tasksolver.answer_types": answer_types_mod,
            "tasksolver.common": common_mod,
            "tasksolver.keychain": keychain_mod,
            "tasksolver.utils": utils_mod,
            "agents": agents_mod,
        }

        for name, module in stub_modules.items():
            sys.modules.setdefault(name, module)

        module_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "inference_prompt_optimization"
            / "evaluator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "inference_prompt_optimization_evaluator_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_stage1_failure_keeps_prompt_length_metric(self):
        with patch.object(self.module, "_stage1_metrics", side_effect=RuntimeError("boom")):
            with patch.object(self.module, "_load_candidate_prompt", return_value="abcd"):
                result = self.module.evaluate_stage1("candidate.py")

        self.assertEqual(result.metrics["prompt_length"], 4.0)
        self.assertEqual(result.metrics["stage1_passed"], 0.0)
        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertIn("stage1_error", result.artifacts)

    def test_stage2_failed_stage1_becomes_skip_with_zeroed_feature_metrics(self):
        with patch.object(self.module, "_stage1_metrics", side_effect=RuntimeError("stage1 failed")):
            with patch.object(self.module, "_load_candidate_prompt", return_value="abcdef"):
                result = self.module.evaluate_stage2("candidate.py")

        self.assertEqual(result.metrics["prompt_length"], 6.0)
        self.assertEqual(result.metrics["stage1_passed"], 0.0)
        self.assertEqual(result.metrics["stage2_passed"], 0.0)
        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertIn("stage1_error", result.artifacts)
        self.assertEqual(result.artifacts["stage2_skipped"], "stage1 returned non-executable result")

    def test_stage2_skips_when_stage1_is_non_executable(self):
        stage1_result = self.module.EvaluationResult(
            metrics={
                "prompt_length": 7.0,
                "executable": 0.0,
                "stage1_passed": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"metadata_path": "dummy.json"},
        )

        with patch.object(self.module, "evaluate_stage1", return_value=stage1_result):
            with patch.object(self.module, "_evaluate_3d") as mock_evaluate_3d:
                result = self.module.evaluate_stage2("candidate.py")

        mock_evaluate_3d.assert_not_called()
        self.assertEqual(result.metrics["stage2_passed"], 0.0)
        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertEqual(result.metrics["prompt_length"], 7.0)
        self.assertEqual(result.artifacts["stage2_skipped"], "stage1 returned non-executable result")

    def test_stage2_runtime_error_after_stage1_success_sets_stage2_error(self):
        stage1_result = self.module.EvaluationResult(
            metrics={
                "prompt_length": 5.0,
                "executable": 1.0,
                "stage1_passed": 1.0,
                "combined_score": 1.0,
            },
            artifacts={"metadata_path": "dummy.json"},
        )

        with patch.object(self.module, "evaluate_stage1", return_value=stage1_result):
            with patch.object(self.module, "_evaluate_3d", side_effect=RuntimeError("stage2 failed")):
                with patch.object(
                    self.module,
                    "_stage1_metrics",
                    return_value=(dict(stage1_result.metrics), dict(stage1_result.artifacts)),
                ):
                    result = self.module.evaluate_stage2("candidate.py")

        self.assertEqual(result.metrics["stage2_passed"], 0.0)
        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertIn("stage2_error", result.artifacts)

    def test_stage3_skips_when_stage2_does_not_pass(self):
        stage2_result = self.module.EvaluationResult(
            metrics={
                "prompt_length": 8.0,
                "executable": 0.0,
                "stage1_passed": 0.0,
                "stage2_passed": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"stage2_skipped": "stage1 returned non-executable result"},
        )

        with patch.object(self.module, "evaluate_stage2", return_value=stage2_result):
            with patch.object(self.module, "_verifier_scores") as mock_verifier:
                result = self.module.evaluate_stage3("candidate.py")

        mock_verifier.assert_not_called()
        self.assertEqual(result.metrics["stage3_passed"], 0.0)
        self.assertEqual(result.metrics["vlm_verifier"], 0.0)
        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertEqual(result.artifacts["stage3_skipped"], "stage2 did not pass")
