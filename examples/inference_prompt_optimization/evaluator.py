"""
OpenEvolve evaluator for one-shot Infinigen generator prompt optimization.

The candidate program is a text prompt. A fixed Python prompt loader is passed
to inference so the one-shot pipeline can still import get_generator_prompt().
Evaluation is staged:
1. run inference and score executability,
2. reuse inference output for 3D Chamfer and aligned n-CLIP,
3. optionally run a target-vs-candidate visual verifier.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parent
OPENEVOLVE_DIR = EXAMPLE_DIR.parents[1]
PROJECT_ROOT = EXAMPLE_DIR.parents[2]
SYSTEM_DIR = PROJECT_ROOT / "system"
TASKSOLVER_DIR = PROJECT_ROOT / "TaskSolver"
PROMPT_LOADER_PATH = EXAMPLE_DIR / "templates" / "evolve_inference_prompt.py"

for path in (PROJECT_ROOT, OPENEVOLVE_DIR, SYSTEM_DIR, TASKSOLVER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openevolve.evaluation_result import EvaluationResult
from evaluation3d import (  # noqa: E402
    DEFAULT_GEOMETRY_EXPORT_SCRIPT_PATH,
    DEFAULT_RENDER_WORKER_SCRIPT_PATH,
    evaluate_geometry_instance_3d,
    _completed_proposal_items,
    _infer_dataset_type,
    _load_json,
    _normalize_instance_info_paths,
    _save_json,
)
from tasksolver.answer_types import PythonExecutableAnswer  # noqa: E402
from tasksolver.common import ParsedAnswer, Question, TaskSpec  # noqa: E402
from tasksolver.keychain import KeyChain  # noqa: E402
from tasksolver.utils import docs_for_GPT4  # noqa: E402
from agents import GeneralAgent  # noqa: E402


WEIGHTS = {
    "executable": 0.40,
    "chamfer_score": 0.30,
    "n_clip_score": 0.10,
    "vlm_verifier": 0.20,
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _cache_root() -> Path:
    root = Path(_env("EVOLVE_INFERENCE_CACHE_DIR", str(PROJECT_ROOT / "cache")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_candidate_prompt(program_path: str) -> str:
    path = Path(program_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate prompt text file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        prompt = file.read()
    if not prompt.strip():
        raise ValueError("Candidate prompt is empty.")
    return prompt.strip()


def _candidate_hash(program_path: str) -> str:
    prompt = _load_candidate_prompt(program_path)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _candidate_cache_dir(program_path: str) -> Path:
    cache_dir = _cache_root() / _candidate_hash(program_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _state_path(program_path: str) -> Path:
    return _candidate_cache_dir(program_path) / "state.json"


def _read_state(program_path: str) -> Dict[str, Any]:
    path = _state_path(program_path)
    if path.is_file():
        with open(path, "r") as file:
            return json.load(file)
    return {}


def _write_state(program_path: str, state: Mapping[str, Any]) -> None:
    with open(_state_path(program_path), "w") as file:
        json.dump(state, file, indent=2)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _score_from_loss(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(1.0 / (1.0 + max(0.0, float(value))))


def calculate_prompt_features(prompt: str) -> Tuple[float, float]:
    """Return raw MAP-Elites features for prompt text candidates."""
    prompt_length = float(len(prompt))
    prompt_lower = prompt.lower()
    reasoning_strategy = 0.0

    if len(prompt) >= 500:
        reasoning_strategy += 0.1
    if any(term in prompt_lower for term in ("infer", "analyze", "before writing code", "step")):
        reasoning_strategy += 0.25
    if any(term in prompt_lower for term in ("depth", "normal", "grey", "geometry", "surface orientation")):
        reasoning_strategy += 0.25
    if any(term in prompt_lower for term in ("constraint", "exactly", "must", "do not")):
        reasoning_strategy += 0.15
    if "reference example" in prompt_lower or bool(re.search(r"```python\nclass", prompt)):
        reasoning_strategy += 0.25

    return prompt_length, float(min(1.0, max(0.0, reasoning_strategy)))


def _candidate_features(program_path: str) -> Tuple[float, float]:
    try:
        return calculate_prompt_features(_load_candidate_prompt(program_path))
    except Exception:
        return 0.0, 0.0


def _combined(
    executable: float,
    chamfer_score: float = 0.0,
    n_clip_score: float = 0.0,
    vlm_verifier: float = 0.0,
) -> float:
    return float(
        WEIGHTS["executable"] * executable
        + WEIGHTS["chamfer_score"] * chamfer_score
        + WEIGHTS["n_clip_score"] * n_clip_score
        + WEIGHTS["vlm_verifier"] * vlm_verifier
    )


def _zero_downstream_metrics() -> Dict[str, float]:
    return {
        "chamfer": 0.0,
        "n_clip": 0.0,
        "chamfer_score": 0.0,
        "n_clip_score": 0.0,
        "stage2_passed": 0.0,
        "vlm_verifier": 0.0,
        "stage3_passed": 0.0,
    }


def _latest_metadata_path(info_dir: Path) -> Optional[Path]:
    candidates = sorted(
        info_dir.glob("intermediate_metadata_*_oneshot_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _replace_output_paths(value: Any, source_root: Path, dest_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _replace_output_paths(item, source_root, dest_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_output_paths(item, source_root, dest_root) for item in value]
    if not isinstance(value, str):
        return value

    source_abs = str(source_root.resolve())
    dest_abs = str(dest_root.resolve())
    source_rel = str(source_root)
    source_posix = source_rel.replace(os.sep, "/")
    source_project_rel = str(source_root.relative_to(PROJECT_ROOT))
    source_project_posix = source_project_rel.replace(os.sep, "/")

    if value == source_abs or value.startswith(source_abs + os.sep):
        return dest_abs + value[len(source_abs):]
    if value == source_rel or value.startswith(source_rel + os.sep):
        return dest_abs + value[len(source_rel):]
    if value == source_posix or value.startswith(source_posix + "/"):
        return dest_abs + value[len(source_posix):]
    if value == source_project_rel or value.startswith(source_project_rel + os.sep):
        return dest_abs + value[len(source_project_rel):]
    if value == source_project_posix or value.startswith(source_project_posix + "/"):
        return dest_abs + value[len(source_project_posix):]
    return value


def _archive_inference_outputs(metadata_path: Path, cache_dir: Path) -> Optional[Path]:
    metadata = _load_json(str(metadata_path))
    output_dir_name = metadata.get("output_dir_name")
    if not output_dir_name:
        return None

    source_root = PROJECT_ROOT / "system" / "outputs" / output_dir_name
    if not source_root.is_dir():
        return None

    dest_root = cache_dir / "proposal_outputs" / output_dir_name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_root), str(dest_root))

    rewritten = _replace_output_paths(metadata, source_root, dest_root)
    if isinstance(rewritten, dict):
        rewritten["proposal_output_dir"] = str(dest_root.resolve())
    _save_json(str(metadata_path), rewritten)
    return dest_root


class _Chdir:
    def __init__(self, path: Path):
        self.path = path
        self.previous: Optional[Path] = None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, tb):
        if self.previous is not None:
            os.chdir(self.previous)


def _run_inference(program_path: str) -> Dict[str, Any]:
    state = _read_state(program_path)
    metadata_path = state.get("metadata_path")
    if metadata_path and Path(metadata_path).is_file():
        return state

    cache_dir = _candidate_cache_dir(program_path)
    info_dir = cache_dir / "info_saved"
    info_dir.mkdir(parents=True, exist_ok=True)

    generator_type = _env("EVOLVE_INFERENCE_GENERATOR_TYPE")
    if not generator_type:
        raise ValueError("Set EVOLVE_INFERENCE_GENERATOR_TYPE before running stage1.")

    from argparse import Namespace
    from inference_geometry_oneshot import normalize_generator_type, run_geometry_task

    inference_args = Namespace(
        task=_env("EVOLVE_INFERENCE_TASK", "test"),
        dataset_root=None,
        data_path=str(Path(_env("EVOLVE_INFERENCE_DATA_PATH", str(PROJECT_ROOT / "data" / "blendergym_generation"))).resolve()),
        vlm_code_skeleton_path=str(Path(_env("EVOLVE_INFERENCE_VLM_CODE_SKELETON_PATH", str(PROJECT_ROOT / "vlm_context" / "code_skeleton.py"))).resolve()),
        starter_blend_path=str(Path(_env("EVOLVE_INFERENCE_STARTER_BLEND_PATH", str(PROJECT_ROOT / "system" / "starter_blends" / "face_animation.blend"))).resolve()),
        info_saving_dir_path=str(info_dir),
        blender_render_script_path=str(Path(_env("EVOLVE_INFERENCE_BLENDER_RENDER_SCRIPT_PATH", str(PROJECT_ROOT / "system" / "blender_base" / "infinigen_asset_render.py"))).resolve()),
        infinigen_installation_path=str(Path(_env("EVOLVE_INFERENCE_BLENDER_PATH", str(PROJECT_ROOT / "infinigen" / "blender" / "blender"))).resolve()),
        generator_type=normalize_generator_type(generator_type),
        image_input_mode=_env("EVOLVE_INFERENCE_IMAGE_INPUT_MODE", "rgb_geometry"),
        num_input_views=_env_int("EVOLVE_INFERENCE_NUM_INPUT_VIEWS", 1),
        sample_num=(
            _env_int("EVOLVE_INFERENCE_SAMPLE_NUM", 0)
            if _env("EVOLVE_INFERENCE_SAMPLE_NUM") is not None
            else None
        ),
        tree_dims="1x1",
        render_device=_env("EVOLVE_INFERENCE_RENDER_DEVICE", "auto"),
        prompt_program_path=str(PROMPT_LOADER_PATH.resolve()),
    )

    command = {
        "mode": "import",
        "function": "inference_geometry_oneshot.run_geometry_task",
        "candidate_prompt_text_path": str(Path(program_path).resolve()),
        "args": vars(inference_args),
    }

    started_at = time.time()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    returncode = 0
    error_message = None
    previous_prompt_text_path = os.environ.get("EVOLVE_INFERENCE_PROMPT_TEXT_PATH")
    try:
        os.environ["EVOLVE_INFERENCE_PROMPT_TEXT_PATH"] = str(Path(program_path).resolve())
        if inference_args.render_device == "cpu":
            os.environ["BLENDERGYM_FORCE_CPU"] = "1"
        elif inference_args.render_device == "gpu":
            os.environ.pop("BLENDERGYM_FORCE_CPU", None)

        with _Chdir(PROJECT_ROOT), redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            run_geometry_task(inference_args)
    except Exception as exc:
        returncode = 1
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        if previous_prompt_text_path is None:
            os.environ.pop("EVOLVE_INFERENCE_PROMPT_TEXT_PATH", None)
        else:
            os.environ["EVOLVE_INFERENCE_PROMPT_TEXT_PATH"] = previous_prompt_text_path

    metadata_path_obj = _latest_metadata_path(info_dir)
    proposal_output_dir = None
    if metadata_path_obj is not None:
        proposal_output_dir = _archive_inference_outputs(metadata_path_obj, cache_dir)
    state = {
        **state,
        "command": command,
        "returncode": returncode,
        "stdout_tail": stdout_buffer.getvalue()[-4000:],
        "stderr_tail": stderr_buffer.getvalue()[-4000:],
        "runtime_seconds": time.time() - started_at,
        "metadata_path": str(metadata_path_obj) if metadata_path_obj else None,
        "proposal_output_dir": str(proposal_output_dir) if proposal_output_dir else None,
    }
    _write_state(program_path, state)

    if returncode != 0:
        raise RuntimeError(
            "inference_geometry_oneshot.run_geometry_task failed "
            f"with code {returncode}: {error_message or state['stderr_tail'] or state['stdout_tail']}"
        )
    if metadata_path_obj is None:
        raise RuntimeError(f"Inference completed but no metadata JSON was found in {info_dir}.")

    return state


def _geometry_entries(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return metadata.get("geometry", {}) if isinstance(metadata.get("geometry"), dict) else {}


def _executable_fraction(metadata: Mapping[str, Any]) -> Tuple[float, int, int]:
    entries = _geometry_entries(metadata)
    if not entries:
        return 0.0, 0, 0
    successful = 0
    for instance_info in entries.values():
        proposal_paths = instance_info.get("proposal_edits_paths") or []
        selected_edit_path = instance_info.get("selected_edit_path")
        if proposal_paths or selected_edit_path:
            successful += 1
    return float(successful / len(entries)), successful, len(entries)


def _stage1_metrics(program_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    state = _run_inference(program_path)
    metadata_path = state.get("metadata_path")
    metadata = _load_json(metadata_path)
    executable, successful, total = _executable_fraction(metadata)
    prompt_length, reasoning_strategy = _candidate_features(program_path)
    metrics = {
        "prompt_length": prompt_length,
        "reasoning_strategy": reasoning_strategy,
        "executable": executable,
        "stage1_passed": executable,
        "combined_score": executable,
    }
    metrics.update(_zero_downstream_metrics())
    artifacts = {
        "metadata_path": str(metadata_path),
        "successful_instances": str(successful),
        "total_instances": str(total),
    }
    return metrics, artifacts


def _failure_stage1_metrics(program_path: str) -> Dict[str, float]:
    prompt_length, reasoning_strategy = _candidate_features(program_path)
    metrics = {
        "prompt_length": prompt_length,
        "reasoning_strategy": reasoning_strategy,
        "executable": 0.0,
        "stage1_passed": 0.0,
        "combined_score": 0.0,
    }
    metrics.update(_zero_downstream_metrics())
    return metrics


def evaluate_stage1(program_path: str) -> EvaluationResult:
    try:
        metrics, artifacts = _stage1_metrics(program_path)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        return EvaluationResult(
            metrics=_failure_stage1_metrics(program_path),
            artifacts={"stage1_error": f"{type(exc).__name__}: {exc}"},
        )


def _evaluate_3d(program_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    state = _run_inference(program_path)
    metadata_path = state.get("metadata_path")
    metadata = _load_json(metadata_path)
    executable, successful, total = _executable_fraction(metadata)
    prompt_length, reasoning_strategy = _candidate_features(program_path)

    cache_dir = _candidate_cache_dir(program_path)
    eval_dir = cache_dir / "eval_renders_3d"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "geometry_scores_3d.json"
    if summary_path.is_file():
        geometry_summary = _load_json(str(summary_path))
    else:
        geometry_entries = _geometry_entries(metadata)
        if not geometry_entries:
            raise ValueError(f"No geometry entries found in {metadata_path}.")

        chamfer_values: List[float] = []
        n_clip_values: List[float] = []
        intermediate_scores: Dict[str, Any] = {}
        non_executable_count = 0
        non_executable_details: List[Dict[str, Any]] = []

        for task_instance, instance_info in geometry_entries.items():
            normalized_instance_info = _normalize_instance_info_paths(instance_info)
            normalized_instance_info["dataset_type"] = _infer_dataset_type(
                normalized_instance_info,
                metadata.get("dataset_type"),
            )
            task_instance_dir = eval_dir / task_instance
            task_instance_dir.mkdir(parents=True, exist_ok=True)
            task_instance_scores, instance_non_exec, instance_non_exec_details = evaluate_geometry_instance_3d(
                instance_info=normalized_instance_info,
                task_instance=task_instance,
                task_instance_dir=str(task_instance_dir),
                blender_executable_path=_env("EVOLVE_INFERENCE_BLENDER_PATH", str(PROJECT_ROOT / "infinigen" / "blender" / "blender")),
                render_worker_script_path=_env("EVOLVE_INFERENCE_RENDER_WORKER_SCRIPT_PATH", DEFAULT_RENDER_WORKER_SCRIPT_PATH),
                geometry_export_script_path=_env("EVOLVE_INFERENCE_GEOMETRY_EXPORT_SCRIPT_PATH", DEFAULT_GEOMETRY_EXPORT_SCRIPT_PATH),
                chamfer_num_points=_env_int("EVOLVE_INFERENCE_CHAMFER_NUM_POINTS", 1024),
                chamfer_chunk_size=_env_int("EVOLVE_INFERENCE_CHAMFER_CHUNK_SIZE", 512),
                icp_max_iterations=_env_int("EVOLVE_INFERENCE_ICP_MAX_ITERATIONS", 50),
                icp_tolerance=_env_float("EVOLVE_INFERENCE_ICP_TOLERANCE", 1e-5),
                icp_rejection_percentile=_env_float("EVOLVE_INFERENCE_ICP_REJECTION_PERCENTILE", 95.0),
                go_icp_mse_threshold=None,
                go_icp_trim_fraction=None,
                go_icp_distance_transform_size=_env_int("EVOLVE_INFERENCE_GO_ICP_DISTANCE_TRANSFORM_SIZE", 256),
                go_icp_distance_transform_expand_factor=_env_float("EVOLVE_INFERENCE_GO_ICP_DISTANCE_TRANSFORM_EXPAND_FACTOR", 2.0),
                go_icp_downsample_points=_env_int("EVOLVE_INFERENCE_GO_ICP_DOWNSAMPLE_POINTS", 1024),
                go_icp_timeout_seconds=_env_float("EVOLVE_INFERENCE_GO_ICP_TIMEOUT_SECONDS", 120.0),
                status_log_fn=print,
            )
            task_instance_scores["non_executable_count"] = instance_non_exec
            task_instance_scores["non_executable_details"] = instance_non_exec_details
            _save_json(str(task_instance_dir / "score_3d.json"), task_instance_scores)
            intermediate_scores[task_instance] = task_instance_scores
            non_executable_count += instance_non_exec
            non_executable_details.extend(instance_non_exec_details)

            for _, proposal_scores in _completed_proposal_items(task_instance_scores):
                if proposal_scores.get("chamfer") is not None:
                    chamfer_values.append(float(proposal_scores["chamfer"]))
                if proposal_scores.get("avg_aligned_n_clip") is not None:
                    n_clip_values.append(float(proposal_scores["avg_aligned_n_clip"]))

        geometry_summary = {
            "chamfer": _mean(chamfer_values),
            "n_clip": _mean(n_clip_values),
            "non_executable_count": non_executable_count,
            "non_executable_details": non_executable_details,
        }
        _save_json(str(summary_path), geometry_summary)
        _save_json(str(eval_dir / "intermediate_scores_3d.json"), intermediate_scores)

    chamfer = geometry_summary.get("chamfer")
    n_clip = geometry_summary.get("n_clip")
    chamfer_score = _score_from_loss(chamfer)
    n_clip_score = _score_from_loss(n_clip)
    metrics = {
        "prompt_length": prompt_length,
        "reasoning_strategy": reasoning_strategy,
        "executable": executable,
        "stage1_passed": executable,
        "chamfer": float(chamfer) if chamfer is not None else 0.0,
        "n_clip": float(n_clip) if n_clip is not None else 0.0,
        "chamfer_score": chamfer_score,
        "n_clip_score": n_clip_score,
        "stage2_passed": _combined(executable, chamfer_score, n_clip_score, 0.0),
        "combined_score": _combined(executable, chamfer_score, n_clip_score, 0.0),
    }
    artifacts = {
        "metadata_path": str(metadata_path),
        "eval_dir": str(eval_dir),
        "successful_instances": str(successful),
        "total_instances": str(total),
    }
    return metrics, artifacts


def evaluate_stage2(program_path: str) -> EvaluationResult:
    try:
        metrics, artifacts = _evaluate_3d(program_path)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        try:
            metrics, artifacts = _stage1_metrics(program_path)
        except Exception as stage1_exc:
            metrics = _failure_stage1_metrics(program_path)
            artifacts = {"stage1_error": f"{type(stage1_exc).__name__}: {stage1_exc}"}
        metrics["stage2_passed"] = 0.0
        metrics["combined_score"] = 0.0
        for key, value in _zero_downstream_metrics().items():
            metrics.setdefault(key, value)
        artifacts["stage2_error"] = f"{type(exc).__name__}: {exc}"
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


class VisualSimilarityScore(ParsedAnswer):
    def __init__(self, score: float, raw: str = None):
        self.data = max(0.0, min(1.0, float(score)))
        self.raw = raw

    @staticmethod
    def parser(gpt_raw: str) -> "VisualSimilarityScore":
        """
        @GPT4-doc-begin
        The response must contain exactly one numeric score from 0.0 to 1.0 in a fenced code block.
        Example:
        ```
        0.75
        ```
        @GPT4-doc-end
        """
        import re

        code_blocks = re.findall(r"```\s*([0-9]*\.?[0-9]+)\s*```", gpt_raw)
        candidates = code_blocks or re.findall(r"([0-9]*\.?[0-9]+)", gpt_raw)
        if not candidates:
            raise ValueError("Verifier response did not contain a numeric score.")
        return VisualSimilarityScore(float(candidates[-1]), raw=gpt_raw)

    def __str__(self) -> str:
        return str(self.data)


VERIFIER_TASK = TaskSpec(
    name="Target-vs-candidate 3D asset visual similarity scoring",
    description="Score how visually and geometrically similar a generated asset render is to a target reference render.",
    answer_type=VisualSimilarityScore,
    followup_func=None,
    completed_func=None,
)
VERIFIER_TASK.add_background(
    Question([
        "Return exactly one numeric similarity score from 0.0 to 1.0 in a fenced code block.",
        "0.0 means no meaningful similarity; 1.0 means the candidate closely matches the target geometry and appearance.",
        docs_for_GPT4(VisualSimilarityScore.parser),
    ])
)


def _build_keychain() -> KeyChain:
    keychain = KeyChain()
    credentials = {
        "openai": SYSTEM_DIR / "credentials" / "openai_api.txt",
        "claude": SYSTEM_DIR / "credentials" / "claude_api.txt",
        "gemini": SYSTEM_DIR / "credentials" / "gemini_api.txt",
        "vllm": SYSTEM_DIR / "credentials" / "vllm_api.txt",
    }
    for name, path in credentials.items():
        if path.is_file():
            keychain.add_key(name, str(path))
    return keychain


def _first_existing_target_image(instance_info: Mapping[str, Any]) -> Optional[str]:
    for item in instance_info.get("target_input_images", []) or []:
        if item.get("label") == "render" and item.get("path"):
            path = _resolve_project_path(item["path"])
            if path.is_file():
                return str(path)
    for item in instance_info.get("target_input_images", []) or []:
        if item.get("path"):
            path = _resolve_project_path(item["path"])
            if path.is_file():
                return str(path)
    return None


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verifier_scores(program_path: str) -> Tuple[float, Dict[str, str]]:
    verifier_type = _env("EVOLVE_INFERENCE_VERIFIER_TYPE")
    if not verifier_type:
        return 0.0, {"vlm_verifier_status": "disabled: EVOLVE_INFERENCE_VERIFIER_TYPE is not set"}

    state = _run_inference(program_path)
    metadata = _load_json(state["metadata_path"])
    geometry_entries = _geometry_entries(metadata)
    if not geometry_entries:
        return 0.0, {"vlm_verifier_status": "no geometry entries"}

    agent = GeneralAgent(_build_keychain(), VERIFIER_TASK, vision_model=verifier_type)
    scores: List[float] = []
    raw_outputs: Dict[str, str] = {}
    zero_scored_instances: List[str] = []
    for task_instance, instance_info in geometry_entries.items():
        target_path = _first_existing_target_image(instance_info)
        candidate_value = instance_info.get("selected_render_path")
        candidate_path = _resolve_project_path(candidate_value) if candidate_value else None
        if not target_path or candidate_path is None or not candidate_path.is_file():
            scores.append(0.0)
            zero_scored_instances.append(task_instance)
            continue
        with Image.open(target_path) as target_image, Image.open(candidate_path) as candidate_image:
            question = Question([
                "The first image is the target asset reference. The second image is the generated candidate render.",
                "Score the candidate's visual and geometric similarity to the target from 0.0 to 1.0.",
                "Prioritize shape, proportions, large parts, depth cues, and surface orientation over exact color.",
                "Return only the final score in a fenced code block.",
                "Target image:",
                target_image.convert("RGB").copy(),
                "Candidate image:",
                candidate_image.convert("RGB").copy(),
            ])
        parsed = agent.think(question, num_tokens=512, agent_idx=0)
        scores.append(float(parsed.data))
        raw_outputs[task_instance] = str(parsed.raw)

    return float(_mean(scores) or 0.0), {
        "vlm_verifier_status": "complete" if scores else "no geometry entries",
        "vlm_verifier_raw": json.dumps(raw_outputs, ensure_ascii=False),
        "vlm_verifier_zero_scored_instances": json.dumps(zero_scored_instances, ensure_ascii=False),
    }


def _evaluate_stage3_from_stage2(program_path: str, stage2_result: EvaluationResult) -> EvaluationResult:
    stage2_metrics = dict(stage2_result.metrics)
    stage2_artifacts = dict(stage2_result.artifacts)

    if stage2_metrics.get("stage2_passed", 0.0) <= 0.0:
        stage2_metrics["stage3_passed"] = 0.0
        stage2_metrics["vlm_verifier"] = 0.0
        stage2_metrics["combined_score"] = 0.0
        stage2_artifacts["stage3_skipped"] = "stage2 did not pass"
        return EvaluationResult(metrics=stage2_metrics, artifacts=stage2_artifacts)

    try:
        vlm_score, vlm_artifacts = _verifier_scores(program_path)
    except Exception as exc:
        stage2_metrics["vlm_verifier"] = 0.0
        stage2_metrics["stage3_passed"] = 0.0
        stage2_metrics["combined_score"] = 0.0
        stage2_artifacts["stage3_error"] = f"{type(exc).__name__}: {exc}"
        return EvaluationResult(metrics=stage2_metrics, artifacts=stage2_artifacts)

    executable = stage2_metrics.get("executable", 0.0)
    chamfer_score = stage2_metrics.get("chamfer_score", 0.0)
    n_clip_score = stage2_metrics.get("n_clip_score", 0.0)
    stage2_metrics["vlm_verifier"] = vlm_score
    stage2_metrics["stage3_passed"] = _combined(
        executable,
        chamfer_score,
        n_clip_score,
        vlm_score,
    )
    stage2_metrics["combined_score"] = stage2_metrics["stage3_passed"]
    stage2_artifacts.update(vlm_artifacts)
    return EvaluationResult(metrics=stage2_metrics, artifacts=stage2_artifacts)


def evaluate_stage3(program_path: str) -> EvaluationResult:
    return _evaluate_stage3_from_stage2(program_path, evaluate_stage2(program_path))


def evaluate(program_path: str) -> EvaluationResult:
    stage2_result = evaluate_stage2(program_path)
    return _evaluate_stage3_from_stage2(program_path, stage2_result)
