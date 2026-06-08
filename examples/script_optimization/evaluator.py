"""
OpenEvolve evaluator for per-sample generated script optimization.

The candidate program is a Blender/Infinigen Python script for one fixed
geometry sample. The generator prompt is not evaluated or mutated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parent
OPENEVOLVE_DIR = EXAMPLE_DIR.parents[1]
PROJECT_ROOT = EXAMPLE_DIR.parents[2]
SYSTEM_DIR = PROJECT_ROOT / "system"
TASKSOLVER_DIR = PROJECT_ROOT / "TaskSolver"

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
    _derive_chamfer,
    _load_json,
    _normalize_instance_info_paths,
    _save_json,
)
from tasksolver.common import ParsedAnswer, Question, TaskSpec  # noqa: E402
from tasksolver.keychain import KeyChain  # noqa: E402
from tasksolver.utils import docs_for_GPT4  # noqa: E402
from agents import GeneralAgent  # noqa: E402


WEIGHTS = {
    "executable": 0.50,
    "chamfer_score": 0.30,
    "n_clip_score": 0.10,
    "vlm_verifier": 0.10,
}

GEOMETRY_CHAMFER_KEYS = (
    "pre_icp_chamfer",
    "aligned_go_icp_chamfer",
    "aligned_icp_chamfer",
)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _sample_id() -> str:
    sample_id = _env("EVOLVE_SCRIPT_SAMPLE_ID")
    if not sample_id:
        raise ValueError("Set EVOLVE_SCRIPT_SAMPLE_ID before evaluating script candidates.")
    return sample_id


def _sample_tag() -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", _sample_id()).strip("_") or "sample"


def _cache_root() -> Path:
    default_root = PROJECT_ROOT / "cache" / "script_optimization" / _sample_tag()
    root = Path(_env("EVOLVE_SCRIPT_SAMPLE_CACHE_DIR", str(default_root))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _candidate_hash(program_path: str) -> str:
    with open(program_path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()[:16]


def _candidate_cache_dir(program_path: str) -> Path:
    cache_dir = _cache_root() / _candidate_hash(program_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_sample_context() -> Dict[str, Any]:
    context_path = _env("EVOLVE_SCRIPT_SAMPLE_CONTEXT_PATH")
    if not context_path:
        raise ValueError("Set EVOLVE_SCRIPT_SAMPLE_CONTEXT_PATH before evaluating script candidates.")
    return _load_json(str(Path(context_path).expanduser().resolve()))


def _score_from_loss(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(1.0 / (1.0 + max(0.0, float(value))))


def _candidate_features(program_path: str) -> Tuple[float, float]:
    with open(program_path, "r", encoding="utf-8", errors="replace") as file:
        code = file.read()
    code_length = float(len(code))
    uses_factory = 1.0 if "class MyAssetFactory" in code and "spawn_asset" in code else 0.0
    return code_length, uses_factory


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


def _proposal_summary(task_instance_scores: Mapping[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    completed = _completed_proposal_items(task_instance_scores)
    if completed:
        return completed[0]
    for proposal_name, proposal_scores in task_instance_scores.items():
        if isinstance(proposal_scores, dict) and proposal_scores.get("score_status") == "failed":
            return proposal_name, proposal_scores
    return None, {}


def _read_excerpt(path_value: Optional[str], max_chars: int = 6000) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.is_file():
        return f"<missing file: {path_value}>"
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        content = file.read(max_chars + 1)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"
    return content


def _extract_bootstrap_error(context: Mapping[str, Any]) -> str:
    candidates: List[str] = []
    env_error = _env("EVOLVE_SCRIPT_BOOTSTRAP_ERROR")
    if env_error:
        candidates.append(env_error)
    if context.get("error"):
        candidates.append(str(context["error"]))

    for path_value in context.get("failed_response_paths") or []:
        path = Path(str(path_value)).expanduser()
        if not path.is_file():
            continue
        try:
            payload = _load_json(str(path))
        except Exception as exc:
            candidates.append(f"Failed to load failed response {path}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(payload, Mapping):
            for key in ("error", "raw_response", "reasoning_process", "explicit_reasoning_output"):
                value = payload.get(key)
                if value:
                    candidates.append(f"{key}: {value}")
                    break

    return "\n".join(candidates[:3])


def _artifact_feedback(
    *,
    program_path: str,
    context: Mapping[str, Any],
    task_instance_scores: Mapping[str, Any],
    proposal_name: Optional[str],
    proposal_scores: Mapping[str, Any],
    executable: float,
    chamfer: Optional[float],
    n_clip: Optional[float],
    verifier_suggestions: str = "",
) -> str:
    initial_script_path = _env("EVOLVE_SCRIPT_INITIAL_SCRIPT_PATH")
    initial_source_kind = _env("EVOLVE_SCRIPT_INITIAL_SOURCE_KIND", "unknown")
    non_exec_details = task_instance_scores.get("non_executable_details") or []
    lines = [
        f"sample_id: {_sample_id()}",
        f"candidate_script_path: {Path(program_path).resolve()}",
        f"bootstrap_script_path: {initial_script_path or ''}",
        f"bootstrap_source_kind: {initial_source_kind}",
        f"executable: {executable:.4f}",
        f"chamfer: {chamfer if chamfer is not None else 'none'}",
        f"avg_aligned_n_clip: {n_clip if n_clip is not None else 'none'}",
        f"proposal_name: {proposal_name or 'none'}",
    ]

    for key in GEOMETRY_CHAMFER_KEYS:
        if proposal_scores.get(key) is not None:
            lines.append(f"{key}: {proposal_scores[key]}")
    if proposal_scores.get("used_code_skeleton_fallback") is not None:
        lines.append(f"used_code_skeleton_fallback: {proposal_scores['used_code_skeleton_fallback']}")
    if proposal_scores.get("error"):
        lines.append(f"candidate_error: {proposal_scores['error']}")
    if non_exec_details:
        lines.append("non_executable_details:")
        for detail in non_exec_details[:5]:
            lines.append(json.dumps(detail, ensure_ascii=False))
    bootstrap_error = _extract_bootstrap_error(context)
    if bootstrap_error:
        lines.append("bootstrap_error:")
        lines.append(bootstrap_error[:4000])
    if verifier_suggestions:
        lines.append("vision_verifier_suggestions:")
        lines.append(verifier_suggestions[:4000])

    lines.append("\n# Current candidate script excerpt")
    lines.append(_read_excerpt(program_path))
    if initial_script_path and str(Path(initial_script_path).resolve()) != str(Path(program_path).resolve()):
        lines.append("\n# Bootstrap script / failed_script excerpt")
        lines.append(_read_excerpt(initial_script_path))
    return "\n".join(lines)


class VisualScriptFeedback(ParsedAnswer):
    def __init__(self, score: float, suggestions: str = "", raw: str = ""):
        self.data = max(0.0, min(1.0, float(score)))
        self.suggestions = suggestions.strip()
        self.raw = raw

    @staticmethod
    def parser(gpt_raw: str) -> "VisualScriptFeedback":
        """
        @GPT4-doc-begin
        Return JSON in a fenced code block with:
        {"score": 0.75, "suggestions": "Concise script-level edits."}
        @GPT4-doc-end
        """
        blocks = re.findall(r"```\s*(?:json)?\s*(.*?)```", gpt_raw, re.DOTALL)
        for block in blocks:
            try:
                payload = json.loads(block)
            except Exception:
                continue
            if isinstance(payload, Mapping) and payload.get("score") is not None:
                return VisualScriptFeedback(
                    float(payload["score"]),
                    str(payload.get("suggestions") or ""),
                    raw=gpt_raw,
                )
        numbers = re.findall(r"([0-9]*\.?[0-9]+)", gpt_raw)
        score = float(numbers[-1]) if numbers else 0.0
        return VisualScriptFeedback(score, suggestions=gpt_raw, raw=gpt_raw)

    def __str__(self) -> str:
        return f"{self.data}: {self.suggestions}"


VERIFIER_TASK = TaskSpec(
    name="Generated 3D script visual feedback",
    description="Score target-vs-candidate render similarity and suggest concrete Python script edits.",
    answer_type=VisualScriptFeedback,
    followup_func=None,
    completed_func=None,
)
VERIFIER_TASK.add_background(
    Question([
        "Return JSON with a numeric score from 0.0 to 1.0 and concise script-level suggestions.",
        "Focus suggestions on geometry, proportions, parts, spatial layout, and surface orientation.",
        docs_for_GPT4(VisualScriptFeedback.parser),
    ])
)


def _build_keychain() -> KeyChain:
    keychain = KeyChain()
    credentials = {
        "openai": SYSTEM_DIR / "credentials" / "openai_api.txt",
        "claude": SYSTEM_DIR / "credentials" / "claude_api.txt",
        "gemini": SYSTEM_DIR / "credentials" / "gemini_api.txt",
        "vllm": SYSTEM_DIR / "credentials" / "vllm_api.txt",
        "moonshot": SYSTEM_DIR / "credentials" / "moonshot_api.txt",
    }
    for name, path in credentials.items():
        if path.is_file():
            keychain.add_key(name, str(path))
    return keychain


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first_existing_target_image(instance_info: Mapping[str, Any]) -> Optional[str]:
    for item in instance_info.get("target_input_images", []) or []:
        if isinstance(item, Mapping) and item.get("label") == "render" and item.get("path"):
            path = _resolve_project_path(item["path"])
            if path.is_file():
                return str(path)
    for item in instance_info.get("target_input_images", []) or []:
        if isinstance(item, Mapping) and item.get("path"):
            path = _resolve_project_path(item["path"])
            if path.is_file():
                return str(path)
    return None


def _first_existing_candidate_image(proposal_scores: Mapping[str, Any]) -> Optional[str]:
    aligned_render_dir = proposal_scores.get("aligned_render_dir")
    if not aligned_render_dir:
        return None
    render_dir = _resolve_project_path(str(aligned_render_dir))
    if render_dir.is_file():
        return str(render_dir)
    if render_dir.is_dir():
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            matches = sorted(render_dir.glob(pattern))
            if matches:
                return str(matches[0])
    return None


def _verifier_feedback(
    instance_info: Mapping[str, Any],
    proposal_scores: Mapping[str, Any],
    program_path: str,
) -> Tuple[float, str, Dict[str, str]]:
    verifier_type = _env("EVOLVE_INFERENCE_VERIFIER_TYPE")
    if not verifier_type:
        return 0.0, "", {"vlm_verifier_status": "disabled: EVOLVE_INFERENCE_VERIFIER_TYPE is not set"}

    target_path = _first_existing_target_image(instance_info)
    candidate_path = _first_existing_candidate_image(proposal_scores)
    if not target_path or not candidate_path:
        return 0.0, "No target or candidate render was available for visual verification.", {
            "vlm_verifier_status": "missing render",
        }

    script_excerpt = _read_excerpt(program_path, max_chars=5000)
    agent = GeneralAgent(_build_keychain(), VERIFIER_TASK, vision_model=verifier_type)
    with Image.open(target_path) as target_image, Image.open(candidate_path) as candidate_image:
        question = Question([
            "The first image is the target asset reference. The second image is the generated candidate render.",
            "Score visual/geometric similarity from 0.0 to 1.0.",
            "Then suggest concrete edits to the Python script that could reduce geometry mismatch.",
            "Prioritize shape, proportions, large parts, depth cues, and surface orientation over exact color.",
            "Return only fenced JSON with keys score and suggestions.",
            "Candidate script excerpt:",
            script_excerpt,
            "Target image:",
            target_image.convert("RGB").copy(),
            "Candidate image:",
            candidate_image.convert("RGB").copy(),
        ])
    parsed = agent.think(question, num_tokens=768, agent_idx=0)
    return float(parsed.data), parsed.suggestions, {
        "vlm_verifier_status": "complete",
        "vlm_verifier_raw": str(parsed.raw),
        "vlm_verifier_suggestions": parsed.suggestions,
    }


def _evaluate_script(program_path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    context = _load_sample_context()
    instance_info = _normalize_instance_info_paths(context)
    if _env("EVOLVE_SCRIPT_DATASET_TYPE"):
        instance_info["dataset_type"] = _env("EVOLVE_SCRIPT_DATASET_TYPE")
    instance_info["proposal_edits_paths"] = [str(Path(program_path).expanduser().resolve())]
    instance_info["selected_edit_path"] = str(Path(program_path).expanduser().resolve())

    cache_dir = _candidate_cache_dir(program_path)
    score_path = cache_dir / "score_3d.json"
    if score_path.is_file():
        task_instance_scores = _load_json(str(score_path))
    else:
        task_instance_scores, instance_non_exec, instance_non_exec_details = evaluate_geometry_instance_3d(
            instance_info=instance_info,
            task_instance=_sample_id(),
            task_instance_dir=str(cache_dir),
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
        _save_json(str(score_path), task_instance_scores)

    proposal_name, proposal_scores = _proposal_summary(task_instance_scores)
    completed = bool(proposal_scores.get("score_status") == "complete")
    used_fallback = bool(proposal_scores.get("used_code_skeleton_fallback"))
    executable = 1.0 if completed and not used_fallback else 0.0
    chamfer = _derive_chamfer(proposal_scores) if completed else None
    n_clip = (
        float(proposal_scores["avg_aligned_n_clip"])
        if proposal_scores.get("avg_aligned_n_clip") is not None
        else None
    )
    chamfer_score = _score_from_loss(chamfer)
    n_clip_score = _score_from_loss(n_clip)

    verifier_score = 0.0
    verifier_suggestions = ""
    verifier_artifacts: Dict[str, str] = {}
    if completed:
        try:
            verifier_score, verifier_suggestions, verifier_artifacts = _verifier_feedback(
                instance_info,
                proposal_scores,
                program_path,
            )
        except Exception as exc:
            verifier_artifacts = {"vlm_verifier_error": f"{type(exc).__name__}: {exc}"}

    code_length, uses_factory = _candidate_features(program_path)
    metrics = {
        "combined_score": _combined(executable, chamfer_score, n_clip_score, verifier_score),
        "executable": executable,
        "chamfer": float(chamfer) if chamfer is not None else 0.0,
        "n_clip": float(n_clip) if n_clip is not None else 0.0,
        "chamfer_score": chamfer_score,
        "n_clip_score": n_clip_score,
        "vlm_verifier": verifier_score,
        "code_length": code_length,
        "uses_factory": uses_factory,
    }
    artifacts = {
        "sample_feedback": _artifact_feedback(
            program_path=program_path,
            context=context,
            task_instance_scores=task_instance_scores,
            proposal_name=proposal_name,
            proposal_scores=proposal_scores,
            executable=executable,
            chamfer=chamfer,
            n_clip=n_clip,
            verifier_suggestions=verifier_suggestions,
        ),
        "score_path": str(score_path),
    }
    artifacts.update(verifier_artifacts)
    return metrics, artifacts


def evaluate(program_path: str) -> EvaluationResult:
    try:
        metrics, artifacts = _evaluate_script(program_path)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        code_length, uses_factory = _candidate_features(program_path)
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "executable": 0.0,
                "chamfer": 0.0,
                "n_clip": 0.0,
                "chamfer_score": 0.0,
                "n_clip_score": 0.0,
                "vlm_verifier": 0.0,
                "code_length": code_length,
                "uses_factory": uses_factory,
            },
            artifacts={
                "sample_feedback": (
                    f"sample_id: {_env('EVOLVE_SCRIPT_SAMPLE_ID', 'unknown')}\n"
                    f"candidate_script_path: {Path(program_path).resolve()}\n"
                    f"evaluation_error: {type(exc).__name__}: {exc}\n\n"
                    "# Current candidate script excerpt\n"
                    f"{_read_excerpt(program_path)}"
                )
            },
        )
