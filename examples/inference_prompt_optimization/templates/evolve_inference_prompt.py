"""
Fixed prompt loader for one-shot Infinigen generator prompt optimization.

OpenEvolve optimizes the sibling ``evolve_prompt.txt`` text file. This Python
module stays stable so ``inference_geometry_oneshot.py --prompt_program_path``
can import ``get_generator_prompt()``.
"""

from __future__ import annotations

import os
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_TEXT_PATH = EXAMPLE_DIR / "evolve_prompt.txt"
REQUIRED_PLACEHOLDERS = (
    "{num_views}",
    "{code_skeleton}",
    "{helper_manual}",
    "{reference_example}",
)


def _repo_root() -> Path:
    env_root = os.environ.get("THREED_COT_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "vlm_context").is_dir():
            return parent
    return current.parent


def _load_repo_text(relative_path: str) -> str:
    with open(_repo_root() / relative_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _prompt_text_path() -> Path:
    prompt_path = os.environ.get("EVOLVE_INFERENCE_PROMPT_TEXT_PATH")
    if prompt_path:
        return Path(prompt_path).expanduser().resolve()
    return DEFAULT_PROMPT_TEXT_PATH


def _load_prompt_template() -> str:
    prompt_path = _prompt_text_path()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt text file does not exist: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _validate_prompt_template(prompt_template: str) -> None:
    missing = [
        placeholder for placeholder in REQUIRED_PLACEHOLDERS if placeholder not in prompt_template
    ]
    if missing:
        raise ValueError(f"Prompt template is missing required placeholders: {', '.join(missing)}")


def _num_views(num_views: int | str | None = None) -> str:
    if num_views is not None:
        return str(num_views)
    return os.environ.get("EVOLVE_INFERENCE_NUM_INPUT_VIEWS", "1")


def get_generator_prompt(num_views: int | str | None = None) -> str:
    """Return the complete generator instruction text."""
    prompt_template = _load_prompt_template()
    _validate_prompt_template(prompt_template)
    replacements = {
        "{num_views}": _num_views(num_views),
        "{code_skeleton}": _load_repo_text("vlm_context/code_skeleton.py"),
        "{helper_manual}": _load_repo_text("vlm_context/helper_manual.md"),
        "{reference_example}": _load_repo_text("vlm_context/examples/mushroom.py"),
    }
    prompt = prompt_template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt.strip()


def validate_generator_prompt() -> None:
    """Validate that the prompt loader exposes a usable generator prompt."""
    prompt = get_generator_prompt()
    if not prompt:
        raise ValueError("Generator prompt is empty.")
    if "```python" not in prompt:
        raise ValueError("Generator prompt must require a python code block response.")

    unfilled = [
        placeholder
        for placeholder in REQUIRED_PLACEHOLDERS
        if placeholder in prompt
    ]
    if unfilled:
        raise ValueError(f"Generator prompt contains unfilled placeholders: {', '.join(unfilled)}")


if __name__ == "__main__":
    validate_generator_prompt()
    print(get_generator_prompt())
