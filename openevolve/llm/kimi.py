"""
Moonshot / Kimi API interface for OpenEvolve LLMs.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

KIMI_BASE_URL = "https://api.kimi.com/coding"
KIMI_DEFAULT_MODEL = "k2p6"
KIMI_DEFAULT_HEADERS = {"User-Agent": "KimiCLI/1.12.0"}


def resolve_moonshot_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    if explicit_api_key:
        return explicit_api_key
    return os.environ.get("MOONSHOT_API_KEY")


def resolve_moonshot_api_base(explicit_api_base: Optional[str] = None) -> str:
    if explicit_api_base and "kimi.com" in explicit_api_base:
        return explicit_api_base

    for env_name in ("MOONSHOT_API_BASE", "KIMI_API_BASE"):
        value = os.environ.get(env_name)
        if value:
            return value

    return KIMI_BASE_URL


class KimiLLM(LLMInterface):
    """LLM interface using Moonshot's Anthropic-compatible Messages API."""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for kimi2-6 in OpenEvolve. "
                "Install it in the active environment."
            ) from exc

        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = resolve_moonshot_api_key(model_cfg.api_key)
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.api_base = resolve_moonshot_api_base(getattr(model_cfg, "api_base", None))
        self.kimi_model = KIMI_DEFAULT_MODEL

        if not self.api_key:
            raise ValueError(
                "kimi2-6 requires MOONSHOT_API_KEY or llm.api_key. "
                "OpenEvolve cannot call this backend without a Moonshot API key."
            )

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
            default_headers=KIMI_DEFAULT_HEADERS,
        )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(
                f"Initialized Kimi LLM with model: {self.model} -> {self.kimi_model} @ {self.api_base}"
            )
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        payload_messages = []
        for message in messages:
            payload_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                }
            )

        params: Dict[str, Any] = {
            "model": self.kimi_model,
            "system": system_message or self.system_message,
            "messages": payload_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "thinking": {"type": "enabled"},
        }
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            params["temperature"] = temperature
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            params["top_p"] = top_p

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as exc:
                error_details = self._format_exception(exc)
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {error_details}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {error_details}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self.client.messages.create(**params)
            )
        except TypeError as exc:
            if "thinking" not in str(exc):
                raise
            fallback_params = dict(params)
            fallback_params.pop("thinking", None)
            logger.warning(
                "Kimi client does not support the 'thinking' parameter. "
                "Retrying request without it."
            )
            response = await loop.run_in_executor(
                None, lambda: self.client.messages.create(**fallback_params)
            )
        content = self._combine_text_blocks(getattr(response, "content", None))
        if content is None:
            raise ValueError(
                f"Kimi returned no visible text content for model={self.kimi_model}."
            )
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {content}")
        return content

    @staticmethod
    def _combine_text_blocks(content_blocks: Any) -> Optional[str]:
        text_blocks: List[str] = []
        for block in content_blocks or []:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "text":
                continue

            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                text_blocks.append(text)
        if not text_blocks:
            return None
        return "\n".join(text_blocks).strip()

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        details = [f"{type(exc).__name__}: {exc}"]
        cause = getattr(exc, "__cause__", None)
        context = getattr(exc, "__context__", None)
        if cause is not None:
            details.append(f"cause={type(cause).__name__}: {cause}")
        if context is not None and context is not cause:
            details.append(f"context={type(context).__name__}: {context}")
        return " | ".join(details)
