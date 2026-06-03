"""
Tests for LLMEnsemble in openevolve.llm.ensemble
"""

import unittest
from typing import Dict, List
from unittest.mock import patch

from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface

class TestLLMEnsemble(unittest.TestCase):
    def test_weighted_sampling(self):
        models = [
            LLMModelConfig(name="a", weight=0.0, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=1.0, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should always sample model 'b'
        for _ in range(10):
            self.assertEqual(ensemble._sample_model().model, "b")

        models = [
            LLMModelConfig(name="a", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="c", weight=0.3, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should sample both models. Track sampled models in a set
        sampled_models = set()
        for _ in range(1000):
            sampled_models.add(ensemble._sample_model().model)
            # Cancel once we have both models
            if len(sampled_models) == len(models):
                break
        self.assertEqual(len(sampled_models), len(models))

    @patch("openevolve.llm.ensemble.KimiLLM")
    @patch("openevolve.llm.ensemble.OpenAILLM")
    def test_kimi2_6_uses_kimi_backend(self, mock_openai_llm, mock_kimi_llm):
        mock_kimi_llm.return_value.model = "kimi2-6"
        mock_openai_llm.return_value.model = "gpt-4o-mini"
        models = [
            LLMModelConfig(name="kimi2-6", weight=1.0),
            LLMModelConfig(name="gpt-4o-mini", weight=1.0),
        ]

        ensemble = LLMEnsemble(models)

        self.assertEqual(len(ensemble.models), 2)
        mock_kimi_llm.assert_called_once_with(models[0])
        mock_openai_llm.assert_called_once_with(models[1])



class TestEnsembleInit(unittest.TestCase):
    class MyCustomLLM(LLMInterface):
        def __init__(self, model, some_field):
            self.model = model
            self.some_field = some_field

        async def generate(self, prompt: str, **kwargs) -> str:
            return "custom-generate"

        async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
            return "custom-generate-with-context"

    def init_custom_llm(self, model_cfg):
        return self.MyCustomLLM(model=model_cfg.name, some_field="value")

    def test_ensemble_initialization(self):
        models = [
            LLMModelConfig(name="a"),
            LLMModelConfig(name="b", init_client=self.init_custom_llm),
        ]
        ensemble = LLMEnsemble(models)
        self.assertEqual(len(ensemble.models), len(models))
        self.assertEqual(ensemble.models[0].model, "a")
        self.assertEqual(ensemble.models[1].model, "b")
        self.assertEqual(ensemble.models[1].some_field, "value")


if __name__ == "__main__":
    unittest.main()
