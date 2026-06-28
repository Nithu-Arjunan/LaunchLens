import importlib
import os
import sys
import unittest
import unittest.mock
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class GetLlmTests(unittest.TestCase):
    def test_get_llm_uses_configured_summary_model(self):
        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ.pop("OPENAI_MODEL", None)

            helpers = importlib.import_module("nodes._helpers")
            helpers._llm = None

            llm = helpers._get_llm()

        self.assertEqual(llm.model_name, "gpt-4o-mini")

    def test_get_llm_uses_openai_model_env_when_summary_model_missing(self):
        with unittest.mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4.1-mini"},
            clear=False,
        ):
            helpers = importlib.import_module("nodes._helpers")
            helpers._llm = None

            llm = helpers._get_llm()

        self.assertEqual(llm.model_name, "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
