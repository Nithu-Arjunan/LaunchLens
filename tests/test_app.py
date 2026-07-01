import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app


class AppResponseHelperTests(unittest.TestCase):
    def test_memory_route_uses_only_current_turn_messages(self):
        result = {
            "route": "memory",
            "messages": [
                HumanMessage(content="Research launching vegan cosmetics in India"),
                AIMessage(
                    content="old agent answer",
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 40,
                        "total_tokens": 140,
                    },
                ),
                HumanMessage(content="What products did you research earlier?"),
                AIMessage(
                    content="Research was conducted on vegan cosmetics in India.",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 8,
                        "total_tokens": 20,
                    },
                ),
            ],
        }

        self.assertEqual(
            app._latest_answer(result),
            "Research was conducted on vegan cosmetics in India.",
        )
        self.assertEqual(
            app._agent_answer(result),
            "Research was conducted on vegan cosmetics in India.",
        )
        self.assertEqual(
            app._token_usage_from_messages(result),
            {"prompt": 12, "completion": 8, "total": 20},
        )

    def test_memory_route_can_hide_stale_research_payloads(self):
        result = {
            "route": "memory",
            "trends_result": {"old": "trends"},
            "amazon_result": {"old": "amazon"},
            "amazon_products_result": {"old": "products"},
            "news_result": {"old": "news"},
        }

        self.assertEqual(
            app._fanout(result, include_research=False),
            {
                "trends": None,
                "amazon_search": None,
                "amazon_products": None,
                "news": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
