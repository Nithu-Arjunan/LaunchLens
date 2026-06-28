import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class _FakeToolBoundLlm:
    def invoke(self, messages):
        prompt = messages[0].content
        return AIMessage(content=f"fake agent response\n{prompt}")


class _FailingLlm:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("router unavailable")


class _FakeRouterVerdictLlm:
    def __init__(self, route, region="United States", search_query="smart mugs"):
        self.route = route
        self.region = region
        self.search_query = search_query

    def with_structured_output(self, schema):
        return _FakeStructuredRouter(self.route, self.region, self.search_query)

    def invoke(self, messages):
        first_content = messages[0].content if messages else ""
        if "thread memory" in first_content:
            return AIMessage(content="fake memory answer")
        return AIMessage(content='{"decision":"NICHE","confidence":"Medium","reasoning":"Test verdict.","key_factors":["A","B","C"]}')


class _FakeStructuredRouter:
    def __init__(self, route, region, search_query):
        self.route = route
        self.region = region
        self.search_query = search_query

    def invoke(self, messages):
        return SimpleNamespace(
            route=self.route,
            reason=f"test route {self.route}",
            target_region=self.region,
            target_region_reason=f"test region {self.region}",
            search_query=self.search_query,
            search_query_reason=f"test search query {self.search_query}",
        )


class GraphImportTests(unittest.TestCase):
    def test_graph_module_imports(self):
        graph = importlib.import_module("graph")

        self.assertTrue(callable(graph.build_graph))
        self.assertTrue(callable(graph.get_sqlite_checkpointer_cm))


class GraphFanOutTests(unittest.TestCase):
    def _invoke(self, question):
        graph = importlib.import_module("graph")
        app = graph.build_graph(MemorySaver())
        return app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": f"test-{question}"}},
        )

    def test_demand_route_runs_trends_branch_only(self):
        with patch("nodes.fanout.fetch_google_trends", return_value={"source": "trends"}) as trends:
            with patch("nodes.fanout.fetch_amazon_search", return_value={"source": "amazon"}) as amazon:
                with patch("nodes.fanout.fetch_amazon_product", return_value={"source": "amazon_product"}) as product:
                    with patch("nodes.fanout.fetch_google_news", return_value={"source": "news"}) as news:
                        with patch("nodes._helpers._get_llm", return_value=_FakeRouterVerdictLlm("demand")):
                            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()):
                                result = self._invoke("is there demand for smart mugs?")

        self.assertEqual(result["route"], "demand")
        self.assertEqual(result["route_reason"], "test route demand")
        self.assertEqual(result["target_region"], "United States")
        self.assertEqual(result["target_region_reason"], "test region United States")
        self.assertEqual(result["search_query"], "smart mugs")
        self.assertEqual(result["search_query_reason"], "test search query smart mugs")
        self.assertEqual(result["trends_result"], {"source": "trends"})
        self.assertNotIn("amazon_result", result)
        self.assertNotIn("amazon_products_result", result)
        self.assertNotIn("news_result", result)
        trends.assert_called_once_with("smart mugs", region="United States")
        amazon.assert_not_called()
        product.assert_not_called()
        news.assert_not_called()

    def test_pricing_route_runs_amazon_search_then_product_enrichment_only(self):
        amazon_result = {
            "source": "amazon",
            "domain": "com",
            "listings": [{"asin": "A1"}, {"asin": "A2"}],
        }
        with patch("nodes.fanout.fetch_google_trends", return_value={"source": "trends"}) as trends:
            with patch("nodes.fanout.fetch_amazon_search", return_value=amazon_result) as amazon:
                with patch("nodes.fanout.fetch_amazon_product", return_value={"source": "amazon_product"}) as product:
                    with patch("nodes.fanout.fetch_google_news", return_value={"source": "news"}) as news:
                        with patch("nodes._helpers._get_llm", return_value=_FakeRouterVerdictLlm("pricing")):
                            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()):
                                result = self._invoke("what price should smart mugs have?")

        self.assertEqual(result["route"], "pricing")
        self.assertEqual(result["target_region"], "United States")
        self.assertEqual(result["search_query"], "smart mugs")
        self.assertEqual(result["amazon_result"], amazon_result)
        self.assertEqual(result["amazon_products_result"]["source"], "amazon_products")
        self.assertEqual(len(result["amazon_products_result"]["products"]), 2)
        self.assertNotIn("trends_result", result)
        self.assertNotIn("news_result", result)
        trends.assert_not_called()
        amazon.assert_called_once_with("smart mugs", region="United States")
        product.assert_any_call("A1", region="United States", domain="com")
        product.assert_any_call("A2", region="United States", domain="com")
        news.assert_not_called()

    def test_full_report_route_runs_all_branches_before_agent(self):
        amazon_result = {
            "source": "amazon",
            "domain": "in",
            "listings": [{"asin": "B1"}, {"asin": "B2"}, {"asin": "B3"}],
        }
        with patch("nodes.fanout.fetch_google_trends", return_value={"source": "trends"}) as trends:
            with patch("nodes.fanout.fetch_amazon_search", return_value=amazon_result) as amazon:
                with patch("nodes.fanout.fetch_amazon_product", return_value={"source": "amazon_product"}) as product:
                    with patch("nodes.fanout.fetch_google_news", return_value={"source": "news"}) as news:
                        with patch(
                            "nodes._helpers._get_llm",
                            return_value=_FakeRouterVerdictLlm(
                                "full_report",
                                "India",
                                "eco-friendly lunch boxes",
                            ),
                        ):
                            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()):
                                result = self._invoke("analyze smart mugs market")

        self.assertEqual(result["route"], "full_report")
        self.assertEqual(result["target_region"], "India")
        self.assertEqual(result["search_query"], "eco-friendly lunch boxes")
        self.assertEqual(result["trends_result"], {"source": "trends"})
        self.assertEqual(result["amazon_result"], amazon_result)
        self.assertEqual(result["amazon_products_result"]["source"], "amazon_products")
        self.assertEqual(result["news_result"], {"source": "news"})
        trends.assert_called_once_with("eco-friendly lunch boxes", region="India")
        amazon.assert_called_once_with("eco-friendly lunch boxes", region="India")
        product.assert_any_call("B1", region="India", domain="in")
        product.assert_any_call("B2", region="India", domain="in")
        product.assert_any_call("B3", region="India", domain="in")
        news.assert_called_once_with("eco-friendly lunch boxes", region="India")
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        agent_message = next(m for m in ai_messages if m.content.startswith("fake agent response"))
        self.assertIn("Target launch region: India", agent_message.content)
        self.assertIn("Search query used: eco-friendly lunch boxes", agent_message.content)
        self.assertIn("Google Trends", agent_message.content)
        self.assertIn("Amazon search", agent_message.content)
        self.assertIn("Amazon product enrichment", agent_message.content)
        self.assertIn("Google News", agent_message.content)
        self.assertEqual(result["verdict"]["decision"], "NICHE")

    def test_memory_route_skips_research_and_ends_after_memory_node(self):
        graph = importlib.import_module("graph")
        app = graph.build_graph(MemorySaver())
        config = {"configurable": {"thread_id": "test-memory-route"}}

        with patch("nodes.fanout.fetch_google_trends", return_value={"source": "trends"}):
            with patch("nodes.fanout.fetch_amazon_search", return_value={"source": "amazon"}):
                with patch("nodes.fanout.fetch_amazon_product", return_value={"source": "amazon_product"}):
                    with patch("nodes.fanout.fetch_google_news", return_value={"source": "news"}):
                        with patch("nodes._helpers._get_llm", return_value=_FakeRouterVerdictLlm("full_report")):
                            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()):
                                app.invoke(
                                    {"messages": [HumanMessage(content="analyze smart mugs market")]},
                                    config=config,
                                )

        with patch("nodes.fanout.fetch_google_trends", return_value={"source": "trends"}) as trends:
            with patch("nodes.fanout.fetch_amazon_search", return_value={"source": "amazon"}) as amazon:
                with patch("nodes.fanout.fetch_amazon_product", return_value={"source": "amazon_product"}) as product:
                    with patch("nodes.fanout.fetch_google_news", return_value={"source": "news"}) as news:
                        with patch("nodes._helpers._get_llm", return_value=_FakeRouterVerdictLlm("memory")):
                            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()) as tool_llm:
                                result = app.invoke(
                                    {"messages": [HumanMessage(content="what did I ask earlier?")]},
                                    config=config,
                                )

        self.assertEqual(result["route"], "memory")
        self.assertEqual(
            result["route_reason"],
            "Detected a question about prior conversation or checkpoint memory.",
        )
        self.assertEqual(result["search_query"], "smart mugs")
        self.assertEqual(result["target_region"], "United States")
        trends.assert_not_called()
        amazon.assert_not_called()
        product.assert_not_called()
        news.assert_not_called()
        tool_llm.assert_not_called()
        self.assertEqual(result["messages"][-1].content, "fake memory answer")

    def test_memory_route_is_detected_without_router_llm(self):
        graph = importlib.import_module("graph")
        app = graph.build_graph(MemorySaver())
        config = {"configurable": {"thread_id": "test-memory-router-fallback"}}

        with patch("nodes._helpers._get_llm", return_value=_FailingLlm()):
            with patch("nodes._helpers._get_llm_with_tools", return_value=_FakeToolBoundLlm()) as tool_llm:
                with patch("nodes.fanout.fetch_google_trends") as trends:
                    with patch("nodes.fanout.fetch_amazon_search") as amazon:
                        with patch("nodes.fanout.fetch_amazon_product") as product:
                            with patch("nodes.fanout.fetch_google_news") as news:
                                result = app.invoke(
                                    {"messages": [HumanMessage(content="What products did you research earlier?")]},
                                    config=config,
                                )

        self.assertEqual(result["route"], "memory")
        self.assertIn("not have enough previous conversation context", result["messages"][-1].content)
        tool_llm.assert_not_called()
        trends.assert_not_called()
        amazon.assert_not_called()
        product.assert_not_called()
        news.assert_not_called()


if __name__ == "__main__":
    unittest.main()
