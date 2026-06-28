import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from config import DEFAULT_CHECKPOINT_DB
from state import AgentState
from tools import ALL_TOOLS
from nodes import (
    summarize_node,
    router_node,
    memory_node,
    fetch_trends_node,
    fetch_amazon_node,
    fetch_amazon_products_node,
    fetch_news_node,
    agent_node,
    verdict_node,
)

logger = logging.getLogger(__name__)

# fan-out branch(es) to run for each route. full_report fans out to
# all four at once — returning a list of node names from a conditional
# edge function is what triggers LangGraph to run them in parallel.
ROUTE_TO_BRANCHES = {
    "memory": ["memory"],
    "demand": ["fetch_trends"],
    "pricing": ["fetch_amazon"],
    "full_report": ["fetch_trends", "fetch_amazon", "fetch_news"],
}


def _route_after_router(state: AgentState) -> list[str]:
    route = state.get("route", "full_report")
    branches = ROUTE_TO_BRANCHES.get(route, ROUTE_TO_BRANCHES["full_report"])
    logger.info(
        "graph.router_path: route=%s region=%s search_query=%r branches=%s",
        route,
        state.get("target_region"),
        state.get("search_query"),
        branches,
    )
    return branches


def _route_after_agent(state: AgentState) -> str:
    path = tools_condition(state)
    target = "tools" if path == "tools" else "verdict"
    logger.info("graph.agent_path: condition=%s target=%s", path, target)
    return path


def _research_join(state: AgentState) -> dict:
    logger.info(
        "graph.research_join: fanout_present=%s",
        [
            key
            for key in (
                "trends_result",
                "amazon_result",
                "amazon_products_result",
                "news_result",
            )
            if state.get(key) is not None
        ],
    )
    return {}


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("summarize", summarize_node)
    graph.add_node("router", router_node)
    graph.add_node("memory", memory_node)
    graph.add_node("fetch_trends", fetch_trends_node)
    graph.add_node("fetch_amazon", fetch_amazon_node)
    graph.add_node("fetch_amazon_products", fetch_amazon_products_node)
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("research_join", _research_join)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("verdict", verdict_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "router")

    # Fan-out: router dispatches to 1-3 branches depending on route.
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "memory": "memory",
            "fetch_trends": "fetch_trends",
            "fetch_amazon": "fetch_amazon",
            "fetch_news": "fetch_news",
        },
    )

    # Merge: every branch converges back into agent. Branches that don't
    # run for a given route simply aren't invoked — agent only waits on
    # whichever branches were actually scheduled this step.
    graph.add_edge("fetch_trends", "research_join")
    graph.add_edge("fetch_amazon", "fetch_amazon_products")
    graph.add_edge("fetch_amazon_products", "research_join")
    graph.add_edge("fetch_news", "research_join")
    graph.add_edge("research_join", "agent")
    graph.add_edge("memory", END)

    # Agent <-> tools loop. tools_condition reads the last message: if it
    # has tool_calls, returns "tools"; otherwise "__end__", which we remap
    # to "verdict" so the graph continues instead of actually ending.
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "__end__": "verdict"},
    )
    graph.add_edge("tools", "agent")

    graph.add_edge("verdict", END)

    return graph.compile(checkpointer=checkpointer)


def get_sqlite_checkpointer_cm(db_path: str = DEFAULT_CHECKPOINT_DB):
    """Returns the context manager for the SQLite checkpointer.
    Use with `with get_sqlite_checkpointer_cm() as cp:` so the connection
    closes cleanly."""
    return SqliteSaver.from_conn_string(db_path)
