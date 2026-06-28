"""router_node — classifies the user question into a research route."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import DEFAULT_REGION
from state import AgentState, RouteType

from . import _helpers

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    route: RouteType = Field(description="One of: demand, pricing, full_report, memory.")
    reason: str = Field(description="Brief reason for choosing the route.")
    target_region: str = Field(
        description=(
            "Launch country or region for the business research. Use United States "
            "when the user does not specify a location."
        )
    )
    target_region_reason: str = Field(description="Brief reason for the selected region.")
    search_query: str = Field(
        description=(
            "Concise product, service, or category phrase to use for external research. "
            "Remove launch-intent wording and location text."
        )
    )
    search_query_reason: str = Field(description="Brief reason for the selected search query.")


def router_node(state: AgentState) -> dict:
    """Classify the user question into the research route using structured LLM output."""
    last_user_msg = state["messages"][-1].content if state["messages"] else ""
    if _helpers._is_memory_question(last_user_msg):
        reason = "Detected a question about prior conversation or checkpoint memory."
        logger.info("router: deterministic_memory_route reason=%s query=%r", reason, last_user_msg)
        return {"route": "memory", "route_reason": reason}

    router_prompt = [
        SystemMessage(content=(
            "You are a routing classifier for LaunchLens. "
            "Classify the founder's question into exactly one route.\n\n"
            "Routes:\n"
            "- demand: market demand, customer interest, trends, popularity, search behavior.\n"
            "- pricing: price, competitor pricing, Amazon listings, product costs, willingness to pay.\n"
            "- full_report: broad validation, market research, go/no-go, or questions needing multiple sources.\n\n"
            "- memory: questions about prior conversation, saved context, remembered preferences, "
            "thread history, earlier ideas, previous verdicts, or summaries of what has already been discussed.\n\n"
            "Also extract the target launch region for the research. If the user mentions a city, "
            "country, geography, marketplace, or phrase like 'in India' or 'for UAE', infer the "
            "country or region. Normalize common regions to names like United States, United Kingdom, "
            "India, United Arab Emirates, Canada, or Australia. If no region is stated, use United States.\n\n"
            "Also extract a clean search query for external research. This must be a concise product, "
            "service, or category phrase, not the full founder question. Remove wording like 'can I launch', "
            "'is there demand for', 'should I sell', 'analyze the market for', and remove location text "
            "because location belongs in target_region. Keep important qualifiers such as eco-friendly, "
            "premium, kids, B2B, subscription, handmade, or vegan. If unsure, use the founder's product "
            "or category phrase from the question.\n\n"
            "For memory questions, do not force a market-research query; use the latest user question as search_query. "
            "When unsure about the route, choose full_report. Do not invent routes or locations."
        )),
        HumanMessage(content=last_user_msg),
    ]

    try:
        decision = _helpers._get_llm().with_structured_output(RouteDecision).invoke(router_prompt)
        route = decision.route
        reason = decision.reason
        target_region = _helpers._normalize_target_region(decision.target_region)
        target_region_reason = decision.target_region_reason
        search_query = (decision.search_query or last_user_msg).strip()
        search_query_reason = decision.search_query_reason
    except Exception as exc:
        route = "full_report"
        reason = f"Router LLM failed; defaulted to full_report ({exc.__class__.__name__})."
        target_region = DEFAULT_REGION
        target_region_reason = (
            f"Router LLM failed; defaulted target region to {DEFAULT_REGION} ({exc.__class__.__name__})."
        )
        search_query = last_user_msg
        search_query_reason = (
            f"Router LLM failed; defaulted search query to the founder question ({exc.__class__.__name__})."
        )
        logger.warning("router: llm_failed fallback_route=%s error=%s", route, exc)

    logger.info(
        "router: route=%s region=%s search_query=%r route_reason=%s region_reason=%s search_query_reason=%s query=%r",
        route,
        target_region,
        search_query,
        reason,
        target_region_reason,
        search_query_reason,
        last_user_msg,
    )
    if route == "memory":
        return {
            "route": route,
            "route_reason": reason,
        }
    return {
        "route": route,
        "route_reason": reason,
        "target_region": target_region,
        "target_region_reason": target_region_reason,
        "search_query": search_query,
        "search_query_reason": search_query_reason,
    }
