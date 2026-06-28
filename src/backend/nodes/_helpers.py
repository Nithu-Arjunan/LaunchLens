"""Shared helpers and LLM singletons used across node modules."""

import logging
import os

from langchain_openai import ChatOpenAI

from config import DEFAULT_OPENAI_MODEL, DEFAULT_REGION, MEMORY_KEYWORDS, REGION_ALIASES
from state import AgentState
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)

_llm = None
_llm_with_tools = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    return _llm


def _get_llm_with_tools() -> ChatOpenAI:
    """Separate instance from _get_llm() — the summarizer never needs
    tools bound, and binding adds tool-schema overhead to every call."""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools


def _normalize_target_region(region: str | None) -> str:
    cleaned = (region or "").strip()
    if not cleaned:
        return DEFAULT_REGION
    return REGION_ALIASES.get(cleaned.casefold(), cleaned)


def _target_region(state: AgentState) -> str:
    return _normalize_target_region(state.get("target_region"))


def _latest_user_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            return message.content
    return ""


def _search_query(state: AgentState) -> str:
    return (state.get("search_query") or _latest_user_question(state)).strip()


def _preview_titles(items: list[dict], limit: int = 3) -> list[str]:
    return [str(item.get("title")) for item in items[:limit] if item.get("title")]


def _top_asins_from_amazon_result(amazon_result: dict | None, limit: int = 3) -> list[str]:
    asins = []
    for listing in (amazon_result or {}).get("listings", []):
        asin = listing.get("asin")
        if asin and asin not in asins:
            asins.append(asin)
        if len(asins) >= limit:
            break
    return asins


def _is_memory_question(text: str) -> bool:
    normalized = text.casefold()
    return any(keyword in normalized for keyword in MEMORY_KEYWORDS)


def _latest_ai_answer(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type == "ai" and not getattr(message, "tool_calls", None):
            return message.content
    return ""
