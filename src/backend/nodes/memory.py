"""memory_node — answers questions from checkpoint-restored graph state."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import AgentState

from . import _helpers

logger = logging.getLogger(__name__)


def _recent_conversation_lines(state: AgentState, limit: int = 8) -> list[str]:
    lines = []
    messages = list(state.get("messages", []))
    if messages and messages[-1].type == "human":
        messages = messages[:-1]
    for message in messages:
        if message.type not in {"human", "ai"}:
            continue
        content = str(message.content).strip()
        if not content:
            continue
        lines.append(f"{message.type}: {content}")
    return lines[-limit:]


def _format_memory_answer(memory_facts: list[str]) -> str:
    if not memory_facts:
        return "I do not have enough previous conversation context in this thread yet."
    return "Here is what I found in this thread's checkpoint memory:\n\n" + "\n\n".join(memory_facts)


def memory_node(state: AgentState) -> dict:
    """Answer memory questions from checkpoint-restored graph state."""
    summary = state.get("summary")
    recent_lines = _recent_conversation_lines(state)
    verdict = state.get("verdict")

    memory_facts = []
    if summary:
        memory_facts.append(f"Running summary: {summary}")
    if state.get("search_query"):
        memory_facts.append(f"Latest research query: {state.get('search_query')}")
    if state.get("target_region"):
        memory_facts.append(f"Latest target region: {state.get('target_region')}")
    if state.get("route") and state.get("route") != "memory":
        memory_facts.append(f"Latest research route: {state.get('route')}")
    if verdict:
        memory_facts.append(
            "Latest verdict: "
            f"{verdict.get('decision')} ({verdict.get('confidence')} confidence)"
        )
    if recent_lines:
        memory_facts.append("Recent conversation:\n" + "\n".join(recent_lines))

    if not memory_facts:
        answer = "I do not have enough previous conversation context in this thread yet."
    else:
        prompt = [
            SystemMessage(content=(
                "You answer questions about the current LaunchLens thread memory only. "
                "Use only the checkpoint-restored context provided. Do not do market research, "
                "do not call tools, and do not invent missing details. Keep the answer concise."
            )),
            HumanMessage(content=(
                f"User memory question:\n{_helpers._latest_user_question(state)}\n\n"
                "Checkpoint-restored context:\n"
                + "\n\n".join(memory_facts)
            )),
        ]
        try:
            response = _helpers._get_llm().invoke(prompt)
            answer = response.content
        except Exception as exc:
            logger.warning("memory: llm_failed fallback_to_direct_summary error=%s", exc)
            answer = _format_memory_answer(memory_facts)

    logger.info(
        "memory: answered from checkpoint_state summary_present=%s recent_lines=%s verdict_present=%s",
        bool(summary),
        len(recent_lines),
        bool(verdict),
    )
    return {"messages": [AIMessage(content=answer)]}
