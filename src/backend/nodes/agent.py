"""agent_node — ReAct loop step that reasons over collected fan-out data."""

import logging

from langchain_core.messages import SystemMessage

from state import AgentState

from . import _helpers

logger = logging.getLogger(__name__)


def _messages_safe_for_openai(messages: list) -> list:
    """Remove messages that would cause an OpenAI 400 error.

    """
    safe: list = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            # Collect the tool_call_ids this AI message expects to have answered.
            expected: set[str] = {
                (call.get("id") if isinstance(call, dict) else getattr(call, "id", None))
                for call in tool_calls
            } - {None}

            # Scan the messages that immediately follow for tool responses.
            j = i + 1
            responded: set[str] = set()
            while j < len(messages) and messages[j].type == "tool":
                tid = getattr(messages[j], "tool_call_id", None)
                if tid:
                    responded.add(tid)
                j += 1

            missing = expected - responded
            if missing:
                logger.warning(
                    "agent: dropped AI tool-call message with unanswered tool_call_ids=%s",
                    sorted(missing),
                )
                # Jump past the AI message and any adjacent tool responses —
                # those responses would be orphaned without their request.
                i = j
            else:
                safe.append(msg)
                for tool_msg in messages[i + 1 : j]:
                    tid = getattr(tool_msg, "tool_call_id", None)
                    if tid in expected:
                        safe.append(tool_msg)
                    else:
                        logger.warning("agent: dropped unexpected tool message id=%s", tid)
                i = j

        elif msg.type == "tool":
            # Tool response not immediately after an AI tool-call — orphan.
            logger.warning(
                "agent: dropped orphan tool message id=%s",
                getattr(msg, "tool_call_id", None),
            )
            i += 1

        else:
            safe.append(msg)
            i += 1

    return safe


def agent_node(state: AgentState) -> dict:
    """Real ReAct-loop step. Builds an ephemeral system prompt from
    whatever fan-out results + summary are currently in state, asks the
    tool-bound LLM to respond, and returns ONLY its reply.

    The system prompt is rebuilt fresh every time this node runs (e.g.
    once before any tool calls, and again after each tool result comes
    back) — it is NOT added to state["messages"]. Only the model's actual
    output (which may include tool_calls) gets persisted to history.
    """
    target_region = _helpers._target_region(state)
    search_query = _helpers._search_query(state)

    fanout_summary_lines = []
    for key, label in (
        ("trends_result", "Google Trends"),
        ("amazon_result", "Amazon search"),
        ("amazon_products_result", "Amazon product enrichment"),
        ("news_result", "Google News"),
    ):
        if state.get(key) is not None:
            fanout_summary_lines.append(f"- {label}: {state[key]}")
    fanout_block = "\n".join(fanout_summary_lines) or "(no fan-out data collected for this route)"

    system_prompt = SystemMessage(content=(
        "You are LaunchLens, a market-research assistant helping a founder "
        "assess product viability. You have baseline data already collected "
        "for this turn:\n\n"
        f"Search query used: {search_query}\n"
        f"Target launch region: {target_region}\n\n"
        f"{fanout_block}\n\n"
        f"Running conversation summary: {state.get('summary') or '(none yet)'}\n\n"
        "If the baseline data above is sufficient to answer, just answer. "
        "If you genuinely need more — e.g. a specific competitor, a "
        "narrower query, or a data source not covered above — call one of "
        "your tools. Don't call a tool for data you already have."
    ))

    messages_for_llm = [system_prompt, *_messages_safe_for_openai(state["messages"])]
    response = _helpers._get_llm_with_tools().invoke(messages_for_llm)

    n_tool_calls = len(getattr(response, "tool_calls", []) or [])
    logger.info(
        "agent: route=%s region=%s search_query=%r fanout_present=%s tool_calls=%s",
        state.get("route"),
        target_region,
        search_query,
        [k for k in ("trends_result", "amazon_result", "amazon_products_result", "news_result")
         if state.get(k) is not None],
        n_tool_calls,
    )

    return {"messages": [response]}
