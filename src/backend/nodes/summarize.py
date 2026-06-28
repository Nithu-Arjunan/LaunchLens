"""summarize_node — compresses old messages into a running summary."""

import logging

from langchain_core.messages import RemoveMessage

from config import KEEP_RECENT_N_MESSAGES, SUMMARIZE_AFTER_N_MESSAGES
from state import AgentState

from . import _helpers

logger = logging.getLogger(__name__)


def _split_messages_for_summary(messages: list) -> tuple[list, list]:
    split_at = max(0, len(messages) - KEEP_RECENT_N_MESSAGES)
    while split_at > 0 and messages[split_at].type == "tool":
        split_at -= 1
    return messages[:split_at], messages[split_at:]


def summarize_node(state: AgentState) -> dict:
    """Short-term memory: once the conversation gets long, compress older
    turns into a running summary and drop the raw messages, so the context
    window doesn't grow unbounded across a long chat."""
    messages = state["messages"]

    if len(messages) <= SUMMARIZE_AFTER_N_MESSAGES:
        logger.info(
            "summarize: skipped; message_count=%s threshold=%s",
            len(messages),
            SUMMARIZE_AFTER_N_MESSAGES,
        )
        return {}

    to_summarize, _to_keep = _split_messages_for_summary(messages)
    if not to_summarize:
        logger.info("summarize: skipped; recent tool-call group must stay intact")
        return {}

    existing_summary = state.get("summary", "")
    prompt = (
        "You are maintaining a running summary of a conversation between a "
        "founder and a market-research assistant. Update the summary below "
        "with the new messages, keeping it concise (max ~150 words) and "
        "preserving any concrete facts, product ideas, or decisions made.\n\n"
        f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
        "New messages:\n"
        + "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
    )

    response = _helpers._get_llm().invoke(prompt)
    new_summary = response.content

    logger.info(
        "summarize: compressed_messages=%s kept_recent=%s summary_chars=%s",
        len(to_summarize),
        KEEP_RECENT_N_MESSAGES,
        len(new_summary),
    )

    # RemoveMessage tells LangGraph's add_messages reducer to delete those
    # specific messages from state rather than appending more.
    removals = [RemoveMessage(id=m.id) for m in to_summarize]

    return {
        "summary": new_summary,
        "messages": removals,
    }
