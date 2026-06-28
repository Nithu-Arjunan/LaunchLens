"""verdict_node — fuses all evidence into a final GO / NICHE / NO-GO decision."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import AgentState

from . import _helpers

logger = logging.getLogger(__name__)


def _parse_verdict_response(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("verdict: model returned non-json response")
        parsed = {}

    return {
        "decision": parsed.get("decision", "NO-GO"),
        "confidence": parsed.get("confidence", "Low"),
        "reasoning": parsed.get("reasoning", content),
        "key_factors": parsed.get("key_factors", []),
    }


def _format_verdict_message(verdict: dict) -> str:
    key_factors = verdict.get("key_factors") or []
    factors_text = "\n".join(f"- {factor}" for factor in key_factors) or "- None provided"
    return (
        f"Verdict: {verdict.get('decision')}\n"
        f"Confidence: {verdict.get('confidence')}\n"
        f"Reasoning: {verdict.get('reasoning')}\n"
        "Key factors:\n"
        f"{factors_text}"
    )


def verdict_node(state: AgentState) -> dict:
    """Fuse fan-out data and agent findings into a final Go / No-Go / Niche verdict."""
    question = _helpers._latest_user_question(state)
    agent_answer = _helpers._latest_ai_answer(state)
    target_region = _helpers._target_region(state)
    search_query = _helpers._search_query(state)

    verdict_prompt = SystemMessage(content=(
        "You are LaunchLens's final decision engine. "
        "Your job is to give a practical, founder-facing startup/product verdict using only the evidence provided. "
        "Do not use outside knowledge. Do not assume missing facts. Do not be optimistic unless the evidence supports it.\n\n"

        "You must choose exactly one verdict: GO, NICHE, or NO-GO.\n\n"

        "Decision definitions:\n"
        "- GO: There is a meaningful demand signal, the market appears reachable, competition/pricing looks workable, "
        "and there are no major red flags. A founder could reasonably proceed to validation or MVP.\n"
        "- NICHE: There is an opportunity, but it only looks attractive for a specific customer segment, use case, price point, "
        "distribution channel, geography, or positioning angle. Broad market entry would be risky.\n"
        "- NO-GO: Demand is weak, evidence is thin, pricing room is poor, the category is too crowded, signals are negative/mixed, "
        "or there is not enough reliable evidence to justify moving forward.\n\n"

        "Evidence evaluation rules:\n"
        "- Prefer concrete evidence over general claims.\n"
        "- If demand signals are unclear or missing, reduce confidence.\n"
        "- If Amazon competition is high but differentiation is weak, avoid GO.\n"
        "- If Google Trends shows declining or flat interest with no strong supporting evidence, avoid GO.\n"
        "- If Google News shows hype but weak buyer intent, do not treat hype as demand.\n"
        "- If evidence sources disagree, mention the conflict and lower confidence.\n"
        "- If evidence is insufficient, choose NO-GO or NICHE with Low confidence, not GO.\n"
        "- Do not recommend GO based only on the agent analysis if the raw evidence does not support it.\n\n"

        "Verdict calibration:\n"
        "- Choose GO only when at least two independent evidence sources support the opportunity.\n"
        "- Choose NICHE when demand exists but success depends on targeting a narrower segment or differentiated positioning.\n"
        "- Choose NO-GO when the idea lacks clear demand, has weak commercial signals, or the evidence is too limited.\n\n"

        "Confidence rules:\n"
        "- High: multiple evidence sources strongly agree and the decision is clear.\n"
        "- Medium: evidence is useful but has some gaps, mixed signals, or assumptions.\n"
        "- Low: evidence is thin, missing, contradictory, or mostly qualitative.\n\n"

        "Output requirements:\n"
        "- Return only valid JSON.\n"
        "- Do not wrap the JSON in markdown.\n"
        "- Do not include extra keys.\n"
        "- Do not include comments.\n"
        "- The reasoning must be one concise paragraph, written for a non-technical founder.\n"
        "- key_factors must contain exactly 3 short, specific factors based on the evidence.\n\n"

        "The JSON object must have exactly these keys:\n"
        "{\n"
        '  "decision": "GO | NICHE | NO-GO",\n'
        '  "confidence": "Low | Medium | High",\n'
        '  "reasoning": "one concise founder-facing paragraph explaining the decision using the evidence",\n'
        '  "key_factors": ["specific factor 1", "specific factor 2", "specific factor 3"]\n'
        "}"
    ))

    evidence_prompt = HumanMessage(content=(
        "Evaluate the following startup/product idea and return the final LaunchLens verdict.\n\n"
        f"Founder question:\n{question}\n\n"
        f"Search query used:\n{search_query}\n\n"
        f"Target launch region:\n{target_region}\n\n"
        f"Selected research route:\n{state.get('route')}\n\n"
        "Evidence collected:\n\n"
        f"1. Google Trends result:\n{state.get('trends_result') or '(not available)'}\n\n"
        f"2. Amazon result:\n{state.get('amazon_result') or '(not available)'}\n\n"
        f"3. Amazon product enrichment result:\n{state.get('amazon_products_result') or '(not available)'}\n\n"
        f"4. Google News result:\n{state.get('news_result') or '(not available)'}\n\n"
        f"5. Agent analysis:\n{agent_answer or '(not available)'}\n\n"
        f"6. Conversation summary:\n{state.get('summary') or '(none)'}\n\n"
        "Important instruction:\n"
        "Base the verdict only on the evidence above. "
        "If the evidence is missing, weak, or contradictory, reflect that in the decision and confidence. "
        "Do not invent market data, competitors, pricing, customer behavior, or demand signals."
    ))

    response = _helpers._get_llm().invoke([verdict_prompt, evidence_prompt])
    verdict = _parse_verdict_response(response.content)
    verdict_message = _format_verdict_message(verdict)

    logger.info(
        "verdict: generated route=%s decision=%s confidence=%s",
        state.get("route"),
        verdict.get("decision"),
        verdict.get("confidence"),
    )
    return {
        "verdict": verdict,
        "messages": [AIMessage(content=verdict_message)],
    }
