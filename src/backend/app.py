import os
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
GRAPH_PNG = BACKEND_DIR / "graph_out" / "graph.png"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config import DEFAULT_CHECKPOINT_DB, DEFAULT_THREAD_ID
from graph import build_graph, get_sqlite_checkpointer_cm


load_dotenv(BACKEND_DIR / ".env")


class RunRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str = DEFAULT_THREAD_ID


class RunResponse(BaseModel):
    thread_id: str
    question: str
    answer: str
    path: list[str]
    route: str | None
    route_reason: str | None
    search_query: str | None
    target_region: str | None
    checkpoint: dict[str, Any]
    summarization: dict[str, Any]
    fanout: dict[str, Any]
    agent: dict[str, Any]
    verdict: dict[str, Any] | None
    metrics: dict[str, Any]


app = FastAPI(title="LaunchLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _checkpoint_db_path() -> Path:
    configured = Path(os.environ.get("CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB))
    if configured.is_absolute():
        return configured
    return BACKEND_DIR / configured


def _message_content(message: Any) -> str:
    return str(getattr(message, "content", "") or "")


def _current_turn_messages(result: dict[str, Any]) -> list[Any]:
    messages = result.get("messages", [])
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            return messages[index:]
    return messages


def _ai_messages(result: dict[str, Any], *, current_turn_only: bool = False) -> list[Any]:
    messages = _current_turn_messages(result) if current_turn_only else result.get("messages", [])
    return [message for message in messages if getattr(message, "type", None) == "ai"]


def _latest_answer(result: dict[str, Any]) -> str:
    messages = _ai_messages(result, current_turn_only=True)
    return _message_content(messages[-1]) if messages else ""


def _agent_answer(result: dict[str, Any]) -> str:
    messages = _ai_messages(result, current_turn_only=True)
    if not messages:
        return ""
    if result.get("verdict") and len(messages) >= 2:
        return _message_content(messages[-2])
    return _message_content(messages[-1])


def _token_usage_from_messages(result: dict[str, Any]) -> dict[str, int | None]:
    prompt = completion = total = 0
    found = False
    for message in _ai_messages(result, current_turn_only=True):
        usage = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or {}

        input_tokens = usage.get("input_tokens") or token_usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or token_usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens") or token_usage.get("total_tokens")

        if input_tokens is not None:
            prompt += int(input_tokens)
            found = True
        if output_tokens is not None:
            completion += int(output_tokens)
            found = True
        if total_tokens is not None:
            total += int(total_tokens)
            found = True

    if not found:
        return {"prompt": None, "completion": None, "total": None}
    return {
        "prompt": prompt,
        "completion": completion,
        "total": total or prompt + completion,
    }


def _execution_path(result: dict[str, Any]) -> list[str]:
    route = result.get("route")
    path = ["summarize", "router"]
    if route == "memory":
        return path + ["memory"]
    if result.get("trends_result") is not None:
        path.append("fetch_trends")
    if result.get("amazon_result") is not None:
        path.append("fetch_amazon")
    if result.get("amazon_products_result") is not None:
        path.append("fetch_amazon_products")
    if result.get("news_result") is not None:
        path.append("fetch_news")
    if any(result.get(key) is not None for key in ("trends_result", "amazon_result", "amazon_products_result", "news_result")):
        path.append("research_join")
    path.append("agent")
    if result.get("verdict") is not None:
        path.append("verdict")
    return path


def _empty_fanout() -> dict[str, Any]:
    return {
        "trends": None,
        "amazon_search": None,
        "amazon_products": None,
        "news": None,
    }


def _fanout(result: dict[str, Any], *, include_research: bool = True) -> dict[str, Any]:
    if not include_research:
        return _empty_fanout()
    return {
        "trends": result.get("trends_result"),
        "amazon_search": result.get("amazon_result"),
        "amazon_products": result.get("amazon_products_result"),
        "news": result.get("news_result"),
    }


def _checkpoint_info(result: dict[str, Any], thread_id: str, db_path: Path) -> dict[str, Any]:
    return {
        "db_path": str(db_path),
        "thread_id": thread_id,
        "message_count": len(result.get("messages", [])),
        "summary_present": bool(result.get("summary")),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph.png")
def graph_png() -> FileResponse:
    if not GRAPH_PNG.exists():
        raise HTTPException(status_code=404, detail="graph.png has not been generated yet")
    return FileResponse(GRAPH_PNG)


@app.post("/api/run", response_model=RunResponse)
def run_graph(request: RunRequest) -> RunResponse:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    db_path = _checkpoint_db_path()
    start = time.perf_counter()
    try:
        with get_sqlite_checkpointer_cm(str(db_path)) as checkpointer:
            graph_app = build_graph(checkpointer)
            result = graph_app.invoke(
                {"messages": [HumanMessage(content=request.question)]},
                config={"configurable": {"thread_id": request.thread_id}},
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc

    latency_ms = round((time.perf_counter() - start) * 1000)
    summary = result.get("summary")
    is_memory_route = result.get("route") == "memory"
    current_turn_ai_messages = _ai_messages(result, current_turn_only=True)

    return RunResponse(
        thread_id=request.thread_id,
        question=request.question,
        answer=_latest_answer(result),
        path=_execution_path(result),
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        search_query=None if is_memory_route else result.get("search_query"),
        target_region=None if is_memory_route else result.get("target_region"),
        checkpoint=_checkpoint_info(result, request.thread_id, db_path),
        summarization={
            "summary": summary,
            "summary_chars": len(summary or ""),
            "enabled": bool(summary),
        },
        fanout=_fanout(result, include_research=not is_memory_route),
        agent={
            "answer": _agent_answer(result),
            "tool_calls": sum(
                len(getattr(message, "tool_calls", []) or [])
                for message in current_turn_ai_messages
            ),
        },
        verdict=None if is_memory_route else result.get("verdict"),
        metrics={
            "latency_ms": latency_ms,
            "tokens": _token_usage_from_messages(result),
        },
    )


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
