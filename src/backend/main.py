import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from config import DEFAULT_CHECKPOINT_DB, DEFAULT_THREAD_ID
from diagram import show_graph
from graph import build_graph, get_sqlite_checkpointer_cm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="LaunchLens CLI (Phase 1)")
    parser.add_argument(
        "--thread", default=DEFAULT_THREAD_ID,
        help="Conversation thread id — reuse this to resume a past session via the checkpointer.",
    )
    parser.add_argument(
        "--db", default=DEFAULT_CHECKPOINT_DB,
        help="Path to the SQLite checkpoint database.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    print(f"LaunchLens (Phase 1 — basic graph + SQLite memory). Thread: '{args.thread}'")
    print("Type 'exit' to quit.\n")

    with get_sqlite_checkpointer_cm(args.db) as checkpointer:
        app = build_graph(checkpointer)
        show_graph(app)
        config = {"configurable": {"thread_id": args.thread}}

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            ai_messages = [m for m in result["messages"] if m.type == "ai"]
            if ai_messages:
                print(f"\nLaunchLens: {ai_messages[-1].content}\n")

            if result.get("summary"):
                print(f"[memory] running summary present ({len(result['summary'])} chars)\n")


if __name__ == "__main__":
    main()
