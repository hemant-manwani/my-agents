import os
import random
import sqlite3
import getpass
import warnings
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import tool

warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message=r".*allowed_objects.*",
)

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.config import get_store  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.store.sqlite import SqliteStore  # noqa: E402

load_dotenv()


def _ensure_env(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(f"{var}: ")


_ensure_env("GOOGLE_API_KEY")


USER_ID = "hemant"
MEMORY_NAMESPACE = ("memories", USER_ID)


SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory about the user. "
    "When the user asks a yes-or-no (binary) question, call the "
    "`yes_or_no_oracle` tool exactly once and use its result ('yes' or "
    "'no') to phrase a confident, single-sentence reply. "
    "Whenever the user shares a preference, identifying detail, or "
    "anything you should remember in future conversations, call the "
    "`remember` tool with a concise statement of the fact (do not call "
    "it for one-off requests or transient questions). "
    "For everything else, answer normally without using any tool."
)


@tool
def yes_or_no_oracle() -> str:
    """Decide a yes-or-no answer using a randomized oracle.

    Use this tool whenever the user asks a binary (yes/no) question,
    for example: "Should I take the job?", "Will it rain tomorrow?",
    "Is X better than Y?". Returns the literal string "yes" or "no".
    """
    return "no" if random.random() < 0.5 else "yes"


@tool
def remember(fact: str) -> str:
    """Save a single concise fact about the user to long-term memory.

    Use this whenever the user shares a preference, identifying detail,
    work context, or anything else worth remembering across future
    conversations. Pass a self-contained statement (e.g. "User prefers
    concise answers", "User's name is Hemant"). Do not store transient
    or one-off requests.
    """
    store = get_store()
    store.put(MEMORY_NAMESPACE, key=str(uuid4()), value={"content": fact})
    return f"Remembered: {fact}"


TOOLS = [yes_or_no_oracle, remember]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)


def _load_memory_block() -> str:
    store = get_store()
    items = list(store.search(MEMORY_NAMESPACE))
    if not items:
        return ""
    bullets = "\n".join(f"- {item.value['content']}" for item in items)
    return f"\n\nKnown facts about the user:\n{bullets}"


def _chat_node(state: MessagesState) -> dict:
    system = SYSTEM_PROMPT + _load_memory_block()
    messages = [SystemMessage(content=system), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _route_after_chat(state: MessagesState) -> Literal["tools", "__end__"]:
    """Decide where to go after the chat node has run.

    Inspect the last message produced by the LLM:
    - if it contains tool calls, dispatch them via the `tools` node.
    - otherwise, the turn is complete and we end the run.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


CHECKPOINT_DB_PATH = Path(__file__).parent / "db" / "agent.sqlite"
STORE_DB_PATH = Path(__file__).parent / "db" / "memories.sqlite"


def _open_checkpointer() -> SqliteSaver:
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def _open_store() -> SqliteStore:
    STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(STORE_DB_PATH),
        check_same_thread=False,
        isolation_level=None,
    )
    store = SqliteStore(conn)
    store.setup()
    return store


def _build_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("chat", _chat_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "chat")

    builder.add_conditional_edges(
        source="chat",
        path=_route_after_chat,
        path_map={
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "chat")

    return builder.compile(
        checkpointer=_open_checkpointer(),
        store=_open_store(),
    )


graph = _build_graph()


GRAPH_IMAGE_PATH = Path(__file__).parent / "images" / "graph.png"


def save_graph_image(path: Path = GRAPH_IMAGE_PATH) -> Path | None:
    """Render the compiled graph to a PNG and save it to `path`.

    Returns the path on success, or `None` if rendering failed (e.g.
    offline / mermaid.ink unreachable). In that case the mermaid source
    is printed as a fallback so the structure is still inspectable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
    except Exception as exc:
        print(f"(could not render PNG: {exc}; printing mermaid source)\n")
        print(graph.get_graph().draw_mermaid())
        return None

    path.write_bytes(png_bytes)
    return path


def chat(prompt: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    return result["messages"][-1].content


def _chunk_text(chunk_content) -> str:
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        parts = []
        for part in chunk_content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def stream_chat(prompt: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    for msg_chunk, _metadata in graph.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="messages",
    ):
        if not isinstance(msg_chunk, AIMessageChunk):
            continue
        text = _chunk_text(getattr(msg_chunk, "content", ""))
        if text:
            yield text


def _repl() -> None:
    print(
        "Notion Agent ready (LangGraph). Type your question.\n"
        "Commands: 'graph' saves the graph image, "
        "'reset' clears memory, 'exit' / 'quit' to leave.\n"
    )
    saved = save_graph_image()
    if saved is not None:
        print(f"Graph image saved to {saved}\n")

    session = 1
    thread_id = f"session-{session}"

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered == "graph":
            saved = save_graph_image()
            if saved is not None:
                print(f"\nGraph image saved to {saved}\n")
            continue
        if lowered == "reset":
            session += 1
            thread_id = f"session-{session}"
            print("(memory cleared)\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            for chunk in stream_chat(question, thread_id=thread_id):
                print(chunk, end="", flush=True)
        except Exception as exc:
            print(f"\nError: {exc}")
            continue

        print("\n")


if __name__ == "__main__":
    _repl()
