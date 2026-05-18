import os
import getpass
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message=r".*allowed_objects.*",
)

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

load_dotenv()


def _ensure_env(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(f"{var}: ")


_ensure_env("GOOGLE_API_KEY")


SYSTEM_PROMPT = "You are a helpful assistant."

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


def _chat_node(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chat", _chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=MemorySaver())


graph = _build_graph()


def chat(prompt: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    return result["messages"][-1].content


def _repl() -> None:
    print(
        "Notion Agent ready (LangGraph). Type your question.\n"
        "Commands: 'reset' clears memory, 'exit' / 'quit' to leave.\n"
    )
    session = 0
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
        if lowered == "reset":
            session += 1
            thread_id = f"session-{session}"
            print("(memory cleared)\n")
            continue

        try:
            answer = chat(question, thread_id=thread_id)
        except Exception as exc:
            print(f"Error: {exc}\n")
            continue

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    _repl()
