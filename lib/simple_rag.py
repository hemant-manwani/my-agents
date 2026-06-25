"""
Simple RAG for a real estate salesperson, built as a LangGraph graph.

Graph topology:
    START → retrieve → generate → END

State fields:
    question : str           – the user's question
    context  : list[str]     – retrieved document chunks
    answer   : str           – the final generated answer

Usage:
    export NVIDIA_API_KEY="your-nvidia-key"
    export GROQ_API_KEY="your-groq-key"
    python lib/simple_rag.py
"""

import asyncio
import os
import uuid
import warnings
from typing import Annotated, TypedDict

# Suppress before any LangGraph imports — warning fires at module load time
warnings.filterwarnings("ignore", module="langgraph.*")
warnings.filterwarnings("ignore", module="langchain.*")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from weaviate_store import WeaviateStore



# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # auto-appends on update
    context: list[str]


# ---------------------------------------------------------------------------
# RAG graph builder
# ---------------------------------------------------------------------------

def build_rag_graph(
    groq_api_key: str,
    nvidia_api_key: str,
    store: WeaviateStore,
    tools: list[BaseTool],
    checkpointer: AsyncSqliteSaver,
    top_k: int = 3,
):
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=nvidia_api_key,
        truncate="NONE",
    )
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
    ).bind_tools(tools)

    # --- node: retrieve -------------------------------------------------------
    def retrieve(state: RAGState) -> dict:
        last_human: HumanMessage = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        query_vec = embedder.embed_query(last_human.content)
        chunks = store.hybrid_search(
            query_text=last_human.content,
            query_vector=query_vec,
            top_k=top_k,
        )
        return {"context": chunks}

    # --- node: generate -------------------------------------------------------
    def generate(state: RAGState) -> dict:
        context_text = "\n\n".join(f"- {c}" for c in state["context"])
        system = SystemMessage(content=(
            "You are an AI automation opportunity scout. "
            "Your job is to help the user find small, real problems posted by people on Reddit "
            "where AI can automate a painful manual workflow — problems the user can solve as a freelance project.\n\n"

            "The user's skills: building AI agents, RAG pipelines, LLM integrations, and workflow automation.\n\n"

            "## Your behaviour\n"
            "- ALWAYS call the `search_reddit` tool to find real posts. Never invent posts.\n"
            "- Make exactly 2 calls to `search_reddit` with these queries:\n"
            "  1. query='automate manually every day waste hours', subreddit='smallbusiness'\n"
            "  2. query='is there a way to automate workflow copy paste spreadsheet', subreddit='Entrepreneur'\n"
            "- Do NOT call any other tool. Only use `search_reddit`.\n\n"

            "## When you find opportunities, present each one like this:\n"
            "**Problem:** [one sentence summary of what they do manually]\n"
            "**Subreddit:** r/...\n"
            "**Effort:** [Small / Medium] — your estimate of build time\n"
            "**AI solution:** [what you would build: agent, RAG, scraper, workflow, etc.]\n"
            "**Link:** [post URL if available]\n\n"

            "## Rules\n"
            "- Only surface problems that can realistically be solved with AI automation in under 2 weeks.\n"
            "- Skip posts that are already solved, asking for SaaS tool recommendations, or too vague.\n"
            "- If the user asks a follow-up about a specific post, dive deeper into solution design.\n\n"

            + (f"Additional context from knowledge base:\n{context_text}" if context_text.strip() else "")
        ))
        response: AIMessage = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    # --- routing: tool call or done? ------------------------------------------
    def should_use_tools(state: RAGState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # --- wire the graph -------------------------------------------------------
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "generate")  # tool results feed back into generate

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not nvidia_api_key:
        raise EnvironmentError("Set NVIDIA_API_KEY in your .env or environment.")
    if not groq_api_key:
        raise EnvironmentError("Set GROQ_API_KEY in your .env or environment.")

    # Connect to Weaviate (documents are indexed separately via index_documents.py)
    store = WeaviateStore()
    store.create_collection()  # no-op if collection already exists

    # Optional: add REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET to .env for 60 rpm (vs 10 rpm anonymous)
    reddit_env = {}
    reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
    reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if reddit_client_id and reddit_client_secret:
        reddit_env = {
            "REDDIT_CLIENT_ID": reddit_client_id,
            "REDDIT_CLIENT_SECRET": reddit_client_secret,
        }

    # Start the Reddit MCP server and get its tools
    # stderr redirected to /dev/null to suppress the server's [Setup] startup noise
    mcp_client = MultiServerMCPClient(
        {
            "reddit": {
                "command": "sh",
                "args": ["-c", "npx -y reddit-mcp-server 2>/dev/null"],
                "transport": "stdio",
                "env": reddit_env,
            }
        }
    )
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} Reddit tool(s): {[t.name for t in tools]}\n")

    # Compile the LangGraph RAG with async SQLite checkpointer
    db_path = "sessions.db"
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        rag = build_rag_graph(
            groq_api_key=groq_api_key,
            nvidia_api_key=nvidia_api_key,
            store=store,
            tools=tools,
            checkpointer=checkpointer,
        )
        print(f"Sessions persisted to: {db_path}")

        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        print(f"Session started  [id: {session_id}]")
        print("Agent ready. Type 'quit' or 'exit' to end.\n")

        try:
            while True:
                question = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
                if not question:
                    continue
                if question.lower() in {"quit", "exit"}:
                    print("Agent: Thanks for chatting! Have a great day.")
                    break

                result = await rag.ainvoke(
                    {"messages": [HumanMessage(content=question)], "context": []},
                    config=config,
                )

                ai_message: AIMessage = result["messages"][-1]
                print(f"Agent: {ai_message.content}\n")

        except KeyboardInterrupt:
            print("\nAgent: Goodbye!")
        finally:
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
