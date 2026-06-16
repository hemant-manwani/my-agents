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

import os
import sqlite3
import uuid
import warnings
from typing import Annotated, TypedDict

# Must be set before LangGraph imports, which trigger the warning on module load
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from weaviate_store import WeaviateStore

# ---------------------------------------------------------------------------
# Sample knowledge base — real estate sales (text + category)
# ---------------------------------------------------------------------------

DOCUMENTS: list[tuple[str, str]] = [
    # (text, category)

    # Listings
    (
        "123 Maple Street, Austin TX 78701: 3 bed / 2 bath, 1,850 sqft, listed at $485,000. "
        "Recently renovated kitchen, open floor plan, two-car garage, and a large backyard. "
        "Located in the highly rated Brentwood school district.",
        "listing",
    ),
    (
        "456 Oceanview Drive, Miami FL 33101: 4 bed / 3 bath, 2,400 sqft, listed at $1,150,000. "
        "Waterfront property with private dock, heated pool, and panoramic bay views. "
        "HOA is $450/month and covers landscaping and security.",
        "listing",
    ),
    (
        "789 Pine Ridge Road, Denver CO 80201: 2 bed / 2 bath condo, 1,100 sqft, listed at $320,000. "
        "Top-floor unit with mountain views, in-unit laundry, and one assigned parking spot. "
        "Pet-friendly building, walkable to downtown dining and light rail.",
        "listing",
    ),
    (
        "22 Elm Court, Chicago IL 60601: 5 bed / 4 bath, 3,200 sqft, listed at $795,000. "
        "Historic brownstone, original hardwood floors, finished basement, and a private rooftop deck. "
        "Walking distance to Lincoln Park and top-rated public schools.",
        "listing",
    ),
    (
        "501 Desert Rose Lane, Scottsdale AZ 85251: 3 bed / 2 bath, 2,000 sqft, listed at $620,000. "
        "Single-story, no-step entry, solar panels, resort-style pool. "
        "55+ community with golf course access and low HOA of $180/month.",
        "listing",
    ),

    # Buying process
    (
        "The homebuying process starts with mortgage pre-approval, which shows sellers you are a "
        "serious buyer and defines your budget. Pre-approval typically requires proof of income, "
        "tax returns, and a credit check. Most lenders require a credit score of at least 620.",
        "buying_process",
    ),
    (
        "A conventional loan typically requires a 20% down payment to avoid private mortgage "
        "insurance (PMI). FHA loans allow as little as 3.5% down for buyers with a credit score "
        "of 580 or higher. VA loans offer 0% down for eligible veterans and active-duty service members.",
        "buying_process",
    ),
    (
        "Closing costs are typically 2–5% of the loan amount and include lender fees, title "
        "insurance, appraisal, and prepaid taxes and insurance. Buyers can sometimes negotiate "
        "seller concessions to offset these costs.",
        "buying_process",
    ),
    (
        "After an offer is accepted, buyers usually have a 10–17 day inspection period. A licensed "
        "home inspector will check the structure, roof, plumbing, electrical, and HVAC. Buyers can "
        "request repairs or a price reduction based on inspection findings.",
        "buying_process",
    ),

    # Market & seller guidance
    (
        "In a seller's market, inventory is low and homes often receive multiple offers above asking "
        "price. Strategies for buyers include escalation clauses, waiving minor contingencies, and "
        "writing a personal letter to the seller.",
        "market_seller",
    ),
    (
        "Comparable sales (comps) are recently sold homes that are similar in size, condition, and "
        "location. Agents use comps to price listings accurately and to advise buyers on offer amounts.",
        "market_seller",
    ),
    (
        "Negotiation tactics include asking for closing cost contributions, requesting repairs before "
        "closing, or negotiating a home warranty. A skilled agent can help structure offers that are "
        "competitive without overpaying.",
        "market_seller",
    ),
    (
        "Staging a home before listing increases perceived value and helps buyers visualize living "
        "there. Decluttering, neutralizing paint colors, and enhancing curb appeal are among the "
        "most cost-effective steps a seller can take.",
        "market_seller",
    ),
    (
        "The listing agent typically earns a commission of 5–6% of the sale price, split between "
        "the buyer's and seller's agents. This is paid at closing from the seller's proceeds.",
        "market_seller",
    ),
    (
        "Days on Market (DOM) measures how long a property has been listed. Homes with high DOM "
        "may signal overpricing or issues. Pricing correctly from day one attracts the most attention "
        "during the critical first two weeks on the market.",
        "market_seller",
    ),
]


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
    db_path: str = "sessions.db",
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
    )

    # --- node: retrieve -------------------------------------------------------
    def retrieve(state: RAGState) -> dict:
        last_human: HumanMessage = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        query_vec = embedder.embed_query(last_human.content)
        # hybrid search: dense (NVIDIA) + BM25, alpha=0.5
        chunks = store.hybrid_search(
            query_text=last_human.content,
            query_vector=query_vec,
            top_k=top_k,
        )
        print(chunks)
        return {"context": chunks}

    # --- node: generate -------------------------------------------------------
    def generate(state: RAGState) -> dict:
        context_text = "\n\n".join(f"- {c}" for c in state["context"])
        system = SystemMessage(content=(
            "You are a knowledgeable real estate sales agent. "
            "Use only the context below to answer the customer's questions. "
            "If the answer is not in the context, say so politely.\n\n"
            f"Context:\n{context_text}"
        ))
        # pass system message + full conversation history directly to the LLM
        response: AIMessage = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    # --- wire the graph -------------------------------------------------------
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return graph.compile(checkpointer=SqliteSaver(conn))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not nvidia_api_key:
        raise EnvironmentError("Set NVIDIA_API_KEY in your .env or environment.")
    if not groq_api_key:
        raise EnvironmentError("Set GROQ_API_KEY in your .env or environment.")

    # Connect to Weaviate and index documents (skips if collection already exists)
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=nvidia_api_key,
        truncate="NONE",
    )
    store = WeaviateStore()
    store.create_collection()

    if not store.client.collections.get("RealEstateDoc").query.fetch_objects(limit=1).objects:
        texts = [doc for doc, _ in DOCUMENTS]
        categories = [cat for _, cat in DOCUMENTS]
        print(f"Indexing {len(texts)} document(s)…")
        vecs = embedder.embed_documents(texts)
        store.index_documents(texts, categories, vecs)
    else:
        print("Documents already indexed — skipping.\n")

    # Compile the LangGraph RAG (sessions persisted to SQLite)
    db_path = "sessions.db"
    rag = build_rag_graph(groq_api_key=groq_api_key, nvidia_api_key=nvidia_api_key, store=store, db_path=db_path)
    print(f"Sessions persisted to: {db_path}")

    # session_id = str(uuid.uuid4())
    session_id = "4a01ed2b-1452-4259-87e9-90d9a55ca287"
    config = {"configurable": {"thread_id": session_id}}
    print(f"Session started  [id: {session_id}]")
    print("Real Estate Agent ready. Type 'quit' or 'exit' to end.\n")

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                print("Agent: Thanks for chatting! Have a great day.")
                break

            result = rag.invoke(
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
    main()
