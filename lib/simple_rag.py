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

import math
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Sample knowledge base — real estate sales
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # Listings
    "123 Maple Street, Austin TX 78701: 3 bed / 2 bath, 1,850 sqft, listed at $485,000. "
    "Recently renovated kitchen, open floor plan, two-car garage, and a large backyard. "
    "Located in the highly rated Brentwood school district.",

    "456 Oceanview Drive, Miami FL 33101: 4 bed / 3 bath, 2,400 sqft, listed at $1,150,000. "
    "Waterfront property with private dock, heated pool, and panoramic bay views. "
    "HOA is $450/month and covers landscaping and security.",

    "789 Pine Ridge Road, Denver CO 80201: 2 bed / 2 bath condo, 1,100 sqft, listed at $320,000. "
    "Top-floor unit with mountain views, in-unit laundry, and one assigned parking spot. "
    "Pet-friendly building, walkable to downtown dining and light rail.",

    "22 Elm Court, Chicago IL 60601: 5 bed / 4 bath, 3,200 sqft, listed at $795,000. "
    "Historic brownstone, original hardwood floors, finished basement, and a private rooftop deck. "
    "Walking distance to Lincoln Park and top-rated public schools.",

    "501 Desert Rose Lane, Scottsdale AZ 85251: 3 bed / 2 bath, 2,000 sqft, listed at $620,000. "
    "Single-story, no-step entry, solar panels, resort-style pool. "
    "55+ community with golf course access and low HOA of $180/month.",

    # Buying process
    "The homebuying process starts with mortgage pre-approval, which shows sellers you are a "
    "serious buyer and defines your budget. Pre-approval typically requires proof of income, "
    "tax returns, and a credit check. Most lenders require a credit score of at least 620.",

    "A conventional loan typically requires a 20% down payment to avoid private mortgage "
    "insurance (PMI). FHA loans allow as little as 3.5% down for buyers with a credit score "
    "of 580 or higher. VA loans offer 0% down for eligible veterans and active-duty service members.",

    "Closing costs are typically 2–5% of the loan amount and include lender fees, title "
    "insurance, appraisal, and prepaid taxes and insurance. Buyers can sometimes negotiate "
    "seller concessions to offset these costs.",

    "After an offer is accepted, buyers usually have a 10–17 day inspection period. A licensed "
    "home inspector will check the structure, roof, plumbing, electrical, and HVAC. Buyers can "
    "request repairs or a price reduction based on inspection findings.",

    # Market & negotiation
    "In a seller's market, inventory is low and homes often receive multiple offers above asking "
    "price. Strategies for buyers include escalation clauses, waiving minor contingencies, and "
    "writing a personal letter to the seller.",

    "Comparable sales (comps) are recently sold homes that are similar in size, condition, and "
    "location. Agents use comps to price listings accurately and to advise buyers on offer amounts.",

    "Negotiation tactics include asking for closing cost contributions, requesting repairs before "
    "closing, or negotiating a home warranty. A skilled agent can help structure offers that are "
    "competitive without overpaying.",

    # Seller guidance
    "Staging a home before listing increases perceived value and helps buyers visualize living "
    "there. Decluttering, neutralizing paint colors, and enhancing curb appeal are among the "
    "most cost-effective steps a seller can take.",

    "The listing agent typically earns a commission of 5–6% of the sale price, split between "
    "the buyer's and seller's agents. This is paid at closing from the seller's proceeds.",

    "Days on Market (DOM) measures how long a property has been listed. Homes with high DOM "
    "may signal overpricing or issues. Pricing correctly from day one attracts the most attention "
    "during the critical first two weeks on the market.",
]

# ---------------------------------------------------------------------------
# Pure-Python cosine similarity (no numpy required)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Minimal in-memory vector store
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self) -> None:
        self._docs: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, docs: list[str], vecs: list[list[float]]) -> None:
        self._docs.extend(docs)
        self._vecs.extend(vecs)

    def search(self, query_vec: list[float], top_k: int = 3) -> list[str]:
        scored = sorted(
            range(len(self._vecs)),
            key=lambda i: _cosine(query_vec, self._vecs[i]),
            reverse=True,
        )
        return [self._docs[i] for i in scored[:top_k]]


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
    store: VectorStore,
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
        # embed the latest human message to find relevant chunks
        last_human: HumanMessage = next(
            m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        query_vec = embedder.embed_query(last_human.content)
        chunks = store.search(query_vec, top_k=top_k)
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

    return graph.compile()


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

    # Build and populate the vector store using NVIDIA embeddings
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=nvidia_api_key,
        truncate="NONE",
    )
    print(f"Indexing {len(DOCUMENTS)} document(s)…")
    vecs = embedder.embed_documents(DOCUMENTS)
    store = VectorStore()
    store.add(DOCUMENTS, vecs)
    print("Done.\n")

    # Compile the LangGraph RAG
    rag = build_rag_graph(groq_api_key=groq_api_key, nvidia_api_key=nvidia_api_key, store=store)

    print("Real Estate Agent ready. Type 'quit' or 'exit' to end the conversation.\n")
    messages: list[BaseMessage] = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Agent: Thanks for chatting! Have a great day.")
            break

        messages.append(HumanMessage(content=question))

        result = rag.invoke({"messages": messages, "context": []})

        ai_message: AIMessage = result["messages"][-1]
        messages.append(ai_message)
        print(f"Agent: {ai_message.content}\n")


if __name__ == "__main__":
    main()
