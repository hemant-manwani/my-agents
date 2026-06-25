"""
Load, chunk (parent-child), embed, and index documents into Weaviate.

Chunking strategy: parent-child
    Parent  = Heading 1 section  (broad context)
    Child   = Heading 2/3 + its paragraphs  (precise, searchable)

    Each child chunk carries:
        text        — child content (what gets embedded + searched)
        parent_text — full H1 section text (passed to LLM for broader context)
        section     — the child's own heading title
        category    — document-level label

    H1 sections with no H2/H3 children are stored as their own self-contained chunk
    (parent_text == text).

Usage:
    .venv/bin/python lib/index_documents.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv

load_dotenv()

from docx import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

sys.path.insert(0, os.path.dirname(__file__))
from weaviate_store import WeaviateStore


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _para_text(para) -> str:
    return para.text.strip()

def _style(para) -> str:
    return para.style.name if para.style else ""


def chunk_docx_parent_child(path: str, category: str) -> list[dict]:
    """
    Parse a .docx into parent-child chunks.

    Returns list of dicts:
        { text, parent_text, section, category }
    """
    doc = Document(path)
    paragraphs = [p for p in doc.paragraphs if _para_text(p)]

    # ------------------------------------------------------------------
    # Pass 1 — build H1 parent sections
    # Each parent = { heading, lines: [str] }
    # ------------------------------------------------------------------
    parents: list[dict] = []
    current_parent: dict | None = None

    for para in paragraphs:
        style = _style(para)
        text = _para_text(para)

        if style == "Heading 1":
            current_parent = {"heading": text, "lines": [text]}
            parents.append(current_parent)
        else:
            if current_parent is None:
                # content before any H1 (e.g. title/intro) — create implicit parent
                current_parent = {"heading": "Introduction", "lines": []}
                parents.append(current_parent)
            current_parent["lines"].append(text)

    # ------------------------------------------------------------------
    # Pass 2 — split each parent into H2/H3 children
    # ------------------------------------------------------------------
    chunks: list[dict] = []

    for parent in parents:
        parent_text = "\n".join(parent["lines"])
        current_child_lines: list[str] = []
        current_section: str = parent["heading"]
        has_children = False

        # walk only the lines that belong to this parent
        for line in parent["lines"]:
            # find the matching paragraph to check its style
            matching = next(
                (p for p in paragraphs
                 if _para_text(p) == line and _style(p) in {"Heading 2", "Heading 3"}),
                None,
            )

            if matching:
                # flush previous child
                if current_child_lines:
                    chunks.append({
                        "text":        "\n".join(current_child_lines),
                        "parent_text": parent_text,
                        "section":     current_section,
                        "category":    category,
                    })
                current_child_lines = [line]
                current_section = line
                has_children = True
            else:
                current_child_lines.append(line)

        # flush last child
        if current_child_lines:
            chunks.append({
                "text":        "\n".join(current_child_lines),
                "parent_text": parent_text,
                "section":     current_section,
                "category":    category,
            })

        # H1 with no H2/H3 children — store as self-contained chunk
        if not has_children:
            chunks.append({
                "text":        parent_text,
                "parent_text": parent_text,
                "section":     parent["heading"],
                "category":    category,
            })

    # deduplicate (self-contained H1 chunks appear twice otherwise)
    seen: set[str] = set()
    unique: list[dict] = []
    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise EnvironmentError("Set NVIDIA_API_KEY in your .env or environment.")

    docx_path = "/Users/hemantmanwani/Downloads/AI_Automation_Lead_Playbook.docx"
    print(f"Loading: {docx_path}")
    chunks = chunk_docx_parent_child(docx_path, category="reddit_lead_gen")
    print(f"Produced {len(chunks)} chunk(s).\n")

    for i, c in enumerate(chunks):
        preview = c["text"][:70].replace("\n", " ")
        parent_preview = c["parent_text"][:50].replace("\n", " ")
        print(f"  [{i+1:02d}] section : {c['section']}")
        print(f"        child   : {preview}…")
        print(f"        parent  : {parent_preview}…")
        print()

    # --- embed child text only -----------------------------------------------
    print(f"Embedding {len(chunks)} chunk(s) with NVIDIA…")
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=nvidia_api_key,
        truncate="NONE",
    )
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    print("Embedding done.\n")

    # --- store in Weaviate ----------------------------------------------------
    store = WeaviateStore()
    store.create_collection(recreate=True)
    store.index_documents(chunks, vectors)
    store.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
