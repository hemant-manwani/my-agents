"""
Weaviate vector store with parent-child chunk support.

Collection schema:
    LeadGenKnowledge
    ├── text        : TEXT   — child chunk (H2/H3 heading + its paragraphs)
    ├── parent_text : TEXT   — parent H1 section (broad context)
    ├── section     : TEXT   — heading title of this chunk
    ├── category    : TEXT   — document-level label (e.g. "reddit_lead_gen")
    └── vector      : float[] — from NVIDIA nv-embed-v1 (on child text only)

Retrieval strategy:
    Search is run against child chunks (small → precise match).
    Both child text and parent_text are returned to the LLM (small → big context).
    Hybrid search: dense vector + BM25, alpha=0.5.
"""

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

COLLECTION_NAME = "LeadGenKnowledge"


class WeaviateStore:
    def __init__(self, host: str = "localhost", port: int = 8082) -> None:
        self.client = weaviate.connect_to_local(host=host, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        if self.client.collections.exists(COLLECTION_NAME):
            if recreate:
                self.client.collections.delete(COLLECTION_NAME)
                print(f"Dropped existing collection '{COLLECTION_NAME}'.")
            else:
                print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
                return

        self.client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text",        data_type=DataType.TEXT),
                Property(name="parent_text", data_type=DataType.TEXT),
                Property(name="section",     data_type=DataType.TEXT),
                Property(name="category",    data_type=DataType.TEXT),
            ],
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> None:
        """
        Batch-insert chunks. Each chunk dict must have:
            text, parent_text, section, category
        Vectors must correspond 1-to-1 with chunks.
        """
        collection = self.client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    properties={
                        "text":        chunk["text"],
                        "parent_text": chunk["parent_text"],
                        "section":     chunk["section"],
                        "category":    chunk["category"],
                    },
                    vector=vector,
                )

        print(f"Indexed {len(chunks)} chunk(s) into '{COLLECTION_NAME}'.")

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 3,
        category: str | None = None,
    ) -> list[str]:
        """
        Search child chunks, return parent_text + child text for each hit.
        This gives the LLM precise retrieval with broad section context.
        """
        collection = self.client.collections.get(COLLECTION_NAME)

        filters = None
        if category:
            from weaviate.classes.query import Filter
            filters = Filter.by_property("category").equal(category)

        results = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=0.5,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        contexts = []
        for obj in results.objects:
            p = obj.properties
            # combine parent context + child detail
            context = f"[Section: {p['section']}]\n"
            if p["parent_text"] and p["parent_text"] != p["text"]:
                context += f"{p['parent_text']}\n\n"
            context += p["text"]
            contexts.append(context)

        return contexts

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.client.close()
