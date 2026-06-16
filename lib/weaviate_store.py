"""
Weaviate vector store for the real estate RAG.

Collection schema:
    RealEstateDoc
    ├── text      : TEXT   — full document content (BM25 indexed automatically)
    ├── category  : TEXT   — "listing" | "buying_process" | "market_seller"
    └── vector    : float[] — provided externally (NVIDIA nv-embed-v1)

Search strategy: hybrid (dense vector + BM25) with alpha=0.5
"""

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

COLLECTION_NAME = "RealEstateDoc"


class WeaviateStore:
    def __init__(self, host: str = "localhost", port: int = 8082) -> None:
        self.client = weaviate.connect_to_local(host=host, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection. If recreate=True, drops and rebuilds it."""
        if self.client.collections.exists(COLLECTION_NAME):
            if recreate:
                self.client.collections.delete(COLLECTION_NAME)
                print(f"Dropped existing collection '{COLLECTION_NAME}'.")
            else:
                print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
                return

        self.client.collections.create(
            name=COLLECTION_NAME,
            # We supply vectors ourselves (NVIDIA embeddings)
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
            ],
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(
        self,
        documents: list[str],
        categories: list[str],
        vectors: list[list[float]],
    ) -> None:
        """Batch-insert documents with their category labels and vectors."""
        collection = self.client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for text, category, vector in zip(documents, categories, vectors):
                batch.add_object(
                    properties={"text": text, "category": category},
                    vector=vector,
                )

        print(f"Indexed {len(documents)} document(s) into '{COLLECTION_NAME}'.")

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
        Hybrid search combining dense vector (NVIDIA) and BM25.
        alpha=0.5 → equal weight to both.
        Optionally filter by category.
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

        return [obj.properties["text"] for obj in results.objects]

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.client.close()
