import uuid
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

_CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"
_COLLECTION_NAME = "protocol_chunks"

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
_collection = _client.get_or_create_collection(
    name=_COLLECTION_NAME,
    embedding_function=_ef,
)


def add_documents(texts: list[str], metadatas: list[dict] | None = None) -> None:
    """Embed and add text chunks to the vector store."""
    ids = [str(uuid.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{"i": i} for i in range(len(texts))]
    else:
        metadatas = [m if m else {"i": i} for i, m in enumerate(metadatas)]
    _collection.add(documents=texts, metadatas=metadatas, ids=ids)


def search(query: str, k: int = 4) -> list[dict]:
    """Semantic search; returns list of {content, metadata, distance}."""
    result = _collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"content": doc, "metadata": meta or {}, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def list_chunks(limit: int = 100) -> list[dict]:
    """List stored chunks with content and metadata (e.g. source file). Returns list of {content, metadata}."""
    result = _collection.get(
        limit=limit,
        include=["documents", "metadatas"],
    )
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    return [
        {"content": doc, "metadata": meta or {}}
        for doc, meta in zip(documents, metadatas)
    ]


def list_documents() -> list[str]:
    """Return a sorted list of distinct document sources (filenames)."""
    result = _collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    sources: set[str] = set()
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source")
        if source:
            sources.add(str(source))
    return sorted(sources)


def delete_document(source: str) -> None:
    """Delete all chunks belonging to a given source filename."""
    _collection.delete(where={"source": source})


def clear_all() -> None:
    """Clear all documents from the vector store."""
    import chromadb
    _client.delete_collection(name=_COLLECTION_NAME)
    global _collection
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_ef,
    )