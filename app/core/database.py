"""Local, disk-backed Qdrant client (no server, no network)."""
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def create_qdrant_client(storage_path: str) -> QdrantClient:
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=storage_path)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
