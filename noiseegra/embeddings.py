"""Embedding-model registry for the diversity metrics.

The published Arabic runs used ``BAAI/bge-m3``, chosen because it is strong on
Arabic. It is still a solid multilingual model, but it is no longer at the top
of the multilingual MTEB board. The default here is ``Qwen/Qwen3-Embedding-0.6B``:
Apache 2.0, ungated, 100+ languages (so Arabic is still covered), 1024-dim, and
about 1.2 GB in float16, which leaves plenty of room on a single T4.

Pass ``--embedding-model bge-m3`` to reproduce the published numbers exactly.

Diversity scores are only comparable within one embedding model. Never put a
bge-m3 Vendi score and a Qwen3 Vendi score in the same table.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional


class EmbeddingModel(NamedTuple):
    hf_id: str
    dim: int
    note: str


EMBEDDING_MODELS: Dict[str, EmbeddingModel] = {
    "qwen3-0.6b": EmbeddingModel(
        "Qwen/Qwen3-Embedding-0.6B", 1024,
        "default; Apache 2.0, 100+ languages, fits alongside anything on one T4",
    ),
    "qwen3-4b": EmbeddingModel(
        "Qwen/Qwen3-Embedding-4B", 2560,
        "stronger, ~8 GB in float16; use when the GPU is otherwise free",
    ),
    "qwen3-8b": EmbeddingModel(
        "Qwen/Qwen3-Embedding-8B", 4096,
        "top of the multilingual MTEB board at release; needs a full 16 GB card",
    ),
    "bge-m3": EmbeddingModel(
        "BAAI/bge-m3", 1024,
        "what the published Arabic paper used; keep for replication",
    ),
    "e5-large": EmbeddingModel(
        "intfloat/multilingual-e5-large-instruct", 1024,
        "well-established multilingual baseline",
    ),
    "gemma-300m": EmbeddingModel(
        "google/embeddinggemma-300m", 768,
        "smallest option; gated on Hugging Face",
    ),
}

DEFAULT_EMBEDDING_MODEL = "qwen3-0.6b"
PAPER_EMBEDDING_MODEL = "bge-m3"


def resolve_embedding_model(name: Optional[str]) -> str:
    """Map a registry key to an HF id. Unknown names pass through unchanged.

    ``resolve_embedding_model("bge-m3") == "BAAI/bge-m3"`` and
    ``resolve_embedding_model("BAAI/bge-m3") == "BAAI/bge-m3"``, so both the short
    keys and raw HF ids work everywhere.
    """
    if name is None:
        name = DEFAULT_EMBEDDING_MODEL
    entry = EMBEDDING_MODELS.get(name.strip().lower())
    return entry.hf_id if entry is not None else name


def load_embedder(name: Optional[str] = None, *, device: Optional[str] = None):
    """Load a ``SentenceTransformer`` for ``name`` (registry key or HF id)."""
    from sentence_transformers import SentenceTransformer

    hf_id = resolve_embedding_model(name)
    model = SentenceTransformer(hf_id, trust_remote_code=True, device=device)
    # Qwen3-Embedding pools the last token, which is only correct with left
    # padding. The repo config sets this, but not every version of
    # sentence-transformers reads it, so pin it here too.
    if "qwen3-embedding" in hf_id.lower():
        try:
            model.tokenizer.padding_side = "left"
        except AttributeError:
            pass
    return model


def describe(name: Optional[str] = None) -> str:
    key = (name or DEFAULT_EMBEDDING_MODEL).strip().lower()
    entry = EMBEDDING_MODELS.get(key)
    if entry is None:
        return f"{name} (not in the registry)"
    return f"{key} -> {entry.hf_id}, {entry.dim}-dim ({entry.note})"
