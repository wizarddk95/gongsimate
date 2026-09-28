"""Pure dense baseline over the same chunks BM25 searches.

BGE-M3 is used only as a dense CLS encoder. Sparse weights, ColBERT vectors,
hybrid fusion, and ANN indexes are not loaded.
"""

from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any

from disclosure_agent.chunking import CHUNKER_VERSION, chunker_config_hash, load_json
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import (
    FAILURE_K,
    KS,
    _aggregate,
    _failures,
    _metric_scope,
    _q07_calculation,
    compare_retrievers,
    load_eval_corpus,
    reproducibility_record,
    score_search,
    write_eval_run,
)

DENSE_MODEL_ID = "BAAI/bge-m3"
DENSE_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
MAX_SEQ_LENGTH = 8192
QUERY_ENCODING_POLICY = "raw_question_text_no_instruction"
DOCUMENT_ENCODING_POLICY = "chunk_retrieval_text"
NORMALIZATION = "l2"
SIMILARITY_METRIC = "normalized_dot_product"
SEARCH_METHOD = "exact_exhaustive"


def exact_normalized_search(
    doc_matrix: Any,
    query_vector: Any,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank every row. L2-normalized dot product is cosine similarity."""
    import numpy as np

    docs = _unit_vectors(np.asarray(doc_matrix, dtype=np.float32))
    query = _unit_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    scores = docs @ query
    order = sorted(
        range(len(chunks)),
        key=lambda index: (
            -float(scores[index]),
            chunks[index]["ordinal"],
            chunks[index]["chunk_id"],
        ),
    )
    return [
        {
            "rank": rank,
            "chunk_id": chunks[index]["chunk_id"],
            "ordinal": chunks[index]["ordinal"],
            "score": float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _unit_vectors(matrix: Any) -> Any:
    import numpy as np

    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.float32(1e-12))


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_dense_model(device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        DENSE_MODEL_ID,
        revision=DENSE_MODEL_REVISION,
        device=device,
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    prompts = dict(getattr(model, "prompts", None) or {})
    default_prompt = getattr(model, "default_prompt_name", None)
    active = {key: value for key, value in prompts.items() if value}
    if active or default_prompt:
        raise UsageError(
            "BGE-M3 dense baseline does not add a query or document prompt. "
            f"Found prompts={active!r} default_prompt_name={default_prompt!r}."
        )
    _assert_dense_contract(model)
    return model


def _child_modules(model: Any) -> list[Any]:
    modules = []
    for index in range(8):
        try:
            modules.append(model[index])
        except (KeyError, IndexError):
            return modules
    return modules


def _embedding_dimension(model: Any) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def _assert_dense_contract(model: Any) -> None:
    modules = _child_modules(model)
    names = [type(module).__name__ for module in modules]
    if names != ["Transformer", "Pooling", "Normalize"]:
        raise UsageError(
            "BGE-M3 dense baseline expected Transformer, CLS pooling, and L2 normalize. "
            f"Loaded modules: {names}."
        )
    pooling = modules[1]
    mode = pooling.pooling_mode
    modes = (mode,) if isinstance(mode, str) else tuple(mode)
    if modes != ("cls",):
        raise UsageError(f"BGE-M3 dense baseline requires CLS pooling, got {modes!r}")
    dimension = _embedding_dimension(model)
    if dimension != EMBEDDING_DIMENSION:
        raise UsageError(
            f"expected embedding dimension {EMBEDDING_DIMENSION}, got {dimension}"
        )


def encode_texts(model: Any, texts: list[str], batch_size: int) -> Any:
    import numpy as np

    vectors = model.encode(
        texts,
        prompt=None,
        prompt_name=None,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIMENSION:
        raise UsageError(
            f"dense encode returned shape {getattr(array, 'shape', None)}; "
            f"expected (*, {EMBEDDING_DIMENSION})"
        )
    if len(array) != len(texts):
        raise UsageError("dense encode did not return one vector per input")
    return array


def token_length_stats(
    model: Any, texts: list[str], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count tokens with the model tokenizer. Truncation follows the 8192-token contract."""
    tokenizer = model.tokenizer
    limit = int(model.max_seq_length)
    lengths = [len(tokenizer.encode(text or "", add_special_tokens=True)) for text in texts]
    truncated = [
        int(chunks[index]["ordinal"])
        for index, length in enumerate(lengths)
        if length > limit
    ]
    return {
        "max_seq_length": limit,
        "max_tokens": max(lengths) if lengths else 0,
        "truncated_document_count": len(truncated),
        "truncated_ordinals": truncated,
        "truncation_policy": "model_max_seq_length",
    }


def write_embedding_matrix(path: Path, matrix: Any) -> str:
    import numpy as np

    raw = np.ascontiguousarray(matrix, dtype=np.float32).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def run_dense_baseline(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: int = 8,
    bm25_per_question: Path | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise UsageError("--batch-size must be >= 1")
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    resolved = resolve_device(device)
    load_started = time.perf_counter()
    model = load_dense_model(resolved)
    model_load_seconds = time.perf_counter() - load_started
    texts = [chunk.get("text") or "" for chunk in corpus["chunks"]]
    lengths = token_length_stats(model, texts, corpus["chunks"])
    embed_started = time.perf_counter()
    matrix = encode_texts(model, texts, batch_size)
    document_embedding_seconds = time.perf_counter() - embed_started
    index_started = time.perf_counter()
    matrix = _unit_vectors(matrix)
    index_build_seconds = time.perf_counter() - index_started
    query_latencies: list[float] = []

    def search_fn(text: str) -> list[dict[str, Any]]:
        started = time.perf_counter()
        query = encode_texts(model, [text], batch_size=1)[0]
        hits = exact_normalized_search(matrix, query, corpus["chunks"])
        query_latencies.append(time.perf_counter() - started)
        return hits

    questions = score_search(corpus, search_fn)
    embedding_path = output_dir / "embeddings.f32"
    embedding_sha = write_embedding_matrix(embedding_path, matrix)
    runtime = _runtime_report(
        device=resolved,
        batch_size=batch_size,
        model_load_seconds=model_load_seconds,
        document_embedding_seconds=document_embedding_seconds,
        index_build_seconds=index_build_seconds,
        query_latencies=query_latencies,
    )
    retrieval = {
        "retriever_type": "dense",
        "embedding_model": DENSE_MODEL_ID,
        "embedding_model_revision": DENSE_MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "pooling": "cls",
        "normalization": NORMALIZATION,
        "similarity_metric": SIMILARITY_METRIC,
        "query_encoding_policy": QUERY_ENCODING_POLICY,
        "document_encoding_policy": DOCUMENT_ENCODING_POLICY,
        "max_seq_length": MAX_SEQ_LENGTH,
        "search_method": SEARCH_METHOD,
        "ann": False,
        "sparse_output": False,
        "colbert_output": False,
        "hybrid": False,
        "reranker": False,
        "zero_score_omitted_from_topk": False,
        "embedding_row_order": "chunk_list_order",
        "embedding_artifact_sha256": embedding_sha,
        "embedding_dtype": "float32",
        "loaded_prompts": dict(model.prompts),
        "default_prompt_name": model.default_prompt_name,
        "token_stats": lengths,
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    comparison = None
    if bm25_per_question is not None:
        baseline_questions = load_json(bm25_per_question)["questions"]
        comparison = compare_retrievers(baseline_questions, questions)
    payload = {
        "retriever_type": "dense",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "chunks": corpus["chunk_summary"],
        "dense": retrieval,
        "runtime": runtime,
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "metrics": _aggregate(
            corpus["dataset"],
            questions,
            KS,
            index_scope=_metric_scope(corpus["scope"]),
        ),
        "comparison_to_bm25": comparison,
        "questions": questions,
        "failures": _failures(questions, FAILURE_K),
    }
    written = write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=(
            "retriever_type",
            "dataset",
            "dataset_sha256",
            "reproducibility",
            "scope",
            "chunker_version",
            "chunker_config_hash",
            "chunks",
            "dense",
            "runtime",
            "calculation_q07",
            "metrics",
            "comparison_to_bm25",
            "failures",
        ),
    )
    written["paths"]["embeddings"] = str(embedding_path)
    return written


def _runtime_report(
    *,
    device: str,
    batch_size: int,
    model_load_seconds: float,
    document_embedding_seconds: float,
    index_build_seconds: float,
    query_latencies: list[float],
) -> dict[str, Any]:
    import sentence_transformers
    import torch

    mean = sum(query_latencies) / len(query_latencies) if query_latencies else None
    report: dict[str, Any] = {
        "device": device,
        "batch_size": batch_size,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_load_seconds": model_load_seconds,
        "document_embedding_seconds": document_embedding_seconds,
        "index_build_seconds": index_build_seconds,
        "query_count": len(query_latencies),
        "query_latency_seconds": {
            "mean": mean,
            "max": max(query_latencies) if query_latencies else None,
            "per_query": query_latencies,
        },
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        report["gpu_name"] = torch.cuda.get_device_name(0)
    return report
