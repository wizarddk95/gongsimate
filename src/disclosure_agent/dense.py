"""Pure dense baseline over the same chunks BM25 searches.

BGE-M3 is used only as a dense CLS encoder. Sparse weights, ColBERT vectors,
hybrid fusion, and ANN indexes are not loaded.
"""

from __future__ import annotations

import hashlib
import json
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
    chunk_contains_element,
    compare_retrievers,
    load_eval_corpus,
    reproducibility_record,
    score_search,
    write_eval_run,
)
from disclosure_agent.representation import (
    D1_STRUCTURED_TABLES,
    D1_TARGET_ELEMENT_IDS,
    build_retrieval_representations,
    embedding_input_record,
    representation_sha256,
)
from disclosure_agent.retrieval_unit import (
    D2A_VERSION,
    D2B_VERSION,
    build_retrieval_units,
    compact_row_examples,
    retrieval_units_sha256,
    scope_inheritance_audit,
    scoring_chunks,
    slot_diagnostic,
    summarize_retrieval_units,
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
FOCUS_ELEMENTS = (
    {
        "element_key": "E2362",
        "element_id": "9b792cedd0b5cac5fb96d20943d300ae51dcf509b090e917367dad1152e341de",
        "body_only": True,
    },
    {
        "element_key": "E2361",
        "element_id": "af118098848327c5bfb3f4768cfc6b36d6fd34c59044b42638f467f506e512aa",
        "body_only": False,
    },
    {
        "element_key": "E2357",
        "element_id": "a1e286435bd81607af861e8e7c86ebe870ed8879e66dbfd569871790c9a4d60f",
        "body_only": False,
    },
    {
        "element_key": "E228",
        "element_id": "f55b61e36dd52bfcfeda562f3dcc163dd8bc4051f2176d3b248b13034f60f244",
        "body_only": True,
    },
    {
        "element_key": "E186",
        "element_id": "78be6e94020c610d2411a83f2187ca58f098c8d9f071da7c4d78cb9db0ae5b62",
        "body_only": True,
    },
)


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
            int(chunks[index].get("unit_index") or 0),
            chunks[index]["chunk_id"],
        ),
    )
    hits = []
    for rank, index in enumerate(order, start=1):
        row = chunks[index]
        hit = {
            "rank": rank,
            "chunk_id": row["chunk_id"],
            "ordinal": row["ordinal"],
            "score": float(scores[index]),
        }
        if row.get("canonical_chunk_id"):
            hit["canonical_chunk_id"] = row["canonical_chunk_id"]
            hit["retrieval_unit_id"] = row.get("retrieval_unit_id", row["chunk_id"])
            hit["unit_index"] = int(row.get("unit_index") or 0)
        hits.append(hit)
    return hits


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


def encode_texts(
    model: Any, texts: list[str], batch_size: int, *, show_progress_bar: bool = False
) -> Any:
    import numpy as np

    vectors = model.encode(
        texts,
        prompt=None,
        prompt_name=None,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
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
        "truncation_policy": "recorded_not_stored_as_prefix",
        "token_counts": lengths,
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
    representation_mode: str = "canonical",
    reuse_embeddings: Path | None = None,
    compare_dense_per_question: Path | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise UsageError("--batch-size must be >= 1")
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    if representation_mode not in {"canonical", "structured-table-v1"}:
        raise UsageError(f"unknown representation mode {representation_mode!r}")
    if representation_mode == "canonical" and reuse_embeddings is not None:
        raise UsageError("reuse-embeddings applies only to a derived representation run")
    records = None
    if representation_mode == "canonical":
        texts = [chunk.get("text") or "" for chunk in corpus["chunks"]]
    else:
        records = build_retrieval_representations(
            corpus["chunks"],
            corpus["elements_by_id"],
            mode=representation_mode,
            target_element_ids=D1_TARGET_ELEMENT_IDS,
        )
        texts = [record["text"] for record in records]
        _assert_canonical_chunks_unchanged(corpus["chunks"], records)
    resolved = resolve_device(device)
    load_started = time.perf_counter()
    model = load_dense_model(resolved)
    model_load_seconds = time.perf_counter() - load_started
    lengths = token_length_stats(model, texts, corpus["chunks"])
    token_counts = lengths.pop("token_counts")
    token_deltas = _structured_token_deltas(model, records, corpus["chunks"], token_counts)
    embed_started = time.perf_counter()
    matrix, reuse_info = _encode_document_matrix(
        model,
        texts,
        corpus["chunks"],
        batch_size,
        reuse_embeddings,
    )
    document_embedding_seconds = time.perf_counter() - embed_started
    index_started = time.perf_counter()
    matrix = _unit_vectors(matrix)
    index_build_seconds = time.perf_counter() - index_started
    query_latencies: list[float] = []
    query_vectors: dict[str, Any] = {}
    ranked_hits: dict[str, list[dict[str, Any]]] = {}

    def search_fn(text: str) -> list[dict[str, Any]]:
        started = time.perf_counter()
        if text not in query_vectors:
            query_vectors[text] = encode_texts(model, [text], batch_size=1)[0]
        hits = exact_normalized_search(matrix, query_vectors[text], corpus["chunks"])
        ranked_hits[text] = hits
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
        "document_encoding_policy": (
            DOCUMENT_ENCODING_POLICY
            if representation_mode == "canonical"
            else "structured_table_v1_for_d1_targets_else_canonical_text"
        ),
        "representation_mode": representation_mode,
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
    dense_comparison = None
    if compare_dense_per_question is not None:
        dense_baseline_questions = load_json(compare_dense_per_question)["questions"]
        dense_comparison = compare_retrievers(dense_baseline_questions, questions)
    representation_payload = _representation_payload(
        records,
        corpus["chunks"],
        reuse_info,
        questions,
        ranked_hits,
        token_deltas,
    )
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
        "comparison_to_dense_d0": dense_comparison,
        "representation": representation_payload,
        "questions": questions,
        "failures": _failures(questions, FAILURE_K),
    }
    metric_keys = [
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
        "comparison_to_dense_d0",
        "representation",
        "failures",
    ]
    written = write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=tuple(metric_keys),
    )
    _write_representation_sidecars(output_dir, records, corpus["chunks"], token_counts)
    written["paths"]["embeddings"] = str(embedding_path)
    return written


def _encode_document_matrix(
    model: Any,
    texts: list[str],
    chunks: list[dict[str, Any]],
    batch_size: int,
    reuse_embeddings: Path | None,
) -> tuple[Any, dict[str, Any] | None]:
    import numpy as np

    if reuse_embeddings is None:
        return encode_texts(model, texts, batch_size), None
    raw = reuse_embeddings.read_bytes()
    expected = len(chunks) * EMBEDDING_DIMENSION * 4
    if len(raw) != expected:
        raise UsageError(
            f"reused embedding file has {len(raw)} bytes; expected {expected}"
        )
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(len(chunks), EMBEDDING_DIMENSION).copy()
    changed = [
        index
        for index, (text, chunk) in enumerate(zip(texts, chunks, strict=True))
        if text != (chunk.get("text") or "")
    ]
    if not changed:
        raise UsageError("derived representation did not change any embedded row")
    encoded = _unit_vectors(encode_texts(model, [texts[index] for index in changed], batch_size))
    for row, vector in zip(changed, encoded, strict=True):
        matrix[row] = vector
    return matrix, {
        "reused_embedding_sha256": hashlib.sha256(raw).hexdigest(),
        "reencoded_ordinals": [int(chunks[index]["ordinal"]) for index in changed],
        "reencoded_count": len(changed),
    }


def _assert_canonical_chunks_unchanged(
    chunks: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    if len(chunks) != len(records):
        raise UsageError("representation rows do not match canonical chunks")
    for chunk, record in zip(chunks, records, strict=True):
        if record["chunk_id"] != chunk["chunk_id"] or record["ordinal"] != chunk["ordinal"]:
            raise UsageError("representation identity does not match the canonical chunk")
        if record["source_refs"] != list(chunk.get("chunk_sources") or []):
            raise UsageError("representation dropped canonical source lineage")
        if record["canonical_text_hash"] != chunk.get("text_hash"):
            raise UsageError("representation run rewrote canonical chunk text")


def _structured_token_deltas(
    model: Any,
    records: list[dict[str, Any]] | None,
    chunks: list[dict[str, Any]],
    token_counts: list[int],
) -> list[dict[str, Any]]:
    if not records:
        return []
    by_ordinal = {int(chunk["ordinal"]): chunk for chunk in chunks}
    tokenizer = model.tokenizer
    previous = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    deltas = []
    try:
        for record, count in zip(records, token_counts, strict=True):
            if record["type"] != "structured_table":
                continue
            chunk = by_ordinal[int(record["ordinal"])]
            canonical_tokens = len(
                tokenizer.encode(chunk.get("text") or "", add_special_tokens=True, truncation=False)
            )
            deltas.append(
                {
                    "ordinal": int(record["ordinal"]),
                    "chunk_id": record["chunk_id"],
                    "canonical_tokens": canonical_tokens,
                    "representation_tokens": int(count),
                    "token_delta": int(count) - canonical_tokens,
                    "canonical_chars": len(chunk.get("text") or ""),
                    "representation_chars": len(record["text"]),
                }
            )
    finally:
        tokenizer.model_max_length = previous
    return deltas


def _first_element_ranks(
    hits: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> dict[str, int | None]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    found: dict[str, int | None] = {}
    for spec in FOCUS_ELEMENTS:
        rank = None
        for hit in hits:
            chunk = by_id[hit["chunk_id"]]
            if chunk_contains_element(
                chunk, spec["element_id"], body_only=bool(spec["body_only"])
            ):
                rank = int(hit["rank"])
                break
        found[str(spec["element_key"])] = rank
    return found


def _representation_payload(
    records: list[dict[str, Any]] | None,
    chunks: list[dict[str, Any]],
    reuse_info: dict[str, Any] | None,
    questions: list[dict[str, Any]],
    ranked_hits: dict[str, list[dict[str, Any]]],
    token_deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    element_ranks = {
        question["question_id"]: _first_element_ranks(ranked_hits[question["text"]], chunks)
        for question in questions
    }
    focus_ids = ("Q01", "Q05", "Q07", "Q08", "Q09")
    focus = []
    for question in questions:
        if question["question_id"] not in focus_ids:
            continue
        classification = question["evidence_classification"]
        focus.append(
            {
                "question_id": question["question_id"],
                "first_complete_evidence_path_rank": classification[
                    "first_complete_evidence_path_rank"
                ],
                "first_positive_answer_evidence_rank": classification[
                    "first_positive_answer_evidence_rank"
                ],
                "path_ranks": question["answer_evidence"].get("path_ranks"),
                "element_ranks": element_ranks[question["question_id"]],
                "answer_evidence_success_at_5": question["at_k"]["5"]["answer_evidence_success"],
                "answer_evidence_success_at_20": question["at_k"]["20"]["answer_evidence_success"],
            }
        )
    if records is None:
        return {
            "mode": "canonical",
            "version": "canonical-v1",
            "canonical_chunk_artifact_unchanged": True,
            "focus_questions": focus,
        }
    return {
        "mode": "structured-table-v1",
        "version": "structured-table-v1",
        "targets": list(D1_STRUCTURED_TABLES),
        "target_rule": (
            "Serialize a chunk only when its table body element is in the D1 target set. "
            "The serializer does not branch on these ids."
        ),
        "representation_sha256": representation_sha256(records),
        "structured_chunk_count": sum(record["type"] == "structured_table" for record in records),
        "canonical_chunk_artifact_unchanged": True,
        "reuse": reuse_info,
        "token_deltas": token_deltas,
        "focus_questions": focus,
        "embedding_fields": [
            "embedding_token_count",
            "embedding_truncated",
            "embedding_model_max_tokens",
        ],
    }


def _write_representation_sidecars(
    output_dir: Path,
    records: list[dict[str, Any]] | None,
    chunks: list[dict[str, Any]],
    token_counts: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk.get("text") or "" for chunk in chunks] if records is None else [
        record["text"] for record in records
    ]
    meta_path = output_dir / "embedding_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk, text, count in zip(chunks, texts, token_counts, strict=True):
            recorded = embedding_input_record(
                text, token_count=int(count), max_tokens=MAX_SEQ_LENGTH
            )
            if recorded["stored_text"] != text:
                raise UsageError("embedding metadata replaced the full representation")
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "ordinal": chunk["ordinal"],
                        "embedding_token_count": recorded["embedding_token_count"],
                        "embedding_truncated": recorded["embedding_truncated"],
                        "embedding_model_max_tokens": recorded["embedding_model_max_tokens"],
                        "stored_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "stored_text_chars": len(text),
                        "stored_text_is_full_representation": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    if records is None:
        return
    path = output_dir / "representations.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def run_retrieval_unit_dense(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    output_dir: Path,
    mode: str,
    device: str = "auto",
    batch_size: int = 8,
    reuse_embeddings: Path | None = None,
    compare_dense_per_question: Path | None = None,
    bm25_per_question: Path | None = None,
) -> dict[str, Any]:
    """Dense exact search over D2a or D2b retrieval units. Canonical chunks stay put."""
    if mode not in {"d2a", "d2b"}:
        raise UsageError(f"retrieval unit dense mode must be d2a or d2b, got {mode!r}")
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
    token_counter = _bge_token_counter(model.tokenizer)
    try:
        units = build_retrieval_units(
            corpus["chunks"],
            corpus["elements_by_id"],
            mode=mode,
            token_counter=token_counter,
        )
    finally:
        _restore_tokenizer_limit(model.tokenizer)
    summary = summarize_retrieval_units(
        units,
        target_tokens=512,
        hard_max_tokens=1024,
        model_max_tokens=MAX_SEQ_LENGTH,
    )
    if summary["embedding_truncated"] and mode == "d2a":
        raise UsageError("D2a left an embedding-truncated retrieval unit")
    view = scoring_chunks(units)
    embed_started = time.perf_counter()
    matrix, reuse_info = _encode_retrieval_units(
        model,
        units,
        corpus["chunks"],
        batch_size,
        reuse_embeddings,
    )
    document_embedding_seconds = time.perf_counter() - embed_started
    index_started = time.perf_counter()
    matrix = _unit_vectors(matrix)
    index_build_seconds = time.perf_counter() - index_started
    query_latencies: list[float] = []
    query_vectors: dict[str, Any] = {}
    ranked_hits: dict[str, list[dict[str, Any]]] = {}

    def search_fn(text: str) -> list[dict[str, Any]]:
        started = time.perf_counter()
        if text not in query_vectors:
            query_vectors[text] = encode_texts(model, [text], batch_size=1)[0]
        hits = exact_normalized_search(matrix, query_vectors[text], view)
        ranked_hits[text] = hits
        query_latencies.append(time.perf_counter() - started)
        return hits

    scoring_corpus = dict(corpus)
    scoring_corpus["chunks"] = view
    questions = score_search(scoring_corpus, search_fn)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.f32"
    embedding_sha = write_embedding_matrix(embedding_path, matrix)
    _write_retrieval_units(output_dir, units)
    scope_audit = scope_inheritance_audit(
        list(corpus["elements_by_id"].values()),
        corpus["chunks"],
        scope_element_id=next(
            item["element_id"] for item in FOCUS_ELEMENTS if item["element_key"] == "E2357"
        ),
        title_element_id=next(
            item["element_id"] for item in FOCUS_ELEMENTS if item["element_key"] == "E2361"
        ),
        table_element_id=next(
            item["element_id"] for item in FOCUS_ELEMENTS if item["element_key"] == "E2362"
        ),
    )
    audit_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "audits"
        / "e2357_e2362_scope_audit.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(scope_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    version = D2A_VERSION if mode == "d2a" else D2B_VERSION
    runtime = _runtime_report(
        device=resolved,
        batch_size=batch_size,
        model_load_seconds=model_load_seconds,
        document_embedding_seconds=document_embedding_seconds,
        index_build_seconds=index_build_seconds,
        query_latencies=query_latencies,
    )
    positive_slots = [
        slot_diagnostic(ranked_hits[question["text"]])
        for question in questions
        if question["in_positive_recall_denominator"]
    ]
    retrieval = {
        "retriever_type": "dense",
        "embedding_model": DENSE_MODEL_ID,
        "embedding_model_revision": DENSE_MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "pooling": "cls",
        "normalization": NORMALIZATION,
        "similarity_metric": SIMILARITY_METRIC,
        "query_encoding_policy": QUERY_ENCODING_POLICY,
        "document_encoding_policy": (
            "d2a_bounded_retrieval_unit"
            if mode == "d2a"
            else "d2b_compact_table_row_or_canonical_text"
        ),
        "representation_mode": mode,
        "representation_version": version,
        "max_seq_length": MAX_SEQ_LENGTH,
        "search_method": SEARCH_METHOD,
        "ann": False,
        "sparse_output": False,
        "colbert_output": False,
        "hybrid": False,
        "reranker": False,
        "query_rewrite": False,
        "zero_score_omitted_from_topk": False,
        "embedding_row_order": "retrieval_unit_order",
        "embedding_artifact_sha256": embedding_sha,
        "embedding_dtype": "float32",
        "loaded_prompts": dict(model.prompts),
        "default_prompt_name": model.default_prompt_name,
        "token_stats": {
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_tokens": summary["max_tokens"],
            "truncated_document_count": summary["embedding_truncated"],
            "truncation_policy": "refused_when_a_unit_exceeds_model_max",
        },
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    reproducibility["retrieval_unit"] = {
        "representation_version": version,
        "artifact_sha256": retrieval_units_sha256(units),
        "retrieval_unit_count": len(units),
        "canonical_chunk_artifact_unchanged": True,
        "reuse": reuse_info,
    }
    comparison = None
    if bm25_per_question is not None:
        comparison = compare_retrievers(
            load_json(bm25_per_question)["questions"], questions
        )
    dense_comparison = None
    if compare_dense_per_question is not None:
        dense_comparison = compare_retrievers(
            load_json(compare_dense_per_question)["questions"], questions
        )
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
        "comparison_to_dense_d0": dense_comparison,
        "retrieval_units": {
            "mode": mode,
            "version": version,
            "summary": summary,
            "artifact_sha256": reproducibility["retrieval_unit"]["artifact_sha256"],
            "reuse": reuse_info,
            "slot_diagnostic_positive_questions": positive_slots,
            "slot_diagnostic_mean": _mean_slot_diagnostic(positive_slots),
            "focus_questions": _focus_question_report(questions),
            "compact_row_examples": compact_row_examples(units) if mode == "d2b" else None,
            "scope_audit_path": str(audit_path),
            "canonical_chunk_artifact_unchanged": True,
            "combined_with_other_experiment": False,
        },
        "questions": questions,
        "failures": _failures(questions, FAILURE_K),
    }
    metric_keys = [
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
        "comparison_to_dense_d0",
        "retrieval_units",
        "failures",
    ]
    written = write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=tuple(metric_keys),
    )
    written["paths"]["embeddings"] = str(embedding_path)
    written["paths"]["retrieval_units"] = str(output_dir / "retrieval_units.jsonl")
    return written


def _bge_token_counter(tokenizer: Any) -> Any:
    tokenizer._gongsimate_previous_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9

    def count(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=True, truncation=False))

    return count


def _restore_tokenizer_limit(tokenizer: Any) -> None:
    previous = getattr(tokenizer, "_gongsimate_previous_max_length", None)
    if previous is not None:
        tokenizer.model_max_length = previous


def _encode_retrieval_units(
    model: Any,
    units: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    batch_size: int,
    reuse_embeddings: Path | None,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    matrix = np.zeros((len(units), EMBEDDING_DIMENSION), dtype=np.float32)
    index_by_id = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
    reused_ordinals: list[int] = []
    encode_at: list[int] = []
    texts: list[str] = []
    baseline = None
    reused_sha = None
    if reuse_embeddings is not None:
        raw = reuse_embeddings.read_bytes()
        expected = len(chunks) * EMBEDDING_DIMENSION * 4
        if len(raw) != expected:
            raise UsageError(
                f"reused embedding file has {len(raw)} bytes; expected {expected}"
            )
        baseline = np.frombuffer(raw, dtype=np.float32).reshape(len(chunks), EMBEDDING_DIMENSION)
        reused_sha = hashlib.sha256(raw).hexdigest()
    for index, unit in enumerate(units):
        chunk_index = index_by_id[unit["canonical_chunk_id"]]
        same_text = unit["text"] == (chunks[chunk_index].get("text") or "")
        if (
            baseline is not None
            and same_text
            and unit["representation_type"] == "canonical_text"
        ):
            matrix[index] = baseline[chunk_index]
            reused_ordinals.append(int(unit["canonical_ordinal"]))
            continue
        encode_at.append(index)
        texts.append(unit["text"])
    if texts:
        encoded = encode_texts(model, texts, batch_size, show_progress_bar=True)
        for row, vector in zip(encode_at, encoded, strict=True):
            matrix[row] = vector
    return matrix, {
        "reused_embedding_sha256": reused_sha,
        "reused_row_count": len(reused_ordinals),
        "encoded_row_count": len(texts),
        "reused_only_when_text_matches_canonical_chunk": True,
    }


def _write_retrieval_units(output_dir: Path, units: list[dict[str, Any]]) -> None:
    path = output_dir / "retrieval_units.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n")


def _focus_question_report(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"Q01", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09"}
    rows = []
    for question in questions:
        if question["question_id"] not in wanted:
            continue
        classification = question["evidence_classification"]
        rows.append(
            {
                "question_id": question["question_id"],
                "first_complete_evidence_path_rank": classification[
                    "first_complete_evidence_path_rank"
                ],
                "first_positive_answer_evidence_rank": classification[
                    "first_positive_answer_evidence_rank"
                ],
                "path_ranks": question["answer_evidence"].get("path_ranks"),
                "element_body_ranks": classification.get("element_body_ranks"),
                "hard_negative_ranks": question.get("hard_negative_ranks"),
                "answer_evidence_success": {
                    key: question["at_k"][key]["answer_evidence_success"]
                    for key in ("1", "3", "5", "10", "20")
                },
            }
        )
    return rows


def _mean_slot_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    means = {}
    for key in rows[0]:
        uniques = [int(row[key]["unique_canonical_chunks"]) for row in rows]
        duplicates = [int(row[key]["duplicate_same_source_ru_count"]) for row in rows]
        means[key] = {
            "questions": len(rows),
            "mean_unique_canonical_chunks": sum(uniques) / len(rows),
            "mean_duplicate_same_source_ru_count": sum(duplicates) / len(rows),
            "max_duplicate_same_source_ru_count": max(duplicates),
            "role": "diagnostic",
        }
    return means
