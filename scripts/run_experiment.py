from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_CACHE_DIR = ROOT / ".cache" / "embeddings"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "while",
    "why",
    "with",
}
METHOD_ALIASES = {
    "summary_memory": "summary_memory_extractive",
    "retrieval_memory": "retrieval_memory_keyword",
    "hybrid_memory": "retrieval_plus_summary",
    "adaptive_memory": "adaptive_context",
}


@dataclass
class DocumentSpec:
    document_id: str
    path: Path
    text: str


@dataclass
class QuestionSpec:
    question_id: str
    document_id: str
    category: str
    question: str
    expected_points: list[Any]


@dataclass
class ContextBuild:
    context_label: str
    context: str
    instruction: str
    build_seconds: float
    metadata: dict[str, Any]


@dataclass
class MonitorStats:
    peak_vram_mib: int | None = None
    peak_ram_percent: float | None = None
    samples: int = 0


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.stats = MonitorStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            vram = query_vram_mib()
            ram = query_ram_percent()
            if vram is not None:
                self.stats.peak_vram_mib = max(self.stats.peak_vram_mib or 0, vram)
            if ram is not None:
                self.stats.peak_ram_percent = max(self.stats.peak_ram_percent or 0.0, ram)
            self.stats.samples += 1
            self._stop.wait(self.interval_seconds)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    allowed_root = (ROOT / "results").resolve()
    if resolved_output == allowed_root or allowed_root not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clean output directory outside {allowed_root}: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def approximate_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def approximate_token_count(text: str) -> int:
    return len(approximate_tokens(text))


def trim_to_token_budget(text: str, token_budget: int, from_end: bool = False) -> str:
    tokens = approximate_tokens(text)
    if len(tokens) <= token_budget:
        return text.strip()
    selected = tokens[-token_budget:] if from_end else tokens[:token_budget]
    return detokenize(selected)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text.strip()) if sentence.strip()]


def content_terms(text: str) -> list[str]:
    terms = []
    for token in approximate_tokens(text.lower()):
        if len(token) > 2 and token not in STOPWORDS and re.search(r"\w", token):
            terms.append(token)
    return terms


def make_sliding_window(document: str, token_budget: int) -> str:
    return trim_to_token_budget(document, token_budget, from_end=True)


def make_extractive_summary(document: str, question: str, token_budget: int) -> str:
    sentences = split_sentences(document)
    if not sentences:
        return trim_to_token_budget(document, token_budget)

    question_terms = set(content_terms(question))
    frequencies = Counter(content_terms(document))
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        terms = content_terms(sentence)
        if not terms:
            score = 0.0
        else:
            frequency_score = sum(math.log1p(frequencies.get(term, 0)) for term in terms)
            question_bonus = sum(2.0 for term in terms if term in question_terms)
            position_bonus = 0.15 if index in (0, len(sentences) - 1) else 0.0
            score = ((frequency_score + question_bonus) / math.sqrt(len(terms))) + position_bonus
        scored.append((score, index, sentence))

    selected: list[tuple[int, str]] = []
    used_tokens = 0
    for _, index, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
        sentence_tokens = approximate_token_count(sentence)
        if selected and used_tokens + sentence_tokens > token_budget:
            continue
        selected.append((index, sentence))
        used_tokens += sentence_tokens
        if used_tokens >= token_budget:
            break

    ordered_sentences = [sentence for _, sentence in sorted(selected)]
    return trim_to_token_budget(" ".join(ordered_sentences), token_budget)


def chunk_document(document: str, chunk_token_budget: int) -> list[str]:
    sentences = split_sentences(document)
    if not sentences:
        return [trim_to_token_budget(document, chunk_token_budget)]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = approximate_token_count(sentence)
        if current and current_tokens + sentence_tokens > chunk_token_budget:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def keyword_retrieval(document: str, question: str, chunk_tokens: int, top_k: int) -> tuple[str, dict[str, Any]]:
    chunks = chunk_document(document, chunk_tokens)
    question_terms = set(content_terms(question))
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        terms = content_terms(chunk)
        overlap = sum(1 for term in terms if term in question_terms)
        unique_overlap = len(set(terms) & question_terms)
        score = float(overlap + (2 * unique_overlap))
        scored.append((score, index, chunk))
    context, meta = format_selected_chunks(scored, top_k)
    return context, {"retrieval_backend": "keyword", "chunk_count": len(chunks)} | meta


def tfidf_retrieval(document: str, question: str, chunk_tokens: int, top_k: int) -> tuple[str, dict[str, Any]]:
    chunks = chunk_document(document, chunk_tokens)
    chunk_terms = [content_terms(chunk) for chunk in chunks]
    question_terms = content_terms(question)
    document_frequency: Counter[str] = Counter()
    for terms in chunk_terms:
        document_frequency.update(set(terms))

    total_chunks = max(1, len(chunks))
    query_tf = Counter(question_terms)
    query_vector = {
        term: count * math.log((1 + total_chunks) / (1 + document_frequency.get(term, 0))) + 1
        for term, count in query_tf.items()
    }
    scored: list[tuple[float, int, str]] = []
    for index, terms in enumerate(chunk_terms):
        chunk_tf = Counter(terms)
        chunk_vector = {
            term: count * math.log((1 + total_chunks) / (1 + document_frequency.get(term, 0))) + 1
            for term, count in chunk_tf.items()
        }
        scored.append((cosine_similarity(query_vector, chunk_vector), index, chunks[index]))
    context, meta = format_selected_chunks(scored, top_k)
    return context, {"retrieval_backend": "tfidf", "chunk_count": len(chunks)} | meta


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def embedding_retrieval(
    document: str,
    question: str,
    chunk_tokens: int,
    top_k: int,
    host: str,
    embedding_model: str,
) -> tuple[str, dict[str, Any]]:
    chunks = chunk_document(document, chunk_tokens)
    query_embedding = call_ollama_embedding(host, embedding_model, question)
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_embedding = call_ollama_embedding(host, embedding_model, chunk)
        score = dense_cosine_similarity(query_embedding, chunk_embedding)
        scored.append((score, index, chunk))
    context, meta = format_selected_chunks(scored, top_k)
    return context, {
        "retrieval_backend": "embedding",
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
    } | meta


def dense_cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(l_item * r_item for l_item, r_item in zip(left, right))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def format_selected_chunks(scored: list[tuple[float, int, str]], top_k: int) -> tuple[str, dict[str, Any]]:
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected = ranked[:top_k]
    selected_in_order = sorted(selected, key=lambda item: item[1])
    top_score = ranked[0][0] if ranked else 0.0
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    score_gap = top_score - second_score
    selected_ids = [str(index + 1) for _, index, _ in selected_in_order]
    selected_scores = [f"{score:.4f}" for score, _, _ in selected_in_order]
    context = "\n\n".join(
        f"Chunk {index + 1} (score={score:.4f}):\n{chunk}"
        for score, index, chunk in selected_in_order
    )
    meta = {
        "retrieval_top_score": f"{top_score:.6f}",
        "retrieval_second_score": f"{second_score:.6f}",
        "retrieval_score_gap": f"{score_gap:.6f}",
        "retrieval_confidence": f"{score_gap:.6f}",
        "selected_chunk_ids": ",".join(selected_ids),
        "selected_chunk_scores": ",".join(selected_scores),
    }
    return context, meta


def make_retrieval_plus_summary_context(document: str, question: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    summary_tokens = int(config.get("hybrid_summary_tokens", max(160, int(config["summary_tokens"]) // 2)))
    retrieval_top_k = int(config.get("hybrid_retrieval_top_k", max(1, int(config["retrieval_top_k"]) - 1)))
    retrieval_chunk_tokens = int(config.get("hybrid_retrieval_chunk_tokens", config["retrieval_chunk_tokens"]))
    summary = make_extractive_summary(document, question, summary_tokens)
    retrieved, retrieval_meta = tfidf_retrieval(document, question, retrieval_chunk_tokens, retrieval_top_k)
    context = (
        "Summary:\n"
        f"{summary}\n\n"
        "Retrieved chunks:\n"
        f"{retrieved}"
    )
    meta = {
        "summary_backend": "extractive",
        "retrieval_backend": "tfidf",
        "hybrid_summary_tokens": summary_tokens,
        "hybrid_retrieval_top_k": retrieval_top_k,
    } | retrieval_meta
    return context, meta


def is_recent_context_question(question: str, question_category: str) -> bool:
    if question_category == "fact_near_end":
        return True
    terms = set(content_terms(question))
    return bool(terms & {"recent", "latest", "last", "newest", "final"})


def choose_adaptive_context(
    document: str,
    question: str,
    question_category: str,
    config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    document_tokens = approximate_token_count(document)
    num_ctx = int(config["num_ctx"])
    num_predict = int(config["num_predict"])
    prompt_margin = int(config.get("safety", {}).get("prompt_token_margin", 64))
    full_context_limit = int(config.get("adaptive_full_context_max_document_tokens", 700))
    prompt_budget = int(config.get("adaptive_prompt_token_budget", max(256, num_ctx - num_predict - prompt_margin - 180)))
    confidence_threshold = float(config.get("adaptive_retrieval_confidence_threshold", 0.04))

    if document_tokens <= full_context_limit and document_tokens <= prompt_budget:
        return document.strip(), "full_context", {
            "adaptive_selected_method": "full_context",
            "adaptive_reason": "document_within_small_document_budget",
            "adaptive_prompt_budget": prompt_budget,
        }

    if is_recent_context_question(question, question_category):
        return make_sliding_window(document, int(config["sliding_window_tokens"])), "sliding_window", {
            "adaptive_selected_method": "sliding_window",
            "adaptive_reason": "question_category_or_terms_indicate_recent_context",
            "adaptive_prompt_budget": prompt_budget,
        }

    retrieved, retrieval_meta = tfidf_retrieval(
        document,
        question,
        int(config["retrieval_chunk_tokens"]),
        int(config["retrieval_top_k"]),
    )
    confidence = float(retrieval_meta.get("retrieval_confidence", 0.0) or 0.0)
    if confidence >= confidence_threshold:
        return retrieved, "retrieval_memory_tfidf", {
            "adaptive_selected_method": "retrieval_memory_tfidf",
            "adaptive_reason": "retrieval_confidence_above_threshold",
            "adaptive_prompt_budget": prompt_budget,
            "adaptive_confidence_threshold": confidence_threshold,
        } | retrieval_meta

    hybrid_context, hybrid_meta = make_retrieval_plus_summary_context(document, question, config)
    if approximate_token_count(hybrid_context) <= prompt_budget:
        return hybrid_context, "retrieval_plus_summary", {
            "adaptive_selected_method": "retrieval_plus_summary",
            "adaptive_reason": "retrieval_confidence_below_threshold_but_hybrid_fits_budget",
            "adaptive_prompt_budget": prompt_budget,
            "adaptive_confidence_threshold": confidence_threshold,
        } | hybrid_meta

    summary = make_extractive_summary(document, question, int(config["summary_tokens"]))
    return summary, "summary_memory_extractive", {
        "adaptive_selected_method": "summary_memory_extractive",
        "adaptive_reason": "retrieval_confidence_low_and_hybrid_exceeds_budget",
        "adaptive_prompt_budget": prompt_budget,
        "adaptive_confidence_threshold": confidence_threshold,
        "summary_backend": "extractive",
    }


def build_context(
    method: str,
    document: str,
    question: str,
    config: dict[str, Any],
    allow_model_calls: bool = True,
    question_category: str = "",
) -> ContextBuild:
    method = METHOD_ALIASES.get(method, method)
    start = time.perf_counter()
    metadata: dict[str, Any] = {"canonical_method": method}

    if method == "full_context":
        result = ContextBuild(
            context_label="Document",
            context=document.strip(),
            instruction="Read the full document below and answer the question.",
            build_seconds=0.0,
            metadata=metadata,
        )
    elif method == "sliding_window":
        result = ContextBuild(
            context_label="Recent context",
            context=make_sliding_window(document, int(config["sliding_window_tokens"])),
            instruction="Use only the recent context below and answer the question.",
            build_seconds=0.0,
            metadata=metadata,
        )
    elif method == "summary_memory_extractive":
        result = ContextBuild(
            context_label="Extractive summary",
            context=make_extractive_summary(document, question, int(config["summary_tokens"])),
            instruction="Use only the compressed extractive summary below and answer the question.",
            build_seconds=0.0,
            metadata=metadata | {"summary_backend": "extractive"},
        )
    elif method == "summary_memory_llm":
        if allow_model_calls:
            summary = build_llm_summary(document, question, config)
            metadata_extra = {"summary_backend": "llm", "summary_model": config["summary_model"]}
        else:
            summary = make_extractive_summary(document, question, int(config["summary_tokens"]))
            metadata_extra = {
                "summary_backend": "llm",
                "summary_model": config["summary_model"],
                "dry_run_fallback": "extractive_summary_preview",
            }
        result = ContextBuild(
            context_label="LLM summary",
            context=trim_to_token_budget(summary, int(config["summary_tokens"])),
            instruction="Use only the compressed LLM summary below and answer the question.",
            build_seconds=0.0,
            metadata=metadata | metadata_extra,
        )
    elif method == "retrieval_memory_keyword":
        context, retrieval_meta = keyword_retrieval(
            document,
            question,
            int(config["retrieval_chunk_tokens"]),
            int(config["retrieval_top_k"]),
        )
        result = ContextBuild(
            context_label="Retrieved chunks",
            context=context,
            instruction="Use only the retrieved relevant chunks below and answer the question.",
            build_seconds=0.0,
            metadata=metadata | retrieval_meta,
        )
    elif method == "retrieval_memory_tfidf":
        context, retrieval_meta = tfidf_retrieval(
            document,
            question,
            int(config["retrieval_chunk_tokens"]),
            int(config["retrieval_top_k"]),
        )
        result = ContextBuild(
            context_label="Retrieved chunks",
            context=context,
            instruction="Use only the TF-IDF retrieved chunks below and answer the question.",
            build_seconds=0.0,
            metadata=metadata | retrieval_meta,
        )
    elif method == "retrieval_memory_embedding":
        embedding_model = str(config.get("embedding_model", "nomic-embed-text:latest"))
        if allow_model_calls:
            context, retrieval_meta = embedding_retrieval(
                document,
                question,
                int(config["retrieval_chunk_tokens"]),
                int(config["retrieval_top_k"]),
                str(config["ollama_host"]),
                embedding_model,
            )
        else:
            context, retrieval_meta = tfidf_retrieval(
                document,
                question,
                int(config["retrieval_chunk_tokens"]),
                int(config["retrieval_top_k"]),
            )
            retrieval_meta = retrieval_meta | {
                "retrieval_backend": "embedding",
                "embedding_model": embedding_model,
                "dry_run_fallback": "tfidf_retrieval_preview",
            }
        result = ContextBuild(
            context_label="Retrieved chunks",
            context=context,
            instruction="Use only the embedding-retrieved chunks below and answer the question.",
            build_seconds=0.0,
            metadata=metadata | retrieval_meta,
        )
    elif method == "retrieval_plus_summary":
        context, hybrid_meta = make_retrieval_plus_summary_context(document, question, config)
        result = ContextBuild(
            context_label="Hybrid context",
            context=context,
            instruction="Use the global summary for orientation and the retrieved chunks for detailed evidence.",
            build_seconds=0.0,
            metadata=metadata | hybrid_meta | {"hybrid_strategy": "extractive_summary_plus_tfidf_retrieval"},
        )
    elif method == "adaptive_context":
        context, selected_method, adaptive_meta = choose_adaptive_context(document, question, question_category, config)
        if selected_method == "full_context":
            context_label = "Document"
            instruction = "Read the full document below and answer the question."
        elif selected_method == "sliding_window":
            context_label = "Recent context"
            instruction = "Use only the recent context below and answer the question."
        elif selected_method == "summary_memory_extractive":
            context_label = "Extractive summary"
            instruction = "Use only the compressed extractive summary below and answer the question."
        elif selected_method == "retrieval_plus_summary":
            context_label = "Hybrid context"
            instruction = "Use the global summary for orientation and the retrieved chunks for detailed evidence."
        else:
            context_label = "Retrieved chunks"
            instruction = "Use only the selected retrieved chunks below and answer the question."
        result = ContextBuild(
            context_label=context_label,
            context=context,
            instruction=instruction,
            build_seconds=0.0,
            metadata=metadata | adaptive_meta,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result.build_seconds = time.perf_counter() - start
    return result


def build_llm_summary(document: str, question: str, config: dict[str, Any]) -> str:
    summary_prompt = (
        "Summarize the document for answering the question below.\n"
        "Keep only facts grounded in the document. Do not add outside information.\n"
        f"Maximum approximate summary tokens: {int(config['summary_tokens'])}.\n\n"
        f"Question:\n{question}\n\n"
        f"Document:\n{trim_to_token_budget(document, int(config.get('summary_source_tokens', 1600)))}"
    )
    response = call_ollama_generate(
        host=str(config["ollama_host"]),
        model=str(config["summary_model"]),
        prompt=summary_prompt,
        num_ctx=int(config.get("summary_num_ctx", 2048)),
        num_predict=int(config.get("summary_num_predict", 220)),
        temperature=float(config.get("summary_temperature", 0.0)),
    )
    return str(response.get("response", "")).strip()


def build_prompt(method: str, context_build: ContextBuild, question: str) -> str:
    return (
        "You are part of an experiment on memory-efficient long-context inference.\n\n"
        f"Context compression method: {method.upper()}\n\n"
        f"{context_build.instruction}\n\n"
        f"{context_build.context_label}:\n{context_build.context}\n\n"
        f"Question:\n{question}\n\n"
        "Give a concise answer using only the provided context."
    )


def call_ollama_generate(
    host: str,
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    return post_json(f"{host.rstrip('/')}/api/generate", payload, timeout=900)


def call_ollama_embedding(host: str, model: str, text: str) -> list[float]:
    cache_key = hashlib.sha256(f"{host}\n{model}\n{text}".encode("utf-8")).hexdigest()
    cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        cached = load_json(cache_path)
        embedding = cached.get("embedding") if isinstance(cached, dict) else None
        if isinstance(embedding, list):
            return [float(item) for item in embedding]

    payload = {"model": model, "input": text}
    try:
        response = post_json(f"{host.rstrip('/')}/api/embed", payload, timeout=300)
    except urllib.error.HTTPError:
        legacy = post_json(f"{host.rstrip('/')}/api/embeddings", {"model": model, "prompt": text}, timeout=300)
        embedding = legacy.get("embedding")
        if isinstance(embedding, list):
            vector = [float(item) for item in embedding]
            write_embedding_cache(cache_path, model, vector)
            return vector
        raise

    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        vector = [float(item) for item in embeddings[0]]
        write_embedding_cache(cache_path, model, vector)
        return vector
    embedding = response.get("embedding")
    if isinstance(embedding, list):
        vector = [float(item) for item in embedding]
        write_embedding_cache(cache_path, model, vector)
        return vector
    raise RuntimeError(f"Ollama embedding response from {model} did not contain an embedding vector.")


def write_embedding_cache(cache_path: Path, model: str, embedding: list[float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"model": model, "embedding": embedding}), encoding="utf-8")


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def query_vram_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def query_total_vram_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def query_ram_percent() -> float | None:
    if os.name != "nt":
        return None
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$os=Get-CimInstance Win32_OperatingSystem; "
                "[math]::Round((($os.TotalVisibleMemorySize-$os.FreePhysicalMemory)/$os.TotalVisibleMemorySize)*100,2)",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def ns_to_seconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def seconds_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_documents(config: dict[str, Any]) -> dict[str, DocumentSpec]:
    document_entries = config.get("documents")
    if not document_entries:
        document_entries = [
            {
                "id": Path(config["document_path"]).stem,
                "path": config["document_path"],
            }
        ]

    documents: dict[str, DocumentSpec] = {}
    for entry in document_entries:
        document_id = str(entry["id"])
        path = resolve_path(str(entry["path"]))
        documents[document_id] = DocumentSpec(document_id=document_id, path=path, text=path.read_text(encoding="utf-8"))
    return documents


def load_questions(config: dict[str, Any]) -> list[QuestionSpec]:
    if "questions_path" in config:
        raw_questions = load_json(resolve_path(str(config["questions_path"])))
    else:
        document_id = Path(str(config["document_path"])).stem
        raw_questions = [
            {
                "id": f"q{index}",
                "document_id": document_id,
                "category": "legacy",
                "question": question,
                "expected_points": [],
            }
            for index, question in enumerate(config["questions"], start=1)
        ]

    questions: list[QuestionSpec] = []
    for item in raw_questions:
        questions.append(
            QuestionSpec(
                question_id=str(item["id"]),
                document_id=str(item["document_id"]),
                category=str(item.get("category", "")),
                question=str(item["question"]),
                expected_points=list(item.get("expected_points", [])),
            )
        )
    return questions


def filter_questions(
    questions: list[QuestionSpec],
    max_questions: int | None,
    document_ids: set[str] | None,
    categories: set[str] | None,
) -> list[QuestionSpec]:
    filtered = questions
    if document_ids:
        filtered = [question for question in filtered if question.document_id in document_ids]
    if categories:
        filtered = [question for question in filtered if question.category in categories]
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("--max-questions must be at least 1.")
        filtered = filtered[:max_questions]
    return filtered


def model_safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def check_safety(
    config: dict[str, Any],
    model: str,
    num_ctx: int,
    max_prompt_tokens: int,
    unsafe: bool,
) -> list[str]:
    safety = config.get("safety", {})
    if unsafe or not safety.get("enabled", True):
        return []

    errors: list[str] = []
    max_ctx_by_model = safety.get("max_num_ctx_by_model", {})
    max_ctx = max_ctx_by_model.get(model)
    if max_ctx is not None and num_ctx > int(max_ctx):
        errors.append(f"{model} num_ctx={num_ctx} exceeds safe limit {max_ctx}. Use --unsafe to override.")

    model_size_gb = safety.get("model_size_gb", {}).get(model)
    total_vram_mib = query_total_vram_mib() or safety.get("gpu_vram_limit_mib")
    if model_size_gb is not None and total_vram_mib is not None:
        model_size_mib = float(model_size_gb) * 1024
        if model_size_mib > float(total_vram_mib):
            errors.append(
                f"{model} size is about {model_size_gb} GB, larger than detected VRAM "
                f"{int(total_vram_mib)} MiB. Use --unsafe or lower num_ctx for an intentional stress test."
            )

    prompt_margin = int(safety.get("prompt_token_margin", 64))
    if max_prompt_tokens + prompt_margin > num_ctx:
        errors.append(
            f"approx_prompt_tokens={max_prompt_tokens} is too close to num_ctx={num_ctx}; "
            f"keep at least {prompt_margin} tokens for safety or raise num_ctx."
        )
    return errors


def should_abort_after_run(config: dict[str, Any], peak_vram_mib: int | None, unsafe: bool) -> bool:
    if unsafe or peak_vram_mib is None:
        return False
    safety = config.get("safety", {})
    abort_at = safety.get("abort_if_peak_vram_mib_gte")
    return abort_at is not None and peak_vram_mib >= int(abort_at)


def build_result_row(
    *,
    model: str,
    method: str,
    canonical_method: str,
    document: DocumentSpec,
    question: QuestionSpec,
    run_number: int,
    phase: str,
    prompt: str,
    context_build: ContextBuild,
    response: dict[str, Any],
    elapsed_seconds: float,
    monitor_stats: MonitorStats,
    config: dict[str, Any],
) -> dict[str, Any]:
    eval_count = response.get("eval_count")
    eval_duration_seconds = ns_to_seconds(response.get("eval_duration"))
    prompt_eval_duration_seconds = ns_to_seconds(response.get("prompt_eval_duration"))
    total_duration_seconds = ns_to_seconds(response.get("total_duration"))
    tokens_per_second = None
    if eval_count and eval_duration_seconds and eval_duration_seconds > 0:
        tokens_per_second = float(eval_count) / eval_duration_seconds

    return {
        "model": model,
        "method": method,
        "canonical_method": canonical_method,
        "document_id": document.document_id,
        "document_path": str(document.path),
        "document_approx_tokens": approximate_token_count(document.text),
        "question_id": question.question_id,
        "question_category": question.category,
        "question": question.question,
        "phase": phase,
        "run_number": run_number,
        "included_in_summary": "yes" if phase == "measured" else "no",
        "num_ctx": config["num_ctx"],
        "num_predict": config["num_predict"],
        "temperature": config["temperature"],
        "approx_prompt_tokens": approximate_token_count(prompt),
        "approx_context_tokens": approximate_token_count(context_build.context),
        "ollama_prompt_tokens": response.get("prompt_eval_count", ""),
        "generated_tokens": eval_count or "",
        "context_build_seconds": seconds_text(context_build.build_seconds),
        "prompt_eval_seconds": seconds_text(prompt_eval_duration_seconds),
        "generation_seconds": seconds_text(eval_duration_seconds),
        "ollama_total_seconds": seconds_text(total_duration_seconds),
        "wall_seconds": f"{elapsed_seconds:.3f}",
        "tokens_per_second": f"{tokens_per_second:.3f}" if tokens_per_second is not None else "",
        "peak_vram_mib": monitor_stats.peak_vram_mib or "",
        "peak_ram_percent": f"{monitor_stats.peak_ram_percent:.2f}" if monitor_stats.peak_ram_percent is not None else "",
        "retrieval_backend": context_build.metadata.get("retrieval_backend", ""),
        "retrieval_top_score": context_build.metadata.get("retrieval_top_score", ""),
        "retrieval_second_score": context_build.metadata.get("retrieval_second_score", ""),
        "retrieval_score_gap": context_build.metadata.get("retrieval_score_gap", ""),
        "retrieval_confidence": context_build.metadata.get("retrieval_confidence", ""),
        "selected_chunk_ids": context_build.metadata.get("selected_chunk_ids", ""),
        "selected_chunk_scores": context_build.metadata.get("selected_chunk_scores", ""),
        "embedding_model": context_build.metadata.get("embedding_model", ""),
        "summary_backend": context_build.metadata.get("summary_backend", ""),
        "summary_model": context_build.metadata.get("summary_model", ""),
        "hybrid_strategy": context_build.metadata.get("hybrid_strategy", ""),
        "adaptive_selected_method": context_build.metadata.get("adaptive_selected_method", ""),
        "adaptive_reason": context_build.metadata.get("adaptive_reason", ""),
        "adaptive_prompt_budget": context_build.metadata.get("adaptive_prompt_budget", ""),
        "adaptive_confidence_threshold": context_build.metadata.get("adaptive_confidence_threshold", ""),
        "chunk_count": context_build.metadata.get("chunk_count", ""),
        "prompt_sha256_16": hash_prompt(prompt),
        "answer": response.get("response", "").strip(),
    }


def build_dry_run_row(
    *,
    model: str,
    method: str,
    canonical_method: str,
    document: DocumentSpec,
    question: QuestionSpec,
    run_number: int,
    phase: str,
    prompt: str,
    context_build: ContextBuild,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "method": method,
        "canonical_method": canonical_method,
        "document_id": document.document_id,
        "document_path": str(document.path),
        "document_approx_tokens": approximate_token_count(document.text),
        "question_id": question.question_id,
        "question_category": question.category,
        "question": question.question,
        "phase": phase,
        "run_number": run_number,
        "included_in_summary": "yes" if phase == "measured" else "no",
        "num_ctx": config["num_ctx"],
        "num_predict": config["num_predict"],
        "temperature": config["temperature"],
        "approx_prompt_tokens": approximate_token_count(prompt),
        "approx_context_tokens": approximate_token_count(context_build.context),
        "ollama_prompt_tokens": "",
        "generated_tokens": "",
        "context_build_seconds": f"{context_build.build_seconds:.3f}",
        "prompt_eval_seconds": "",
        "generation_seconds": "",
        "ollama_total_seconds": "",
        "wall_seconds": "",
        "tokens_per_second": "",
        "peak_vram_mib": "",
        "peak_ram_percent": "",
        "retrieval_backend": context_build.metadata.get("retrieval_backend", ""),
        "retrieval_top_score": context_build.metadata.get("retrieval_top_score", ""),
        "retrieval_second_score": context_build.metadata.get("retrieval_second_score", ""),
        "retrieval_score_gap": context_build.metadata.get("retrieval_score_gap", ""),
        "retrieval_confidence": context_build.metadata.get("retrieval_confidence", ""),
        "selected_chunk_ids": context_build.metadata.get("selected_chunk_ids", ""),
        "selected_chunk_scores": context_build.metadata.get("selected_chunk_scores", ""),
        "embedding_model": context_build.metadata.get("embedding_model", ""),
        "summary_backend": context_build.metadata.get("summary_backend", ""),
        "summary_model": context_build.metadata.get("summary_model", ""),
        "hybrid_strategy": context_build.metadata.get("hybrid_strategy", ""),
        "adaptive_selected_method": context_build.metadata.get("adaptive_selected_method", ""),
        "adaptive_reason": context_build.metadata.get("adaptive_reason", ""),
        "adaptive_prompt_budget": context_build.metadata.get("adaptive_prompt_budget", ""),
        "adaptive_confidence_threshold": context_build.metadata.get("adaptive_confidence_threshold", ""),
        "chunk_count": context_build.metadata.get("chunk_count", ""),
        "prompt_sha256_16": hash_prompt(prompt),
        "answer": "DRY RUN",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context memory experiments with Ollama.")
    parser.add_argument("--config", default="config/safe_6gb_config.json", help="Path to config JSON.")
    parser.add_argument("--models", nargs="+", help="Override model list.")
    parser.add_argument("--methods", nargs="+", help="Override methods.")
    parser.add_argument("--documents", nargs="+", help="Only run these document ids.")
    parser.add_argument("--categories", nargs="+", help="Only run these question categories.")
    parser.add_argument("--num-ctx", type=int, help="Override Ollama num_ctx.")
    parser.add_argument("--num-predict", type=int, help="Override Ollama num_predict.")
    parser.add_argument("--repeat-runs", type=int, help="Override measured repeat runs.")
    parser.add_argument("--warmup-runs", type=int, help="Override warmup runs.")
    parser.add_argument("--max-questions", type=int, help="Run only the first N filtered questions.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--clean-output", action="store_true", help="Delete the selected output directory before running. Only allowed under project results/.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and rows without calling Ollama.")
    parser.add_argument("--unsafe", action="store_true", help="Bypass configured 6 GB safety checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_json(config_path)

    if args.models:
        config["models"] = args.models
    if args.methods:
        config["methods"] = args.methods
    if args.num_ctx:
        config["num_ctx"] = args.num_ctx
    if args.num_predict:
        config["num_predict"] = args.num_predict
    if args.repeat_runs is not None:
        config["repeat_runs"] = args.repeat_runs
    if args.warmup_runs is not None:
        config["warmup_runs"] = args.warmup_runs
    if args.output_dir:
        config["output_dir"] = args.output_dir

    documents = load_documents(config)
    if args.documents:
        requested_documents = set(args.documents)
        missing_requested = sorted(requested_documents - set(documents.keys()))
        if missing_requested:
            print(f"Requested document ids are not configured: {', '.join(missing_requested)}")
            return 1
        document_filter = requested_documents
    else:
        document_filter = set(documents.keys())
    category_filter = set(args.categories) if args.categories else None
    questions = filter_questions(load_questions(config), args.max_questions, document_filter, category_filter)

    if not questions:
        print("No questions matched the selected filters.")
        return 1

    missing_documents = sorted({question.document_id for question in questions if question.document_id not in documents})
    if missing_documents:
        print(f"Questions reference missing document ids: {', '.join(missing_documents)}")
        return 1

    output_dir = resolve_path(config["output_dir"])
    if args.clean_output:
        try:
            clean_output_dir(output_dir)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
    prompts_dir = output_dir / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    repeat_runs = int(config.get("repeat_runs", 1))
    warmup_runs = int(config.get("warmup_runs", 0))
    if repeat_runs < 1:
        print("repeat_runs must be at least 1.")
        return 1
    if warmup_runs < 0:
        print("warmup_runs cannot be negative.")
        return 1

    rows: list[dict[str, Any]] = []
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Methods: {', '.join(config['methods'])}")
    print(f"Documents: {', '.join(sorted({question.document_id for question in questions}))}")
    print(f"Questions: {len(questions)}")
    print(f"num_ctx={config['num_ctx']}, num_predict={config['num_predict']}")
    print(f"warmup_runs={warmup_runs}, repeat_runs={repeat_runs}")

    for model in config["models"]:
        for method in config["methods"]:
            canonical_method = METHOD_ALIASES.get(method, method)
            for question in questions:
                document = documents[question.document_id]
                try:
                    context_build = build_context(
                        canonical_method,
                        document.text,
                        question.question,
                        config,
                        allow_model_calls=not args.dry_run,
                        question_category=question.category,
                    )
                except urllib.error.HTTPError as exc:
                    print(f"ERROR: context build failed for {method}. HTTP {exc.code}: {exc.reason}")
                    return 1
                except urllib.error.URLError as exc:
                    print(f"ERROR: context build failed for {method}: {exc}")
                    return 1
                except RuntimeError as exc:
                    print(f"ERROR: context build failed for {method}: {exc}")
                    return 1

                prompt = build_prompt(canonical_method, context_build, question.question)
                prompt_file = prompts_dir / (
                    f"{question.question_id}_{document.document_id}_{model_safe_name(model)}_{canonical_method}.txt"
                )
                prompt_file.write_text(prompt, encoding="utf-8")
                approx_tokens = approximate_token_count(prompt)

                safety_errors = check_safety(
                    config=config,
                    model=model,
                    num_ctx=int(config["num_ctx"]),
                    max_prompt_tokens=approx_tokens,
                    unsafe=args.unsafe,
                )
                if safety_errors:
                    print("Safety check stopped this run:")
                    for error in safety_errors:
                        print(f"  - {error}")
                    print("Lower --num-ctx, reduce context, or rerun with --unsafe only for an intentional stress test.")
                    return 1

                schedule = [("warmup", run) for run in range(1, warmup_runs + 1)]
                schedule += [("measured", run) for run in range(1, repeat_runs + 1)]
                for phase, run_number in schedule:
                    print(
                        f"\nRunning model={model}, method={canonical_method}, "
                        f"doc={document.document_id}, question={question.question_id}, "
                        f"phase={phase}, run={run_number}, approx_prompt_tokens={approx_tokens}"
                    )

                    if args.dry_run:
                        rows.append(
                            build_dry_run_row(
                                model=model,
                                method=method,
                                canonical_method=canonical_method,
                                document=document,
                                question=question,
                                run_number=run_number,
                                phase=phase,
                                prompt=prompt,
                                context_build=context_build,
                                config=config,
                            )
                        )
                        continue

                    try:
                        start = time.perf_counter()
                        with ResourceMonitor() as monitor:
                            response = call_ollama_generate(
                                host=str(config["ollama_host"]),
                                model=model,
                                prompt=prompt,
                                num_ctx=int(config["num_ctx"]),
                                num_predict=int(config["num_predict"]),
                                temperature=float(config["temperature"]),
                            )
                        elapsed_seconds = time.perf_counter() - start
                    except urllib.error.URLError as exc:
                        print(f"ERROR: Ollama request failed for {model}/{canonical_method}: {exc}")
                        return 1

                    row = build_result_row(
                        model=model,
                        method=method,
                        canonical_method=canonical_method,
                        document=document,
                        question=question,
                        run_number=run_number,
                        phase=phase,
                        prompt=prompt,
                        context_build=context_build,
                        response=response,
                        elapsed_seconds=elapsed_seconds,
                        monitor_stats=monitor.stats,
                        config=config,
                    )
                    rows.append(row)
                    print(
                        "Done: "
                        f"wall={row['wall_seconds']}s, "
                        f"prompt_eval={row['prompt_eval_seconds'] or 'n/a'}s, "
                        f"tok/s={row['tokens_per_second'] or 'n/a'}, "
                        f"peak_vram={row['peak_vram_mib'] or 'n/a'} MiB"
                    )

                    if should_abort_after_run(config, monitor.stats.peak_vram_mib, args.unsafe):
                        print(
                            f"Safety abort: peak VRAM {monitor.stats.peak_vram_mib} MiB reached "
                            f"configured threshold {config['safety']['abort_if_peak_vram_mib_gte']} MiB."
                        )
                        write_csv(output_dir / "results.csv", rows)
                        (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                        return 1

    write_csv(output_dir / "results.csv", rows)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata = {
        "config_path": str(config_path),
        "config": config,
        "generated_at_unix": time.time(),
        "python": sys.version,
        "total_vram_mib": query_total_vram_mib(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'results.json'}")
    print(f"Wrote {output_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
