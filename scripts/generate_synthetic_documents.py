from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_REPLACEMENT_PARAGRAPHS = [
    "Machine learning systems learn statistical patterns from data. Local inference keeps data private, but local hardware has strict memory and latency limits.",
    "Transformer models use attention to relate tokens in a context. During inference, the model stores previous key and value tensors in the KV cache.",
    "Quantization reduces model weight size by storing parameters with fewer bits. It helps the model fit, but it does not remove KV-cache growth from long prompts.",
    "Retrieval memory splits a document into chunks and selects only relevant chunks for the question. This can reduce prompt length while preserving exact evidence.",
    "Summary memory compresses older context into shorter text. It preserves broad ideas but can lose exact names, numbers, code lines, or edge-case details.",
    "Sliding window memory keeps only recent context. It is efficient for recent facts but unreliable when the answer appears near the beginning of a long document.",
    "On a 6 GB GPU, model weights, KV cache, temporary buffers, and runtime overhead compete for the same limited memory budget.",
    "A budget-aware context controller chooses the cheapest context strategy likely to answer the question under the current prompt and memory budget.",
]


def approximate_tokens(text: str) -> list[str]:
    return text.replace(".", " .").replace(",", " ,").split()


def build_document(target_tokens: int) -> str:
    paragraphs: list[str] = [
        f"Synthetic scaling document target: about {target_tokens} tokens.",
        "Anchor fact near beginning: the recommended safe baseline is llama3:latest with num_ctx 2048.",
    ]
    index = 0
    while len(approximate_tokens("\n\n".join(paragraphs))) < target_tokens - 80:
        paragraphs.append(TOKEN_REPLACEMENT_PARAGRAPHS[index % len(TOKEN_REPLACEMENT_PARAGRAPHS)])
        index += 1
    paragraphs.append("Anchor fact near end: adaptive context should use retrieval plus summary when retrieval confidence is low.")
    return "\n\n".join(paragraphs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic long documents and matching questions.")
    parser.add_argument("--sizes", default="1000,2000,4000,8000", help="Comma-separated approximate token targets.")
    args = parser.parse_args()

    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    documents_dir = ROOT / "data" / "documents"
    questions_dir = ROOT / "data" / "questions"
    configs_dir = ROOT / "config"
    documents_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    document_entries = []
    questions = []
    for size in sizes:
        document_id = f"synthetic_scale_{size}"
        path = documents_dir / f"{document_id}.txt"
        path.write_text(build_document(size), encoding="utf-8")
        document_entries.append({"id": document_id, "path": f"data/documents/{document_id}.txt"})
        questions.extend(
            [
                {
                    "id": f"q_{document_id}_beginning",
                    "document_id": document_id,
                    "category": "fact_near_beginning",
                    "question": "What safe baseline model and num_ctx are recommended near the beginning of the document?",
                    "expected_points": [
                        {"label": "baseline is llama3:latest", "keywords": ["llama3:latest"]},
                        {"label": "num_ctx is 2048", "keywords": ["num_ctx", "2048"]},
                    ],
                },
                {
                    "id": f"q_{document_id}_end",
                    "document_id": document_id,
                    "category": "fact_near_end",
                    "question": "What should adaptive context do when retrieval confidence is low?",
                    "expected_points": [
                        {"label": "use retrieval plus summary", "aliases": ["retrieval plus summary", "summary", "retrieval"]},
                        {"label": "condition is low retrieval confidence", "keywords": ["retrieval confidence", "low"]},
                    ],
                },
            ]
        )

    questions_path = questions_dir / "synthetic_scale_questions.json"
    questions_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    config = {
        "models": ["llama3:latest"],
        "documents": document_entries,
        "questions_path": "data/questions/synthetic_scale_questions.json",
        "methods": ["full_context", "sliding_window", "retrieval_memory_tfidf", "retrieval_plus_summary", "adaptive_context"],
        "num_ctx": 4096,
        "num_predict": 120,
        "temperature": 0.0,
        "repeat_runs": 2,
        "warmup_runs": 1,
        "sliding_window_tokens": 500,
        "summary_tokens": 600,
        "summary_model": "llama3:latest",
        "summary_source_tokens": 1800,
        "summary_num_ctx": 2048,
        "summary_num_predict": 220,
        "summary_temperature": 0.0,
        "retrieval_chunk_tokens": 220,
        "retrieval_top_k": 3,
        "hybrid_summary_tokens": 260,
        "hybrid_retrieval_top_k": 2,
        "hybrid_retrieval_chunk_tokens": 220,
        "adaptive_full_context_max_document_tokens": 700,
        "adaptive_prompt_token_budget": 3200,
        "adaptive_retrieval_confidence_threshold": 0.04,
        "embedding_model": "nomic-embed-text:latest",
        "ollama_host": "http://127.0.0.1:11434",
        "output_dir": "results/synthetic_scale",
        "safety": {
            "enabled": True,
            "gpu_vram_limit_mib": 6141,
            "abort_if_peak_vram_mib_gte": 5950,
            "prompt_token_margin": 96,
            "max_num_ctx_by_model": {"llama3:latest": 4096},
            "model_size_gb": {"llama3:latest": 4.7},
        },
    }
    config_path = configs_dir / "synthetic_scale_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Generated {len(document_entries)} synthetic documents in {documents_dir}")
    print(f"Generated {questions_path}")
    print(f"Generated {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

