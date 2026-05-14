from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("List cannot be empty.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate budget sweep configs from a base config.")
    parser.add_argument("--base-config", default="config/safe_6gb_config.json")
    parser.add_argument("--output-dir", default="config/budget_sweeps")
    parser.add_argument("--summary-tokens", default="256,512,768")
    parser.add_argument("--retrieval-top-k", default="1,2,3,4")
    parser.add_argument("--retrieval-chunk-tokens", default="100,180,256")
    parser.add_argument("--methods", nargs="+", default=["retrieval_memory_tfidf", "retrieval_plus_summary", "adaptive_context"])
    parser.add_argument("--max-questions", type=int, default=None, help="Optional note stored in config for manual use.")
    args = parser.parse_args()

    base = load_json(resolve_path(args.base_config))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for summary_tokens in parse_int_list(args.summary_tokens):
        for top_k in parse_int_list(args.retrieval_top_k):
            for chunk_tokens in parse_int_list(args.retrieval_chunk_tokens):
                config = deepcopy(base)
                config["methods"] = args.methods
                config["summary_tokens"] = summary_tokens
                config["retrieval_top_k"] = top_k
                config["retrieval_chunk_tokens"] = chunk_tokens
                config["hybrid_summary_tokens"] = max(96, summary_tokens // 2)
                config["hybrid_retrieval_top_k"] = max(1, min(top_k, 2))
                config["hybrid_retrieval_chunk_tokens"] = chunk_tokens
                config["adaptive_prompt_token_budget"] = int(config.get("num_ctx", 2048)) - int(config.get("num_predict", 160)) - 240
                name = f"s{summary_tokens}_k{top_k}_c{chunk_tokens}"
                config["output_dir"] = f"results/budget_sweeps/{name}"
                if args.max_questions is not None:
                    config["sweep_max_questions_note"] = args.max_questions
                path = output_dir / f"{name}.json"
                path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                generated.append(path)

    run_script = output_dir / "run_budget_sweeps.ps1"
    commands = [
        '$ErrorActionPreference = "Stop"',
        'Set-Location -Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))',
        "",
    ]
    for path in generated:
        relative = path.relative_to(ROOT)
        run_name = path.stem
        extra = f" --max-questions {args.max_questions}" if args.max_questions is not None else ""
        commands.append(
            f"python scripts\\run_experiment.py --config {relative} --output-dir results\\budget_sweeps\\{run_name} --clean-output{extra}"
        )
        commands.append(
            f"python scripts\\analyze_results.py --results-dir results\\budget_sweeps\\{run_name} --reports-dir reports\\budget_sweeps\\{run_name}"
        )
        commands.append("")
    run_script.write_text("\n".join(commands), encoding="utf-8")

    print(f"Generated {len(generated)} config files in {output_dir}")
    print(f"Generated {run_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

