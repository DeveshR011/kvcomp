from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("kv-cache", "kv cache")
    text = text.replace("6gb", "6 gb")
    text = re.sub(r"[^a-z0-9:.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def point_matches(answer_norm: str, point: Any) -> bool:
    if isinstance(point, str):
        return normalize(point) in answer_norm
    keywords = [normalize(str(item)) for item in point.get("keywords", [])]
    aliases = [normalize(str(item)) for item in point.get("aliases", [])]
    keyword_match = bool(keywords) and all(keyword in answer_norm for keyword in keywords)
    alias_match = bool(aliases) and any(alias in answer_norm for alias in aliases)
    return keyword_match or alias_match


def score_answer(answer: str, expected_points: list[Any]) -> tuple[int, int, int, str]:
    if not expected_points:
        return 0, 0, 0, ""
    answer_norm = normalize(answer)
    matched_labels: list[str] = []
    for point in expected_points:
        if point_matches(answer_norm, point):
            if isinstance(point, dict):
                matched_labels.append(str(point.get("label", point)))
            else:
                matched_labels.append(str(point))
    matched = len(matched_labels)
    total = len(expected_points)
    ratio = matched / total if total else 0.0
    if matched == 0:
        score = 0
    elif ratio < 0.5:
        score = 1
    elif ratio < 0.85:
        score = 2
    else:
        score = 3
    return score, matched, total, "; ".join(matched_labels)


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def fmt(value: float | None, places: int = 3) -> str:
    return "" if value is None else f"{value:.{places}f}"


def load_questions(questions_path: Path) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in load_json(questions_path)}


def load_manual_annotations(path: Path | None) -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = read_csv(path)
    manual: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        score = row.get("manual_quality_score", "").strip()
        failure_mode = row.get("failure_mode", "").strip()
        notes = row.get("manual_notes", "").strip()
        if not score and not failure_mode and not notes:
            continue
        key = (
            row.get("model", ""),
            row.get("canonical_method", row.get("method", "")),
            row.get("document_id", ""),
            row.get("question_id", ""),
            row.get("phase", "measured"),
            row.get("run_number", ""),
        )
        manual[key] = {
            "manual_quality_score": score,
            "failure_mode": failure_mode,
            "manual_notes": notes,
        }
    return manual


def score_rows(
    rows: list[dict[str, str]],
    questions_by_id: dict[str, dict[str, Any]],
    manual_annotations: dict[tuple[str, str, str, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        question = questions_by_id.get(row.get("question_id", ""), {})
        auto_score, matched, total, matched_labels = score_answer(row.get("answer", ""), question.get("expected_points", []))
        key = (
            row.get("model", ""),
            row.get("canonical_method", row.get("method", "")),
            row.get("document_id", ""),
            row.get("question_id", ""),
            row.get("phase", ""),
            row.get("run_number", ""),
        )
        manual = manual_annotations.get(key, {})
        manual_score = manual.get("manual_quality_score", "")
        final_score = manual_score if manual_score != "" else str(auto_score)
        enriched = dict(row)
        enriched.update(
            {
                "auto_quality_score": auto_score,
                "matched_expected_points": matched,
                "total_expected_points": total,
                "matched_point_labels": matched_labels,
                "manual_quality_score": manual_score,
                "failure_mode": manual.get("failure_mode", ""),
                "manual_notes": manual.get("manual_notes", ""),
                "final_quality_score": final_score,
            }
        )
        scored.append(enriched)
    return scored


def aggregate(
    rows: list[dict[str, Any]],
    group_fields: list[str],
) -> list[dict[str, Any]]:
    measured = [row for row in rows if row.get("included_in_summary") == "yes"]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        numeric_fields = {
            "wall_seconds": [value for row in group_rows if (value := to_float(str(row.get("wall_seconds", "")))) is not None],
            "prompt_eval_seconds": [value for row in group_rows if (value := to_float(str(row.get("prompt_eval_seconds", "")))) is not None],
            "generation_seconds": [value for row in group_rows if (value := to_float(str(row.get("generation_seconds", "")))) is not None],
            "tokens_per_second": [value for row in group_rows if (value := to_float(str(row.get("tokens_per_second", "")))) is not None],
            "peak_vram_mib": [value for row in group_rows if (value := to_float(str(row.get("peak_vram_mib", "")))) is not None],
            "peak_ram_percent": [value for row in group_rows if (value := to_float(str(row.get("peak_ram_percent", "")))) is not None],
            "approx_prompt_tokens": [value for row in group_rows if (value := to_float(str(row.get("approx_prompt_tokens", "")))) is not None],
            "ollama_prompt_tokens": [value for row in group_rows if (value := to_float(str(row.get("ollama_prompt_tokens", "")))) is not None],
            "auto_quality_score": [value for row in group_rows if (value := to_float(str(row.get("auto_quality_score", "")))) is not None],
            "final_quality_score": [value for row in group_rows if (value := to_float(str(row.get("final_quality_score", "")))) is not None],
            "retrieval_top_score": [value for row in group_rows if (value := to_float(str(row.get("retrieval_top_score", "")))) is not None],
            "retrieval_second_score": [value for row in group_rows if (value := to_float(str(row.get("retrieval_second_score", "")))) is not None],
            "retrieval_score_gap": [value for row in group_rows if (value := to_float(str(row.get("retrieval_score_gap", "")))) is not None],
            "retrieval_confidence": [value for row in group_rows if (value := to_float(str(row.get("retrieval_confidence", "")))) is not None],
        }
        output: dict[str, Any] = {field: key[index] for index, field in enumerate(group_fields)}
        output["measured_runs"] = len(group_rows)
        for field, values in numeric_fields.items():
            output[f"mean_{field}"] = fmt(mean(values))
            output[f"std_{field}"] = fmt(stdev(values))
            output[f"max_{field}"] = fmt(max(values), 3) if values else ""
            output[f"min_{field}"] = fmt(min(values), 3) if values else ""
        aggregates.append(output)
    return aggregates


def preferred_prompt_tokens(row: dict[str, Any]) -> float | None:
    return to_float(str(row.get("mean_ollama_prompt_tokens", ""))) or to_float(str(row.get("mean_approx_prompt_tokens", "")))


def add_efficiency_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        quality = to_float(str(row.get("mean_final_quality_score", "")))
        latency = to_float(str(row.get("mean_wall_seconds", "")))
        vram = to_float(str(row.get("max_peak_vram_mib", ""))) or to_float(str(row.get("mean_peak_vram_mib", "")))
        if quality is not None and latency is not None and latency > 0:
            enriched["quality_latency_efficiency"] = fmt(quality / latency, 6)
        else:
            enriched["quality_latency_efficiency"] = ""
        if quality is not None and vram is not None and vram > 0:
            enriched["quality_memory_efficiency"] = fmt(quality / vram, 8)
        else:
            enriched["quality_memory_efficiency"] = ""
        if quality is not None and latency is not None and vram is not None:
            enriched["tradeoff_score"] = fmt(quality / (1.0 + latency + (vram / 6000.0)), 6)
        else:
            enriched["tradeoff_score"] = ""
        output.append(enriched)
    return output


def add_relative_metrics_against_full_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_question: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("canonical_method") == "full_context":
            key = (str(row.get("model", "")), str(row.get("document_id", "")), str(row.get("question_id", "")))
            baseline_by_question[key] = row

    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        key = (str(row.get("model", "")), str(row.get("document_id", "")), str(row.get("question_id", "")))
        baseline = baseline_by_question.get(key)
        method_tokens = preferred_prompt_tokens(row)
        method_quality = to_float(str(row.get("mean_final_quality_score", "")))
        method_latency = to_float(str(row.get("mean_wall_seconds", "")))
        method_vram = to_float(str(row.get("max_peak_vram_mib", ""))) or to_float(str(row.get("mean_peak_vram_mib", "")))
        if baseline:
            base_tokens = preferred_prompt_tokens(baseline)
            base_quality = to_float(str(baseline.get("mean_final_quality_score", "")))
            base_latency = to_float(str(baseline.get("mean_wall_seconds", "")))
            base_vram = to_float(str(baseline.get("max_peak_vram_mib", ""))) or to_float(str(baseline.get("mean_peak_vram_mib", "")))
            if method_tokens is not None and base_tokens and base_tokens > 0:
                compression_ratio = method_tokens / base_tokens
                enriched["compression_ratio_vs_full"] = fmt(compression_ratio, 6)
                enriched["token_saving_percent_vs_full"] = fmt(100.0 * (1.0 - compression_ratio), 3)
            else:
                enriched["compression_ratio_vs_full"] = ""
                enriched["token_saving_percent_vs_full"] = ""
            if method_quality is not None and base_quality and base_quality > 0:
                enriched["quality_retention_vs_full"] = fmt(method_quality / base_quality, 6)
            else:
                enriched["quality_retention_vs_full"] = ""
            if method_latency is not None and base_latency and base_latency > 0:
                enriched["latency_saving_percent_vs_full"] = fmt(100.0 * (1.0 - (method_latency / base_latency)), 3)
            else:
                enriched["latency_saving_percent_vs_full"] = ""
            if method_vram is not None and base_vram and base_vram > 0:
                enriched["vram_saving_percent_vs_full"] = fmt(100.0 * (1.0 - (method_vram / base_vram)), 3)
            else:
                enriched["vram_saving_percent_vs_full"] = ""
        else:
            enriched["compression_ratio_vs_full"] = ""
            enriched["token_saving_percent_vs_full"] = ""
            enriched["quality_retention_vs_full"] = ""
            enriched["latency_saving_percent_vs_full"] = ""
            enriched["vram_saving_percent_vs_full"] = ""
        output.append(enriched)
    return add_efficiency_columns(output)


def aggregate_derived_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("model", "")), str(row.get("canonical_method", "")))].append(row)
    numeric_fields = [
        "compression_ratio_vs_full",
        "token_saving_percent_vs_full",
        "quality_retention_vs_full",
        "latency_saving_percent_vs_full",
        "vram_saving_percent_vs_full",
        "quality_latency_efficiency",
        "quality_memory_efficiency",
        "tradeoff_score",
    ]
    output: list[dict[str, Any]] = []
    for (model, method), group_rows in sorted(groups.items()):
        row: dict[str, Any] = {"model": model, "canonical_method": method, "question_count": len(group_rows)}
        for field in numeric_fields:
            values = [value for item in group_rows if (value := to_float(str(item.get(field, "")))) is not None]
            row[f"mean_{field}"] = fmt(mean(values), 6)
            row[f"std_{field}"] = fmt(stdev(values), 6)
        output.append(row)
    return output


def summarize_failure_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [row for row in rows if row.get("included_in_summary") == "yes"]
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in measured:
        failure_mode = str(row.get("failure_mode", "")).strip()
        if not failure_mode:
            continue
        key = (str(row.get("model", "")), str(row.get("canonical_method", "")), failure_mode)
        counts[key] += 1
    return [
        {"model": model, "canonical_method": method, "failure_mode": failure_mode, "count": count}
        for (model, method, failure_mode), count in sorted(counts.items())
    ]


def manual_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [row for row in rows if row.get("included_in_summary") == "yes"]
    output: list[dict[str, Any]] = []
    for row in measured:
        output.append(
            {
                "model": row.get("model", ""),
                "canonical_method": row.get("canonical_method", row.get("method", "")),
                "document_id": row.get("document_id", ""),
                "question_id": row.get("question_id", ""),
                "phase": row.get("phase", ""),
                "run_number": row.get("run_number", ""),
            "auto_quality_score": row.get("auto_quality_score", ""),
            "manual_quality_score": "",
            "failure_mode": "",
            "allowed_failure_modes": "none;lost_early_fact;bad_retrieval;summary_omission;hallucination;partial_answer;memory_stress;code_mismatch;other",
            "manual_notes": "",
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            }
        )
    return output


def svg_bar_chart(title: str, rows: list[dict[str, Any]], label_field: str, value_field: str, path: Path) -> None:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = to_float(str(row.get(value_field, "")))
        if value is not None:
            values.append((str(row.get(label_field, "")), value))
    if not values:
        return
    width = 980
    height = 120 + (len(values) * 34)
    left = 230
    right = 40
    bar_width = width - left - right
    max_value = max(value for _, value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{escape_xml(title)}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = 68 + (index * 34)
        length = (value / max_value) * bar_width
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial" font-size="13">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{length:.1f}" height="22" fill="#2563eb"/>')
        parts.append(f'<text x="{left + length + 8:.1f}" y="{y + 16}" font-family="Arial" font-size="13">{value:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_scatter_chart(title: str, rows: list[dict[str, Any]], x_field: str, y_field: str, label_field: str, path: Path) -> None:
    points: list[tuple[str, float, float]] = []
    for row in rows:
        x_value = to_float(str(row.get(x_field, "")))
        y_value = to_float(str(row.get(y_field, "")))
        if x_value is not None and y_value is not None:
            points.append((str(row.get(label_field, "")), x_value, y_value))
    if not points:
        return
    width = 900
    height = 520
    left = 72
    top = 62
    chart_width = 720
    chart_height = 360
    max_x = max(point[1] for point in points) or 1.0
    max_y = max(point[2] for point in points) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{escape_xml(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#111827"/>',
        f'<text x="{left + chart_width - 120}" y="{top + chart_height + 42}" font-family="Arial" font-size="13">{escape_xml(x_field)}</text>',
        f'<text x="20" y="{top + 14}" font-family="Arial" font-size="13">{escape_xml(y_field)}</text>',
    ]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for index, (label, x_value, y_value) in enumerate(points):
        x = left + (x_value / max_x) * chart_width
        y = top + chart_height - ((y_value / max_y) * chart_height)
        color = colors[index % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" font-family="Arial" font-size="12">{escape_xml(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def make_method_label(row: dict[str, Any]) -> str:
    return f"{row.get('model', '')} / {row.get('canonical_method', '')}"


def add_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copy = dict(row)
        copy["model_method"] = make_method_label(copy)
        output.append(copy)
    return output


def write_charts(
    charts_dir: Path,
    aggregate_by_method: list[dict[str, Any]],
    derived_by_method: list[dict[str, Any]],
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    labeled = add_labels(aggregate_by_method)
    svg_bar_chart("Average Latency by Method", labeled, "model_method", "mean_wall_seconds", charts_dir / "latency_by_method.svg")
    svg_bar_chart("Average Prompt Tokens by Method", labeled, "model_method", "mean_ollama_prompt_tokens", charts_dir / "prompt_tokens_by_method.svg")
    svg_bar_chart("Average Tokens Per Second by Method", labeled, "model_method", "mean_tokens_per_second", charts_dir / "tokens_per_second_by_method.svg")
    svg_bar_chart("Peak VRAM by Method", labeled, "model_method", "max_peak_vram_mib", charts_dir / "peak_vram_by_method.svg")
    svg_scatter_chart(
        "Quality vs Latency",
        labeled,
        "mean_wall_seconds",
        "mean_final_quality_score",
        "model_method",
        charts_dir / "quality_vs_latency.svg",
    )
    derived_labeled = add_labels(derived_by_method)
    svg_bar_chart(
        "Mean Token Saving vs Full Context",
        derived_labeled,
        "model_method",
        "mean_token_saving_percent_vs_full",
        charts_dir / "token_saving_vs_full_by_method.svg",
    )
    svg_bar_chart(
        "Mean Tradeoff Score",
        derived_labeled,
        "model_method",
        "mean_tradeoff_score",
        charts_dir / "tradeoff_score_by_method.svg",
    )


def query_gpu() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
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
        return "nvidia-smi unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "nvidia-smi query failed"


def write_report(
    path: Path,
    results_dir: Path,
    aggregate_by_method: list[dict[str, Any]],
    aggregate_by_question: list[dict[str, Any]],
    derived_by_method: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> None:
    measured_rows = sum(int(row.get("measured_runs", 0)) for row in aggregate_by_question)
    best_latency = sorted(
        aggregate_by_method,
        key=lambda row: to_float(str(row.get("mean_wall_seconds", ""))) if to_float(str(row.get("mean_wall_seconds", ""))) is not None else float("inf"),
    )[:5]
    best_quality = sorted(
        aggregate_by_method,
        key=lambda row: to_float(str(row.get("mean_final_quality_score", ""))) if to_float(str(row.get("mean_final_quality_score", ""))) is not None else -1,
        reverse=True,
    )[:5]
    tradeoff = sorted(
        aggregate_by_method,
        key=tradeoff_key,
        reverse=True,
    )[:5]
    token_saving = sorted(
        derived_by_method,
        key=lambda row: to_float(str(row.get("mean_token_saving_percent_vs_full", ""))) if to_float(str(row.get("mean_token_saving_percent_vs_full", ""))) is not None else -999,
        reverse=True,
    )[:5]

    lines = [
        "# Experiment Summary",
        "",
        "## Hardware",
        "",
        f"- GPU: `{query_gpu()}`",
    ]
    if metadata:
        lines.append(f"- Config: `{metadata.get('config_path', '')}`")
        lines.append(f"- Recorded total VRAM MiB: `{metadata.get('total_vram_mib', '')}`")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Results directory: `{results_dir}`",
            "- `scored_results.csv` contains per-answer automatic quality helper scores.",
            "- `aggregate_by_method.csv` contains method-level averages over measured runs.",
            "- `aggregate_by_question.csv` contains question-level averages over measured runs.",
            "- `derived_by_method.csv` contains relative savings and efficiency metrics.",
            "- `failure_modes.csv` summarizes manual failure-mode labels when provided.",
            "- `manual_quality_template.csv` is for human scoring.",
            "",
            "## Run Count",
            "",
            f"- Measured rows used in aggregates: `{measured_rows}`",
            "",
            "## Fastest Methods",
            "",
        ]
    )
    lines.extend(markdown_table(best_latency, ["model", "canonical_method", "mean_wall_seconds", "mean_tokens_per_second", "max_peak_vram_mib"]))
    lines.extend(["", "## Highest Automatic Quality", ""])
    lines.extend(markdown_table(best_quality, ["model", "canonical_method", "mean_final_quality_score", "mean_wall_seconds", "mean_ollama_prompt_tokens"]))
    lines.extend(["", "## Best Quality-Latency Tradeoff", ""])
    lines.extend(markdown_table(tradeoff, ["model", "canonical_method", "mean_final_quality_score", "mean_wall_seconds", "max_peak_vram_mib"]))
    lines.extend(["", "## Highest Token Saving vs Full Context", ""])
    lines.extend(markdown_table(token_saving, ["model", "canonical_method", "mean_token_saving_percent_vs_full", "mean_quality_retention_vs_full", "mean_tradeoff_score"]))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Warmup runs are stored in `results.csv` but excluded from aggregate tables.",
            "- `ollama_prompt_tokens` should be preferred over approximate token counts when it is available.",
            "- Automatic quality scores are helper scores based on expected point matching. Use `manual_quality_template.csv` for thesis-grade human scoring.",
            "- If a heavier model runs slower, state whether it may have offloaded work to CPU/RAM under the 6 GB VRAM constraint.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def tradeoff_key(row: dict[str, Any]) -> float:
    quality = to_float(str(row.get("mean_final_quality_score", ""))) or 0.0
    latency = to_float(str(row.get("mean_wall_seconds", ""))) or 999.0
    vram = to_float(str(row.get("max_peak_vram_mib", ""))) or 9999.0
    return quality / (1.0 + latency + (vram / 6000.0))


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["No data."]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score, aggregate, chart, and report experiment results.")
    parser.add_argument("--results-dir", default="results/latest", help="Directory containing results.csv.")
    parser.add_argument("--questions", default="data/questions/thesis_questions.json", help="Question JSON with expected points.")
    parser.add_argument("--manual-scores", help="Optional CSV with manual_quality_score values.")
    parser.add_argument("--reports-dir", default="reports/latest", help="Output report directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    reports_dir = resolve_path(args.reports_dir)
    charts_dir = reports_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.csv"
    if not results_path.exists():
        print(f"Missing results file: {results_path}")
        return 1

    questions = load_questions(resolve_path(args.questions))
    manual_annotations = load_manual_annotations(resolve_path(args.manual_scores) if args.manual_scores else None)
    rows = read_csv(results_path)
    scored = score_rows(rows, questions, manual_annotations)
    aggregate_by_method = add_efficiency_columns(aggregate(scored, ["model", "canonical_method"]))
    aggregate_by_question = add_relative_metrics_against_full_context(
        aggregate(scored, ["model", "canonical_method", "document_id", "question_id", "question_category"])
    )
    derived_by_method = aggregate_derived_by_method(aggregate_by_question)
    failure_modes = summarize_failure_modes(scored)

    write_csv(reports_dir / "scored_results.csv", scored)
    write_csv(reports_dir / "aggregate_by_method.csv", aggregate_by_method)
    write_csv(reports_dir / "aggregate_by_question.csv", aggregate_by_question)
    write_csv(reports_dir / "derived_by_method.csv", derived_by_method)
    if failure_modes:
        write_csv(reports_dir / "failure_modes.csv", failure_modes)
    else:
        (reports_dir / "failure_modes.csv").write_text("model,canonical_method,failure_mode,count\n", encoding="utf-8")
    write_csv(reports_dir / "manual_quality_template.csv", manual_template(scored))
    write_charts(charts_dir, aggregate_by_method, derived_by_method)

    metadata_path = results_dir / "run_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else None
    write_report(reports_dir / "experiment_summary.md", results_dir, aggregate_by_method, aggregate_by_question, derived_by_method, metadata)

    print(f"Wrote {reports_dir / 'scored_results.csv'}")
    print(f"Wrote {reports_dir / 'aggregate_by_method.csv'}")
    print(f"Wrote {reports_dir / 'aggregate_by_question.csv'}")
    print(f"Wrote {reports_dir / 'derived_by_method.csv'}")
    print(f"Wrote {reports_dir / 'failure_modes.csv'}")
    print(f"Wrote {reports_dir / 'manual_quality_template.csv'}")
    print(f"Wrote charts in {charts_dir}")
    print(f"Wrote {reports_dir / 'experiment_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
