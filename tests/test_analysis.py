"""Tests for result aggregation and reporting."""

from __future__ import annotations

import json

import pytest

from kvcomp.analysis import (
    aggregate,
    bootstrap_ci,
    build_reports,
    group_by,
    load_results,
)


NEWLINE = chr(10)


def _row(**overrides):
    row = {
        "policy": "snapkv",
        "budget": 512,
        "task": "niah_single_1",
        "context_length": 4096,
        "index": 0,
        "score": 1.0,
        "oom": False,
        "error": None,
        "peak_vram_bytes": 2**31,
        "cache_bytes": 2**26,
        "compression_ratio": 0.9,
        "prefill_seconds": 1.0,
        "compress_seconds": 0.1,
        "decode_tokens_per_second": 10.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def results_file(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        _row(index=0, score=1.0),
        _row(index=1, score=0.0),
        _row(index=2, score=1.0),
        _row(policy="full", budget=-1, index=0, score=1.0, compression_ratio=0.0),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class TestLoading:
    def test_reads_all_rows(self, results_file):
        assert len(load_results(results_file)) == 4

    def test_skips_a_truncated_final_line(self, tmp_path):
        """A run killed mid-write leaves a partial line; it must not abort the
        report for every sample that completed successfully."""
        path = tmp_path / "partial.jsonl"
        path.write_text(json.dumps(_row()) + "\n{\"policy\": \"snap", encoding="utf-8")
        assert len(load_results(path)) == 1

    def test_deduplicates_repeated_keys(self, tmp_path):
        """Two processes sharing an output file can log the same sample twice;
        counting it twice would silently skew every aggregate."""
        path = tmp_path / "dupes.jsonl"
        rows = [_row(index=0, score=1.0), _row(index=0, score=0.0), _row(index=1)]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        loaded = load_results(path)
        assert len(loaded) == 2

    def test_deduplication_keeps_the_last_write(self, tmp_path):
        path = tmp_path / "dupes.jsonl"
        rows = [_row(index=0, score=1.0), _row(index=0, score=0.0)]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert load_results(path)[0]["score"] == 0.0

    def test_niah_depths_are_not_collapsed(self, tmp_path):
        """Depth must be part of the identity key.

        Every NIAH sample carries task="niah" and distinguishes itself by needle
        depth, so a key without depth maps all depths onto the same index. A
        945-run sweep deduplicated to 135 rows, dropping 86% of the data from
        every published aggregate while the report still looked well-formed.
        """
        path = tmp_path / "niah.jsonl"
        rows = [
            _row(task="niah", index=i, depth=d, score=1.0)
            for i in range(3)
            for d in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        path.write_text(NEWLINE.join(json.dumps(r) for r in rows), encoding="utf-8")

        loaded = load_results(path)
        assert len(loaded) == 15, "distinct depths were collapsed onto one index"
        assert len({r["depth"] for r in loaded}) == 5

    def test_deduplication_can_be_disabled(self, tmp_path):
        path = tmp_path / "dupes.jsonl"
        rows = [_row(index=0), _row(index=0)]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert len(load_results(path, deduplicate=False)) == 2


class TestBootstrap:
    def test_interval_contains_the_mean(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        low, high = bootstrap_ci(values)
        assert low <= sum(values) / len(values) <= high

    def test_identical_values_give_a_zero_width_interval(self):
        assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)

    def test_single_observation_is_handled(self):
        assert bootstrap_ci([0.5]) == (0.5, 0.5)

    def test_empty_input_is_handled(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_more_samples_narrow_the_interval(self):
        """The reason intervals are reported at all: small cells are uncertain
        and a 3-point gap between methods is usually noise."""
        pattern = [1.0, 0.0]
        low_small, high_small = bootstrap_ci(pattern * 3, seed=1)
        low_large, high_large = bootstrap_ci(pattern * 50, seed=1)
        assert (high_large - low_large) < (high_small - low_small)

    def test_is_reproducible(self):
        values = [1.0, 0.0, 1.0, 0.0, 1.0]
        assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


class TestAggregate:
    def test_mean_score(self):
        summary = aggregate([_row(score=1.0), _row(score=0.0)])
        assert summary.mean == 0.5
        assert summary.count == 2

    def test_empty_group(self):
        assert aggregate([]).count == 0

    def test_oom_rate_is_tracked_separately_from_score(self):
        """A method that could not run has not 'scored zero' in the same sense
        as one that ran and answered incorrectly."""
        summary = aggregate([_row(score=0.0, oom=True, error="OOM"), _row(score=1.0)])
        assert summary.oom_rate == 0.5
        assert summary.mean == 0.5

    def test_failed_runs_are_excluded_from_timing_averages(self):
        """Timings from a failed run describe the failure, not the method."""
        summary = aggregate(
            [_row(prefill_seconds=2.0), _row(prefill_seconds=0.0, error="OOM", oom=True)]
        )
        assert summary.mean_prefill_s == 2.0

    def test_memory_is_converted_to_gib(self):
        summary = aggregate([_row(peak_vram_bytes=2**30)])
        assert summary.mean_peak_vram_gib == pytest.approx(1.0)


class TestGrouping:
    def test_groups_by_multiple_keys(self, results_file):
        rows = load_results(results_file)
        assert len(group_by(rows, ["policy", "budget"])) == 2

    def test_group_contents_are_complete(self, results_file):
        rows = load_results(results_file)
        assert len(group_by(rows, ["policy", "budget"])[("snapkv", 512)]) == 3


class TestReports:
    def test_writes_every_report(self, results_file, tmp_path):
        written = build_reports(results_file, tmp_path / "out")
        assert set(written) == {
            "by_method",
            "by_method_length",
            "by_method_task",
            "by_depth",
            "summary",
        }
        assert all(path.exists() for path in written.values())

    def test_summary_mentions_each_policy(self, results_file, tmp_path):
        written = build_reports(results_file, tmp_path / "out")
        text = written["summary"].read_text(encoding="utf-8")
        assert "snapkv" in text and "full" in text

    def test_csv_has_a_row_per_group(self, results_file, tmp_path):
        written = build_reports(results_file, tmp_path / "out")
        lines = written["by_method"].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # header + two policies
