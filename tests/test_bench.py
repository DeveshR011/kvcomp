"""Tests for benchmark generation and scoring.

A benchmark bug is worse than a code bug: it produces plausible numbers that are
simply wrong. These tests check that samples are the length they claim, that
answers are actually present in the context, and that generation is reproducible.
"""

from __future__ import annotations

import pytest

from kvcomp.bench.base import Sample, fit_to_length, insert_at_depth, score_sample
from kvcomp.bench.niah import needle_haystack_sweep
from kvcomp.bench.ruler import RULER_TASKS, generate_ruler


class WordTokenizer:
    """Whitespace tokenizer standing in for a real one.

    Length control must not depend on a specific vocabulary, so the tests use a
    trivial tokenizer and assert on proportional behaviour rather than exact
    token counts.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [hash(word) for word in text.split()]


@pytest.fixture
def tokenizer() -> WordTokenizer:
    return WordTokenizer()


class TestScoring:
    def test_full_recall(self):
        sample = Sample("t", 128, 0, "p", ["1234", "5678"])
        assert score_sample(sample, "the numbers are 1234 and 5678") == 1.0

    def test_partial_recall(self):
        sample = Sample("t", 128, 0, "p", ["1234", "5678"])
        assert score_sample(sample, "only 1234 here") == 0.5

    def test_no_recall(self):
        sample = Sample("t", 128, 0, "p", ["1234"])
        assert score_sample(sample, "no idea") == 0.0

    def test_scoring_is_case_insensitive(self):
        sample = Sample("t", 128, 0, "p", ["Golden Gate"])
        assert score_sample(sample, "the golden gate bridge") == 1.0

    def test_substring_credit_tolerates_surrounding_prose(self):
        """Models wrap answers in commentary; exact match would understate
        genuine accuracy, which is why RULER scores by containment."""
        sample = Sample("t", 128, 0, "p", ["8421337"])
        verbose = "Based on the document, the special magic number is 8421337."
        assert score_sample(sample, verbose) == 1.0

    def test_empty_answers_score_zero(self):
        assert score_sample(Sample("t", 128, 0, "p", []), "anything") == 0.0


class TestLengthControl:
    def test_respects_the_target_budget(self, tokenizer):
        text = fit_to_length(tokenizer, ["alpha beta gamma"] * 50, 60)
        assert len(text.split()) <= 60

    def test_gets_reasonably_close_to_target(self, tokenizer):
        """Loose packing would make 'context length' meaningless as an axis."""
        text = fit_to_length(tokenizer, ["alpha beta gamma"] * 500, 300)
        assert len(text.split()) >= 240

    def test_empty_units_yield_empty_text(self, tokenizer):
        assert fit_to_length(tokenizer, [], 100) == ""


class TestNeedleInsertion:
    @pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_needle_is_present_at_every_depth(self, depth):
        haystack = " ".join(f"Sentence number {i}." for i in range(50))
        result = insert_at_depth(haystack, "The secret is 42.", depth)
        assert "The secret is 42." in result

    def test_depth_controls_relative_position(self):
        haystack = " ".join(f"Sentence number {i}." for i in range(100))
        early = insert_at_depth(haystack, "NEEDLE.", 0.1).index("NEEDLE")
        late = insert_at_depth(haystack, "NEEDLE.", 0.9).index("NEEDLE")
        assert early < late

    def test_single_sentence_haystack_is_handled(self):
        assert "NEEDLE" in insert_at_depth("Only one.", "NEEDLE.", 0.5)


class TestRuler:
    def test_every_task_generates(self, tokenizer):
        samples = generate_ruler(
            tokenizer, list(RULER_TASKS), [512], samples_per_cell=1
        )
        assert {s.task for s in samples} == set(RULER_TASKS)

    @pytest.mark.parametrize("task", RULER_TASKS)
    def test_answers_appear_in_the_prompt(self, tokenizer, task):
        """A sample whose answer is absent from its own context is unanswerable
        and would silently depress every method's score."""
        for sample in generate_ruler(tokenizer, [task], [1024], samples_per_cell=3):
            assert sample.answers, f"{task} produced no answers"
            for answer in sample.answers:
                assert answer.lower() in sample.prompt.lower(), (
                    f"{task}: answer {answer!r} missing from its own context"
                )

    def test_generation_is_reproducible(self, tokenizer):
        first = generate_ruler(tokenizer, ["niah_single_1"], [512], 2, seed=7)
        second = generate_ruler(tokenizer, ["niah_single_1"], [512], 2, seed=7)
        assert [s.prompt for s in first] == [s.prompt for s in second]

    def test_different_seeds_give_different_samples(self, tokenizer):
        first = generate_ruler(tokenizer, ["niah_single_1"], [512], 1, seed=1)
        second = generate_ruler(tokenizer, ["niah_single_1"], [512], 1, seed=2)
        assert first[0].prompt != second[0].prompt

    def test_adding_a_task_does_not_perturb_others(self, tokenizer):
        """Per-cell seeding means a sweep can be extended without invalidating
        results already collected."""
        alone = generate_ruler(tokenizer, ["vt"], [512], 2, seed=3)
        together = generate_ruler(tokenizer, ["cwe", "vt"], [512], 2, seed=3)
        vt_only = [s.prompt for s in together if s.task == "vt"]
        assert [s.prompt for s in alone] == vt_only

    def test_longer_targets_produce_longer_prompts(self, tokenizer):
        short = generate_ruler(tokenizer, ["niah_single_1"], [512], 1)[0]
        long = generate_ruler(tokenizer, ["niah_single_1"], [4096], 1)[0]
        assert len(long.prompt) > len(short.prompt) * 2

    def test_multivalue_asks_for_every_value(self, tokenizer):
        sample = generate_ruler(tokenizer, ["niah_multivalue"], [1024], 1)[0]
        assert len(sample.answers) == 4

    def test_multiquery_asks_about_several_keys(self, tokenizer):
        sample = generate_ruler(tokenizer, ["niah_multiquery"], [1024], 1)[0]
        assert len(sample.answers) == 4

    def test_multikey_has_distractor_needles(self, tokenizer):
        """Distractors are what stop the task being solvable by spotting the
        one sentence that looks different from the filler."""
        sample = generate_ruler(tokenizer, ["niah_multikey_1"], [1024], 1)[0]
        assert sample.prompt.count("One of the special magic numbers") == 4
        assert len(sample.answers) == 1

    def test_vt_answers_form_an_assignment_chain(self, tokenizer):
        sample = generate_ruler(tokenizer, ["vt"], [1024], 1)[0]
        value = sample.metadata["value"]
        assert f"= {value}" in sample.prompt
        assert len(sample.answers) == sample.metadata["chain_length"]

    def test_cwe_answers_are_the_most_frequent_words(self, tokenizer):
        sample = generate_ruler(tokenizer, ["cwe"], [2048], 1)[0]
        words = sample.prompt.split()
        counts = {answer: words.count(answer) for answer in sample.answers}
        assert all(count > 1 for count in counts.values())

    def test_unknown_task_raises(self, tokenizer):
        with pytest.raises(ValueError, match="unknown RULER tasks"):
            generate_ruler(tokenizer, ["not_a_task"], [512], 1)


class TestNiah:
    def test_grid_covers_every_cell(self, tokenizer):
        samples = needle_haystack_sweep(tokenizer, [512, 1024], [0.0, 0.5, 1.0])
        assert len(samples) == 6

    def test_depth_is_recorded(self, tokenizer):
        samples = needle_haystack_sweep(tokenizer, [512], [0.0, 0.5, 1.0])
        assert sorted(s.metadata["depth"] for s in samples) == [0.0, 0.5, 1.0]

    def test_needle_is_in_the_prompt(self, tokenizer):
        for sample in needle_haystack_sweep(tokenizer, [1024], [0.0, 0.5, 1.0]):
            assert sample.answers[0] in sample.prompt

    def test_invalid_depth_raises(self, tokenizer):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            needle_haystack_sweep(tokenizer, [512], [1.5])

    def test_reproducible(self, tokenizer):
        first = needle_haystack_sweep(tokenizer, [512], [0.5], seed=4)
        second = needle_haystack_sweep(tokenizer, [512], [0.5], seed=4)
        assert first[0].prompt == second[0].prompt
