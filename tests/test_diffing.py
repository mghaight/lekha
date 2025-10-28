from __future__ import annotations

import unittest

from lekha.diffing import BaseToken, WordConsensus, compute_word_consensus, tokenize


class TokenizeTests(unittest.TestCase):
    def test_tokenize_splits_on_whitespace(self) -> None:
        result = tokenize("hello world foo bar")
        self.assertEqual(result, ["hello", "world", "foo", "bar"])

    def test_tokenize_handles_empty_string(self) -> None:
        result = tokenize("")
        self.assertEqual(result, [])

    def test_tokenize_handles_multiple_spaces(self) -> None:
        result = tokenize("hello  world   foo")
        self.assertEqual(result, ["hello", "world", "foo"])


class WordConsensusPropertyTests(unittest.TestCase):
    def test_has_conflict_false_when_no_alternatives(self) -> None:
        consensus = WordConsensus(base="hello", line_index=0, word_index=0)
        self.assertFalse(consensus.has_conflict)

    def test_has_conflict_false_when_alternatives_match_base(self) -> None:
        consensus = WordConsensus(
            base="hello", line_index=0, word_index=0, alternatives={"model1": "hello", "model2": "hello"}
        )
        self.assertFalse(consensus.has_conflict)

    def test_has_conflict_true_when_alternative_differs(self) -> None:
        consensus = WordConsensus(
            base="hello", line_index=0, word_index=0, alternatives={"model1": "hallo", "model2": "hello"}
        )
        self.assertTrue(consensus.has_conflict)

    def test_has_conflict_false_when_alternative_is_empty(self) -> None:
        consensus = WordConsensus(base="hello", line_index=0, word_index=0, alternatives={"model1": ""})
        self.assertFalse(consensus.has_conflict)

    def test_display_text_shows_base_when_no_alternatives(self) -> None:
        consensus = WordConsensus(base="hello", line_index=0, word_index=0)
        self.assertEqual(consensus.display_text, "hello")

    def test_display_text_shows_base_when_alternatives_match(self) -> None:
        consensus = WordConsensus(
            base="hello", line_index=0, word_index=0, alternatives={"model1": "hello", "model2": "hello"}
        )
        self.assertEqual(consensus.display_text, "hello")

    def test_display_text_shows_bracket_notation_for_conflicts(self) -> None:
        consensus = WordConsensus(
            base="hello", line_index=0, word_index=0, alternatives={"model1": "hallo", "model2": "helo"}
        )
        display = consensus.display_text
        self.assertTrue(display.startswith("["))
        self.assertTrue(display.endswith("]"))
        self.assertIn("hello", display)
        self.assertIn("hallo", display)
        self.assertIn("helo", display)

    def test_display_text_removes_duplicates(self) -> None:
        consensus = WordConsensus(
            base="hello", line_index=0, word_index=0, alternatives={"model1": "hello", "model2": "hallo"}
        )
        display = consensus.display_text
        # Should be [hello/hallo] not [hello/hello/hallo]
        parts = display.strip("[]").split("/")
        self.assertEqual(len(parts), 2)
        self.assertIn("hello", parts)
        self.assertIn("hallo", parts)


class ComputeWordConsensusTests(unittest.TestCase):
    def test_consensus_with_no_models_returns_base(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].base, "hello")
        self.assertEqual(result[1].base, "world")
        self.assertFalse(result[0].has_conflict)
        self.assertFalse(result[1].has_conflict)

    def test_consensus_with_matching_model_text(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {"model1": "hello world"})
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].has_conflict)
        self.assertFalse(result[1].has_conflict)

    def test_consensus_with_replacement(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {"model1": "hallo world"})
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].has_conflict)
        self.assertEqual(result[0].alternatives["model1"], "hallo")
        self.assertFalse(result[1].has_conflict)

    def test_consensus_with_deletion(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {"model1": "world"})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].alternatives.get("model1"), "")
        self.assertFalse(result[1].has_conflict)

    def test_consensus_with_insertion(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {"model1": "hello beautiful world"})
        self.assertEqual(len(result), 2)
        # "beautiful" should be appended to "hello"
        self.assertIn("model1", result[0].alternatives)
        self.assertIn("beautiful", result[0].alternatives["model1"])

    def test_consensus_handles_empty_model_text(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
        ]
        result = compute_word_consensus(base_tokens, {"model1": ""})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].base, "hello")

    def test_consensus_with_multiple_models(self) -> None:
        base_tokens = [
            BaseToken(text="hello", line_index=0, word_index=0),
            BaseToken(text="world", line_index=0, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {"model1": "hallo world", "model2": "hi world"})
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].has_conflict)
        self.assertEqual(result[0].alternatives["model1"], "hallo")
        self.assertEqual(result[0].alternatives["model2"], "hi")

    def test_consensus_with_complex_replacement(self) -> None:
        base_tokens = [
            BaseToken(text="the", line_index=0, word_index=0),
            BaseToken(text="quick", line_index=0, word_index=1),
            BaseToken(text="brown", line_index=0, word_index=2),
        ]
        # Replace "quick brown" with "fast red"
        result = compute_word_consensus(base_tokens, {"model1": "the fast red"})
        self.assertEqual(len(result), 3)
        self.assertFalse(result[0].has_conflict)
        self.assertTrue(result[1].has_conflict)
        self.assertEqual(result[1].alternatives["model1"], "fast")
        self.assertTrue(result[2].has_conflict)
        self.assertEqual(result[2].alternatives["model1"], "red")

    def test_consensus_preserves_line_and_word_indices(self) -> None:
        base_tokens = [
            BaseToken(text="first", line_index=0, word_index=0),
            BaseToken(text="second", line_index=1, word_index=0),
            BaseToken(text="third", line_index=1, word_index=1),
        ]
        result = compute_word_consensus(base_tokens, {})
        self.assertEqual(result[0].line_index, 0)
        self.assertEqual(result[0].word_index, 0)
        self.assertEqual(result[1].line_index, 1)
        self.assertEqual(result[1].word_index, 0)
        self.assertEqual(result[2].line_index, 1)
        self.assertEqual(result[2].word_index, 1)
