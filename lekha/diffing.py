"""Word-level diff helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections.abc import Mapping, Sequence


@dataclass
class BaseToken:
    text: str
    line_index: int
    word_index: int


@dataclass
class WordConsensus:
    base: str
    line_index: int
    word_index: int
    alternatives: dict[str, str] = field(default_factory=dict)

    @property
    def has_conflict(self) -> bool:
        return any(alt and alt != self.base for alt in self.alternatives.values())

    @property
    def display_text(self) -> str:
        alts: list[str] = [self.base]
        for alt in self.alternatives.values():
            if alt and alt not in alts:
                alts.append(alt)
        # Filter duplicates of base
        unique: list[str] = []
        for alt in alts:
            if alt not in unique:
                unique.append(alt)
        if len(unique) <= 1:
            return unique[0] if unique else self.base
        return "[" + "/".join(unique) + "]"


def tokenize(text: str) -> list[str]:
    return text.split()


def compute_word_consensus(
    base_tokens: Sequence[BaseToken],
    model_texts: Mapping[str, str],
) -> list[WordConsensus]:
    consensus = [
        WordConsensus(base=token.text, line_index=token.line_index, word_index=token.word_index) for token in base_tokens
    ]
    base_words = [token.text for token in base_tokens]
    for model_name, text in model_texts.items():
        if not text:
            continue
        tokens = tokenize(text)
        matcher = SequenceMatcher(None, base_words, tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                span = min(i2 - i1, j2 - j1)
                for offset in range(span):
                    base_idx = i1 + offset
                    alt = tokens[j1 + offset]
                    consensus[base_idx].alternatives[model_name] = alt
                if (j2 - j1) > span and i2 - 1 >= i1:
                    trailing_tokens = tokens[j1 + span : j2]
                    base_idx = max(i1, i2 - 1)
                    joined = " ".join(trailing_tokens)
                    _append_alternative(consensus[base_idx], model_name, joined)
                if (i2 - i1) > span:
                    for base_idx in range(i1 + span, i2):
                        consensus[base_idx].alternatives[model_name] = ""
            elif tag == "delete":
                for base_idx in range(i1, i2):
                    consensus[base_idx].alternatives[model_name] = ""
            elif tag == "insert":
                # Only append insertion if we have base tokens to attach it to
                if consensus:
                    target_idx = max(i1 - 1, 0)
                    insertion = " ".join(tokens[j1:j2])
                    _append_alternative(consensus[target_idx], model_name, insertion)
    return consensus


def _append_alternative(entry: WordConsensus, model_name: str, text: str) -> None:
    if not text:
        return
    if model_name in entry.alternatives and entry.alternatives[model_name]:
        entry.alternatives[model_name] = " ".join([entry.alternatives[model_name], text])
    else:
        entry.alternatives[model_name] = text
