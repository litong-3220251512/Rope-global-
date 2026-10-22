"""Deterministic encoding for original CLEVR question text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


_SPECIAL_TOKENS = ("<NULL>", "<START>", "<END>", "<UNK>")
_START_TOKEN = _SPECIAL_TOKENS[1]
_END_TOKEN = _SPECIAL_TOKENS[2]
_UNK_ID = 3


@dataclass(frozen=True, slots=True)
class Vocabulary:
    token_to_id: Mapping[str, int]
    id_to_token: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "token_to_id", MappingProxyType(dict(self.token_to_id))
        )
        object.__setattr__(self, "id_to_token", tuple(self.id_to_token))


def tokenize_question(question: str) -> tuple[str, ...]:
    """Apply the frozen punctuation and literal-space tokenization contract."""
    for separator in (";", ","):
        question = question.replace(separator, f" {separator}")
    for punctuation in ("?", "."):
        question = question.replace(punctuation, "")
    return (_START_TOKEN, *question.split(" "), _END_TOKEN)


def build_vocabulary(training_questions: Sequence[str]) -> Vocabulary:
    """Build fixed specials plus sorted ordinary tokens from training only."""
    ordinary_tokens: set[str] = set()
    for question in training_questions:
        ordinary_tokens.update(tokenize_question(question)[1:-1])
    ordinary_tokens.difference_update(_SPECIAL_TOKENS)

    id_to_token = _SPECIAL_TOKENS + tuple(sorted(ordinary_tokens))
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


def encode_question(
    question: str, vocabulary: Vocabulary
) -> tuple[int, ...]:
    """Encode with the frozen vocabulary, mapping unseen tokens to UNK."""
    return tuple(
        vocabulary.token_to_id.get(token, _UNK_ID)
        for token in tokenize_question(question)
    )
