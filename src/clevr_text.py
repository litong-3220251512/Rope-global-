"""Deterministic encoding for original CLEVR question text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


_SPECIAL_TOKENS = ("<NULL>", "<START>", "<END>", "<UNK>")
_START_TOKEN = _SPECIAL_TOKENS[1]
_END_TOKEN = _SPECIAL_TOKENS[2]
_UNK_TOKEN = _SPECIAL_TOKENS[3]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    token_to_id: Mapping[str, int]
    id_to_token: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.token_to_id, Mapping):
            raise ValueError("token_to_id must be a mapping")
        if not isinstance(self.id_to_token, tuple):
            raise ValueError("id_to_token must be a tuple")

        token_to_id = dict(self.token_to_id)
        if any(
            type(token) is not str or type(token_id) is not int
            for token, token_id in token_to_id.items()
        ):
            raise ValueError("token_to_id must map strings to integer IDs")
        if any(type(token) is not str for token in self.id_to_token):
            raise ValueError("id_to_token must contain only strings")
        if self.id_to_token[: len(_SPECIAL_TOKENS)] != _SPECIAL_TOKENS:
            raise ValueError("id_to_token must begin with the fixed special tokens")
        if len(set(self.id_to_token)) != len(self.id_to_token):
            raise ValueError("id_to_token must not contain duplicate tokens")

        ordinary_tokens = self.id_to_token[len(_SPECIAL_TOKENS) :]
        if ordinary_tokens != tuple(sorted(ordinary_tokens)):
            raise ValueError("ordinary tokens must be sorted lexicographically")
        expected_mapping = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }
        if token_to_id != expected_mapping:
            raise ValueError("token_to_id must exactly invert id_to_token")

        object.__setattr__(self, "token_to_id", MappingProxyType(token_to_id))


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
    unknown_id = vocabulary.token_to_id[_UNK_TOKEN]
    return tuple(
        vocabulary.token_to_id.get(token, unknown_id)
        for token in tokenize_question(question)
    )
