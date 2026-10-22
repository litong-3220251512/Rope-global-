"""Data-free tests for the frozen CLEVR question-text contract."""

import unittest
from dataclasses import FrozenInstanceError

from src.clevr_text import (
    Vocabulary,
    build_vocabulary,
    encode_question,
    tokenize_question,
)


class QuestionTokenizationTests(unittest.TestCase):
    def test_preserves_case_and_separates_commas_and_semicolons(self):
        self.assertEqual(
            tokenize_question("Is the Red cube, left; or right?"),
            (
                "<START>",
                "Is",
                "the",
                "Red",
                "cube",
                ",",
                "left",
                ";",
                "or",
                "right",
                "<END>",
            ),
        )

    def test_removes_question_marks_and_periods_before_splitting(self):
        self.assertEqual(
            tokenize_question("A.B? C."),
            ("<START>", "AB", "C", "<END>"),
        )

    def test_literal_space_split_preserves_empty_tokens(self):
        self.assertEqual(
            tokenize_question("What  color is it?"),
            ("<START>", "What", "", "color", "is", "it", "<END>"),
        )


class VocabularyTests(unittest.TestCase):
    def test_fixed_special_ids_precede_lexicographically_sorted_training_tokens(self):
        vocabulary = build_vocabulary(("zebra cube?", "Apple  cube."))

        self.assertEqual(
            dict(vocabulary.token_to_id),
            {
                "<NULL>": 0,
                "<START>": 1,
                "<END>": 2,
                "<UNK>": 3,
                "": 4,
                "Apple": 5,
                "cube": 6,
                "zebra": 7,
            },
        )
        self.assertEqual(
            vocabulary.id_to_token,
            (
                "<NULL>",
                "<START>",
                "<END>",
                "<UNK>",
                "",
                "Apple",
                "cube",
                "zebra",
            ),
        )

    def test_vocabulary_record_and_mapping_reject_mutation(self):
        vocabulary = build_vocabulary(("red cube?",))

        with self.assertRaises(TypeError):
            vocabulary.token_to_id["blue"] = 99
        with self.assertRaises(FrozenInstanceError):
            vocabulary.id_to_token = ()

    def test_public_constructor_rejects_invalid_fixed_id_vocabularies(self):
        invalid_vocabularies = (
            (
                "wrong special IDs",
                {"<NULL>": 1, "<START>": 0, "<END>": 2, "<UNK>": 3},
                ("<NULL>", "<START>", "<END>", "<UNK>"),
            ),
            (
                "wrong special order",
                {"<NULL>": 0, "<END>": 1, "<START>": 2, "<UNK>": 3},
                ("<NULL>", "<END>", "<START>", "<UNK>"),
            ),
            (
                "duplicate token",
                {
                    "<NULL>": 0,
                    "<START>": 1,
                    "<END>": 2,
                    "<UNK>": 3,
                    "red": 4,
                },
                ("<NULL>", "<START>", "<END>", "<UNK>", "red", "red"),
            ),
            (
                "unsorted ordinary tokens",
                {
                    "<NULL>": 0,
                    "<START>": 1,
                    "<END>": 2,
                    "<UNK>": 3,
                    "red": 4,
                    "blue": 5,
                },
                ("<NULL>", "<START>", "<END>", "<UNK>", "red", "blue"),
            ),
            (
                "non-dense inverse mapping",
                {
                    "<NULL>": 0,
                    "<START>": 1,
                    "<END>": 2,
                    "<UNK>": 3,
                    "red": 5,
                },
                ("<NULL>", "<START>", "<END>", "<UNK>", "red"),
            ),
            (
                "non-mapping token IDs",
                [],
                ("<NULL>", "<START>", "<END>", "<UNK>"),
            ),
            (
                "mutable token sequence",
                {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3},
                ["<NULL>", "<START>", "<END>", "<UNK>"],
            ),
            (
                "non-string token",
                {
                    "<NULL>": 0,
                    "<START>": 1,
                    "<END>": 2,
                    "<UNK>": 3,
                    7: 4,
                },
                ("<NULL>", "<START>", "<END>", "<UNK>", 7),
            ),
            (
                "non-integer ID",
                {
                    "<NULL>": 0,
                    "<START>": 1,
                    "<END>": 2,
                    "<UNK>": 3,
                    "red": "4",
                },
                ("<NULL>", "<START>", "<END>", "<UNK>", "red"),
            ),
        )
        for name, token_to_id, id_to_token in invalid_vocabularies:
            with self.subTest(name=name), self.assertRaises(ValueError):
                Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)

    def test_encoding_maps_unseen_tokens_to_unk_without_expanding_vocabulary(self):
        vocabulary = build_vocabulary(("Is the cube red?",))
        original_mapping = dict(vocabulary.token_to_id)

        encoded = encode_question("Is the sphere blue?", vocabulary)

        self.assertIsInstance(encoded, tuple)
        self.assertEqual(
            encoded,
            (
                1,
                vocabulary.token_to_id["Is"],
                vocabulary.token_to_id["the"],
                3,
                3,
                2,
            ),
        )
        self.assertEqual(dict(vocabulary.token_to_id), original_mapping)
        self.assertNotIn("sphere", vocabulary.token_to_id)
        self.assertNotIn("blue", vocabulary.token_to_id)


if __name__ == "__main__":
    unittest.main()
