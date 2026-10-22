"""Data-free tests for selecting and auditing CLEVR functional programs."""

import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from src.clevr_programs import (
    AnswerMismatchError,
    ProgramNode,
    ProgramSchemaError,
    SceneSchemaError,
    SelectionDecision,
    audit_official_answer,
    evaluate_selected_program,
    parse_program,
    select_single_relation_exist,
)


class ProgramParsingTests(unittest.TestCase):
    def test_parse_program_preserves_valid_nodes_as_immutable_records(self):
        raw = [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {
                "function": "filter_color",
                "inputs": [0],
                "value_inputs": ["red"],
                "ignored_official_metadata": "allowed",
            },
        ]

        nodes = parse_program(raw)

        self.assertEqual(
            nodes,
            (
                ProgramNode("scene", (), ()),
                ProgramNode("filter_color", (0,), ("red",)),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            nodes[1].function = "filter_shape"

    def test_parse_program_rejects_a_non_list_or_empty_program(self):
        for raw in (None, {}, (), "scene", []):
            with self.subTest(raw=raw), self.assertRaises(ProgramSchemaError):
                parse_program(raw)

    def test_parse_program_rejects_non_mapping_nodes_and_missing_fields(self):
        malformed = [
            ["scene"],
            {"inputs": [], "value_inputs": []},
            {"function": "scene", "value_inputs": []},
            {"function": "scene", "inputs": []},
        ]
        for node in malformed:
            with self.subTest(node=node), self.assertRaises(ProgramSchemaError):
                parse_program([node])

    def test_parse_program_rejects_wrong_function_inputs_and_value_types(self):
        malformed = [
            {"function": 1, "inputs": [], "value_inputs": []},
            {"function": "", "inputs": [], "value_inputs": []},
            {"function": "scene", "inputs": (), "value_inputs": []},
            {"function": "scene", "inputs": [True], "value_inputs": []},
            {"function": "scene", "inputs": [-1], "value_inputs": []},
            {"function": "scene", "inputs": ["0"], "value_inputs": []},
            {"function": "scene", "inputs": [], "value_inputs": ()},
            {"function": "scene", "inputs": [], "value_inputs": [1]},
        ]
        for node in malformed:
            with self.subTest(node=node), self.assertRaises(ProgramSchemaError):
                parse_program([node])


class ProgramSelectionTests(unittest.TestCase):
    def test_accepts_every_literal_filter_domain_value(self):
        domains = {
            "filter_color": (
                "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"
            ),
            "filter_shape": ("cube", "sphere", "cylinder"),
            "filter_size": ("small", "large"),
            "filter_material": ("rubber", "metal"),
        }
        for function, values in domains.items():
            for value in values:
                raw = [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": function, "inputs": [0], "value_inputs": [value]},
                    {"function": "unique", "inputs": [1], "value_inputs": []},
                    {"function": "relate", "inputs": [2], "value_inputs": ["front"]},
                    {"function": "exist", "inputs": [3], "value_inputs": []},
                ]
                with self.subTest(function=function, value=value):
                    self.assertEqual(
                        select_single_relation_exist(parse_program(raw), "yes").relation,
                        "front",
                    )

    def test_accepts_zero_or_multiple_filters_and_all_four_relations(self):
        zero_filter_template = [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "unique", "inputs": [0], "value_inputs": []},
            {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
            {"function": "exist", "inputs": [2], "value_inputs": []},
        ]
        for relation in ("left", "right", "front", "behind"):
            raw = [dict(node) for node in zero_filter_template]
            raw[2] = {"function": "relate", "inputs": [1], "value_inputs": [relation]}
            with self.subTest(relation=relation):
                self.assertEqual(
                    select_single_relation_exist(parse_program(raw), "yes"),
                    SelectionDecision(relation=relation, exclusion_reason=None),
                )

        multiple_filters = parse_program([
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
            {"function": "filter_shape", "inputs": [1], "value_inputs": ["cube"]},
            {"function": "filter_size", "inputs": [2], "value_inputs": ["small"]},
            {"function": "unique", "inputs": [3], "value_inputs": []},
            {"function": "relate", "inputs": [4], "value_inputs": ["behind"]},
            {"function": "filter_material", "inputs": [5], "value_inputs": ["metal"]},
            {"function": "filter_color", "inputs": [6], "value_inputs": ["cyan"]},
            {"function": "exist", "inputs": [7], "value_inputs": []},
        ])
        self.assertEqual(
            select_single_relation_exist(multiple_filters, "no"),
            SelectionDecision(relation="behind", exclusion_reason=None),
        )

    def test_rejects_target_nodes_with_invalid_value_arguments_as_schema_errors(self):
        def with_filter(node):
            return [
                {"function": "scene", "inputs": [], "value_inputs": []},
                node,
                {"function": "unique", "inputs": [1], "value_inputs": []},
                {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [3], "value_inputs": []},
            ]

        cases = [
            with_filter(
                {"function": "filter_color", "inputs": [0], "value_inputs": ["orange"]}
            ),
            with_filter(
                {
                    "function": "filter_shape",
                    "inputs": [0],
                    "value_inputs": ["cube", "sphere"],
                }
            ),
            with_filter(
                {"function": "filter_size", "inputs": [0], "value_inputs": []}
            ),
            with_filter(
                {"function": "filter_material", "inputs": [0], "value_inputs": ["wood"]}
            ),
            [
                {"function": "scene", "inputs": [], "value_inputs": ["extra"]},
                {"function": "unique", "inputs": [0], "value_inputs": []},
                {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [2], "value_inputs": []},
            ],
            [
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": "unique", "inputs": [0], "value_inputs": ["extra"]},
                {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [2], "value_inputs": []},
            ],
            [
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": "unique", "inputs": [0], "value_inputs": []},
                {"function": "relate", "inputs": [1], "value_inputs": ["above"]},
                {"function": "exist", "inputs": [2], "value_inputs": []},
            ],
            [
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": "unique", "inputs": [0], "value_inputs": []},
                {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [2], "value_inputs": ["extra"]},
            ],
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ProgramSchemaError):
                select_single_relation_exist(parse_program(raw), "yes")

    def test_rejects_non_binary_answers_only_for_target_shaped_programs(self):
        target = parse_program([
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "unique", "inputs": [0], "value_inputs": []},
            {"function": "relate", "inputs": [1], "value_inputs": ["right"]},
            {"function": "exist", "inputs": [2], "value_inputs": []},
        ])
        for answer in (True, 1, "maybe", None):
            with self.subTest(answer=answer), self.assertRaises(ProgramSchemaError):
                select_single_relation_exist(target, answer)

        count_program = parse_program([
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "count", "inputs": [0], "value_inputs": []},
        ])
        self.assertEqual(
            select_single_relation_exist(count_program, 3),
            SelectionDecision(relation=None, exclusion_reason="unsupported_function"),
        )

    def test_excludes_branches_skips_and_wrong_first_inputs_as_non_linear(self):
        cases = [
            [
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": "unique", "inputs": [0, 0], "value_inputs": []},
                {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [2], "value_inputs": []},
            ],
            [
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
                {"function": "unique", "inputs": [0], "value_inputs": []},
                {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [3], "value_inputs": []},
            ],
            [
                {"function": "scene", "inputs": [0], "value_inputs": []},
                {"function": "unique", "inputs": [0], "value_inputs": []},
                {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [2], "value_inputs": []},
            ],
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    select_single_relation_exist(parse_program(raw), "yes"),
                    SelectionDecision(relation=None, exclusion_reason="non_linear_program"),
                )

    def test_excludes_wrong_counts_order_and_terminal_with_stable_categories(self):
        cases = [
            (
                "unique_count",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "relate", "inputs": [0], "value_inputs": ["left"]},
                    {"function": "exist", "inputs": [1], "value_inputs": []},
                ],
            ),
            (
                "unique_count",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "unique", "inputs": [0], "value_inputs": []},
                    {"function": "unique", "inputs": [1], "value_inputs": []},
                    {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
                    {"function": "exist", "inputs": [3], "value_inputs": []},
                ],
            ),
            (
                "relate_count",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "unique", "inputs": [0], "value_inputs": []},
                    {"function": "exist", "inputs": [1], "value_inputs": []},
                ],
            ),
            (
                "relate_count",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "unique", "inputs": [0], "value_inputs": []},
                    {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                    {"function": "relate", "inputs": [2], "value_inputs": ["right"]},
                    {"function": "exist", "inputs": [3], "value_inputs": []},
                ],
            ),
            (
                "unique_relate_adjacency",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "relate", "inputs": [0], "value_inputs": ["left"]},
                    {"function": "unique", "inputs": [1], "value_inputs": []},
                    {"function": "exist", "inputs": [2], "value_inputs": []},
                ],
            ),
            (
                "scene_layout",
                [
                    {"function": "filter_color", "inputs": [], "value_inputs": ["red"]},
                    {"function": "unique", "inputs": [0], "value_inputs": []},
                    {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                    {"function": "exist", "inputs": [2], "value_inputs": []},
                ],
            ),
            (
                "exist_layout",
                [
                    {"function": "scene", "inputs": [], "value_inputs": []},
                    {"function": "unique", "inputs": [0], "value_inputs": []},
                    {"function": "relate", "inputs": [1], "value_inputs": ["left"]},
                    {"function": "filter_color", "inputs": [2], "value_inputs": ["red"]},
                ],
            ),
        ]
        for reason, raw in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    select_single_relation_exist(parse_program(raw), "yes"),
                    SelectionDecision(relation=None, exclusion_reason=reason),
                )

    def test_selection_decision_requires_exactly_one_outcome(self):
        for arguments in (
            {"relation": None, "exclusion_reason": None},
            {"relation": "left", "exclusion_reason": "non_linear_program"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SelectionDecision(**arguments)


class ProgramEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "objects": [
                {"color": "red", "shape": "cube", "size": "small", "material": "rubber"},
                {"color": "blue", "shape": "sphere", "size": "large", "material": "metal"},
                {"color": "green", "shape": "cylinder", "size": "small", "material": "metal"},
            ],
            "relationships": {
                "left": [[1, 2], [2], []],
                "right": [[], [0], [0, 1]],
                "front": [[2], [], [1]],
                "behind": [[1], [2], []],
            },
        }

    def _program(self, relation, post_filter, post_value):
        return parse_program([
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
            {"function": "filter_shape", "inputs": [1], "value_inputs": ["cube"]},
            {"function": "filter_size", "inputs": [2], "value_inputs": ["small"]},
            {"function": "filter_material", "inputs": [3], "value_inputs": ["rubber"]},
            {"function": "unique", "inputs": [4], "value_inputs": []},
            {"function": "relate", "inputs": [5], "value_inputs": [relation]},
            {"function": post_filter, "inputs": [6], "value_inputs": [post_value]},
            {"function": "exist", "inputs": [7], "value_inputs": []},
        ])

    def test_evaluator_executes_each_filter_and_relation_with_literal_results(self):
        cases = [
            ("left", "filter_color", "blue", True),
            ("right", "filter_color", "blue", False),
            ("front", "filter_material", "metal", True),
            ("behind", "filter_size", "large", True),
        ]
        for relation, post_filter, post_value, expected in cases:
            nodes = self._program(relation, post_filter, post_value)
            with self.subTest(relation=relation):
                self.assertIs(evaluate_selected_program(nodes, self.scene), expected)

    def test_official_answer_audit_accepts_hand_checked_yes_and_no(self):
        yes_nodes = self._program("left", "filter_shape", "sphere")
        no_nodes = self._program("right", "filter_shape", "sphere")

        self.assertIsNone(audit_official_answer(yes_nodes, self.scene, "yes"))
        self.assertIsNone(audit_official_answer(no_nodes, self.scene, "no"))

    def test_unique_rejects_zero_or_multiple_matching_objects(self):
        cases = [
            ("filter_color", "purple"),
            ("filter_size", "small"),
        ]
        for filter_function, value in cases:
            nodes = parse_program([
                {"function": "scene", "inputs": [], "value_inputs": []},
                {"function": filter_function, "inputs": [0], "value_inputs": [value]},
                {"function": "unique", "inputs": [1], "value_inputs": []},
                {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
                {"function": "exist", "inputs": [3], "value_inputs": []},
            ])
            with self.subTest(filter_function=filter_function), self.assertRaises(
                SceneSchemaError
            ):
                evaluate_selected_program(nodes, self.scene)

    def test_scene_rejects_wrong_relationship_row_count_and_out_of_range_indices(self):
        nodes = self._program("left", "filter_color", "blue")
        malformed_scenes = []

        wrong_row_count = deepcopy(self.scene)
        wrong_row_count["relationships"]["left"] = [[1, 2], [2]]
        malformed_scenes.append(wrong_row_count)

        out_of_range = deepcopy(self.scene)
        out_of_range["relationships"]["left"][0] = [3]
        malformed_scenes.append(out_of_range)

        boolean_index = deepcopy(self.scene)
        boolean_index["relationships"]["left"][0] = [True]
        malformed_scenes.append(boolean_index)

        for scene in malformed_scenes:
            with self.subTest(scene=scene), self.assertRaises(SceneSchemaError):
                evaluate_selected_program(nodes, scene)

    def test_scene_rejects_missing_or_invalid_object_attributes(self):
        nodes = self._program("left", "filter_color", "blue")
        mutations = [
            ("color", None),
            ("color", "orange"),
            ("shape", "cone"),
            ("size", "medium"),
            ("material", "wood"),
        ]
        for attribute, replacement in mutations:
            scene = deepcopy(self.scene)
            if replacement is None:
                del scene["objects"][1][attribute]
            else:
                scene["objects"][1][attribute] = replacement
            with self.subTest(attribute=attribute, replacement=replacement), self.assertRaises(
                SceneSchemaError
            ):
                evaluate_selected_program(nodes, scene)

    def test_scene_rejects_malformed_top_level_collections_and_relation_rows(self):
        nodes = self._program("left", "filter_color", "blue")
        malformed = [
            None,
            {"objects": {}, "relationships": self.scene["relationships"]},
            {"objects": self.scene["objects"], "relationships": []},
            {"objects": self.scene["objects"], "relationships": {"left": [[], [], []]}},
            {
                "objects": self.scene["objects"],
                "relationships": {
                    "left": [[1, 2], [2], "not-a-row"],
                    "right": [[], [0], [0, 1]],
                    "front": [[2], [], [1]],
                    "behind": [[1], [2], []],
                },
            },
        ]
        for scene in malformed:
            with self.subTest(scene=scene), self.assertRaises(SceneSchemaError):
                evaluate_selected_program(nodes, scene)

    def test_official_answer_disagreement_is_a_hard_failure_in_both_directions(self):
        yes_result = self._program("left", "filter_shape", "sphere")
        no_result = self._program("right", "filter_shape", "sphere")

        with self.assertRaises(AnswerMismatchError):
            audit_official_answer(yes_result, self.scene, "no")
        with self.assertRaises(AnswerMismatchError):
            audit_official_answer(no_result, self.scene, "yes")


if __name__ == "__main__":
    unittest.main()
