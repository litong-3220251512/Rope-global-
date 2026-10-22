"""Data-free tests for selecting and auditing CLEVR functional programs."""

import unittest
from dataclasses import FrozenInstanceError

from src.clevr_programs import (
    ProgramNode,
    ProgramSchemaError,
    SelectionDecision,
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


if __name__ == "__main__":
    unittest.main()
