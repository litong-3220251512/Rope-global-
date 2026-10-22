"""Pure validation and selection helpers for official CLEVR programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ProgramSchemaError(ValueError):
    """Raised when a CLEVR functional program violates its source schema."""


class SceneSchemaError(ValueError):
    """Raised when a scene cannot support faithful official-answer auditing."""


class AnswerMismatchError(ValueError):
    """Raised when a recomputed answer disagrees with the published answer."""


@dataclass(frozen=True, slots=True)
class ProgramNode:
    function: str
    inputs: tuple[int, ...]
    value_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.function, str) or not self.function:
            raise ProgramSchemaError("node function must be a non-empty string")
        if not isinstance(self.inputs, tuple) or any(
            type(index) is not int or index < 0 for index in self.inputs
        ):
            raise ProgramSchemaError("node inputs must be non-negative integer indices")
        if not isinstance(self.value_inputs, tuple) or any(
            not isinstance(value, str) for value in self.value_inputs
        ):
            raise ProgramSchemaError("node value_inputs must contain strings")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    relation: str | None
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        if (self.relation is None) == (self.exclusion_reason is None):
            raise ValueError("selection decisions require exactly one outcome")
        if self.relation is not None and self.relation not in _RELATIONS:
            raise ValueError("selection relation is not supported")
        if self.exclusion_reason is not None and (
            not isinstance(self.exclusion_reason, str) or not self.exclusion_reason
        ):
            raise ValueError("exclusion_reason must be a non-empty string")


_RELATIONS = frozenset({"left", "right", "front", "behind"})
_FILTER_VALUES = {
    "filter_color": frozenset(
        {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
    ),
    "filter_shape": frozenset({"cube", "sphere", "cylinder"}),
    "filter_size": frozenset({"small", "large"}),
    "filter_material": frozenset({"rubber", "metal"}),
}
_OBJECT_VALUES = {
    function.removeprefix("filter_"): values
    for function, values in _FILTER_VALUES.items()
}
_TARGET_FUNCTIONS = frozenset(
    {"scene", "unique", "relate", "exist", *_FILTER_VALUES}
)


def parse_program(raw: object) -> tuple[ProgramNode, ...]:
    """Parse the published function/inputs/value_inputs representation."""
    if not isinstance(raw, list) or not raw:
        raise ProgramSchemaError("program must be a non-empty list")

    parsed: list[ProgramNode] = []
    for position, raw_node in enumerate(raw):
        if not isinstance(raw_node, Mapping):
            raise ProgramSchemaError(f"program node {position} must be a mapping")
        missing = {"function", "inputs", "value_inputs"}.difference(raw_node)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ProgramSchemaError(f"program node {position} is missing: {fields}")

        function = raw_node["function"]
        inputs = raw_node["inputs"]
        value_inputs = raw_node["value_inputs"]
        if not isinstance(inputs, list):
            raise ProgramSchemaError(f"program node {position} inputs must be a list")
        if not isinstance(value_inputs, list):
            raise ProgramSchemaError(f"program node {position} value_inputs must be a list")
        try:
            parsed.append(ProgramNode(function, tuple(inputs), tuple(value_inputs)))
        except ProgramSchemaError as exc:
            raise ProgramSchemaError(f"program node {position}: {exc}") from exc

    return tuple(parsed)


def _excluded(reason: str) -> SelectionDecision:
    return SelectionDecision(relation=None, exclusion_reason=reason)


def _validate_program_node_arguments(nodes: tuple[ProgramNode, ...]) -> None:
    for position, node in enumerate(nodes):
        if any(input_index >= position for input_index in node.inputs):
            raise ProgramSchemaError(
                f"program node {position} references itself or a later node"
            )

        if node.function not in _TARGET_FUNCTIONS:
            continue
        expected_input_count = 0 if node.function == "scene" else 1
        if len(node.inputs) != expected_input_count:
            raise ProgramSchemaError(
                f"target node {position} {node.function} requires "
                f"{expected_input_count} input(s)"
            )
        if node.function in {"scene", "unique", "exist"}:
            if node.value_inputs:
                raise ProgramSchemaError(
                    f"target node {position} {node.function} takes no value_inputs"
                )
        elif node.function in _FILTER_VALUES:
            allowed_values = _FILTER_VALUES[node.function]
            if len(node.value_inputs) != 1 or node.value_inputs[0] not in allowed_values:
                raise ProgramSchemaError(
                    f"target node {position} has invalid {node.function} value_inputs"
                )
        elif node.function == "relate":
            if len(node.value_inputs) != 1 or node.value_inputs[0] not in _RELATIONS:
                raise ProgramSchemaError(
                    f"target node {position} has invalid relate value_inputs"
                )


def select_single_relation_exist(
    nodes: tuple[ProgramNode, ...], answer: object
) -> SelectionDecision:
    """Select the frozen linear single-relation existence program family."""
    if not isinstance(nodes, tuple) or not nodes or any(
        not isinstance(node, ProgramNode) for node in nodes
    ):
        raise ProgramSchemaError("nodes must be a non-empty tuple of ProgramNode records")
    _validate_program_node_arguments(nodes)

    functions = tuple(node.function for node in nodes)
    if any(function not in _TARGET_FUNCTIONS for function in functions):
        return _excluded("unsupported_function")

    if functions.count("unique") != 1:
        return _excluded("unique_count")
    if functions.count("relate") != 1:
        return _excluded("relate_count")
    if functions.count("scene") != 1 or functions[0] != "scene":
        return _excluded("scene_layout")
    if functions.count("exist") != 1 or functions[-1] != "exist":
        return _excluded("exist_layout")

    expected_inputs = tuple(
        () if position == 0 else (position - 1,) for position in range(len(nodes))
    )
    if tuple(node.inputs for node in nodes) != expected_inputs:
        return _excluded("non_linear_program")

    unique_index = functions.index("unique")
    relate_index = functions.index("relate")
    if relate_index != unique_index + 1:
        return _excluded("unique_relate_adjacency")

    if not isinstance(answer, str) or answer not in {"yes", "no"}:
        raise ProgramSchemaError("target program answer must be exactly 'yes' or 'no'")

    return SelectionDecision(
        relation=nodes[relate_index].value_inputs[0], exclusion_reason=None
    )


def evaluate_selected_program(nodes: tuple[ProgramNode, ...], scene: object) -> bool:
    """Evaluate only the selected linear program family on one scene graph."""
    decision = select_single_relation_exist(nodes, "yes")
    if decision.relation is None:
        raise ProgramSchemaError("only selected relation-existence programs can be evaluated")

    objects, relationships = _validate_scene(scene)
    output: list[int] | int | bool = []
    for node in nodes:
        if node.function == "scene":
            output = list(range(len(objects)))
        elif node.function in _FILTER_VALUES:
            attribute = node.function.removeprefix("filter_")
            value = node.value_inputs[0]
            output = [index for index in output if objects[index][attribute] == value]
        elif node.function == "unique":
            if len(output) != 1:
                raise SceneSchemaError("unique must receive exactly one object")
            output = output[0]
        elif node.function == "relate":
            output = list(relationships[node.value_inputs[0]][output])
        elif node.function == "exist":
            output = bool(output)

    return output


def _validate_scene(
    scene: object,
) -> tuple[list[Mapping[str, object]], Mapping[str, list[list[int]]]]:
    if not isinstance(scene, Mapping):
        raise SceneSchemaError("scene must be a mapping")
    objects = scene.get("objects")
    relationships = scene.get("relationships")
    if not isinstance(objects, list):
        raise SceneSchemaError("scene objects must be a list")
    if not isinstance(relationships, Mapping):
        raise SceneSchemaError("scene relationships must be a mapping")

    for object_index, scene_object in enumerate(objects):
        if not isinstance(scene_object, Mapping):
            raise SceneSchemaError(f"scene object {object_index} must be a mapping")
        for attribute, allowed_values in _OBJECT_VALUES.items():
            value = scene_object.get(attribute)
            if not isinstance(value, str) or value not in allowed_values:
                raise SceneSchemaError(
                    f"scene object {object_index} has invalid or missing {attribute}"
                )

    object_count = len(objects)
    for relation in _RELATIONS:
        rows = relationships.get(relation)
        if not isinstance(rows, list) or len(rows) != object_count:
            raise SceneSchemaError(
                f"relationship {relation} must have one row per object"
            )
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                raise SceneSchemaError(
                    f"relationship {relation} row {row_index} must be a list"
                )
            if any(
                type(related_index) is not int
                or not 0 <= related_index < object_count
                for related_index in row
            ):
                raise SceneSchemaError(
                    f"relationship {relation} row {row_index} contains an invalid object index"
                )

    return objects, relationships


def validate_scene(scene: object) -> None:
    """Validate the complete object and relationship schema without executing a program."""
    _validate_scene(scene)


def audit_official_answer(
    nodes: tuple[ProgramNode, ...], scene: object, answer: object
) -> None:
    """Require the official yes/no answer to match independent evaluation."""
    select_single_relation_exist(nodes, answer)
    recomputed = evaluate_selected_program(nodes, scene)
    expected = answer == "yes"
    if recomputed is not expected:
        raise AnswerMismatchError(
            f"official answer {answer!r} disagrees with recomputed "
            f"{'yes' if recomputed else 'no'!r}"
        )
