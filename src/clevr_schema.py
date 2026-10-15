"""Data-free schema primitives for Stage 1 CLEVR spatial queries."""

from __future__ import annotations

import math
from dataclasses import InitVar, asdict, dataclass
from numbers import Real
from typing import Literal

import torch
from torch import Tensor


_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ObjectLocation:
    object_index: int
    x: float
    y: float
    patch_x: int
    patch_y: int
    region_x: int
    region_y: int
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("ObjectLocation records must be created by GridSpec.locate")


@dataclass(frozen=True, slots=True)
class QueryRecord:
    scene_id: int
    official_split: str
    image_filename: str
    source_width: int
    source_height: int
    query_kind: Literal["absolute", "relative"]
    object_a: ObjectLocation
    object_b: ObjectLocation | None
    label: int | str
    same_region: bool | None
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("QueryRecord records must be created by query builders")

    def to_dict(self) -> dict[str, object]:
        """Return a nested record suitable for JSON-lines manifests."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GridSpec:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    patch_width: int
    patch_height: int

    def __post_init__(self) -> None:
        dimensions = (
            self.source_width,
            self.source_height,
            self.input_width,
            self.input_height,
            self.patch_width,
            self.patch_height,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in dimensions):
            raise ValueError("image and patch dimensions must be positive integers")
        if self.input_width % self.patch_width or self.input_height % self.patch_height:
            raise ValueError("input dimensions must be divisible by patch dimensions")

    @property
    def patch_columns(self) -> int:
        return self.input_width // self.patch_width

    @property
    def patch_rows(self) -> int:
        return self.input_height // self.patch_height

    def locate(self, object_index: int, x: float, y: float) -> ObjectLocation:
        if not isinstance(object_index, int) or isinstance(object_index, bool) or object_index < 0:
            raise ValueError("object_index must be a non-negative integer")
        if (not isinstance(x, Real) or isinstance(x, bool) or not math.isfinite(x)
                or not isinstance(y, Real) or isinstance(y, bool) or not math.isfinite(y)):
            raise ValueError("object coordinates must be finite real numbers")
        if not 0 <= x < self.source_width or not 0 <= y < self.source_height:
            raise ValueError("object coordinates lie outside the source image")
        normalized_x = x if isinstance(x, int) else float(x)
        normalized_y = y if isinstance(y, int) else float(y)
        patch_x = int(self.patch_columns * normalized_x / self.source_width)
        patch_y = int(self.patch_rows * normalized_y / self.source_height)
        return ObjectLocation(
            object_index=object_index,
            x=normalized_x,
            y=normalized_y,
            patch_x=patch_x,
            patch_y=patch_y,
            region_x=(2 * patch_x) // self.patch_columns,
            region_y=(2 * patch_y) // self.patch_rows,
            _construction_token=_CONSTRUCTION_TOKEN,
        )

    def marker(self, location: ObjectLocation) -> Tensor:
        if not isinstance(location, ObjectLocation):
            raise ValueError("marker location must be an ObjectLocation")
        if location != self.locate(location.object_index, location.x, location.y):
            raise ValueError("marker location fields do not match its source coordinates")
        marker = torch.zeros((self.input_height, self.input_width), dtype=torch.float32)
        x_start = location.patch_x * self.patch_width
        y_start = location.patch_y * self.patch_height
        marker[y_start:y_start + self.patch_height, x_start:x_start + self.patch_width] = 1
        return marker


def _validate_query_metadata(scene_id: int, official_split: str, image_filename: str) -> None:
    if not isinstance(scene_id, int) or isinstance(scene_id, bool) or scene_id < 0:
        raise ValueError("scene_id must be a non-negative integer")
    if not isinstance(official_split, str) or official_split not in {"train", "val"}:
        raise ValueError("official_split must be train or val")
    if not isinstance(image_filename, str) or not image_filename.strip():
        raise ValueError("image_filename must be a non-empty string")


def build_absolute_query(
    *,
    grid: GridSpec,
    scene_id: int,
    official_split: str,
    image_filename: str,
    object_index: int,
    x: float,
    y: float,
) -> QueryRecord:
    _validate_query_metadata(scene_id, official_split, image_filename)
    location = grid.locate(object_index, x, y)
    return QueryRecord(
        scene_id=scene_id,
        official_split=official_split,
        image_filename=image_filename,
        source_width=grid.source_width,
        source_height=grid.source_height,
        query_kind="absolute",
        object_a=location,
        object_b=None,
        label=2 * location.region_y + location.region_x,
        same_region=None,
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def build_relative_query(
    *,
    grid: GridSpec,
    scene_id: int,
    official_split: str,
    image_filename: str,
    a_index: int,
    a_x: float,
    a_y: float,
    b_index: int,
    b_x: float,
    b_y: float,
) -> QueryRecord:
    _validate_query_metadata(scene_id, official_split, image_filename)
    if a_index == b_index:
        raise ValueError("relative queries require two distinct object indices")
    object_a = grid.locate(a_index, a_x, a_y)
    object_b = grid.locate(b_index, b_x, b_y)
    if a_x == b_x or a_y == b_y:
        raise ValueError("relative quadrant is undefined when either coordinate difference is zero")
    if (object_a.patch_x, object_a.patch_y) == (object_b.patch_x, object_b.patch_y):
        raise ValueError("relative objects must map to distinguishable patches")
    vertical = "upper" if a_y < b_y else "lower"
    horizontal = "left" if a_x < b_x else "right"
    return QueryRecord(
        scene_id=scene_id,
        official_split=official_split,
        image_filename=image_filename,
        source_width=grid.source_width,
        source_height=grid.source_height,
        query_kind="relative",
        object_a=object_a,
        object_b=object_b,
        label=f"{vertical}-{horizontal}",
        same_region=(object_a.region_x, object_a.region_y)
        == (object_b.region_x, object_b.region_y),
        _construction_token=_CONSTRUCTION_TOKEN,
    )
