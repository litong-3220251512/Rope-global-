"""Boundary tests for the data-free Stage 1 CLEVR query schema."""

import importlib
import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace

import torch

from src.local_rope import build_axial_cis


class CLEVRSchemaTests(unittest.TestCase):
    def setUp(self):
        try:
            self.schema = importlib.import_module("src.clevr_schema")
        except ModuleNotFoundError as exc:
            self.fail(f"The Stage 1 CLEVR schema is missing: {exc}")

    def test_source_pixels_map_to_hand_derived_patch_and_region_boundaries(self):
        grid = self.schema.GridSpec(
            source_width=480,
            source_height=320,
            input_width=224,
            input_height=224,
            patch_width=16,
            patch_height=16,
        )

        expected = {
            (0, 0): (0, 0, 0, 0),
            (479, 319): (13, 13, 1, 1),
            (239, 159): (6, 6, 0, 0),
            (240, 160): (7, 7, 1, 1),
        }
        for (x, y), (patch_x, patch_y, region_x, region_y) in expected.items():
            with self.subTest(x=x, y=y):
                location = grid.locate(object_index=3, x=x, y=y)
                self.assertEqual(
                    (location.patch_x, location.patch_y, location.region_x, location.region_y),
                    (patch_x, patch_y, region_x, region_y),
                )

    def test_invalid_grid_and_source_coordinates_are_rejected_without_clamping(self):
        invalid_grids = [
            dict(source_width=0, source_height=320, input_width=224, input_height=224,
                 patch_width=16, patch_height=16),
            dict(source_width=480, source_height=-1, input_width=224, input_height=224,
                 patch_width=16, patch_height=16),
            dict(source_width=480, source_height=320, input_width=225, input_height=224,
                 patch_width=16, patch_height=16),
            dict(source_width=480, source_height=320, input_width=224, input_height=224,
                 patch_width=0, patch_height=16),
        ]
        for arguments in invalid_grids:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.schema.GridSpec(**arguments)

        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        for x, y in [(-1, 0), (0, -1), (480, 0), (0, 320),
                     (math.nan, 0), (0, math.inf)]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                grid.locate(object_index=0, x=x, y=y)
        with self.assertRaises(ValueError):
            grid.locate(object_index=-1, x=0, y=0)

    def test_marker_activates_exactly_one_complete_patch_with_independent_storage(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        first = grid.marker(grid.locate(object_index=0, x=239, y=159))
        second = grid.marker(grid.locate(object_index=1, x=240, y=160))

        self.assertEqual(tuple(first.shape), (224, 224))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(first.sum().item(), 16 * 16)
        self.assertTrue(torch.equal(first[96:112, 96:112], torch.ones(16, 16)))
        self.assertEqual(first[:96].count_nonzero().item(), 0)
        self.assertEqual(first[112:].count_nonzero().item(), 0)
        self.assertEqual(first[:, :96].count_nonzero().item(), 0)
        self.assertEqual(first[:, 112:].count_nonzero().item(), 0)
        self.assertNotEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        self.assertEqual(second[112:128, 112:128].sum().item(), 16 * 16)

    def test_marker_rejects_location_fields_inconsistent_with_source_coordinates(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        location = grid.locate(object_index=0, x=239, y=159)
        forged = replace(location, patch_x=7, region_x=1)

        with self.assertRaises(ValueError):
            grid.marker(forged)

    def test_absolute_query_derives_label_and_preserves_required_source_fields(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        query = self.schema.build_absolute_query(
            grid=grid,
            scene_id=17,
            official_split="train",
            image_filename="CLEVR_train_000017.png",
            object_index=4,
            x=479,
            y=319,
        )

        self.assertEqual(query.scene_id, 17)
        self.assertEqual(query.official_split, "train")
        self.assertEqual(query.image_filename, "CLEVR_train_000017.png")
        self.assertEqual((query.source_width, query.source_height), (480, 320))
        self.assertEqual(query.query_kind, "absolute")
        self.assertEqual(query.object_a.object_index, 4)
        self.assertIsNone(query.object_b)
        self.assertEqual(query.label, 3)
        self.assertIsNone(query.same_region)
        with self.assertRaises(FrozenInstanceError):
            query.label = 0

    def test_relative_query_uses_strict_coordinate_signs_for_all_four_labels(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        cases = [
            ((100, 80), "upper-left"),
            ((300, 80), "upper-right"),
            ((100, 240), "lower-left"),
            ((300, 240), "lower-right"),
        ]
        for (a_x, a_y), expected_label in cases:
            with self.subTest(expected_label=expected_label):
                query = self.schema.build_relative_query(
                    grid=grid,
                    scene_id=18,
                    official_split="val",
                    image_filename="CLEVR_val_000018.png",
                    a_index=2,
                    a_x=a_x,
                    a_y=a_y,
                    b_index=5,
                    b_x=200,
                    b_y=160,
                )
                self.assertEqual(query.query_kind, "relative")
                self.assertEqual(query.label, expected_label)
                self.assertIsInstance(query.same_region, bool)
                self.assertEqual(query.object_a.object_index, 2)
                self.assertEqual(query.object_b.object_index, 5)

    def test_relative_query_rejects_undefined_axes_and_same_patch_not_small_distance(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        common = dict(
            grid=grid,
            scene_id=19,
            official_split="train",
            image_filename="CLEVR_train_000019.png",
            a_index=0,
            b_index=1,
        )
        invalid_coordinates = [
            dict(a_x=100, a_y=80, b_x=100, b_y=160),
            dict(a_x=100, a_y=80, b_x=200, b_y=80),
            dict(a_x=10, a_y=10, b_x=20, b_y=20),
        ]
        for coordinates in invalid_coordinates:
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                self.schema.build_relative_query(**common, **coordinates)

        adjacent = self.schema.build_relative_query(
            **common,
            a_x=239.999,
            a_y=100,
            b_x=240,
            b_y=101,
        )
        self.assertNotEqual(
            (adjacent.object_a.patch_x, adjacent.object_a.patch_y),
            (adjacent.object_b.patch_x, adjacent.object_b.patch_y),
        )
        self.assertEqual(adjacent.label, "upper-left")

    def test_swapping_objects_inverts_label_and_preserves_region_slice(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        metadata = dict(
            grid=grid,
            scene_id=20,
            official_split="train",
            image_filename="CLEVR_train_000020.png",
        )
        forward = self.schema.build_relative_query(
            **metadata,
            a_index=0,
            a_x=20,
            a_y=20,
            b_index=1,
            b_x=200,
            b_y=100,
        )
        reverse = self.schema.build_relative_query(
            **metadata,
            a_index=1,
            a_x=200,
            a_y=100,
            b_index=0,
            b_x=20,
            b_y=20,
        )

        self.assertEqual((forward.label, reverse.label), ("upper-left", "lower-right"))
        self.assertTrue(forward.same_region)
        self.assertEqual(forward.same_region, reverse.same_region)

    def test_query_metadata_and_relative_object_identity_are_validated(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        valid = dict(
            grid=grid,
            scene_id=21,
            official_split="train",
            image_filename="CLEVR_train_000021.png",
            object_index=0,
            x=20,
            y=20,
        )
        for replacement in [
            dict(scene_id=-1),
            dict(scene_id=True),
            dict(official_split="test"),
            dict(image_filename=""),
        ]:
            arguments = dict(valid)
            arguments.update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.schema.build_absolute_query(**arguments)

        with self.assertRaises(ValueError):
            self.schema.build_relative_query(
                grid=grid,
                scene_id=21,
                official_split="train",
                image_filename="CLEVR_train_000021.png",
                a_index=0,
                a_x=20,
                a_y=20,
                b_index=0,
                b_x=200,
                b_y=100,
            )

    def test_query_record_exports_a_complete_json_serializable_manifest_entry(self):
        grid = self.schema.GridSpec(480, 320, 224, 224, 16, 16)
        query = self.schema.build_relative_query(
            grid=grid,
            scene_id=22,
            official_split="val",
            image_filename="CLEVR_val_000022.png",
            a_index=2,
            a_x=239,
            a_y=159,
            b_index=5,
            b_x=240,
            b_y=160,
        )

        entry = query.to_dict()
        self.assertEqual(json.loads(json.dumps(entry)), {
            "scene_id": 22,
            "official_split": "val",
            "image_filename": "CLEVR_val_000022.png",
            "source_width": 480,
            "source_height": 320,
            "query_kind": "relative",
            "object_a": {
                "object_index": 2,
                "x": 239,
                "y": 159,
                "patch_x": 6,
                "patch_y": 6,
                "region_x": 0,
                "region_y": 0,
            },
            "object_b": {
                "object_index": 5,
                "x": 240,
                "y": 160,
                "patch_x": 7,
                "patch_y": 7,
                "region_x": 1,
                "region_y": 1,
            },
            "label": "upper-left",
            "same_region": False,
        })

    def test_all_14_patch_axis_pairs_match_region_offset_and_rope_phase_ratio(self):
        phases = build_axial_cis(1, 14, mode="B", lam=1.0)
        global_phases = build_axial_cis(1, 14, mode="G")
        same_count = 0
        cross_count = 0

        for a in range(14):
            for b in range(14):
                region_a = (2 * a) // 14
                region_b = (2 * b) // 14
                global_difference = a - b
                local_difference = (a - 7 * region_a) - (b - 7 * region_b)
                self.assertEqual(
                    local_difference,
                    global_difference - 7 * (region_a - region_b),
                )
                local_ratio = (phases[a, 8] * phases[b, 8].conj()).item()
                expected_ratio = complex(
                    math.cos(0.1 * local_difference),
                    math.sin(0.1 * local_difference),
                )
                self.assertAlmostEqual(local_ratio.real, expected_ratio.real, places=6)
                self.assertAlmostEqual(local_ratio.imag, expected_ratio.imag, places=6)
                if region_a == region_b:
                    same_count += 1
                    self.assertEqual(local_difference, global_difference)
                    torch.testing.assert_close(
                        phases[a, 8] * phases[b, 8].conj(),
                        global_phases[a, 8] * global_phases[b, 8].conj(),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                else:
                    cross_count += 1

        self.assertEqual((same_count, cross_count), (98, 98))


if __name__ == "__main__":
    unittest.main()
