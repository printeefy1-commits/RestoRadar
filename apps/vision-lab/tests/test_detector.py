"""YOLO11 tensor semantics, coordinate recovery and person-only NMS."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import PersonDetector


def predictions(*rows):
    result = np.zeros((1, 84, 8400), np.float32)
    for index, row in enumerate(rows):
        result[0, : len(row), index] = row
    return result


class Yolo11DetectorTests(unittest.TestCase):
    def test_letterbox_is_centered_rgb_float_and_normalized(self):
        frame = np.full((100, 200, 3), (10, 20, 30), np.uint8)
        tensor, ratio, left, top = PersonDetector.preprocess(frame)
        self.assertEqual((ratio, left, top), (3.2, 0, 160))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 160, 0], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(tensor[0, :, 159, 0], [114 / 255] * 3)
        portrait = PersonDetector.preprocess(np.zeros((200, 100, 3), np.uint8))
        self.assertEqual(portrait[1:], (3.2, 160, 0))

    def test_xywh_recovered_after_padding_for_landscape_and_portrait(self):
        for shape in ((100, 200, 3), (200, 100, 3), (101, 203, 3)):
            with self.subTest(shape=shape):
                _, ratio, left, top = PersonDetector.preprocess(
                    np.zeros(shape, np.uint8)
                )
                raw = predictions(
                    [50 * ratio + left, 40 * ratio + top, 40 * ratio, 60 * ratio, 0.9]
                )
                people = PersonDetector.postprocess(raw, shape, ratio, left, top)
                np.testing.assert_allclose(people.xyxy, [[30, 10, 70, 70]], atol=1e-4)

    def test_filters_other_coco_classes_bad_scores_and_boxes(self):
        raw = predictions(
            [200, 200, 100, 100, 0.9],
            [400, 400, 100, 100, 0.8, 0.95],  # bicycle wins over person
            [100, 100, 100, 100, 0.09],
            [100, 100, -10, 100, 0.9],
            [100, 100, 1, 1, 0.9],
            [100, 100, 100, 100, float("nan")],
            [100, 100, 100, 100, 1.1],
            [100, 100, 100, 100, 0.9, float("inf")],
        )
        people = PersonDetector.postprocess(raw, (640, 640, 3), 1, 0, 0)
        self.assertEqual(len(people), 1)
        np.testing.assert_allclose(people.xyxy, [[150, 150, 250, 250]])
        np.testing.assert_equal(people.class_id, [0])

    def test_nms_precedes_cap_and_keeps_best_score_and_other_people(self):
        duplicate = [100, 100, 80, 160, 0.9]
        raw = predictions(*([duplicate] * 40 + [[500, 400, 60, 80, 0.5]]))
        raw[0, 4, 20] = 0.99
        people = PersonDetector.postprocess(raw, (640, 640, 3), 1, 0, 0)
        self.assertEqual(len(people), 2)
        np.testing.assert_allclose(people.confidence, [0.99, 0.5])

    def test_contained_torso_is_removed_before_limit_without_losing_other_people(self):
        rows = [
            [30 + (i % 8) * 70, 30 + (i // 8) * 100, 20, 60, 0.8] for i in range(28)
        ]
        rows += [
            [610, 500, 50, 200, 0.7],
            [610, 460, 50, 120, 0.9],
            [520, 330, 20, 60, 0.6],
        ]
        result = PersonDetector.postprocess(predictions(*rows), (640, 640, 3), 1, 0, 0)
        self.assertEqual(len(result), 30)
        self.assertEqual(
            sum(np.all(np.isclose(result.xyxy, [585, 400, 635, 600]), axis=1)), 1
        )
        self.assertFalse(
            np.any(np.all(np.isclose(result.xyxy, [585, 400, 635, 520]), axis=1))
        )
        self.assertAlmostEqual(float(min(result.confidence)), 0.6, places=5)

    def test_clipping_empty_and_rejects_old_yolox_output(self):
        raw = predictions([0, 0, 100, 100, 0.9], [-100, -100, 50, 50, 0.95])
        people = PersonDetector.postprocess(raw, (640, 640, 3), 1, 0, 0)
        np.testing.assert_allclose(people.xyxy, [[0, 0, 50, 50]])
        self.assertEqual(
            len(PersonDetector.postprocess(predictions(), (640, 640, 3), 1, 0, 0)), 0
        )
        with self.assertRaisesRegex(ValueError, "YOLO11l"):
            PersonDetector.postprocess(np.zeros((1, 30, 5)), (640, 640, 3), 1, 0, 0)


if __name__ == "__main__":
    unittest.main()
