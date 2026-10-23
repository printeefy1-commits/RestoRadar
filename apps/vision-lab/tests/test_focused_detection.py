"""Synthetic geometry regressions; these do not measure person-detection accuracy."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/restoradar-test-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import supervision as sv

from focused_detection import detect_with_seat_focus, merge_views


def detections(boxes, scores):
    return sv.Detections(
        xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        confidence=np.array(scores, np.float32),
        class_id=np.zeros(len(boxes), int),
    )


class ViewMergeTests(unittest.TestCase):
    def test_aligned_torso_yields_to_longer_body_with_its_original_score(self):
        body = detections([[200, 450, 310, 760]], [0.55])
        torso = detections([[205, 465, 290, 590]], [0.76])
        # IoU is below 0.6 even though the small box is entirely inside the body.
        for parts in ([body, torso], [torso, body]):
            result = merge_views(parts)
            np.testing.assert_array_equal(result.xyxy, body.xyxy)
            np.testing.assert_array_equal(result.confidence, body.confidence)

    def test_complete_views_still_choose_the_higher_original_score(self):
        body = [[100, 100, 200, 400]]
        result = merge_views([detections(body, [0.4]), detections(body, [0.9])])
        np.testing.assert_array_equal(result.xyxy, body)
        np.testing.assert_allclose(result.confidence, [0.9])

    def test_neighbor_at_different_horizontal_position_is_not_merged(self):
        outer = detections([[100, 100, 260, 400]], [0.6])
        neighbor = detections([[150, 100, 260, 220]], [0.8])
        self.assertEqual(len(merge_views([outer, neighbor])), 2)

    def test_partial_overlap_does_not_count_as_near_complete_containment(self):
        outer = detections([[100, 100, 200, 400]], [0.6])
        overlapping = detections([[94, 100, 194, 200]], [0.8])
        self.assertEqual(len(merge_views([outer, overlapping])), 2)

    def test_occluded_person_lower_in_the_image_is_not_merged(self):
        outer = detections([[100, 100, 200, 400]], [0.6])
        neighbor = detections([[110, 220, 190, 380]], [0.8])
        self.assertEqual(len(merge_views([outer, neighbor])), 2)

    def test_narrow_contained_person_is_not_assumed_to_be_a_duplicate(self):
        outer = detections([[100, 100, 300, 400]], [0.6])
        neighbor = detections([[170, 100, 230, 250]], [0.8])
        self.assertEqual(len(merge_views([outer, neighbor])), 2)

    def test_containment_in_one_inference_view_is_not_enough(self):
        same_view = detections(
            [[200, 450, 310, 760], [205, 465, 290, 590]], [0.55, 0.76]
        )
        self.assertEqual(len(merge_views([same_view])), 2)

    def test_weak_longer_body_cannot_replace_or_borrow_score_from_torso(self):
        body = detections([[200, 450, 310, 760]], [0.16])
        torso = detections([[205, 465, 290, 590]], [0.76])
        result = merge_views([body, torso])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.xyxy[0], torso.xyxy[0])
        np.testing.assert_array_equal(
            result.confidence, [*torso.confidence, *body.confidence]
        )

    def test_suppressed_box_cannot_extend_a_match_transitively(self):
        body = detections([[100, 100, 200, 400]], [0.6])
        intermediate = detections([[100, 105, 200, 250]], [0.7])
        lower = detections([[100, 115, 200, 175]], [0.8])
        # Intermediate matches both boxes, but body's top does not match lower's.
        result = merge_views([body, intermediate, lower])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.xyxy, [lower.xyxy[0], body.xyxy[0]])

    def test_unrelated_boxes_survive_and_candidate_limit_remains_thirty(self):
        boxes = [[10 * i, 0, 10 * i + 5, 10] for i in range(35)]
        scores = np.linspace(0.4, 0.9, len(boxes), dtype=np.float32)
        result = merge_views([detections(boxes, scores), sv.Detections.empty()])
        self.assertEqual(len(result), 30)
        np.testing.assert_array_equal(result.xyxy, boxes[:4:-1])
        np.testing.assert_array_equal(result.confidence, scores[:4:-1])
        self.assertEqual(len(merge_views([sv.Detections.empty()])), 0)

    def test_detection_path_remaps_crop_before_comparing_body_and_torso(self):
        frame = np.zeros((1000, 1000, 3), np.uint8)
        regions = [
            {"tableId": "table", "x": 0.4, "y": 0.4, "width": 0.1, "height": 0.2}
        ]

        def detector(image):
            if image.shape[:2] == (1000, 1000):
                return detections([[400, 280, 500, 650]], [0.55])
            return detections([[50, 30, 150, 140]], [0.9])

        result, crops = detect_with_seat_focus(detector, frame, regions)
        self.assertEqual(crops, 1)
        np.testing.assert_array_equal(result.xyxy, [[400, 280, 500, 650]])
        np.testing.assert_allclose(result.confidence, [0.55])


if __name__ == "__main__":
    unittest.main()
