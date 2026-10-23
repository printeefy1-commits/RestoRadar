"""Nano pose tensor semantics and preservation of each detection's own pose."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/restoradar-test-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import supervision as sv

from analyze import NanoPoseDetector
from focused_detection import detect_with_seat_focus


def predictions(*rows):
    result = np.zeros((1, 300, 57), dtype=np.float32)
    for index, (box, score, point) in enumerate(rows):
        result[0, index, :6] = [*box, score, 0]
        result[0, index, 6:] = np.tile([*point, 0.8], 17)
    return result


class NanoPoseTests(unittest.TestCase):
    def test_boxes_and_keypoints_share_letterbox_recovery(self):
        for shape in ((100, 200, 3), (200, 100, 3), (101, 203, 3)):
            with self.subTest(shape=shape):
                _, ratio, left, top = NanoPoseDetector.preprocess(
                    np.zeros(shape, np.uint8)
                )
                box = [
                    30 * ratio + left,
                    10 * ratio + top,
                    70 * ratio + left,
                    70 * ratio + top,
                ]
                raw = predictions((box, 0.9, [50 * ratio + left, 40 * ratio + top]))
                people = NanoPoseDetector.postprocess(raw, shape, ratio, left, top)
                np.testing.assert_allclose(people.xyxy, [[30, 10, 70, 70]], atol=1e-4)
                np.testing.assert_allclose(
                    people.data["keypoints"][0, :, :2], [[50, 40]] * 17, atol=1e-4
                )
                np.testing.assert_allclose(people.data["keypoints"][0, :, 2], 0.8)

    def test_invalid_class_scores_coordinates_and_keypoint_scores_are_rejected(self):
        row = ([100, 100, 200, 300], 0.9, [150, 200])
        raw = predictions(*([row] * 9))
        raw[0, 1, 5] = 1
        raw[0, 2, 4] = 0.09
        raw[0, 3, 4] = 1.1
        raw[0, 4, 6] = np.nan
        raw[0, 5, 8] = 1.01
        raw[0, 6, 8] = -0.01
        raw[0, 7, 2] = 99
        raw[0, 8, 2] = 101
        result = NanoPoseDetector.postprocess(raw, (640, 640, 3), 1, 0, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.data["keypoints"].shape, (1, 17, 3))

    def test_torso_filter_retains_the_selected_box_own_score_and_pose(self):
        raw = predictions(
            ([100, 100, 200, 400], 0.7, [150, 250]),
            ([100, 100, 200, 280], 0.99, [155, 160]),
            ([300, 100, 400, 400], 0.8, [350, 250]),
        )
        result = NanoPoseDetector.postprocess(raw, (640, 640, 3), 1, 0, 0)
        np.testing.assert_allclose(result.confidence, [0.8, 0.7])
        np.testing.assert_allclose(
            result.data["keypoints"][:, 0, :2], [[350, 250], [150, 250]]
        )

    def test_bytetrack_preserves_current_poses_when_confidence_order_changes(self):
        tracker = sv.ByteTrack(track_activation_threshold=0.35, frame_rate=2)
        for step in range(3):
            rows = [
                ([10, 10, 90, 200], 0.9 if step != 1 else 0.7, [50, 50 + step]),
                ([200, 10, 280, 200], 0.8, [240, 70 + step]),
            ]
            current = NanoPoseDetector.postprocess(
                predictions(*rows), (640, 640, 3), 1, 0, 0
            )
            tracked = tracker.update_with_detections(current)
            self.assertEqual(len(tracked), 2)
            for box, keypoints in zip(tracked.xyxy, tracked.data["keypoints"]):
                np.testing.assert_allclose(
                    keypoints[0, :2],
                    [50, 50 + step] if box[0] < 100 else [240, 70 + step],
                )
        empty = NanoPoseDetector.postprocess(predictions(), (640, 640, 3), 1, 0, 0)
        self.assertEqual(len(tracker.update_with_detections(empty)), 0)

    def test_focused_detection_offsets_keypoints_and_keeps_selected_views_pose(self):
        def detector(frame):
            if frame.shape[:2] == (1000, 1000):
                box, point, score = [400, 400, 500, 600], [450, 450], 0.5
            else:
                box, point, score = [50, 150, 150, 350], [110, 210], 0.9
            return sv.Detections(
                xyxy=np.array([box], np.float32),
                confidence=np.array([score], np.float32),
                class_id=np.array([0]),
                data={"keypoints": np.array([[[*point, 0.8]] * 17], np.float32)},
            )

        result, crops = detect_with_seat_focus(
            detector,
            np.zeros((1000, 1000, 3), np.uint8),
            [{"tableId": "t", "x": 0.4, "y": 0.4, "width": 0.1, "height": 0.2}],
        )
        self.assertEqual(crops, 1)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result.xyxy, [[400, 400, 500, 600]])
        np.testing.assert_allclose(result.data["keypoints"][0, 0], [460, 460, 0.8])

    def test_empty_output_and_incompatible_export_are_handled(self):
        result = NanoPoseDetector.postprocess(predictions(), (640, 640, 3), 1, 0, 0)
        self.assertEqual(result.data["keypoints"].shape, (0, 17, 3))
        for shape in ((1, 56, 8400), (1, 84, 8400), (1, 300, 56)):
            with self.assertRaisesRegex(ValueError, "YOLO26n-Pose"):
                NanoPoseDetector.postprocess(np.zeros(shape), (640, 640, 3), 1, 0, 0)


if __name__ == "__main__":
    unittest.main()
