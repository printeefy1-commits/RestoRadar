"""Same-view duplicate regressions, including adjacent/occluded people."""

from pathlib import Path
import sys
import unittest

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from person_duplicates import suppress_aligned_torsos


def detections(boxes, scores):
    return sv.Detections(
        xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        confidence=np.array(scores, np.float32),
        class_id=np.zeros(len(boxes), int),
    )


class PersonDuplicateTests(unittest.TestCase):
    def test_aligned_torso_keeps_longer_body_and_its_own_score_in_either_order(self):
        body, torso = [100, 100, 200, 400], [101, 101, 199, 300]
        for boxes, scores in (
            ([body, torso], [0.55, 0.9]),
            ([torso, body], [0.9, 0.55]),
        ):
            source = detections(boxes, scores)
            result = suppress_aligned_torsos(source)
            np.testing.assert_allclose(result.xyxy, [body])
            np.testing.assert_allclose(result.confidence, [0.55])
            np.testing.assert_allclose(source.xyxy, boxes)
            np.testing.assert_allclose(source.confidence, scores)

    def test_weak_long_box_cannot_borrow_a_stronger_torso_score(self):
        source = detections([[100, 100, 200, 400], [100, 100, 200, 300]], [0.16, 0.9])
        result = suppress_aligned_torsos(source)
        np.testing.assert_allclose(result.xyxy, [source.xyxy[1]])
        np.testing.assert_allclose(result.confidence, [0.9])
        source.confidence[:] = [0.25, 0.3]
        np.testing.assert_allclose(suppress_aligned_torsos(source).confidence, [0.3])

    def test_adjacent_lower_narrow_partial_or_loose_alignment_is_preserved(self):
        body = [100, 100, 200, 400]
        others = (
            [150, 100, 250, 300],  # adjacent person
            [100, 200, 200, 380],  # lower head under occlusion
            [130, 100, 170, 300],  # narrower person inside body
            [104, 100, 204, 300],  # only 96% containment
            [105, 115, 195, 250],  # weaker cross-view alignment is insufficient
            [100, 100, 200, 160],  # very short contained box
        )
        for other in others:
            with self.subTest(other=other):
                self.assertEqual(
                    len(suppress_aligned_torsos(detections([body, other], [0.6, 0.8]))),
                    2,
                )

    def test_suppressed_intermediate_cannot_merge_another_person(self):
        boxes = [[100, 100, 200, 500], [100, 105, 200, 360], [100, 111, 200, 251]]
        result = suppress_aligned_torsos(detections(boxes, [0.7, 0.8, 0.9]))
        np.testing.assert_allclose(result.xyxy, [boxes[0], boxes[2]])

    def test_weak_outer_cannot_suppress_neighbors_before_losing_to_a_torso(self):
        boxes = [[100, 100, 200, 500], [105, 100, 195, 300], [98.5, 100, 201.5, 400]]
        result = suppress_aligned_torsos(detections(boxes, [0.2, 0.15, 0.3]))
        # Each short box matches the weak outer; they do not match each other.
        np.testing.assert_allclose(result.xyxy, boxes[1:])
        np.testing.assert_allclose(result.confidence, [0.15, 0.3])

    def test_report_seven_final_geometry_removes_only_the_aligned_pair(self):
        # Coordinate-only regression; no video, identities or report is stored.
        boxes = [
            [211.2, 467.2, 290.6, 605.6],
            [122.7, 597.2, 230.5, 863.3],
            [122.9, 511.9, 189.6, 662.2],
            [225.8, 544.5, 308.4, 778.7],
            [807.3, 400.9, 860.4, 561.0],
            [683.8, 374.7, 750.6, 556.1],  # torso duplicate
            [684.2, 375.4, 752.0, 638.2],  # longer body
        ]
        result = suppress_aligned_torsos(
            detections(boxes, [0.7866, 0.7571, 0.7514, 0.7093, 0.4706, 0.4322, 0.3717])
        )
        self.assertEqual(len(result), 6)
        np.testing.assert_allclose(result.xyxy, boxes[:5] + boxes[6:])
        np.testing.assert_allclose(result.confidence[-1], 0.3717)

    def test_empty_single_and_unrelated_detections_are_unchanged(self):
        for boxes, scores in (
            ([], []),
            ([[0, 0, 10, 20]], [0.8]),
            ([[0, 0, 10, 20], [30, 0, 40, 20]], [0.8, 0.9]),
        ):
            source = detections(boxes, scores)
            result = suppress_aligned_torsos(source)
            np.testing.assert_equal(result.xyxy, source.xyxy)
            np.testing.assert_equal(result.confidence, source.confidence)


if __name__ == "__main__":
    unittest.main()
