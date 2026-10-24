"""Conservative same-view duplicate filtering before person tracking."""

import numpy as np

from decisions import MIN_DETECTION_SCORE

PERSON_MERGE_POLICY = "STRICT_UPPER_BODY_ALIGNMENT_V1"
POSE_MERGE_POLICY = "NEAR_IDENTICAL_BOX_AND_SHOULDERS_V1"


def suppress_pose_duplicates(detected):
    """Remove same-size duplicates only when their reliable shoulders agree.

    End-to-end pose exports can emit two almost identical complete detections.
    Keep one row intact, never move a score or keypoints to another box. A
    suppressed row cannot become a reference for a transitive merge.
    """
    if len(detected) < 2 or "keypoints" not in detected.data:
        return detected
    boxes = detected.xyxy
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keypoints = detected.data["keypoints"]
    removed = set()
    for i in np.argsort(-detected.confidence, kind="stable"):
        if i in removed:
            continue
        for j in range(len(detected)):
            if j == i or j in removed:
                continue
            if detected.confidence[j] > detected.confidence[i]:
                continue
            w, h = min(widths[i], widths[j]), min(heights[i], heights[j])
            if (
                w <= 0
                or h <= 0
                or w < 0.9 * max(widths[i], widths[j])
                or h < 0.9 * max(heights[i], heights[j])
            ):
                continue
            if (
                abs((boxes[i, 0] + boxes[i, 2] - boxes[j, 0] - boxes[j, 2]) / 2)
                > 0.05 * w
                or abs(boxes[i, 1] - boxes[j, 1]) > 0.05 * h
            ):
                continue
            overlap = np.maximum(
                0,
                np.minimum(boxes[i, 2:], boxes[j, 2:])
                - np.maximum(boxes[i, :2], boxes[j, :2]),
            )
            intersection = float(np.prod(overlap))
            union = widths[i] * heights[i] + widths[j] * heights[j] - intersection
            if union <= 0 or intersection / union < 0.85:
                continue
            shoulders = keypoints[[i, j]][:, [5, 6]]
            if not np.isfinite(shoulders).all() or (shoulders[:, :, 2] < 0.4).any():
                continue
            displacement = np.abs(shoulders[0, :, :2] - shoulders[1, :, :2])
            if (displacement[:, 0] > 0.15 * w).any() or (
                displacement[:, 1] > 0.1 * h
            ).any():
                continue
            removed.add(j)
    return detected[np.array([i not in removed for i in range(len(detected))])]


def suppress_aligned_torsos(detected):
    """Collapse nearly identical upper bodies with different box bottoms.

    Ordinary IoU NMS can retain both a person and their shorter torso box.
    Same-view containment is ambiguous, so require much closer head/width
    alignment than the separate cross-view merge. Geometry remains a heuristic,
    particularly when two people align almost perfectly under occlusion.
    """
    if len(detected) < 2:
        return detected
    boxes, scores = detected.xyxy, detected.confidence
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    areas = widths * heights
    suppressed = set()
    for outer in np.argsort(-areas, kind="stable"):
        if outer in suppressed:
            continue
        matches = []
        for inner in range(len(detected)):
            if inner == outer or inner in suppressed:
                continue
            if not 0.5 * heights[outer] <= heights[inner] <= 0.85 * heights[outer]:
                continue
            if min(widths[inner], widths[outer]) < 0.9 * max(
                widths[inner], widths[outer]
            ):
                continue
            if abs(centers[inner] - centers[outer]) > 0.05 * min(
                widths[inner], widths[outer]
            ):
                continue
            if abs(boxes[inner, 1] - boxes[outer, 1]) > 0.05 * heights[inner]:
                continue
            overlap = np.maximum(
                0,
                np.minimum(boxes[inner, 2:], boxes[outer, 2:])
                - np.maximum(boxes[inner, :2], boxes[outer, :2]),
            )
            if areas[inner] <= 0 or float(np.prod(overlap)) < 0.97 * areas[inner]:
                continue
            matches.append(inner)
        if not matches:
            continue
        # Decide the reference before suppressing anything. If a weak outer
        # loses to a stronger torso, it cannot remove any other matched box.
        if scores[outer] >= MIN_DETECTION_SCORE or scores[outer] >= max(
            scores[matches]
        ):
            suppressed.update(matches)
        else:
            suppressed.add(outer)
    # A suppressed box never participates as an outer reference, avoiding
    # transitive matches through an intermediate box.
    return detected[np.array([i not in suppressed for i in range(len(detected))])]
