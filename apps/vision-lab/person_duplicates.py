"""Conservative same-view duplicate filtering before person tracking."""

import numpy as np

from decisions import MIN_DETECTION_SCORE

PERSON_MERGE_POLICY = "STRICT_UPPER_BODY_ALIGNMENT_V1"


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
