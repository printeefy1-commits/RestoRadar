"""Optional full-frame plus seat-area inference using the same verified model."""

import math

import numpy as np
import supervision as sv

from decisions import MAX_FOCUS_TABLES, MIN_DETECTION_SCORE

FOCUS_MERGE_POLICY = "ALIGNED_CONTAINMENT_V1"


def merge_views(parts):
    """Prefer a corroborated longer body over a contained upper-body view.

    Containment alone can join different people during occlusion. Only compare
    different inference views with nearly the same upper-body position/width.
    Keep a selected box's own score; a confident torso cannot upgrade a weak
    full-body detection. These are camera-dependent heuristics, not identity.
    """
    merged = sv.Detections.merge(parts)
    if not len(merged):
        return merged
    views = np.concatenate([np.full(len(part), i) for i, part in enumerate(parts)])
    boxes = merged.xyxy
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    suppressed = set()
    # Direct comparisons only: a suppressed intermediate box cannot join two
    # otherwise distinct people through a transitive chain of overlaps.
    for outer in np.argsort(-areas, kind="stable"):
        if outer in suppressed or merged.confidence[outer] < MIN_DETECTION_SCORE:
            continue
        for inner in range(len(merged)):
            if inner == outer or inner in suppressed or views[inner] == views[outer]:
                continue
            if areas[inner] <= 0 or heights[inner] > 0.85 * heights[outer]:
                continue
            if min(widths[inner], widths[outer]) < 0.65 * max(
                widths[inner], widths[outer]
            ):
                continue
            if abs(centers_x[inner] - centers_x[outer]) > 0.2 * min(
                widths[inner], widths[outer]
            ):
                continue
            if abs(boxes[inner, 1] - boxes[outer, 1]) > 0.2 * heights[inner]:
                continue
            overlap = np.maximum(
                0,
                np.minimum(boxes[inner, 2:], boxes[outer, 2:])
                - np.maximum(boxes[inner, :2], boxes[outer, :2]),
            )
            if float(np.prod(overlap)) >= 0.95 * areas[inner]:
                suppressed.add(inner)
    merged = merged[np.array([i not in suppressed for i in range(len(merged))])]
    # Similar complete-body views still use ordinary confidence-based NMS.
    merged = merged.with_nms(threshold=0.6, class_agnostic=True)
    return merged[np.argsort(-merged.confidence, kind="stable")[:30]]


def seat_crops(regions, width, height):
    grouped = {}
    for region in regions:
        grouped.setdefault(region["tableId"], []).append(region)
    if len(grouped) > MAX_FOCUS_TABLES:
        raise ValueError("Focused detection supports at most four tables per trial")
    crops = []
    for table_id in sorted(grouped):
        seats = grouped[table_id]
        # Include context above hips for heads/shoulders and below for legs.
        x1 = max(0, math.floor((min(r["x"] for r in seats) - 0.05) * width))
        y1 = max(0, math.floor((min(r["y"] for r in seats) - 0.15) * height))
        x2 = min(
            width, math.ceil((max(r["x"] + r["width"] for r in seats) + 0.05) * width)
        )
        y2 = min(
            height,
            math.ceil((max(r["y"] + r["height"] for r in seats) + 0.05) * height),
        )
        crop = (x1, y1, x2, y2)
        if x2 > x1 and y2 > y1 and crop != (0, 0, width, height) and crop not in crops:
            crops.append(crop)
    return crops


def detect_with_seat_focus(detector, frame, regions):
    height, width = frame.shape[:2]
    crops = seat_crops(regions, width, height)
    parts = [detector(frame)]
    for x1, y1, x2, y2 in crops:
        detected = detector(frame[y1:y2, x1:x2])
        boxes = detected.xyxy.copy()
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
        parts.append(
            sv.Detections(
                xyxy=boxes,
                confidence=detected.confidence,
                class_id=detected.class_id,
            )
        )
    return merge_views(parts), len(crops)
