"""Optional full-frame plus seat-area inference using the same verified model."""

import math

import numpy as np
import supervision as sv

from decisions import MAX_FOCUS_TABLES


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
    merged = sv.Detections.merge(parts)
    if len(merged):
        # Merge duplicate views before tracking so one person has one candidate box.
        merged = merged.with_nms(threshold=0.6, class_agnostic=True)
        merged = merged[np.argsort(-merged.confidence)[:30]]
    return merged, len(crops)
