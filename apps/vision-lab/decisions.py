"""Conservative trial decisions. No detector miss is evidence of an empty seat."""

import math

MIN_DETECTION_SCORE = 0.35
MIN_KEYPOINT_SCORE = 0.4
MAX_FOCUS_TABLES = 4
DURATION_COMPARISON = "FLOAT_ULP_TOLERANCE_V1"


def duration_reached(start, now, duration):
    """Ignore only floating-point representation error at a source-time boundary."""
    elapsed = now - start
    tolerance = math.ulp(start) + math.ulp(now) + math.ulp(duration)
    return elapsed >= duration or math.isclose(
        elapsed, duration, rel_tol=0, abs_tol=tolerance
    )


def seated_anchor(points, scores, threshold=MIN_KEYPOINT_SCORE):
    # COCO body indices: shoulders 5/6, hips 11/12, knees 13/14.
    # A bent torso/thigh angle is a provisional, camera-dependent posture cue.
    anchors = []
    for shoulder, hip, knee in ((5, 11, 13), (6, 12, 14)):
        if min(scores[shoulder], scores[hip], scores[knee]) < threshold:
            continue
        a = [points[shoulder][i] - points[hip][i] for i in (0, 1)]
        b = [points[knee][i] - points[hip][i] for i in (0, 1)]
        denominator = math.hypot(*a) * math.hypot(*b)
        if denominator < 4:
            continue
        angle = math.degrees(
            math.acos(max(-1, min(1, sum(x * y for x, y in zip(a, b)) / denominator)))
        )
        if 35 <= angle <= 140:
            anchors.append(points[hip])
    if not anchors:
        return None
    return [sum(float(p[i]) for p in anchors) / len(anchors) for i in (0, 1)]


def matching_tables(anchor, regions, width, height):
    if anchor is None:
        return set()
    x, y = anchor[0] / width, anchor[1] / height
    return {
        r["tableId"]
        for r in regions
        if r["x"] <= x <= r["x"] + r["width"] and r["y"] <= y <= r["y"] + r["height"]
    }


def mapped_table(anchor, regions, width, height):
    matches = matching_tables(anchor, regions, width, height)
    # Ambiguous mappings cannot assign a person to two tables.
    return next(iter(matches)) if len(matches) == 1 else None


def seating_evidence(anchor, regions, width, height, detection_score):
    if anchor is None:
        return None, "POSE_UNCERTAIN"
    matches = matching_tables(anchor, regions, width, height)
    if not matches:
        return None, "UNMAPPED"
    if len(matches) > 1:
        return None, "AMBIGUOUS_REGIONS"
    table_id = next(iter(matches))
    if not math.isfinite(detection_score) or detection_score < MIN_DETECTION_SCORE:
        return table_id, "LOW_DETECTION_SCORE"
    return table_id, "ACCEPTED"


class SeatingVerifier:
    def __init__(self, table_ids):
        self.table_ids = set(table_ids)
        self.candidates = {}
        self.confirmed = set()
        self.last_time = None

    def observe(self, seconds, occupied_tables):
        if not math.isfinite(seconds) or (
            self.last_time is not None and seconds <= self.last_time
        ):
            raise ValueError("Video timestamps must increase")
        if self.last_time is not None and seconds - self.last_time > 0.8:
            self.candidates.clear()
        self.last_time = seconds
        events = []
        states = {}
        for table_id in self.table_ids:
            if table_id not in occupied_tables:
                self.candidates.pop(table_id, None)
                states[table_id] = "UNKNOWN"
                continue
            self.candidates.setdefault(table_id, seconds)
            verified = duration_reached(self.candidates[table_id], seconds, 5.0)
            states[table_id] = "OCCUPIED" if verified else "UNKNOWN"
            if verified and table_id not in self.confirmed:
                self.confirmed.add(table_id)
                events.append(
                    {
                        "tableId": table_id,
                        "kind": "SEATING_CONFIRMED",
                        "observedSeconds": self.candidates[table_id],
                        "confirmedSeconds": seconds,
                    }
                )
        # This trial deliberately never confirms departure or closes a live session.
        return states, events
