"""Optional image-space wrist cue. Never confirms seating, touch or departure."""

import math
from collections import Counter

from decisions import MIN_DETECTION_SCORE, MIN_KEYPOINT_SCORE, matching_tables

WRIST_POLICY = {
    "version": "WRIST_SEAT_DWELL_V1",
    "seconds": 5,
    "minKeypointScore": MIN_KEYPOINT_SCORE,
    "maxDriftBoxWidths": 0.25,
    "maxDriftBoxHeights": 0.15,
}


def reliable(points, scores, index):
    return (
        math.isfinite(float(scores[index]))
        and scores[index] >= MIN_KEYPOINT_SCORE
        and all(math.isfinite(float(v)) for v in points[index])
    )


def visibly_standing(points, scores):
    for shoulder, hip, knee in ((5, 11, 13), (6, 12, 14)):
        if not all(reliable(points, scores, i) for i in (shoulder, hip, knee)):
            continue
        a = [float(points[shoulder][i] - points[hip][i]) for i in (0, 1)]
        b = [float(points[knee][i] - points[hip][i]) for i in (0, 1)]
        denominator = math.hypot(*a) * math.hypot(*b)
        if denominator < 4:
            continue
        cosine = max(-1, min(1, sum(x * y for x, y in zip(a, b)) / denominator))
        if math.degrees(math.acos(cosine)) >= 160:
            return True
    return False


def wrist_observation(points, scores, box, track_id, score, seats, tops, width, height):
    result = {
        "state": "UNKNOWN",
        "reason": "WRIST_UNCERTAIN",
        "tableId": None,
        "seatRegionId": None,
        "bodyAnchor": None,
        "anchorSource": None,
        "stableSeconds": 0,
    }
    if track_id is None:
        return {**result, "reason": "NO_TRACK"}
    if not math.isfinite(score) or score < MIN_DETECTION_SCORE:
        return {**result, "reason": "LOW_DETECTION_SCORE"}
    if visibly_standing(points, scores):
        return {**result, "reason": "STANDING"}
    wrists = [points[i] for i in (9, 10) if reliable(points, scores, i)]
    if not wrists:
        return result
    tables = set().union(*(matching_tables(p, tops, width, height) for p in wrists))
    if len(tables) != 1:
        return {
            **result,
            "reason": "AMBIGUOUS_TABLETOP" if tables else "NO_TABLETOP_MATCH",
        }
    table_id = next(iter(tables))
    result["tableId"] = table_id
    hips = [points[i] for i in (11, 12) if reliable(points, scores, i)]
    anchor = (
        [sum(float(p[i]) for p in hips) / len(hips) for i in (0, 1)]
        if hips
        else [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    )
    result.update(
        bodyAnchor=[float(anchor[0] / width), float(anchor[1] / height)],
        anchorSource="HIP" if hips else "BOX_CENTER",
    )
    matches = [r for r in seats if matching_tables(anchor, [r], width, height)]
    if len(matches) != 1:
        return {**result, "reason": "AMBIGUOUS_SEAT" if matches else "NO_SEAT_MATCH"}
    result["seatRegionId"] = matches[0]["id"]
    if matches[0]["tableId"] != table_id:
        return {**result, "reason": "TABLE_MISMATCH"}
    return {**result, "state": "CANDIDATE", "reason": "DWELLING"}


class WristTrial:
    def __init__(self, table_ids):
        self.table_ids = set(table_ids)
        self.candidates = {}
        self.last_time = None

    def observe(self, seconds, people):
        if not math.isfinite(seconds) or (
            self.last_time is not None and seconds <= self.last_time
        ):
            raise ValueError("Video timestamps must increase")
        if self.last_time is not None and seconds - self.last_time > 0.8:
            self.candidates.clear()
        self.last_time = seconds
        valid = [p for p in people if p["wristEvidence"]["state"] == "CANDIDATE"]
        seats = Counter(p["wristEvidence"]["seatRegionId"] for p in valid)
        tracks = Counter(p["trackId"] for p in valid)
        current = {}
        states = {table: "UNKNOWN" for table in self.table_ids}
        for person in valid:
            cue = person["wristEvidence"]
            if seats[cue["seatRegionId"]] > 1 or tracks[person["trackId"]] > 1:
                cue.update(state="UNKNOWN", reason="SEAT_CONFLICT")
                continue
            key = (
                person["trackId"],
                cue["seatRegionId"],
                cue["tableId"],
                cue["anchorSource"],
            )
            previous = self.candidates.get(key)
            box = person["box"]  # Normalized coordinates in the source image.
            if previous:
                ref = previous["box"]
                dx = WRIST_POLICY["maxDriftBoxWidths"] * (ref[2] - ref[0])
                dy = WRIST_POLICY["maxDriftBoxHeights"] * (ref[3] - ref[1])
                anchor = cue["bodyAnchor"]
                # Compare with the first sample, not the previous step: slow
                # walking must not accumulate a stationary five-second dwell.
                moving = (
                    abs(anchor[0] - previous["anchor"][0]) > dx
                    or abs(anchor[1] - previous["anchor"][1]) > dy
                    or abs((box[0] + box[2] - ref[0] - ref[2]) / 2) > dx
                    or abs((box[1] + box[3] - ref[1] - ref[3]) / 2) > dy
                    or not 0.75 <= (box[2] - box[0]) / (ref[2] - ref[0]) <= 1.33
                    or not 0.75 <= (box[3] - box[1]) / (ref[3] - ref[1]) <= 1.33
                )
                if moving:
                    cue.update(state="UNKNOWN", reason="MOVING")
                    continue
            else:
                previous = {
                    "start": seconds,
                    "box": box[:],
                    "anchor": cue["bodyAnchor"][:],
                }
            current[key] = previous
            cue["stableSeconds"] = round(seconds - previous["start"], 3)
            if seconds - previous["start"] >= WRIST_POLICY["seconds"]:
                cue.update(state="PROBABLE", reason="PROBABLE")
                states[cue["tableId"]] = "PROBABLE"
        # A missing/weak wrist, changed track, missing frame or conflict drops
        # that candidate. There is no last-known occupied state or arrival event.
        self.candidates = current
        return states
