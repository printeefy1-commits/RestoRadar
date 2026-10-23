"""Optional image-space wrist cue. Never confirms seating, touch or departure."""

import math
from collections import Counter

from decisions import MIN_DETECTION_SCORE, MIN_KEYPOINT_SCORE, matching_tables

WRIST_POLICY = {
    "version": "WRIST_SEAT_DWELL_V3",
    "startDetectionScore": MIN_DETECTION_SCORE,
    "continueDetectionScore": 0.30,
    "maxLowScoreSeconds": 1.0,
    "motionReference": "SHOULDERS_THEN_BOX_CENTER",
    "overlapResolution": "PREVIOUS_SEAT_WHILE_STATIONARY",
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
        "motionAnchor": None,
        "motionSource": None,
        "seatRegionCandidates": [],
        "seatResolution": None,
        "stableSeconds": 0,
        "detectionScore": float(score) if math.isfinite(score) else None,
        "scoreContinuation": False,
        "lowScoreSeconds": 0,
    }
    if track_id is None:
        return {**result, "reason": "NO_TRACK"}
    if not math.isfinite(score) or score < WRIST_POLICY["continueDetectionScore"]:
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
    # Motion and seat mapping have different references. Stable shoulders
    # survive a full-body/torso box change and a fluctuating hip score.
    shoulders = all(reliable(points, scores, i) for i in (5, 6))
    motion = (
        [(float(points[5][i]) + float(points[6][i])) / 2 for i in (0, 1)]
        if shoulders
        else [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    )
    result.update(
        motionAnchor=[float(motion[0] / width), float(motion[1] / height)],
        motionSource="SHOULDERS" if shoulders else "BOX_CENTER",
    )
    matches = [r for r in seats if matching_tables(anchor, [r], width, height)]
    # Mixed-table ambiguity must never be resolved by history.
    if all(r["tableId"] == table_id for r in matches):
        result["seatRegionCandidates"] = [r["id"] for r in matches]
    if len(matches) != 1:
        return {**result, "reason": "AMBIGUOUS_SEAT" if matches else "NO_SEAT_MATCH"}
    result["seatRegionId"] = matches[0]["id"]
    if matches[0]["tableId"] != table_id:
        return {**result, "reason": "TABLE_MISMATCH"}
    return {
        **result,
        "state": "CANDIDATE" if score >= MIN_DETECTION_SCORE else "UNKNOWN",
        "reason": "DWELLING" if score >= MIN_DETECTION_SCORE else "LOW_DETECTION_SCORE",
        "seatResolution": "DIRECT",
    }


def stationary(previous, person):
    cue, box = person["wristEvidence"], person["box"]
    ref, anchor = previous["box"], cue["motionAnchor"]
    if cue["motionSource"] != previous["motionSource"]:
        return False
    dx = WRIST_POLICY["maxDriftBoxWidths"] * (ref[2] - ref[0])
    dy = WRIST_POLICY["maxDriftBoxHeights"] * (ref[3] - ref[1])
    if dx <= 0 or dy <= 0 or any(not math.isfinite(v) for v in [*box, *anchor]):
        return False
    # Always compare with the first position, never accumulate small steps.
    if (
        abs(anchor[0] - previous["anchor"][0]) > dx
        or abs(anchor[1] - previous["anchor"][1]) > dy
    ):
        return False
    if (
        abs((box[0] + box[2] - ref[0] - ref[2]) / 2) > dx
        or not 0.75 <= (box[2] - box[0]) / (ref[2] - ref[0]) <= 1.33
    ):
        return False
    if cue["motionSource"] == "BOX_CENTER":
        # Without two reliable shoulders retain the conservative height guard.
        return 0.75 <= (box[3] - box[1]) / (ref[3] - ref[1]) <= 1.33
    # Shoulder motion, horizontal position/width and the top edge must stay
    # stable. Only the bottom edge may change as body/torso crops alternate.
    return abs(box[1] - ref[1]) <= dy


def can_continue_score(previous, person, seconds):
    cue = person["wristEvidence"]
    score = cue["detectionScore"]
    return (
        previous is not None
        and previous["qualified"]
        and score is not None
        and math.isfinite(score)
        and WRIST_POLICY["continueDetectionScore"] <= score < MIN_DETECTION_SCORE
        and 0
        < seconds - previous["lastStrongSeconds"]
        <= WRIST_POLICY["maxLowScoreSeconds"]
        and cue["motionSource"] == "SHOULDERS"
        and stationary(previous, person)
    )


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
        for person in people:
            cue = person["wristEvidence"]
            if (
                cue["reason"] == "LOW_DETECTION_SCORE"
                and cue["seatRegionId"] is not None
            ):
                key = (person["trackId"], cue["seatRegionId"], cue["tableId"])
                history = self.candidates.get(key)
                if can_continue_score(history, person, seconds):
                    cue.update(
                        state="CANDIDATE",
                        reason="DWELLING",
                        scoreContinuation=True,
                        lowScoreSeconds=round(
                            seconds - history["lastStrongSeconds"], 3
                        ),
                    )
            if cue["reason"] != "AMBIGUOUS_SEAT":
                continue
            previous = [
                (key, history)
                for key, history in self.candidates.items()
                if key[0] == person["trackId"]
                and key[2] == cue["tableId"]
                and key[1] in cue["seatRegionCandidates"]
            ]
            # No cold-start tie-breaking, cross-table carry or missing-frame
            # bridging. Only an immediately preceding stationary track can keep
            # its already established seat when the current regions overlap.
            if len(previous) != 1 or not stationary(previous[0][1], person):
                continue
            history = previous[0][1]
            strong = cue["detectionScore"] >= MIN_DETECTION_SCORE
            continued = can_continue_score(history, person, seconds)
            if strong or continued:
                cue.update(
                    state="CANDIDATE",
                    reason="DWELLING",
                    seatRegionId=previous[0][0][1],
                    seatResolution="CONTINUITY",
                    scoreContinuation=continued,
                    lowScoreSeconds=(
                        round(seconds - history["lastStrongSeconds"], 3)
                        if continued
                        else 0
                    ),
                )
        valid = [p for p in people if p["wristEvidence"]["state"] == "CANDIDATE"]
        seats = Counter(p["wristEvidence"]["seatRegionId"] for p in valid)
        tracks = Counter(p["trackId"] for p in valid)
        current = {}
        states = {table: "UNKNOWN" for table in self.table_ids}
        for person in valid:
            cue = person["wristEvidence"]
            if seats[cue["seatRegionId"]] > 1 or tracks[person["trackId"]] > 1:
                cue.update(
                    state="UNKNOWN",
                    reason="SEAT_CONFLICT",
                    scoreContinuation=False,
                    lowScoreSeconds=0,
                )
                continue
            key = (person["trackId"], cue["seatRegionId"], cue["tableId"])
            previous = self.candidates.get(key)
            if previous and previous["motionSource"] != cue["motionSource"]:
                previous = None
            if previous and not stationary(previous, person):
                cue.update(state="UNKNOWN", reason="MOVING")
                continue
            if not previous:
                previous = {
                    "start": seconds,
                    "box": person["box"][:],
                    "anchor": cue["motionAnchor"][:],
                    "motionSource": cue["motionSource"],
                    "lastStrongSeconds": seconds,
                    "qualified": False,
                }
            # Never refresh the strong-score clock from a low-score sample.
            # This bounds a dip even when a detector keeps emitting weak boxes.
            previous = {
                **previous,
                "lastStrongSeconds": (
                    seconds
                    if cue["detectionScore"] >= MIN_DETECTION_SCORE
                    else previous["lastStrongSeconds"]
                ),
                "qualified": previous["qualified"]
                or seconds - previous["start"] >= WRIST_POLICY["seconds"],
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
