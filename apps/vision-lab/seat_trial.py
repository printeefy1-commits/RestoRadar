"""Opt-in per-seat evidence. Never emits live arrivals, departures or sessions."""

import math
from collections import Counter

from decisions import (
    MIN_DETECTION_SCORE,
    MIN_KEYPOINT_SCORE,
    duration_reached,
    matching_tables,
)
from wrist_trial import reliable, visibly_standing, WristTrial, wrist_observation
from occlusion import (
    OCCLUSION_POLICY,
    occluded_knees,
    supported_standing_legs,
    recent_standing_ambiguity,
)

SEAT_POLICY = {
    "version": "SEAT_REVALIDATION_V2",
    "occlusionPolicy": OCCLUSION_POLICY,
    "wristSupportPolicy": "WRIST_SEAT_DWELL_V4",
    "seatingSeconds": 5,
    "minDetectionScore": MIN_DETECTION_SCORE,
    "minKeypointScore": MIN_KEYPOINT_SCORE,
    "maxRevalidationSeconds": 2.0,
    "maxSampleGapSeconds": 0.8,
    "initialHipDriftBoxWidths": 0.25,
    "initialHipDriftBoxHeights": 0.15,
    "revalidationRequires": "CURRENT_SEATED_POSE_SAME_TRACK_AND_SEAT",
}


def within_duration(start, now, duration):
    elapsed = now - start
    tolerance = math.ulp(start) + math.ulp(now) + math.ulp(duration)
    return elapsed <= duration or math.isclose(
        elapsed, duration, rel_tol=0, abs_tol=tolerance
    )


def seat_observation(points, scores, detection_score, regions, width, height):
    # Both hips are required for the new pose path and revalidation. A current
    # qualified wrist cue may independently supply its already verified seat.
    hips = all(reliable(points, scores, i) for i in (11, 12))
    anchor = (
        [(float(points[11][i]) + float(points[12][i])) / 2 for i in (0, 1)]
        if hips
        else None
    )
    return {
        "points": points,
        "scores": scores,
        "width": width,
        "height": height,
        "upperAnchor": (
            [
                (float(points[5][0]) + float(points[6][0])) / (2 * width),
                (float(points[5][1]) + float(points[6][1])) / (2 * height),
            ]
            if all(reliable(points, scores, i) for i in (5, 6))
            else None
        ),
        "score": (
            float(detection_score)
            if math.isfinite(detection_score) and 0 <= detection_score <= 1
            else None
        ),
        "standing": visibly_standing(points, scores),
        "hipAnchor": [anchor[0] / width, anchor[1] / height] if anchor else None,
        "matches": [
            r["id"]
            for r in regions
            if anchor and matching_tables(anchor, [r], width, height)
        ],
    }


class SeatTrial:
    def __init__(self, regions, tabletops=None):
        self.regions = {r["id"]: r for r in regions}
        self.tabletops = tabletops or []
        self.wrists = (
            WristTrial({r["tableId"] for r in regions}) if self.tabletops else None
        )
        self.history = {}
        self.standing_recent = {}
        self.last_time = None

    def observe(self, seconds, people, observations, occluders=None):
        if not math.isfinite(seconds) or (
            self.last_time is not None and seconds <= self.last_time
        ):
            raise ValueError("Video timestamps must increase")
        if len(people) != len(observations):
            raise ValueError("Seat observations must match current people")
        if self.last_time is not None and not within_duration(
            self.last_time, seconds, SEAT_POLICY["maxSampleGapSeconds"]
        ):
            self.history.clear()
            self.standing_recent.clear()
        self.last_time = seconds
        observations = [dict(o) for o in observations]
        context = (
            occluders
            if occluders is not None
            else [
                {"box": p["box"], "upperAnchor": o.get("upperAnchor")}
                for p, o in zip(people, observations)
                if o["score"] is not None and o["score"] >= MIN_DETECTION_SCORE
            ]
        )
        support = []
        standing_current = {}
        for person, obs in zip(people, observations):
            previous = self.history.get(person["trackId"], {})
            same_seat = (
                previous.get("seat") in obs["matches"]
                and len(obs["matches"]) == 1
                and previous.get("qualifiedAt") is not None
                and within_duration(
                    previous["lastValid"],
                    seconds,
                    SEAT_POLICY["maxRevalidationSeconds"],
                )
            )
            blocked = occluded_knees(
                obs,
                person["box"],
                context,
                previous.get("poseReference") if same_seat else None,
            )
            scores = obs["scores"].copy()
            for i in blocked:
                scores[i] = 0
            torso_standing = visibly_standing(obs["points"], scores)
            leg_standing = supported_standing_legs(obs["points"], scores)
            obs["standing"] = torso_standing or leg_standing
            obs["postureGuard"] = (
                "TORSO_ANGLE"
                if torso_standing
                else (
                    "LEG_CHAIN"
                    if leg_standing
                    else "OCCLUDED_LOWER_BODY" if blocked else "NONE"
                )
            )
            recent = self.standing_recent.get(person["trackId"])
            if (
                obs["standing"]
                and obs["score"] is not None
                and obs["score"] >= MIN_DETECTION_SCORE
                and obs["hipAnchor"] is not None
                and obs["upperAnchor"] is not None
            ):
                standing_current[person["trackId"]] = {
                    "seconds": seconds,
                    "hip": obs["hipAnchor"][:],
                    "upper": obs["upperAnchor"][:],
                    "box": person["box"][:],
                }
            elif (
                not obs["standing"]
                and not (
                    person.get("evidence") == "ACCEPTED"
                    and all(reliable(obs["points"], obs["scores"], i) for i in (13, 14))
                )
                and recent_standing_ambiguity(
                    obs, person["box"], context, recent, seconds
                )
            ):
                obs["standing"] = True
                obs["postureGuard"] = "RECENT_STANDING"
                standing_current[person["trackId"]] = recent
            copy = dict(person)
            if self.wrists:
                copy["wristEvidence"] = wrist_observation(
                    obs["points"],
                    scores,
                    [
                        person["box"][i]
                        * (obs["width"] if i % 2 == 0 else obs["height"])
                        for i in range(4)
                    ],
                    person["trackId"],
                    obs["score"] if obs["score"] is not None else float("nan"),
                    list(self.regions.values()),
                    self.tabletops,
                    obs["width"],
                    obs["height"],
                )
                if obs["standing"]:
                    copy["wristEvidence"].update(
                        state="UNKNOWN", reason="STANDING", seatRegionId=None
                    )
            support.append(copy)
        if self.wrists:
            self.wrists.observe(seconds, support)
        self.standing_recent = standing_current
        entries = []
        for person, obs in zip(support, observations):
            wrist = person.get("wristEvidence", {})
            strong = obs["score"] is not None and obs["score"] >= MIN_DETECTION_SCORE
            wrist_seat = self.regions.get(wrist.get("seatRegionId"))
            wrist_ready = (
                strong
                and not obs["standing"]
                and wrist.get("state") == "PROBABLE"
                and wrist.get("stableSeconds", 0) >= SEAT_POLICY["seatingSeconds"]
                and wrist_seat is not None
                and wrist_seat["tableId"] == wrist.get("tableId")
                and (obs["hipAnchor"] is None or wrist_seat["id"] in obs["matches"])
            )
            selected = (
                obs["matches"][0]
                if len(obs["matches"]) == 1
                else wrist_seat["id"] if wrist_ready else None
            )
            claims = (
                []
                if obs["standing"]
                or not strong
                or not (
                    person.get("evidence") == "ACCEPTED"
                    or wrist.get("state") in ("CANDIDATE", "PROBABLE")
                )
                else [selected] if wrist_ready else obs["matches"]
            )
            entries.append((person, obs, strong, wrist_ready, selected, claims))
        seats = Counter(seat for *_, claims in entries for seat in claims)
        tracks = Counter(p["trackId"] for p in people if p["trackId"] is not None)
        current = {}
        counts = {r["tableId"]: 0 for r in self.regions.values()}
        for original, (person, obs, strong, wrist_ready, seat_id, claims) in zip(
            people, entries
        ):
            track = person["trackId"]
            region = self.regions.get(seat_id)
            cue = {
                "state": "UNKNOWN",
                "reason": "POSE_UNCERTAIN",
                "source": None,
                "tableId": region["tableId"] if region else None,
                "seatRegionId": seat_id,
                "hipAnchor": obs["hipAnchor"],
                "detectionScore": obs["score"],
                "poseSeconds": 0,
                "qualifiedAtSeconds": None,
                "qualifiedBy": None,
                "resumed": False,
                "revalidationPending": False,
                "gapSeconds": 0,
                "postureGuard": obs["postureGuard"],
                "wristState": person.get("wristEvidence", {}).get("state", "UNKNOWN"),
                "wristReason": person.get("wristEvidence", {}).get(
                    "reason", "WRIST_UNCERTAIN"
                ),
                "wristStableSeconds": person.get("wristEvidence", {}).get(
                    "stableSeconds", 0
                ),
            }
            original["seatEvidence"] = cue
            if track is None:
                cue["reason"] = "NO_TRACK"
                continue
            if obs["standing"]:
                cue["reason"] = (
                    "POSTURE_UNCERTAIN"
                    if obs["postureGuard"] == "RECENT_STANDING"
                    else "STANDING"
                )
                continue
            if obs["score"] is None:
                cue["reason"] = "LOW_DETECTION_SCORE"
                continue
            if tracks[track] > 1 or any(seats[s] > 1 for s in claims):
                cue["reason"] = "SEAT_CONFLICT"
                continue
            if region is None:
                cue["reason"] = (
                    "HIP_UNCERTAIN"
                    if obs["hipAnchor"] is None
                    else (
                        "AMBIGUOUS_SEAT" if len(obs["matches"]) > 1 else "NO_SEAT_MATCH"
                    )
                )
                continue
            previous = self.history.get(track)
            if previous and (
                previous["seat"] != seat_id or previous["table"] != region["tableId"]
            ):
                previous = None
            if (
                previous
                and previous["qualifiedAt"] is not None
                and not within_duration(
                    previous["lastValid"],
                    seconds,
                    SEAT_POLICY["maxRevalidationSeconds"],
                )
            ):
                previous = None
            qualified = previous is not None and previous["qualifiedAt"] is not None
            pose_ready = (
                strong
                and obs["hipAnchor"] is not None
                and person.get("evidence") == "ACCEPTED"
                and person.get("tableId") == region["tableId"]
                and len(obs["matches"]) == 1
            )
            if not pose_ready and not wrist_ready:
                cue["reason"] = (
                    "LOW_DETECTION_SCORE"
                    if not strong
                    else (
                        "WRIST_DWELLING"
                        if person.get("wristEvidence", {}).get("state") == "CANDIDATE"
                        else "POSE_UNCERTAIN"
                    )
                )
                # Retention is invisible to the count and cannot establish an
                # initial dwell. Missing/ambiguous hips cannot retain history.
                if (
                    qualified
                    and obs["hipAnchor"] is not None
                    and len(obs["matches"]) == 1
                ):
                    current[track] = {**previous, "poseStart": None, "pending": True}
                    cue.update(
                        revalidationPending=True,
                        qualifiedAtSeconds=previous["qualifiedAt"],
                        qualifiedBy=previous["qualifiedBy"],
                        gapSeconds=round(seconds - previous["lastValid"], 3),
                    )
                elif (
                    strong
                    and person.get("wristEvidence", {}).get("state") == "CANDIDATE"
                ):
                    cue["state"] = "CANDIDATE"
                continue
            record = (
                dict(previous)
                if previous
                else {
                    "seat": seat_id,
                    "table": region["tableId"],
                    "poseStart": None,
                    "qualifiedAt": None,
                    "qualifiedBy": None,
                    "lastValid": None,
                    "pending": False,
                }
            )
            resumed = qualified and previous["pending"] and pose_ready
            moving = False
            if pose_ready:
                if record["poseStart"] is not None and not qualified:
                    anchor, ref = obs["hipAnchor"], record["poseAnchor"]
                    box = record["poseBox"]
                    moving = abs(anchor[0] - ref[0]) > SEAT_POLICY[
                        "initialHipDriftBoxWidths"
                    ] * (box[2] - box[0]) or abs(anchor[1] - ref[1]) > SEAT_POLICY[
                        "initialHipDriftBoxHeights"
                    ] * (
                        box[3] - box[1]
                    )
                if record["poseStart"] is None or moving:
                    record.update(
                        poseStart=seconds,
                        poseAnchor=obs["hipAnchor"][:],
                        poseBox=person["box"][:],
                    )
                cue["poseSeconds"] = round(seconds - record["poseStart"], 3)
            else:
                record["poseStart"] = None
            pose_qualified = pose_ready and duration_reached(
                record["poseStart"], seconds, SEAT_POLICY["seatingSeconds"]
            )
            if (
                pose_qualified
                and "poseReference" not in record
                and obs.get("upperAnchor") is not None
            ):
                record["poseReference"] = {
                    "hip": obs["hipAnchor"][:],
                    "upper": obs["upperAnchor"][:],
                    "box": person["box"][:],
                }
            cue["source"] = (
                "BOTH"
                if pose_ready and wrist_ready
                else "POSE" if pose_ready else "WRIST"
            )
            if pose_qualified or wrist_ready or (qualified and pose_ready):
                if record["qualifiedAt"] is None:
                    record.update(
                        qualifiedAt=seconds,
                        qualifiedBy=(
                            "BOTH"
                            if pose_qualified and wrist_ready
                            else "POSE" if pose_qualified else "WRIST"
                        ),
                    )
                gap = round(seconds - record["lastValid"], 3) if resumed else 0
                record.update(lastValid=seconds, pending=False)
                cue.update(
                    state="PROBABLE",
                    reason=(
                        "REVALIDATED"
                        if resumed
                        else (
                            "BOTH_QUALIFIED"
                            if pose_qualified and wrist_ready
                            else "WRIST_QUALIFIED" if wrist_ready else "POSE_QUALIFIED"
                        )
                    ),
                    qualifiedAtSeconds=record["qualifiedAt"],
                    qualifiedBy=record["qualifiedBy"],
                    resumed=resumed,
                    gapSeconds=gap,
                )
                counts[region["tableId"]] += 1
            else:
                cue.update(
                    state="CANDIDATE", reason="POSE_MOVING" if moving else "DWELLING"
                )
            current[track] = record
        # A missing ID is a hard break, including during the revalidation window.
        self.history = current
        return counts
