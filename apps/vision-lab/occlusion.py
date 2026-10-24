"""Conservative image-space posture checks for the opt-in per-seat trial."""

import math

from wrist_trial import reliable

OCCLUSION_POLICY = "KNOWN_SEATED_LOWER_BODY_V1"


def supported_standing_legs(points, scores):
    # A bent torso alone can look seated. Require both thighs to project down,
    # plus a reliable lower leg of comparable projected length. This is a
    # camera-dependent posture cue, not an identification of a person's role.
    for hip, knee in ((11, 13), (12, 14)):
        if not all(reliable(points, scores, i) for i in (hip, knee)):
            return False
        dx = float(points[knee][0] - points[hip][0])
        dy = float(points[knee][1] - points[hip][1])
        if dy <= 0 or abs(dx) > 0.7 * dy:
            return False
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if not reliable(points, scores, ankle):
            continue
        a = [float(points[hip][i] - points[knee][i]) for i in (0, 1)]
        b = [float(points[ankle][i] - points[knee][i]) for i in (0, 1)]
        thigh, calf = math.hypot(*a), math.hypot(*b)
        if b[1] <= 0 or calf < 2 or not 0.65 <= thigh / calf <= 1.8:
            continue
        cosine = max(-1, min(1, sum(x * y for x, y in zip(a, b)) / (thigh * calf)))
        if math.degrees(math.acos(cosine)) >= 140:
            return True
    return False


def occluded_knees(observation, box, context, reference):
    """Flag potentially borrowed knees only for a previously pose-qualified seat.

    Overlap is not proof of visibility/identity. It cannot establish initial
    seating or bridge missing people. Current wrists are independently verified
    by the caller; rising hips/shoulders disable this exception immediately.
    """
    hip, upper = observation["hipAnchor"], observation.get("upperAnchor")
    if not reference or hip is None or upper is None:
        return []
    width = reference["box"][2] - reference["box"][0]
    height = reference["box"][3] - reference["box"][1]
    if (
        abs(hip[0] - reference["hip"][0]) > 0.5 * width
        or reference["hip"][1] - hip[1] > 0.15 * height
        or reference["upper"][1] - upper[1] > 0.15 * height
    ):
        return []
    blocked = set()
    for other in context:
        other_box, other_upper = other["box"], other["upperAnchor"]
        other_width = other_box[2] - other_box[0]
        other_height = other_box[3] - other_box[1]
        if (
            other_upper is None
            or other_upper[1] - upper[1] < 0.12 * min(box[3] - box[1], other_height)
            or min(box[2], other_box[2]) <= max(box[0], other_box[0])
            or min(box[3], other_box[3]) <= max(box[1], other_box[1])
        ):
            continue
        for i in (13, 14):
            x = float(observation["points"][i][0]) / observation["width"]
            y = float(observation["points"][i][1]) / observation["height"]
            if (
                other_box[0] - 0.25 * other_width
                <= x
                <= other_box[2] + 0.25 * other_width
                and max(other_upper[1], other_box[1]) <= y <= other_box[3]
            ):
                blocked.add(i)
    # One potentially borrowed knee makes the paired lower-body projection
    # ambiguous too. No points are reassigned from the other person.
    return [13, 14] if blocked else []


def recent_standing_ambiguity(observation, box, context, previous, seconds):
    if previous is None or seconds - previous["seconds"] > 2:
        return False
    hip, upper = observation["hipAnchor"], observation.get("upperAnchor")
    if hip is None or upper is None:
        return False
    width = previous["box"][2] - previous["box"][0]
    height = previous["box"][3] - previous["box"][1]
    if (
        abs(hip[0] - previous["hip"][0]) > 0.25 * width
        or abs(hip[1] - previous["hip"][1]) > 0.15 * height
        or abs(upper[1] - previous["upper"][1]) > 0.15 * height
    ):
        return False
    return any(
        other["upperAnchor"] is not None
        and abs(other["upperAnchor"][1] - upper[1])
        > 0.1 * min(height, other["box"][3] - other["box"][1])
        and min(box[2], other["box"][2]) > max(box[0], other["box"][0])
        and min(box[3], other["box"][3]) > max(box[1], other["box"][1])
        for other in context
    )
