"""Occluded seated evidence must not become a stale occupied-seat lock."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occlusion import supported_standing_legs
from seat_trial import SeatTrial, seat_observation


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.regions = [
            {
                "id": "seat",
                "tableId": "table",
                "x": 0.3,
                "y": 0.3,
                "width": 0.3,
                "height": 0.55,
            }
        ]
        self.tops = [
            {
                "id": "top",
                "tableId": "table",
                "x": 0.48,
                "y": 0.35,
                "width": 0.22,
                "height": 0.12,
            }
        ]
        self.context = [{"box": [0.38, 0.4, 0.55, 0.9], "upperAnchor": [0.43, 0.45]}]

    def person(
        self, track=1, pose=True, false_legs=False, rising=False, wrists=True, score=0.9
    ):
        points = np.full((17, 2), 400.0, dtype=np.float32)
        scores = np.full(17, 0.9, dtype=np.float32)
        points[5:7] = [[400, 300], [440, 300]]
        points[11:13] = [[400, 500], [440, 500]]
        points[13:15] = [[550, 500], [590, 500]]
        points[15:17] = [[550, 750], [590, 750]]
        points[9:11] = [[520, 400], [540, 410]]
        if false_legs:
            points[13:15] = [[410, 650], [450, 650]]
            points[15:17] = [[410, 800], [450, 800]]
        if rising:
            points[5:7, 1] = 160
            points[11:13, 1] = 340
        if not wrists:
            scores[9:11] = 0.1
        p = {
            "trackId": track,
            "score": score,
            "box": [0.35, 0.25, 0.62, 0.8],
            "evidence": "ACCEPTED" if pose else "POSE_UNCERTAIN",
            "tableId": "table" if pose else None,
            "wristEvidence": {
                "state": "UNKNOWN",
                "reason": "STANDING",
                "stableSeconds": 0,
            },
        }
        return p, seat_observation(points, scores, score, self.regions, 1000, 1000)

    def step(self, trial, t, pair, context=None):
        p, obs = pair
        return trial.observe(
            t, [p], [obs], self.context if context is None else context
        )

    def qualified(self, wrists=True):
        trial = SeatTrial(self.regions, self.tops if wrists else [])
        for i in range(11):
            pair = self.person()
            count = self.step(trial, i / 2, pair, [])
        self.assertEqual(count, {"table": 1})
        return trial

    def test_current_wrists_support_previously_seated_person_when_knees_overlap(self):
        trial = self.qualified()
        for i in range(11, 20):
            pair = self.person(pose=False, false_legs=True)
            raw_points, raw_scores = pair[1]["points"].copy(), pair[1]["scores"].copy()
            original = deepcopy(pair[0]["wristEvidence"])
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 1})
            cue = pair[0]["seatEvidence"]
            self.assertEqual(cue["postureGuard"], "OCCLUDED_LOWER_BODY")
            self.assertEqual(cue["source"], "WRIST")
            self.assertEqual(cue["wristState"], "PROBABLE")
            self.assertEqual(pair[0]["wristEvidence"], original)
            np.testing.assert_array_equal(pair[1]["points"], raw_points)
            np.testing.assert_array_equal(pair[1]["scores"], raw_scores)

    def test_occlusion_cannot_cold_start_or_establish_initial_qualification(self):
        trial = SeatTrial(self.regions, self.tops)
        for i in range(20):
            pair = self.person(pose=False, false_legs=True)
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})
            self.assertEqual(pair[0]["seatEvidence"]["reason"], "STANDING")

    def test_rising_hips_and_shoulders_still_clear_seating_despite_overlap(self):
        trial = self.qualified()
        for i in range(11, 25):
            pair = self.person(pose=False, false_legs=True, rising=True)
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})
            self.assertEqual(pair[0]["seatEvidence"]["reason"], "STANDING")
            self.assertEqual(trial.history, {})

    def test_missing_wrists_are_not_replaced_by_old_qualification(self):
        trial = self.qualified()
        pair = self.person(pose=False, false_legs=True, wrists=False)
        self.assertEqual(self.step(trial, 5.5, pair), {"table": 0})
        self.assertEqual(pair[0]["seatEvidence"]["state"], "UNKNOWN")
        for i in range(12, 23):
            pair = self.person(pose=False, false_legs=True)
            count = self.step(trial, i / 2, pair)
            self.assertEqual(count, {"table": 0})
            if i == 15:
                self.assertEqual(pair[0]["seatEvidence"]["reason"], "STANDING")
        # Once the bounded history expires, a visible pose must qualify again.
        for i in range(24, 35):
            pair = self.person()
            self.assertEqual(self.step(trial, i / 2, pair, []), {"table": int(i == 34)})

    def test_masking_without_enabled_wrist_trial_does_not_create_positive_pose(self):
        trial = self.qualified(wrists=False)
        pair = self.person(pose=False, false_legs=True)
        self.assertEqual(self.step(trial, 5.5, pair), {"table": 0})
        self.assertEqual(pair[0]["seatEvidence"]["state"], "UNKNOWN")

    def test_low_current_score_is_uncounted_even_with_qualified_wrist_support(self):
        trial = self.qualified()
        pair = self.person(pose=False, false_legs=True, score=0.31)
        self.assertEqual(self.step(trial, 5.5, pair), {"table": 0})
        self.assertEqual(pair[0]["seatEvidence"]["reason"], "LOW_DETECTION_SCORE")

    def test_missing_track_or_source_gap_cannot_keep_the_occlusion_exception(self):
        for gap in (False, True):
            with self.subTest(gap=gap):
                trial = self.qualified()
                if not gap:
                    trial.observe(5.5, [], [], [])
                pair = self.person(pose=False, false_legs=True)
                self.assertEqual(
                    self.step(trial, 6.5 if gap else 6, pair), {"table": 0}
                )
                self.assertEqual(pair[0]["seatEvidence"]["reason"], "STANDING")

    def test_two_genuine_seated_claims_still_conflict(self):
        trial = self.qualified()
        a, b = self.person(), self.person(track=2)
        counts = trial.observe(5.5, [a[0], b[0]], [a[1], b[1]], [])
        self.assertEqual(counts, {"table": 0})
        self.assertTrue(
            all(p["seatEvidence"]["reason"] == "SEAT_CONFLICT" for p in (a[0], b[0]))
        )

    def test_bent_torso_with_supported_downward_legs_is_still_excluded(self):
        trial = SeatTrial(self.regions, self.tops)
        for i in range(25):
            p, obs = self.person()
            points = obs["points"].copy()
            points[5:7] = [[600, 500], [640, 500]]
            points[13:15] = [[400, 650], [440, 650]]
            points[15:17] = [[400, 800], [440, 800]]
            obs = seat_observation(points, obs["scores"], 0.9, self.regions, 1000, 1000)
            self.assertTrue(supported_standing_legs(points, obs["scores"]))
            self.assertEqual(self.step(trial, i / 2, (p, obs), []), {"table": 0})
            self.assertEqual(p["seatEvidence"]["postureGuard"], "LEG_CHAIN")

    def test_short_projected_thighs_or_unreliable_ankles_do_not_add_standing(self):
        p, obs = self.person()
        points = obs["points"].copy()
        points[13:15] = [[400, 550], [440, 550]]
        points[15:17] = [[400, 850], [440, 850]]
        self.assertFalse(supported_standing_legs(points, obs["scores"]))
        points[13:15] = [[400, 650], [440, 650]]
        points[15:17] = [[400, 800], [440, 800]]
        scores = obs["scores"].copy()
        scores[15:17] = 0.39
        self.assertFalse(supported_standing_legs(points, scores))

    def test_recent_standing_is_uncounted_uncertainty_and_does_not_refresh_itself(self):
        trial = SeatTrial(self.regions, self.tops)
        context = [{"box": [0.35, 0.1, 0.62, 0.9], "upperAnchor": [0.42, 0.15]}]
        p, obs = self.person(pose=False, false_legs=True)
        self.assertEqual(self.step(trial, 0, (p, obs), context), {"table": 0})
        for t in (0.5, 1, 1.5, 2, 2.5):
            p, obs = self.person(pose=False, false_legs=True)
            obs["scores"][13:15] = 0.1
            self.assertEqual(self.step(trial, t, (p, obs), context), {"table": 0})
            self.assertEqual(
                p["seatEvidence"]["postureGuard"],
                "RECENT_STANDING" if t <= 2 else "NONE",
            )
        self.assertEqual(trial.standing_recent, {})


if __name__ == "__main__":
    unittest.main()
