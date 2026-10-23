"""Synthetic seated, standing and moving cases for the optional wrist cue."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrist_trial import WristTrial, wrist_observation


class WristTrialTests(unittest.TestCase):
    def setUp(self):
        self.seats = [
            {
                "id": "seat",
                "tableId": "table",
                "x": 0.2,
                "y": 0.4,
                "width": 0.6,
                "height": 0.3,
            }
        ]
        self.tops = [
            {
                "id": "top",
                "tableId": "table",
                "x": 0.5,
                "y": 0.3,
                "width": 0.3,
                "height": 0.15,
            }
        ]
        self.points = [[400.0, 500.0] for _ in range(17)]
        self.scores = [0.9] * 17
        for shoulder, hip, knee in ((5, 11, 13), (6, 12, 14)):
            self.points[shoulder] = [400.0, 300.0]
            self.points[hip] = [400.0, 500.0]
            self.points[knee] = [550.0, 500.0]
        self.points[9] = self.points[10] = [550.0, 400.0]

    def person(self, track=1, shift=0, score=0.9, box=None):
        points = [[x + shift, y] for x, y in self.points]
        box = box if box is not None else [300 + shift, 250, 500 + shift, 700]
        cue = wrist_observation(
            points, self.scores, box, track, score, self.seats, self.tops, 1000, 1000
        )
        return {"trackId": track, "box": [v / 1000 for v in box], "wristEvidence": cue}

    def test_stationary_wrists_require_five_source_seconds_and_never_confirm_seating(
        self,
    ):
        verifier = WristTrial(["table"])
        for step in range(13):
            person = self.person()
            states = verifier.observe(step / 2, [person])
            self.assertEqual(states["table"], "PROBABLE" if step >= 10 else "UNKNOWN")
            self.assertEqual(person["wristEvidence"]["stableSeconds"], step / 2)
            self.assertNotEqual(person["wristEvidence"]["state"], "ACCEPTED")
        self.assertEqual(verifier.observe(6.5, []), {"table": "UNKNOWN"})

    def test_waiter_standing_with_wrists_in_tabletop_for_twelve_seconds_is_rejected(
        self,
    ):
        self.points[13] = self.points[14] = [400.0, 700.0]
        verifier = WristTrial(["table"])
        for step in range(25):
            person = self.person()
            self.assertEqual(person["wristEvidence"]["reason"], "STANDING")
            self.assertEqual(verifier.observe(step / 2, [person]), {"table": "UNKNOWN"})

    def test_slow_walking_is_compared_with_first_position_not_each_small_step(self):
        verifier = WristTrial(["table"])
        reasons = []
        for step in range(25):
            person = self.person(shift=step * 6)
            states = verifier.observe(step / 2, [person])
            self.assertEqual(states["table"], "UNKNOWN")
            reasons.append(person["wristEvidence"]["reason"])
        self.assertIn("MOVING", reasons)

    def test_unknown_hips_can_only_produce_probable_box_center_evidence(self):
        self.scores[11] = self.scores[12] = 0.1
        self.scores[13] = self.scores[14] = 0.1
        verifier = WristTrial(["table"])
        for step in range(11):
            person = self.person()
            self.assertEqual(person["wristEvidence"]["anchorSource"], "BOX_CENTER")
            verifier.observe(step / 2, [person])
        self.assertEqual(person["wristEvidence"]["state"], "PROBABLE")

    def test_missing_wrists_unknown_track_low_detection_and_non_finite_points(self):
        self.assertEqual(self.person(track=None)["wristEvidence"]["reason"], "NO_TRACK")
        self.assertEqual(
            self.person(score=0.16)["wristEvidence"]["reason"], "LOW_DETECTION_SCORE"
        )
        self.scores[9] = self.scores[10] = 0.1
        self.assertEqual(self.person()["wristEvidence"]["reason"], "WRIST_UNCERTAIN")
        self.scores[9] = self.scores[10] = 0.9
        self.points[9] = self.points[10] = [float("nan"), 400]
        self.assertEqual(self.person()["wristEvidence"]["reason"], "WRIST_UNCERTAIN")

    def test_wrists_must_map_to_body_seat_at_the_same_unambiguous_table(self):
        self.points[9] = self.points[10] = [100.0, 100.0]
        self.assertEqual(self.person()["wristEvidence"]["reason"], "NO_TABLETOP_MATCH")
        self.points[9] = self.points[10] = [550.0, 400.0]
        self.tops.append({**self.tops[0], "id": "other-top", "tableId": "other"})
        self.assertEqual(self.person()["wristEvidence"]["reason"], "AMBIGUOUS_TABLETOP")
        self.tops = [self.tops[1]]
        self.assertEqual(self.person()["wristEvidence"]["reason"], "TABLE_MISMATCH")
        self.tops[0]["tableId"] = "table"
        self.seats.append({**self.seats[0], "id": "overlapping-seat"})
        self.assertEqual(self.person()["wristEvidence"]["reason"], "AMBIGUOUS_SEAT")
        self.seats = []
        self.assertEqual(self.person()["wristEvidence"]["reason"], "NO_SEAT_MATCH")

    def test_missing_sample_gap_track_or_motion_reference_change_cannot_pool_dwell(
        self,
    ):
        for change in ("missing", "gap", "track", "motion"):
            with self.subTest(change=change):
                verifier = WristTrial(["table"])
                self.scores[5] = self.scores[6] = 0.9
                self.scores[11] = self.scores[12] = 0.9
                for step in range(10):
                    verifier.observe(step / 2, [self.person()])
                seconds, track = 5.0, 1
                if change == "missing":
                    verifier.observe(seconds, [])
                    seconds += 0.5
                if change == "gap":
                    seconds = 6.0
                if change == "track":
                    track = 2
                if change == "motion":
                    self.scores[5] = self.scores[6] = 0.1
                person = self.person(track=track)
                self.assertEqual(
                    verifier.observe(seconds, [person]), {"table": "UNKNOWN"}
                )
                self.assertEqual(person["wristEvidence"]["stableSeconds"], 0)

    def test_two_tracks_in_same_seat_cannot_both_count_or_continue_previous_dwell(self):
        verifier = WristTrial(["table"])
        for step in range(11):
            verifier.observe(step / 2, [self.person()])
        people = [self.person(), self.person(track=2)]
        self.assertEqual(verifier.observe(5.5, people), {"table": "UNKNOWN"})
        self.assertTrue(
            all(p["wristEvidence"]["reason"] == "SEAT_CONFLICT" for p in people)
        )
        returned = self.person()
        verifier.observe(6, [returned])
        self.assertEqual(returned["wristEvidence"]["stableSeconds"], 0)

    def test_weak_wrist_after_valid_dwell_resets_candidate(self):
        verifier = WristTrial(["table"])
        for step in range(10):
            verifier.observe(step / 2, [self.person()])
        self.scores[9] = self.scores[10] = 0.2
        self.assertEqual(verifier.observe(5, [self.person()]), {"table": "UNKNOWN"})
        self.scores[9] = self.scores[10] = 0.9
        returned = self.person()
        verifier.observe(5.5, [returned])
        self.assertEqual(returned["wristEvidence"]["stableSeconds"], 0)
        with self.assertRaises(ValueError):
            verifier.observe(5.5, [])

    def test_changing_box_bottom_with_stable_shoulders_keeps_dwell(self):
        verifier = WristTrial(["table"])
        for step in range(15):
            bottom = 700 if step % 2 == 0 else 500
            person = self.person(box=[300, 250, 500, bottom])
            verifier.observe(step / 2, [person])
            cue = person["wristEvidence"]
            self.assertEqual(cue["motionSource"], "SHOULDERS")
            self.assertEqual(cue["stableSeconds"], step / 2)
            self.assertEqual(cue["state"], "PROBABLE" if step >= 10 else "CANDIDATE")

    def test_hip_and_box_mapping_can_alternate_with_stable_motion_reference(self):
        verifier = WristTrial(["table"])
        for step in range(12):
            self.scores[11] = self.scores[12] = 0.9 if step % 2 else 0.1
            person = self.person()
            verifier.observe(step / 2, [person])
            self.assertEqual(person["wristEvidence"]["motionSource"], "SHOULDERS")
            self.assertEqual(person["wristEvidence"]["stableSeconds"], step / 2)
        self.assertEqual(person["wristEvidence"]["state"], "PROBABLE")

    def test_box_fallback_cannot_ignore_crop_changes_or_slow_walking(self):
        self.scores[5] = self.scores[6] = 0.1
        verifier = WristTrial(["table"])
        for step in range(10):
            verifier.observe(step / 2, [self.person()])
        person = self.person(box=[300, 250, 500, 500])
        verifier.observe(5, [person])
        self.assertEqual(person["wristEvidence"]["reason"], "MOVING")
        verifier = WristTrial(["table"])
        for step in range(25):
            self.assertEqual(
                verifier.observe(step / 2, [self.person(shift=step * 6)]),
                {"table": "UNKNOWN"},
            )

    def test_frozen_pose_cannot_hide_box_translation_or_width_change(self):
        for box in ([360, 250, 560, 700], [300, 320, 500, 770], [250, 250, 550, 700]):
            with self.subTest(box=box):
                verifier = WristTrial(["table"])
                for step in range(10):
                    verifier.observe(step / 2, [self.person()])
                person = self.person(box=box)
                self.assertEqual(verifier.observe(5, [person]), {"table": "UNKNOWN"})
                self.assertEqual(person["wristEvidence"]["reason"], "MOVING")

    def overlap(self, other_table="table"):
        self.seats.append(
            {
                **self.seats[0],
                "id": "overlap",
                "tableId": other_table,
                "y": 0.55,
                "height": 0.15,
            }
        )

    def ambiguous_hips(self):
        self.points[11] = self.points[12] = [400.0, 560.0]

    def test_overlap_can_only_keep_an_established_stationary_seat(self):
        self.overlap()
        verifier = WristTrial(["table"])
        for step in range(10):
            verifier.observe(step / 2, [self.person()])
        self.ambiguous_hips()
        for step in range(10, 14):
            person = self.person()
            self.assertEqual(person["wristEvidence"]["reason"], "AMBIGUOUS_SEAT")
            self.assertEqual(
                verifier.observe(step / 2, [person]), {"table": "PROBABLE"}
            )
            self.assertEqual(person["wristEvidence"]["seatRegionId"], "seat")
            self.assertEqual(person["wristEvidence"]["seatResolution"], "CONTINUITY")
            self.assertEqual(person["wristEvidence"]["stableSeconds"], step / 2)

    def test_mapping_source_and_overlap_can_alternate_without_pooling_new_seats(self):
        self.overlap()
        verifier = WristTrial(["table"])
        verifier.observe(0, [self.person()])
        self.ambiguous_hips()
        for step in range(1, 13):
            self.scores[11] = self.scores[12] = 0.9 if step % 2 else 0.1
            person = self.person()
            verifier.observe(step / 2, [person])
            cue = person["wristEvidence"]
            self.assertEqual(cue["seatRegionId"], "seat")
            self.assertEqual(cue["stableSeconds"], step / 2)
            self.assertEqual(cue["state"], "PROBABLE" if step >= 10 else "CANDIDATE")

    def test_ambiguous_seat_cannot_start_cold_or_bridge_missing_gap_or_new_track(self):
        self.overlap()
        for change in ("cold", "missing", "gap", "track", "motion"):
            with self.subTest(change=change):
                self.points[11] = self.points[12] = [400.0, 500.0]
                self.scores[5] = self.scores[6] = 0.9
                verifier = WristTrial(["table"])
                if change != "cold":
                    for step in range(10):
                        verifier.observe(step / 2, [self.person()])
                seconds = 5.0
                if change == "missing":
                    verifier.observe(5, [])
                    seconds = 5.5
                if change == "gap":
                    seconds = 6
                if change == "motion":
                    self.scores[5] = self.scores[6] = 0.1
                self.ambiguous_hips()
                person = self.person(track=2 if change == "track" else 1)
                self.assertEqual(
                    verifier.observe(seconds, [person]), {"table": "UNKNOWN"}
                )
                self.assertEqual(person["wristEvidence"]["reason"], "AMBIGUOUS_SEAT")
                self.assertEqual(person["wristEvidence"]["stableSeconds"], 0)

    def test_overlap_cannot_follow_a_moving_person_or_cross_table_boundaries(self):
        for other_table, shift in (("table", 60), ("other", 0)):
            with self.subTest(table=other_table, shift=shift):
                self.seats = self.seats[:1]
                self.points[11] = self.points[12] = [400.0, 500.0]
                self.overlap(other_table)
                verifier = WristTrial(["table", "other"])
                for step in range(10):
                    verifier.observe(step / 2, [self.person()])
                self.ambiguous_hips()
                person = self.person(shift=shift)
                self.assertTrue(
                    all(v == "UNKNOWN" for v in verifier.observe(5, [person]).values())
                )
                self.assertEqual(person["wristEvidence"]["reason"], "AMBIGUOUS_SEAT")

    def test_resolved_overlap_still_rejects_two_people_claiming_one_seat(self):
        self.overlap()
        verifier = WristTrial(["table"])
        for step in range(10):
            verifier.observe(step / 2, [self.person()])
        other = self.person(track=2)
        self.ambiguous_hips()
        person = self.person()
        self.assertEqual(verifier.observe(5, [person, other]), {"table": "UNKNOWN"})
        self.assertTrue(
            all(
                p["wristEvidence"]["reason"] == "SEAT_CONFLICT" for p in (person, other)
            )
        )

    def test_distinct_people_in_distinct_seats_can_each_be_probable(self):
        self.seats = [
            {**self.seats[0], "x": 0.3, "width": 0.15},
            {**self.seats[0], "id": "seat2", "x": 0.55, "width": 0.15},
        ]
        verifier = WristTrial(["table"])
        for step in range(11):
            people = [self.person(), self.person(track=2, shift=200)]
            states = verifier.observe(step / 2, people)
        self.assertEqual(states, {"table": "PROBABLE"})
        self.assertTrue(all(p["wristEvidence"]["state"] == "PROBABLE" for p in people))


if __name__ == "__main__":
    unittest.main()
