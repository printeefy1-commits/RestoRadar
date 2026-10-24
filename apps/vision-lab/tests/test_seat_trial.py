"""Per-seat qualification and bounded strong revalidation, without real video."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seat_trial import SeatTrial, seat_observation


class SeatTrialTests(unittest.TestCase):
    def setUp(self):
        self.regions = [
            {
                "id": "seat",
                "tableId": "table",
                "x": 0.2,
                "y": 0.2,
                "width": 0.4,
                "height": 0.65,
            }
        ]

    def person(
        self,
        track=1,
        score=0.9,
        pose=True,
        shift=0,
        wrist=None,
        standing=False,
        hips=True,
        table="table",
        seat="seat",
        box=None,
    ):
        points = [[400.0 + shift, 500.0] for _ in range(17)]
        scores = [0.9] * 17
        for i in (5, 6):
            points[i] = [400.0 + shift, 300.0]
        for i in (13, 14):
            points[i] = [400.0 + shift, 700.0] if standing else [550.0 + shift, 500.0]
        if not hips:
            scores[11] = scores[12] = 0.1
        person = {
            "trackId": track,
            "score": round(score, 4),
            "box": box or [(300 + shift) / 1000, 0.25, (500 + shift) / 1000, 0.7],
            "evidence": "ACCEPTED" if pose else "POSE_UNCERTAIN",
            "tableId": table if pose else None,
        }
        if wrist:
            person["wristEvidence"] = {
                "state": wrist,
                "seatRegionId": seat,
                "tableId": table,
                "stableSeconds": 5 if wrist == "PROBABLE" else 0,
            }
        obs = seat_observation(points, scores, score, self.regions, 1000, 1000)
        return person, obs

    def step(self, trial, t, *pairs):
        return trial.observe(t, [p for p, _ in pairs], [o for _, o in pairs])

    def verified(self):
        trial = SeatTrial(self.regions)
        for i in range(11):
            pair = self.person()
            counts = self.step(trial, i / 2, pair)
            self.assertEqual(counts["table"], int(i == 10))
        return trial

    def test_pose_qualifies_five_seconds_without_a_wrist_requirement(self):
        trial = self.verified()
        for i in range(11, 25):
            pair = self.person(
                box=[0.15, 0.2, 0.65, 0.9]
            )  # Upper body/box changes after qualification.
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 1})
            self.assertEqual(pair[0]["seatEvidence"]["qualifiedBy"], "POSE")

    def test_cold_pelvis_motion_cannot_accumulate_a_slow_walker(self):
        trial = SeatTrial(self.regions)
        for i in range(26):
            pair = self.person(shift=i * 6)
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})

    def test_standing_never_qualifies_even_with_a_reported_wrist_probability(self):
        trial = SeatTrial(self.regions)
        for i in range(25):
            pair = self.person(standing=True, wrist="PROBABLE")
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})
            self.assertEqual(pair[0]["seatEvidence"]["reason"], "STANDING")

    def test_initial_dwell_cannot_use_a_low_score_or_rounded_display_score(self):
        for score in (0.34998, 0.31, 0.12, float("nan"), float("inf"), 1.1):
            with self.subTest(score=score):
                trial = SeatTrial(self.regions)
                for i in range(15):
                    pair = self.person(score=score, wrist="PROBABLE")
                    self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})
        trial = SeatTrial(self.regions)
        for i in range(10):
            self.step(trial, i / 2, self.person())
        self.assertEqual(self.step(trial, 5, self.person(score=0.3265)), {"table": 0})
        for i in range(11, 22):
            pair = self.person()
            self.assertEqual(self.step(trial, i / 2, pair), {"table": int(i == 21)})

    def test_qualified_history_is_not_counted_while_suspended(self):
        trial = self.verified()
        for t in (5.5, 6):
            pair = self.person(score=0.12)
            self.assertEqual(self.step(trial, t, pair), {"table": 0})
            self.assertTrue(pair[0]["seatEvidence"]["revalidationPending"])
            self.assertEqual(pair[0]["seatEvidence"]["state"], "UNKNOWN")
        pair = self.person()
        self.assertEqual(self.step(trial, 6.5, pair), {"table": 1})
        cue = pair[0]["seatEvidence"]
        self.assertTrue(cue["resumed"])
        self.assertEqual(cue["gapSeconds"], 1.5)
        self.assertEqual(cue["poseSeconds"], 0)
        self.assertEqual(cue["qualifiedAtSeconds"], 5)

    def test_weak_samples_do_not_refresh_the_revalidation_deadline(self):
        trial = self.verified()
        for i in range(11, 16):
            self.assertEqual(
                self.step(trial, i / 2, self.person(score=0.12)), {"table": 0}
            )
        pair = self.person()
        self.assertEqual(self.step(trial, 8, pair), {"table": 0})
        self.assertIsNone(pair[0]["seatEvidence"]["qualifiedAtSeconds"])

    def test_revalidation_requires_current_accepted_pose_not_a_strong_box_alone(self):
        trial = self.verified()
        pair = self.person(pose=False, wrist="CANDIDATE")
        self.assertEqual(self.step(trial, 5.5, pair), {"table": 0})
        self.assertTrue(pair[0]["seatEvidence"]["revalidationPending"])
        pair = self.person()
        self.assertEqual(self.step(trial, 6, pair), {"table": 1})
        self.assertTrue(pair[0]["seatEvidence"]["resumed"])

    def test_missing_id_or_timestamp_gap_breaks_history(self):
        for change in ("missing", "new-track", "gap"):
            with self.subTest(change=change):
                trial = self.verified()
                seconds = 5.5
                if change == "missing":
                    self.step(trial, seconds)
                    seconds = 6
                if change == "gap":
                    seconds = 6.5
                pair = self.person(track=2 if change == "new-track" else 1)
                self.assertEqual(self.step(trial, seconds, pair), {"table": 0})
                self.assertFalse(pair[0]["seatEvidence"]["resumed"])

    def test_two_second_limit_allows_only_float_representation_error(self):
        for excess in (0, 1e-9):
            with self.subTest(excess=excess):
                trial = self.verified()
                for t in (5.5, 6, 6.5):
                    self.step(trial, t, self.person(score=0.12))
                self.assertEqual(
                    self.step(trial, 7 + excess, self.person()),
                    {"table": int(excess == 0)},
                )

    def test_standing_outside_missing_hips_and_ambiguity_clear_qualified_history(self):
        for change in ("standing", "outside", "hips", "ambiguous"):
            with self.subTest(change=change):
                self.setUp()
                if change == "ambiguous":
                    self.regions.append(
                        {**self.regions[0], "id": "other-seat", "x": 0.5}
                    )
                trial = self.verified()
                pair = self.person(
                    standing=change == "standing",
                    shift=(
                        300
                        if change == "outside"
                        else 150 if change == "ambiguous" else 0
                    ),
                    hips=change != "hips",
                )
                self.assertEqual(sum(self.step(trial, 5.5, pair).values()), 0)
                self.assertEqual(trial.history, {})

    def test_new_seat_or_table_cannot_inherit_qualification(self):
        for table in ("table", "other-table"):
            with self.subTest(table=table):
                self.setUp()
                self.regions[0]["width"] = 0.25
                self.regions.append(
                    {
                        **self.regions[0],
                        "id": "second-seat",
                        "tableId": table,
                        "x": 0.55,
                    }
                )
                trial = self.verified()
                pair = self.person(shift=200, table=table)
                self.assertEqual(sum(self.step(trial, 5.5, pair).values()), 0)
                self.assertIsNone(pair[0]["seatEvidence"]["qualifiedAtSeconds"])

    def test_same_seat_or_duplicate_track_conflicts_never_double_count(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                trial = self.verified()
                a, b = self.person(), self.person(track=1 if duplicate else 2)
                self.assertEqual(self.step(trial, 5.5, a, b), {"table": 0})
                self.assertEqual(trial.history, {})
                self.assertTrue(
                    all(
                        p["seatEvidence"]["reason"] == "SEAT_CONFLICT"
                        for p, _ in (a, b)
                    )
                )

    def test_standing_waiter_does_not_claim_the_seated_customers_count(self):
        trial = self.verified()
        self.assertEqual(
            self.step(trial, 5.5, self.person(), self.person(track=2, standing=True)),
            {"table": 1},
        )

    def test_wrist_and_pose_windows_are_not_pooled_before_five_seconds(self):
        trial = SeatTrial(self.regions)
        for i in range(13):
            pair = self.person(pose=i < 6, wrist="CANDIDATE")
            self.assertEqual(self.step(trial, i / 2, pair), {"table": 0})
        pair = self.person(pose=False, wrist="PROBABLE")
        self.assertEqual(self.step(trial, 6.5, pair), {"table": 1})
        self.assertEqual(pair[0]["seatEvidence"]["qualifiedBy"], "WRIST")

    def test_current_qualified_wrist_can_supply_its_seat_when_hips_are_hidden(self):
        trial = SeatTrial(self.regions)
        pair = self.person(pose=False, hips=False, wrist="PROBABLE")
        self.assertEqual(self.step(trial, 0, pair), {"table": 1})
        trial = SeatTrial(self.regions)
        pair = self.person(pose=False, hips=False, wrist="PROBABLE", table="wrong")
        self.assertEqual(self.step(trial, 0, pair), {"table": 0})
        trial = SeatTrial(self.regions)
        pair = self.person(pose=False, wrist="PROBABLE")
        pair[0]["wristEvidence"]["stableSeconds"] = 4.5
        self.assertEqual(self.step(trial, 0, pair), {"table": 0})

    def test_distinct_seats_and_both_sources_are_counted_once_each(self):
        self.regions[0]["width"] = 0.25
        self.regions.append({**self.regions[0], "id": "second", "x": 0.55})
        trial = SeatTrial(self.regions)
        for i in range(11):
            people = [
                self.person(wrist="PROBABLE" if i == 10 else "CANDIDATE"),
                self.person(track=2, shift=200),
            ]
            counts = self.step(trial, i / 2, *people)
        self.assertEqual(counts, {"table": 2})
        self.assertEqual(people[0][0]["seatEvidence"]["source"], "BOTH")

    def test_source_time_roundoff_and_invalid_timestamps(self):
        trial = SeatTrial(self.regions)
        start = math.nextafter(14.5, math.inf)
        for i in range(11):
            counts = self.step(trial, start if i == 0 else 14.5 + i / 2, self.person())
        self.assertEqual(counts, {"table": 1})
        for value in (19.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.step(trial, value, self.person())
        with self.assertRaises(ValueError):
            trial.observe(20, [self.person()[0]], [])


if __name__ == "__main__":
    unittest.main()
