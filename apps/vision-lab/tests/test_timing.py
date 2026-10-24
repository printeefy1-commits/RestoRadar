"""Source-time boundaries must survive binary roundoff, never real shortfalls."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decisions import SeatingVerifier, duration_reached


class SourceTimingTests(unittest.TestCase):
    def test_five_seconds_survive_roundoff_at_multiple_source_offsets(self):
        for offset in (14.5, 30.5, 86390.0):
            with self.subTest(offset=offset):
                start = math.nextafter(offset, math.inf)
                now = offset + 5
                self.assertLess(now - start, 5)
                self.assertTrue(duration_reached(start, now, 5))

    def test_even_a_nanosecond_shortfall_cannot_qualify(self):
        for start in (0.0, 14.5, 30.5, 86390.0):
            for missing in (0.5, 0.000001, 0.000000001):
                with self.subTest(start=start, missing=missing):
                    self.assertFalse(duration_reached(start, start + 5 - missing, 5))

    def test_table_confirms_at_the_source_boundary_without_an_extra_sample(self):
        verifier = SeatingVerifier(["table"])
        start = math.nextafter(14.5, math.inf)
        for index in range(11):
            seconds = start if index == 0 else 14.5 + index / 2
            states, events = verifier.observe(seconds, {"table"})
            self.assertEqual(states["table"], "OCCUPIED" if index == 10 else "UNKNOWN")
            self.assertEqual(len(events), int(index == 10))
        self.assertEqual(events[0]["observedSeconds"], start)
        self.assertEqual(events[0]["confirmedSeconds"], 19.5)


if __name__ == "__main__":
    unittest.main()
