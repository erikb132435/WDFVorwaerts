"""Tests der gemeinsam verwendeten Echtzeitmetriken."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realtime_metrics as metrics


class RealtimeMetricsTests(unittest.TestCase):
    def test_timing_summary_contains_p999(self):
        values = np.arange(1, 1001, dtype=float) * 1e-6
        summary = metrics.timing_summary(values)
        self.assertIn("p999_us", summary)
        self.assertGreater(summary["p999_us"], summary["p99_us"])
        self.assertEqual(summary["max_us"], 1000.0)

    def test_deadline_misses_are_grouped_into_episodes(self):
        misses = np.asarray([False, True, True, False, True, False, True, True, True])
        deadlines = np.full(len(misses), 0.005)
        episodes = metrics.deadline_episodes(misses, deadlines)
        np.testing.assert_array_equal(episodes.start_indices, [1, 4, 6])
        np.testing.assert_array_equal(episodes.lengths, [2, 1, 3])
        self.assertEqual(episodes.missed_block_count, 6)
        self.assertEqual(episodes.episode_count, 3)
        self.assertEqual(episodes.longest_episode_blocks, 3)
        self.assertAlmostEqual(episodes.longest_episode_duration_s, 0.015)

    def test_empty_episode_sequence_is_well_defined(self):
        episodes = metrics.deadline_episodes(np.zeros(4, dtype=bool), 0.01)
        self.assertEqual(episodes.episode_count, 0)
        self.assertEqual(episodes.longest_episode_blocks, 0)
        self.assertEqual(episodes.longest_episode_duration_s, 0.0)


if __name__ == "__main__":
    unittest.main()
