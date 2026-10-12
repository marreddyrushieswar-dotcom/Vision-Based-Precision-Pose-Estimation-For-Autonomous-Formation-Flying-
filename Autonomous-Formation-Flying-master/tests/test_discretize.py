"""Regression tests for pose discretisation and soft pose estimates.

Run from the repository root:
    python -m unittest discover -s Autonomous-Formation-Flying-master/tests -v

These tests check conversion invariants; they do not require trained model files.
"""
import unittest

import numpy as np

from config import BIN_SHAPE, N_LABELS, POSE_BINS
from discretize import BIN_CENTERS, label_to_pose, pose_to_label, soft_pose


class PoseDiscretizationTests(unittest.TestCase):
    def test_all_bin_centres_map_back_to_their_labels(self):
        labels = np.arange(N_LABELS)
        recovered = pose_to_label(BIN_CENTERS)
        np.testing.assert_array_equal(recovered, labels)

    def test_label_conversion_preserves_expected_shapes(self):
        self.assertEqual(label_to_pose(0).shape, (1, 4))
        self.assertEqual(label_to_pose([0, N_LABELS - 1]).shape, (2, 4))
        self.assertEqual(pose_to_label(BIN_CENTERS[:3]).shape, (3,))

    def test_values_outside_range_are_clipped_to_edge_bins(self):
        lows = [POSE_BINS[k][0] - 1e6 for k in ("x", "y", "z", "phi")]
        highs = [POSE_BINS[k][1] + 1e6 for k in ("x", "y", "z", "phi")]
        labels = pose_to_label([lows, highs])
        self.assertEqual(int(labels[0]), 0)
        self.assertEqual(int(labels[1]), N_LABELS - 1)

    def test_bin_centres_are_inside_configured_ranges(self):
        for column, key in enumerate(("x", "y", "z", "phi")):
            lo, hi, _ = POSE_BINS[key]
            self.assertTrue(np.all(BIN_CENTERS[:, column] >= lo))
            self.assertTrue(np.all(BIN_CENTERS[:, column] <= hi))

    def test_one_hot_soft_pose_returns_selected_bin_centre(self):
        selected = np.array([0, N_LABELS // 2, N_LABELS - 1])
        probs = np.zeros((len(selected), N_LABELS), dtype=float)
        probs[np.arange(len(selected)), selected] = 1.0
        np.testing.assert_allclose(soft_pose(probs), BIN_CENTERS[selected])

    def test_uniform_probabilities_return_mean_of_bin_centres(self):
        probs = np.full((1, N_LABELS), 1.0 / N_LABELS)
        np.testing.assert_allclose(soft_pose(probs), BIN_CENTERS.mean(axis=0)[None, :])


if __name__ == "__main__":
    unittest.main()
