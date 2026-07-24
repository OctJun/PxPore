import unittest

from _test_env import configure_test_threads

configure_test_threads()

import numpy as np

from PxPore.pores import (
    get_psd_from_centerline,
    get_psd_from_voxels_mc,
)


class PsdTests(unittest.TestCase):
    def test_center_psd_is_sorted_and_normalized(self):
        nodes = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
        ], dtype=np.float32)
        radii = np.array([0.1, 0.3, 0.2], dtype=np.float32)
        psd, centers = get_psd_from_centerline(
            nodes, radii, bin_size=0.1)
        self.assertAlmostEqual(centers[0, 4], 0.6, places=6)
        self.assertAlmostEqual(centers[-1, 4], 0.2, places=6)
        self.assertAlmostEqual(float(psd[:, 3].sum()), 1.0)
        self.assertAlmostEqual(float(psd[-1, 4]), 1.0)

    def test_mc_fixed_seed_and_normalization(self):
        shape = (5, 4, 3)
        dmin = np.full(shape, 0.2, dtype=np.float32)
        accessible = np.ones(shape, dtype=np.uint8)
        grid_info = (*shape, 0.1, 0.1, 0.1)
        first, first_promoted = get_psd_from_voxels_mc(
            dmin, accessible, grid_info,
            bin_size=0.05, n_samples=2000, seed=11451466)
        second, second_promoted = get_psd_from_voxels_mc(
            dmin, accessible, grid_info,
            bin_size=0.05, n_samples=2000, seed=11451466)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_promoted, second_promoted)
        self.assertEqual(first_promoted, 0.0)
        self.assertAlmostEqual(float(first[:, 3].sum()), 1.0)
        self.assertAlmostEqual(float((first[:, 4] * 0.05).sum()), 1.0)

    def test_mc_assigns_larger_containing_ball(self):
        shape = (7, 7, 7)
        dmin = np.full(shape, 0.05, dtype=np.float32)
        dmin[3, 3, 3] = 0.25
        accessible = np.ones(shape, dtype=np.uint8)
        grid_info = (*shape, 0.1, 0.1, 0.1)
        data, promoted = get_psd_from_voxels_mc(
            dmin, accessible, grid_info,
            bin_size=0.05, n_samples=10000, seed=11451466)
        self.assertGreater(promoted, 0.0)
        self.assertAlmostEqual(float(data[:, 3].sum()), 1.0)
        self.assertGreater(np.count_nonzero(data[:, 2]), 1)

    def test_mc_rejects_invalid_inputs(self):
        shape = (2, 2, 2)
        grid_info = (*shape, 0.1, 0.1, 0.1)
        with self.assertRaisesRegex(ValueError, "bin_size"):
            get_psd_from_voxels_mc(
                np.ones(shape), np.ones(shape), grid_info, bin_size=0)
        with self.assertRaisesRegex(ValueError, "n_samples"):
            get_psd_from_voxels_mc(
                np.ones(shape), np.ones(shape), grid_info, n_samples=0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            get_psd_from_voxels_mc(
                np.ones(shape), np.ones((2, 2, 3)), grid_info)


if __name__ == "__main__":
    unittest.main()
