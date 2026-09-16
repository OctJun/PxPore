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
        accessible = np.ones(shape, dtype=bool)
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
        accessible = np.ones(shape, dtype=bool)
        grid_info = (*shape, 0.1, 0.1, 0.1)
        data, promoted = get_psd_from_voxels_mc(
            dmin, accessible, grid_info,
            bin_size=0.05, n_samples=10000, seed=11451466)
        self.assertGreater(promoted, 0.0)
        self.assertAlmostEqual(float(data[:, 3].sum()), 1.0)
        self.assertGreater(np.count_nonzero(data[:, 2]), 1)

    def test_mc_matches_exhaustive_voxel_ball_search(self):
        shape = (6, 5, 4)
        spacing = np.array([0.09, 0.11, 0.13])
        grid_info = (*shape, *spacing)
        rng = np.random.default_rng(20260916)
        dmin = rng.uniform(0.02, 0.18, size=shape).astype(np.float32)
        accessible = rng.random(shape) > 0.2
        n_samples = 500
        seed = 99173
        bin_size = 0.02

        data, promoted = get_psd_from_voxels_mc(
            dmin,
            accessible,
            grid_info,
            bin_size=bin_size,
            n_samples=n_samples,
            seed=seed,
        )

        valid_flat = np.flatnonzero(accessible.ravel())
        sample_rng = np.random.default_rng(seed)
        sampled_flat = valid_flat[
            sample_rng.integers(0, valid_flat.size, size=n_samples)
        ]
        sample_xyz = np.column_stack(np.unravel_index(sampled_flat, shape))
        center_xyz = np.column_stack(np.unravel_index(valid_flat, shape))
        center_radii = dmin.ravel()[valid_flat]
        assigned = np.empty(n_samples, dtype=np.float32)
        box = np.asarray(shape) * spacing
        for i in range(sample_xyz.shape[0]):
            delta = np.abs(center_xyz - sample_xyz[i]) * spacing
            delta = np.minimum(delta, box - delta)
            contains = np.sum(delta * delta, axis=1) <= center_radii**2
            assigned[i] = np.max(center_radii[contains])

        n_bins = int(np.ceil((2.0 * center_radii.max()) / bin_size))
        expected_hist, _ = np.histogram(
            2.0 * assigned,
            bins=np.arange(n_bins + 1) * bin_size,
        )
        np.testing.assert_array_equal(data[:, 2], expected_hist)
        expected_promoted = np.mean(
            assigned > dmin.ravel()[sampled_flat] + 1e-7
        )
        self.assertAlmostEqual(promoted, float(expected_promoted))

if __name__ == "__main__":
    unittest.main()
