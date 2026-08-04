import unittest

from _test_env import configure_test_threads

configure_test_threads()

import numpy as np
from numba import get_num_threads, set_num_threads

from PxPore.connectivity_multicore import (
    percolation_masks_directional,
    percolation_masks_periodic,
)
from PxPore.octree import OCC_VOID
from PxPore.octree_connectivity_hybrid import (
    percolation_masks_with_octree_hybrid,
)


def single_leaf_forest(void_mask, refined_mask):
    """Represent every selected coarse voxel by one level-0 void leaf."""
    roots = np.argwhere(refined_mask)
    count = roots.shape[0]
    centers = [
        np.asarray(roots[:, axis] + 0.5, dtype=np.float32)
        for axis in range(3)
    ]
    distance = np.ones(count, dtype=np.float32)
    parent = np.full(count, -1, dtype=np.int32)
    child = np.full(count, -1, dtype=np.int32)
    level = np.zeros(count, dtype=np.uint8)
    occupancy = np.where(
        void_mask[tuple(roots.T)] != 0, OCC_VOID, 0
    ).astype(np.uint8)
    grid_mask = np.zeros(void_mask.shape, dtype=np.uint8)
    grid_mask[refined_mask] |= np.uint8(128)
    forest = (
        centers[0], centers[1], centers[2], distance,
        parent, child, level, occupancy,
    )
    shape = void_mask.shape
    grid_info = (*shape, 1.0, 1.0, 1.0)
    return grid_mask, grid_info, forest


def hybrid_labels(void_mask, refined_mask, connectivity, direction):
    grid_mask, grid_info, forest = single_leaf_forest(
        void_mask, refined_mask
    )
    labels, _ = percolation_masks_with_octree_hybrid(
        void_mask,
        grid_mask,
        grid_info,
        forest,
        connectivity,
        direction,
    )
    return labels


class OctreeConnectivityHybridTests(unittest.TestCase):
    def test_all_refined_matches_uniform_directional(self):
        void_mask = np.zeros((7, 5, 3), dtype=np.uint8)
        void_mask[:, 2, 1] = 1
        refined = void_mask != 0
        for direction in ("x", "y", "z", "any"):
            with self.subTest(direction=direction):
                expected, _ = percolation_masks_directional(
                    void_mask, direction
                )
                observed = hybrid_labels(
                    void_mask, refined, "legacy", direction
                )
                np.testing.assert_array_equal(observed, expected)

    def test_mixed_coarse_leaf_periodic_ring(self):
        void_mask = np.zeros((8, 5, 3), dtype=np.uint8)
        void_mask[:, 2, 1] = 1
        refined = np.zeros_like(void_mask, dtype=bool)
        refined[::2, 2, 1] = True
        for direction in ("x", "y", "z", "any"):
            with self.subTest(direction=direction):
                expected, _ = percolation_masks_periodic(
                    void_mask, direction
                )
                observed = hybrid_labels(
                    void_mask, refined, "periodic", direction
                )
                np.testing.assert_array_equal(observed, expected)

    def test_leaf_leaf_periodic_seam_is_not_itself_winding(self):
        void_mask = np.zeros((7, 5, 3), dtype=np.uint8)
        void_mask[0, 2, 1] = 1
        void_mask[-1, 2, 1] = 1
        refined = void_mask != 0
        expected, _ = percolation_masks_periodic(void_mask, "x")
        observed = hybrid_labels(
            void_mask, refined, "periodic", "x"
        )
        np.testing.assert_array_equal(observed, expected)

    def test_results_are_thread_count_independent(self):
        available = get_num_threads()
        if available < 2:
            self.skipTest("requires at least two Numba threads")
        void_mask = np.zeros((12, 7, 5), dtype=np.uint8)
        void_mask[:, 3, 2] = 1
        void_mask[6, :, 2] = 1
        refined = np.zeros_like(void_mask, dtype=bool)
        refined[void_mask != 0] = (
            np.indices(void_mask.shape).sum(axis=0)[void_mask != 0] % 2 == 0
        )
        try:
            set_num_threads(1)
            serial = hybrid_labels(
                void_mask, refined, "periodic", "any"
            )
            set_num_threads(min(4, available))
            parallel = hybrid_labels(
                void_mask, refined, "periodic", "any"
            )
        finally:
            set_num_threads(available)
        np.testing.assert_array_equal(parallel, serial)


if __name__ == "__main__":
    unittest.main()
