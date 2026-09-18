import itertools
import unittest

from _test_env import configure_test_threads

configure_test_threads()

import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.ndimage import label

from PxPore.connectivity_multicore import (
    percolation_masks,
    percolation_masks_directional,
    percolation_masks_periodic,
)


NEIGHBOR6 = np.zeros((3, 3, 3), dtype=np.uint8)
NEIGHBOR6[1, 1, 1] = 1
NEIGHBOR6[0, 1, 1] = NEIGHBOR6[2, 1, 1] = 1
NEIGHBOR6[1, 0, 1] = NEIGHBOR6[1, 2, 1] = 1
NEIGHBOR6[1, 1, 0] = NEIGHBOR6[1, 1, 2] = 1


def tiled_reference(void_mask, direction):
    """使用三倍周期复制获得独立的 winding 参考结果。"""
    tiled = np.tile(void_mask, (3, 3, 3))
    components, _ = label(tiled, structure=NEIGHBOR6)
    shape = void_mask.shape
    center_slice = tuple(slice(n, 2 * n) for n in shape)
    center = components[center_slice]
    selected = (
        range(3) if direction == "any"
        else ("xyz".index(direction),)
    )
    accessible = np.zeros_like(void_mask, dtype=bool)

    for shift in itertools.product((-1, 0, 1), repeat=3):
        if shift == (0, 0, 0):
            continue
        if not any(shift[axis] != 0 for axis in selected):
            continue
        shifted_slice = tuple(
            slice((1 + step) * n, (2 + step) * n)
            for step, n in zip(shift, shape)
        )
        accessible |= (
            (center != 0)
            & (center == components[shifted_slice])
        )

    expected = np.zeros_like(void_mask, dtype=np.int8)
    expected[void_mask != 0] = 1
    expected[accessible] = 2
    return expected


class ConnectivityTests(unittest.TestCase):
    def test_channels_across_slab_boundaries(self):
        # 覆盖完整分块、末尾不完整分块，以及接缝处被封闭的通道。
        available = get_num_threads()
        try:
            for threads in sorted({1, min(8, available)}):
                set_num_threads(threads)
                for gz in (31, 32, 33, 65):
                    for closed in (False, True):
                        void = np.zeros((5, 5, gz), dtype=np.uint8)
                        void[2, 2, :] = 1
                        if closed:
                            void[2, 2, min(32, gz // 2)] = 0
                        expected = void.astype(np.int8) * (1 if closed else 2)
                        with self.subTest(threads=threads, gz=gz, closed=closed):
                            observed, _ = percolation_masks(void)
                            np.testing.assert_array_equal(observed, expected)
                            observed, _ = percolation_masks_periodic(void, 'z')
                            np.testing.assert_array_equal(observed, expected)
                            observed, _ = percolation_masks_directional(void, 'z')
                            np.testing.assert_array_equal(observed, expected)
        finally:
            set_num_threads(available)

    def test_periodic_voxel_truth_models(self):
        xy_open = np.zeros((7, 7, 7), dtype=np.uint8)
        xy_open[:, :, 3] = 1

        z_channel = np.zeros((7, 7, 7), dtype=np.uint8)
        z_channel[3, 3, :] = 1

        single_seam = np.zeros((7, 7, 7), dtype=np.uint8)
        single_seam[3, 3, 0] = 1
        single_seam[3, 3, -1] = 1

        diagonal_xy = np.zeros((7, 7, 3), dtype=np.uint8)
        for index in range(7):
            diagonal_xy[index, index, 1] = 1
            diagonal_xy[(index + 1) % 7, index, 1] = 1

        models = {
            "xy_open_z_closed": (
                xy_open,
                {"x": True, "y": True, "z": False, "any": True},
            ),
            "z_channel": (
                z_channel,
                {"x": False, "y": False, "z": True, "any": True},
            ),
            "single_periodic_seam": (
                single_seam,
                {"x": False, "y": False, "z": False, "any": False},
            ),
            "diagonal_xy_winding": (
                diagonal_xy,
                {"x": True, "y": True, "z": False, "any": True},
            ),
        }
        for name, (void_mask, expected) in models.items():
            for direction, accessible in expected.items():
                with self.subTest(model=name, direction=direction):
                    labels, _ = percolation_masks_periodic(
                        void_mask, direction)
                    self.assertEqual(
                        bool(np.any(labels == 2)), accessible)

    def test_single_periodic_seam_is_not_winding(self):
        void_mask = np.zeros((7, 5, 3), dtype=np.uint8)
        void_mask[0, 2, 1] = 1
        void_mask[-1, 2, 1] = 1
        labels, _ = percolation_masks_periodic(void_mask, "x")
        self.assertEqual(np.count_nonzero(labels == 2), 0)
        self.assertEqual(np.count_nonzero(labels == 1), 2)

    def test_directional_periodic_rings(self):
        void_mask = np.zeros((7, 7, 7), dtype=np.uint8)
        void_mask[:, 3, 3] = 1
        void_mask[3, :, 3] = 1
        expected = {"x": True, "y": True, "z": False, "any": True}
        for direction, accessible in expected.items():
            with self.subTest(direction=direction):
                labels, _ = percolation_masks_periodic(
                    void_mask, direction)
                self.assertEqual(
                    bool(np.any(labels == 2)), accessible)

    def test_nonperiodic_direction_selection(self):
        void_mask = np.zeros((7, 7, 7), dtype=np.uint8)
        void_mask[:, 3, 3] = 1
        self.assertTrue(np.any(
            percolation_masks_directional(void_mask, "x")[0] == 2))
        self.assertFalse(np.any(
            percolation_masks_directional(void_mask, "y")[0] == 2))
        legacy, _ = percolation_masks(void_mask)
        directional_any, _ = percolation_masks_directional(
            void_mask, "any")
        np.testing.assert_array_equal(legacy, directional_any)

    def test_degenerate_and_noncontiguous_inputs(self):
        void_mask = np.ones((1, 4, 3), dtype=bool)
        labels, _ = percolation_masks_periodic(void_mask, "x")
        self.assertTrue(np.all(labels == 2))

        transposed = np.zeros((5, 4, 3), dtype=np.uint8).transpose(2, 1, 0)
        transposed[:, 2, 2] = 1
        labels, _ = percolation_masks_periodic(transposed, "x")
        self.assertTrue(np.any(labels == 2))

    def test_random_masks_match_tiled_reference(self):
        rng = np.random.default_rng(11451466)
        for shape in ((2, 2, 2), (3, 3, 3), (4, 3, 2)):
            for _ in range(12):
                void_mask = (rng.random(shape) < 0.4).astype(np.uint8)
                for direction in ("x", "y", "z", "any"):
                    with self.subTest(shape=shape, direction=direction):
                        observed, _ = percolation_masks_periodic(
                            void_mask, direction)
                        expected = tiled_reference(
                            void_mask, direction)
                        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
