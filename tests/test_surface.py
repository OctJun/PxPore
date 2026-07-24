import unittest

from _test_env import configure_test_threads

configure_test_threads()

import numpy as np

from PxPore.geometry import build_cell_list
from PxPore.surface import fibonacci_sphere_surface_area


class SurfaceTests(unittest.TestCase):
    def test_single_sphere_matches_analytic_area(self):
        box = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        pos = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        radius = np.array([0.2], dtype=np.float64)
        grid_info = (20, 20, 20, 0.1, 0.1, 0.1)
        grid_mask = np.zeros((20, 20, 20), dtype=np.uint8)
        label_mask = np.full((20, 20, 20), 2, dtype=np.int8)
        cell_list = build_cell_list(pos, box, 0.2)
        expected = 4.0 * np.pi * radius[0] ** 2

        for samples in (32, 1000):
            with self.subTest(samples=samples):
                accessible, total = fibonacci_sphere_surface_area(
                    pos,
                    radius,
                    box,
                    0.0,
                    grid_info,
                    grid_mask,
                    label_mask,
                    cell_list,
                    None,
                    nsample=samples,
                )
                self.assertAlmostEqual(accessible, expected, places=12)
                self.assertAlmostEqual(total, expected, places=12)


if __name__ == "__main__":
    unittest.main()
