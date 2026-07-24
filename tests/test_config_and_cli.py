import inspect
import unittest

from _test_env import configure_test_threads

configure_test_threads()

from PxPore import analyse
from PxPore.api import analyse as api_analyse
from PxPore.cli import build_parser, namespace_to_config
from PxPore.config import AnalyseConfig


class ConfigAndCliTests(unittest.TestCase):
    def test_new_defaults_preserve_legacy_behavior(self):
        cfg = namespace_to_config(
            build_parser().parse_args(["sample.gro"]))
        self.assertEqual(cfg.connectivity, "legacy")
        self.assertEqual(cfg.transport_direction, "any")
        self.assertFalse(cfg.no_octree)
        self.assertEqual(cfg.psd_method, "centers")
        self.assertEqual(cfg.psd_mc_samples, 50000)
        self.assertEqual(cfg.psd_mc_seed, 11451466)
        self.assertIsNone(cfg.psd_mc_bin_size)
        self.assertEqual(cfg.surface_samples, 1000)

    def test_custom_cli_values_reach_config(self):
        cfg = namespace_to_config(build_parser().parse_args([
            "sample.gro",
            "--connectivity", "periodic",
            "--transport-direction", "z",
            "--no-octree",
            "--psd-method", "mc",
            "--psd-mc-samples", "1234",
            "--psd-mc-seed", "99",
            "--psd-mc-bin-size", "0.025",
            "--surface-samples", "2500",
        ]))
        self.assertEqual(cfg.connectivity, "periodic")
        self.assertEqual(cfg.transport_direction, "z")
        self.assertTrue(cfg.no_octree)
        self.assertEqual(cfg.psd_method, "mc")
        self.assertEqual(cfg.psd_mc_samples, 1234)
        self.assertEqual(cfg.psd_mc_seed, 99)
        self.assertEqual(cfg.psd_mc_bin_size, 0.025)
        self.assertEqual(cfg.surface_samples, 2500)

    def test_api_parameters_are_appended(self):
        names = list(inspect.signature(api_analyse).parameters)
        self.assertEqual(names[-7:], [
            "connectivity",
            "transport_direction",
            "psd_method",
            "psd_mc_samples",
            "psd_mc_seed",
            "psd_mc_bin_size",
            "surface_samples",
        ])

    def test_dataclass_parameters_are_appended(self):
        names = list(AnalyseConfig.__dataclass_fields__)
        self.assertEqual(names[-7:], [
            "connectivity",
            "transport_direction",
            "psd_method",
            "psd_mc_samples",
            "psd_mc_seed",
            "psd_mc_bin_size",
            "surface_samples",
        ])

    def test_invalid_new_values_fail_before_input(self):
        with self.assertRaisesRegex(ValueError, "psd_method"):
            analyse("missing.gro", psd_method="invalid")
        with self.assertRaisesRegex(ValueError, "psd_mc_samples"):
            analyse("missing.gro", psd_method="mc", psd_mc_samples=0)
        with self.assertRaisesRegex(ValueError, "psd_mc_bin_size"):
            analyse("missing.gro", psd_mc_bin_size=0.0)
        with self.assertRaisesRegex(ValueError, "surface_samples"):
            analyse("missing.gro", surface_samples=0)
        with self.assertRaisesRegex(ValueError, "--no-octree"):
            analyse("missing.gro", connectivity="periodic")


if __name__ == "__main__":
    unittest.main()
