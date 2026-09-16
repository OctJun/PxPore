import tempfile
import unittest
from pathlib import Path

from _test_env import configure_test_threads

configure_test_threads()

from PxPore import analyse


class NoPoreTests(unittest.TestCase):
    def test_fully_occupied_system_writes_empty_pore_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            structure = root / "fully_occupied.gro"
            positions = [
                (x, y, z)
                for x in (0.1, 0.3)
                for y in (0.1, 0.3)
                for z in (0.1, 0.3)
            ]
            lines = ["fully occupied 2x2x2 carbon lattice", str(len(positions))]
            for index, (x, y, z) in enumerate(positions, start=1):
                lines.append(
                    f"{1:5d}{'BEN':<5}{'C':>5}{index:5d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                )
            lines.append(f"{0.4:10.5f}{0.4:10.5f}{0.4:10.5f}")
            structure.write_text("\n".join(lines) + "\n", encoding="ascii")

            result = analyse(
                str(structure),
                grid=0.2,
                pore=True,
                stats=True,
                no_octree=True,
                no_surface=True,
                out_prefix="no_pore",
            )

            stats = result["stats"]["stats"]
            self.assertEqual(stats["Vacc_nm3"], 0.0)
            self.assertEqual(stats["PLD_nm"], -1.0)
            self.assertEqual(stats["LCD_nm"], -1.0)
            self.assertEqual(stats["LCD_global_nm"], -1.0)
            self.assertTrue((root / "no_pore_stats.json").is_file())
            self.assertTrue((root / "no_pore_voxel_mc_psd.txt").is_file())
            self.assertFalse((root / "no_pore_center.txt").exists())
            self.assertFalse((root / "no_pore_psd.txt").exists())


if __name__ == "__main__":
    unittest.main()
