import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_regression_suite.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_regression_suite", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RegressionRunnerTests(unittest.TestCase):
    def test_runner_does_not_import_pxpore(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from PxPore", source)
        self.assertNotIn("import PxPore", source)

    def test_empty_manifest_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "cases": [],
            }), encoding="utf-8")
            self.assertEqual(RUNNER.load_manifest(path), [])

    def test_metric_path_lookup(self):
        payload = {"stats": {"Vacc_nm3": 1.25}}
        self.assertEqual(
            RUNNER.get_metric(payload, "stats.Vacc_nm3"), 1.25)
        with self.assertRaises(KeyError):
            RUNNER.get_metric(payload, "stats.missing")

    def test_numeric_and_exact_comparisons(self):
        self.assertTrue(RUNNER.compare_values(
            1.00001, 1.0, 0.0, 1e-4))
        self.assertFalse(RUNNER.compare_values(
            1.1, 1.0, 0.0, 1e-4))
        self.assertTrue(RUNNER.compare_values(
            "periodic", "periodic", 0.0, 0.0))
        self.assertFalse(RUNNER.compare_values(
            "legacy", "periodic", 0.0, 0.0))

    def test_skipped_case_reports_each_expected_metric(self):
        records = RUNNER.run_case({
            "name": "slow-case",
            "input": "not-read-when-skipped.gro",
            "slow": True,
            "metrics": {
                "stats.Vacc_nm3": {"reference": 1.25},
            },
        }, threads=2)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            [record["status"] for record in records],
            ["SKIP"],
        )
        self.assertEqual(
            [record["reference"] for record in records], [1.25])
        self.assertEqual(
            [record["observed"] for record in records], [None])

    def test_arguments_are_converted_to_cli_options(self):
        self.assertEqual(
            RUNNER.arguments_to_cli({
                "grid": 0.01,
                "no_surface": True,
                "pore": False,
                "psd_mc_bin_size": None,
                "threads": 2,
                "stats": True,
            }),
            ["--grid", "0.01", "--no-surface"],
        )

    def test_output_includes_name_result_output_and_expected_value(self):
        records = [{
            "case": "sample-case",
            "metric": "stats.Vacc_nm3",
            "status": "PASS",
            "observed": 1.25,
            "reference": 1.25,
        }]
        output = io.StringIO()
        with redirect_stdout(output):
            RUNNER.print_records(records, use_color=False)
        lines = output.getvalue().splitlines()
        self.assertEqual(
            [cell.strip() for cell in lines[0].split("|")],
            ["RESULT", "TEST NAME", "OUTPUT VALUE", "EXPECTED VALUE", "METRIC"],
        )
        self.assertTrue(lines[2].startswith(
            "PASS   | sample-case"))

    def test_status_color_is_applied_when_enabled(self):
        output = io.StringIO()
        with redirect_stdout(output):
            RUNNER.print_records([{
                "case": "sample-case",
                "metric": "stats.Vacc_nm3",
                "status": "PASS",
                "observed": 1.25,
                "reference": 1.25,
            }], use_color=True)
        self.assertIn("\033[32mPASS", output.getvalue())

    def test_case_runs_from_staged_input(self):
        gro_text = (
            "single H regression fixture\n"
            "1\n"
            "    1MOL      H    1   0.200   0.200   0.200\n"
            "   0.40000   0.40000   0.40000\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "single_H.gro"
            input_path.write_text(gro_text, encoding="ascii")
            records = RUNNER.run_case({
                "name": "runner-smoke",
                "input": str(input_path),
                "arguments": {
                    "grid": 0.05,
                    "no_surface": True,
                    "no_octree": True,
                },
                "metrics": {
                    "settings.psd_method": {
                        "reference": "mc",
                    },
                    "stats.Vacc_nm3": {
                        "reference": None,
                    },
                },
            }, threads=2)
        statuses = {
            record["metric"]: record["status"] for record in records
        }
        self.assertEqual(statuses["settings.psd_method"], "PASS")
        self.assertEqual(statuses["stats.Vacc_nm3"], "UNSET")


if __name__ == "__main__":
    unittest.main()
