import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from leitor_gas import persistence


class PersistenceTests(unittest.TestCase):
    def test_calibration_round_trip(self):
        data = {
            "width": 640,
            "height": 480,
            "backend": "CPU",
            "rois": [{"x1": 10, "y1": 20, "x2": 30, "y2": 40}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            persistence.save_calibration(path, data)

            self.assertEqual(persistence.load_calibration(path), data)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_calibration_rejects_out_of_bounds_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "width": 20,
                        "height": 20,
                        "rois": [{"x1": 0, "y1": 0, "x2": 21, "y2": 10}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                persistence.load_calibration(path)

    def test_editing_calibration_validates_and_sorts_digit_positions(self):
        calibration = {
            "width": 100,
            "height": 50,
            "rois": [
                {"x1": 60, "y1": 5, "x2": 70, "y2": 25},
                {"x1": 10, "y1": 5, "x2": 20, "y2": 25},
            ],
        }

        updated = persistence.update_calibration_regions(
            calibration,
            [(30, 5, 40, 25), (5, 5, 15, 25)],
        )

        self.assertEqual(updated["rois"][0]["x1"], 5)
        self.assertEqual(updated["rois"][1]["x1"], 30)
        self.assertEqual(calibration["rois"][0]["x1"], 60)

    def test_editing_calibration_rejects_invalid_coordinates(self):
        calibration = {
            "width": 100,
            "height": 50,
            "rois": [{"x1": 10, "y1": 5, "x2": 20, "y2": 25}],
        }

        with self.assertRaises(ValueError):
            persistence.update_calibration_regions(
                calibration, [(10, 5, 101, 25)]
            )

    def test_append_reading_writes_header_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readings.csv"
            persistence.append_reading(path, ["2026-10-30", "camera", "123.4", "CPU", 4])
            persistence.append_reading(path, ["2026-10-31", "camera", "123.5", "CPU", 4])

            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file, delimiter=";"))

        self.assertEqual(rows[0], ["data_hora", "fonte", "leitura", "backend", "rois"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2][2], "123.5")

    def test_append_reading_reports_filesystem_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-file"
            path.mkdir()

            with self.assertRaises(OSError):
                persistence.append_reading(path, ["row"])

    def test_legacy_migration_preserves_source_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            current = root / "current"
            source_calibration_dir = legacy / "calibracoes"
            source_calibration_dir.mkdir(parents=True)
            (source_calibration_dir / "saved.json").write_text("old", encoding="utf-8")
            (legacy / "leituras.csv").write_text("old csv", encoding="utf-8")
            destination_calibration_dir = current / "calibracoes"
            destination_calibration_dir.mkdir(parents=True)
            (destination_calibration_dir / "saved.json").write_text(
                "new", encoding="utf-8"
            )

            with (
                patch.object(persistence, "LEGACY_RESULTS_DIR", legacy),
                patch.object(persistence, "RESULTS_DIR", current),
                patch.object(
                    persistence,
                    "CALIBRATIONS_DIR",
                    destination_calibration_dir,
                ),
            ):
                error = persistence.migrate_legacy_results()

            self.assertIsNone(error)
            self.assertEqual(
                (destination_calibration_dir / "saved.json").read_text(
                    encoding="utf-8"
                ),
                "new",
            )
            self.assertTrue((legacy / "leituras.csv").exists())
            self.assertEqual((current / "leituras.csv").read_text(encoding="utf-8"), "old csv")


if __name__ == "__main__":
    unittest.main()
