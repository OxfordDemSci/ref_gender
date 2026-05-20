import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.pipeline_manifest import append_manifest_row


class PipelineManifestTest(unittest.TestCase):
    def test_append_manifest_row_creates_header_and_row(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            infile = root / "in.txt"
            outfile = root / "out.txt"
            infile.write_text("input", encoding="utf-8")
            outfile.write_text("output", encoding="utf-8")

            append_manifest_row(
                manifest_path=manifest,
                project_root=root,
                step="unit_test",
                status="success",
                started_at_utc="2026-01-01T00:00:00+00:00",
                finished_at_utc="2026-01-01T00:00:01+00:00",
                duration_seconds=1.0,
                parameters={"a": 1},
                input_paths={"in": infile},
                output_paths={"out": outfile},
                row_counts={"n": 2},
                notes="ok",
            )

            with manifest.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], "unit_test")
            self.assertEqual(json.loads(rows[0]["input_paths"])["in"], "in.txt")
            self.assertEqual(json.loads(rows[0]["output_paths"])["out"], "out.txt")


if __name__ == "__main__":
    unittest.main()
