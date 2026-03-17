from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.pipeline_manifest import append_manifest_row


class PipelineManifestTest(unittest.TestCase):
    def test_append_manifest_row_creates_header_and_row(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            infile = Path(tmp) / "in.txt"
            outfile = Path(tmp) / "out.txt"
            infile.write_text("input", encoding="utf-8")
            outfile.write_text("output", encoding="utf-8")

            append_manifest_row(
                manifest_path=manifest,
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

            lines = manifest.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)
            self.assertIn("unit_test", lines[1])


if __name__ == "__main__":
    unittest.main()

