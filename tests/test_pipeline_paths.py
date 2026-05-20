from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.pipeline_paths import (
    build_paths,
    format_relative_path,
    normalize_path_values,
    resolve_enhanced_ref_data_path,
    resolve_outputs_concat_path,
)


class PipelinePathsTest(unittest.TestCase):
    def test_resolve_prefers_analysis_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_paths(project_root=root)
            (paths.analysis_dir).mkdir(parents=True, exist_ok=True)
            (paths.legacy_final_dir).mkdir(parents=True, exist_ok=True)
            analysis = paths.analysis_dir / "enhanced_ref_data.parquet"
            legacy = paths.legacy_final_dir / "enhanced_ref_data.csv"
            analysis.write_bytes(b"PAR1fake")
            legacy.write_text("a,b\n3,4\n", encoding="utf-8")
            self.assertEqual(resolve_enhanced_ref_data_path(paths), analysis)

    def test_resolve_outputs_falls_back_to_legacy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_paths(project_root=root)
            paths.legacy_dimensions_dir.mkdir(parents=True, exist_ok=True)
            legacy = paths.legacy_dimensions_dir / "outputs_concat_with_positive_authors.csv"
            legacy.write_text("x,y\n1,2\n", encoding="utf-8")
            self.assertEqual(resolve_outputs_concat_path(paths), legacy)

    def test_format_relative_path_and_normalize_path_values(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "outputs" / "tables" / "report.txt"
            self.assertEqual(format_relative_path(report, root), "outputs/tables/report.txt")

            payload = normalize_path_values(
                {
                    "input_path": str(report),
                    "artifacts": [report],
                    "nested": {"output_path": report},
                },
                root,
            )
            self.assertEqual(payload["input_path"], "outputs/tables/report.txt")
            self.assertEqual(payload["artifacts"], ["outputs/tables/report.txt"])
            self.assertEqual(payload["nested"]["output_path"], "outputs/tables/report.txt")


if __name__ == "__main__":
    unittest.main()
