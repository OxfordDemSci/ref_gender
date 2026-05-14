from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.pipeline_io import read_table


class PipelineIoTest(unittest.TestCase):
    def test_read_table_falls_back_to_csv_when_parquet_engine_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "sample.parquet"
            csv_path = root / "sample.csv"
            parquet_path.write_bytes(b"PAR1fake")
            csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

            with patch("src.pipeline_io.pd.read_parquet", side_effect=ImportError("missing parquet engine")):
                df = read_table(parquet_path)

            self.assertEqual(df.to_dict(orient="records"), [{"a": 1, "b": 2}])


if __name__ == "__main__":
    unittest.main()
