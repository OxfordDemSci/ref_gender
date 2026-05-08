import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from src.pipeline_io import atomic_write_parquet
from src.step03_get_dimensions_research_outputs import _assemble_outputs_with_authors
from src.step03_get_dimensions_research_outputs import _infer_gender_list
from src.step03_get_dimensions_research_outputs import _normalise_complex_output_columns


class _PrimaryDetector:
    def get_gender(self, name):
        return {
            "Alice": "female",
            "Bob": "mostly_male",
            "Casey": "unknown",
            "Pat": "andy",
        }.get(name, "unknown")


class _FallbackDetector:
    def guess(self, name):
        return {
            "Casey": "female",
            "Pat": "unknown",
            "Sam": "male",
        }.get(name, "unknown")


class Step03GenderInferenceTest(unittest.TestCase):
    def test_gender_detector_fallback_only_for_unknown_primary_labels(self):
        detector = (_PrimaryDetector(), _FallbackDetector())

        out = _infer_gender_list(["Alice", "Bob", "Casey", "Pat", "Sam", ""], detector)

        self.assertEqual(out, ["female", "male", "female", "unknown", "male", "unknown"])

    def test_complex_dimensions_columns_are_parquet_safe_json_strings(self):
        df = pd.DataFrame(
            {
                "authors": [
                    [{"first_name": "A", "corresponding": True}],
                    [{"first_name": "B", "corresponding": ""}],
                    None,
                ],
                "category_for_2020": [[{"id": "1", "name": "Field"}], None, ""],
                "author_forenames": [["A"], ["B"], []],
                "author_genders": [["female"], ["unknown"], []],
            }
        )

        out = _normalise_complex_output_columns(df)

        self.assertTrue(out["authors"].map(lambda value: isinstance(value, str)).all())
        self.assertIn('"corresponding": true', out.loc[0, "authors"])
        self.assertEqual(out.loc[2, "authors"], "")
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.parquet"
            atomic_write_parquet(out, path)
            reread = pd.read_parquet(path)
            self.assertEqual(len(reread), 3)

    def test_assemble_outputs_prefers_doi_match_and_keeps_zero_author_rows(self):
        raw = pd.DataFrame(
            {
                "REF2ID": ["1", "2", "3"],
                "DOI": ["10.1/a", "", "10.1/missing"],
                "ISBN": ["9780000000001", "9780000000002", ""],
            }
        )
        dim_df = pd.DataFrame(
            {
                "id": ["doi-hit", "isbn-hit-a", "isbn-hit-b"],
                "doi": ["10.1/a", "", ""],
                "isbn": ["", "9780000000001", "9780000000002"],
                "authors": [
                    '[{"first_name": "Alice"}]',
                    '[{"first_name": "Bob"}]',
                    '[{"first_name": "Casey"}]',
                ],
                "category_for_2020": ["doi-cat", "isbn-cat-a", "isbn-cat-b"],
                "year": [2020, 2021, 2022],
                "doi_norm": ["10.1/a", "", ""],
                "isbn_norm": ["", "9780000000001", "9780000000002"],
            }
        )

        with TemporaryDirectory() as tmp:
            path = f"{tmp}/raw_outputs.xlsx"
            with pd.ExcelWriter(path) as writer:
                raw.to_excel(writer, index=False, startrow=4)

            any_authors, positive_authors = _assemble_outputs_with_authors(path, dim_df)

        self.assertEqual(len(any_authors), 3)
        self.assertEqual(len(positive_authors), 2)
        self.assertEqual(int(any_authors.loc[0, "number_people"]), 1)
        self.assertIn("Alice", any_authors.loc[0, "author_forenames"])
        self.assertNotIn("Bob", any_authors.loc[0, "author_forenames"])
        self.assertEqual(any_authors.loc[0, "category_for_2020"], "doi-cat")
        self.assertEqual(int(any_authors.loc[1, "number_people"]), 1)
        self.assertEqual(int(any_authors.loc[2, "number_people"]), 0)


if __name__ == "__main__":
    unittest.main()
