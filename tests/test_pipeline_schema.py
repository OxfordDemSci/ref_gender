import unittest

import pandas as pd

from src.pipeline_schema import validate_enhanced_ref_data, validate_outputs_concat


class PipelineSchemaTest(unittest.TestCase):
    def test_validate_enhanced_ref_data_pass(self):
        df = pd.DataFrame(
            {
                "REF impact case study identifier": ["id1"],
                "Institution name": ["Inst A"],
                "Unit of assessment number": [11],
                "number_male": [2],
                "number_female": [3],
                "number_unknown": [0],
                "number_people": [5],
            }
        )
        out = validate_enhanced_ref_data(df, use_pandera=False)
        self.assertEqual(len(out), 1)

    def test_validate_outputs_concat_fails_missing_column(self):
        df = pd.DataFrame(
            {
                "Institution name": ["Inst A"],
                "Unit of assessment number": [11],
            }
        )
        with self.assertRaises(ValueError):
            validate_outputs_concat(df, use_pandera=False)


if __name__ == "__main__":
    unittest.main()

