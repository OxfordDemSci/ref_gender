from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figure_one_data import prepare_figure_one_data
from src.figure_one_plots import plot_figure_one
from src.figure_two_llm import load_llm_tables
from src.step05_build_regression_models import build_and_save_models


def _make_enhanced_df() -> pd.DataFrame:
    rows = []
    for i, panel in enumerate(["A", "B", "C", "D"] * 3, start=1):
        male = 8 + (i % 5)
        female = 6 + (i % 4)
        rows.append(
            {
                "REF impact case study identifier": f"ics-{i}",
                "Institution name": f"Uni {i % 4}",
                "Unit of assessment number": 10 + (i % 4),
                "Main Panel": panel,
                "inst_id": 1000 + (i % 4),
                "uoa_id": str(10 + (i % 4)),
                "number_male": male,
                "number_female": female,
                "number_unknown": 0,
                "number_people": male + female,
                "OxBridge": 1 if i % 6 == 0 else 0,
                "RussellGroup": 1 if i % 2 == 0 else 0,
                "Redbrick": 1 if i % 3 == 0 else 0,
                "Ancient": 1 if i % 5 == 0 else 0,
                "llm_museum": i % 2,
                "llm_nhs": i % 3 == 0,
                "llm_drug_trial": i % 4 == 0,
                "llm_school": i % 5 == 0,
                "llm_legislation": i % 3 == 0,
                "llm_heritage": i % 4 == 0,
                "llm_manufacturing": i % 2 == 0,
                "llm_software": i % 2 == 1,
                "llm_patent": i % 3 == 1,
                "llm_startup": i % 4 == 1,
                "llm_charity": i % 5 == 1,
            }
        )
    df = pd.DataFrame(rows)
    llm_cols = [c for c in df.columns if c.startswith("llm_")]
    for c in llm_cols:
        df[c] = df[c].astype(int)
    return df


def _make_outputs_df() -> pd.DataFrame:
    rows = []
    for i in range(1, 25):
        male = 5 + (i % 3)
        female = 4 + (i % 2)
        rows.append(
            {
                "Institution UKPRN code": 1000 + (i % 4),
                "Institution name": f"Uni {i % 4}",
                "Main panel": ["A", "B", "C", "D"][i % 4],
                "Unit of assessment number": 10 + (i % 4),
                "number_male": male,
                "number_female": female,
                "number_unknown": 0,
                "number_people": male + female,
            }
        )
    return pd.DataFrame(rows)


class SmokeAnalysisTest(unittest.TestCase):
    def test_smoke_pipeline_components(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            (data_root / "gold").mkdir(parents=True, exist_ok=True)
            (data_root / "manual" / "university_category").mkdir(parents=True, exist_ok=True)
            (root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

            enhanced_path = data_root / "gold" / "enhanced_ref_data.csv"
            outputs_path = data_root / "gold" / "outputs_concat_with_positive_authors.csv"
            uniclass_path = data_root / "manual" / "university_category" / "ref_unique_institutions.csv"

            _make_enhanced_df().to_csv(enhanced_path, index=False)
            _make_outputs_df().to_csv(outputs_path, index=False)
            pd.DataFrame({"Institution name": [f"Uni {i}" for i in range(4)], "RussellGroup": [0, 1, 0, 1]}).to_csv(
                uniclass_path, index=False
            )

            df_ics, df_uoa_m, _df_uni_m, _df_uniuoa_m = prepare_figure_one_data(data_root, uniclass_path=uniclass_path)
            fig, _axes = plot_figure_one(df_ics, df_uoa_m, show_unit_names=False)
            self.assertIsNotNone(fig)

            out_model = root / "outputs" / "models" / "regression_results.pkl"
            build_and_save_models(data_csv_path=enhanced_path, out_path=out_model)
            self.assertTrue(out_model.exists())

            llm_overall, llm_by_panel = load_llm_tables(data_root=data_root)
            self.assertGreater(len(llm_overall), 0)
            self.assertGreater(len(llm_by_panel), 0)


if __name__ == "__main__":
    unittest.main()

