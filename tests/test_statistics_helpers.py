from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.statistics_helpers import _panel_table, _physics_vs_chemistry_table, _uoa_table


TOPIC_COLS = [
    "llm_charity",
    "llm_nhs",
    "llm_museum",
    "llm_school",
    "llm_legislation",
    "llm_heritage",
    "llm_software",
    "llm_startup",
    "llm_patent",
    "llm_manufacturing",
    "llm_drug_trial",
]


def _make_coef_df() -> pd.DataFrame:
    ols_values = {
        "llm_charity": 0.09,
        "llm_nhs": 0.07,
        "llm_school": 0.06,
        "llm_museum": 0.05,
        "llm_legislation": 0.01,
        "llm_heritage": -0.02,
        "llm_startup": -0.03,
        "llm_software": -0.04,
        "llm_manufacturing": -0.05,
        "llm_patent": -0.06,
        "llm_drug_trial": -0.08,
    }
    glm_values = {
        "llm_charity": 0.40,
        "llm_nhs": 0.30,
        "llm_school": 0.25,
        "llm_museum": 0.20,
        "llm_legislation": 0.05,
        "llm_heritage": -0.10,
        "llm_startup": -0.15,
        "llm_software": -0.20,
        "llm_manufacturing": -0.30,
        "llm_patent": -0.40,
        "llm_drug_trial": -0.50,
    }
    rows = []
    for variable, coef in ols_values.items():
        rows.append({"variable": variable, "coef": coef, "model": "OLS (3)"})
    for variable, coef in glm_values.items():
        rows.append({"variable": variable, "coef": coef, "model": "GLM (3)"})
    return pd.DataFrame(rows)


def _make_ics_df() -> pd.DataFrame:
    rows = [
        {
            "Unit of assessment number": 9,
            "llm_charity": 1,
            "llm_nhs": 1,
            "llm_museum": 0,
            "llm_school": 1,
            "llm_legislation": 0,
            "llm_heritage": 0,
            "llm_software": 1,
            "llm_startup": 0,
            "llm_patent": 1,
            "llm_manufacturing": 0,
            "llm_drug_trial": 0,
        },
        {
            "Unit of assessment number": 9,
            "llm_charity": 0,
            "llm_nhs": 1,
            "llm_museum": 1,
            "llm_school": 0,
            "llm_legislation": 1,
            "llm_heritage": 0,
            "llm_software": 0,
            "llm_startup": 1,
            "llm_patent": 0,
            "llm_manufacturing": 1,
            "llm_drug_trial": 0,
        },
        {
            "Unit of assessment number": 8,
            "llm_charity": 1,
            "llm_nhs": 0,
            "llm_museum": 0,
            "llm_school": 0,
            "llm_legislation": 1,
            "llm_heritage": 1,
            "llm_software": 0,
            "llm_startup": 0,
            "llm_patent": 1,
            "llm_manufacturing": 1,
            "llm_drug_trial": 1,
        },
        {
            "Unit of assessment number": 8,
            "llm_charity": 0,
            "llm_nhs": 0,
            "llm_museum": 1,
            "llm_school": 0,
            "llm_legislation": 0,
            "llm_heritage": 0,
            "llm_software": 1,
            "llm_startup": 1,
            "llm_patent": 1,
            "llm_manufacturing": 0,
            "llm_drug_trial": 1,
        },
    ]
    df = pd.DataFrame(rows)
    for col in TOPIC_COLS:
        df[col] = df[col].astype(int)
    return df


class PhysicsVsChemistryTableTest(unittest.TestCase):
    def test_generates_tex_with_glm_column_and_removes_legacy_csvs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "tables"
            out_dir.mkdir(parents=True, exist_ok=True)

            model_path = root / "regression_results.pkl"
            with model_path.open("wb") as f:
                pickle.dump({"coef_df": _make_coef_df()}, f)

            legacy_csv = out_dir / "physics_vs_chemistry.csv"
            legacy_detail_csv = out_dir / "physics_vs_chemistry_detail.csv"
            legacy_csv.write_text("legacy", encoding="utf-8")
            legacy_detail_csv.write_text("legacy", encoding="utf-8")

            table = _physics_vs_chemistry_table(_make_ics_df(), model_path=model_path, out_dir=out_dir)

            self.assertIn("Delta Women (pp, OLS)", table.columns)
            self.assertIn("Delta Women (log-odds, GLM)", table.columns)
            self.assertEqual(table.iloc[0]["Impact domain"], "Charity")
            self.assertFalse(legacy_csv.exists())
            self.assertFalse(legacy_detail_csv.exists())

            tex = (out_dir / "physics_vs_chemistry.tex").read_text(encoding="utf-8")
            self.assertIn("Delta Women (log-odds, GLM)", tex)
            self.assertIn("Domain predicted $\\Delta$ Women (OLS)", tex)
            self.assertIn("Domain predicted $\\Delta$ log-odds (GLM)", tex)
            self.assertIn("+0.40", tex)


class ScaleMetricAggregationTest(unittest.TestCase):
    def test_panel_and_uoa_tables_count_institution_uoa_scale_metrics_once(self):
        df_ics = pd.DataFrame(
            [
                {
                    "inst_id": "1001",
                    "Institution name": "Example University",
                    "Unit of assessment number": 8,
                    "Unit of assessment name": "Chemistry",
                    "Panel": "B",
                    "fte": 10.0,
                    "num_doc_degrees_total": 100.0,
                    "tot_income": 1_000.0,
                    "tot_inc_kind": 100.0,
                    "number_female": 1,
                    "number_male": 1,
                },
                {
                    "inst_id": "1001",
                    "Institution name": "Example University",
                    "Unit of assessment number": 8,
                    "Unit of assessment name": "Chemistry",
                    "Panel": "B",
                    "fte": 10.0,
                    "num_doc_degrees_total": 100.0,
                    "tot_income": 1_000.0,
                    "tot_inc_kind": 100.0,
                    "number_female": 2,
                    "number_male": 0,
                },
                {
                    "inst_id": "1002",
                    "Institution name": "Other University",
                    "Unit of assessment number": 8,
                    "Unit of assessment name": "Chemistry",
                    "Panel": "B",
                    "fte": 5.0,
                    "num_doc_degrees_total": 50.0,
                    "tot_income": 500.0,
                    "tot_inc_kind": 50.0,
                    "number_female": 0,
                    "number_male": 1,
                },
            ]
        )
        df_output = pd.DataFrame(
            [
                {
                    "Unit of assessment number": 8,
                    "Panel": "B",
                    "number_female": 1,
                    "number_male": 1,
                }
            ]
        )

        panel_b = _panel_table(df_ics, df_output).set_index("Panel").loc["B"]
        self.assertEqual(panel_b["FTE"], 15.0)
        self.assertEqual(panel_b["PhDs"], 150.0)
        self.assertEqual(panel_b["Total inc"], 1_650.0)
        self.assertEqual(panel_b["Number of ICS"], 3)
        self.assertEqual(panel_b["% Female Authors (ICS)"], 60.0)

        uoa_8 = _uoa_table(df_ics, df_output)
        uoa_8 = uoa_8[uoa_8["Unit of Assessment"].str.startswith("8 - ")].iloc[0]
        self.assertEqual(uoa_8["FTE"], 15.0)
        self.assertEqual(uoa_8["PhDs"], 150.0)
        self.assertEqual(uoa_8["Total inc"], 1_650.0)
        self.assertEqual(uoa_8["Number of ICS"], 3)

    def test_explicit_scale_frame_can_include_submission_rows_without_case_rows(self):
        df_ics = pd.DataFrame(
            [
                {
                    "inst_id": "1001",
                    "Institution name": "Example University",
                    "Unit of assessment number": 8,
                    "Panel": "B",
                    "number_female": 1,
                    "number_male": 1,
                }
            ]
        )
        df_output = pd.DataFrame(
            [
                {
                    "Unit of assessment number": 8,
                    "Panel": "B",
                    "number_female": 1,
                    "number_male": 1,
                }
            ]
        )
        scale_df = pd.DataFrame(
            [
                {
                    "inst_id": "1001",
                    "Unit of assessment number": 8,
                    "Panel": "B",
                    "fte": 10.0,
                    "num_doc_degrees_total": 100.0,
                    "tot_income": 1_000.0,
                    "tot_inc_kind": 100.0,
                },
                {
                    "inst_id": "1002",
                    "Unit of assessment number": 8,
                    "Panel": "B",
                    "fte": 5.0,
                    "num_doc_degrees_total": 50.0,
                    "tot_income": 500.0,
                    "tot_inc_kind": 50.0,
                },
            ]
        )

        panel_b = _panel_table(df_ics, df_output, scale_df=scale_df).set_index("Panel").loc["B"]
        self.assertEqual(panel_b["FTE"], 15.0)
        self.assertEqual(panel_b["PhDs"], 150.0)
        self.assertEqual(panel_b["Total inc"], 1_650.0)
        self.assertEqual(panel_b["Number of ICS"], 1)


if __name__ == "__main__":
    unittest.main()
