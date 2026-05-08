import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.step01_make_enhanced_data import (
    _parse_thematic_batch_flags,
    _response_output_text,
    get_ics_staff_rows,
    get_thematic_indicators,
)

THEMATIC_GROUPS = [
    "charity",
    "startup",
    "patent",
    "museum",
    "nhs",
    "drug_trial",
    "school",
    "legislation",
    "heritage",
    "manufacturing",
    "software",
]


class FakeResponse:
    def __init__(self, text):
        self.output_text = text
        self.status = "completed"
        self.incomplete_details = None
        self.error = None


class FakeResponsesClient:
    def __init__(self):
        self.batch_call_sizes = []

    def create(self, **kwargs):
        raw_items = kwargs["input"].split("ITEMS:\n", 1)[1]
        items = __import__("json").loads(raw_items)
        self.batch_call_sizes.append(len(items))
        if len(items) > 1:
            return FakeResponse("this is not json")
        item_id = items[0]["id"]
        result = {"id": item_id, **{g: False for g in THEMATIC_GROUPS}}
        result["software"] = True
        return FakeResponse(__import__("json").dumps({"results": [result]}))


class FakeOpenAIClient:
    last_responses_client = None

    def __init__(self, api_key):
        self.responses = FakeResponsesClient()
        FakeOpenAIClient.last_responses_client = self.responses


class FakeBatchContent:
    def __init__(self, text):
        self.text = text


class FakeBatchFiles:
    def __init__(self):
        self.input_jsonl = ""

    def create(self, file, purpose):
        self.input_jsonl = file.read().decode("utf-8")
        return SimpleNamespace(id="file-input")

    def content(self, file_id):
        rows = []
        for line in self.input_jsonl.splitlines():
            req = __import__("json").loads(line)
            prompt = req["body"]["input"]
            items = __import__("json").loads(prompt.split("ITEMS:\n", 1)[1])
            results = []
            for item in items:
                result = {"id": item["id"], **{g: False for g in THEMATIC_GROUPS}}
                result["software"] = True
                results.append(result)
            rows.append(
                {
                    "custom_id": req["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {"output_text": __import__("json").dumps({"results": results})},
                    },
                    "error": None,
                }
            )
        return FakeBatchContent("\n".join(__import__("json").dumps(row) for row in rows))


class FakeBatchOpenAIClient:
    def __init__(self, api_key):
        self.files = FakeBatchFiles()
        self.batches = self

    def create(self, **kwargs):
        return SimpleNamespace(
            id="batch-1",
            status="completed",
            input_file_id=kwargs["input_file_id"],
            output_file_id="file-output",
            error_file_id=None,
            endpoint=kwargs["endpoint"],
        )

    def retrieve(self, batch_id):
        return self.create(input_file_id="file-input", endpoint="/v1/responses")


class Step01ThematicTextFieldsTest(unittest.TestCase):
    def test_response_output_text_reads_nested_responses_content(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"results": [{"id": "abc", "charity": true}]}',
                        }
                    ],
                }
            ]
        }

        self.assertIn('"results"', _response_output_text(response))

    def test_batch_parser_accepts_fenced_json_and_single_flat_item(self):
        groups = ["charity", "startup"]
        raw = """```json
{"id": "case-1", "charity": true, "startup": false}
```"""

        out, status, error = _parse_thematic_batch_flags(raw, groups, expected_ids=["case-1"])

        self.assertEqual(status, "ok")
        self.assertEqual(error, "")
        self.assertEqual(out["case-1"], {"charity": 1, "startup": 0})

    def test_llm_batch_parse_failures_switch_to_single_item_mode(self):
        with TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "1. Summary of the impact": [f"Software impact {i}" for i in range(7)],
                    "2. Underpinning research": ["" for _ in range(7)],
                    "3. References to the research": ["" for _ in range(7)],
                    "4. Details of the impact": ["" for _ in range(7)],
                    "5. Sources to corroborate the impact": ["" for _ in range(7)],
                }
            )

            with (
                patch("src.step01_make_enhanced_data.read_secret", return_value="sk-test"),
                patch("src.step01_make_enhanced_data.OpenAI", FakeOpenAIClient),
            ):
                out = get_thematic_indicators(
                    df,
                    llm_enabled=True,
                    llm_batch_size=2,
                    cache_path=Path(tmp) / "categories.csv",
                )

            self.assertTrue(out["llm_status"].astype(str).str.startswith("ok").all())
            self.assertEqual(int(out["llm_software"].sum()), 7)
            self.assertEqual(FakeOpenAIClient.last_responses_client.batch_call_sizes.count(2), 2)

    def test_thematic_openai_batch_collects_completed_job(self):
        with TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "1. Summary of the impact": ["Software impact A", "Software impact B"],
                    "2. Underpinning research": ["", ""],
                    "3. References to the research": ["", ""],
                    "4. Details of the impact": ["", ""],
                    "5. Sources to corroborate the impact": ["", ""],
                }
            )

            with (
                patch("src.step01_make_enhanced_data.read_secret", return_value="sk-test"),
                patch("src.step01_make_enhanced_data.OpenAI", FakeBatchOpenAIClient),
            ):
                out = get_thematic_indicators(
                    df,
                    llm_enabled=True,
                    llm_batch_size=2,
                    cache_path=Path(tmp) / "categories.csv",
                    openai_processing_mode="batch",
                    batch_wait=True,
                    batch_dir=Path(tmp) / "batches",
                )

            self.assertEqual(set(out["llm_status"]), {"ok_batch"})
            self.assertEqual(int(out["llm_software"].sum()), 2)

    def test_regex_uses_all_five_case_study_text_fields(self):
        with TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "1. Summary of the impact": [""],
                    "2. Underpinning research": ["The underpinning research produced software."],
                    "3. References to the research": ["A patent is listed in the references."],
                    "4. Details of the impact": [""],
                    "5. Sources to corroborate the impact": ["A charity corroborated the impact."],
                }
            )

            out = get_thematic_indicators(
                df,
                llm_enabled=False,
                cache_path=Path(tmp) / "categories.csv",
            )

            self.assertEqual(int(out.loc[0, "regex_software"]), 1)
            self.assertEqual(int(out.loc[0, "regex_patent"]), 1)
            self.assertEqual(int(out.loc[0, "regex_charity"]), 1)

    def test_missing_staff_file_fails_canonical_build(self):
        with TemporaryDirectory() as tmp:
            df = pd.DataFrame({"REF impact case study identifier": ["case-1", "case-2"]})

            with self.assertRaises(FileNotFoundError):
                get_ics_staff_rows(df, Path(tmp) / "missing_staff_dir")

    def test_existing_staff_file_replaces_bootstrap_count_columns(self):
        with TemporaryDirectory() as tmp:
            staff_dir = Path(tmp) / "staff"
            staff_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "REF impact case study identifier": ["case-1"],
                    "staff_block": ["Name(s): Jane Doe"],
                    "extraction_status": ["flex"],
                    "names": ["['Jane Doe']"],
                    "given_names": ["['Jane']"],
                    "roles": ["[]"],
                    "genders": ["['female']"],
                    "number_people": [1],
                    "number_male": [0],
                    "number_female": [1],
                    "number_unknown": [0],
                }
            ).to_csv(staff_dir / "ref_case_level.csv", index=False)
            df = pd.DataFrame(
                {
                    "REF impact case study identifier": ["case-1"],
                    "number_people": [0],
                    "number_male": [0],
                    "number_female": [0],
                    "number_unknown": [0],
                }
            )

            out = get_ics_staff_rows(df, staff_dir)

            self.assertEqual(int(out.loc[0, "number_people"]), 1)
            self.assertEqual(int(out.loc[0, "number_female"]), 1)
            self.assertEqual(out.loc[0, "extraction_status"], "flex")

    def test_staff_file_can_carry_unresolved_nan_counts(self):
        with TemporaryDirectory() as tmp:
            staff_dir = Path(tmp) / "staff"
            staff_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "REF impact case study identifier": ["case-1"],
                    "staff_block": ["Fallback source text"],
                    "extraction_status": ["case_text_fallback"],
                    "staff_extraction_status": ["llm_failed"],
                    "staff_extraction_error": ["connection reset"],
                    "names": [pd.NA],
                    "given_names": [pd.NA],
                    "roles": [pd.NA],
                    "genders": [pd.NA],
                    "number_people": [pd.NA],
                    "number_male": [pd.NA],
                    "number_female": [pd.NA],
                    "number_unknown": [pd.NA],
                }
            ).to_csv(staff_dir / "ref_case_level.csv", index=False)
            df = pd.DataFrame({"REF impact case study identifier": ["case-1"]})

            out = get_ics_staff_rows(df, staff_dir)

            self.assertTrue(pd.isna(out.loc[0, "number_people"]))
            self.assertTrue(pd.isna(out.loc[0, "names"]))
            self.assertEqual(out.loc[0, "staff_extraction_status"], "llm_failed")


if __name__ == "__main__":
    unittest.main()
