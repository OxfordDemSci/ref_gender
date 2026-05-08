import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pandas as pd

from src.step02_make_ref_staff import _get_pdf_bytes, isolate_staff_names_block_with_status
from src.step02_make_ref_staff import _run_staff_openai_batch
from src.step02_make_ref_staff import get_staff_rows, parse_staff_block_locally
from src.step02_make_ref_staff import parse_staff_with_llm, parse_staff_with_llm_batch


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise AssertionError("temperature must not be sent for GPT-5 staff extraction")
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(arguments='{"people": [{"name": "Jane Doe", "roles": []}]}')
                )
            ],
            content=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeBatchCompletions(_FakeCompletions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise AssertionError("temperature must not be sent for GPT-5 staff extraction")
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        arguments='{"cases": [{"case_id": "case-1", "people": [{"name": "Jane Doe", "roles": []}]}]}'
                    )
                )
            ],
            content=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _RetryThenSuccessCompletions(_FakeCompletions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("upstream connect error or disconnect/reset before headers")
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(arguments='{"people": [{"name": "Jane Doe", "roles": []}]}')
                )
            ],
            content=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _AlwaysFailCompletions(_FakeCompletions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("upstream connect error or disconnect/reset before headers")


class _FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class _FakeBatchContent:
    def __init__(self, text):
        self.text = text


class _FakeBatchFiles:
    def __init__(self):
        self.input_jsonl = ""

    def create(self, file, purpose):
        self.input_jsonl = file.read().decode("utf-8")
        return SimpleNamespace(id="file-input")

    def content(self, file_id):
        rows = []
        for line in self.input_jsonl.splitlines():
            req = __import__("json").loads(line)
            payload = __import__("json").loads(req["body"]["messages"][1]["content"])
            cases = []
            for case in payload["cases"]:
                cases.append(
                    {
                        "case_id": case["case_id"],
                        "people": [{"name": "Jane Doe", "roles": []}],
                    }
                )
            body = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": __import__("json").dumps({"cases": cases})
                                    }
                                }
                            ],
                            "content": None,
                        }
                    }
                ]
            }
            rows.append({"custom_id": req["custom_id"], "response": {"status_code": 200, "body": body}, "error": None})
        return _FakeBatchContent("\n".join(__import__("json").dumps(row) for row in rows))


class _FakeBatchClient:
    def __init__(self):
        self.files = _FakeBatchFiles()
        self.batches = self

    def create(self, **kwargs):
        return SimpleNamespace(
            id="batch-staff",
            status="completed",
            input_file_id=kwargs["input_file_id"],
            output_file_id="file-output",
            error_file_id=None,
            endpoint=kwargs["endpoint"],
        )

    def retrieve(self, batch_id):
        return self.create(input_file_id="file-input", endpoint="/v1/chat/completions")


class Step02StaffBlockIsolationTest(unittest.TestCase):
    def test_staff_llm_calls_do_not_send_temperature(self):
        completions = _FakeCompletions()
        client = _FakeClient(completions)

        people = parse_staff_with_llm(client, "Name(s): Jane Doe", model="gpt-5.5")

        self.assertEqual(people[0]["name"], "Jane Doe")
        self.assertNotIn("temperature", completions.calls[0])

    def test_staff_llm_retries_transient_connection_errors(self):
        completions = _RetryThenSuccessCompletions()
        client = _FakeClient(completions)

        people = parse_staff_with_llm(
            client,
            "Name(s): Jane Doe",
            model="gpt-5.5",
            max_retries=1,
            retry_base_sleep=0.001,
        )

        self.assertEqual(people[0]["name"], "Jane Doe")
        self.assertEqual(len(completions.calls), 2)

    def test_staff_batch_llm_calls_do_not_send_temperature(self):
        completions = _FakeBatchCompletions()
        client = _FakeClient(completions)

        people_by_case = parse_staff_with_llm_batch(client, [("case-1", "Name(s): Jane Doe")], model="gpt-5.5")

        self.assertEqual(people_by_case["case-1"][0]["name"], "Jane Doe")
        self.assertNotIn("temperature", completions.calls[0])

    def test_staff_openai_batch_collects_completed_job(self):
        with TemporaryDirectory() as tmp:
            people, errors = _run_staff_openai_batch(
                _FakeBatchClient(),
                llm_items=[("case-1", "Fallback text"), ("case-2", "Fallback text")],
                model_staff="gpt-5.5",
                llm_batch_size=2,
                batch_dir=Path(tmp),
                batch_wait=True,
                batch_poll_interval_seconds=5,
            )

        self.assertEqual(errors, {})
        self.assertEqual(people["case-1"][0]["name"], "Jane Doe")
        self.assertEqual(people["case-2"][0]["name"], "Jane Doe")

    def test_extracts_staff_block_with_field_prefixes(self):
        text = (
            "4BDetails of staff conducting the underpinning research from the submitting unit:\n"
            "5BName(s):\n"
            "Jane Doe\n"
            "6BRole(s) (e.g. job title):\n"
            "Professor\n"
            "7BPeriod(s) employed by submitting HEI:\n"
            "2010-present\n"
            "8BPeriod when the claimed impact occurred: 2018-2020\n"
        )
        block, status = isolate_staff_names_block_with_status(text, service_mode="flex")

        self.assertEqual(status, "flex")
        self.assertIsNotNone(block)
        self.assertIn("Name(s):", block or "")
        self.assertIn("Role(s):", block or "")

    def test_returns_none_when_no_staff_headers_exist(self):
        text = "Summary of impact only. No staff details section exists in this document."
        block, status = isolate_staff_names_block_with_status(text, service_mode="flex")
        self.assertEqual(status, "none")
        self.assertIsNone(block)

    def test_extracts_author_header_template(self):
        text = (
            "REF Impact Case Study Template\n"
            "Title of case study\n"
            "Author(s)\n"
            "Dr Kate Pike and Dr Emma Wadsworth\n"
            "1. Summary of the impact\n"
            "Impact text follows.\n"
        )
        block, status = isolate_staff_names_block_with_status(text, service_mode="flex")

        self.assertEqual(status, "flex")
        self.assertIn("Name(s):", block or "")
        self.assertIn("Kate Pike", block or "")
        local_people = parse_staff_block_locally(block or "")
        self.assertEqual([p["name"] for p in local_people], ["Kate Pike", "Emma Wadsworth"])

    def test_get_pdf_bytes_uses_cache_when_present(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            case_id = "abc-123"
            pdf_path = cache_dir / f"{case_id}.pdf"
            pdf_path.write_bytes(b"cached-pdf")

            class _Session:
                def get(self, *_args, **_kwargs):
                    raise AssertionError("session.get should not be called on cache hit")

            data, source = _get_pdf_bytes(
                case_id=case_id,
                target_url="https://example.invalid/file.pdf",
                session=_Session(),
                timeout_seconds=1,
                pdf_cache_dir=cache_dir,
            )
            self.assertEqual(source, "cache")
            self.assertEqual(data, b"cached-pdf")

    def test_get_pdf_bytes_downloads_and_writes_cache(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            case_id = "abc-123"

            class _Resp:
                content = b"downloaded-pdf"

                def raise_for_status(self):
                    return None

            class _Session:
                def get(self, *_args, **_kwargs):
                    return _Resp()

            data, source = _get_pdf_bytes(
                case_id=case_id,
                target_url="https://example.invalid/file.pdf",
                session=_Session(),
                timeout_seconds=1,
                pdf_cache_dir=cache_dir,
            )
            self.assertEqual(source, "download")
            self.assertEqual(data, b"downloaded-pdf")
            self.assertEqual((cache_dir / f"{case_id}.pdf").read_bytes(), b"downloaded-pdf")

    def test_get_staff_rows_uses_prior_text_when_download_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "ids.csv"
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            case_id = "case-1"
            pd.DataFrame({"REF impact case study identifier": [case_id]}).to_csv(input_csv, index=False)
            prior_text = (
                "4BDetails of staff conducting the underpinning research from the submitting unit:\n"
                "5BName(s):\nJane Doe\n"
                "6BRole(s):\nProfessor\n"
                "7BPeriod(s) employed by submitting HEI:\n2010-present\n"
                "8BPeriod when the claimed impact occurred: 2018-2020\n"
            )
            pd.DataFrame(
                {
                    "REF impact case study identifier": [case_id],
                    "Extracted Text": [prior_text],
                    "staff_block": [None],
                    "extraction_status": ["none"],
                }
            ).to_csv(out_dir / "ref_text_and_staff_blocks.csv", index=False)

            class _FailingSession:
                def get(self, *_args, **_kwargs):
                    raise RuntimeError("network failure")

            _rows, case_level = get_staff_rows(
                input_data_path=input_csv,
                out_dir=out_dir,
                session=_FailingSession(),
                llm_enabled=False,
                client=None,
                service_mode="flex",
                sleep_between_calls=0.0,
                timeout_seconds=1,
            )

            self.assertEqual(int(case_level["number_people"].sum()), 1)
            master = pd.read_csv(out_dir / "ref_text_and_staff_blocks.csv")
            self.assertEqual(master.loc[0, "extraction_status"], "flex")
            audit = pd.read_csv(out_dir / "ref_staff_extraction_audit.csv")
            self.assertEqual(audit.loc[0, "pdf_source"], "previous_text")

    def test_get_staff_rows_uses_local_staff_parser_before_llm(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "ids.csv"
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            case_id = "case-local"
            pd.DataFrame({"REF impact case study identifier": [case_id]}).to_csv(input_csv, index=False)
            prior_text = (
                "4BDetails of staff conducting the underpinning research from the submitting unit:\n"
                "5BName(s):\nJane Doe\n"
                "6BRole(s):\nProfessor\n"
                "7BPeriod(s) employed by submitting HEI:\n2010-present\n"
                "8BPeriod when the claimed impact occurred: 2018-2020\n"
            )
            pd.DataFrame(
                {
                    "REF impact case study identifier": [case_id],
                    "Extracted Text": [prior_text],
                    "staff_block": [None],
                    "extraction_status": ["none"],
                }
            ).to_csv(out_dir / "ref_text_and_staff_blocks.csv", index=False)

            class _FailingSession:
                def get(self, *_args, **_kwargs):
                    raise RuntimeError("network failure")

            completions = _FakeCompletions()
            _rows, case_level = get_staff_rows(
                input_data_path=input_csv,
                out_dir=out_dir,
                session=_FailingSession(),
                llm_enabled=True,
                client=_FakeClient(completions),
                service_mode="flex",
                sleep_between_calls=0.0,
                timeout_seconds=1,
                local_first=True,
            )

            self.assertEqual(len(completions.calls), 0)
            self.assertEqual(int(case_level.loc[0, "number_people"]), 1)
            self.assertEqual(case_level.loc[0, "staff_extraction_status"], "local_first")

    def test_get_staff_rows_uses_case_text_fallback_when_pdf_has_no_staff_block(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "ids.csv"
            out_dir = root / "out"
            case_id = "case-2"
            pd.DataFrame(
                {
                    "REF impact case study identifier": [case_id],
                    "Title": ["Removed PDF with workbook text"],
                    "1. Summary of the impact": ["Summary"],
                    "2. Underpinning research": ["The research was led by Dr Jane Example at the submitting institution."],
                    "3. References to the research": [""],
                    "4. Details of the impact": ["Impact"],
                    "5. Sources to corroborate the impact": [""],
                }
            ).to_csv(input_csv, index=False)

            class _NoStaffPdfSession:
                def get(self, *_args, **_kwargs):
                    class _Resp:
                        content = b"%PDF-not-really-used"

                        def raise_for_status(self):
                            return None

                    return _Resp()

            _rows, case_level = get_staff_rows(
                input_data_path=input_csv,
                out_dir=out_dir,
                session=_NoStaffPdfSession(),
                llm_enabled=False,
                client=None,
                service_mode="flex",
                sleep_between_calls=0.0,
                timeout_seconds=1,
            )

            self.assertTrue(pd.isna(case_level.loc[0, "number_people"]))
            self.assertEqual(case_level.loc[0, "staff_extraction_status"], "unresolved_no_llm")
            master = pd.read_csv(out_dir / "ref_text_and_staff_blocks.csv")
            self.assertEqual(master.loc[0, "extraction_status"], "case_text_fallback")

    def test_get_staff_rows_marks_nan_after_repeated_llm_failures(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "ids.csv"
            out_dir = root / "out"
            case_id = "case-3"
            pd.DataFrame(
                {
                    "REF impact case study identifier": [case_id],
                    "Title": ["Fallback text case"],
                    "1. Summary of the impact": ["Summary"],
                    "2. Underpinning research": ["The research was described in prose without an explicit staff table."],
                    "3. References to the research": [""],
                    "4. Details of the impact": ["Impact"],
                    "5. Sources to corroborate the impact": [""],
                }
            ).to_csv(input_csv, index=False)

            class _NoStaffPdfSession:
                def get(self, *_args, **_kwargs):
                    class _Resp:
                        content = b"%PDF-not-really-used"

                        def raise_for_status(self):
                            return None

                    return _Resp()

            completions = _AlwaysFailCompletions()
            _rows, case_level = get_staff_rows(
                input_data_path=input_csv,
                out_dir=out_dir,
                session=_NoStaffPdfSession(),
                llm_enabled=True,
                client=_FakeClient(completions),
                service_mode="flex",
                sleep_between_calls=0.0,
                timeout_seconds=1,
                llm_max_retries=1,
                llm_retry_base_sleep=0.001,
                require_people=False,
            )

            self.assertGreaterEqual(len(completions.calls), 2)
            self.assertTrue(pd.isna(case_level.loc[0, "number_people"]))
            self.assertEqual(case_level.loc[0, "staff_extraction_status"], "llm_failed")
            audit = pd.read_csv(out_dir / "ref_staff_extraction_audit.csv")
            self.assertTrue(bool(audit.loc[0, "is_unresolved"]))


if __name__ == "__main__":
    unittest.main()
