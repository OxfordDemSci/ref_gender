import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.step02_make_ref_staff import _get_pdf_bytes, isolate_staff_names_block_with_status
from src.step02_make_ref_staff import get_staff_rows


class Step02StaffBlockIsolationTest(unittest.TestCase):
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

            self.assertEqual(int(case_level["number_people"].sum()), 0)
            master = pd.read_csv(out_dir / "ref_text_and_staff_blocks.csv")
            self.assertEqual(master.loc[0, "extraction_status"], "flex")
            audit = pd.read_csv(out_dir / "ref_staff_extraction_audit.csv")
            self.assertEqual(audit.loc[0, "pdf_source"], "previous_text")


if __name__ == "__main__":
    unittest.main()
