import unittest

from src.step03_get_dimensions_research_outputs import _infer_gender_list


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


if __name__ == "__main__":
    unittest.main()
