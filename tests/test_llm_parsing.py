"""Tests for LLM response parsing (no API calls needed)."""
import unittest

from src.openly.llm import OpenlyLLM, NLUResult


class TestNLUResponseParsing(unittest.TestCase):
    """Test the NLU response parser without making actual API calls."""

    def _parse(self, raw: str) -> NLUResult:
        """Helper to test _parse_nlu_response directly."""
        # Create instance without API key to test parsing only
        llm = object.__new__(OpenlyLLM)
        return llm._parse_nlu_response(raw, all_known_tags={"tag_a", "tag_b", "sensory_overstimulation"})

    def test_parses_valid_json(self):
        raw = '{"matched_tags": ["tag_a", "tag_b"], "discovered_domains": [], "intake_fields": {}, "safety_flags": []}'
        result = self._parse(raw)
        self.assertEqual(result.matched_tags, ["tag_a", "tag_b"])
        self.assertEqual(result.discovered_domains, [])

    def test_parses_json_with_markdown_fences(self):
        raw = '```json\n{"matched_tags": ["sensory_overstimulation"], "discovered_domains": ["motor_and_sensory_development"], "intake_fields": {"frequency": "daily"}, "safety_flags": []}\n```'
        result = self._parse(raw)
        self.assertEqual(result.matched_tags, ["sensory_overstimulation"])
        self.assertEqual(result.discovered_domains, ["motor_and_sensory_development"])
        self.assertEqual(result.intake_fields, {"frequency": "daily"})

    def test_handles_invalid_json_gracefully(self):
        raw = "This is not JSON at all"
        result = self._parse(raw)
        self.assertEqual(result.matched_tags, [])
        self.assertEqual(result.raw_llm_response, raw)

    def test_normalizes_tag_format(self):
        raw = '{"matched_tags": ["Tag-With-Dashes", "TAG WITH SPACES"], "discovered_domains": [], "intake_fields": {}, "safety_flags": []}'
        result = self._parse(raw)
        self.assertIn("tag_with_dashes", result.matched_tags)
        self.assertIn("tag_with_spaces", result.matched_tags)

    def test_filters_empty_values(self):
        raw = '{"matched_tags": ["tag_a", "", null], "discovered_domains": ["", null], "intake_fields": {"frequency": "", "intensity": "high"}, "safety_flags": []}'
        result = self._parse(raw)
        self.assertEqual(result.matched_tags, ["tag_a"])
        self.assertEqual(result.discovered_domains, [])
        self.assertEqual(result.intake_fields, {"intensity": "high"})

    def test_extracts_safety_flags(self):
        raw = '{"matched_tags": [], "discovered_domains": [], "intake_fields": {}, "safety_flags": ["self_harm_signals", "abuse_disclosure"]}'
        result = self._parse(raw)
        self.assertEqual(len(result.safety_flags), 2)
        self.assertIn("self_harm_signals", result.safety_flags)


if __name__ == "__main__":
    unittest.main()
