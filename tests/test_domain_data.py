"""Tests for comprehensive domain tree and cross-domain data parsers."""
import unittest
from pathlib import Path

from src.openly.domain_data import load_all_domains, load_cross_domain_data


DATA_DIR = Path(__file__).resolve().parent.parent
DOMAIN_TREE = DATA_DIR / "Domain_tree_UPDATED.xlsx"
CROSS_DOMAIN = DATA_DIR / "Cross_Domain_Logic_UPDATED.xlsx"


class TestDomainTreeParsing(unittest.TestCase):
    """Validate full domain tree parsing from Excel."""

    @classmethod
    def setUpClass(cls):
        cls.domains = load_all_domains(DOMAIN_TREE)

    def test_parses_all_nine_domains(self):
        self.assertEqual(len(self.domains), 9)

    def test_each_domain_has_concerns(self):
        for d in self.domains:
            self.assertGreater(len(d.concerns), 0, f"{d.domain_id} has no concerns")

    def test_each_domain_has_questions(self):
        for d in self.domains:
            self.assertGreater(len(d.all_questions), 0, f"{d.domain_id} has no questions")

    def test_each_domain_has_tags(self):
        for d in self.domains:
            self.assertGreater(len(d.all_tags), 0, f"{d.domain_id} has no tags")

    def test_total_questions_reasonable(self):
        total = sum(len(d.all_questions) for d in self.domains)
        self.assertGreater(total, 200, "Expected at least 200 questions across all domains")

    def test_total_tags_reasonable(self):
        all_tags = set()
        for d in self.domains:
            all_tags.update(d.all_tags)
        self.assertGreater(len(all_tags), 100, "Expected at least 100 unique tags")

    def test_domain_ids_are_normalized(self):
        for d in self.domains:
            self.assertFalse(" " in d.domain_id, f"Domain ID has space: {d.domain_id}")
            self.assertEqual(d.domain_id, d.domain_id.lower(), f"Domain ID not lowercase: {d.domain_id}")

    def test_find_concern_by_tag(self):
        # behavioral_development should have low_frustration_tolerance
        behav = next((d for d in self.domains if "behavioral" in d.domain_id), None)
        self.assertIsNotNone(behav)
        concern = behav.find_concern_by_tag("low_frustration_tolerance")
        self.assertIsNotNone(concern, "Should find concern containing low_frustration_tolerance tag")


class TestCrossDomainParsing(unittest.TestCase):
    """Validate cross-domain logic parsing from Excel."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_cross_domain_data(CROSS_DOMAIN)

    def test_convergence_patterns_loaded(self):
        self.assertEqual(len(self.data.convergence_patterns), 30)

    def test_confound_rules_loaded(self):
        self.assertEqual(len(self.data.confound_rules), 14)

    def test_severity_escalations_loaded(self):
        self.assertEqual(len(self.data.severity_escalations), 11)

    def test_differential_rules_loaded(self):
        self.assertEqual(len(self.data.differential_rules), 15)

    def test_age_logic_rules_loaded(self):
        self.assertEqual(len(self.data.age_logic_rules), 25)

    def test_tag_reference_loaded(self):
        self.assertEqual(len(self.data.tag_reference), 177)

    def test_convergence_has_rule_ids(self):
        for p in self.data.convergence_patterns:
            self.assertTrue(p.rule_id.startswith("CP-"), f"Invalid rule ID: {p.rule_id}")

    def test_confound_has_rule_ids(self):
        for r in self.data.confound_rules:
            self.assertTrue(r.rule_id.startswith("CF-"), f"Invalid rule ID: {r.rule_id}")

    def test_severity_has_rule_ids(self):
        for s in self.data.severity_escalations:
            self.assertTrue(s.rule_id.startswith("SE-"), f"Invalid rule ID: {s.rule_id}")

    def test_convergence_has_domain_tags(self):
        for p in self.data.convergence_patterns:
            self.assertGreater(len(p.domain_tags), 0, f"{p.rule_id} has no domain tags")

    def test_age_logic_has_bands(self):
        bands = {r.age_band for r in self.data.age_logic_rules}
        self.assertGreater(len(bands), 1, "Expected multiple age bands")


if __name__ == "__main__":
    unittest.main()
