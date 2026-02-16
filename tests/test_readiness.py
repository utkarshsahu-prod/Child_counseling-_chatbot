import unittest
from pathlib import Path

from src.openly.readiness import run_readiness_review


class TestReadinessReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.domain_tree = cls.repo_root / "Domain_tree_UPDATED.xlsx"
        cls.cross_domain = cls.repo_root / "Cross_Domain_Logic_UPDATED.xlsx"

    def test_readiness_passes_when_gates_met(self):
        report = run_readiness_review(
            domain_tree_path=self.domain_tree,
            cross_domain_path=self.cross_domain,
            executed_test_count=27,
            minimum_test_count=25,
        )
        self.assertTrue(report.passed)
        self.assertTrue(all(g.passed for g in report.gates))

    def test_readiness_fails_when_test_threshold_not_met(self):
        report = run_readiness_review(
            domain_tree_path=self.domain_tree,
            cross_domain_path=self.cross_domain,
            executed_test_count=10,
            minimum_test_count=25,
        )
        self.assertFalse(report.passed)
        test_gate = next(g for g in report.gates if g.gate == "test_coverage_threshold")
        self.assertFalse(test_gate.passed)


if __name__ == "__main__":
    unittest.main()
