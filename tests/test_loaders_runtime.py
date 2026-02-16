import unittest
from pathlib import Path

from src.openly.loaders import load_domain_questions, load_severity_rules
from src.openly.runtime import build_orchestrator_from_excels
from src.openly.state_schema import ConversationState


class TestExcelLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.domain_tree = cls.repo_root / "Domain_tree_UPDATED.xlsx"
        cls.cross_domain = cls.repo_root / "Cross_Domain_Logic_UPDATED.xlsx"

    def test_load_domain_questions_parses_non_empty_records(self):
        records = load_domain_questions(self.domain_tree)
        self.assertGreater(len(records), 100)
        first = records[0]
        self.assertTrue(first.domain_id)
        self.assertTrue(first.question)

    def test_load_severity_rules_parses_non_empty(self):
        rules = load_severity_rules(self.cross_domain)
        self.assertGreater(len(rules), 0)
        self.assertTrue(all(r.rule_id for r in rules))

    def test_runtime_builder_wires_orchestrator(self):
        orchestrator, assets = build_orchestrator_from_excels(self.domain_tree, self.cross_domain)
        self.assertGreater(assets.domain_question_count, 100)
        self.assertGreater(assets.severity_rule_count, 0)

        state = ConversationState(session_id="runtime_test")
        result = orchestrator.process_turn(
            state,
            new_tags={"speech_delay"},
            discovered_domains=["speech_language"],
        )
        self.assertFalse(result["should_escalate"])


if __name__ == "__main__":
    unittest.main()
