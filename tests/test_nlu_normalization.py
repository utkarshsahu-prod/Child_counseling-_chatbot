import unittest

from src.openly.nlu import TagOntology
from src.openly.orchestrator import Orchestrator
from src.openly.severity import SeverityEngine, SeverityRule
from src.openly.state_schema import ConversationState, SeverityLevel


class TestTagOntology(unittest.TestCase):
    def test_normalizes_aliases_and_rejects_unknown(self):
        ontology = TagOntology(
            allowed_tags={"speech_delay", "self_harm_signals"},
            aliases={"speech delay": "speech_delay"},
        )

        result = ontology.normalize({"speech delay", "mystery-tag"})
        self.assertEqual(result.canonical_tags, {"speech_delay"})
        self.assertEqual(result.invalid_tags, {"mystery-tag"})


class TestOrchestratorNormalizationIntegration(unittest.TestCase):
    def test_invalid_tags_do_not_enter_state_or_trigger_rules(self):
        rules = [
            SeverityRule(
                rule_id="R1",
                trigger_tags=frozenset({"speech_delay"}),
                outcome=SeverityLevel.MILD_CONCERN,
                priority=1,
            )
        ]
        ontology = TagOntology(allowed_tags={"speech_delay"}, aliases={"speech delay": "speech_delay"})
        orchestrator = Orchestrator(severity_engine=SeverityEngine(rules), tag_ontology=ontology)

        state = ConversationState(session_id="nlu1")
        orchestrator.process_turn(
            state,
            new_tags={"speech delay", "unknown_issue"},
            discovered_domains=[],
        )

        self.assertIn("speech_delay", state.discovered_tag_ids)
        self.assertNotIn("unknown_issue", state.discovered_tag_ids)
        self.assertEqual(state.severity_level, SeverityLevel.MILD_CONCERN)

        normalization_events = [u for u in state.value_updates if u["variable"] == "nlu_tag_normalization"]
        self.assertEqual(len(normalization_events), 1)
        self.assertIn("unknown_issue", normalization_events[0]["new_value"]["invalid_tags"])


if __name__ == "__main__":
    unittest.main()
