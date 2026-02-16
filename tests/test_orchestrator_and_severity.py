import unittest

from src.openly.orchestrator import Orchestrator
from src.openly.severity import SeverityEngine, SeverityRule
from src.openly.state_schema import ConversationState, SeverityLevel
from src.openly.trace import build_turn_trace


class TestOrchestrator(unittest.TestCase):
    def test_safety_precheck_short_circuits(self):
        orch = Orchestrator()
        state = ConversationState(session_id="s1")
        result = orch.process_turn(
            state,
            new_tags={"self_harm_signals"},
            discovered_domains=["behavioral_development"],
        )
        self.assertTrue(result["should_escalate"])
        self.assertEqual(result["reason"], "immediate_red_flag_escalation")

    def test_routing_enqueues_and_activates_branch(self):
        orch = Orchestrator()
        state = ConversationState(session_id="s2")
        result = orch.process_turn(
            state,
            new_tags={"speech_delay"},
            discovered_domains=["speech_language"],
        )
        self.assertFalse(result["should_escalate"])
        self.assertEqual(result["active_branch"], "speech_language")


class TestSeverityEngine(unittest.TestCase):
    def test_deterministic_priority_and_tie_breaking(self):
        engine = SeverityEngine(
            rules=[
                SeverityRule(
                    rule_id="r2",
                    trigger_tags=frozenset({"a", "b"}),
                    outcome=SeverityLevel.MODERATE,
                    priority=2,
                ),
                SeverityRule(
                    rule_id="r1",
                    trigger_tags=frozenset({"a", "b"}),
                    outcome=SeverityLevel.HIGH,
                    priority=2,
                ),
            ]
        )
        level, fired = engine.resolve({"a", "b"}, base=SeverityLevel.LOW)
        self.assertEqual(level, SeverityLevel.HIGH)
        self.assertEqual(fired, ["r1", "r2"])


class TestTrace(unittest.TestCase):
    def test_trace_from_state_artifacts(self):
        state = ConversationState(session_id="s3")
        state.log_update("severity_level", "low", "moderate", "cross_domain_escalation")
        trace = build_turn_trace(
            state,
            routing_decision="route_to_speech_language",
            next_question_reason="primary_concern_deep_dive",
        )
        self.assertIn("severity_level", trace["variables_used"])
        self.assertEqual(trace["routing_decision"], "route_to_speech_language")


if __name__ == "__main__":
    unittest.main()
