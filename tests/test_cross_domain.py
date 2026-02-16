import unittest

from src.openly.cross_domain import CrossDomainEngine, CrossDomainRule, CrossRuleType
from src.openly.orchestrator import Orchestrator
from src.openly.state_schema import ConversationState, SeverityLevel


class TestCrossDomainEngine(unittest.TestCase):
    def test_evaluates_domains_and_min_severity_deterministically(self):
        rules = [
            CrossDomainRule(
                rule_id="CD1",
                rule_type=CrossRuleType.CONVERGENCE,
                trigger_tags=frozenset({"speech_delay", "social_delay"}),
                suggested_domains=("speech_language",),
                priority=2,
            ),
            CrossDomainRule(
                rule_id="CD2",
                rule_type=CrossRuleType.ESCALATION,
                trigger_tags=frozenset({"social_delay", "regression"}),
                min_severity=SeverityLevel.HIGH,
                priority=3,
            ),
        ]
        engine = CrossDomainEngine(rules)

        decision = engine.evaluate(
            {"speech_delay", "social_delay", "regression"},
            base_severity=SeverityLevel.LOW,
        )
        self.assertEqual(decision.fired_rule_ids, ["CD2", "CD1"])
        self.assertEqual(decision.suggested_domains, {"speech_language"})
        self.assertEqual(decision.min_severity, SeverityLevel.HIGH)


class TestCrossDomainOrchestratorIntegration(unittest.TestCase):
    def test_cross_domain_suggested_domains_are_enqueued(self):
        cross_engine = CrossDomainEngine(
            [
                CrossDomainRule(
                    rule_id="CDX",
                    rule_type=CrossRuleType.DIFFERENTIAL,
                    trigger_tags=frozenset({"speech_delay", "autism_red_flag"}),
                    suggested_domains=("speech_language", "social_interaction"),
                    priority=1,
                )
            ]
        )
        orchestrator = Orchestrator(cross_domain_engine=cross_engine)
        state = ConversationState(session_id="cd-orch")

        orchestrator.process_turn(
            state,
            new_tags={"speech_delay", "autism_red_flag"},
            discovered_domains=[],
        )

        queued_domains = {b.domain_id for b in state.domain_queue}
        self.assertTrue({"speech_language", "social_interaction"}.issubset(queued_domains))

        events = [u for u in state.value_updates if u["variable"] == "cross_domain_decision"]
        self.assertEqual(len(events), 1)
        self.assertIn("CDX", events[0]["new_value"]["fired_rule_ids"])


if __name__ == "__main__":
    unittest.main()
