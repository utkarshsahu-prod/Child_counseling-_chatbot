import unittest

from src.openly.engine import ConversationEngine
from src.openly.intake import PrimaryConcernIntake, REQUIRED_PRIMARY_INTAKE_FIELDS
from src.openly.orchestrator import Orchestrator


class TestPrimaryIntakeModel(unittest.TestCase):
    def test_missing_fields_and_completion(self):
        intake = PrimaryConcernIntake()
        intake.update({"frequency": "daily", "intensity": "high"})
        self.assertFalse(intake.is_complete)
        self.assertIn("current_methods", intake.missing_fields())

        intake.update(
            {
                "current_methods": "timeouts",
                "where_happening": "home",
                "life_impact": "family stress",
            }
        )
        self.assertTrue(intake.is_complete)
        self.assertEqual(sorted(intake.as_dict().keys()), sorted(REQUIRED_PRIMARY_INTAKE_FIELDS))


class TestPrimaryIntakeEngineIntegration(unittest.TestCase):
    def test_engine_updates_state_and_logs(self):
        engine = ConversationEngine(orchestrator=Orchestrator(), domain_questions=[])
        state = engine.start_session("intake1", "tantrums")

        status = engine.update_primary_concern_intake(
            state,
            {
                "frequency": "daily",
                "intensity": "high",
                "current_methods": "timeouts",
            },
        )
        self.assertFalse(status["is_complete"])
        self.assertIn("where_happening", status["missing_fields"])
        self.assertIn("primary_intake", state.variables_used)

        status2 = engine.update_primary_concern_intake(
            state,
            {
                "where_happening": "home and school",
                "life_impact": "disrupting routines",
            },
        )
        self.assertTrue(status2["is_complete"])
        self.assertEqual(len(status2["missing_fields"]), 0)


if __name__ == "__main__":
    unittest.main()
