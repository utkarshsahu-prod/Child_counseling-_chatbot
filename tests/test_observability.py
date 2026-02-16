import unittest
from pathlib import Path

from src.openly.engine import TurnInput
from src.openly.runtime import build_engine_from_excels


class TestObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.domain_tree = cls.repo_root / "Domain_tree_UPDATED.xlsx"
        cls.cross_domain = cls.repo_root / "Cross_Domain_Logic_UPDATED.xlsx"
        cls.engine, _assets = build_engine_from_excels(cls.domain_tree, cls.cross_domain)

    def test_run_scenario_emits_traversal_snapshot(self):
        state = self.engine.start_session("obs1", "speech delay")
        domain = self.engine.available_domains[0]
        run = self.engine.run_scenario(
            state,
            [
                TurnInput(new_tags={"speech_delay"}, discovered_domains=[domain]),
                TurnInput(new_tags=set(), discovered_domains=[]),
            ],
        )

        self.assertGreaterEqual(len(run.records), 1)
        self.assertGreaterEqual(run.average_latency_ms, 0)

        snapshot = run.traversal_snapshot()
        self.assertEqual(snapshot[0]["turn_index"], 1)
        self.assertIn("active_domain", snapshot[0])
        self.assertIn("should_stop", snapshot[0])


if __name__ == "__main__":
    unittest.main()
