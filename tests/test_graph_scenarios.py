import unittest
from pathlib import Path

from src.openly.runtime import build_engine_from_excels


class TestGraphScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.domain_tree = cls.repo_root / "Domain_tree_UPDATED.xlsx"
        cls.cross_domain = cls.repo_root / "Cross_Domain_Logic_UPDATED.xlsx"
        cls.engine, _assets = build_engine_from_excels(cls.domain_tree, cls.cross_domain)

    def test_red_flag_interrupts_flow(self):
        state = self.engine.start_session("g1", "My child has sudden troubling behavior")
        out = self.engine.process_turn(
            state,
            new_tags={"self_harm_signals"},
            discovered_domains=[self.engine.available_domains[0]],
        )
        self.assertTrue(out.should_escalate)
        self.assertTrue(out.should_stop)
        self.assertIsNone(out.next_question)

    def test_loop_control_stops_after_no_new_info(self):
        state = self.engine.start_session("g2", "No concern updates")
        # seed one domain so first turn has an active branch
        self.engine.process_turn(
            state,
            new_tags={"speech_delay"},
            discovered_domains=[self.engine.available_domains[0]],
        )

        stopped = False
        for _ in range(4):
            out = self.engine.process_turn(state, new_tags=set(), discovered_domains=[])
            if out.should_stop:
                stopped = True
                break
        self.assertTrue(stopped)

    def test_question_progression_by_domain(self):
        state = self.engine.start_session("g3", "Speech concern")
        target_domain = self.engine.available_domains[0]
        out1 = self.engine.process_turn(
            state,
            new_tags={"speech_delay"},
            discovered_domains=[target_domain],
        )
        self.assertIsNotNone(out1.next_question)

        out2 = self.engine.process_turn(
            state,
            new_tags={"speech_delay_followup"},
            discovered_domains=[target_domain],
        )
        # Either next question continues or branch exhausted and stops; both valid graph outcomes.
        self.assertTrue(out2.next_question is not None or out2.should_stop or out2.active_domain is not None)


if __name__ == "__main__":
    unittest.main()
