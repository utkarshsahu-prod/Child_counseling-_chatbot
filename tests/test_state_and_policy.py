import unittest

from src.openly.policy_tables import ConcernClass, ConcernPriorityPolicy
from src.openly.state_schema import BranchState, ConversationState, DomainBranch


class TestPolicyPriority(unittest.TestCase):
    def test_policy_orders_by_class_then_turn(self):
        policy = ConcernPriorityPolicy.default()
        concerns = [
            {"id": "c3", "concern_class": ConcernClass.NON_CLINICAL.value, "discovered_at_turn": 2},
            {
                "id": "c2",
                "concern_class": ConcernClass.DSM_RED_FLAG_CLINICAL.value,
                "discovered_at_turn": 1,
            },
            {"id": "c1", "concern_class": ConcernClass.PRIMARY.value, "discovered_at_turn": 4},
        ]
        sorted_ids = [c["id"] for c in policy.sort_concerns(concerns)]
        self.assertEqual(sorted_ids, ["c1", "c2", "c3"])


class TestConversationState(unittest.TestCase):
    def test_enqueue_and_activate_branch(self):
        state = ConversationState(session_id="s1")
        b1 = DomainBranch(domain_id="speech", source_tag_ids=["tag_a"])
        b2 = DomainBranch(domain_id="speech", source_tag_ids=["tag_a"])

        self.assertTrue(state.enqueue_branch(b1))
        self.assertFalse(state.enqueue_branch(b2), "duplicate routing key should be rejected")

        active = state.activate_next_branch()
        self.assertIsNotNone(active)
        self.assertEqual(active.state, BranchState.ACTIVE)

        state.complete_active_branch()
        self.assertIn(b1.routing_key, state.processed_branch_keys)

    def test_stop_condition_when_no_new_info(self):
        state = ConversationState(session_id="s2", no_new_info_turns=3)
        self.assertTrue(state.should_stop(max_no_new_info_turns=3))


if __name__ == "__main__":
    unittest.main()
