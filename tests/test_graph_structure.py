"""Tests for the LangGraph conversation structure (without LLM calls)."""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.openly.domain_data import load_all_domains, load_cross_domain_data
from src.openly.graph import OpenlyGraph, GraphState


DATA_DIR = Path(__file__).resolve().parent.parent
DOMAIN_TREE = DATA_DIR / "Domain_tree_UPDATED.xlsx"
CROSS_DOMAIN = DATA_DIR / "Cross_Domain_Logic_UPDATED.xlsx"


class TestGraphConstruction(unittest.TestCase):
    """Test that the graph builds correctly without requiring API key."""

    @classmethod
    def setUpClass(cls):
        cls.domains = load_all_domains(DOMAIN_TREE)
        cls.cross_data = load_cross_domain_data(CROSS_DOMAIN)

    def test_graph_builds_with_mock_llm(self):
        mock_llm = MagicMock()
        graph = OpenlyGraph(
            llm=mock_llm,
            domains=self.domains,
            cross_domain_data=self.cross_data,
        )
        self.assertIsNotNone(graph.graph)

    def test_all_domains_indexed(self):
        mock_llm = MagicMock()
        graph = OpenlyGraph(
            llm=mock_llm,
            domains=self.domains,
            cross_domain_data=self.cross_data,
        )
        self.assertEqual(len(graph.domains), 9)

    def test_all_known_tags_populated(self):
        mock_llm = MagicMock()
        graph = OpenlyGraph(
            llm=mock_llm,
            domains=self.domains,
            cross_domain_data=self.cross_data,
        )
        self.assertGreater(len(graph._all_known_tags), 100)

    def test_trace_json_output(self):
        mock_llm = MagicMock()
        graph = OpenlyGraph(
            llm=mock_llm,
            domains=self.domains,
            cross_domain_data=self.cross_data,
        )
        state = {
            "session_id": "test",
            "phase": "init",
            "severity_level": "low",
            "active_domain_id": None,
            "active_concern_name": None,
            "discovered_tags": {"tag_a", "tag_b"},
            "explored_domains": ["behavioral_development"],
            "domain_queue": [],
            "intake_fields": {"frequency": "daily"},
            "intake_complete": False,
            "convergence_hits": [],
            "confound_notes": [],
            "trace_log": [{"event": "test"}],
        }
        trace_json = graph.get_trace_json(state)
        self.assertIn("session_id", trace_json)
        self.assertIn("tag_a", trace_json)
        self.assertIn("behavioral_development", trace_json)


class TestCrossDomainCheckNode(unittest.TestCase):
    """Test cross-domain check logic without LLM."""

    @classmethod
    def setUpClass(cls):
        cls.domains = load_all_domains(DOMAIN_TREE)
        cls.cross_data = load_cross_domain_data(CROSS_DOMAIN)

    def test_convergence_pattern_detection(self):
        mock_llm = MagicMock()
        graph = OpenlyGraph(
            llm=mock_llm,
            domains=self.domains,
            cross_domain_data=self.cross_data,
        )

        # Simulate state with tags that might trigger convergence
        state = {
            "discovered_tags": {
                "low_social_response_signal",
                "reciprocal_interaction_gap",
                "echolalia",
                "sensory_seeking_movement",
            },
            "explored_domains": ["social_development_and_attachme"],
            "domain_queue": [],
            "severity_level": "low",
            "safety_escalated": False,
            "trace_log": [],
            "convergence_hits": [],
            "confound_notes": [],
        }

        result = graph._node_cross_domain_check(state)
        # Should detect some convergence patterns with ASD-related tags
        self.assertIsInstance(result.get("convergence_hits"), list)
        self.assertTrue(result.get("_cross_domain_done"))


if __name__ == "__main__":
    unittest.main()
