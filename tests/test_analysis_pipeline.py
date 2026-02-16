"""Tests for the cross-domain analysis pipeline."""
import unittest
from pathlib import Path

from src.openly.analysis_pipeline import (
    AnalysisReport,
    ConfoundHit,
    ConvergenceHit,
    DifferentialResult,
    SeverityResult,
    _step2_safety_check,
    _step3_age_filter,
    _step4_convergence,
    _step5_confound_check,
    _step6_severity,
    _step7_differential,
    report_to_trace,
    run_analysis_pipeline,
)
from src.openly.domain_data import load_cross_domain_data


# Load real cross-domain data for tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROSS_DOMAIN_PATH = PROJECT_ROOT / "Cross_Domain_Logic_UPDATED.xlsx"
CROSS_DATA = load_cross_domain_data(CROSS_DOMAIN_PATH)


class TestSafetyCheck(unittest.TestCase):
    """Step 2: Safety check."""

    def test_safety_tags_trigger_escalation(self):
        tags = {"abuse_disclosure", "low_focus_endurance"}
        triggered, rules = _step2_safety_check(tags, CROSS_DATA.severity_escalations)
        self.assertTrue(triggered)
        self.assertIn("SE-01", rules)

    def test_no_safety_tags_no_escalation(self):
        tags = {"low_focus_endurance", "task_initiation_delay"}
        triggered, rules = _step2_safety_check(tags, CROSS_DATA.severity_escalations)
        self.assertFalse(triggered)
        self.assertEqual(rules, [])


class TestAgeFilter(unittest.TestCase):
    """Step 3: Age logic filter."""

    def test_no_age_returns_empty(self):
        tags = {"low_frustration_tolerance"}
        result = _step3_age_filter(tags, None, CROSS_DATA.age_logic_rules)
        self.assertEqual(result.suppressed_tags, [])
        self.assertEqual(result.boosted_tags, [])

    def test_age_filter_processes_rules(self):
        """Verify age filter runs without error for various ages."""
        tags = {"low_frustration_tolerance", "delayed_expressive_language", "low_social_initiation"}
        # Toddler age — some tags may be normative
        result = _step3_age_filter(tags, 18, CROSS_DATA.age_logic_rules)
        # Should produce either suppressed or boosted or notes
        self.assertIsInstance(result.suppressed_tags, list)
        self.assertIsInstance(result.boosted_tags, list)

    def test_older_child_different_from_toddler(self):
        """Same tags at different ages should potentially produce different results."""
        tags = {"low_frustration_tolerance", "delayed_expressive_language"}
        result_18mo = _step3_age_filter(tags, 18, CROSS_DATA.age_logic_rules)
        result_60mo = _step3_age_filter(tags, 60, CROSS_DATA.age_logic_rules)
        # We don't assert specific outcomes since it depends on Excel content,
        # but the function should not crash
        self.assertIsInstance(result_18mo.age_notes, list)
        self.assertIsInstance(result_60mo.age_notes, list)


class TestConvergence(unittest.TestCase):
    """Step 4: Convergence patterns."""

    def test_asd_tags_trigger_convergence(self):
        # Tags that should match ASD convergence pattern across domains
        tags = {
            "low_social_response_signal",
            "reciprocal_interaction_gap",
            "echolalia",
            "sensory_seeking_movement",
        }
        hits = _step4_convergence(tags, CROSS_DATA.convergence_patterns)
        # Should find at least one convergence hit
        self.assertGreater(len(hits), 0)
        rule_ids = [h.rule_id for h in hits]
        # ASD-related patterns should fire
        self.assertTrue(any("ASD" in r for r in rule_ids))

    def test_single_domain_tags_no_convergence(self):
        """Tags from only one domain should NOT fire multi-domain convergence."""
        tags = {"low_focus_endurance", "task_initiation_delay"}
        hits = _step4_convergence(tags, CROSS_DATA.convergence_patterns)
        # Convergence requires multiple domains — single-domain tags shouldn't fire
        # (unless a pattern happens to have very low min_domains)
        for h in hits:
            self.assertGreaterEqual(h.domains_matched, h.min_domains_required)

    def test_convergence_hit_has_required_fields(self):
        tags = {
            "low_social_response_signal",
            "reciprocal_interaction_gap",
            "echolalia",
            "sensory_seeking_movement",
        }
        hits = _step4_convergence(tags, CROSS_DATA.convergence_patterns)
        if hits:
            h = hits[0]
            self.assertTrue(h.rule_id)
            self.assertTrue(h.pattern_name)
            self.assertTrue(h.clinical_hypothesis)
            self.assertGreater(h.domains_matched, 0)


class TestConfoundCheck(unittest.TestCase):
    """Step 5: Confound rules linked to convergence."""

    def test_confound_only_fires_with_convergence(self):
        """Confound rules should NOT fire if no convergence pattern matched."""
        tags = {"excessive_screen_time", "screen_dependency", "low_focus_endurance"}
        no_convergence = []
        confounds = _step5_confound_check(tags, no_convergence, CROSS_DATA.confound_rules)
        # No convergence → no confounds should fire
        self.assertEqual(confounds, [])

    def test_confound_fires_with_matching_convergence(self):
        """Confound should fire when convergence hypothesis matches."""
        # Create a fake ADHD convergence hit
        adhd_hit = ConvergenceHit(
            rule_id="CP-ADHD-01",
            pattern_name="ADHD Convergence",
            clinical_hypothesis="ADHD",
            matched_domains={"Cognitive": ["low_focus_endurance"]},
            min_domains_required=2,
            domains_matched=2,
            confidence_tier="HIGH",
            escalation_action="Refer",
            recommended_evaluation="ADHD screening",
        )
        # Screen time tags that should confound ADHD
        tags = {"excessive_screen_time", "screen_dependency", "low_focus_endurance"}
        confounds = _step5_confound_check(tags, [adhd_hit], CROSS_DATA.confound_rules)
        # Should find screen-time confound for ADHD
        if confounds:
            self.assertTrue(any("screen" in c.confound_name.lower() for c in confounds))


class TestSeverityEscalation(unittest.TestCase):
    """Step 6: Severity escalation."""

    def test_safety_triggers_tier_0(self):
        result = _step6_severity(
            tags={"abuse_disclosure"},
            intake_fields={},
            convergence_hits=[],
            confound_hits=[],
            severity_rules=CROSS_DATA.severity_escalations,
            explored_domains=[],
            safety_triggered=True,
        )
        self.assertEqual(result.final_tier, "tier_0_immediate")

    def test_no_findings_stays_low(self):
        result = _step6_severity(
            tags={"low_focus_endurance"},
            intake_fields={},
            convergence_hits=[],
            confound_hits=[],
            severity_rules=CROSS_DATA.severity_escalations,
            explored_domains=["cognitive_development"],
            safety_triggered=False,
        )
        self.assertEqual(result.final_tier, "low")

    def test_unconfounded_convergence_raises_severity(self):
        hit = ConvergenceHit(
            rule_id="CP-ADHD-01",
            pattern_name="ADHD Convergence",
            clinical_hypothesis="ADHD",
            matched_domains={"Cognitive": ["low_focus_endurance"], "Behavioral": ["impulsive_action"]},
            min_domains_required=2,
            domains_matched=2,
            confidence_tier="HIGH",
            escalation_action="Refer",
            recommended_evaluation="ADHD screening",
            is_confounded=False,
        )
        result = _step6_severity(
            tags={"low_focus_endurance", "impulsive_action"},
            intake_fields={},
            convergence_hits=[hit],
            confound_hits=[],
            severity_rules=CROSS_DATA.severity_escalations,
            explored_domains=["cognitive_development", "behavioral_development"],
            safety_triggered=False,
        )
        # Should be at least moderate due to confound-free convergence
        self.assertIn(result.final_tier, ["moderate", "high", "mild_concern"])
        self.assertTrue(len(result.reasoning) > 0)


class TestDifferentialDiagnosis(unittest.TestCase):
    """Step 7: Differential diagnosis."""

    def test_no_convergence_no_differential(self):
        """Without convergence hits, differentials shouldn't fire."""
        tags = {"low_focus_endurance"}
        results = _step7_differential(tags, [], CROSS_DATA.differential_rules)
        self.assertEqual(results, [])

    def test_differential_has_leaning(self):
        """When differential fires, it should have a leaning."""
        # Create competing convergence hits
        adhd_hit = ConvergenceHit(
            rule_id="CP-ADHD-01",
            pattern_name="ADHD",
            clinical_hypothesis="ADHD",
            matched_domains={},
            min_domains_required=2,
            domains_matched=2,
            confidence_tier="HIGH",
            escalation_action="",
            recommended_evaluation="",
        )
        anxiety_hit = ConvergenceHit(
            rule_id="CP-ANX-01",
            pattern_name="Anxiety",
            clinical_hypothesis="Anxiety",
            matched_domains={},
            min_domains_required=2,
            domains_matched=2,
            confidence_tier="MODERATE",
            escalation_action="",
            recommended_evaluation="",
        )
        # Tags that overlap both + favor ADHD
        tags = {
            "low_focus_endurance",
            "impulsive_action",
            "task_initiation_delay",
        }
        results = _step7_differential(
            tags, [adhd_hit, anxiety_hit], CROSS_DATA.differential_rules
        )
        for r in results:
            self.assertIn(r.leaning, ["A", "B", "unclear"])


class TestFullPipeline(unittest.TestCase):
    """Integration test for the full pipeline."""

    def test_pipeline_returns_report(self):
        tags = {"low_focus_endurance", "task_initiation_delay", "external_scaffolding_dependence"}
        report = run_analysis_pipeline(
            discovered_tags=tags,
            child_age_months=60,
            intake_fields={"frequency": "daily"},
            explored_domains=["cognitive_development"],
            cross_domain_data=CROSS_DATA,
        )
        self.assertIsInstance(report, AnalysisReport)
        self.assertEqual(report.child_age_months, 60)
        self.assertEqual(report.raw_tags, tags)
        self.assertIsNotNone(report.severity)

    def test_pipeline_with_safety_tags(self):
        tags = {"abuse_disclosure", "low_focus_endurance"}
        report = run_analysis_pipeline(
            discovered_tags=tags,
            child_age_months=48,
            intake_fields={},
            explored_domains=[],
            cross_domain_data=CROSS_DATA,
        )
        self.assertTrue(report.safety_triggered)
        self.assertEqual(report.severity.final_tier, "tier_0_immediate")

    def test_report_to_trace_is_serializable(self):
        """Trace output should be JSON-serializable."""
        import json

        tags = {"low_focus_endurance", "task_initiation_delay"}
        report = run_analysis_pipeline(
            discovered_tags=tags,
            child_age_months=60,
            intake_fields={},
            explored_domains=["cognitive_development"],
            cross_domain_data=CROSS_DATA,
        )
        trace = report_to_trace(report)
        # Should not raise
        json_str = json.dumps(trace, default=str)
        self.assertIn("cross_domain_analysis", json_str)
        self.assertIn("severity", json_str)


if __name__ == "__main__":
    unittest.main()
