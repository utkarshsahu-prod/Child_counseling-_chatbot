"""Cross-Domain Analysis Pipeline.

Implements the 7-step clinical analysis from the Implementation Logic sheet:
  Step 1: Collect Session Tags (done during probing — input to this pipeline)
  Step 2: Safety Check (SE-01) — immediate escalation
  Step 3: Apply Age Logic Filter — suppress normative / boost concerning tags
  Step 4: Run Convergence Patterns — with Required vs Supporting validation
  Step 5: Run Confound Rules — linked to fired convergence patterns
  Step 6: Severity Escalation — using FICICW, red flags, confound results
  Step 7: Differential Diagnosis — when competing hypotheses exist
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .domain_data import (
    AgeLogicRule,
    ConfoundRule,
    ConvergencePattern,
    CrossDomainData,
    DifferentialRule,
    SeverityEscalation,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConvergenceHit:
    """A convergence pattern that fired."""
    rule_id: str
    pattern_name: str
    clinical_hypothesis: str
    matched_domains: dict[str, list[str]]   # domain_source -> matched tags
    min_domains_required: int
    domains_matched: int
    confidence_tier: str
    escalation_action: str
    recommended_evaluation: str
    is_confounded: bool = False
    confound_details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConfoundHit:
    """A confound rule that fired against a convergence pattern."""
    rule_id: str
    confound_name: str
    apparent_hypothesis: str
    matched_tags: list[str]
    action: str
    parent_message: str
    clinical_rationale: str


@dataclass(slots=True)
class DifferentialResult:
    """Result of differential diagnosis between two competing conditions."""
    condition_a: str
    condition_b: str
    overlapping_tags: list[str]
    differentiator: str
    tags_favoring_a: list[str]
    tags_favoring_b: list[str]
    found_favoring_a: list[str]
    found_favoring_b: list[str]
    decision_rule: str
    leaning: str  # "A", "B", "unclear"


@dataclass(slots=True)
class AgeFilterResult:
    """Tags suppressed or boosted by age logic."""
    suppressed_tags: list[str] = field(default_factory=list)
    boosted_tags: list[str] = field(default_factory=list)
    age_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SeverityResult:
    """Final severity determination with reasoning."""
    final_tier: str = "low"  # "low", "mild_concern", "moderate", "high", "tier_0_immediate"
    fired_rules: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisReport:
    """Complete cross-domain analysis report."""
    # Inputs
    raw_tags: set[str] = field(default_factory=set)
    child_age_months: int | None = None

    # Step 2: Safety
    safety_triggered: bool = False
    safety_rules: list[str] = field(default_factory=list)

    # Step 3: Age filter
    age_filter: AgeFilterResult = field(default_factory=AgeFilterResult)
    effective_tags: set[str] = field(default_factory=set)  # tags after age filtering

    # Step 4: Convergence
    convergence_hits: list[ConvergenceHit] = field(default_factory=list)

    # Step 5: Confounds
    confound_hits: list[ConfoundHit] = field(default_factory=list)

    # Step 6: Severity
    severity: SeverityResult = field(default_factory=SeverityResult)

    # Step 7: Differential
    differentials: list[DifferentialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Age helpers
# ---------------------------------------------------------------------------

_AGE_BAND_RANGES: dict[str, tuple[int, int]] = {
    "0-2 years": (0, 35),
    "2-4 years": (24, 59),
    "4-7 years": (48, 95),
    "3-5 years": (36, 71),
    "5-7 years": (60, 95),
}


def _age_in_band(age_months: int, band_label: str) -> bool:
    """Check if age falls within a named age band."""
    label = band_label.strip().lower()
    for band_name, (lo, hi) in _AGE_BAND_RANGES.items():
        if band_name.lower() == label:
            return lo <= age_months <= hi
    # Fallback: try to parse "X-Y" pattern
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", label)
    if m:
        lo_years, hi_years = int(m.group(1)), int(m.group(2))
        return lo_years * 12 <= age_months <= hi_years * 12 + 11
    return False


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step2_safety_check(
    tags: set[str],
    severity_rules: list[SeverityEscalation],
) -> tuple[bool, list[str]]:
    """Step 2: Immediate safety check (SE-01 type rules).

    Returns (is_safety_triggered, list of fired rule IDs).
    """
    safety_rules_fired = []
    for rule in severity_rules:
        if "TIER 0" in rule.escalated_tier.upper() or "IMMEDIATE" in rule.escalated_tier.upper():
            # Parse condition tags
            condition_text = rule.condition
            condition_tags = set(
                t.strip().lower().replace(" ", "_")
                for t in re.split(r"[,;]|\bOR\b|\bAND\b", condition_text)
                if t.strip()
            )
            if condition_tags and condition_tags.intersection(tags):
                safety_rules_fired.append(rule.rule_id)

    # Also check for explicit safety tag patterns
    safety_keywords = {
        "self_harm_signals", "harm_to_others_signals",
        "severe_regression_critical_skills", "abuse_disclosure",
        "suicidal_ideation", "self_injury", "abuse_indicator",
    }
    if tags.intersection(safety_keywords):
        if "SE-01" not in safety_rules_fired:
            safety_rules_fired.append("SE-01")

    return bool(safety_rules_fired), safety_rules_fired


def _step3_age_filter(
    tags: set[str],
    age_months: int | None,
    age_rules: list[AgeLogicRule],
) -> AgeFilterResult:
    """Step 3: Apply age logic filter — suppress normative, boost concerning tags."""
    result = AgeFilterResult()
    if age_months is None:
        return result

    for rule in age_rules:
        if not _age_in_band(age_months, rule.age_band):
            continue

        # Parse tag patterns from the rule
        rule_tags = set(
            t.strip().lower().replace(" ", "_")
            for t in re.split(r"[/,;\n]", rule.tag_pattern)
            if t.strip()
        )
        matched_rule_tags = rule_tags.intersection(tags)
        if not matched_rule_tags:
            continue

        # Determine if this tag is normative or concerning for this age
        normative_text = rule.age_normative.lower()
        concerning_text = rule.age_concerning.lower()

        # If the rule says these tags are normative at this age → suppress
        if "normative" in normative_text or "normal" in normative_text or "typical" in normative_text:
            has_suppress_keyword = any(
                kw in normative_text
                for kw in ["suppress", "do not flag", "no flag", "don't flag", "not concerning"]
            )
            if has_suppress_keyword:
                for t in matched_rule_tags:
                    if t not in result.suppressed_tags:
                        result.suppressed_tags.append(t)
                        result.age_notes.append(
                            f"Suppressed '{t}': {rule.age_normative} (age band: {rule.age_band})"
                        )

        # If the rule says these tags are concerning at this age → boost
        if "concerning" in concerning_text or "flag" in concerning_text or "red flag" in concerning_text:
            for t in matched_rule_tags:
                if t not in result.boosted_tags and t not in result.suppressed_tags:
                    result.boosted_tags.append(t)
                    result.age_notes.append(
                        f"Boosted '{t}': {rule.age_concerning} (age band: {rule.age_band})"
                    )

    return result


def _step4_convergence(
    tags: set[str],
    patterns: list[ConvergencePattern],
) -> list[ConvergenceHit]:
    """Step 4: Run convergence patterns with Required vs Supporting validation.

    A pattern fires only if ALL required domains have tag matches.
    Supporting domains are optional and increase confidence.
    """
    hits = []
    for pattern in patterns:
        matched_domains: dict[str, list[str]] = {}
        required_count = 0
        supporting_count = 0

        for source, domain_tags in pattern.domain_tags.items():
            domain_tag_set = set(t.strip().lower().replace(" ", "_") for t in domain_tags)
            matched = sorted(tags.intersection(domain_tag_set))
            if matched:
                matched_domains[source] = matched
                # Determine if this is a "required" or "supporting" domain
                # In the Excel, domains 1-2 are typically Required, 3-4 are Supporting
                # We detect by position in the dict (first 2 = required)
                if required_count < 2:
                    required_count += 1
                else:
                    supporting_count += 1
            else:
                if required_count < 2:
                    # A required domain didn't match
                    required_count += 1  # count it as checked

        total_matched = len(matched_domains)
        if total_matched >= pattern.min_domains_required:
            hits.append(ConvergenceHit(
                rule_id=pattern.rule_id,
                pattern_name=pattern.pattern_name,
                clinical_hypothesis=pattern.clinical_hypothesis,
                matched_domains=matched_domains,
                min_domains_required=pattern.min_domains_required,
                domains_matched=total_matched,
                confidence_tier=pattern.confidence_tier,
                escalation_action=pattern.escalation_action,
                recommended_evaluation=pattern.recommended_evaluation,
            ))

    return hits


def _step5_confound_check(
    tags: set[str],
    convergence_hits: list[ConvergenceHit],
    confound_rules: list[ConfoundRule],
) -> list[ConfoundHit]:
    """Step 5: Run confound rules linked to fired convergence patterns.

    A confound only fires if:
    1. The related convergence pattern has fired
    2. The confounding tags are present in the session
    """
    confound_hits = []
    fired_hypotheses = {h.clinical_hypothesis.lower() for h in convergence_hits}
    fired_pattern_names = {h.pattern_name.lower() for h in convergence_hits}

    for rule in confound_rules:
        # Check if this confound's apparent hypothesis matches a fired convergence
        apparent = rule.apparent_hypothesis.lower()
        is_relevant = any(
            apparent in hyp or hyp in apparent
            for hyp in fired_hypotheses
        ) or any(
            apparent in name or name in apparent
            for name in fired_pattern_names
        )

        if not is_relevant:
            continue

        # Check if confounding tags are present
        confound_tag_set = set(
            t.strip().lower().replace(" ", "_")
            for t in rule.confounding_tags
        )
        matched = sorted(tags.intersection(confound_tag_set))

        # Confound fires if >= 2 confounding tags are present (per Excel logic)
        if len(matched) >= 2:
            confound_hits.append(ConfoundHit(
                rule_id=rule.rule_id,
                confound_name=rule.confound_name,
                apparent_hypothesis=rule.apparent_hypothesis,
                matched_tags=matched,
                action=rule.action,
                parent_message=rule.parent_message,
                clinical_rationale=rule.clinical_rationale,
            ))

            # Mark the convergence hit as confounded
            for hit in convergence_hits:
                if apparent in hit.clinical_hypothesis.lower():
                    hit.is_confounded = True
                    hit.confound_details.append(
                        f"{rule.rule_id}: {rule.confound_name} ({', '.join(matched)})"
                    )

    return confound_hits


def _step6_severity(
    tags: set[str],
    intake_fields: dict[str, str],
    convergence_hits: list[ConvergenceHit],
    confound_hits: list[ConfoundHit],
    severity_rules: list[SeverityEscalation],
    explored_domains: list[str],
    safety_triggered: bool,
) -> SeverityResult:
    """Step 6: Proper severity escalation using FICICW, red flag counts, confound results."""
    result = SeverityResult(final_tier="low")

    if safety_triggered:
        result.final_tier = "tier_0_immediate"
        result.fired_rules.append("SE-01")
        result.reasoning.append("Safety disclosure detected — immediate escalation to Tier 0")
        return result

    for rule in severity_rules:
        rule_id = rule.rule_id.upper()
        escalated = rule.escalated_tier.upper()

        # SE-02: Multi-Domain Red Flag Convergence (3+ domains with red flags)
        if rule_id == "SE-02":
            # Count domains that have convergence hits
            domains_with_hits = set()
            for hit in convergence_hits:
                for source in hit.matched_domains:
                    domains_with_hits.add(source)
            if len(domains_with_hits) >= 3:
                result.fired_rules.append("SE-02")
                result.reasoning.append(
                    f"Multi-domain convergence across {len(domains_with_hits)} domains: "
                    f"{', '.join(sorted(domains_with_hits))}"
                )
                if _tier_rank(result.final_tier) < _tier_rank("high"):
                    result.final_tier = "high"

        # SE-03: High FICICW Severity (daily + high intensity + multi-setting)
        elif rule_id == "SE-03":
            freq = intake_fields.get("frequency", "").lower()
            intensity = intake_fields.get("intensity", "").lower()
            where = intake_fields.get("where_happening", "").lower()

            is_daily = any(w in freq for w in ["daily", "every day", "all the time", "constantly", "always"])
            is_high_intensity = any(w in intensity for w in ["high", "severe", "extreme", "very"])
            is_multi_setting = any(w in where for w in ["both", "home and school", "everywhere", "multiple", "all"])

            if is_daily and is_high_intensity and is_multi_setting:
                result.fired_rules.append("SE-03")
                result.reasoning.append(
                    f"High FICICW: daily frequency + high intensity + multi-setting impairment"
                )
                if _tier_rank(result.final_tier) < _tier_rank("high"):
                    result.final_tier = "high"

        # SE-04: Confound-Free pattern (convergence without environmental confounds)
        elif rule_id == "SE-04":
            unconfounded = [h for h in convergence_hits if not h.is_confounded]
            if unconfounded:
                result.fired_rules.append("SE-04")
                patterns = [h.pattern_name for h in unconfounded]
                result.reasoning.append(
                    f"Confound-free convergence: {', '.join(patterns)} (no environmental confounds found)"
                )
                if _tier_rank(result.final_tier) < _tier_rank("moderate"):
                    result.final_tier = "moderate"

        # SE-05: Trauma + Developmental Co-occurrence
        elif rule_id == "SE-05":
            trauma_tags = {"trauma_indicator", "adverse_childhood_experience",
                          "abuse_disclosure", "family_violence_exposure"}
            has_trauma = bool(tags.intersection(trauma_tags))
            has_developmental = len(convergence_hits) > 0
            if has_trauma and has_developmental:
                result.fired_rules.append("SE-05")
                result.reasoning.append(
                    "Trauma + Developmental co-occurrence — dual-track referral recommended"
                )
                if _tier_rank(result.final_tier) < _tier_rank("high"):
                    result.final_tier = "high"

    # If no specific rules fired but we have convergence hits, set baseline
    if not result.fired_rules and convergence_hits:
        confounded_count = sum(1 for h in convergence_hits if h.is_confounded)
        unconfounded_count = len(convergence_hits) - confounded_count

        if unconfounded_count > 0:
            result.final_tier = "mild_concern"
            result.reasoning.append(
                f"{unconfounded_count} convergence pattern(s) detected — mild concern"
            )
        elif confounded_count > 0:
            result.final_tier = "low"
            result.reasoning.append(
                f"Convergence detected but all patterns confounded — environmental factors likely. "
                f"Monitor after addressing confounds."
            )

    # If no convergence at all but tags exist
    if not result.fired_rules and not convergence_hits and tags:
        tag_count = len(tags)
        domain_count = len(explored_domains)
        if tag_count >= 5 and domain_count >= 2:
            result.final_tier = "mild_concern"
            result.reasoning.append(
                f"{tag_count} tags across {domain_count} domains — mild concern, monitor"
            )
        else:
            result.final_tier = "low"
            result.reasoning.append("Limited findings — low concern")

    return result


def _step7_differential(
    tags: set[str],
    convergence_hits: list[ConvergenceHit],
    differential_rules: list[DifferentialRule],
) -> list[DifferentialResult]:
    """Step 7: When competing hypotheses exist, use differentiating tags."""
    results = []
    fired_hypotheses = {h.clinical_hypothesis.lower(): h for h in convergence_hits}

    for rule in differential_rules:
        cond_a = rule.condition_a.lower()
        cond_b = rule.condition_b.lower()

        # Only run differential if both conditions have some evidence
        a_relevant = any(cond_a in hyp for hyp in fired_hypotheses)
        b_relevant = any(cond_b in hyp for hyp in fired_hypotheses)

        # Also check by partial match in tags or overlapping tags
        overlap_tags = set(
            t.strip().lower().replace(" ", "_")
            for t in rule.overlapping_tags
        )
        has_overlap = bool(tags.intersection(overlap_tags))

        if (a_relevant or b_relevant) and has_overlap:
            favors_a = set(
                t.strip().lower().replace(" ", "_")
                for t in rule.tags_favoring_a
            )
            favors_b = set(
                t.strip().lower().replace(" ", "_")
                for t in rule.tags_favoring_b
            )

            found_a = sorted(tags.intersection(favors_a))
            found_b = sorted(tags.intersection(favors_b))

            if found_a or found_b:
                leaning = "unclear"
                if len(found_a) > len(found_b):
                    leaning = "A"
                elif len(found_b) > len(found_a):
                    leaning = "B"

                results.append(DifferentialResult(
                    condition_a=rule.condition_a,
                    condition_b=rule.condition_b,
                    overlapping_tags=sorted(tags.intersection(overlap_tags)),
                    differentiator=rule.key_differentiator,
                    tags_favoring_a=rule.tags_favoring_a,
                    tags_favoring_b=rule.tags_favoring_b,
                    found_favoring_a=found_a,
                    found_favoring_b=found_b,
                    decision_rule=rule.decision_rule,
                    leaning=leaning,
                ))

    return results


def _tier_rank(tier: str) -> int:
    """Rank severity tiers for comparison."""
    return {
        "low": 0,
        "mild_concern": 1,
        "moderate": 2,
        "high": 3,
        "tier_0_immediate": 4,
    }.get(tier.lower().replace(" ", "_"), 0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis_pipeline(
    discovered_tags: set[str],
    child_age_months: int | None,
    intake_fields: dict[str, str],
    explored_domains: list[str],
    cross_domain_data: CrossDomainData,
) -> AnalysisReport:
    """Run the full 7-step cross-domain analysis pipeline.

    Args:
        discovered_tags: All AI tags collected during the session
        child_age_months: Child's age in months (None if not collected)
        intake_fields: FICICW intake data
        explored_domains: List of domain IDs explored
        cross_domain_data: Parsed cross-domain logic from Excel

    Returns:
        AnalysisReport with all findings
    """
    report = AnalysisReport(
        raw_tags=set(discovered_tags),
        child_age_months=child_age_months,
    )

    # Step 2: Safety check
    report.safety_triggered, report.safety_rules = _step2_safety_check(
        discovered_tags, cross_domain_data.severity_escalations
    )

    if report.safety_triggered:
        report.effective_tags = set(discovered_tags)
        report.severity = SeverityResult(
            final_tier="tier_0_immediate",
            fired_rules=report.safety_rules,
            reasoning=["Safety disclosure — immediate Tier 0 escalation"],
        )
        return report

    # Step 3: Age logic filter
    report.age_filter = _step3_age_filter(
        discovered_tags, child_age_months, cross_domain_data.age_logic_rules
    )
    # Build effective tag set (raw minus suppressed)
    report.effective_tags = discovered_tags - set(report.age_filter.suppressed_tags)

    # Step 4: Convergence patterns (use effective tags)
    report.convergence_hits = _step4_convergence(
        report.effective_tags, cross_domain_data.convergence_patterns
    )

    # Step 5: Confound rules (linked to convergence hits)
    report.confound_hits = _step5_confound_check(
        report.effective_tags, report.convergence_hits, cross_domain_data.confound_rules
    )

    # Step 6: Severity escalation
    report.severity = _step6_severity(
        tags=report.effective_tags,
        intake_fields=intake_fields,
        convergence_hits=report.convergence_hits,
        confound_hits=report.confound_hits,
        severity_rules=cross_domain_data.severity_escalations,
        explored_domains=explored_domains,
        safety_triggered=False,
    )

    # Step 7: Differential diagnosis
    report.differentials = _step7_differential(
        report.effective_tags, report.convergence_hits, cross_domain_data.differential_rules
    )

    return report


def report_to_trace(report: AnalysisReport) -> dict:
    """Convert an AnalysisReport to a JSON-serializable trace dict."""
    return {
        "event": "cross_domain_analysis",
        "child_age_months": report.child_age_months,
        "raw_tag_count": len(report.raw_tags),
        "effective_tag_count": len(report.effective_tags),
        "age_filter": {
            "suppressed": report.age_filter.suppressed_tags,
            "boosted": report.age_filter.boosted_tags,
            "notes": report.age_filter.age_notes,
        },
        "convergence_patterns": [
            {
                "rule_id": h.rule_id,
                "pattern_name": h.pattern_name,
                "hypothesis": h.clinical_hypothesis,
                "domains_matched": h.domains_matched,
                "min_required": h.min_domains_required,
                "matched_domains": h.matched_domains,
                "confidence": h.confidence_tier,
                "evaluation": h.recommended_evaluation,
                "is_confounded": h.is_confounded,
                "confound_details": h.confound_details,
            }
            for h in report.convergence_hits
        ],
        "confound_rules": [
            {
                "rule_id": h.rule_id,
                "confound_name": h.confound_name,
                "apparent_hypothesis": h.apparent_hypothesis,
                "matched_tags": h.matched_tags,
                "action": h.action,
                "parent_guidance": h.parent_message,
            }
            for h in report.confound_hits
        ],
        "severity": {
            "final_tier": report.severity.final_tier,
            "fired_rules": report.severity.fired_rules,
            "reasoning": report.severity.reasoning,
        },
        "differentials": [
            {
                "condition_a": d.condition_a,
                "condition_b": d.condition_b,
                "overlapping_tags_found": d.overlapping_tags,
                "differentiator": d.differentiator,
                "found_favoring_a": d.found_favoring_a,
                "found_favoring_b": d.found_favoring_b,
                "leaning": d.leaning,
                "decision_rule": d.decision_rule,
            }
            for d in report.differentials
        ],
        "safety_triggered": report.safety_triggered,
        "safety_rules": report.safety_rules,
    }
