from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state_schema import SeverityLevel


class CrossRuleType(str, Enum):
    CONVERGENCE = "convergence"
    CONFOUND = "confound"
    DIFFERENTIAL = "differential"
    ESCALATION = "escalation"


@dataclass(frozen=True, slots=True)
class CrossDomainRule:
    rule_id: str
    rule_type: CrossRuleType
    trigger_tags: frozenset[str]
    suggested_domains: tuple[str, ...] = ()
    min_severity: SeverityLevel | None = None
    priority: int = 1


@dataclass(frozen=True, slots=True)
class CrossDomainDecision:
    fired_rule_ids: list[str]
    suggested_domains: set[str]
    min_severity: SeverityLevel | None


class CrossDomainEngine:
    """Deterministic evaluator for cross-domain rule effects and escalation floors."""

    _severity_order = {
        SeverityLevel.LOW: 0,
        SeverityLevel.MILD_CONCERN: 1,
        SeverityLevel.MODERATE: 2,
        SeverityLevel.HIGH: 3,
    }
    _type_order = {
        CrossRuleType.ESCALATION: 0,
        CrossRuleType.CONVERGENCE: 1,
        CrossRuleType.DIFFERENTIAL: 2,
        CrossRuleType.CONFOUND: 3,
    }

    def __init__(self, rules: list[CrossDomainRule]):
        self.rules = rules

    def evaluate(self, tag_ids: set[str], *, base_severity: SeverityLevel) -> CrossDomainDecision:
        fired = [r for r in self.rules if r.trigger_tags and r.trigger_tags.issubset(tag_ids)]
        if not fired:
            return CrossDomainDecision(fired_rule_ids=[], suggested_domains=set(), min_severity=None)

        fired_sorted = sorted(
            fired,
            key=lambda r: (-r.priority, self._type_order[r.rule_type], r.rule_id),
        )

        domains: set[str] = set()
        min_severity = base_severity
        elevated = False
        for rule in fired_sorted:
            domains.update(rule.suggested_domains)
            if rule.min_severity and self._severity_order[rule.min_severity] > self._severity_order[min_severity]:
                min_severity = rule.min_severity
                elevated = True

        return CrossDomainDecision(
            fired_rule_ids=[r.rule_id for r in fired_sorted],
            suggested_domains=domains,
            min_severity=min_severity if elevated else None,
        )
