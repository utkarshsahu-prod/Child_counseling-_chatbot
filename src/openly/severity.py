from __future__ import annotations

from dataclasses import dataclass

from .state_schema import SeverityLevel


@dataclass(frozen=True, slots=True)
class SeverityRule:
    rule_id: str
    trigger_tags: frozenset[str]
    outcome: SeverityLevel
    priority: int


class SeverityEngine:
    """Deterministic severity resolver with explicit precedence/tie-breakers."""

    _order = {
        SeverityLevel.LOW: 0,
        SeverityLevel.MILD_CONCERN: 1,
        SeverityLevel.MODERATE: 2,
        SeverityLevel.HIGH: 3,
    }

    def __init__(self, rules: list[SeverityRule]):
        self.rules = rules

    def resolve(self, tag_ids: set[str], base: SeverityLevel = SeverityLevel.LOW) -> tuple[SeverityLevel, list[str]]:
        fired = [r for r in self.rules if r.trigger_tags.issubset(tag_ids)]
        if not fired:
            return base, []

        # deterministic tie-breaker: higher priority first, then stronger severity, then rule_id
        fired_sorted = sorted(
            fired,
            key=lambda r: (-r.priority, -self._order[r.outcome], r.rule_id),
        )
        winner = fired_sorted[0]
        final = winner.outcome if self._order[winner.outcome] > self._order[base] else base
        return final, [r.rule_id for r in fired_sorted]
