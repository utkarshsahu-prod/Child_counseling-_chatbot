from __future__ import annotations

from dataclasses import dataclass

from .policy_tables import DSM_RED_FLAG_POLICY_TABLE
from .severity import SeverityRule


@dataclass(frozen=True, slots=True)
class TagNormalizationResult:
    canonical_tags: set[str]
    invalid_tags: set[str]


class TagOntology:
    """Canonical tag registry + alias resolver for deterministic routing inputs."""

    def __init__(self, allowed_tags: set[str], aliases: dict[str, str] | None = None):
        self.allowed_tags = {self._tokenize(t) for t in allowed_tags if t}
        self.aliases = {
            self._tokenize(alias): self._tokenize(canonical)
            for alias, canonical in (aliases or {}).items()
            if alias and canonical
        }

    @staticmethod
    def _tokenize(tag: str) -> str:
        return str(tag).strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def from_rules_and_policy(cls, rules: list[SeverityRule]) -> "TagOntology":
        rule_tags = {tag for rule in rules for tag in rule.trigger_tags}
        policy_tags = set(DSM_RED_FLAG_POLICY_TABLE.get("must_escalate_immediately", []))
        policy_tags.update(DSM_RED_FLAG_POLICY_TABLE.get("clinical_priority_tags", []))

        aliases = {
            "self harm signals": "self_harm_signals",
            "harm to others": "harm_to_others_signals",
            "speech delay": "speech_delay",
            "developmental regression": "developmental_regression",
        }

        return cls(allowed_tags=rule_tags.union(policy_tags).union(set(aliases.values())), aliases=aliases)

    def normalize(self, tags: set[str]) -> TagNormalizationResult:
        canonical: set[str] = set()
        invalid: set[str] = set()
        for tag in tags:
            token = self._tokenize(tag)
            mapped = self.aliases.get(token, token)
            if mapped in self.allowed_tags:
                canonical.add(mapped)
            else:
                invalid.add(tag)
        return TagNormalizationResult(canonical_tags=canonical, invalid_tags=invalid)
