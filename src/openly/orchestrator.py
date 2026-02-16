from __future__ import annotations

from dataclasses import dataclass

from .cross_domain import CrossDomainEngine
from .nlu import TagOntology
from .policy_tables import DSM_RED_FLAG_POLICY_TABLE
from .severity import SeverityEngine
from .state_schema import BranchState, ConversationState, DomainBranch


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    max_revisits: int = 2
    max_no_new_info_turns: int = 3


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        severity_engine: SeverityEngine | None = None,
        tag_ontology: TagOntology | None = None,
        cross_domain_engine: CrossDomainEngine | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.severity_engine = severity_engine
        self.tag_ontology = tag_ontology
        self.cross_domain_engine = cross_domain_engine

    def safety_precheck(self, tag_ids: set[str]) -> tuple[bool, str | None]:
        must_escalate = set(DSM_RED_FLAG_POLICY_TABLE["must_escalate_immediately"])
        if must_escalate.intersection(tag_ids):
            return True, "immediate_red_flag_escalation"
        return False, None

    def process_turn(
        self,
        state: ConversationState,
        *,
        new_tags: set[str],
        discovered_domains: list[str],
    ) -> dict:
        raw_tags = set(new_tags)
        if self.tag_ontology is not None:
            normalized = self.tag_ontology.normalize(raw_tags)
            new_tags = normalized.canonical_tags
            state.log_update(
                "nlu_tag_normalization",
                sorted(raw_tags),
                {
                    "canonical_tags": sorted(new_tags),
                    "invalid_tags": sorted(normalized.invalid_tags),
                },
                "nlu_normalization",
            )

        old_tags = set(state.discovered_tag_ids)
        state.discovered_tag_ids.update(new_tags)
        state.log_update("discovered_tag_ids", sorted(old_tags), sorted(state.discovered_tag_ids), "nlp_tag_scan")

        safety_hit, safety_reason = self.safety_precheck(state.discovered_tag_ids)
        if safety_hit:
            return {
                "should_escalate": True,
                "reason": safety_reason,
                "active_branch": None,
            }

        routed_domains = list(discovered_domains)
        if self.cross_domain_engine is not None:
            cross_decision = self.cross_domain_engine.evaluate(
                state.discovered_tag_ids,
                base_severity=state.severity_level,
            )
            routed_domains = sorted(set(routed_domains).union(cross_decision.suggested_domains))
            state.log_update(
                "cross_domain_decision",
                {},
                {
                    "fired_rule_ids": cross_decision.fired_rule_ids,
                    "suggested_domains": sorted(cross_decision.suggested_domains),
                    "min_severity": cross_decision.min_severity.value if cross_decision.min_severity else None,
                },
                "cross_domain_eval",
            )
            if cross_decision.min_severity is not None:
                old_severity = state.severity_level
                state.severity_level = cross_decision.min_severity
                state.log_update(
                    "severity_level",
                    old_severity.value,
                    state.severity_level.value,
                    "cross_domain_min_severity",
                )

        added = 0
        for domain_id in routed_domains:
            branch = DomainBranch(domain_id=domain_id, source_tag_ids=sorted(new_tags), state=BranchState.QUEUED)
            if state.enqueue_branch(branch, max_revisits=self.config.max_revisits):
                added += 1

        state.no_new_info_turns = state.no_new_info_turns + 1 if added == 0 and not new_tags else 0

        if self.severity_engine is not None:
            old_severity = state.severity_level
            new_severity, fired_rules = self.severity_engine.resolve(state.discovered_tag_ids, base=old_severity)
            state.severity_level = new_severity
            state.log_update(
                "severity_level",
                old_severity.value,
                new_severity.value,
                "severity_rules:" + ",".join(fired_rules) if fired_rules else "severity_rules:none",
            )

        if state.active_branch_key is None:
            active = state.activate_next_branch()
        else:
            active = next((b for b in state.domain_queue if b.routing_key == state.active_branch_key), None)

        return {
            "should_escalate": False,
            "reason": "normal_routing",
            "active_branch": active.domain_id if active else None,
            "should_stop": state.should_stop(max_no_new_info_turns=self.config.max_no_new_info_turns),
        }
