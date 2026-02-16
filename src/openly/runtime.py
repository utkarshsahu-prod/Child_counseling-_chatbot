from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cross_domain import CrossDomainEngine, CrossDomainRule, CrossRuleType
from .engine import ConversationEngine
from .loaders import load_domain_questions, load_severity_rules
from .nlu import TagOntology
from .policy_tables import DSM_RED_FLAG_POLICY_TABLE
from .orchestrator import Orchestrator
from .severity import SeverityEngine
from .state_schema import SeverityLevel


@dataclass(frozen=True, slots=True)
class RuntimeAssets:
    domain_question_count: int
    severity_rule_count: int


def build_orchestrator_from_excels(
    domain_tree_path: Path | str,
    cross_domain_path: Path | str,
) -> tuple[Orchestrator, RuntimeAssets]:
    domain_questions = load_domain_questions(domain_tree_path)
    severity_rules = load_severity_rules(cross_domain_path)

    tag_ontology = TagOntology.from_rules_and_policy(severity_rules)
    cross_rules = [
        CrossDomainRule(
            rule_id=f"CD_ESC_{idx+1}",
            rule_type=CrossRuleType.ESCALATION,
            trigger_tags=frozenset({tag}),
            min_severity=SeverityLevel.HIGH,
            priority=10,
        )
        for idx, tag in enumerate(DSM_RED_FLAG_POLICY_TABLE.get("must_escalate_immediately", []))
    ]
    cross_rules.append(
        CrossDomainRule(
            rule_id="CD_DIFF_1",
            rule_type=CrossRuleType.DIFFERENTIAL,
            trigger_tags=frozenset({"speech_delay", "autism_red_flag"}),
            suggested_domains=("speech_language", "social_interaction"),
            priority=2,
        )
    )
    cross_engine = CrossDomainEngine(cross_rules)
    orchestrator = Orchestrator(
        severity_engine=SeverityEngine(severity_rules),
        tag_ontology=tag_ontology,
        cross_domain_engine=cross_engine,
    )
    assets = RuntimeAssets(
        domain_question_count=len(domain_questions),
        severity_rule_count=len(severity_rules),
    )
    return orchestrator, assets


def build_engine_from_excels(
    domain_tree_path: Path | str,
    cross_domain_path: Path | str,
) -> tuple[ConversationEngine, RuntimeAssets]:
    orchestrator, assets = build_orchestrator_from_excels(domain_tree_path, cross_domain_path)
    domain_questions = load_domain_questions(domain_tree_path)
    return ConversationEngine(orchestrator=orchestrator, domain_questions=domain_questions), assets
