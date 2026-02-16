from __future__ import annotations

from dataclasses import dataclass

from .state_schema import BranchState, ConversationState, SeverityLevel


CONTRACT_VERSIONS = {
    "state_schema": "v1",
    "trace_schema": "v1",
    "domain_tree_schema": "v1",
    "cross_domain_schema": "v1",
}


REQUIRED_STATE_FIELDS = {
    "session_id",
    "parent_primary_concern",
    "active_branch_key",
    "domain_queue",
    "processed_branch_keys",
    "discovered_tag_ids",
    "severity_level",
    "variables_used",
    "value_updates",
    "no_new_info_turns",
    "question_progress",
    "primary_intake",
}


EXPECTED_SEVERITY_LEVELS = {
    SeverityLevel.LOW.value,
    SeverityLevel.MILD_CONCERN.value,
    SeverityLevel.MODERATE.value,
    SeverityLevel.HIGH.value,
}

EXPECTED_BRANCH_STATES = {
    BranchState.QUEUED.value,
    BranchState.ACTIVE.value,
    BranchState.COMPLETED.value,
    BranchState.BLOCKED.value,
}


@dataclass(frozen=True, slots=True)
class ContractValidationResult:
    is_valid: bool
    errors: list[str]


def validate_contract_freeze() -> ContractValidationResult:
    errors: list[str] = []

    if set(CONTRACT_VERSIONS.keys()) != {
        "state_schema",
        "trace_schema",
        "domain_tree_schema",
        "cross_domain_schema",
    }:
        errors.append("contract_versions_missing_required_keys")

    current_state_fields = set(ConversationState.__dataclass_fields__.keys())
    missing_state = REQUIRED_STATE_FIELDS.difference(current_state_fields)
    if missing_state:
        errors.append(f"missing_state_fields:{sorted(missing_state)}")

    unexpected_state = current_state_fields.difference(REQUIRED_STATE_FIELDS)
    if unexpected_state:
        errors.append(f"unexpected_state_fields:{sorted(unexpected_state)}")

    if EXPECTED_SEVERITY_LEVELS != {lvl.value for lvl in SeverityLevel}:
        errors.append("severity_levels_changed")

    if EXPECTED_BRANCH_STATES != {b.value for b in BranchState}:
        errors.append("branch_states_changed")

    return ContractValidationResult(is_valid=not errors, errors=errors)
