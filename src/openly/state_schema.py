from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SeverityLevel(str, Enum):
    LOW = "low"
    MILD_CONCERN = "mild_concern"
    MODERATE = "moderate"
    HIGH = "high"


class BranchState(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass(slots=True)
class DomainBranch:
    domain_id: str
    source_tag_ids: list[str] = field(default_factory=list)
    state: BranchState = BranchState.QUEUED
    revisit_count: int = 0

    @property
    def routing_key(self) -> str:
        return f"{self.domain_id}:{','.join(sorted(set(self.source_tag_ids)))}"


@dataclass(slots=True)
class ConversationState:
    session_id: str
    parent_primary_concern: str = ""
    active_branch_key: str | None = None
    domain_queue: list[DomainBranch] = field(default_factory=list)
    processed_branch_keys: set[str] = field(default_factory=set)
    discovered_tag_ids: set[str] = field(default_factory=set)
    severity_level: SeverityLevel = SeverityLevel.LOW
    variables_used: list[str] = field(default_factory=list)
    value_updates: list[dict[str, Any]] = field(default_factory=list)
    no_new_info_turns: int = 0
    question_progress: dict[str, int] = field(default_factory=dict)
    primary_intake: dict[str, str] = field(default_factory=dict)

    def enqueue_branch(self, branch: DomainBranch, max_revisits: int = 2) -> bool:
        """Adds a branch iff it is not completed and revisit limits are respected."""
        key = branch.routing_key
        if key in self.processed_branch_keys:
            return False

        for queued in self.domain_queue:
            if queued.routing_key == key:
                return False

        if branch.revisit_count > max_revisits:
            return False

        self.domain_queue.append(branch)
        return True

    def activate_next_branch(self) -> DomainBranch | None:
        for branch in self.domain_queue:
            if branch.state == BranchState.QUEUED:
                branch.state = BranchState.ACTIVE
                self.active_branch_key = branch.routing_key
                return branch
        self.active_branch_key = None
        return None

    def complete_active_branch(self) -> None:
        if not self.active_branch_key:
            return
        for branch in self.domain_queue:
            if branch.routing_key == self.active_branch_key:
                branch.state = BranchState.COMPLETED
                self.processed_branch_keys.add(branch.routing_key)
                break
        self.active_branch_key = None

    def should_stop(self, *, max_no_new_info_turns: int = 3, max_queue_size: int = 50) -> bool:
        """Loop-control guard while preserving domain-driven stopping."""
        if self.no_new_info_turns >= max_no_new_info_turns:
            return True
        queued_or_active = [
            b for b in self.domain_queue if b.state in {BranchState.QUEUED, BranchState.ACTIVE}
        ]
        if not queued_or_active:
            return True
        if len(self.domain_queue) > max_queue_size:
            return True
        return False

    def log_update(self, variable: str, old_value: Any, new_value: Any, reason: str) -> None:
        self.variables_used.append(variable)
        self.value_updates.append(
            {
                "variable": variable,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            }
        )
