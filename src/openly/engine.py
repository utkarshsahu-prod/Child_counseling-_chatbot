from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .intake import PrimaryConcernIntake
from .loaders import DomainQuestion
from .orchestrator import Orchestrator
from .state_schema import ConversationState
from .trace import build_turn_trace


@dataclass(slots=True)
class TurnOutput:
    should_escalate: bool
    should_stop: bool
    active_domain: str | None
    next_question: str | None
    trace: dict


@dataclass(slots=True, frozen=True)
class TurnInput:
    new_tags: set[str]
    discovered_domains: list[str]


@dataclass(slots=True)
class TurnRecord:
    turn_index: int
    input_tags: list[str]
    input_domains: list[str]
    active_domain: str | None
    next_question: str | None
    should_escalate: bool
    should_stop: bool
    latency_ms: float


@dataclass(slots=True)
class ConversationRun:
    session_id: str
    records: list[TurnRecord]

    @property
    def average_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def traversal_snapshot(self) -> list[dict]:
        return [
            {
                "turn_index": r.turn_index,
                "active_domain": r.active_domain,
                "next_question": r.next_question,
                "should_escalate": r.should_escalate,
                "should_stop": r.should_stop,
            }
            for r in self.records
        ]


class ConversationEngine:
    """Minimal graph-level runtime to drive question progression over active branches."""

    def __init__(self, orchestrator: Orchestrator, domain_questions: list[DomainQuestion]):
        self.orchestrator = orchestrator
        self.domain_questions = domain_questions
        self._questions_by_domain: dict[str, list[DomainQuestion]] = {}
        for record in domain_questions:
            self._questions_by_domain.setdefault(record.domain_id, []).append(record)

    @property
    def available_domains(self) -> list[str]:
        return sorted(self._questions_by_domain.keys())

    def start_session(self, session_id: str, primary_concern: str) -> ConversationState:
        state = ConversationState(session_id=session_id, parent_primary_concern=primary_concern)
        return state

    def update_primary_concern_intake(self, state: ConversationState, payload: dict[str, str]) -> dict:
        intake = PrimaryConcernIntake(fields=dict(state.primary_intake))
        before = intake.as_dict()
        intake.update(payload)
        after = intake.as_dict()

        if before != after:
            state.log_update(
                "primary_intake",
                before,
                after,
                "primary_concern_intake_update",
            )

        state.primary_intake = after
        return {
            "is_complete": intake.is_complete,
            "missing_fields": intake.missing_fields(),
            "captured_fields": sorted(after.keys()),
        }

    def _next_question_for_domain(self, state: ConversationState, domain_id: str | None) -> str | None:
        if domain_id is None:
            return None
        questions = self._questions_by_domain.get(domain_id, [])
        if not questions:
            return None

        idx = state.question_progress.get(domain_id, 0)
        if idx >= len(questions):
            return None

        q = questions[idx].question
        state.question_progress[domain_id] = idx + 1
        state.log_update(
            "question_progress",
            f"{domain_id}:{idx}",
            f"{domain_id}:{idx + 1}",
            "next_question_selected",
        )
        return q

    def process_turn(
        self,
        state: ConversationState,
        *,
        new_tags: set[str],
        discovered_domains: list[str],
    ) -> TurnOutput:
        route = self.orchestrator.process_turn(
            state,
            new_tags=new_tags,
            discovered_domains=discovered_domains,
        )

        if route["should_escalate"]:
            trace = build_turn_trace(
                state,
                routing_decision=route["reason"],
                next_question_reason="escalation_short_circuit",
            )
            return TurnOutput(
                should_escalate=True,
                should_stop=True,
                active_domain=None,
                next_question=None,
                trace=trace,
            )

        active_domain = route["active_branch"]
        next_question = self._next_question_for_domain(state, active_domain)

        if next_question is None and active_domain is not None:
            state.complete_active_branch()

        trace = build_turn_trace(
            state,
            routing_decision=route["reason"],
            next_question_reason="domain_progression" if next_question else "branch_complete_or_stop",
        )
        should_stop = route["should_stop"] or (active_domain is None and next_question is None)
        return TurnOutput(
            should_escalate=False,
            should_stop=should_stop,
            active_domain=active_domain,
            next_question=next_question,
            trace=trace,
        )

    def run_scenario(self, state: ConversationState, turns: list[TurnInput]) -> ConversationRun:
        records: list[TurnRecord] = []
        for idx, turn in enumerate(turns, start=1):
            start = perf_counter()
            out = self.process_turn(
                state,
                new_tags=turn.new_tags,
                discovered_domains=turn.discovered_domains,
            )
            latency_ms = round((perf_counter() - start) * 1000, 3)
            records.append(
                TurnRecord(
                    turn_index=idx,
                    input_tags=sorted(turn.new_tags),
                    input_domains=turn.discovered_domains,
                    active_domain=out.active_domain,
                    next_question=out.next_question,
                    should_escalate=out.should_escalate,
                    should_stop=out.should_stop,
                    latency_ms=latency_ms,
                )
            )
            if out.should_stop:
                break

        return ConversationRun(session_id=state.session_id, records=records)
