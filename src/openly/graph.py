"""LangGraph-based conversation flow for OPenly.

Implements the 8-phase conversation flow as a state graph:
  Phase 0: Session Init
  Phase 1: Initial Input & Multi-Intent Scan
  Phase 2: FICICW Natural Context Gathering (primary concern intake)
  Phase 3: Structured Probing Loop (domain-specific deep-dive)
  Phase 4: Dynamic Discovery & Re-routing
  Phase 5: Branch Completion & Queue Transition
  Phase 6: Cross-Domain Pattern Detection
  Phase 7: Session Summary & Next Steps
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langgraph.graph import END, StateGraph

from .cross_domain import CrossDomainEngine, CrossDomainRule, CrossRuleType
from .domain_data import (
    ClinicalDomain,
    CrossDomainData,
    PresentingConcern,
    load_all_domains,
    load_cross_domain_data,
)
from .intake import REQUIRED_PRIMARY_INTAKE_FIELDS, PrimaryConcernIntake
from .llm import NLUResult, OpenlyLLM
from .pii_guard import redact_payload
from .policy_tables import DSM_RED_FLAG_POLICY_TABLE
from .severity import SeverityEngine, SeverityRule
from .state_schema import BranchState, ConversationState, DomainBranch, SeverityLevel


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

@dataclass
class GraphState:
    """LangGraph state that wraps the conversation context."""
    # Core state
    session_id: str = ""
    phase: str = "init"  # init, intake, probing, transition, summary, ended
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Domain tracking
    active_domain_id: str | None = None
    active_concern_name: str | None = None
    domain_queue: list[str] = field(default_factory=list)
    explored_domains: list[str] = field(default_factory=list)
    processed_concerns: set[str] = field(default_factory=set)

    # Tag and severity tracking
    discovered_tags: set[str] = field(default_factory=set)
    severity_level: str = "low"
    safety_escalated: bool = False

    # Intake tracking
    intake_fields: dict[str, str] = field(default_factory=dict)
    intake_complete: bool = False

    # Question cursor per domain-concern
    question_cursors: dict[str, int] = field(default_factory=dict)

    # Output
    bot_message: str = ""
    should_end: bool = False

    # Trace / reasoning (JSON-friendly)
    trace_log: list[dict[str, Any]] = field(default_factory=list)

    # Parent's latest message
    parent_message: str = ""

    # Convergence patterns found
    convergence_hits: list[str] = field(default_factory=list)
    confound_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Openly Graph Engine
# ---------------------------------------------------------------------------

class OpenlyGraph:
    """Main orchestrator that builds and runs the LangGraph conversation."""

    def __init__(
        self,
        llm: OpenlyLLM,
        domains: list[ClinicalDomain],
        cross_domain_data: CrossDomainData,
    ):
        self.llm = llm
        self.domains = {d.domain_id: d for d in domains}
        self.cross_domain_data = cross_domain_data

        # Build tag reference for quick lookup
        self._all_known_tags: set[str] = set()
        for d in domains:
            self._all_known_tags.update(d.all_tags)
        for tr in cross_domain_data.tag_reference:
            self._all_known_tags.add(tr.ai_tag)

        # Build the LangGraph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph state machine."""
        builder = StateGraph(dict)

        # Add nodes
        builder.add_node("init_session", self._node_init_session)
        builder.add_node("classify_concern", self._node_classify_concern)
        builder.add_node("gather_intake", self._node_gather_intake)
        builder.add_node("probe_domain", self._node_probe_domain)
        builder.add_node("analyze_response", self._node_analyze_response)
        builder.add_node("check_transitions", self._node_check_transitions)
        builder.add_node("cross_domain_check", self._node_cross_domain_check)
        builder.add_node("generate_summary", self._node_generate_summary)

        # Set entry point
        builder.set_entry_point("init_session")

        # Edges
        builder.add_edge("init_session", "classify_concern")
        builder.add_conditional_edges(
            "classify_concern",
            self._route_after_classify,
            {"gather_intake": "gather_intake", "end": END},
        )
        builder.add_edge("gather_intake", "probe_domain")
        builder.add_conditional_edges(
            "probe_domain",
            self._route_after_probe,
            {"wait_for_input": END, "generate_summary": "generate_summary"},
        )
        builder.add_conditional_edges(
            "analyze_response",
            self._route_after_analysis,
            {
                "safety_escalation": "generate_summary",
                "continue_probing": "probe_domain",
                "check_transitions": "check_transitions",
            },
        )
        builder.add_conditional_edges(
            "check_transitions",
            self._route_after_transition,
            {
                "probe_domain": "probe_domain",
                "cross_domain_check": "cross_domain_check",
                "generate_summary": "generate_summary",
            },
        )
        builder.add_edge("cross_domain_check", "probe_domain")
        builder.add_edge("generate_summary", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Node: Session Init
    # ------------------------------------------------------------------

    def _node_init_session(self, state: dict) -> dict:
        opening = self.llm.generate_opening()
        state["bot_message"] = opening
        state["phase"] = "init"
        state["conversation_history"] = [{"role": "assistant", "content": opening}]
        state["trace_log"] = [{"event": "session_init", "phase": "init"}]
        return state

    # ------------------------------------------------------------------
    # Node: Classify Initial Concern (Phase 1)
    # ------------------------------------------------------------------

    def _node_classify_concern(self, state: dict) -> dict:
        parent_msg = state.get("parent_message", "")
        if not parent_msg:
            return state

        state["conversation_history"].append({"role": "user", "content": parent_msg})

        # Build domain reference for LLM
        available = []
        for did, d in self.domains.items():
            sample_concerns = ", ".join(c.name for c in d.concerns[:5])
            available.append({
                "domain_id": did,
                "display_name": d.display_name,
                "sample_concerns": sample_concerns,
            })

        classification = self.llm.classify_initial_concern(parent_msg, available)

        # Check safety flags first
        safety_flags = classification.get("safety_flags", [])
        if safety_flags:
            state["safety_escalated"] = True
            state["trace_log"].append({
                "event": "safety_flag_detected",
                "flags": safety_flags,
                "phase": "classify",
            })

        primary_domain = classification.get("primary_domain", "")
        primary_concern = classification.get("primary_concern", "")
        additional = classification.get("additional_domains", [])

        # Set active domain
        if primary_domain and primary_domain in self.domains:
            state["active_domain_id"] = primary_domain
            state["active_concern_name"] = primary_concern
            state["domain_queue"] = [d for d in additional if d != primary_domain and d in self.domains]
            if primary_domain not in state.get("explored_domains", []):
                state.setdefault("explored_domains", []).append(primary_domain)

        # Capture any intake fields from initial message
        intake_fields = classification.get("intake_fields", {})
        if intake_fields:
            state.setdefault("intake_fields", {}).update(intake_fields)

        state["phase"] = "classify"
        state["trace_log"].append({
            "event": "concern_classified",
            "primary_domain": primary_domain,
            "primary_concern": primary_concern,
            "additional_domains": additional,
            "intake_from_initial": list(intake_fields.keys()),
        })

        return state

    def _route_after_classify(self, state: dict) -> str:
        if state.get("safety_escalated"):
            return "end"
        if state.get("active_domain_id"):
            return "gather_intake"
        return "end"

    # ------------------------------------------------------------------
    # Node: Gather Intake (Phase 2 - FICICW)
    # ------------------------------------------------------------------

    def _node_gather_intake(self, state: dict) -> dict:
        intake = PrimaryConcernIntake(fields=dict(state.get("intake_fields", {})))
        missing = intake.missing_fields()

        state["intake_complete"] = intake.is_complete
        state["phase"] = "intake"
        state["trace_log"].append({
            "event": "intake_check",
            "complete": intake.is_complete,
            "missing": missing,
            "captured": sorted(intake.as_dict().keys()),
        })
        return state

    # ------------------------------------------------------------------
    # Node: Probe Domain (Phase 3)
    # ------------------------------------------------------------------

    def _node_probe_domain(self, state: dict) -> dict:
        domain_id = state.get("active_domain_id")
        concern_name = state.get("active_concern_name")

        if not domain_id or domain_id not in self.domains:
            state["should_end"] = True
            return state

        domain = self.domains[domain_id]

        # Find the active concern in the domain
        concern = None
        if concern_name:
            for c in domain.concerns:
                if c.name.lower() == concern_name.lower():
                    concern = c
                    break
            # Fuzzy match if exact didn't work
            if not concern:
                for c in domain.concerns:
                    if concern_name.lower() in c.name.lower() or c.name.lower() in concern_name.lower():
                        concern = c
                        break

        # If no concern matched, use first unprocessed concern
        if not concern:
            processed = state.get("processed_concerns", set())
            for c in domain.concerns:
                key = f"{domain_id}:{c.name}"
                if key not in processed:
                    concern = c
                    state["active_concern_name"] = c.name
                    break

        if not concern or not concern.questions:
            state["should_end"] = True
            return state

        # Get next question
        cursor_key = f"{domain_id}:{concern.name}"
        cursor = state.get("question_cursors", {}).get(cursor_key, 0)

        if cursor >= len(concern.questions):
            # This concern is exhausted
            state.setdefault("processed_concerns", set()).add(cursor_key)
            state["trace_log"].append({
                "event": "concern_exhausted",
                "domain": domain_id,
                "concern": concern.name,
            })
            # Check if there are more concerns in this domain
            next_concern = self._find_next_concern(state, domain)
            if next_concern:
                state["active_concern_name"] = next_concern.name
                return self._node_probe_domain(state)  # recurse with new concern
            state["should_end"] = True
            return state

        question_obj = concern.questions[cursor]
        state.setdefault("question_cursors", {})[cursor_key] = cursor + 1

        # Check intake status for enriching the question
        intake = PrimaryConcernIntake(fields=dict(state.get("intake_fields", {})))
        intake_status = {
            "is_complete": intake.is_complete,
            "missing_fields": intake.missing_fields(),
        } if not intake.is_complete else None

        # Generate natural question via NLG
        natural_question = self.llm.generate_question(
            base_question=question_obj.question_text,
            concern_name=concern.name,
            domain_name=domain.display_name,
            conversation_history=state.get("conversation_history", []),
            intake_status=intake_status,
        )

        state["bot_message"] = natural_question
        state["conversation_history"].append({"role": "assistant", "content": natural_question})
        state["phase"] = "probing"
        state["trace_log"].append({
            "event": "question_asked",
            "domain": domain_id,
            "concern": concern.name,
            "base_question": question_obj.question_text,
            "cursor": cursor,
            "total_questions": len(concern.questions),
        })

        return state

    def _find_next_concern(self, state: dict, domain: ClinicalDomain) -> PresentingConcern | None:
        processed = state.get("processed_concerns", set())
        for c in domain.concerns:
            key = f"{domain.domain_id}:{c.name}"
            if key not in processed:
                return c
        return None

    def _route_after_probe(self, state: dict) -> str:
        if state.get("should_end"):
            return "generate_summary"
        return "wait_for_input"

    # ------------------------------------------------------------------
    # Node: Analyze Response (Phase 4 - Discovery)
    # ------------------------------------------------------------------

    def _node_analyze_response(self, state: dict) -> dict:
        parent_msg = state.get("parent_message", "")
        if not parent_msg:
            return state

        state["conversation_history"].append({"role": "user", "content": parent_msg})

        domain_id = state.get("active_domain_id")
        concern_name = state.get("active_concern_name")

        active_domain = self.domains.get(domain_id) if domain_id else None
        active_concern = None
        if active_domain and concern_name:
            for c in active_domain.concerns:
                if c.name.lower() == concern_name.lower():
                    active_concern = c
                    break

        # NLU extraction
        nlu_result = self.llm.extract_tags(
            parent_message=parent_msg,
            active_domain=active_domain,
            active_concern=active_concern,
            all_known_tags=self._all_known_tags,
            conversation_history=state.get("conversation_history", []),
        )

        # Update state with NLU results
        old_tags = set(state.get("discovered_tags", set()))
        state.setdefault("discovered_tags", set()).update(nlu_result.matched_tags)

        # Update intake fields
        if nlu_result.intake_fields:
            state.setdefault("intake_fields", {}).update(nlu_result.intake_fields)

        # Check for safety flags
        safety_tags = set(DSM_RED_FLAG_POLICY_TABLE.get("must_escalate_immediately", []))
        if nlu_result.safety_flags or safety_tags.intersection(state.get("discovered_tags", set())):
            state["safety_escalated"] = True

        # Queue newly discovered domains
        for new_domain in nlu_result.discovered_domains:
            if new_domain in self.domains and new_domain not in state.get("explored_domains", []):
                state.setdefault("domain_queue", []).append(new_domain)

        new_tags = state.get("discovered_tags", set()) - old_tags

        state["phase"] = "analyze"
        state["trace_log"].append({
            "event": "response_analyzed",
            "matched_tags": nlu_result.matched_tags,
            "new_tags": sorted(new_tags),
            "discovered_domains": nlu_result.discovered_domains,
            "intake_updates": list(nlu_result.intake_fields.keys()),
            "safety_flags": nlu_result.safety_flags,
        })

        return state

    def _route_after_analysis(self, state: dict) -> str:
        if state.get("safety_escalated"):
            return "safety_escalation"

        domain_id = state.get("active_domain_id")
        concern_name = state.get("active_concern_name")

        if domain_id and domain_id in self.domains:
            domain = self.domains[domain_id]
            cursor_key = f"{domain_id}:{concern_name}" if concern_name else ""
            cursor = state.get("question_cursors", {}).get(cursor_key, 0)

            # Find current concern
            concern = None
            for c in domain.concerns:
                if concern_name and c.name.lower() == concern_name.lower():
                    concern = c
                    break

            if concern and cursor < len(concern.questions):
                return "continue_probing"

        return "check_transitions"

    # ------------------------------------------------------------------
    # Node: Check Transitions (Phase 5)
    # ------------------------------------------------------------------

    def _node_check_transitions(self, state: dict) -> dict:
        domain_id = state.get("active_domain_id")
        concern_name = state.get("active_concern_name")

        # Mark current concern as processed
        if domain_id and concern_name:
            state.setdefault("processed_concerns", set()).add(f"{domain_id}:{concern_name}")

        # Check for more concerns in current domain
        if domain_id and domain_id in self.domains:
            domain = self.domains[domain_id]
            next_concern = self._find_next_concern(state, domain)
            if next_concern:
                old_concern = concern_name or ""
                state["active_concern_name"] = next_concern.name
                state["trace_log"].append({
                    "event": "concern_transition",
                    "domain": domain_id,
                    "from_concern": old_concern,
                    "to_concern": next_concern.name,
                })
                return state

        # Current domain exhausted - check queue
        queue = state.get("domain_queue", [])
        if queue:
            next_domain_id = queue.pop(0)
            old_domain = domain_id or ""

            state["active_domain_id"] = next_domain_id
            state["domain_queue"] = queue

            if next_domain_id not in state.get("explored_domains", []):
                state.setdefault("explored_domains", []).append(next_domain_id)

            # Find first concern in new domain
            new_domain = self.domains.get(next_domain_id)
            if new_domain and new_domain.concerns:
                state["active_concern_name"] = new_domain.concerns[0].name

                # Generate transition message
                transition = self.llm.generate_transition(
                    from_domain=old_domain,
                    to_domain=new_domain.display_name,
                    to_concern=new_domain.concerns[0].name,
                    conversation_history=state.get("conversation_history", []),
                )
                # Prepend transition to next question
                state["_pending_transition"] = transition

            state["trace_log"].append({
                "event": "domain_transition",
                "from_domain": old_domain,
                "to_domain": next_domain_id,
                "remaining_queue": list(queue),
            })
            return state

        # Nothing left - time for cross-domain check
        state["trace_log"].append({"event": "all_domains_exhausted"})
        return state

    def _route_after_transition(self, state: dict) -> str:
        # If we have a new domain queued, continue probing
        if state.get("active_domain_id") and state.get("active_concern_name"):
            cursor_key = f"{state['active_domain_id']}:{state['active_concern_name']}"
            processed = state.get("processed_concerns", set())
            if cursor_key not in processed:
                return "probe_domain"

        # Check if there are still domains in queue
        if state.get("domain_queue"):
            return "probe_domain"

        # All exhausted - do cross-domain check
        if state.get("discovered_tags") and not state.get("_cross_domain_done"):
            return "cross_domain_check"

        return "generate_summary"

    # ------------------------------------------------------------------
    # Node: Cross-Domain Check (Phase 6)
    # ------------------------------------------------------------------

    def _node_cross_domain_check(self, state: dict) -> dict:
        tags = state.get("discovered_tags", set())
        state["_cross_domain_done"] = True

        # Check convergence patterns
        convergence_hits = []
        for pattern in self.cross_domain_data.convergence_patterns:
            all_pattern_tags = set()
            for tag_list in pattern.domain_tags.values():
                all_pattern_tags.update(tag_list)
            matched = tags.intersection(all_pattern_tags)
            domains_hit = 0
            for source, domain_tags in pattern.domain_tags.items():
                if tags.intersection(domain_tags):
                    domains_hit += 1
            if domains_hit >= pattern.min_domains_required:
                convergence_hits.append(f"{pattern.rule_id}: {pattern.pattern_name}")
                # Queue suggested evaluation domains
                for domain_id in self.domains:
                    if domain_id not in state.get("explored_domains", []):
                        state.setdefault("domain_queue", []).append(domain_id)

        state["convergence_hits"] = convergence_hits

        # Check confound rules
        confound_notes = []
        for rule in self.cross_domain_data.confound_rules:
            confound_tags = set(rule.confounding_tags)
            if confound_tags.intersection(tags):
                confound_notes.append(f"{rule.rule_id}: {rule.confound_name} - {rule.action}")

        state["confound_notes"] = confound_notes

        # Severity escalation check
        for esc in self.cross_domain_data.severity_escalations:
            if "TIER 0" in esc.escalated_tier.upper() or "IMMEDIATE" in esc.escalated_tier.upper():
                # Check if safety tags are present
                condition_tags = set(t.strip() for t in esc.condition.replace("OR", ",").replace("AND", ",").split(",") if t.strip())
                if tags.intersection(condition_tags):
                    state["severity_level"] = "high"
                    state["safety_escalated"] = True

        state["trace_log"].append({
            "event": "cross_domain_check",
            "convergence_hits": convergence_hits,
            "confound_notes": [n.split(":")[0] for n in confound_notes],
            "severity": state.get("severity_level", "low"),
        })

        return state

    # ------------------------------------------------------------------
    # Node: Generate Summary (Phase 7)
    # ------------------------------------------------------------------

    def _node_generate_summary(self, state: dict) -> dict:
        tags = state.get("discovered_tags", set())
        severity = state.get("severity_level", "low")
        explored = state.get("explored_domains", [])
        intake = state.get("intake_fields", {})
        convergence = state.get("convergence_hits", [])
        confounds = state.get("confound_notes", [])

        if state.get("safety_escalated"):
            summary = (
                "Thank you for sharing this with me. What you've described is very important, "
                "and I want to make sure your child gets the right support immediately. "
                "I strongly recommend reaching out to a child psychologist or your pediatrician "
                "as soon as possible. If there is any immediate safety concern, please contact "
                "your local emergency services or a crisis helpline. You're doing the right thing "
                "by speaking up about this."
            )
        else:
            summary = self.llm.generate_summary(
                discovered_tags=tags,
                severity_level=severity,
                domains_explored=explored,
                intake_data=redact_payload(intake),
                convergence_patterns=convergence,
                confound_notes=confounds,
            )

        state["bot_message"] = summary
        state["conversation_history"].append({"role": "assistant", "content": summary})
        state["should_end"] = True
        state["phase"] = "summary"

        # Build final trace
        state["trace_log"].append({
            "event": "session_summary",
            "severity": severity,
            "domains_explored": explored,
            "total_tags": len(tags),
            "convergence_patterns": convergence,
            "confound_considerations": [n.split(":")[0] for n in confounds],
            "intake_complete": state.get("intake_complete", False),
            "safety_escalated": state.get("safety_escalated", False),
        })

        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self, session_id: str = "session_001") -> dict:
        """Initialize a new conversation session. Returns initial state with opening message."""
        initial_state = {
            "session_id": session_id,
            "phase": "init",
            "conversation_history": [],
            "active_domain_id": None,
            "active_concern_name": None,
            "domain_queue": [],
            "explored_domains": [],
            "processed_concerns": set(),
            "discovered_tags": set(),
            "severity_level": "low",
            "safety_escalated": False,
            "intake_fields": {},
            "intake_complete": False,
            "question_cursors": {},
            "bot_message": "",
            "should_end": False,
            "trace_log": [],
            "parent_message": "",
            "convergence_hits": [],
            "confound_notes": [],
        }

        result = self.graph.invoke(initial_state)
        return result

    def process_message(self, state: dict, parent_message: str) -> dict:
        """Process a parent's message through the appropriate phase."""
        state["parent_message"] = parent_message

        if state.get("phase") == "init":
            # First message - classify concern
            result = self.graph.invoke(state)
            return result

        # For subsequent messages - run analysis then continue flow
        state = self._node_analyze_response(state)

        # Route based on analysis
        route = self._route_after_analysis(state)

        if route == "safety_escalation":
            state = self._node_generate_summary(state)
        elif route == "continue_probing":
            state = self._node_probe_domain(state)
        else:
            state = self._node_check_transitions(state)
            trans_route = self._route_after_transition(state)

            if trans_route == "probe_domain":
                # If there's a pending transition message, prepend it
                pending = state.pop("_pending_transition", None)
                state = self._node_probe_domain(state)
                if pending and state.get("bot_message"):
                    state["bot_message"] = pending + " " + state["bot_message"]
            elif trans_route == "cross_domain_check":
                state = self._node_cross_domain_check(state)
                # After cross-domain, check if there are new domains to probe
                if state.get("domain_queue"):
                    state = self._node_check_transitions(state)
                    state = self._node_probe_domain(state)
                else:
                    state = self._node_generate_summary(state)
            else:
                state = self._node_generate_summary(state)

        return state

    def get_trace_json(self, state: dict) -> str:
        """Get the reasoning trace as formatted JSON."""
        trace = {
            "session_id": state.get("session_id", ""),
            "phase": state.get("phase", ""),
            "severity_level": state.get("severity_level", ""),
            "active_domain": state.get("active_domain_id", ""),
            "active_concern": state.get("active_concern_name", ""),
            "discovered_tags": sorted(state.get("discovered_tags", set())),
            "explored_domains": state.get("explored_domains", []),
            "domain_queue": state.get("domain_queue", []),
            "intake_fields": redact_payload(state.get("intake_fields", {})),
            "intake_complete": state.get("intake_complete", False),
            "convergence_hits": state.get("convergence_hits", []),
            "confound_notes": state.get("confound_notes", []),
            "trace_events": state.get("trace_log", []),
        }
        return json.dumps(trace, indent=2, default=str)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_graph(
    domain_tree_path: str,
    cross_domain_path: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> OpenlyGraph:
    """Build the full OPenly conversation graph from Excel files."""
    llm = OpenlyLLM(api_key=api_key, model=model)
    domains = load_all_domains(domain_tree_path)
    cross_data = load_cross_domain_data(cross_domain_path)
    return OpenlyGraph(llm=llm, domains=domains, cross_domain_data=cross_data)
