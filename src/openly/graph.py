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

from .analysis_pipeline import AnalysisReport, run_analysis_pipeline, report_to_trace
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

    # Concerns explicitly triggered by NLU discovery
    triggered_concerns: set[str] = field(default_factory=set)

    # Output
    bot_message: str = ""
    should_end: bool = False

    # Trace / reasoning (JSON-friendly)
    trace_log: list[dict[str, Any]] = field(default_factory=list)

    # Parent's latest message
    parent_message: str = ""

    # Child demographics
    child_age_months: int | None = None

    # Convergence patterns found
    convergence_hits: list[str] = field(default_factory=list)
    confound_notes: list[str] = field(default_factory=list)

    # Full analysis report (populated by cross-domain pipeline)
    analysis_report: AnalysisReport | None = None


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
        builder.add_conditional_edges(
            "gather_intake",
            self._route_after_intake,
            {"wait_for_input": END, "probe_domain": "probe_domain"},
        )
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

        # Build domain reference for LLM — include ALL concern names
        available = []
        for did, d in self.domains.items():
            all_concerns = ", ".join(f'"{c.name}"' for c in d.concerns)
            available.append({
                "domain_id": did,
                "display_name": d.display_name,
                "all_concerns": all_concerns,
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
            domain = self.domains[primary_domain]

            # Validate concern name against actual concerns in this domain
            validated_concern = self._validate_concern_name(primary_concern, domain)
            state["active_domain_id"] = primary_domain
            state["active_concern_name"] = validated_concern
            state["domain_queue"] = [d for d in additional if d != primary_domain and d in self.domains]
            if primary_domain not in state.get("explored_domains", []):
                state.setdefault("explored_domains", []).append(primary_domain)

        # Capture child age if mentioned
        child_age = classification.get("child_age_months")
        if child_age is not None:
            try:
                state["child_age_months"] = int(child_age)
            except (ValueError, TypeError):
                pass

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

    def _get_active_concern(self, state: dict) -> PresentingConcern | None:
        """Look up the active concern object from state."""
        domain_id = state.get("active_domain_id")
        concern_name = state.get("active_concern_name")
        domain = self.domains.get(domain_id) if domain_id else None
        if domain and concern_name:
            for c in domain.concerns:
                if c.name.lower() == concern_name.lower():
                    return c
        return None

    def _validate_concern_name(self, llm_concern: str, domain: ClinicalDomain) -> str:
        """Validate and resolve an LLM-returned concern name against actual domain concerns.

        Returns the exact concern name from the domain tree.
        Falls back to the first concern ONLY if nothing matches at all.
        """
        if not llm_concern:
            return domain.concerns[0].name if domain.concerns else ""

        # 1. Exact match (case-insensitive)
        for c in domain.concerns:
            if c.name.lower().strip() == llm_concern.lower().strip():
                return c.name

        # 2. Exact match ignoring quotes and special chars
        def _normalize(s: str) -> str:
            return s.lower().replace('"', '').replace("'", "").replace(""", "").replace(""", "").strip()

        for c in domain.concerns:
            if _normalize(c.name) == _normalize(llm_concern):
                return c.name

        # 3. Substring match — LLM name contains a concern name or vice versa
        llm_lower = llm_concern.lower().strip()
        best_match = None
        best_score = 0
        for c in domain.concerns:
            c_lower = c.name.lower().strip()
            # Check both directions
            if c_lower in llm_lower or llm_lower in c_lower:
                score = len(c_lower)  # longer match = better
                if score > best_score:
                    best_score = score
                    best_match = c

        if best_match:
            return best_match.name

        # 4. Word overlap scoring
        llm_words = set(llm_lower.replace("/", " ").replace("-", " ").split())
        for c in domain.concerns:
            c_words = set(c.name.lower().replace("/", " ").replace("-", " ").split())
            overlap = len(llm_words.intersection(c_words))
            if overlap > best_score:
                best_score = overlap
                best_match = c

        if best_match and best_score >= 2:
            return best_match.name

        # 5. Last resort — return LLM's name as-is (probe_domain will handle no-match)
        return llm_concern

    def _route_after_classify(self, state: dict) -> str:
        if state.get("safety_escalated"):
            return "end"
        if state.get("active_domain_id"):
            return "gather_intake"
        return "end"

    # ------------------------------------------------------------------
    # Node: Gather Intake (Phase 2 - Age + FICICW)
    # ------------------------------------------------------------------

    # Intake fields to collect, in order. Age first, then FICICW.
    _INTAKE_SEQUENCE = [
        "child_age",
        "frequency",
        "current_methods",
        "intensity",
        "where_happening",
        "life_impact",
    ]

    def _node_gather_intake(self, state: dict) -> dict:
        """Ask the next missing intake question (age first, then FICICW).

        This node is called:
        1. Right after classify_concern (first time) — asks the first missing field.
        2. After the parent responds to an intake question — processes the response,
           then asks the next missing field or transitions to probing.
        """
        concern_name = state.get("active_concern_name", "")

        # Determine what's still missing
        missing = self._get_missing_intake_fields(state)

        if not missing:
            # All intake collected — mark complete and move on
            state["intake_complete"] = True
            state["phase"] = "intake_done"
            state["trace_log"].append({
                "event": "intake_complete",
                "captured": self._get_captured_intake_fields(state),
            })
            return state

        # Ask the next missing field
        next_field = missing[0]
        question = self.llm.generate_intake_question(
            field_name=next_field,
            concern_name=concern_name,
            conversation_history=state.get("conversation_history", []),
        )

        state["bot_message"] = question
        state["conversation_history"].append({"role": "assistant", "content": question})
        state["phase"] = "intake"
        state["_current_intake_field"] = next_field
        state["trace_log"].append({
            "event": "intake_question_asked",
            "field": next_field,
            "missing_fields": missing,
            "captured": self._get_captured_intake_fields(state),
        })
        return state

    def _get_missing_intake_fields(self, state: dict) -> list[str]:
        """Return ordered list of intake fields still needed."""
        missing = []
        for field_name in self._INTAKE_SEQUENCE:
            if field_name == "child_age":
                if state.get("child_age_months") is None:
                    missing.append("child_age")
            else:
                if field_name not in state.get("intake_fields", {}):
                    missing.append(field_name)
        return missing

    def _get_captured_intake_fields(self, state: dict) -> list[str]:
        """Return list of intake fields already captured."""
        captured = list(state.get("intake_fields", {}).keys())
        if state.get("child_age_months") is not None:
            captured.append(f"child_age={state['child_age_months']}mo")
        return sorted(captured)

    def _process_intake_response(self, state: dict) -> dict:
        """Process the parent's response to an intake question."""
        parent_msg = state.get("parent_message", "")
        if not parent_msg:
            return state

        state["conversation_history"].append({"role": "user", "content": parent_msg})

        current_field = state.pop("_current_intake_field", None)

        if current_field == "child_age":
            # Extract age using LLM
            age = self.llm.extract_age_from_response(parent_msg)
            if age is not None:
                state["child_age_months"] = age
            state["trace_log"].append({
                "event": "intake_response",
                "field": "child_age",
                "extracted_age_months": age,
            })
        elif current_field:
            # Store the FICICW field value directly
            state.setdefault("intake_fields", {})[current_field] = parent_msg.strip()
            state["trace_log"].append({
                "event": "intake_response",
                "field": current_field,
                "value": parent_msg.strip()[:100],
            })

        # Run NLU extraction during intake ONLY for domain discovery (no tag assignment).
        # Tags are assigned only during the probing phase via question→answer→tag flow.
        domain_id = state.get("active_domain_id")
        active_domain = self.domains.get(domain_id) if domain_id else None
        concern_name = state.get("active_concern_name")
        active_concern = None
        if active_domain and concern_name:
            for c in active_domain.concerns:
                if c.name.lower() == concern_name.lower():
                    active_concern = c
                    break

        nlu_result = self.llm.extract_tags(
            parent_message=parent_msg,
            active_domain=active_domain,
            active_concern=active_concern,
            all_known_tags=self._all_known_tags,
            conversation_history=state.get("conversation_history", []),
        )
        # Do NOT assign tags during intake — tags come only from probing questions.
        # Only use NLU to discover new domains to queue.
        if nlu_result.discovered_domains:
            for new_domain in nlu_result.discovered_domains:
                if new_domain in self.domains and new_domain not in state.get("explored_domains", []):
                    state.setdefault("domain_queue", []).append(new_domain)

        return state

    def _route_after_intake(self, state: dict) -> str:
        """Route after intake: keep asking if fields missing, else probe."""
        if state.get("phase") == "intake_done":
            return "probe_domain"
        return "wait_for_input"

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

        # Find the active concern in the domain using validated matching
        concern = None
        if concern_name:
            validated_name = self._validate_concern_name(concern_name, domain)
            for c in domain.concerns:
                if c.name == validated_name:
                    concern = c
                    break
            if concern and concern.name != concern_name:
                # Update state with the validated name
                state["active_concern_name"] = concern.name

        # If still no match after validation, try LLM-based re-matching
        if not concern:
            state["trace_log"].append({
                "event": "concern_not_found_in_domain",
                "domain": domain_id,
                "attempted_concern": concern_name,
                "available_concerns": [c.name for c in domain.concerns[:10]],
            })
            # Re-match using conversation context and LLM
            rematch_name = self._find_relevant_concern_for_domain(state, domain)
            if rematch_name:
                for c in domain.concerns:
                    if c.name == rematch_name:
                        concern = c
                        state["active_concern_name"] = concern.name
                        state["trace_log"].append({
                            "event": "concern_rematched_by_llm",
                            "domain": domain_id,
                            "rematched_concern": concern.name,
                        })
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
            # Check if discovered tags triggered another concern in this domain
            next_concern = self._find_next_concern(state, domain)
            if next_concern:
                old_concern = concern.name
                state["active_concern_name"] = next_concern.name
                state["trace_log"].append({
                    "event": "concern_transition",
                    "domain": domain_id,
                    "from_concern": old_concern,
                    "to_concern": next_concern.name,
                    "reason": "tag_driven_transition",
                })
                # Generate a transition message so the parent understands the shift
                transition_msg = self.llm.generate_transition(
                    from_domain=f"{domain.display_name} - {old_concern}",
                    to_domain=f"{domain.display_name} - {next_concern.name}",
                    to_concern=next_concern.name,
                    conversation_history=state.get("conversation_history", []),
                )
                state["_pending_transition"] = transition_msg
                return self._node_probe_domain(state)  # recurse with new concern

            # No more tag-triggered concerns in this domain — move on
            state["should_end"] = True
            state["trace_log"].append({
                "event": "domain_probing_complete",
                "domain": domain_id,
                "reason": "no_more_tag_triggered_concerns",
            })
            return state

        question_obj = concern.questions[cursor]
        state.setdefault("question_cursors", {})[cursor_key] = cursor + 1

        # --- Intake-coverage check ---
        # If the parent already provided information during intake that covers
        # this question's triggers, present a confirmation instead of asking
        # the question fresh.
        intake_fields = state.get("intake_fields", {})
        coverage = self.llm.check_intake_coverage(
            question=question_obj,
            intake_fields=intake_fields,
            conversation_history=state.get("conversation_history", []),
        )

        pending_transition = state.pop("_pending_transition", None)

        if coverage:
            # The intake already covers this question — ask for confirmation
            confirmation_msg = coverage["summary"]
            if pending_transition:
                confirmation_msg = pending_transition + " " + confirmation_msg

            state["bot_message"] = confirmation_msg
            state["conversation_history"].append({"role": "assistant", "content": confirmation_msg})
            state["phase"] = "confirming"
            # Store what we're confirming so we can assign the tag on "yes"
            state["_pending_confirmation"] = {
                "tag": coverage["tag"],
                "question": question_obj,
            }
            state["trace_log"].append({
                "event": "intake_coverage_confirmation",
                "domain": domain_id,
                "concern": concern.name,
                "base_question": question_obj.question_text,
                "pending_tag": coverage["tag"],
                "cursor": cursor,
            })
            return state

        # --- Normal flow: ask the question fresh ---
        # Check intake status for enriching the question
        intake = PrimaryConcernIntake(fields=dict(intake_fields))
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

        if pending_transition:
            natural_question = pending_transition + " " + natural_question

        state["bot_message"] = natural_question
        state["conversation_history"].append({"role": "assistant", "content": natural_question})
        state["phase"] = "probing"
        # Store the current question object so _node_analyze_response can scope
        # tag extraction to THIS question's triggers only.
        state["_current_probe_question"] = question_obj
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
        """Find the next concern to probe — TAG-DRIVEN, not sequential.

        Only returns a concern if:
        1. A discovered tag connects to it (tag-triggered), OR
        2. It was explicitly discovered by the NLU from parent responses.

        This prevents the bot from walking through ALL concerns in a domain
        sequentially. The bot should only explore what the evidence leads to.
        """
        processed = state.get("processed_concerns", set())
        discovered_tags = state.get("discovered_tags", set())
        triggered_concerns = state.get("triggered_concerns", set())  # concerns explicitly triggered

        # First pass: find concerns that share tags with discovered tags
        for c in domain.concerns:
            key = f"{domain.domain_id}:{c.name}"
            if key in processed:
                continue
            # Check if any tag from this concern overlaps with discovered tags
            concern_tags = set(c.all_tags)
            if concern_tags.intersection(discovered_tags):
                state["trace_log"].append({
                    "event": "concern_triggered_by_tags",
                    "domain": domain.domain_id,
                    "concern": c.name,
                    "triggering_tags": sorted(concern_tags.intersection(discovered_tags)),
                    "reason": "discovered_tags_overlap",
                })
                return c

        # Second pass: check if any concern was explicitly triggered by NLU
        for c in domain.concerns:
            key = f"{domain.domain_id}:{c.name}"
            if key in processed:
                continue
            if c.name in triggered_concerns:
                state["trace_log"].append({
                    "event": "concern_triggered_explicitly",
                    "domain": domain.domain_id,
                    "concern": c.name,
                    "reason": "nlu_discovery",
                })
                return c

        # No tag-driven concerns found — do NOT fall back to sequential
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

        # Retrieve the question that was just asked (set by _node_probe_domain)
        # so tag extraction is scoped to THAT question's triggers only.
        current_question = state.pop("_current_probe_question", None)

        # NLU extraction — scoped to the current question when available
        nlu_result = self.llm.extract_tags(
            parent_message=parent_msg,
            active_domain=active_domain,
            active_concern=active_concern,
            all_known_tags=self._all_known_tags,
            conversation_history=state.get("conversation_history", []),
            current_question=current_question,
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
            state["domain_queue"] = queue

            new_domain = self.domains.get(next_domain_id)
            if not new_domain or not new_domain.concerns:
                state["trace_log"].append({
                    "event": "domain_skipped",
                    "domain": next_domain_id,
                    "reason": "no_concerns_in_domain",
                })
                return state

            # Use tags + LLM to find the RELEVANT concern in this domain
            matched_concern_name = self._find_relevant_concern_for_domain(state, new_domain)

            if not matched_concern_name:
                # No relevant concern - skip this domain entirely
                state["trace_log"].append({
                    "event": "domain_skipped",
                    "domain": next_domain_id,
                    "reason": "no_relevant_concern_from_conversation_evidence",
                })
                return state

            # Found a relevant concern - transition to it
            state["active_domain_id"] = next_domain_id
            state["active_concern_name"] = matched_concern_name

            if next_domain_id not in state.get("explored_domains", []):
                state.setdefault("explored_domains", []).append(next_domain_id)

            transition = self.llm.generate_transition(
                from_domain=old_domain,
                to_domain=new_domain.display_name,
                to_concern=matched_concern_name,
                conversation_history=state.get("conversation_history", []),
            )
            state["_pending_transition"] = transition

            state["trace_log"].append({
                "event": "domain_transition",
                "from_domain": old_domain,
                "to_domain": next_domain_id,
                "matched_concern": matched_concern_name,
                "reason": "evidence_based_concern_selection",
                "remaining_queue": list(queue),
            })
            return state

        # Nothing left - time for cross-domain check
        state["trace_log"].append({"event": "all_domains_exhausted"})
        return state

    def _find_relevant_concern_for_domain(
        self, state: dict, domain: ClinicalDomain
    ) -> str | None:
        """Find the most relevant concern in a new domain using tags + LLM.

        1. First check if any discovered tags overlap with a concern in this domain.
        2. If no tag overlap, ask the LLM to match based on conversation context.
        3. Returns None if no concern is relevant (domain should be skipped).
        """
        discovered_tags = state.get("discovered_tags", set())
        processed = state.get("processed_concerns", set())

        # Pass 1: Tag-based matching
        best_match = None
        best_overlap = 0
        for c in domain.concerns:
            key = f"{domain.domain_id}:{c.name}"
            if key in processed:
                continue
            concern_tags = set(c.all_tags)
            overlap = len(concern_tags.intersection(discovered_tags))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = c

        if best_match:
            state["trace_log"].append({
                "event": "concern_matched_by_tags",
                "domain": domain.domain_id,
                "concern": best_match.name,
                "overlapping_tags": sorted(set(best_match.all_tags).intersection(discovered_tags)),
            })
            return best_match.name

        # Pass 2: LLM-based matching using conversation context
        available_concerns = [
            c.name for c in domain.concerns
            if f"{domain.domain_id}:{c.name}" not in processed
        ]
        if not available_concerns:
            return None

        # Build conversation summary for the LLM
        history = state.get("conversation_history", [])
        summary_parts = []
        for msg in history[-8:]:
            role = "Parent" if msg.get("role") == "user" else "Counselor"
            summary_parts.append(f"{role}: {msg.get('content', '')[:150]}")
        conv_summary = "\n".join(summary_parts)

        matched = self.llm.match_concern_for_domain(
            domain_name=domain.display_name,
            available_concerns=available_concerns,
            conversation_summary=conv_summary,
            discovered_tags=sorted(discovered_tags),
        )

        if matched:
            state["trace_log"].append({
                "event": "concern_matched_by_llm",
                "domain": domain.domain_id,
                "concern": matched,
                "reason": "llm_conversation_context_match",
            })
            return matched

        return None

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
        """Phase 6: Full cross-domain analysis pipeline.

        Runs the 7-step Implementation Logic:
          Step 2: Safety check
          Step 3: Age logic filter
          Step 4: Convergence patterns (Required vs Supporting)
          Step 5: Confound rules (linked to convergence)
          Step 6: Severity escalation (FICICW-aware)
          Step 7: Differential diagnosis
        """
        tags = state.get("discovered_tags", set())
        state["_cross_domain_done"] = True

        # Run the full analysis pipeline
        report = run_analysis_pipeline(
            discovered_tags=tags,
            child_age_months=state.get("child_age_months"),
            intake_fields=state.get("intake_fields", {}),
            explored_domains=state.get("explored_domains", []),
            cross_domain_data=self.cross_domain_data,
        )

        # Store the full report for summary generation
        state["analysis_report"] = report

        # Update state with pipeline results
        state["convergence_hits"] = [
            f"{h.rule_id}: {h.pattern_name} ({h.clinical_hypothesis})"
            for h in report.convergence_hits
        ]
        state["confound_notes"] = [
            f"{h.rule_id}: {h.confound_name} - {h.action}"
            for h in report.confound_hits
        ]
        state["severity_level"] = report.severity.final_tier
        if report.safety_triggered:
            state["safety_escalated"] = True

        # Queue new domains from convergence patterns (evaluation recommendations)
        for hit in report.convergence_hits:
            for source_domain in hit.matched_domains:
                # Try to map source domain name back to domain_id
                for domain_id, domain in self.domains.items():
                    if (source_domain.lower() in domain.display_name.lower()
                            or domain.display_name.lower() in source_domain.lower()):
                        if domain_id not in state.get("explored_domains", []):
                            state.setdefault("domain_queue", []).append(domain_id)

        # Add the full pipeline trace
        state["trace_log"].append(report_to_trace(report))

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
        report = state.get("analysis_report")

        if state.get("safety_escalated"):
            summary = (
                "What you've described needs attention right away. "
                "Please reach out to a child psychologist or your pediatrician as soon as possible. "
                "If there's an immediate safety concern, contact your local emergency services or "
                "a crisis helpline."
            )
        else:
            summary = self.llm.generate_summary(
                discovered_tags=tags,
                severity_level=severity,
                domains_explored=explored,
                intake_data=redact_payload(intake),
                convergence_patterns=convergence,
                confound_notes=confounds,
                analysis_report=report,
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
            "triggered_concerns": set(),
            "bot_message": "",
            "should_end": False,
            "trace_log": [],
            "parent_message": "",
            "child_age_months": None,
            "convergence_hits": [],
            "confound_notes": [],
            "analysis_report": None,
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

        if state.get("phase") == "intake":
            # Parent is answering an intake question (age or FICICW)
            state = self._process_intake_response(state)
            # Ask next missing field or transition to probing
            state = self._node_gather_intake(state)
            if state.get("phase") == "intake_done":
                # All intake collected — start probing
                state = self._node_probe_domain(state)
            return state

        if state.get("phase") == "confirming":
            # Parent is responding to a confirmation of pre-answered info
            state["conversation_history"].append({"role": "user", "content": parent_message})
            pending = state.pop("_pending_confirmation", None)
            if pending and self.llm.check_confirmation(parent_message):
                # Parent confirmed — assign the tag
                state.setdefault("discovered_tags", set()).add(pending["tag"])
                state["trace_log"].append({
                    "event": "confirmation_accepted",
                    "tag": pending["tag"],
                    "base_question": pending["question"].question_text,
                })
            else:
                # Parent denied or corrected — ask the question fresh
                # The cursor already advanced, so we need to re-ask this question.
                # We do this by running analyze_response on their correction
                # (which may contain the real answer) and then continuing.
                if pending:
                    current_question = pending["question"]
                    nlu_result = self.llm.extract_tags(
                        parent_message=parent_message,
                        active_domain=self.domains.get(state.get("active_domain_id")),
                        active_concern=self._get_active_concern(state),
                        all_known_tags=self._all_known_tags,
                        conversation_history=state.get("conversation_history", []),
                        current_question=current_question,
                    )
                    if nlu_result.matched_tags:
                        state.setdefault("discovered_tags", set()).update(nlu_result.matched_tags)
                    state["trace_log"].append({
                        "event": "confirmation_denied",
                        "base_question": current_question.question_text,
                        "extracted_tags": nlu_result.matched_tags,
                    })
            # Continue to next question
            state = self._node_probe_domain(state)
            return state

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
            "child_age_months": state.get("child_age_months"),
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
