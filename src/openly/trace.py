from __future__ import annotations

from .state_schema import ConversationState


def build_turn_trace(
    state: ConversationState,
    *,
    routing_decision: str,
    next_question_reason: str,
) -> dict:
    """Build trace strictly from logic-state artifacts."""
    return {
        "variables_used": list(state.variables_used),
        "value_updates": list(state.value_updates),
        "routing_decision": routing_decision,
        "next_question_reason": next_question_reason,
    }
