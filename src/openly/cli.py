"""CLI chat interface for testing the OPenly counseling chatbot.

Usage:
    python -m src.openly.cli
    python -m src.openly.cli --model claude-sonnet-4-20250514
    python -m src.openly.cli --no-trace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="OPenly Child Development Chatbot CLI")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Anthropic model to use")
    parser.add_argument("--no-trace", action="store_true", help="Hide reasoning trace output")
    parser.add_argument("--domain-tree", default=None, help="Path to Domain_tree_UPDATED.xlsx")
    parser.add_argument("--cross-domain", default=None, help="Path to Cross_Domain_Logic_UPDATED.xlsx")
    args = parser.parse_args()

    # Load environment
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("Please set it in .env file or export it:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  OR edit .env file in the project root")
        sys.exit(1)

    # Resolve data file paths
    domain_tree = args.domain_tree or str(project_root / "Domain_tree_UPDATED.xlsx")
    cross_domain = args.cross_domain or str(project_root / "Cross_Domain_Logic_UPDATED.xlsx")

    if not Path(domain_tree).exists():
        print(f"\n[ERROR] Domain tree file not found: {domain_tree}")
        sys.exit(1)
    if not Path(cross_domain).exists():
        print(f"\n[ERROR] Cross-domain file not found: {cross_domain}")
        sys.exit(1)

    # Import here to avoid import errors if dependencies missing
    from .graph import build_graph

    print("\n" + "=" * 60)
    print("  OPenly - Child Development Assessment Chatbot (MVP)")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Trace: {'hidden' if args.no_trace else 'shown'}")
    print("  Type 'quit' or 'exit' to end the session")
    print("  Type 'trace' to toggle trace display")
    print("  Type 'state' to see current state summary")
    print("=" * 60)

    print("\n  Loading domain data...")
    try:
        graph = build_graph(
            domain_tree_path=domain_tree,
            cross_domain_path=cross_domain,
            api_key=api_key,
            model=args.model,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to build graph: {e}")
        sys.exit(1)

    print("  Starting session...\n")

    # Start session
    state = graph.start_session()
    show_trace = not args.no_trace

    # Print opening message
    bot_msg = state.get("bot_message", "")
    if bot_msg:
        _print_bot(bot_msg)
        if show_trace:
            _print_trace(graph.get_trace_json(state))

    # Main conversation loop
    while not state.get("should_end", False):
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session ended by user.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\n  Session ended.")
            break

        if user_input.lower() == "trace":
            show_trace = not show_trace
            print(f"  [Trace display: {'ON' if show_trace else 'OFF'}]")
            continue

        if user_input.lower() == "state":
            _print_state_summary(state)
            continue

        # Process message
        state = graph.process_message(state, user_input)

        bot_msg = state.get("bot_message", "")
        if bot_msg:
            _print_bot(bot_msg)

        if show_trace:
            _print_trace(graph.get_trace_json(state))

        if state.get("should_end"):
            print("\n  " + "-" * 40)
            print("  Session complete.")
            if show_trace:
                print("\n  FINAL TRACE:")
                _print_trace(graph.get_trace_json(state))
            break

    # Print final summary
    print("\n" + "=" * 60)
    print("  Session Statistics:")
    print(f"    Domains explored: {state.get('explored_domains', [])}")
    print(f"    Tags discovered: {len(state.get('discovered_tags', set()))}")
    print(f"    Severity level: {state.get('severity_level', 'low')}")
    print(f"    Intake complete: {state.get('intake_complete', False)}")
    print(f"    Safety escalated: {state.get('safety_escalated', False)}")
    print(f"    Turns: {len([m for m in state.get('conversation_history', []) if m.get('role') == 'user'])}")
    print("=" * 60 + "\n")


def _print_bot(message: str):
    """Print bot message with formatting."""
    print()
    print("  " + "-" * 40)
    lines = message.split("\n")
    for line in lines:
        # Word wrap at 70 chars
        while len(line) > 70:
            split_at = line[:70].rfind(" ")
            if split_at == -1:
                split_at = 70
            print(f"  Bot: {line[:split_at]}")
            line = line[split_at:].lstrip()
        print(f"  Bot: {line}")
    print("  " + "-" * 40)


def _print_trace(trace_json: str):
    """Print trace in a compact format."""
    try:
        trace = json.loads(trace_json)
        print("\n  [TRACE]")
        print(f"    Phase: {trace.get('phase', '?')}")
        print(f"    Severity: {trace.get('severity_level', '?')}")
        print(f"    Active: {trace.get('active_domain', 'none')} / {trace.get('active_concern', 'none')}")
        tags = trace.get("discovered_tags", [])
        if tags:
            print(f"    Tags ({len(tags)}): {', '.join(tags[:8])}{'...' if len(tags) > 8 else ''}")
        queue = trace.get("domain_queue", [])
        if queue:
            print(f"    Queue: {queue}")
        intake = trace.get("intake_fields", {})
        if intake:
            print(f"    Intake: {json.dumps(intake)}")
        convergence = trace.get("convergence_hits", [])
        if convergence:
            print(f"    Convergence: {convergence}")

        # Show latest trace event
        events = trace.get("trace_events", [])
        if events:
            latest = events[-1]
            print(f"    Latest event: {latest.get('event', '?')}")
        print("  [/TRACE]")
    except Exception:
        print(f"\n  [TRACE] {trace_json[:200]}")


def _print_state_summary(state: dict):
    """Print a compact state summary."""
    print("\n  === STATE SUMMARY ===")
    print(f"    Phase: {state.get('phase', '?')}")
    print(f"    Active domain: {state.get('active_domain_id', 'none')}")
    print(f"    Active concern: {state.get('active_concern_name', 'none')}")
    print(f"    Domain queue: {state.get('domain_queue', [])}")
    print(f"    Explored: {state.get('explored_domains', [])}")
    print(f"    Severity: {state.get('severity_level', 'low')}")
    print(f"    Tags: {sorted(state.get('discovered_tags', set()))}")
    print(f"    Intake: {state.get('intake_fields', {})}")
    print(f"    Intake complete: {state.get('intake_complete', False)}")
    print(f"    Processed concerns: {state.get('processed_concerns', set())}")
    print(f"    Conversation turns: {len(state.get('conversation_history', []))}")
    print("  =====================")


if __name__ == "__main__":
    main()
