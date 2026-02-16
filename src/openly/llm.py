"""LLM integration layer for NLU (tag extraction) and NLG (question generation).

Uses Anthropic Claude API with tightly constrained prompts and structured
output formats, following the 4-layer architecture principle.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from anthropic import Anthropic

from .domain_data import ClinicalDomain, PresentingConcern


# ---------------------------------------------------------------------------
# NLU output model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NLUResult:
    """Structured output from NLU analysis of a parent response."""
    matched_tags: list[str] = field(default_factory=list)
    discovered_domains: list[str] = field(default_factory=list)
    intake_fields: dict[str, str] = field(default_factory=dict)
    safety_flags: list[str] = field(default_factory=list)
    raw_llm_response: str = ""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class OpenlyLLM:
    """Wrapper around Anthropic Claude for NLU and NLG calls."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key == "your-key-here":
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Please set it in .env or as an environment variable."
            )
        self.client = Anthropic(api_key=key)
        self.model = model

    # ------------------------------------------------------------------
    # NLU: Extract tags from parent response
    # ------------------------------------------------------------------

    def extract_tags(
        self,
        parent_message: str,
        active_domain: ClinicalDomain | None,
        active_concern: PresentingConcern | None,
        all_known_tags: set[str],
        conversation_history: list[dict[str, str]],
    ) -> NLUResult:
        """Analyze parent response and extract AI tags, intake fields, safety flags."""

        # Build the tag reference for the prompt
        tag_context = ""
        if active_domain and active_concern:
            tag_context = f"\nActive domain: {active_domain.display_name}\n"
            tag_context += f"Active concern: {active_concern.name}\n"
            tag_context += "Possible tags for this concern:\n"
            for q in active_concern.questions:
                for t in q.triggers:
                    if t.ai_tag:
                        tag_context += f"  - Trigger: \"{t.trigger_text}\" -> Tag: {t.ai_tag}\n"
        elif active_domain:
            tag_context = f"\nActive domain: {active_domain.display_name}\n"
            tag_context += "Available tags in this domain:\n"
            for tag in sorted(active_domain.all_tags)[:50]:
                tag_context += f"  - {tag}\n"

        # Build conversation context (last 6 turns max)
        recent_turns = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        conv_context = ""
        for turn in recent_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                conv_context += f"Counselor: {content}\n"
            elif role == "user":
                conv_context += f"Parent: {content}\n"

        system_prompt = """You are the NLU (Natural Language Understanding) component of OPenly, a child developmental assessment chatbot for Indian parents/caregivers (children ages 1-7).

Your job is to analyze a parent's message and extract structured information. You must return ONLY valid JSON.

RULES:
1. Match parent responses to AI tags from the domain tree. Use the EXACT tag names provided.
2. If the parent's response matches a trigger pattern, assign the corresponding tag.
3. Extract intake fields if the parent mentions: frequency, intensity, current_methods (what they've tried), where_happening, life_impact.
4. Flag safety concerns: self_harm_signals, harm_to_others_signals, severe_regression_critical_skills, abuse_disclosure.
5. If you detect NEW concerns outside the active domain, list the relevant domain IDs.
6. Be conservative - only assign tags when there's clear evidence in the parent's words.
7. Tags should be in snake_case format.

DOMAIN IDs: behavioral_development, emotional_development, social_development_and_attachme, speech_language_and_communicat, motor_and_sensory_development, cognitive_development, body_and_physiology, environmental_factors, academics_and_learning"""

        user_prompt = f"""Analyze this parent's response and extract structured data.

{tag_context}

Recent conversation:
{conv_context}

Parent's latest message: "{parent_message}"

Return ONLY this JSON structure (no markdown, no explanation):
{{
  "matched_tags": ["tag1", "tag2"],
  "discovered_domains": ["domain_id"],
  "intake_fields": {{"frequency": "...", "intensity": "..."}},
  "safety_flags": []
}}

Only include intake_fields that are explicitly mentioned. Only include discovered_domains for NEW domains not currently being explored."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            return self._parse_nlu_response(raw, all_known_tags)
        except Exception as e:
            return NLUResult(raw_llm_response=f"ERROR: {e}")

    def _parse_nlu_response(self, raw: str, all_known_tags: set[str]) -> NLUResult:
        """Parse and validate LLM JSON response."""
        result = NLUResult(raw_llm_response=raw)

        # Strip markdown code fences if present
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return result

        result.matched_tags = [
            str(t).strip().lower().replace("-", "_").replace(" ", "_")
            for t in data.get("matched_tags", [])
            if t
        ]
        result.discovered_domains = [
            str(d).strip().lower()
            for d in data.get("discovered_domains", [])
            if d
        ]
        result.intake_fields = {
            str(k).strip().lower(): str(v).strip()
            for k, v in data.get("intake_fields", {}).items()
            if v and str(v).strip()
        }
        result.safety_flags = [
            str(f).strip().lower()
            for f in data.get("safety_flags", [])
            if f
        ]

        return result

    # ------------------------------------------------------------------
    # NLG: Generate natural questions and transitions
    # ------------------------------------------------------------------

    def generate_question(
        self,
        base_question: str,
        concern_name: str,
        domain_name: str,
        conversation_history: list[dict[str, str]],
        intake_status: dict | None = None,
    ) -> str:
        """Generate a natural, conversational version of a clinical question."""

        recent_turns = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        conv_context = ""
        for turn in recent_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                conv_context += f"You: {content}\n"
            elif role == "user":
                conv_context += f"Parent: {content}\n"

        intake_note = ""
        if intake_status:
            missing = intake_status.get("missing_fields", [])
            if missing:
                intake_note = f"\nYou still need to gather these intake dimensions naturally: {', '.join(missing)}\nWeave ONE of these into your question if it fits naturally.\n"

        system_prompt = """You are OPenly, a warm, empathetic child developmental assessment chatbot designed for Indian parents/caregivers of children ages 1-7.

TONE GUIDELINES:
- Warm, supportive, non-judgmental
- Use simple language (no clinical jargon with parents)
- Acknowledge what the parent shares before asking the next question
- Use Indian cultural context when appropriate
- Keep responses concise (2-3 sentences max)
- NEVER diagnose. You are identifying presenting concerns, not conditions.
- Frame questions as curiosity about the child, not interrogation

You will be given a base clinical question to ask. Rephrase it naturally as part of the conversation flow. You MUST ask a question - do not just validate."""

        user_prompt = f"""Current domain: {domain_name}
Current concern: {concern_name}
{intake_note}
Recent conversation:
{conv_context}

Base clinical question to ask (rephrase naturally): "{base_question}"

Generate your next conversational message (2-3 sentences, must end with a question):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            # Fallback to base question on error
            return base_question

    # ------------------------------------------------------------------
    # NLG: Generate opening message
    # ------------------------------------------------------------------

    def generate_opening(self) -> str:
        """Generate the initial greeting message."""
        system_prompt = """You are OPenly, a warm, empathetic child developmental assessment chatbot for Indian parents/caregivers of children ages 1-7.

Generate a brief, warm opening message that:
1. Introduces yourself as OPenly
2. Explains you're here to understand their concerns about their child's development
3. Asks them to share their biggest concern about their child
4. Is culturally appropriate for Indian families
5. Is 3-4 sentences max
6. Does NOT claim to be a doctor or diagnostician"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=system_prompt,
                messages=[{"role": "user", "content": "Generate the opening message."}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return (
                "Hello! I'm OPenly, and I'm here to help you understand your child's "
                "developmental journey. Every child grows at their own pace, and it's "
                "completely natural to have questions or concerns. What is the biggest "
                "concern you have about your child's development right now?"
            )

    # ------------------------------------------------------------------
    # NLG: Generate domain transition
    # ------------------------------------------------------------------

    def generate_transition(
        self,
        from_domain: str,
        to_domain: str,
        to_concern: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Generate smooth transition between domains."""
        recent_turns = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        conv_context = ""
        for turn in recent_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                conv_context += f"You: {content}\n"
            elif role == "user":
                conv_context += f"Parent: {content}\n"

        system_prompt = """You are OPenly, a warm child developmental chatbot for Indian parents.
Generate a brief, natural transition from one topic area to another.
1-2 sentences that acknowledge what was discussed and smoothly introduce the new topic.
Do NOT diagnose. Be warm and curious."""

        user_prompt = f"""Recent conversation:
{conv_context}

Transitioning from: {from_domain}
Moving to explore: {to_domain} (concern: {to_concern})

Generate a natural transition (1-2 sentences):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Thank you for sharing that. I'd also like to understand a bit about {to_concern}."

    # ------------------------------------------------------------------
    # NLG: Generate session summary
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        discovered_tags: set[str],
        severity_level: str,
        domains_explored: list[str],
        intake_data: dict[str, str],
        convergence_patterns: list[str],
        confound_notes: list[str],
    ) -> str:
        """Generate a parent-friendly session summary with referral recommendations."""

        system_prompt = """You are OPenly, generating a session summary for a parent.

RULES:
- Be warm, supportive, non-alarming
- NEVER diagnose. Use phrases like "areas worth exploring further" or "patterns we noticed"
- Provide generic referral recommendations (e.g., "developmental pediatrician", "speech therapist")
- Acknowledge the parent's effort in discussing these concerns
- Keep it concise (4-6 sentences)
- Use simple language, no clinical jargon
- Be culturally sensitive for Indian families"""

        user_prompt = f"""Session data:
- Severity level: {severity_level}
- Domains explored: {', '.join(domains_explored)}
- Key patterns identified: {', '.join(sorted(discovered_tags)[:10])}
- Intake data: {json.dumps(intake_data)}
- Convergence patterns noted: {', '.join(convergence_patterns) if convergence_patterns else 'None'}
- Confound considerations: {', '.join(confound_notes) if confound_notes else 'None'}

Generate a parent-friendly summary with generic referral recommendations:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return (
                "Thank you for sharing your concerns today. Based on our conversation, "
                "there are some areas of your child's development worth exploring further "
                "with a professional. We recommend consulting a developmental pediatrician "
                "for a comprehensive evaluation."
            )

    # ------------------------------------------------------------------
    # NLU: Initial concern classification
    # ------------------------------------------------------------------

    def classify_initial_concern(
        self,
        parent_message: str,
        available_domains: list[dict[str, str]],
    ) -> dict:
        """Classify the parent's initial concern into domain(s) and presenting concern(s)."""

        domain_list = "\n".join(
            f"  - {d['domain_id']}: {d['display_name']} (concerns: {d['sample_concerns']})"
            for d in available_domains
        )

        system_prompt = """You are the NLU component of OPenly, a child developmental chatbot.
Analyze the parent's initial concern and classify it into the most relevant domain(s) and presenting concern(s).
Return ONLY valid JSON, no markdown."""

        user_prompt = f"""Available domains:
{domain_list}

Parent says: "{parent_message}"

Return this JSON structure:
{{
  "primary_domain": "domain_id",
  "primary_concern": "the presenting concern name from the domain tree that best matches",
  "additional_domains": ["domain_id"],
  "intake_fields": {{"frequency": "...", "intensity": "..."}},
  "safety_flags": []
}}

Only include fields explicitly mentioned by the parent."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            return json.loads(cleaned.strip())
        except Exception:
            return {
                "primary_domain": available_domains[0]["domain_id"] if available_domains else "",
                "primary_concern": "",
                "additional_domains": [],
                "intake_fields": {},
                "safety_flags": [],
            }
