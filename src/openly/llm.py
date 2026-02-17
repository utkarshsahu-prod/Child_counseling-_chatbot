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

from .domain_data import ClinicalDomain, DomainQuestion, PresentingConcern


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
        current_question: DomainQuestion | None = None,
    ) -> NLUResult:
        """Analyze parent response and extract AI tags, intake fields, safety flags.

        When *current_question* is provided the prompt and post-hoc filter are
        scoped to ONLY that question's triggers/tags.  This ensures tags are
        assigned through the question→answer→tag flow, not broadly inferred.
        """

        # Build the tag reference for the prompt
        tag_context = ""
        if current_question:
            # Scoped mode: only show the current question's triggers
            tag_context = f"\nActive domain: {active_domain.display_name}\n" if active_domain else ""
            tag_context += f"Active concern: {active_concern.name}\n" if active_concern else ""
            tag_context += f"Question just asked: \"{current_question.question_text}\"\n"
            tag_context += "Valid response triggers for THIS question ONLY:\n"
            for t in current_question.triggers:
                if t.ai_tag:
                    tag_context += f"  - Trigger: \"{t.trigger_text}\" -> Tag: {t.ai_tag}\n"
        elif active_domain and active_concern:
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

        # Compute allowed tags for post-hoc filtering
        allowed_tags: set[str] | None = None
        if current_question:
            allowed_tags = {
                t.ai_tag.strip().lower().replace("-", "_").replace(" ", "_")
                for t in current_question.triggers
                if t.ai_tag
            }

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
            return self._parse_nlu_response(raw, all_known_tags, allowed_tags=allowed_tags)
        except Exception as e:
            return NLUResult(raw_llm_response=f"ERROR: {e}")

    def _parse_nlu_response(
        self,
        raw: str,
        all_known_tags: set[str],
        allowed_tags: set[str] | None = None,
    ) -> NLUResult:
        """Parse and validate LLM JSON response.

        When *allowed_tags* is provided, only tags in that set are kept.
        This enforces question-scoped tag assignment during probing.
        """
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

        # Normalize tags
        normalized_tags = [
            str(t).strip().lower().replace("-", "_").replace(" ", "_")
            for t in data.get("matched_tags", [])
            if t
        ]
        # Post-hoc filter: if allowed_tags is set, only keep tags from the current question
        if allowed_tags is not None:
            result.matched_tags = [t for t in normalized_tags if t in allowed_tags]
        else:
            result.matched_tags = normalized_tags

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
    # NLU: Check if intake answers already cover a probing question
    # ------------------------------------------------------------------

    def check_intake_coverage(
        self,
        question: DomainQuestion,
        intake_fields: dict[str, str],
        conversation_history: list[dict[str, str]],
    ) -> dict | None:
        """Check if prior intake answers already cover a probing question's triggers.

        Returns a dict with ``covered=True``, the matching ``tag``, and a
        ``summary`` the bot can present for confirmation — or *None* if the
        intake does not provide enough information.
        """
        if not intake_fields:
            return None

        trigger_list = "\n".join(
            f"  - Trigger: \"{t.trigger_text}\" -> Tag: {t.ai_tag}"
            for t in question.triggers if t.ai_tag
        )
        if not trigger_list:
            return None

        intake_summary = "\n".join(
            f"  {k}: {v}" for k, v in intake_fields.items()
        )

        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        conv_context = "\n".join(
            f"{'Parent' if t.get('role') == 'user' else 'Counselor'}: {t.get('content', '')}"
            for t in recent
        )

        system_prompt = """You are the NLU component of OPenly, a child developmental chatbot.
Determine whether the parent's prior intake answers already provide enough information
to answer a specific probing question. Return ONLY valid JSON, no markdown.

RULES:
1. Only say covered=true if the intake clearly addresses the question's trigger patterns.
2. If covered, pick the BEST matching trigger tag and write a SHORT summary (under 25 words)
   of what the parent said, phrased as a confirmation question.
3. Be conservative — only mark covered if there is clear, direct evidence."""

        user_prompt = f"""Probing question: "{question.question_text}"

Triggers for this question:
{trigger_list}

Intake data collected so far:
{intake_summary}

Conversation context:
{conv_context}

Return this JSON:
{{
  "covered": true/false,
  "matched_tag": "exact_tag_name or null",
  "summary": "Short confirmation question for the parent, or null"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned.strip())
            if data.get("covered"):
                tag = data.get("matched_tag")
                summary = data.get("summary")
                if tag and summary:
                    # Validate tag is actually from this question
                    valid_tags = {
                        t.ai_tag.strip().lower().replace("-", "_").replace(" ", "_")
                        for t in question.triggers if t.ai_tag
                    }
                    norm_tag = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
                    if norm_tag in valid_tags:
                        return {"covered": True, "tag": norm_tag, "summary": summary}
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # NLU: Check if parent's response confirms a pre-answered question
    # ------------------------------------------------------------------

    def check_confirmation(self, parent_message: str) -> bool:
        """Return True if the parent's message is an affirmative confirmation."""
        system_prompt = """Determine if the parent's message is an affirmative confirmation (yes, okay, correct, that's right, etc.) or a denial/correction.
Return ONLY valid JSON: {"confirmed": true} or {"confirmed": false}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=30,
                system=system_prompt,
                messages=[{"role": "user", "content": f'Parent says: "{parent_message}"'}],
            )
            raw = response.content[0].text.strip()
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned.strip())
            return bool(data.get("confirmed", False))
        except Exception:
            return False

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

        system_prompt = """You are OPenly, a child developmental screening chatbot for Indian parents/caregivers (children ages 1-7). You conduct structured intake interviews the way a developmental pediatrician would — professional, efficient, and kind but not effusive.

STYLE RULES:
- Be CONCISE. 1 sentence max before your question. No long validations or restatements.
- DO NOT paraphrase or repeat back what the parent just said. Move forward.
- A brief "Okay" or "Got it" is sufficient acknowledgment. Do NOT say things like "Thank you so much for sharing that" or "I really appreciate you telling me this".
- Ask ONE clear question per turn. No compound questions.
- Use plain, everyday language. No clinical jargon.
- Sound like a real person having a focused conversation, not a therapy chatbot.
- NEVER diagnose or label. You are gathering information, not interpreting it.
- Keep total response under 30 words when possible.

BAD examples (too verbose/empathetic):
- "That sounds really challenging. It's completely normal to feel concerned about your child. Can you tell me more about when this happens?"
- "Thank you for sharing that with me. I can see this is important to you. How often would you say this occurs?"

GOOD examples (professional, direct):
- "Got it. How often does this happen — daily, or more like a few times a week?"
- "Okay. And does this happen more at home or at school?"
- "When did you first start noticing this?"

You will be given a base clinical question. Rephrase it naturally and ask it directly."""

        user_prompt = f"""Current domain: {domain_name}
Current concern: {concern_name}
{intake_note}
Recent conversation:
{conv_context}

Base clinical question to ask (rephrase naturally): "{base_question}"

Generate your next message (1-2 sentences max, under 30 words, must end with a question):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
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
        system_prompt = """You are OPenly, a child developmental screening chatbot for Indian parents/caregivers (children ages 1-7).

Generate a brief, professional opening message that:
1. Introduces yourself as OPenly in one line
2. Says you'll ask a few questions to understand their child's development
3. Asks TWO things: (a) the child's age, and (b) their main concern about the child
4. 2-3 sentences total. No fluff, no motivational language, no "every child is unique" type filler.
5. Does NOT claim to be a doctor or diagnostician
6. Sounds like a professional intake, not a greeting card
7. Example ending: "To start, how old is your child and what's your main concern about their development?"
"""

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

        system_prompt = """You are OPenly, a child developmental screening chatbot.
Generate a brief, natural transition from one topic to another.
1 sentence max. Don't summarize what was discussed — just pivot to the new topic.
Example: "I'd also like to ask about how they're doing with [new topic]."
Do NOT diagnose. Keep it short and direct."""

        user_prompt = f"""Recent conversation:
{conv_context}

Transitioning from: {from_domain}
Moving to explore: {to_domain} (concern: {to_concern})

Generate a natural transition (1-2 sentences):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=60,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"I'd also like to ask about {to_concern}."

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
        analysis_report: object | None = None,
    ) -> str:
        """Generate a parent-friendly session summary with referral recommendations.

        Uses the full analysis report (convergence, confounds, severity, differentials)
        when available.
        """

        system_prompt = """You are OPenly, generating a session summary for a parent.

RULES:
- Be professional and clear, not overly warm or vague.
- NEVER diagnose or label the child. Use phrases like "patterns worth discussing with a specialist" or "areas that may benefit from further evaluation".
- When convergence patterns are found, explain what areas showed connected patterns (without using clinical names like "CP-ASD-01").
- When confounds are found, mention that environmental factors may be contributing and what to try first.
- When differentials exist, note that the patterns could point to different things and a specialist can help clarify.
- Provide SPECIFIC referral recommendations based on the findings (e.g., "developmental pediatrician for attention concerns", "OT for sensory processing").
- Keep it concise — 4-6 sentences. No filler.
- If severity is low and no convergence, say so directly — don't over-pathologize.
- If confounds are present, emphasize trying environmental changes first before specialist referral."""

        # Build structured findings from the analysis report
        findings_section = ""
        if analysis_report is not None:
            findings = []

            # Age filter
            age_filter = getattr(analysis_report, 'age_filter', None)
            if age_filter and age_filter.suppressed_tags:
                findings.append(
                    f"Age-adjusted: {len(age_filter.suppressed_tags)} tag(s) suppressed as age-normative"
                )

            # Convergence
            conv_hits = getattr(analysis_report, 'convergence_hits', [])
            if conv_hits:
                for h in conv_hits:
                    status = "CONFOUNDED" if h.is_confounded else "ACTIVE"
                    findings.append(
                        f"Convergence: {h.pattern_name} — {h.clinical_hypothesis} "
                        f"(matched {h.domains_matched}/{h.min_domains_required} domains, "
                        f"confidence: {h.confidence_tier}, status: {status})"
                    )
                    if h.recommended_evaluation:
                        findings.append(f"  → Recommended evaluation: {h.recommended_evaluation}")
                    if h.confound_details:
                        findings.append(f"  → Confounds: {'; '.join(h.confound_details)}")

            # Confounds
            conf_hits = getattr(analysis_report, 'confound_hits', [])
            if conf_hits:
                for c in conf_hits:
                    findings.append(
                        f"Confound: {c.confound_name} — {c.action}"
                    )
                    if c.parent_message:
                        findings.append(f"  → Parent guidance: {c.parent_message}")

            # Severity reasoning
            sev = getattr(analysis_report, 'severity', None)
            if sev and sev.reasoning:
                for r in sev.reasoning:
                    findings.append(f"Severity reasoning: {r}")

            # Differentials
            diffs = getattr(analysis_report, 'differentials', [])
            if diffs:
                for d in diffs:
                    findings.append(
                        f"Differential: {d.condition_a} vs {d.condition_b} — "
                        f"leaning toward {'A (' + d.condition_a + ')' if d.leaning == 'A' else 'B (' + d.condition_b + ')' if d.leaning == 'B' else 'unclear'}. "
                        f"Key differentiator: {d.differentiator}"
                    )

            if findings:
                findings_section = "\n\nCLINICAL ANALYSIS FINDINGS:\n" + "\n".join(f"- {f}" for f in findings)

        user_prompt = f"""Session data:
- Severity level: {severity_level}
- Domains explored: {', '.join(domains_explored)}
- Key tags: {', '.join(sorted(discovered_tags)[:15])}
- Intake data: {json.dumps(intake_data)}
- Convergence patterns: {', '.join(convergence_patterns) if convergence_patterns else 'None'}
- Confound considerations: {', '.join(confound_notes) if confound_notes else 'None'}
{findings_section}

Generate a parent-friendly summary (4-6 sentences) with specific recommendations:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return (
                "Based on our conversation, there are some areas worth discussing "
                "with a developmental pediatrician for a closer look. "
                "We recommend scheduling a consultation at your convenience."
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
            f"  - {d['domain_id']}: {d['display_name']}\n    Concerns: {d['all_concerns']}"
            for d in available_domains
        )

        system_prompt = """You are the NLU component of OPenly, a child developmental chatbot.
Analyze the parent's initial concern and classify it into the most relevant domain(s) and presenting concern(s).
CRITICAL: The primary_concern MUST be an EXACT concern name from the list below. Copy it exactly as written.
Return ONLY valid JSON, no markdown."""

        user_prompt = f"""Available domains and their concerns:
{domain_list}

Parent says: "{parent_message}"

Return this JSON structure:
{{
  "primary_domain": "domain_id",
  "primary_concern": "EXACT concern name from the list above — copy it character for character",
  "additional_domains": ["domain_id"],
  "child_age_months": null,
  "intake_fields": {{"frequency": "...", "intensity": "..."}},
  "safety_flags": []
}}

IMPORTANT:
- primary_concern MUST be an exact string from the concerns listed above. Do NOT rephrase or create your own name.
- If the parent mentions the child's age, convert it to months and set child_age_months (e.g. "5 years" = 60, "3.5 years" = 42, "18 months" = 18). Set null if not mentioned.
- Only include intake_fields explicitly mentioned by the parent."""

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

    # ------------------------------------------------------------------
    # NLG: Generate intake question (age, FICICW)
    # ------------------------------------------------------------------

    def generate_intake_question(
        self,
        field_name: str,
        concern_name: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Generate a natural intake question for a specific FICICW field."""

        field_prompts = {
            "child_age": "Ask how old the child is (in years and months).",
            "frequency": "Ask how often this concern happens (daily, weekly, etc.).",
            "intensity": "Ask how strong or severe it gets when it does happen.",
            "current_methods": "Ask what they've already tried to address this (any strategies, therapy, etc.).",
            "where_happening": "Ask where this mainly happens — at home, school, or both.",
            "life_impact": "Ask how this is affecting the child's daily life, relationships, or learning.",
        }

        prompt_instruction = field_prompts.get(field_name, f"Ask about {field_name}.")

        recent_turns = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        conv_context = ""
        for turn in recent_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                conv_context += f"You: {content}\n"
            elif role == "user":
                conv_context += f"Parent: {content}\n"

        system_prompt = """You are OPenly, a child developmental screening chatbot. You conduct structured intake interviews the way a developmental pediatrician would — professional, efficient, kind but not effusive.

RULES:
- 1 sentence only. Under 25 words.
- Ask ONE specific question about the intake dimension.
- Don't repeat or paraphrase what the parent already said.
- A brief "Okay" or "Got it" is sufficient acknowledgment.
- Sound like a real professional, not a therapy chatbot."""

        user_prompt = f"""Concern being discussed: {concern_name}

Recent conversation:
{conv_context}

Intake question to ask: {prompt_instruction}

Generate a natural, concise question (1 sentence, under 25 words):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=60,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            # Fallback questions
            fallbacks = {
                "child_age": "How old is your child?",
                "frequency": "How often does this happen?",
                "intensity": "How intense does it get when it happens?",
                "current_methods": "Have you tried anything to address this so far?",
                "where_happening": "Does this mainly happen at home, school, or both?",
                "life_impact": "How is this affecting their daily routine?",
            }
            return fallbacks.get(field_name, f"Can you tell me about the {field_name}?")

    def extract_age_from_response(self, parent_message: str) -> int | None:
        """Extract child age in months from a parent's response."""
        system_prompt = """Extract the child's age from the parent's message and convert to months.
Return ONLY a JSON object, no markdown."""

        user_prompt = f"""Parent says: "{parent_message}"

Return this JSON:
{{"age_months": <integer or null>}}

Examples:
- "She is 5 years old" -> {{"age_months": 60}}
- "He's 3 and a half" -> {{"age_months": 42}}
- "18 months" -> {{"age_months": 18}}
- "Almost 2" -> {{"age_months": 22}}
- No age mentioned -> {{"age_months": null}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned.strip())
            age = data.get("age_months")
            if age is not None:
                return int(age)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # NLU: Match concern when transitioning to a queued domain
    # ------------------------------------------------------------------

    def match_concern_for_domain(
        self,
        domain_name: str,
        available_concerns: list[str],
        conversation_summary: str,
        discovered_tags: list[str],
    ) -> str | None:
        """Identify the most relevant concern in a new domain based on conversation context.

        Returns the concern name or None if no concern is relevant.
        """
        concern_list = "\n".join(f"  - {c}" for c in available_concerns)

        system_prompt = """You are the NLU component of OPenly, a child developmental chatbot.
Given a conversation summary and the tags discovered so far, identify which presenting
concern in the new domain is MOST relevant to explore. Only pick a concern if there is
genuine evidence from the conversation that warrants exploring it.
Return ONLY valid JSON, no markdown."""

        user_prompt = f"""Domain to explore: {domain_name}

Available concerns in this domain:
{concern_list}

Summary of conversation so far: {conversation_summary}

Tags discovered: {', '.join(discovered_tags)}

Return this JSON:
{{
  "selected_concern": "exact concern name from the list above, or null if none are relevant",
  "reason": "1-sentence explanation of why this concern connects to what the parent shared"
}}

IMPORTANT: Only select a concern if the conversation evidence genuinely connects to it.
If nothing from the conversation relates to any concern in this domain, return null."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned.strip())
            selected = data.get("selected_concern")
            if selected and selected != "null" and selected.lower() != "none":
                return selected
            return None
        except Exception:
            return None
