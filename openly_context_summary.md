# Openly — Architecture & Context Summary

This document is a knowledge file for the Openly Build Coach project. It contains all key architectural decisions and existing artifacts so the coach can reference them when generating session prompts and reviewing outputs.

---

## Product Overview

**Openly** (formerly WE Platform) is an AI-powered child developmental assessment chatbot for the Indian market. It sits between "everyday parental concern" and "clinical assessment" — helping parents articulate and understand their child's developmental concerns through natural conversation, then connecting them with appropriate professionals.

**What it is NOT:** A diagnostic tool. It identifies presenting concerns and patterns, not diagnoses.

---

## 4-Layer Architecture

| Layer | Role | Implementation | Uses LLM? |
|-------|------|----------------|-----------|
| **Data Layer** | Stores domain trees, triggers, tags, cross-domain rules | JSON files loaded at runtime | No |
| **Logic Layer** | Deterministic routing: which domain to probe, when to re-route, skip logic, queue management | Python functions as LangGraph nodes | No |
| **NLU Layer** | Understands parent responses: intent classification, trigger matching, discovery scanning, safety detection | LLM API calls with structured prompts | Yes |
| **NLG Layer** | Generates natural questions, FICICW gathering, topic transitions, session summaries | LLM API calls with structured prompts | Yes |

**Key principle:** Logic Layer is deterministic (no LLM randomness). NLU/NLG layers use LLMs but with tightly constrained prompts and structured output formats.

---

## 8-Phase Conversation Flow

| Phase | Name | What Happens |
|-------|------|-------------|
| 0 | Session Init | Load domain trees, initialize state, set conversation parameters |
| 1 | Initial Input & Multi-Intent Scan | Parent describes concerns freely. NLU scans for triggers across ALL 10 domains simultaneously. Creates priority queue of domains to explore. |
| 2 | FICICW Natural Context Gathering | Single fluid conversation gathering Frequency, Intensity, Context, Impact, Coping, What-if for the primary concern. NOT a rigid checklist — woven into natural dialogue. |
| 3 | Structured Probing Loop | Domain-specific deep-dive questions from the domain tree. Skip logic avoids re-asking what was already covered in Phase 2. |
| 4 | Dynamic Discovery & Re-routing | Every parent response is scanned for NEW triggers. If found, new domain branches are added to the queue. Mid-conversation re-routing. |
| 5 | Branch Completion & Queue Transition | Current domain branch reaches a leaf node or sufficient depth. Smooth transition to next domain in queue. |
| 6 | Cross-Domain Pattern Detection | After exploring multiple domains, check for convergence patterns (e.g., sensory + social + communication → possible ASD screen). Apply confound rules and differential diagnosis logic. |
| 7 | Session Summary & Next Steps | Generate parent-friendly summary of identified concerns, recommended next steps, and professional referrals if appropriate. |

**FICICW Framework:**
- **F**requency — How often does this happen?
- **I**ntensity — How severe is it when it happens?
- **C**ontext — When/where does it happen? Any triggers?
- **I**mpact — How does it affect daily life, school, relationships?
- **C**oping — What have you tried? What helps?
- **W**hat-if — What happens if nothing changes? What are you most worried about?

---

## 10 Clinical Domains

1. **Behavioral Development** — Tantrums, aggression, defiance, hyperactivity, impulsivity, self-regulation
2. **Emotional Development** — Anxiety, mood, emotional regulation, attachment, self-esteem
3. **Social Development** — Peer interaction, social skills, empathy, cooperation, play skills
4. **Speech & Language** — Articulation, receptive/expressive language, pragmatics, fluency
5. **Motor & Sensory** — Gross/fine motor, sensory processing, coordination, body awareness
6. **Cognitive Development** — Attention, memory, problem-solving, executive function, processing speed
7. **Body & Physiology** — Sleep, eating, toileting, somatic complaints, physical health
8. **Environmental Factors** — Family dynamics, school environment, life events, socioeconomic factors
9. **Academics** — Reading, writing, math, learning difficulties, school performance
10. **Cross-Domain Logic** — Convergence patterns, confound rules, severity escalation, differential indicators

---

## Cross-Domain Logic (5 Rule Types)

1. **Convergence Patterns** — When tags from multiple domains appear together, suggest a higher-level screen (e.g., social + communication + sensory → ASD screen)
2. **Confound Rules** — Environmental factors that might explain symptoms before assuming a developmental condition (e.g., recent family disruption might explain behavioral regression)
3. **Severity Escalation** — When individual domain findings, combined, indicate higher severity than any single domain alone
4. **Differential Diagnosis Indicators** — Patterns that help distinguish between similar presentations (e.g., ADHD vs anxiety-driven attention problems)
5. **Red Flag Rules** — DSM-5 aligned indicators that require immediate professional referral regardless of conversation stage

---

## Existing Artifacts (Pre-Build Plan)

These files exist and should be uploaded to relevant build sessions:

| File | Description | Upload To |
|------|-------------|-----------|
| `Domain_tree.xlsx` | 10-sheet Excel with all domain trees (triggers, questions, tags, red flags) | Sessions 1, 2 |
| `Cross_Domain_Logic.xlsx` | 5-sheet matrix: Convergence, Confounds, Severity, Differential, Red Flags | Session 2 |
| `NoteBookLM_output.docx` | Architecture discussion: 4-layer design, LangGraph mapping, synthetic personas | Session 3 |
| `userflow.jsx` | Interactive React diagram of the 8-phase flow | Reference only |
| `platform_comparison.jsx` | LangSmith vs Langfuse vs W&B Weave comparison | Session 7 |
| `WE_Project_Instructions.docx` | Legal/clinical/safety framework documentation | Sessions 4, 10 |
| `Openly Tone of Voice` | Brand voice guidelines for NLG prompts | Session 4 |
| `Listening Agent System Prompt` | Existing system prompt with domain coverage requirements | Session 4 |

---

## 5 Synthetic Test Personas

These personas were designed to test different conversation paths and edge cases:

1. **Priya** — First-time mother, articulate, single clear concern (speech delay). Tests: clean single-domain flow.
2. **Rahul** — Father, vague concerns, multiple overlapping issues (behavioral + attention + academic). Tests: multi-intent scan, queue management, cross-domain convergence.
3. **Anita** — Anxious grandmother, over-reports symptoms, emotional language. Tests: NLU robustness, filtering noise from signal, sensitivity in responses.
4. **Deepak** — Skeptical father, minimal responses, needs to be drawn out. Tests: NLG question quality, handling short/evasive answers, conversation momentum.
5. **Meera** — Mother with urgent safety concern (child showing signs of abuse). Tests: safety protocol activation, crisis detection, immediate referral pathway.

---

## Key Technical Decisions (Already Made)

1. **LangGraph over CrewAI/AutoGen** — More control over state management and conditional routing, better for the deterministic Logic layer
2. **Anthropic Claude as primary LLM** — For both NLU and NLG layers
3. **LangSmith for testing** — Tracing, evaluation, dataset management
4. **JSON domain trees (not database)** — Loaded at runtime, versioned in git, easy to update by clinical team
5. **4-layer separation** — Keeps LLM calls isolated to NLU/NLG, making the system testable and predictable
6. **FICICW as fluid conversation** — Not a rigid form-fill, but naturally woven into dialogue (this is a hard NLG problem)
7. **Skip logic** — If Phase 2 (FICICW) already covered certain domain questions, Phase 3 (Structured Probing) should not re-ask them

---

## Safety & Compliance Requirements

**Crisis Detection (must be active at ALL phases):**
- Abuse disclosure → immediate safety protocol
- Self-harm mentions → immediate safety protocol  
- Domestic violence indicators → immediate safety protocol
- Suicidal ideation → immediate safety protocol
- Each triggers mandatory referral language and session handling

**Regulatory Compliance:**
- Mental Healthcare Act 2017 (India) — defines scope of what an AI tool can/cannot do
- DPDP Act 2023 (India) — data protection, consent, data minimization
- No diagnostic claims — only "presenting concerns" and "recommended assessments"
- All data handling must specify: what's collected, why, how long it's stored, who sees it

---

## File Naming Convention (For Build Sessions)

```
Session outputs should follow this pattern:

Handoff docs:     session_01_handoff.md, session_02_handoff.md, ...
Domain JSONs:     behavioral_development.json, emotional_development.json, ...
Architecture:     state_schema.py, node_definitions.md, graph_edges.md, ...
Prompts:          nlu_intent_scan.md, nlg_conversational.md, ...
Code:             graph.py, state.py, nodes/*.py, ...
Tests:            test_edge_cases.py, test_tag_accuracy.py, ...
Documentation:    technical_overview.md, clinical_protocol.md, ...
```

---

## Progress Tracker

(Utkarsh should update this as sessions are completed)

| Session | Status | Artifacts Produced | Notes |
|---------|--------|-------------------|-------|
| 1 | Not started | — | — |
| 2 | Not started | — | — |
| 3 | Not started | — | — |
| 4 | Not started | — | — |
| 5 | Not started | — | — |
| 6 | Not started | — | — |
| 7 | Not started | — | — |
| 8 | Not started | — | — |
| 9 | Not started | — | — |
| 10 | Not started | — | — |
