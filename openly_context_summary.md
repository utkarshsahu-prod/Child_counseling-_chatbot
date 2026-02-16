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
| `Domain_tree_UPDATED.xlsx` | 10-sheet Excel with all domain trees (triggers, questions, tags, red flags) | Sessions 1, 2 |
| `Cross_Domain_Logic_UPDATED.xlsx` | 5-sheet matrix: Convergence, Confounds, Severity, Differential, Red Flags | Session 2 |
| `NoteBookLM output.docx` | Architecture discussion: 4-layer design, LangGraph mapping, synthetic personas | Session 3 |
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

---

## MVP Requirement Decisions (Current)

The following decisions were confirmed during requirement gathering for the first MVP scope:

- **Primary users:** Parents/caregivers only
- **Target child age range:** 1 to 7 years
- **Language support:** English only
- **Scope:** Developmental concerns only (exclude physical health concerns for v1)
- **Output mode:** Risk summary + generic referral recommendation (no home support plan yet)
- **Referral style:** Keep recommendations generic for now (no prioritized first specialist ordering in v1)
- **PII handling:** May ask for identity/contact details in-chat, but do not store in MVP
- **Reasoning visibility:** Include step-by-step reasoning trace with variables used and value updates at each step, shown in a **developer JSON block** format (preferred for MVP speed)
- **Risk/severity levels:** Use **4 levels** in v1
- **Confidence indicator:** Skip confidence score/label in MVP outputs for now
- **Final summary disclaimer:** Not required for the time being in every final response
- **Question flow limit:** No fixed cap in MVP; continue follow-up questions until domain logic determines the flow can stop
- **Questioning strategy:** Begin by asking for the parent/caregiver's biggest issue to identify the initial presenting concern
- **Adaptive domain traversal:** Ask presenting-concern-specific questions to generate AI tags; when new AI tags are discovered, re-check and trigger matching presenting-concern domains dynamically
- **Stop condition for questioning:** Continue this loop until all relevant presenting-concern question paths are exhausted
- **Multi-concern triage priority:** Process concerns in this order: (1) parent's primary concern first, then (2) DSM red-flag presenting concerns with clinical-disorder relevance, then (3) non-clinical presenting concerns
- **Primary concern deep-dive fields:** For the parent's primary concern, always capture: **Frequency**, **Intensity**, **Current methods used by parents**, **Where it happens** (contexts), and **Life impact**

### MVP Data Points Checklist (Implementation Readiness)

| Category | Data Point | Current Status | Source in This Doc |
|---|---|---|---|
| User Scope | Primary user type (parents/caregivers only) | Defined | MVP decisions |
| User Scope | Child age range (1–7 years) | Defined | MVP decisions |
| Product Scope | Developmental-only concerns in v1 | Defined | MVP decisions |
| Product Scope | Output contract (risk summary + generic referral) | Defined | MVP decisions |
| Product Scope | No home support plan in v1 | Defined | MVP decisions |
| Privacy | Ask PII in chat but do not store | Defined | MVP decisions |
| Explainability | JSON reasoning trace with variable updates | Defined | MVP decisions |
| Risk Model | 4-level severity scale | Defined | MVP decisions + risk levels section |
| Orchestration | No fixed question cap | Defined | MVP decisions |
| Orchestration | Start from biggest issue → presenting concern | Defined | MVP decisions |
| Orchestration | Dynamic AI-tag-triggered domain traversal | Defined | MVP decisions |
| Orchestration | Primary concern deep-dive data capture (frequency/intensity/current methods/context/life impact) | Defined | MVP decisions |
| Orchestration | Stop when relevant presenting-concern paths are exhausted | Defined | MVP decisions |
| Safety | Crisis triggers and immediate safety protocol | Defined | Safety & Compliance section |
| Compliance | India MHCA/DPDP constraints and non-diagnostic positioning | Defined | Safety & Compliance section |
| Data Assets | Domain tree/cross-domain source file naming consistency with repo files | Defined (aligned to `Domain_tree_UPDATED.xlsx` and `Cross_Domain_Logic_UPDATED.xlsx`) | Existing Artifacts section |
| Triage Policy | Priority rule when multiple presenting concerns are active simultaneously | Defined (primary concern → DSM red-flag/clinical → non-clinical) | MVP decisions |

### Risk/Severity Levels for v1

1. **Low**
2. **Mild concern**
3. **Moderate**
4. **High**


---

## MVP Execution To-Do List (Next Steps)

1. **Create state schema (`state_schema.py`)**
   - Define conversation state fields: session metadata, active domain queue, AI tags, severity level, and reasoning trace payload.
2. **Define JSON reasoning trace contract**
   - Standardize per-turn trace keys (`variables_used`, `value_updates`, `routing_decision`, `next_question_reason`).
3. **Implement intake node for primary concern**
   - Capture required deep-dive fields for primary concern: Frequency, Intensity, Current methods, Context, and Life impact.
4. **Implement deterministic triage/routing policy**
   - Enforce priority order: primary concern → DSM red-flag/clinical concerns → non-clinical concerns.
5. **Convert domain trees from Excel to runtime JSON**
   - Parse `Domain_tree_UPDATED.xlsx` into domain-specific JSON structures with triggers, questions, and tags.
6. **Implement dynamic discovery + re-routing loop**
   - On each response, rescan for new tags and add matching presenting concerns to queue without losing context.
7. **Implement cross-domain logic checks**
   - Apply convergence/confounds/severity/differential/red-flag rules using `Cross_Domain_Logic_UPDATED.xlsx`.
8. **Build output formatter (MVP contract)**
   - Generate final response as risk summary + generic referral recommendation + JSON reasoning trace.
9. **Add safety and compliance checks**
   - Ensure crisis-trigger handling and non-diagnostic response framing in all terminal summaries.
10. **Create test pack for MVP readiness**
   - Add unit/integration tests for routing order, stop condition, severity mapping, and trace completeness.

### Architecture Risk Review (Major Issues to Anticipate)

1. **Data model instability before schema freeze**
   - Risk: parser, routing, severity, and tests diverge if canonical IDs/schemas are changed mid-implementation.
   - Improvement: freeze canonical schemas/IDs early (`domain_id`, `tag_id`, `rule_id`, `severity_rule_id`) with versioning.
2. **Clinical-rule ambiguity (DSM red-flag vs clinical vs non-clinical)**
   - Risk: triage execution differs across engineers/prompts if policy is not machine-readable.
   - Improvement: define policy tables for triage class and escalation policy with deterministic precedence.
3. **PII leakage despite “do not store” requirement**
   - Risk: PII can leak into traces/logs/analytics/error payloads even when app DB storage is off.
   - Improvement: implement PII scrubbing middleware, trace redaction, and no-PII logging contracts.
4. **State drift across dynamic re-routing**
   - Risk: newly discovered tags can re-open completed branches and create repeated/conflicting paths.
   - Improvement: enforce branch lifecycle states (`queued`, `active`, `completed`, `blocked`) and idempotent routing keys.
5. **Determinism drift in NLU/NLG contracts**
   - Risk: model/prompt updates alter extracted tags and route outcomes.
   - Improvement: strict typed output schemas (enums/bounds), confidence gates, and fallback handling for invalid outputs.
6. **Traceability mismatch (trace vs logic output)**
   - Risk: JSON reasoning trace may not match actual deterministic routing decisions.
   - Improvement: generate trace directly from logic-node outputs; NLG should consume trace, not invent explanations.
7. **Excel-to-JSON schema inconsistency**
   - Risk: malformed/missing spreadsheet fields silently break runtime behavior.
   - Improvement: strict conversion validation + fail-fast parser checks + schema compatibility tests.
8. **Termination and loop-control weakness**
   - Risk: “ask until exhausted” can cause long loops with no new information.
   - Improvement: define stop heuristics (max branch depth, max revisit count, no-new-info counter).
9. **Severity calibration ambiguity across domains**
   - Risk: conflicting escalation rules can under/over-escalate severity.
   - Improvement: define deterministic precedence/tie-breakers and calibration test vectors for the 4-level scale.
10. **Operational observability and graph-level QA gaps**
   - Risk: hard to debug tuning failures without replay, trace dashboards, and traversal snapshots.
   - Improvement: add run-level observability (trace IDs, node timing, branch transitions) and scenario-based graph tests.

### Improved MVP Execution To-Do List (Architecture-First)

0. **Freeze canonical data contracts before coding nodes**
   - Finalize and version: `state_schema.py`, `trace_schema.json`, `domain_tree.schema.json`, `cross_domain.schema.json`, and canonical IDs (`tag_id`, `rule_id`, `domain_id`).
1. **Create machine-readable policy tables for triage and clinical rules**
   - Add explicit DSM red-flag, clinical, and non-clinical classification tables with deterministic precedence/tie-breakers.
2. **Build secure ingestion + validation pipeline (Excel → JSON)**
   - Convert source Excel files with strict validators, schema checks, and fail-fast errors for missing/invalid fields.
3. **Implement PII safety controls at platform boundaries**
   - Add redaction for logs/traces/errors, runtime PII masking, and verify “ask but do not store” enforcement.
4. **Implement deterministic orchestration core with lifecycle states**
   - Queue manager with `queued/active/completed/blocked`, dedupe keys, revisit limits, and branch completion rules.
5. **Implement primary-concern intake node with typed completeness checks**
   - Capture Frequency, Intensity, Current methods, Context, and Life impact in required structured fields.
6. **Add per-turn safety guardrail before normal routing**
   - Run red-flag checks first on every user message; short-circuit to escalation/referral when needed.
7. **Implement NLU normalization + tag ontology mapping**
   - Canonicalize aliases/synonyms; reject/repair invalid model outputs before they reach routing/severity logic.
8. **Implement cross-domain reasoning + severity engine**
   - Centralize convergence/confounds/differential/escalation logic and deterministic 4-level severity resolution.
9. **Bind reasoning trace to node execution artifacts**
   - Emit `variables_used`, `value_updates`, `routing_decision`, and `next_question_reason` from logic outputs only.
10. **Add loop-control termination heuristics**
   - Enforce max depth/revisit/no-new-info thresholds while preserving the domain-driven stop condition.
11. **Establish observability and graph-level evaluation harness**
   - Add run replay, traversal snapshots, node latency metrics, scenario tests (synthetic personas), and acceptance gates.
12. **Run readiness review and freeze MVP interfaces**
   - Lock contracts, policy tables, and release test criteria before scaling implementation.

### Pending Decisions Check

- **Blocking decisions:** None identified from current MVP checklist.
- **Non-blocking decisions to revisit later:**
  - Whether to add confidence indicator in outputs.
  - Whether to include recurring disclaimer in every final summary.
  - Whether to add home support plans after referral output stabilizes.

### Execution Progress Update (Current)

- Added graph-level run observability primitives in code:
  - traversal snapshots per turn,
  - per-turn latency capture,
  - replayable scenario runner (`run_scenario`) for synthetic test personas.
- Added automated test coverage for observability outputs so graph behavior can be audited during tuning.
- These updates advance Improved To-Do item **11 (observability and graph-level evaluation harness)**.
- Implemented a typed primary-concern intake capture in code for required dimensions: Frequency, Intensity, Current methods, Where happening, and Life impact, with completeness checks and state-level reasoning updates.
- Implemented NLU tag normalization + ontology mapping to canonicalize aliases and reject invalid tags before routing/severity logic, with normalization events logged in state updates for auditability.
- Implemented deterministic cross-domain rule evaluation (convergence/confound/differential/escalation) with rule-priority ordering, suggested-domain triggers, and minimum-severity escalation floors integrated into orchestration.
- Added an explicit contract-freeze module (`contracts.py`) with versioned interface constants and guard tests to lock MVP schema/state enums before further scaling.
- Hardened ingestion contract enforcement with duplicate-header detection, required-non-empty column validation, non-empty record guarantees, and ingestion reports to support fail-fast Excel→JSON conversion controls.
- Added an automated readiness review gate (`run_readiness_review`) to enforce contract freeze, runtime asset load health, and minimum test-threshold checks before MVP release sign-off.
