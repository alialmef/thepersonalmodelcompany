# Frontier Personal Model Plan

Status: working architecture plan  
Date: 2026-05-22  
First user: Ali

## Thesis

The product is not a generic assistant with memory. It is a local personal model
factory that turns a person's laptop into a private training, memory,
verification, and action loop.

The user wants a model that sounds like them, thinks like them, acts like them,
and can prove it before it touches important surfaces. Full Disk Access and MCP
are distribution advantages, but the moat is the private correction loop:

1. Observe the user's real writing, decisions, memories, and tool behavior.
2. Train an owned adapter and maintain an owned memory substrate.
3. Test whether outputs are recognizably the user.
4. Capture every approval, edit, rejection, undo, and correction.
5. Convert those corrections into SFT, DPO, action-policy, and memory data.
6. Retrain and re-evaluate until the trust report says the model is ready.

The artifact we are building is therefore not just a chatbot. It is a local,
auditable identity-training system.

## May 2026 Research Anchors

These are the current signals I am using as the planning baseline. Older work
such as MemGPT, MemoryBank, Generative Agents, Reflexion, Toolformer, and DPO
still matters, but only as background.

- Response-Aware User Memory Selection for LLM Personalization, Microsoft
  Research, April 2026:
  https://www.microsoft.com/en-us/research/publication/response-aware-user-memory-selection-for-llm-personalization/
  RUMS argues that memory selection should optimize response utility, not just
  semantic similarity. PMC should eventually score memories by expected output
  influence.

- PlugMem, Microsoft Research, February 2026:
  https://www.microsoft.com/en-us/research/publication/plugmem-a-task-agnostic-plugin-memory-module-for-llm-agents/
  PlugMem pushes memory toward compact propositional and prescriptive knowledge
  graphs instead of raw trajectory stuffing. PMC should preserve raw episodes
  while promoting durable rules and preferences into a compact graph.

- AlpsBench, March/May 2026:
  https://arxiv.org/abs/2603.26680
  The core warning is that recall does not guarantee preference alignment or
  emotional resonance. PMC must have private identity evals, not only retrieval
  metrics.

- MemMachine, April 2026:
  https://arxiv.org/abs/2604.04853
  The most relevant idea is ground-truth-preserving memory: keep full episodes
  and use adaptive retrieval instead of lossy summarization as the source of
  truth.

- MCP authorization and apps specs, current public protocol surface:
  https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
  https://apps.extensions.modelcontextprotocol.io/
  PMC should treat MCP as an audited tool/action substrate with scoped consent,
  not as unrestricted automation.

- MemPalace, May 2026:
  https://github.com/MemPalace/mempalace
  Shows the market is converging on local-first MCP memory with verbatim storage,
  knowledge graphs, and many tools. PMC should not compete as "another memory
  store"; it should own training and verification.

- LycheeMemory, May 2026:
  https://github.com/LycheeMem/LycheeMem
  Shows consolidation, adaptive retrieval, multimodal memory, and agent runtime
  integrations are table stakes.

- SimpleMem, 2026:
  https://github.com/aiming-lab/SimpleMem
  Shows cross-session and multimodal memory are becoming expected. PMC's
  differentiator must be local identity training, not just memory access.

## Product Moat

Memory stores are becoming cheap. MCP servers are becoming cheap. The durable
moat is a closed-loop personal training dataset that nobody else can recreate:

- private voice data from real messages and documents
- private decision data from how the user edits model outputs
- private action data from approval, rejection, undo, and correction traces
- private memory truth data with provenance and contradiction history
- private evals built from held-out user data
- owned adapters, eval reports, memory snapshots, and training manifests

The repo should make these artifacts first-class. Every training run should
answer:

- What did we train on?
- What did we hold out?
- What did the model get wrong about the user?
- What changed since the last run?
- Which memories influenced generation?
- Which actions were proposed, edited, approved, rejected, or undone?
- Is the model ready for chat, sandboxed actions, or supervised autonomy?

## Architecture

The target architecture is a loop:

```text
local substrate
  -> raw episodes
  -> curated completions
  -> personal memory graph
  -> adapter training
  -> private verification probes
  -> user judgments
  -> preference/action datasets
  -> retraining and promotion
```

The repo should keep strict boundaries:

- Ingestion owns raw evidence from local surfaces.
- Curation owns trainable examples and held-out examples.
- Memory owns durable recall, profile facts, episodes, contradictions, and
  action memory.
- Training owns SFT/DPO/adapters and manifests.
- Verification owns probes, user judgments, trust reports, and readiness gates.
- Action sandbox owns MCP/tool proposals, approvals, edits, rejection, undo, and
  replayable traces.
- Serving owns retrieval injection and active adapter routing.

## Key Decisions

1. Weights learn identity. Memory stores mutable facts.

   The adapter should absorb voice, cadence, preferences, decision tendencies,
   and habitual patterns. The memory system should preserve changing facts,
   recent context, names, projects, obligations, and evidence.

2. Raw episodes remain ground truth.

   Summaries, facts, graph nodes, and profiles are derived views. They must keep
   provenance links back to raw evidence.

3. Every correction is data.

   A rejected reply is a negative preference. An edited reply is an SFT target
   and a DPO chosen response. An approved tool action is action-policy data. An
   undone action is high-value negative supervision.

4. Verification precedes autonomy.

   A model can chat before it can act. It can draft before it can send. It can
   propose before it can execute. Readiness is earned through private evals.

5. Retrieval must become response-aware.

   Basic vector search is a baseline. The frontier version ranks memories by
   expected utility for the response, contradiction risk, recency, provenance,
   and the user's judgment history.

6. MCP is an action bus, not a trust model.

   Tool calls require scoped capability labels, dry-run previews, approvals,
   audit logs, and undo paths where possible.

7. The bundle is the product artifact.

   A serious run produces dataset manifests, holdout IDs, adapter files,
   verification probes, judgments, trust reports, memory snapshots, and action
   traces.

## Build Phases

### Phase 1: Verification Spine

Goal: make "does this sound/think/act like me?" an executable loop.

Build:

- `PersonalProbe`: private benchmark item from held-out completions, decisions,
  or actions.
- `ProbeCandidate`: model, base model, real user, or edited candidate.
- `UserJudgment`: approve, reject, choose, edit, not-me, wrong, private.
- `ActionTrace`: proposed action plus approval/edit/rejection/undo.
- `TrustReport`: readiness summary.
- Converters from judgments/traces into SFT and DPO-ready completions.
- API endpoints for eval prompts, judgments, action traces, and reports.

Success bar:

- Eval screen is no longer static.
- User judgments are stored as durable training artifacts.
- Trust report can block promotion when identity confidence is low.

### Phase 2: Personal Eval Gauntlet

Goal: produce private evals from the user's own history.

Build:

- held-out reply reconstruction probes
- pairwise identity choice probes
- memory factuality probes
- privacy probes
- action-choice probes
- judge prompts and human-in-the-loop review queues
- metrics by dimension: voice, factuality, decision, privacy, action

Success bar:

- Every run has a private eval report with pass/fail thresholds.
- Missing identity evals cannot silently pass.

### Phase 3: Memory OS

Goal: evolve memory from recall snippets into a durable personal substrate.

Build:

- raw episode ledger
- semantic memory
- episodic memory
- profile memory
- working memory
- narrative memory
- action memory
- contradiction and decay records
- response-aware retrieval scoring

Success bar:

- Retrieved memory is explainable and provenance-linked.
- The system can say why a memory was injected.
- Corrections update memory and training data.

### Phase 4: Action Sandbox

Goal: turn MCP/local app access into supervised action learning.

Build:

- tool capability registry
- dry-run planner
- proposal UI
- approval/edit/reject/undo trace capture
- risk labels
- replay tests
- surface-specific policies for Messages, Mail, Calendar, Files, Browser, Notes,
  GitHub, and shell

Success bar:

- The model learns not just what the user says, but what the user would do.
- No high-risk action executes without explicit readiness and consent.

### Phase 5: Preference Training Loop

Goal: make the user's edits improve the next model.

Build:

- judgment-to-DPO pipeline
- edit-to-SFT pipeline
- action-trace-to-policy pipeline
- negative example mining
- run-to-run drift tracking
- adapter promotion gates

Success bar:

- The second trained model is measurably more like the user than the first.

### Phase 6: Owned Artifact v2

Goal: make every model portable, auditable, and reproducible.

Build:

- bundle manifest v2
- dataset hashes
- memory snapshot hashes
- probe set hashes
- judgment ledger hash
- trust report
- action trace ledger
- rollback metadata

Success bar:

- The user owns the model and can inspect what made it.

## First Vertical Slice

The first slice should ship the verification spine:

1. Add verification schemas and storage.
2. Generate probes from held-out completions and fallback scenarios.
3. Replace loose eval JSONL with structured judgments.
4. Expose a trust report endpoint.
5. Convert judgments/traces into trainable completions.
6. Add tests that prove the loop persists data and emits SFT/DPO examples.

This does not finish the product, but it changes the repo's center of gravity.
After this slice, evaluation is not a screen. It is the beginning of the private
training flywheel.

