# Event-Level Observer: coarse-to-fine inference experiment

## Motivation

The normal Observer prompt asks every initial video chunk for a full timeline,
all objects, actions, transformations, input/output states, visual details and
uncertainties.  For long CC/NC videos, this makes one 16-frame VLM call produce
roughly 1.1k–1.5k tokens.  The resulting report is slow to generate and forces
FORM to aggregate much irrelevant detail.

## Phase 1: Event Observer

Each initial VLM chunk receives the question context but does **not** answer
the question.  It returns a small event-level evidence packet:

- answer-discriminative event/tool/state plus approximate time;
- start/end state only when relevant;
- an explicit unresolved visual question only when later Focus is needed.

It must omit background objects, people descriptions and generic
cooking/action narration.  It does **not** apply a bullet or token quota to
answer-relevant events: every such event remains a separate timestamped entry.
This preserves a temporal index for later Focus while removing verbose scene
captioning and repeated prose.

## Phase 2: unchanged FORM

The existing LLM FORM receives the chronologically ordered compact reports,
builds the usual Decision State, and either selects an answer or requests a
visual witness.

## Phase 3: unchanged detailed Focus and REVIEW

This experiment changes **only** initial observation.  A `single_video`,
`multi_video_independent`, or `joint_compare` ambiguity still invokes the
existing detailed Focus/Comparer prompt and the normal REVIEW update.  Thus
fine-grained evidence is paid for only when it can resolve a declared
ambiguity.

## What to measure

For a fixed model, QA root and sample ids, compare against the detailed
Observer protocol:

1. accuracy / FSA mean IoU;
2. Observer output tokens and elapsed time;
3. initial visual-call count and Focus-call count;
4. FORM routing validity: `NEED_EVIDENCE` must produce at least one executable
   witness, rather than `not_observable` for a visual question;
5. trace-level error categories (missing event, wrong visual fact, wrong
   routing, wrong final aggregation).

The first controlled test should use the same sampled frames and resolution as
the detailed protocol.  Only after that comparison should sampling density or
resolution be changed.
