# AGENTS.md

## Repository mission

This repository implements a unified, compute-aware, three-agent framework for CrossVid-style multi-video understanding.

The core research hypothesis is:

> Multi-video understanding should be solved through two nested coarse-to-fine processes:
>
> 1. Within each video: event-level scanning followed by local micro-clip focusing.
> 2. Across videos: coarse relation alignment followed by fine-grained visual verification.

The implementation must remain task-agnostic. It must not route different CrossVid task categories into separate pipelines or task-specific solvers.

## Non-negotiable architecture constraints

The system may contain exactly three concrete logical agents:

1. `ScoutAgent`

   * Performs single-video `SCAN`.
   * Performs single-video `FOCUS`.
   * Reads raw video frames or micro-clips.

2. `MasterAgent`

   * Performs evidence-based planning.
   * Allocates the remaining visual budget.
   * Maintains hypotheses and unresolved evidence needs.
   * Produces the final answer.
   * Must not directly receive raw video frames during normal execution.

3. `ComparatorAgent`

   * Performs coarse cross-video `ALIGN`.
   * Performs fine-grained visual `VERIFY`.
   * May read raw visual inputs only during `VERIFY`.

Do not introduce additional logical agents.

The following are tools or modules, not agents:

* video segmenter;
* frame sampler;
* feature extractor;
* feature cache;
* evidence store;
* token budget manager;
* correspondence matcher;
* relation scorer;
* identity clusterer;
* answer formatter;
* dataset adapter;
* evaluation runner.

Do not name these modules with an `Agent` suffix.

## Unified processing requirement

All CrossVid samples must use the same:

* three agents;
* shared evidence-state schema;
* action protocol;
* coarse-to-fine loop;
* correspondence representation;
* budget manager;
* stopping mechanism.

Do not add code branches such as:

```python
if task_type == "FSA":
    ...
elif task_type == "MOC":
    ...
```

Task metadata may be logged or used for evaluation grouping, but it must not select a different reasoning pipeline.

## Required action protocol

The only high-level reasoning operations are:

* `SCAN`
* `PLAN`
* `FOCUS`
* `ALIGN`
* `VERIFY`
* `ANSWER`
* `STOP`

Required ownership:

* `ScoutAgent`: `SCAN`, `FOCUS`
* `MasterAgent`: `PLAN`, `ANSWER`, `STOP`
* `ComparatorAgent`: `ALIGN`, `VERIFY`

## Provenance invariants

Every visual evidence unit must permanently retain:

* `video_id`;
* optional `view_id`;
* `time_span`;
* `frame_indices`;
* feature-cache references;
* confidence;
* extraction level.

Never merge evidence originating from different videos into one anonymous content token.

Cross-video similarity must be represented as a relation edge between source-preserving evidence nodes.

Semantic summaries may be compressed. Provenance fields may not be removed or rewritten.

## Visual-budget invariants

The system must enforce a hard visual budget.

Track at least:

* sampled frames;
* estimated visual tokens;
* actual visual tokens when available;
* visual model calls;
* reasoning rounds;
* per-video warm-start coverage;
* reserved verification budget.

The warm-start stage must not consume the entire budget.

Every video must receive configurable minimum warm-start coverage before question-focused allocation can dominate.

An action that exceeds the remaining budget must be rejected or reduced explicitly. Never silently exceed the configured budget.

## Coarse-to-fine invariants

Warm start produces event-level `VideoSketch` objects, not final answers.

Fine evidence should normally be represented as a key micro-clip rather than a single isolated frame.

A micro-clip should preserve enough temporal information to distinguish:

* before and after states;
* insertion versus removal;
* movement direction;
* event order;
* object continuity;
* multi-view overlap.

Cross-video fine verification must operate only on a small candidate set selected during coarse alignment.

Do not perform exhaustive all-pairs raw-frame comparison by default.

## Model-sharing requirement

Logical multi-agent execution must not require three independently loaded foundation models.

Use one shared model backend or one shared backbone wherever possible.

Agent behavior should be controlled through:

* role-specific prompts;
* role tokens;
* optional lightweight adapters;
* structured-output schemas.

The model registry must prevent accidental duplicate loading of the same large model.

## Engineering requirements

* Inspect the repository before adding a parallel architecture.
* Preserve existing conventions unless they conflict with this file.
* Prefer small, testable modules.
* Use typed Python.
* Validate structured agent outputs.
* Fail loudly on invalid schemas.
* Do not silently return empty evidence.
* Do not hide exceptions behind broad `except Exception`.
* Avoid unnecessary production dependencies.
* Do not leave `TODO`, `pass`, or placeholder exceptions in the core inference path.
* A deterministic mock backend is required for CPU-only tests.
* The real VLM backend must be replaceable through an interface.
* Do not claim a test passed unless it was actually executed.

## Testing requirements

Tests must cover:

* budget accounting;
* per-video minimum warm-start coverage;
* provenance retention;
* structured-output validation;
* event-to-micro-clip focusing;
* top-k correspondence candidate generation;
* temporal relation prediction;
* multi-view duplicate clustering;
* rejection of over-budget actions;
* maximum-round stopping;
* deterministic end-to-end execution with the mock backend;
* absence of task-specific routing;
* exactly three concrete logical agent classes.

After meaningful changes, run the repository’s formatting, linting, type-checking, and test commands where available.

At minimum, run:

```bash
pytest -q
```

## Completion standard

A feature is not complete merely because classes and interfaces exist.

It is complete only when:

* the normal inference path is executable;
* the mock integration test passes;
* errors are actionable;
* configuration is documented;
* CLI examples are present;
* tests cover the core invariants;
* implementation and documentation agree.
