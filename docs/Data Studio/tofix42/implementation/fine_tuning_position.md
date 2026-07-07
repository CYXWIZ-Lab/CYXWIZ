# Fine-Tuning Position

## Purpose

Clarify what fine-tuning is for in the CyxWiz assistant plan, what it cannot
fix, and when it should happen relative to implementation work.

## Short Answer

Fine-tuning is optional.

It is not required to:

- prove the local RAG path
- design the plugin boundary
- design the knowledge pack
- build the first Assistant panel

Fine-tuning is only a later quality-improvement step.

## What Fine-Tuning Can Help

If done carefully, fine-tuning can improve:

- four-section answer compliance
- citation-path copying
- concise grounded wording
- qualification behavior such as `Unknowns` and
  `Unsupported or not implemented`
- consistency on repeated CyxWiz-style retrieval-backed prompts

## What Fine-Tuning Cannot Fix

Fine-tuning will not fix:

- bad retrieval
- missing documentation
- wrong top citations
- weak chunking
- poor corpus coverage
- ambiguous user intent
- broad product-help questions with no good evidence target

This is the key rule:

```text
retrieval problems must be solved in retrieval
fine-tuning problems must be solved in the answer layer
```

## Why This Matters

The broad-question failure we observed is a retrieval and product-help problem,
not a fine-tuning-first problem.

Example:

Question:

```text
what can u assist me with in cyxwiz engine?
```

Observed problem:

- retrieval found `EngineConfig::Save`
- model answered from that unrelated evidence

Fine-tuning will not reliably fix that by itself.

The real fixes are:

1. add better capability/help documents to the indexed corpus
2. define question-routing rules
3. detect broad capability/help intent
4. improve semantic relevance checks

## Correct Order

The implementation order should be:

1. define plugin boundary
2. define knowledge pack
3. define backend runtime contract
4. define Assistant panel and Command Window routing rules
5. implement retrieval-backed product surface
6. observe real usage and failure patterns
7. only then run a controlled fine-tuning experiment if still justified

Do not invert this order.

## When Fine-Tuning Becomes Justified

Fine-tuning becomes worth doing when all of these are true:

- retrieval quality is already acceptable
- knowledge-pack content is stable enough
- prompt contract is stable enough
- UI/backend contract is stable enough
- failures are mostly answer-formatting or grounded-answer consistency failures

If those are not true, fine-tuning is premature.

## Best Current Role of Fine-Tuning

For the current CyxWiz assistant, fine-tuning should be positioned as:

- a formatting and grounded-answer reliability experiment
- not a replacement for retrieval quality
- not a substitute for help documentation
- not a prerequisite for first plugin integration

## Safe Fine-Tuning Targets

The safest first fine-tuning targets remain:

- section-format compliance
- citation inclusion
- concise answer discipline
- honest `Unknowns`
- refusal to invent unsupported behavior

Do not target:

- broader product marketing answers
- generic tutorial generation
- open-ended design advice
- uncited procedural synthesis

## Relation to Implementation

Before implementation, the correct work is:

- make retrieval packaging real
- make the backend contract real
- make the first UI surface real

After that, compare:

- baseline backend behavior
- post-integration failure patterns

Only then decide whether tuning is still worth the complexity.

## Recommendation

Use this policy:

```text
Implement the first retrieval-backed assistant surface first.
Treat fine-tuning as an optional later optimization for answer discipline,
not as a prerequisite for shipping the first assistant integration.
```

That keeps the project grounded in the part that actually carries truth:
retrieval.
