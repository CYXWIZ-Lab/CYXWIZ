# To Fix 50 - GPT-Style Generation Controls And UX

## Purpose

Extend generation controls from core runtime proof into a practical GPT-style
prompting workflow.

`tofix28` added reusable generation controls and a Studio panel. This ticket
makes the user-facing generation workflow complete enough for iterative prompt
experiments.

## Scope

- Prompt text input with packaged tokenizer.
- Raw token-ID debugging mode.
- Max new tokens.
- Temperature.
- Top-k.
- Top-p.
- Greedy vs multinomial sampling.
- Seed control.
- EOS stop.
- Include/exclude prompt in output.
- Token-by-token generation progress.
- Candidate distribution diagnostics.

## UX requirements

Studio should show:

- Current prompt length.
- Max context length.
- Remaining generation budget.
- Stop reason: EOS, max tokens, error, or user cancel.
- Generated token IDs.
- Decoded generated text.
- Sampling settings used for the run.
- Optional candidate table for last token.

## Engine requirements

- Generation config validation must run before inference.
- Prompt length must be checked against model/tokenizer max length.
- Sampling must be reproducible with seed.
- Invalid logits shape must fail with a clear reason.
- Invalid probability distribution must fail safely.

## Tests

Add tests for:

- EOS stop reason.
- Max-token stop reason.
- Seed reproducibility.
- Prompt too long.
- Empty prompt handling.
- Invalid temperature/top-k/top-p.
- Candidate distribution reporting.

## Non-goals

- Do not claim full GPT training support.
- Do not add beam search unless a real user need appears.
- Do not add streaming server APIs in this ticket unless Studio needs them.

## Completion criteria

- Studio generation is usable for controlled GPT-style prompt experiments.
- Every generation run has visible settings, output, and stop reason.
- Core tests and Studio-facing smoke tests cover success and failure paths.
