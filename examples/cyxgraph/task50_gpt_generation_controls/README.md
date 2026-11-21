# Task 50 - GPT-Style Generation Controls

This folder groups the CyxGraph example for `done50`.

## Files

- `causal_lm_decoder_generation_experiment.cyxgraph` - a small decoder-only
  causal language model for controlled generation experiments.
- `causal_lm_tiny_corpus.csv` - ten short sentences (`document_id`, `split`,
  `text`) the graph trains on.

## The graph

Tiny text corpus (Data Input, text) -> Word tokenizer (Text Tokenizer, Output
**Causal document windows**, 8 tokens; the vocabulary is fitted on the train
rows) -> Split 80/10/10 -> Causal LM loader -> Token embedding 64x32 ->
Position encoding -> two Transformer decoder blocks -> Token logits 64 per
position (TimeDistributed) -> Causal LM output -> Next-token cross entropy
(with the loader's labels) -> Adam.

Open it in any project (its data path is relative to the repository; the
Engine finds it from the graph's folder), then Compile and Train. The corpus is far too small to learn real language: it
shows the pieces work.

## Scope Reminder

Task 50 covers prompt-generation controls and metadata around causal-LM runs. It does not claim full GPT training parity or pretrained checkpoint compatibility.
