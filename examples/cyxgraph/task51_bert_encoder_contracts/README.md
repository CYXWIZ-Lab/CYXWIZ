# Task 51 - BERT Encoder Contracts

This folder groups CyxGraph examples for `done51`.

## Files

- `bert_sequence_classifier_contract.cyxgraph` - explicit BERT-style sequence classifier path: token IDs -> embedding -> positional encoding -> TransformerEncoder -> CLS select -> classifier.
- `bert_token_classifier_contract.cyxgraph` - explicit BERT-style token classifier path: token IDs -> embedding -> positional encoding -> TransformerEncoder -> TimeDistributed token head.

## Scope Reminder

Task 51 covers tested BERT-style encoder graph contracts, metadata, export/import, packaged text inference contract, and fail-closed unsupported features. These examples do not claim pretrained BERT checkpoint compatibility, HuggingFace loading, token_type/segment IDs, or GPU transformer kernels.
