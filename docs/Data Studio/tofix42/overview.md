# tofix42 RAG Overview

## Purpose

This document explains the high-level flow of the current local RAG system used
by tofix42.

The design is intentionally simple:

- retrieval is the truth layer
- the model is the answer synthesizer
- the proxy is only a transport adapter
- validation checks that the model follows the required answer contract

## Core Idea

The system does not ask the model to answer from memory about CyxWiz.

It first retrieves relevant local source, docs, and graph evidence from the
repository. That evidence is packed into a strict prompt. The local model then
answers only from that evidence.

## Main Components

### 1. Index and Retrieval

File:

- [phase1a_retrieval.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase1a_retrieval.py>)

Responsibilities:

- scan selected local files
- chunk source, markdown, and graph content
- build a local JSON index
- run lexical search for a user query
- return an answer packet with top evidence and citations

This stage decides what evidence is relevant.

### 2. Prompt and Answer Adapter

File:

- [phase1b_answer.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase1b_answer.py>)

Responsibilities:

- take the retrieval answer packet
- build the strict evidence-grounded prompt
- call a configured local runtime endpoint
- parse the returned text into:
  - `Answer`
  - `Evidence`
  - `Unknowns`
  - `Unsupported or not implemented`

This stage defines the answer contract.

### 3. Real Local Model Server

Runtime:

- `llama-server` on `http://127.0.0.1:1235/v1/chat/completions`

Model:

- `D:\tmp\models\qwen2.5-coder-3b-instruct-q4_k_m.gguf`

Responsibilities:

- load the local GGUF model
- run local inference
- expose an OpenAI-compatible chat endpoint

This is the actual model runtime.

### 4. OpenAI-Compatible Proxy

File:

- [phase2_openai_compat_proxy.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase2_openai_compat_proxy.py>)

Live endpoint:

- `http://127.0.0.1:8768/completion`

Responsibilities:

- accept the simpler tofix42 `/completion` request format
- adapt the Phase 1B prompt for chat-model use
- translate the request into OpenAI chat format
- forward it to the real `llama-server`
- return the model text in the shape expected by the rest of the harness

The proxy is not retrieval and not inference logic. It is a bridge.

### 5. Validation

Files:

- [phase2_probe_suite.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase2_probe_suite.py>)
- [phase2_real_model_check.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase2_real_model_check.py>)

Responsibilities:

- run fixed probe questions
- verify expected evidence paths
- verify required answer sections
- reject stub-only proof
- record whether real localhost validation passed

## Current Accepted Runtime Path

The accepted real localhost path is:

- real model server: `127.0.0.1:1235`
- proxy: `127.0.0.1:8768`

The accepted result is:

- `real_model_probe_accepted = true`

## End-to-End Flow

1. User or harness asks a question.
2. Retrieval searches the local index.
3. Retrieval returns top evidence with citations.
4. Phase 1B builds a strict prompt from that evidence.
5. The prompt is sent to the proxy at `8768`.
6. The proxy rewrites the request into OpenAI chat format.
7. The proxy forwards the request to Qwen on `1235`.
8. Qwen returns answer text.
9. The proxy returns that text to the Phase 1B adapter.
10. Phase 1B parses the structured sections.
11. Validation scripts confirm the output meets the contract.

## ASCII Diagram

```text
                       LOCAL CYXWIZ RAG FLOW

  user question
       |
       v
  +------------------------+
  | phase1a_retrieval.py   |
  | local index + search   |
  +------------------------+
       |
       | answer packet
       | question + top evidence + citations
       v
  +------------------------+
  | phase1b_answer.py      |
  | strict prompt builder  |
  | + section parser       |
  +------------------------+
       |
       | POST /completion
       | prompt, n_predict, stream
       v
  +-------------------------------+
  | phase2_openai_compat_proxy.py |
  | 127.0.0.1:8768                |
  | transport + prompt adapter    |
  +-------------------------------+
       |
       | POST /v1/chat/completions
       | system + user chat messages
       v
  +-------------------------------+
  | llama-server                  |
  | 127.0.0.1:1235                |
  | Qwen2.5-Coder-3B-Instruct     |
  +-------------------------------+
       |
       | model text
       v
  +------------------------+
  | phase1b_answer.py      |
  | parse sections:        |
  | Answer                 |
  | Evidence               |
  | Unknowns               |
  | Unsupported...         |
  +------------------------+
       |
       v
  final structured answer
```

## Request Shape

Input accepted by the proxy:

```json
{
  "prompt": "<strict evidence-grounded prompt>",
  "n_predict": 384,
  "stream": false
}
```

Request sent from the proxy to `llama-server`:

```json
{
  "model": "local-model",
  "messages": [
    { "role": "system", "content": "<format contract>" },
    { "role": "user", "content": "<adapted evidence prompt>" }
  ],
  "max_tokens": 384,
  "stream": false,
  "temperature": 0
}
```

## Boundaries

What retrieval does:

- decide relevant evidence
- provide citations

What the model does:

- synthesize an answer from retrieved evidence
- follow the required section format

What the proxy does:

- translate request and response shapes

What validation does:

- prove the local runtime is real
- prove the answer contract is satisfied

## Important Operational Point

If the model is replaced later, the retrieval layer should stay the same.

That is the main architectural point of this RAG setup:

- CyxWiz knowledge lives in the repo and index
- model quality improves answer formatting and synthesis
- model memory is not treated as the source of truth
