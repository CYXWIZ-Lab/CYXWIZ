---
title: Server node HTTP API
description: The HTTP routes a server node serves for deployed models, the predict request and reply, and the limits of this build.
type: reference
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - settings.md
  - ../ports.md
  - ../security.md
---

A server node serves the models deployed on it (the **Deploy** page of the
GUI, or **Deploy > Deploy to Server Node...** in the Engine) over HTTP on
port 8082 (`--http-port`, or `network.http_api_port`).

> **Experimental, and open.** The HTTP API listens on every network
> interface, answers any caller (no key is checked, whatever **Settings >
> API > Require Authentication** says), allows every web origin, and has no
> TLS. Keep port 8082 closed to other machines unless you mean to serve
> anyone who can reach it. Deployment and serving have not been checked end
> to end for this release.

## Routes

| Method and path | What it does |
| --- | --- |
| `GET /health` | `{"status": "healthy", "server_type": "cyxwiz-server-node", "active_deployments": N}` |
| `GET /v1/models` | The deployed models, in the OpenAI list format |
| `GET /v1/deployments` | The deployments with their state |
| `GET /v1/deployments/:id` | One deployment |
| `POST /v1/predict` | Run a deployed model on a tensor (below) |
| `POST /v1/completions`, `POST /v1/chat/completions`, `POST /v1/embeddings` | OpenAI-shaped routes. They pass the prompt to the deployment as raw bytes and return what it outputs; no tokenizer runs on the node, so a model trained in the Engine does not answer them usefully yet. |

Errors come back as
`{"error": {"message": "...", "type": "...", "code": "..."}}` with an HTTP
status: `missing_field` and `parse_error` (400), `deployment_not_found`
(404), `inference_error` (500), `service_unavailable` when the deployment
manager is not running.

## Predict

```bash
curl -s http://localhost:8082/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"deployment_id": "<id from /v1/deployments>", "input": [[5.1, 3.5, 1.4, 0.2]]}'
```

- `deployment_id` (required): the deployment to run.
- `input` (required): a number array, nested to the model's input shape.

The reply carries `deployment_id`, `latency_ms`, `output` (the values, flat)
and `shape`.

## Which models load

The node has loaders for CyxWiz models (`.cyxmodel`), ONNX, GGUF and, when
it was built with LibTorch, TorchScript. A CyxWiz model that uses
an extension node (a C++ plugin node or a Python node) is not loaded: the
log says "the Server Node cannot run extension nodes yet". Serve such a
model from the Engine (**Deploy > Model Serving**).

## Related

- [Ports](../ports.md): the gRPC deployment (50056) and inference (50057)
  services the Engine uses for the same deployments.
- [Security notes](../security.md)
