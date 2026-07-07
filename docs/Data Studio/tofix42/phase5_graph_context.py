#!/usr/bin/env python3
"""Phase 5 graph-aware help harness for tofix42.

This is deterministic and read-only. It reads a .cyxgraph JSON file, selects a
node, and emits graph/node context, a basic explanation, or a Phase 1B-compatible
retrieval packet. It does not mutate graphs, generate graph drafts, start a
model, or depend on Studio UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval


CONTEXT_SCHEMA = "cyxwiz.tofix42.phase5.graph_node_context.v1"
EXPLANATION_SCHEMA = "cyxwiz.tofix42.phase5.graph_node_explanation.v1"
PATH_CONTEXT_SCHEMA = "cyxwiz.tofix42.phase5.graph_path_context.v1"
PATH_EXPLANATION_SCHEMA = "cyxwiz.tofix42.phase5.graph_path_explanation.v1"
SUGGESTIONS_SCHEMA = "cyxwiz.tofix42.phase5.graph_suggestions.v1"
AUDIT_SCHEMA = "cyxwiz.tofix42.phase5.graph_audit.v1"
DRAFT_PLAN_SCHEMA = "cyxwiz.tofix42.phase5.graph_draft_plan.v1"
PACKET_SCHEMA = "cyxwiz.tofix42.phase1a.answer_packet.v1"
SENSITIVE_KEY_MARKERS = (
    "path",
    "file",
    "dataset",
    "raw",
    "preview",
    "token",
    "password",
    "secret",
    "credential",
)
INSPECTION_NODE_NAME_MARKERS = (
    "data profiler",
    "descriptive stats",
    "value counts",
    "row sampler",
    "correlation matrix",
    "data validator",
)
DATASET_NODE_NAME_MARKERS = ("dataset", "csv", "input", "source")
LOSS_NODE_NAME_MARKERS = ("loss", "crossentropy", "cross entropy", "nll")
OPTIMIZER_NODE_NAME_MARKERS = ("optimizer", "adam", "adamw", "sgd", "rmsprop")
MODEL_NODE_NAME_MARKERS = (
    "dense",
    "linear",
    "conv",
    "lstm",
    "gru",
    "rnn",
    "attention",
    "dropout",
    "relu",
    "softmax",
    "model",
)


def read_text_any(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json(path: Path | None) -> dict[str, Any]:
    raw = sys.stdin.read() if path is None or str(path) == "-" else read_text_any(path)
    raw = raw.lstrip("\ufeff")
    raw = raw.removeprefix(chr(239) + chr(187) + chr(191))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def is_sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(marker in lower for marker in SENSITIVE_KEY_MARKERS)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if is_sensitive_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        for marker in ("token=", "password=", "secret=", "credential="):
            pos = lowered.find(marker)
            if pos >= 0:
                return value[: pos + len(marker)] + "[REDACTED]"
    return value


def int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in graph.get("nodes", []) if isinstance(node, dict)]


def links(graph: dict[str, Any]) -> list[dict[str, Any]]:
    raw = graph.get("links", graph.get("connections", []))
    return [link for link in raw if isinstance(link, dict)]


def select_node(
    graph_nodes: list[dict[str, Any]],
    *,
    node_id: int | None,
    node_name: str,
    node_type: int | None,
    node_index: int | None,
) -> tuple[int, dict[str, Any]]:
    if not graph_nodes:
        raise ValueError("Graph has no nodes")

    if node_index is not None:
        if node_index < 0 or node_index >= len(graph_nodes):
            raise ValueError(f"No node exists at index {node_index}")
        return node_index, graph_nodes[node_index]

    candidates = list(enumerate(graph_nodes))
    if node_id is not None:
        candidates = [
            (idx, node)
            for idx, node in candidates
            if int_value(node.get("id"), -1) == node_id
        ]
    if node_type is not None:
        candidates = [
            (idx, node)
            for idx, node in candidates
            if int_value(node.get("type"), -1) == node_type
        ]
    if node_name:
        wanted = node_name.lower()
        candidates = [
            (idx, node)
            for idx, node in candidates
            if wanted in str(node.get("name", "")).lower()
        ]

    if not candidates:
        raise ValueError("No nodes matched the selection filters")
    return candidates[0]


def node_by_id(graph_nodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int_value(node.get("id"), -1): node for node in graph_nodes}


def compact_link(link: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    from_id = int_value(link.get("from_node"), -1)
    to_id = int_value(link.get("to_node"), -1)
    return {
        "id": link.get("id"),
        "from_node": from_id,
        "from_name": by_id.get(from_id, {}).get("name", ""),
        "from_pin_index": link.get("from_pin_index", link.get("from_pin")),
        "to_node": to_id,
        "to_name": by_id.get(to_id, {}).get("name", ""),
        "to_pin_index": link.get("to_pin_index", link.get("to_pin")),
    }


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int_value(node.get("id"), -1),
        "name": node.get("name", ""),
        "type": int_value(node.get("type"), -1),
        "category": node.get("category", ""),
        "description": node.get("description", ""),
        "parameters": redact(node.get("parameters", {})),
    }


def select_parameter(node: dict[str, Any], name: str) -> dict[str, Any]:
    if not name:
        return {}
    params = node.get("parameters", {})
    if not isinstance(params, dict):
        return {
            "name": name,
            "exists": False,
            "value": "",
            "sensitive": False,
        }
    for key, value in params.items():
        if str(key).lower() == name.lower():
            sensitive = is_sensitive_key(str(key))
            return {
                "name": key,
                "exists": True,
                "value": "[REDACTED]" if sensitive else redact(value),
                "sensitive": sensitive,
            }
    return {
        "name": name,
        "exists": False,
        "value": "",
        "sensitive": False,
    }


def build_context(graph: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    graph_nodes = nodes(graph)
    graph_links = links(graph)
    selected_index, selected = select_node(
        graph_nodes,
        node_id=args.node_id,
        node_name=args.node_name,
        node_type=args.node_type,
        node_index=args.node_index,
    )
    selected_id = int_value(selected.get("id"), -1)
    by_id = node_by_id(graph_nodes)
    incoming = [
        compact_link(link, by_id)
        for link in graph_links
        if int_value(link.get("to_node"), -1) == selected_id
    ]
    outgoing = [
        compact_link(link, by_id)
        for link in graph_links
        if int_value(link.get("from_node"), -1) == selected_id
    ]
    return {
        "schema": CONTEXT_SCHEMA,
        "graph_summary": {
            "path": str(args.graph) if args.graph else "",
            "version": graph.get("version", ""),
            "workflow_description": graph.get("workflow_description", ""),
            "node_count": len(graph_nodes),
            "link_count": len(graph_links),
        },
        "selection": {
            "node_index": selected_index,
            "node_id": selected_id,
            "node_name": selected.get("name", ""),
            "node_type": int_value(selected.get("type"), -1),
        },
        "selected_parameter": select_parameter(selected, args.parameter),
        "selected_node": compact_node(selected),
        "incoming_links": incoming,
        "outgoing_links": outgoing,
        "neighbor_nodes": [
            compact_node(by_id[node_id])
            for node_id in sorted(
                {
                    int_value(link.get("from_node"), -1)
                    for link in graph_links
                    if int_value(link.get("to_node"), -1) == selected_id
                }
                | {
                    int_value(link.get("to_node"), -1)
                    for link in graph_links
                    if int_value(link.get("from_node"), -1) == selected_id
                }
            )
            if node_id in by_id
        ],
        "answer_contract": {
            "must_answer": [
                "what the selected node is",
                "where it sits in the graph",
                "important configured parameters",
                "what to inspect next",
            ],
            "must_not": [
                "mutate the graph",
                "invent unsupported node behavior",
                "expose redacted local paths or dataset values",
            ],
        },
    }


def graph_evidence_queries(context: dict[str, Any]) -> list[str]:
    node = context["selected_node"]
    name = str(node.get("name", "")).strip()
    node_type = str(node.get("type", "")).strip()
    selected_parameter = context.get("selected_parameter", {})
    queries = [
        "TFIDFVectorizer sentiment graph",
        f"{name} {node_type} graph node",
    ]
    if selected_parameter and selected_parameter.get("exists"):
        queries.insert(0, f"{name} {selected_parameter.get('name', '')}")
        queries.insert(1, f"{selected_parameter.get('name', '')} {node_type} graph parameter")
    params = node.get("parameters", {})
    if isinstance(params, dict):
        for key in list(params.keys())[:5]:
            queries.append(f"{name} {key}")
    return [query for query in queries if query.strip()]


def source_evidence(index: dict[str, Any], context: dict[str, Any], top: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in graph_evidence_queries(context):
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}:{chunk['title']}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": item["score"],
                    "citation": retrieval.citation_for(chunk),
                    "text": chunk["text"],
                }
            )
            if len(evidence) >= top:
                return evidence
    return evidence


def citation_strings(evidence: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in evidence:
        citation = item.get("citation", {})
        if isinstance(citation, dict):
            out.append(
                f"{citation.get('path', '')}:"
                f"{citation.get('line_start', '')}-{citation.get('line_end', '')}"
            )
    return out


def graph_evidence(context: dict[str, Any]) -> list[str]:
    selected = context["selected_node"]
    selected_parameter = context.get("selected_parameter", {})
    evidence = [
        f"graph_summary.node_count={context['graph_summary'].get('node_count', 0)}",
        f"graph_summary.link_count={context['graph_summary'].get('link_count', 0)}",
        f"selected_node.id={selected.get('id', -1)}",
        f"selected_node.name={selected.get('name', '')}",
        f"selected_node.type={selected.get('type', -1)}",
        f"incoming_links.count={len(context.get('incoming_links', []))}",
        f"outgoing_links.count={len(context.get('outgoing_links', []))}",
    ]
    if selected_parameter:
        evidence.append(f"selected_parameter.name={selected_parameter.get('name', '')}")
        evidence.append(f"selected_parameter.exists={selected_parameter.get('exists', False)}")
        evidence.append(f"selected_parameter.value={selected_parameter.get('value', '')}")
    params = selected.get("parameters", {})
    if isinstance(params, dict):
        for key, value in list(params.items())[:8]:
            evidence.append(f"selected_node.parameters.{key}={value}")
    return evidence


def make_explanation(context: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    node = context["selected_node"]
    selected_parameter = context.get("selected_parameter", {})
    name = node.get("name", "") or f"node {node.get('id', -1)}"
    node_type = node.get("type", -1)
    incoming = len(context.get("incoming_links", []))
    outgoing = len(context.get("outgoing_links", []))
    params = node.get("parameters", {})
    param_keys = list(params.keys()) if isinstance(params, dict) else []
    if selected_parameter:
        param_name = selected_parameter.get("name", "")
        if selected_parameter.get("exists"):
            answer = (
                f"{name} is graph node {node.get('id', -1)} with type id {node_type}. "
                f"Focused parameter {param_name} is set to {selected_parameter.get('value', '')}."
            )
            inspect_first = f"Inspect selected_node.parameters.{param_name} and source_evidence for validation rules."
        else:
            answer = (
                f"{name} is graph node {node.get('id', -1)} with type id {node_type}. "
                f"Focused parameter {param_name} is not present on this node."
            )
            inspect_first = f"Confirm whether parameter {param_name} is supported for this node type."
    else:
        answer = f"{name} is graph node {node.get('id', -1)} with type id {node_type}."
        inspect_first = "Inspect selected_node.parameters for configuration values."

    return {
        "schema": EXPLANATION_SCHEMA,
        "answer": answer,
        "where": (
            f"It has {incoming} incoming link(s), {outgoing} outgoing link(s), "
            f"inside a graph with {context['graph_summary'].get('node_count', 0)} node(s)."
        ),
        "likely_why": (
            "Fact: this deterministic explanation only describes the selected graph node. "
            "It does not infer runtime behavior beyond graph fields and cited source evidence."
        ),
        "inspect_next": [
            inspect_first,
            "Inspect incoming_links and outgoing_links to confirm data flow.",
            "Use source_evidence to check whether the node type has implemented behavior.",
        ],
        "graph_evidence": graph_evidence(context),
        "source_evidence": citation_strings(evidence),
        "evidence": graph_evidence(context) + citation_strings(evidence),
        "unknowns": [
            "Node type ids are reported as stored in the graph; metadata names are not resolved in this harness.",
            "Runtime support depends on engine code and active graph execution, not this static graph view.",
        ],
        "unsupported_or_not_implemented": [
            "No graph mutation or graph draft generation is performed.",
            "No source mutation or training launch is performed.",
        ],
        "graph_context": context,
    }


def packet_question(context: dict[str, Any]) -> str:
    selected_parameter = context.get("selected_parameter", {})
    target = "selected CyxWiz graph node"
    if selected_parameter:
        target += f" parameter {selected_parameter.get('name', '')}"
    lines = [
        f"Explain the {target}.",
        "",
        "Graph facts:",
    ]
    for item in graph_evidence(context):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Answer what this node or parameter is, where it sits in the graph,",
            "which configuration values matter, and what to inspect next. Use",
            "graph facts and cited source evidence only. Do not mutate the graph.",
        ]
    )
    return "\n".join(lines)


def make_answer_packet(context: dict[str, Any], evidence: list[dict[str, Any]], top: int) -> dict[str, Any]:
    clipped = evidence[:top]
    missing_notes = []
    if not clipped:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(clipped) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": packet_question(context),
        "answer_contract": {
            "mode": "graph_node_context_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from graph facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not mutate graphs or source",
                "do not expose redacted local paths, dataset values, or secrets",
            ],
        },
        "graph_context": context,
        "evidence": clipped,
        "missing_evidence_notes": missing_notes,
    }


def find_directed_path(
    graph_links: list[dict[str, Any]],
    start_node: int,
    end_node: int,
) -> list[dict[str, Any]]:
    adjacency: dict[int, list[dict[str, Any]]] = {}
    for link in graph_links:
        adjacency.setdefault(int_value(link.get("from_node"), -1), []).append(link)

    queue: list[tuple[int, list[dict[str, Any]]]] = [(start_node, [])]
    visited = {start_node}
    while queue:
        node_id, path = queue.pop(0)
        if node_id == end_node:
            return path
        for link in adjacency.get(node_id, []):
            next_node = int_value(link.get("to_node"), -1)
            if next_node in visited:
                continue
            visited.add(next_node)
            queue.append((next_node, path + [link]))
    return []


def build_path_context(graph: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.from_node_id is None or args.to_node_id is None:
        raise ValueError("--from-node-id and --to-node-id are required")
    graph_nodes = nodes(graph)
    graph_links = links(graph)
    by_id = node_by_id(graph_nodes)
    if args.from_node_id not in by_id:
        raise ValueError(f"No node exists with id {args.from_node_id}")
    if args.to_node_id not in by_id:
        raise ValueError(f"No node exists with id {args.to_node_id}")

    path_links = find_directed_path(graph_links, args.from_node_id, args.to_node_id)
    path_node_ids = [args.from_node_id]
    for link in path_links:
        path_node_ids.append(int_value(link.get("to_node"), -1))

    return {
        "schema": PATH_CONTEXT_SCHEMA,
        "graph_summary": {
            "path": str(args.graph) if args.graph else "",
            "version": graph.get("version", ""),
            "workflow_description": graph.get("workflow_description", ""),
            "node_count": len(graph_nodes),
            "link_count": len(graph_links),
        },
        "selection": {
            "from_node_id": args.from_node_id,
            "from_node_name": by_id[args.from_node_id].get("name", ""),
            "to_node_id": args.to_node_id,
            "to_node_name": by_id[args.to_node_id].get("name", ""),
        },
        "path_found": bool(path_links) or args.from_node_id == args.to_node_id,
        "path_node_ids": path_node_ids if path_links or args.from_node_id == args.to_node_id else [],
        "path_nodes": [
            compact_node(by_id[node_id])
            for node_id in path_node_ids
            if node_id in by_id and (path_links or args.from_node_id == args.to_node_id)
        ],
        "path_links": [compact_link(link, by_id) for link in path_links],
        "answer_contract": {
            "must_answer": [
                "whether a directed path exists",
                "which nodes and links are on the path",
                "what the path implies about graph flow",
                "what to inspect next",
            ],
            "must_not": [
                "mutate the graph",
                "claim runtime execution succeeded",
                "invent missing links",
            ],
        },
    }


def path_evidence(context: dict[str, Any]) -> list[str]:
    evidence = [
        f"graph_summary.node_count={context['graph_summary'].get('node_count', 0)}",
        f"graph_summary.link_count={context['graph_summary'].get('link_count', 0)}",
        f"selection.from_node_id={context['selection'].get('from_node_id', -1)}",
        f"selection.from_node_name={context['selection'].get('from_node_name', '')}",
        f"selection.to_node_id={context['selection'].get('to_node_id', -1)}",
        f"selection.to_node_name={context['selection'].get('to_node_name', '')}",
        f"path_found={context.get('path_found', False)}",
        f"path_node_ids={context.get('path_node_ids', [])}",
    ]
    for idx, link in enumerate(context.get("path_links", [])[:8]):
        evidence.append(
            f"path_links[{idx}]={link.get('from_node')}:{link.get('from_name')} -> "
            f"{link.get('to_node')}:{link.get('to_name')}"
        )
    return evidence


def path_evidence_queries(context: dict[str, Any]) -> list[str]:
    names = [
        str(context["selection"].get("from_node_name", "")),
        str(context["selection"].get("to_node_name", "")),
    ]
    for node in context.get("path_nodes", []):
        name = str(node.get("name", ""))
        if name:
            names.append(name)
    return [
        "TFIDFVectorizer sentiment graph",
        "GraphCompiler ValidateRequiredInputsConnected links graph path",
        "GraphCompiler Compile graph links",
        " ".join(name for name in names if name),
    ]


def path_source_evidence(index: dict[str, Any], context: dict[str, Any], top: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in path_evidence_queries(context):
        if not query.strip():
            continue
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}:{chunk['title']}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": item["score"],
                    "citation": retrieval.citation_for(chunk),
                    "text": chunk["text"],
                }
            )
            if len(evidence) >= top:
                return evidence
    return evidence


def make_path_explanation(context: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    source = context["selection"].get("from_node_name", "") or context["selection"].get("from_node_id", "")
    target = context["selection"].get("to_node_name", "") or context["selection"].get("to_node_id", "")
    if context.get("path_found"):
        answer = f"A directed graph path exists from {source} to {target}."
        where = " -> ".join(node.get("name", str(node.get("id", ""))) for node in context.get("path_nodes", []))
        likely = "Fact: the path is derived directly from the stored graph links."
    else:
        answer = f"No directed graph path was found from {source} to {target}."
        where = "No path node sequence is available."
        likely = "Fact: no route was found by following from_node to to_node links."
    return {
        "schema": PATH_EXPLANATION_SCHEMA,
        "answer": answer,
        "where": where,
        "likely_why": likely,
        "inspect_next": [
            "Inspect path_links to confirm pin-level connectivity.",
            "Inspect each path node's parameters before interpreting runtime behavior.",
            "Use source_evidence to check graph compiler link validation behavior.",
        ],
        "graph_evidence": path_evidence(context),
        "source_evidence": citation_strings(evidence),
        "evidence": path_evidence(context) + citation_strings(evidence),
        "unknowns": [
            "This static path does not prove the graph will execute successfully.",
            "Pin semantics are reported from stored link indices but not interpreted here.",
        ],
        "unsupported_or_not_implemented": [
            "No graph path mutation is performed.",
            "No graph draft generation is performed.",
        ],
        "graph_path_context": context,
    }


def path_packet_question(context: dict[str, Any]) -> str:
    lines = [
        "Explain the selected CyxWiz graph path.",
        "",
        "Graph path facts:",
    ]
    for item in path_evidence(context):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Answer whether a directed path exists, which nodes and links are",
            "on the path, what that implies about graph flow, and what to inspect",
            "next. Use graph facts and cited source evidence only. Do not mutate the graph.",
        ]
    )
    return "\n".join(lines)


def make_path_packet(context: dict[str, Any], evidence: list[dict[str, Any]], top: int) -> dict[str, Any]:
    clipped = evidence[:top]
    missing_notes = []
    if not clipped:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(clipped) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": path_packet_question(context),
        "answer_contract": {
            "mode": "graph_path_context_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from graph path facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not mutate graphs or source",
                "do not claim runtime success from static graph links alone",
            ],
        },
        "graph_path_context": context,
        "evidence": clipped,
        "missing_evidence_notes": missing_notes,
    }


def graph_summary(graph: dict[str, Any], graph_path: Path | None) -> dict[str, Any]:
    graph_nodes = nodes(graph)
    graph_links = links(graph)
    return {
        "path": str(graph_path) if graph_path else "",
        "version": graph.get("version", ""),
        "workflow_description": graph.get("workflow_description", ""),
        "node_count": len(graph_nodes),
        "link_count": len(graph_links),
        "nodes": [compact_node(node) for node in graph_nodes],
    }


def has_pretrain_inspection_node(graph_nodes: list[dict[str, Any]]) -> bool:
    for node in graph_nodes:
        name = str(node.get("name", "")).lower()
        if any(marker in name for marker in INSPECTION_NODE_NAME_MARKERS):
            return True
    return False


def dataloader_nodes(graph_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        node
        for node in graph_nodes
        if int_value(node.get("type"), -1) == 104
        or "dataloader" in str(node.get("name", "")).lower()
    ]


def tfidf_nodes(graph_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        node
        for node in graph_nodes
        if int_value(node.get("type"), -1) == 264
        or "tf-idf" in str(node.get("name", "")).lower()
        or "tfidf" in str(node.get("name", "")).lower()
    ]


def graph_suggestion_queries() -> list[str]:
    return [
        "No pre-train data inspection node found consider adding DataProfiler DescribeStats ValueCounts",
        "DataLoader pin_memory unsupported current batchers compatibility",
        "TFIDFVectorizer min_df must be >= 1 Configure",
    ]


def suggestion_source_evidence(index: dict[str, Any], top: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in graph_suggestion_queries():
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}:{chunk['title']}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": item["score"],
                    "citation": retrieval.citation_for(chunk),
                    "text": chunk["text"],
                }
            )
            if len(evidence) >= top:
                return evidence
    return evidence


def node_name(node: dict[str, Any]) -> str:
    return str(node.get("name", "")).lower()


def nodes_with_name_markers(
    graph_nodes: list[dict[str, Any]],
    markers: tuple[str, ...],
    extra_types: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    return [
        node
        for node in graph_nodes
        if int_value(node.get("type"), -1) in extra_types
        or any(marker in node_name(node) for marker in markers)
    ]


def graph_adjacency(graph_links: list[dict[str, Any]]) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {}
    for link in graph_links:
        from_id = int_value(link.get("from_node"), -1)
        to_id = int_value(link.get("to_node"), -1)
        if from_id < 0 or to_id < 0:
            continue
        adjacency.setdefault(from_id, []).append(to_id)
    return adjacency


def graph_has_cycle(graph_nodes: list[dict[str, Any]], graph_links: list[dict[str, Any]]) -> bool:
    adjacency = graph_adjacency(graph_links)
    ids = [int_value(node.get("id"), -1) for node in graph_nodes]
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node_id: int) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for next_id in adjacency.get(node_id, []):
            if visit(next_id):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in ids if node_id >= 0)


def reachable_node_ids(start_node_id: int, graph_links: list[dict[str, Any]]) -> set[int]:
    adjacency = graph_adjacency(graph_links)
    pending = [start_node_id]
    seen = {start_node_id}
    while pending:
        node_id = pending.pop(0)
        for next_id in adjacency.get(node_id, []):
            if next_id not in seen:
                seen.add(next_id)
                pending.append(next_id)
    return seen


def make_audit_check(
    check_id: str,
    status: str,
    title: str,
    detail: str,
    graph_evidence: list[str],
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "title": title,
        "detail": detail,
        "graph_evidence": graph_evidence,
        "mutation_required": False,
    }


def make_graph_audit(graph: dict[str, Any], graph_path: Path | None, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    graph_nodes = nodes(graph)
    graph_links = links(graph)
    summary = graph_summary(graph, graph_path)
    node_ids = {int_value(node.get("id"), -1) for node in graph_nodes}
    checks: list[dict[str, Any]] = []

    dataset_nodes = nodes_with_name_markers(graph_nodes, DATASET_NODE_NAME_MARKERS, (157,))
    loss_nodes = nodes_with_name_markers(graph_nodes, LOSS_NODE_NAME_MARKERS)
    optimizer_nodes = nodes_with_name_markers(graph_nodes, OPTIMIZER_NODE_NAME_MARKERS)
    model_nodes = nodes_with_name_markers(graph_nodes, MODEL_NODE_NAME_MARKERS)
    cycle_present = graph_has_cycle(graph_nodes, graph_links)

    checks.append(
        make_audit_check(
            "has_nodes",
            "pass" if graph_nodes else "fail",
            "Graph contains nodes",
            "The graph must contain at least one node before it can be explained or compiled.",
            [f"graph_summary.node_count={summary['node_count']}"],
        )
    )
    checks.append(
        make_audit_check(
            "has_dataset_source",
            "pass" if dataset_nodes else "fail",
            "Graph has a dataset or source-like node",
            "A source-like node was detected by node type or name marker.",
            [f"source_like_node_ids={[int_value(node.get('id'), -1) for node in dataset_nodes]}"],
        )
    )
    checks.append(
        make_audit_check(
            "has_model_layer",
            "pass" if model_nodes else "warning",
            "Graph has a model or layer-like node",
            "A model-like node was detected by common layer or activation name markers.",
            [f"model_like_node_ids={[int_value(node.get('id'), -1) for node in model_nodes[:12]]}"],
        )
    )
    checks.append(
        make_audit_check(
            "has_loss",
            "pass" if loss_nodes else "warning",
            "Graph has a loss-like node",
            "A loss-like node was detected by common loss name markers.",
            [f"loss_like_node_ids={[int_value(node.get('id'), -1) for node in loss_nodes]}"],
        )
    )
    checks.append(
        make_audit_check(
            "has_optimizer",
            "pass" if optimizer_nodes else "warning",
            "Graph has an optimizer-like node",
            "An optimizer-like node was detected by common optimizer name markers.",
            [f"optimizer_like_node_ids={[int_value(node.get('id'), -1) for node in optimizer_nodes]}"],
        )
    )
    checks.append(
        make_audit_check(
            "acyclic",
            "fail" if cycle_present else "pass",
            "Directed graph is acyclic",
            "The graph should not contain a directed cycle in its stored links.",
            [f"cycle_present={cycle_present}"],
        )
    )

    if dataset_nodes:
        start_id = int_value(dataset_nodes[0].get("id"), -1)
        reachable = reachable_node_ids(start_id, graph_links)
        disconnected = sorted(node_id for node_id in node_ids if node_id >= 0 and node_id not in reachable)
        checks.append(
            make_audit_check(
                "reachable_from_first_source",
                "pass" if not disconnected else "warning",
                "Nodes are reachable from the first source-like node",
                "Reachability follows stored from_node to to_node links from the first detected source-like node.",
                [
                    f"start_node_id={start_id}",
                    f"disconnected_node_ids={disconnected[:20]}",
                ],
            )
        )

    checks.append(
        make_audit_check(
            "pretrain_inspection_node",
            "pass" if has_pretrain_inspection_node(graph_nodes) else "warning",
            "Graph includes a pre-train data inspection node",
            (
                "No obvious DataProfiler, DescribeStats, ValueCounts, SampleRows, "
                "CorrelationMatrix, or DataValidator node was detected."
            ),
            [f"pretrain_inspection_node_present={has_pretrain_inspection_node(graph_nodes)}"],
        )
    )

    for node in dataloader_nodes(graph_nodes):
        params = node.get("parameters", {})
        pin_memory = params.get("pin_memory") if isinstance(params, dict) else None
        if str(pin_memory).lower() == "true":
            checks.append(
                make_audit_check(
                    f"dataloader_{node.get('id', '')}_pin_memory",
                    "warning",
                    "DataLoader pin_memory is currently unsupported",
                    "pin_memory=true is configured but current batchers ignore it.",
                    [
                        f"node.id={node.get('id', -1)}",
                        f"node.name={node.get('name', '')}",
                        "node.parameters.pin_memory=true",
                    ],
                )
            )

    for node in tfidf_nodes(graph_nodes):
        params = node.get("parameters", {})
        if not isinstance(params, dict) or "min_df" not in params:
            continue
        try:
            min_df = int(str(params.get("min_df", "")).strip())
        except ValueError:
            min_df = 0
        checks.append(
            make_audit_check(
                f"tfidf_{node.get('id', '')}_min_df",
                "pass" if min_df >= 1 else "fail",
                "TF-IDF min_df is valid",
                "TF-IDF min_df should be at least 1.",
                [
                    f"node.id={node.get('id', -1)}",
                    f"node.name={node.get('name', '')}",
                    f"node.parameters.min_df={params.get('min_df', '')}",
                ],
            )
        )

    if any(check["status"] == "fail" for check in checks):
        overall_status = "fail"
    elif any(check["status"] == "warning" for check in checks):
        overall_status = "warning"
    else:
        overall_status = "pass"

    return {
        "schema": AUDIT_SCHEMA,
        "overall_status": overall_status,
        "graph_summary": summary,
        "checks": checks,
        "source_evidence": citation_strings(evidence),
        "evidence": [
            item
            for check in checks
            for item in check.get("graph_evidence", [])
        ] + citation_strings(evidence),
        "unknowns": [
            "This audit is deterministic and structural; it is not a full GraphCompiler validation.",
            "Name-marker checks can miss custom node names or custom node types.",
        ],
        "unsupported_or_not_implemented": [
            "No graph edits are applied.",
            "No graph draft generation is performed.",
            "No training run or compiler call is launched.",
        ],
    }


def audit_packet_question(audit: dict[str, Any]) -> str:
    lines = [
        "Explain a read-only CyxWiz graph audit.",
        "",
        "Graph audit facts:",
        f"- overall_status={audit.get('overall_status', '')}",
    ]
    for check in audit.get("checks", []):
        lines.append(
            f"- {check.get('status', '')}: {check.get('title', '')}: "
            f"{check.get('detail', '')}"
        )
        for item in check.get("graph_evidence", []):
            lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "Explain which audit checks passed, which need attention, and what",
            "source evidence supports the findings. Do not mutate or draft graph changes.",
        ]
    )
    return "\n".join(lines)


def make_audit_packet(audit: dict[str, Any], evidence: list[dict[str, Any]], top: int) -> dict[str, Any]:
    clipped = evidence[:top]
    missing_notes = []
    if not clipped:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(clipped) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": audit_packet_question(audit),
        "answer_contract": {
            "mode": "graph_audit_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from audit facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not mutate graphs or source",
                "do not generate graph drafts",
            ],
        },
        "graph_audit": audit,
        "evidence": clipped,
        "missing_evidence_notes": missing_notes,
    }


def graph_draft_queries(goal: str, template: str) -> list[str]:
    return [
        f"{goal} {template} supported CyxWiz graph nodes",
        "TFIDFVectorizer min_df must be >= 1 Configure",
        "DataLoader batch_size epochs shuffle num_workers GraphCompiler",
        "CrossEntropy AdamW Dense ReLU Dropout sentiment cyxgraph",
        "GraphCompiler FindDatasetInputNode reachable loss node",
    ]


def draft_source_evidence(index: dict[str, Any], goal: str, template: str, top: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in graph_draft_queries(goal, template):
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}:{chunk['title']}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": item["score"],
                    "citation": retrieval.citation_for(chunk),
                    "text": chunk["text"],
                }
            )
            if len(evidence) >= top:
                return evidence
    return evidence


def make_text_classification_draft_plan(goal: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    planned_nodes = [
        {
            "draft_id": "n1",
            "role": "dataset_source",
            "name": "Text CSV source",
            "node_type_hint": 157,
            "parameters_to_set": ["file_path", "text_column", "label_column", "header", "delimiter"],
            "notes": "Use a local CSV source and keep private file paths out of assistant output.",
        },
        {
            "draft_id": "n2",
            "role": "feature_extraction",
            "name": "TF-IDF vectorizer",
            "node_type_hint": 264,
            "parameters_to_set": ["text_col", "label_col", "max_features", "ngram_range", "min_df"],
            "notes": "Keep min_df >= 1 before preflight or compile.",
        },
        {
            "draft_id": "n3",
            "role": "split",
            "name": "Train/validation/test split",
            "node_type_hint": 106,
            "parameters_to_set": ["train_ratio", "val_ratio", "test_ratio", "seed", "stratified"],
            "notes": "Use a deterministic seed and stratification when labels are categorical.",
        },
        {
            "draft_id": "n4",
            "role": "training_loop",
            "name": "DataLoader",
            "node_type_hint": 104,
            "parameters_to_set": ["batch_size", "epochs", "shuffle", "num_workers", "pin_memory"],
            "notes": "pin_memory=true is currently only compatibility metadata and may be ignored.",
        },
        {
            "draft_id": "n5",
            "role": "model_layer",
            "name": "Dense hidden layer",
            "node_type_hint": 0,
            "parameters_to_set": ["units", "activation", "use_bias"],
            "notes": "Start with a modest hidden size before widening the model.",
        },
        {
            "draft_id": "n6",
            "role": "activation",
            "name": "ReLU",
            "node_type_hint": 29,
            "parameters_to_set": [],
            "notes": "Activation follows the hidden dense layer.",
        },
        {
            "draft_id": "n7",
            "role": "regularization",
            "name": "Dropout",
            "node_type_hint": 14,
            "parameters_to_set": ["rate"],
            "notes": "Use only if validation performance suggests overfitting risk.",
        },
        {
            "draft_id": "n8",
            "role": "output_layer",
            "name": "Dense output layer",
            "node_type_hint": 0,
            "parameters_to_set": ["units", "activation", "use_bias"],
            "notes": "Set units to the number of target classes.",
        },
        {
            "draft_id": "n9",
            "role": "loss",
            "name": "CrossEntropy",
            "node_type_hint": 73,
            "parameters_to_set": ["reduction", "label_smoothing", "class_weight"],
            "notes": "Use for multi-class classification unless the graph target says otherwise.",
        },
        {
            "draft_id": "n10",
            "role": "optimizer",
            "name": "AdamW",
            "node_type_hint": 82,
            "parameters_to_set": ["learning_rate", "weight_decay", "beta1", "beta2", "epsilon"],
            "notes": "Start with conservative learning rate and weight decay values.",
        },
        {
            "draft_id": "n11",
            "role": "output",
            "name": "Output",
            "node_type_hint": 71,
            "parameters_to_set": ["classes"],
            "notes": "Expose the classification result count.",
        },
    ]
    planned_links = [
        {"from": "n1", "to": "n2", "purpose": "source rows to text features"},
        {"from": "n2", "to": "n3", "purpose": "features and labels to split"},
        {"from": "n3", "to": "n4", "purpose": "split data to training batches"},
        {"from": "n4", "to": "n5", "purpose": "batches to first trainable layer"},
        {"from": "n5", "to": "n6", "purpose": "hidden activations"},
        {"from": "n6", "to": "n7", "purpose": "regularized activations"},
        {"from": "n7", "to": "n8", "purpose": "classification logits"},
        {"from": "n8", "to": "n9", "purpose": "logits to loss"},
        {"from": "n9", "to": "n10", "purpose": "loss optimized by AdamW"},
        {"from": "n8", "to": "n11", "purpose": "predictions to graph output"},
    ]
    return {
        "schema": DRAFT_PLAN_SCHEMA,
        "status": "draft_plan_only",
        "template": "text-classification-tfidf-mlp",
        "goal": goal,
        "planned_nodes": planned_nodes,
        "planned_links": planned_links,
        "assumptions": [
            "Input data is tabular text with one text column and one label column.",
            "The task is multi-class or single-label text classification.",
            "The number of output classes is known before creating the final graph JSON.",
        ],
        "preflight_checks": [
            "Confirm selected node types exist in the target CyxWiz build.",
            "Confirm TF-IDF min_df is >= 1.",
            "Confirm source node reaches loss and optimizer nodes through directed links.",
            "Confirm class count matches the output dense layer and output node.",
            "Run graph audit before any future apply step.",
        ],
        "risks": [
            "This is a graph plan, not a validated .cyxgraph file.",
            "Custom builds may rename nodes or change node type identifiers.",
            "DataLoader pin_memory may be serialized but ignored by current batchers.",
        ],
        "source_evidence": citation_strings(evidence),
        "evidence": citation_strings(evidence),
        "unsupported_or_not_implemented": [
            "No .cyxgraph JSON is generated by this command.",
            "No graph edits are applied.",
            "No training run or compiler call is launched.",
        ],
    }


def make_graph_draft_plan(template: str, goal: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if template != "text-classification-tfidf-mlp":
        return {
            "schema": DRAFT_PLAN_SCHEMA,
            "status": "unsupported_template",
            "template": template,
            "goal": goal,
            "planned_nodes": [],
            "planned_links": [],
            "assumptions": [],
            "preflight_checks": ["Choose a supported draft template before generating a plan."],
            "risks": ["Unsupported templates intentionally do not produce draft graph structure."],
            "source_evidence": citation_strings(evidence),
            "evidence": citation_strings(evidence),
            "unsupported_or_not_implemented": [
                "Only text-classification-tfidf-mlp is supported in this deterministic slice.",
                "No graph edits are applied.",
            ],
        }
    return make_text_classification_draft_plan(goal, evidence)


def draft_plan_packet_question(plan: dict[str, Any]) -> str:
    lines = [
        "Explain a read-only CyxWiz graph draft plan.",
        "",
        "Graph draft plan facts:",
        f"- status={plan.get('status', '')}",
        f"- template={plan.get('template', '')}",
        f"- goal={plan.get('goal', '')}",
    ]
    lines.append("")
    lines.append("Planned nodes:")
    for node in plan.get("planned_nodes", []):
        lines.append(
            f"- {node.get('draft_id', '')}: {node.get('name', '')} "
            f"role={node.get('role', '')} type_hint={node.get('node_type_hint', '')}"
        )
        if node.get("notes"):
            lines.append(f"  - {node.get('notes', '')}")
    lines.append("")
    lines.append("Planned links:")
    for link in plan.get("planned_links", []):
        lines.append(
            f"- {link.get('from', '')} -> {link.get('to', '')}: {link.get('purpose', '')}"
        )
    lines.extend(
        [
            "",
            "Explain the plan, assumptions, risks, and preflight checks. Do not",
            "generate final .cyxgraph JSON and do not mutate or draft graph changes beyond this plan.",
        ]
    )
    return "\n".join(lines)


def make_draft_plan_packet(plan: dict[str, Any], evidence: list[dict[str, Any]], top: int) -> dict[str, Any]:
    clipped = evidence[:top]
    missing_notes = []
    if not clipped:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(clipped) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": draft_plan_packet_question(plan),
        "answer_contract": {
            "mode": "graph_draft_plan_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from draft plan facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not emit final .cyxgraph JSON",
                "do not mutate graphs or source",
            ],
        },
        "graph_draft_plan": plan,
        "evidence": clipped,
        "missing_evidence_notes": missing_notes,
    }


def make_suggestions(graph: dict[str, Any], graph_path: Path | None, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    graph_nodes = nodes(graph)
    suggestions: list[dict[str, Any]] = []
    summary = graph_summary(graph, graph_path)

    if graph_nodes and not has_pretrain_inspection_node(graph_nodes):
        suggestions.append(
            {
                "id": "pretrain_inspection_node",
                "severity": "info",
                "title": "Consider a pre-train data inspection node",
                "detail": (
                    "The graph has no obvious DataProfiler, DescribeStats, "
                    "ValueCounts, SampleRows, CorrelationMatrix, or DataValidator node."
                ),
                "graph_evidence": [
                    f"graph_summary.node_count={summary['node_count']}",
                    "pretrain_inspection_node_present=False",
                ],
                "mutation_required": False,
            }
        )

    for node in dataloader_nodes(graph_nodes):
        params = node.get("parameters", {})
        pin_memory = params.get("pin_memory") if isinstance(params, dict) else None
        if str(pin_memory).lower() == "true":
            suggestions.append(
                {
                    "id": f"dataloader_{node.get('id', '')}_pin_memory",
                    "severity": "warning",
                    "title": "DataLoader pin_memory is currently unsupported",
                    "detail": "pin_memory=true is configured but current batchers ignore it.",
                    "graph_evidence": [
                        f"node.id={node.get('id', -1)}",
                        f"node.name={node.get('name', '')}",
                        "node.parameters.pin_memory=true",
                    ],
                    "mutation_required": False,
                }
            )

    for node in tfidf_nodes(graph_nodes):
        params = node.get("parameters", {})
        if not isinstance(params, dict) or "min_df" not in params:
            continue
        try:
            min_df = int(str(params.get("min_df", "")).strip())
        except ValueError:
            min_df = 0
        if min_df < 1:
            suggestions.append(
                {
                    "id": f"tfidf_{node.get('id', '')}_min_df",
                    "severity": "error",
                    "title": "TF-IDF min_df must be at least 1",
                    "detail": "The selected TF-IDF vectorizer has an invalid min_df value.",
                    "graph_evidence": [
                        f"node.id={node.get('id', -1)}",
                        f"node.name={node.get('name', '')}",
                        f"node.parameters.min_df={params.get('min_df', '')}",
                    ],
                    "mutation_required": False,
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "id": "no_deterministic_suggestions",
                "severity": "info",
                "title": "No deterministic graph improvement suggestions triggered",
                "detail": "The current read-only checks did not find a known issue pattern.",
                "graph_evidence": [
                    f"graph_summary.node_count={summary['node_count']}",
                    f"graph_summary.link_count={summary['link_count']}",
                ],
                "mutation_required": False,
            }
        )

    return {
        "schema": SUGGESTIONS_SCHEMA,
        "graph_summary": summary,
        "suggestions": suggestions,
        "source_evidence": citation_strings(evidence),
        "evidence": [
            item
            for suggestion in suggestions
            for item in suggestion.get("graph_evidence", [])
        ] + citation_strings(evidence),
        "unknowns": [
            "Suggestions are deterministic checks, not full graph validation.",
            "Runtime behavior still depends on GraphCompiler and active execution.",
        ],
        "unsupported_or_not_implemented": [
            "No graph edits are applied.",
            "No graph draft generation is performed.",
        ],
    }


def suggestions_packet_question(suggestions: dict[str, Any]) -> str:
    lines = [
        "Explain read-only CyxWiz graph improvement suggestions.",
        "",
        "Suggestion facts:",
    ]
    for suggestion in suggestions.get("suggestions", []):
        lines.append(
            f"- {suggestion.get('severity', '')}: {suggestion.get('title', '')}: "
            f"{suggestion.get('detail', '')}"
        )
        for item in suggestion.get("graph_evidence", []):
            lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "Explain which suggestions are facts, what source evidence supports",
            "them, and what to inspect next. Do not mutate or draft graph changes.",
        ]
    )
    return "\n".join(lines)


def make_suggestions_packet(suggestions: dict[str, Any], evidence: list[dict[str, Any]], top: int) -> dict[str, Any]:
    clipped = evidence[:top]
    missing_notes = []
    if not clipped:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(clipped) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": suggestions_packet_question(suggestions),
        "answer_contract": {
            "mode": "graph_suggestions_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from suggestion facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not mutate graphs or source",
                "do not generate graph drafts",
            ],
        },
        "graph_suggestions": suggestions,
        "evidence": clipped,
        "missing_evidence_notes": missing_notes,
    }


def sample_graph() -> dict[str, Any]:
    path = Path("examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph")
    if path.exists():
        return json.loads(read_text_any(path))
    return {
        "version": 1,
        "workflow_description": "fallback sample",
        "nodes": [
            {
                "id": 1,
                "name": "Sentiment CSV",
                "type": 157,
                "parameters": {"file_path": "D:/private/data.csv", "text_column": "statement"},
            },
            {
                "id": 2,
                "name": "TF-IDF",
                "type": 264,
                "parameters": {"text_col": "statement", "max_features": "8000", "min_df": "1"},
            },
        ],
        "links": [{"id": 1, "from_node": 1, "to_node": 2}],
    }


def context_from_args(args: argparse.Namespace) -> dict[str, Any]:
    graph = load_json(args.graph)
    return build_context(graph, args)


def cmd_context(args: argparse.Namespace) -> int:
    context = context_from_args(args)
    print(json.dumps(context, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    context = context_from_args(args)
    evidence = source_evidence(retrieval.load_index(args.index), context, args.top)
    explanation = make_explanation(context, evidence)
    if args.json:
        print(json.dumps(explanation, indent=2 if args.pretty else None, sort_keys=args.pretty))
    else:
        print("Answer:")
        print(explanation["answer"])
        print("\nWhere:")
        print(explanation["where"])
        print("\nWhat to inspect next:")
        for item in explanation["inspect_next"]:
            print(f"- {item}")
        print("\nEvidence:")
        print("Graph fields:")
        for item in explanation["graph_evidence"]:
            print(f"- {item}")
        print("Source citations:")
        for item in explanation["source_evidence"] or ["No source evidence was retrieved."]:
            print(f"- {item}")
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    context = context_from_args(args)
    evidence = source_evidence(retrieval.load_index(args.index), context, args.top)
    packet = make_answer_packet(context, evidence, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    context = build_path_context(graph, args)
    evidence = path_source_evidence(retrieval.load_index(args.index), context, args.top)
    explanation = make_path_explanation(context, evidence)
    if args.json:
        print(json.dumps(explanation, indent=2 if args.pretty else None, sort_keys=args.pretty))
    else:
        print("Answer:")
        print(explanation["answer"])
        print("\nWhere:")
        print(explanation["where"])
        print("\nWhat to inspect next:")
        for item in explanation["inspect_next"]:
            print(f"- {item}")
        print("\nEvidence:")
        print("Graph path fields:")
        for item in explanation["graph_evidence"]:
            print(f"- {item}")
        print("Source citations:")
        for item in explanation["source_evidence"] or ["No source evidence was retrieved."]:
            print(f"- {item}")
    return 0


def cmd_path_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    context = build_path_context(graph, args)
    evidence = path_source_evidence(retrieval.load_index(args.index), context, args.top)
    packet = make_path_packet(context, evidence, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    evidence = suggestion_source_evidence(retrieval.load_index(args.index), args.top)
    suggestions = make_suggestions(graph, args.graph, evidence)
    print(json.dumps(suggestions, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_suggest_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    evidence = suggestion_source_evidence(retrieval.load_index(args.index), args.top)
    suggestions = make_suggestions(graph, args.graph, evidence)
    packet = make_suggestions_packet(suggestions, evidence, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    evidence = suggestion_source_evidence(retrieval.load_index(args.index), args.top)
    audit = make_graph_audit(graph, args.graph, evidence)
    print(json.dumps(audit, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_audit_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    graph = load_json(args.graph)
    evidence = suggestion_source_evidence(retrieval.load_index(args.index), args.top)
    audit = make_graph_audit(graph, args.graph, evidence)
    packet = make_audit_packet(audit, evidence, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_draft_plan(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    evidence = draft_source_evidence(retrieval.load_index(args.index), args.goal, args.template, args.top)
    plan = make_graph_draft_plan(args.template, args.goal, evidence)
    print(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_draft_plan_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    evidence = draft_source_evidence(retrieval.load_index(args.index), args.goal, args.template, args.top)
    plan = make_graph_draft_plan(args.template, args.goal, evidence)
    packet = make_draft_plan_packet(plan, evidence, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    graph = sample_graph()
    base_args = argparse.Namespace(
        graph=Path("examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph"),
        node_id=2,
        node_name="",
        node_type=None,
        node_index=None,
        parameter="min_df",
    )
    context = build_context(graph, base_args)
    index = retrieval.load_index(args.index) if args.index.exists() else {"chunks": []}
    evidence = source_evidence(index, context, args.top)
    explanation = make_explanation(context, evidence)
    packet = make_answer_packet(context, evidence, args.top)
    path_args = argparse.Namespace(
        graph=base_args.graph,
        from_node_id=1,
        to_node_id=4,
    )
    path_context = build_path_context(graph, path_args)
    path_evidence_items = path_source_evidence(index, path_context, args.top)
    path_explanation = make_path_explanation(path_context, path_evidence_items)
    path_packet = make_path_packet(path_context, path_evidence_items, args.top)
    suggestion_evidence_items = suggestion_source_evidence(index, args.top)
    suggestions = make_suggestions(graph, base_args.graph, suggestion_evidence_items)
    suggestions_packet = make_suggestions_packet(suggestions, suggestion_evidence_items, args.top)
    audit = make_graph_audit(graph, base_args.graph, suggestion_evidence_items)
    audit_packet = make_audit_packet(audit, suggestion_evidence_items, args.top)
    audit_check_ids = {item.get("id") for item in audit.get("checks", [])}
    draft_goal = "Generate a draft text-classification graph using supported CyxWiz nodes."
    draft_evidence_items = draft_source_evidence(index, draft_goal, "text-classification-tfidf-mlp", args.top)
    draft_plan = make_graph_draft_plan("text-classification-tfidf-mlp", draft_goal, draft_evidence_items)
    draft_packet = make_draft_plan_packet(draft_plan, draft_evidence_items, args.top)
    checks = [
        context["schema"] == CONTEXT_SCHEMA,
        context["selection"]["node_id"] == 2,
        context["selected_parameter"]["exists"],
        context["selected_parameter"]["value"] == "2",
        "file_path" not in json.dumps(context["selected_node"]),
        explanation["schema"] == EXPLANATION_SCHEMA,
        "selected_node.name=" in "\n".join(explanation["graph_evidence"]),
        packet["schema"] == PACKET_SCHEMA,
        "Graph facts:" in packet["question"],
        bool(packet["evidence"]) if args.index.exists() else True,
        path_context["schema"] == PATH_CONTEXT_SCHEMA,
        path_context["path_found"],
        path_context["path_node_ids"] == [1, 3, 4],
        path_explanation["schema"] == PATH_EXPLANATION_SCHEMA,
        path_packet["schema"] == PACKET_SCHEMA,
        "Graph path facts:" in path_packet["question"],
        suggestions["schema"] == SUGGESTIONS_SCHEMA,
        any(item.get("id") == "pretrain_inspection_node" for item in suggestions["suggestions"]),
        suggestions_packet["schema"] == PACKET_SCHEMA,
        "Suggestion facts:" in suggestions_packet["question"],
        audit["schema"] == AUDIT_SCHEMA,
        audit["overall_status"] in {"pass", "warning", "fail"},
        {"has_nodes", "has_dataset_source", "has_model_layer", "has_loss", "has_optimizer", "acyclic"}.issubset(audit_check_ids),
        audit_packet["schema"] == PACKET_SCHEMA,
        "Graph audit facts:" in audit_packet["question"],
        draft_plan["schema"] == DRAFT_PLAN_SCHEMA,
        draft_plan["status"] == "draft_plan_only",
        any(item.get("name") == "TF-IDF vectorizer" for item in draft_plan["planned_nodes"]),
        draft_packet["schema"] == PACKET_SCHEMA,
        "Graph draft plan facts:" in draft_packet["question"],
    ]
    if not all(checks):
        print(json.dumps({
            "context": context,
            "explanation": explanation,
            "packet": packet,
            "path_context": path_context,
            "path_explanation": path_explanation,
            "path_packet": path_packet,
            "suggestions": suggestions,
            "suggestions_packet": suggestions_packet,
            "audit": audit,
            "audit_packet": audit_packet,
            "draft_plan": draft_plan,
            "draft_packet": draft_packet,
        }, indent=2, sort_keys=True))
        print("Phase 5 graph context check failed", file=sys.stderr)
        return 1
    print("All Phase 5 graph context checks passed.")
    return 0


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--node-name", default="")
    parser.add_argument("--node-type", type=int, default=None)
    parser.add_argument("--node-index", type=int, default=None)
    parser.add_argument("--parameter", default="", help="optional selected node parameter to explain")


def add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    parser.add_argument("--from-node-id", type=int, required=True)
    parser.add_argument("--to-node-id", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    context = sub.add_parser("context", help="build selected graph node context")
    add_selection_args(context)
    context.add_argument("--pretty", action="store_true")
    context.set_defaults(func=cmd_context)

    explain = sub.add_parser("explain", help="explain selected graph node")
    add_selection_args(explain)
    explain.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    explain.add_argument("--top", type=int, default=5)
    explain.add_argument("--json", action="store_true")
    explain.add_argument("--pretty", action="store_true")
    explain.set_defaults(func=cmd_explain)

    packet = sub.add_parser("packet", help="build graph-node Phase 1B packet")
    add_selection_args(packet)
    packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    packet.add_argument("--top", type=int, default=5)
    packet.add_argument("--pretty", action="store_true")
    packet.set_defaults(func=cmd_packet)

    path = sub.add_parser("path", help="explain a directed graph path")
    add_path_args(path)
    path.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    path.add_argument("--top", type=int, default=5)
    path.add_argument("--json", action="store_true")
    path.add_argument("--pretty", action="store_true")
    path.set_defaults(func=cmd_path)

    path_packet = sub.add_parser("path-packet", help="build graph-path Phase 1B packet")
    add_path_args(path_packet)
    path_packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    path_packet.add_argument("--top", type=int, default=5)
    path_packet.add_argument("--pretty", action="store_true")
    path_packet.set_defaults(func=cmd_path_packet)

    suggest = sub.add_parser("suggest", help="emit read-only graph improvement suggestions")
    suggest.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    suggest.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    suggest.add_argument("--top", type=int, default=5)
    suggest.add_argument("--pretty", action="store_true")
    suggest.set_defaults(func=cmd_suggest)

    suggest_packet = sub.add_parser("suggest-packet", help="build graph-suggestions Phase 1B packet")
    suggest_packet.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    suggest_packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    suggest_packet.add_argument("--top", type=int, default=5)
    suggest_packet.add_argument("--pretty", action="store_true")
    suggest_packet.set_defaults(func=cmd_suggest_packet)

    audit = sub.add_parser("audit", help="emit read-only graph audit checks")
    audit.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    audit.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    audit.add_argument("--top", type=int, default=5)
    audit.add_argument("--pretty", action="store_true")
    audit.set_defaults(func=cmd_audit)

    audit_packet = sub.add_parser("audit-packet", help="build graph-audit Phase 1B packet")
    audit_packet.add_argument("--graph", type=Path, default=None, help=".cyxgraph JSON path, or stdin")
    audit_packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    audit_packet.add_argument("--top", type=int, default=5)
    audit_packet.add_argument("--pretty", action="store_true")
    audit_packet.set_defaults(func=cmd_audit_packet)

    draft_plan = sub.add_parser("draft-plan", help="emit a read-only graph draft plan")
    draft_plan.add_argument(
        "--template",
        default="text-classification-tfidf-mlp",
        choices=["text-classification-tfidf-mlp"],
    )
    draft_plan.add_argument(
        "--goal",
        default="Generate a draft text-classification graph using supported CyxWiz nodes.",
    )
    draft_plan.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    draft_plan.add_argument("--top", type=int, default=5)
    draft_plan.add_argument("--pretty", action="store_true")
    draft_plan.set_defaults(func=cmd_draft_plan)

    draft_plan_packet = sub.add_parser("draft-plan-packet", help="build graph-draft-plan Phase 1B packet")
    draft_plan_packet.add_argument(
        "--template",
        default="text-classification-tfidf-mlp",
        choices=["text-classification-tfidf-mlp"],
    )
    draft_plan_packet.add_argument(
        "--goal",
        default="Generate a draft text-classification graph using supported CyxWiz nodes.",
    )
    draft_plan_packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    draft_plan_packet.add_argument("--top", type=int, default=5)
    draft_plan_packet.add_argument("--pretty", action="store_true")
    draft_plan_packet.set_defaults(func=cmd_draft_plan_packet)

    check = sub.add_parser("check", help="run deterministic Phase 5 checks")
    check.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    check.add_argument("--top", type=int, default=5)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
