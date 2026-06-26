#!/usr/bin/env python3
"""
Validate the NER example graph and generated artifacts.

This script performs a lightweight "smoke" validation for local example users:
- graph JSON parses and contains expected nodes/links
- graph path fields are repo-local (no absolute local paths)
- referenced artifacts exist relative to the graph file
- generated metadata references repo-local assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ner_inference


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent.parent
DEFAULT_GRAPH = EXAMPLE_DIR / "ner_bilstm_sequence_tagger.cyxgraph"
DEFAULT_METADATA = EXAMPLE_DIR / "generated" / "ner_metadata.json"
REQUIRED_PATH_KEYS = {"file_path", "raw_source_path", "vocab_file", "tag_vocab_file"}
METADATA_KEYS = {
    "source_csv",
    "sentence_csv",
    "word_vocab_file",
    "pos_vocab_file",
    "tag_vocab_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NER example graph assets.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help="Path to ner_bilstm_sequence_tagger.cyxgraph",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to ner_metadata.json",
    )
    return parser.parse_args()


def add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def validate_json_object(path: Path, issues: list[str]) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        add_issue(issues, f"Missing JSON file: {path}")
        return {}
    except json.JSONDecodeError as exc:
        add_issue(issues, f"Invalid JSON in {path}: {exc}")
        return {}


def resolve_relative_path(raw: str, base_dir: Path) -> Path:
    raw_path = Path(raw)
    candidates = [
        (base_dir / raw_path).resolve(),
        (REPO_ROOT / raw_path).resolve(),
        (EXAMPLE_DIR / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_relative_path(
    owner: str,
    key: str,
    raw_value: str,
    issues: list[str],
    base_dir: Path,
) -> Path:
    if not raw_value:
        add_issue(issues, f"{owner}: {key} is missing.")
        return base_dir

    path_obj = Path(raw_value)
    if path_obj.is_absolute():
        add_issue(
            issues,
            f"{owner}: {key} must be repository-relative, found absolute path: {raw_value}",
        )

    resolved = resolve_relative_path(raw_value, base_dir)
    if not resolved.exists():
        add_issue(
            issues,
            f"{owner}: {key} does not exist: {resolved}",
        )
    return resolved


def validate_graph_nodes(graph: dict, issues: list[str]) -> None:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        add_issue(issues, "Graph has no nodes.")
        return

    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            add_issue(issues, "Encountered a non-object node entry.")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, int):
            add_issue(issues, f"Node has non-integer id: {node}")
            continue
        node_ids.add(node_id)

        name = node.get("name", "<unnamed>")
        params = node.get("parameters")
        if not isinstance(params, dict):
            continue

        for key in REQUIRED_PATH_KEYS:
            if key not in params:
                continue
            validate_relative_path(
                owner=f"Node {node_id}:{name}",
                key=key,
                raw_value=str(params[key]),
                issues=issues,
                base_dir=REPO_ROOT,
            )

    # Verify a handful of expected nodes are present by name.
    names = [node.get("name", "") for node in nodes if isinstance(node, dict)]
    for expected in [
        "NER Sentence CSV",
        "NER Tag Vocabulary",
        "NER Metrics",
        "NER Output",
    ]:
        if expected not in names:
            add_issue(issues, f"Expected node '{expected}' is missing from the saved graph.")

    links = graph.get("links")
    if not isinstance(links, list):
        add_issue(issues, "Graph links block missing or invalid.")
        return

    for link in links:
        if not isinstance(link, dict):
            add_issue(issues, "Encountered non-object link entry.")
            continue
        from_node = link.get("from_node")
        to_node = link.get("to_node")
        if not isinstance(from_node, int) or from_node not in node_ids:
            add_issue(issues, f"Link references unknown from_node: {from_node}")
        if not isinstance(to_node, int) or to_node not in node_ids:
            add_issue(issues, f"Link references unknown to_node: {to_node}")


def validate_metadata(path: Path, issues: list[str]) -> dict:
    metadata = validate_json_object(path, issues)
    if not metadata:
        return {}

    base_dir = path.parent
    for key in METADATA_KEYS:
        if key not in metadata:
            add_issue(issues, f"Metadata missing required key: {key}")
            continue
        validate_relative_path(
            owner=f"Metadata:{path.name}",
            key=key,
            raw_value=str(metadata[key]),
            issues=issues,
                base_dir=base_dir,
            )
    return metadata


def validate_inference_payload(metadata_path: Path, metadata: dict, issues: list[str]) -> None:
    if not metadata:
        return

    try:
        word_vocab = ner_inference.load_vocab(
            ner_inference.resolve_metadata_path(metadata_path, metadata["word_vocab_file"])
        )
        pos_vocab = ner_inference.load_vocab(
            ner_inference.resolve_metadata_path(metadata_path, metadata["pos_vocab_file"])
        )
        max_length = int(metadata.get("configured_max_length", 96))
        tokens = ["British", "troops", "marched", "through", "London", "."]
        pos_tags = ["JJ", "NNS", "VBD", "IN", "NNP", "."]
        word_ids, pos_ids, attention_mask, visible_tokens = ner_inference.encode_sequence(
            tokens=tokens,
            pos_tags=pos_tags,
            word_vocab=word_vocab,
            pos_vocab=pos_vocab,
            max_length=max_length,
        )
        payload = ner_inference.build_payload(
            word_ids=word_ids,
            pos_ids=pos_ids,
            attention_mask=attention_mask,
            sequence_length=len(visible_tokens),
        )
    except Exception as exc:
        add_issue(issues, f"Failed to build NER inference payload: {exc}")
        return

    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        add_issue(issues, "Inference payload missing input object.")
        return

    expected_keys = {"word_ids", "pos_ids", "attention_mask", "sequence_lengths"}
    missing_keys = expected_keys.difference(input_payload)
    if missing_keys:
        add_issue(issues, f"Inference payload missing keys: {sorted(missing_keys)}")

    if input_payload.get("sequence_lengths") != [len(visible_tokens)]:
        add_issue(
            issues,
            "Inference payload sequence_lengths does not match visible token count.",
        )

    for key in ["word_ids", "pos_ids", "attention_mask"]:
        values = input_payload.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], list)
            or len(values[0]) != max_length
        ):
            add_issue(
                issues,
                f"Inference payload {key} must be shaped [1, max_length].",
            )


def main() -> int:
    args = parse_args()
    graph_path = args.graph
    metadata_path = args.metadata
    issues: list[str] = []

    graph = validate_json_object(graph_path, issues)
    if graph:
        validate_graph_nodes(graph, issues)

    metadata = {}
    if metadata_path.exists():
        metadata = validate_metadata(metadata_path, issues)
        validate_inference_payload(metadata_path, metadata, issues)
    else:
        add_issue(issues, f"Metadata file not found: {metadata_path}")

    if issues:
        print("NER graph validation failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print("NER graph validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
