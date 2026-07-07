#!/usr/bin/env python3
"""Build and query a versioned CyxWiz assistant knowledge pack.

This is the first Phase B implementation from the assistant roadmap. It turns
the Phase 1A dev index into a directory that can be loaded without rescanning
the repository at query time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase1a_retrieval as phase1a


DEFAULT_PACK_DIR = Path(__file__).with_name("knowledge_pack")
MANIFEST_SCHEMA = "cyxwiz.assistant.knowledge_pack.v1"
RETRIEVAL_BACKEND = "lexical-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


def chunk_tokens(chunk: dict[str, Any]) -> list[str]:
    fields = [
        str(chunk.get("title", "")),
        str(chunk.get("path", "")),
        str(chunk.get("source_type", "")),
        " ".join(str(tag) for tag in chunk.get("tags", [])),
        str(chunk.get("text", "")),
    ]
    return phase1a.tokenize("\n".join(fields))


def build_retrieval_assets(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    lexicon: dict[str, dict[str, int]] = {}
    postings: dict[str, list[dict[str, int]]] = defaultdict(list)

    for chunk_index, chunk in enumerate(chunks):
        counts = Counter(chunk_tokens(chunk))
        for token, count in sorted(counts.items()):
            postings[token].append({"chunk": chunk_index, "count": count})
            entry = lexicon.setdefault(token, {"df": 0, "tf": 0})
            entry["df"] += 1
            entry["tf"] += count

    return lexicon, dict(sorted(postings.items()))


def source_types(chunks: list[dict[str, Any]]) -> list[str]:
    return sorted({str(chunk.get("source_type", "")) for chunk in chunks if chunk.get("source_type")})


def build_pack(
    pack_dir: Path,
    config_path: Path,
    engine_version: str,
    build_id: str,
    content_revision: str,
) -> dict[str, Any]:
    config = phase1a.load_config(config_path)
    index = phase1a.build_index(config)
    chunks = index.get("chunks", [])
    files = index.get("files", [])
    lexicon, postings = build_retrieval_assets(chunks)

    pack_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pack_dir / "chunks.jsonl", chunks)
    write_json(pack_dir / "lexicon.json", lexicon)
    write_json(pack_dir / "postings.json", postings)
    write_json(
        pack_dir / "metadata.json",
        {
            "schema": "cyxwiz.assistant.knowledge_pack.metadata.v1",
            "repo_root": str(phase1a.REPO_ROOT),
            "source_patterns": index.get("source_patterns", []),
            "config_schema": index.get("config_schema", ""),
            "config_path": index.get("config_path", ""),
            "config_hash": index.get("config_hash", ""),
            "files": files,
        },
    )
    write_json(
        pack_dir / "diagnostics.json",
        {
            "schema": "cyxwiz.assistant.knowledge_pack.diagnostics.v1",
            "warnings": [],
            "notes": [
                "Initial lexical knowledge pack generated from Phase 1A chunking.",
                "Diagnostics metadata is reserved for future structured trace/operator records.",
            ],
        },
    )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "engine_version": engine_version,
        "build_id": build_id,
        "content_revision": content_revision,
        "created_at_utc": utc_now(),
        "chunk_count": len(chunks),
        "source_file_count": len(files),
        "source_types": source_types(chunks),
        "retrieval_backend": RETRIEVAL_BACKEND,
        "assets": {
            "chunks": "chunks.jsonl",
            "lexicon": "lexicon.json",
            "postings": "postings.json",
            "metadata": "metadata.json",
            "diagnostics": "diagnostics.json",
        },
    }
    write_json(pack_dir / "manifest.json", manifest)
    return manifest


def load_pack(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"knowledge pack manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported knowledge pack schema: {manifest.get('schema')}")

    assets = manifest.get("assets", {})
    chunks = read_jsonl(pack_dir / assets.get("chunks", "chunks.jsonl"))
    lexicon = read_json(pack_dir / assets.get("lexicon", "lexicon.json"))
    postings = read_json(pack_dir / assets.get("postings", "postings.json"))
    metadata = read_json(pack_dir / assets.get("metadata", "metadata.json"))
    diagnostics = read_json(pack_dir / assets.get("diagnostics", "diagnostics.json"))

    return {
        "manifest": manifest,
        "chunks": chunks,
        "lexicon": lexicon,
        "postings": postings,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }


def validate_pack(pack_dir: Path, expected_engine_version: str = "") -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    try:
        pack = load_pack(pack_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [str(exc)], {}

    manifest = pack["manifest"]
    chunks = pack["chunks"]
    lexicon = pack["lexicon"]
    postings = pack["postings"]

    if manifest.get("chunk_count") != len(chunks):
        failures.append(
            f"chunk_count_mismatch: manifest={manifest.get('chunk_count')} actual={len(chunks)}"
        )
    if not isinstance(lexicon, dict) or not lexicon:
        failures.append("lexicon_missing_or_empty")
    if not isinstance(postings, dict) or not postings:
        failures.append("postings_missing_or_empty")
    if expected_engine_version and manifest.get("engine_version") != expected_engine_version:
        failures.append(
            "engine_version_mismatch: "
            f"expected={expected_engine_version} actual={manifest.get('engine_version')}"
        )

    required_chunk_fields = {
        "id",
        "source_type",
        "path",
        "line_start",
        "line_end",
        "title",
        "text",
        "content_hash",
        "tags",
    }
    for idx, chunk in enumerate(chunks[: min(len(chunks), 25)]):
        missing = sorted(required_chunk_fields - set(chunk))
        if missing:
            failures.append(f"chunk_{idx}_missing_fields:{','.join(missing)}")
            break

    return not failures, failures, pack


def filters_from_args(args: argparse.Namespace) -> phase1a.SearchFilters:
    return phase1a.SearchFilters(
        source_types=phase1a.normalize_filter_values(getattr(args, "source_type", None)),
        path_contains=phase1a.normalize_filter_values(getattr(args, "path_contains", None)),
        title_contains=phase1a.normalize_filter_values(getattr(args, "title_contains", None)),
        tags=phase1a.normalize_filter_values(getattr(args, "tag", None)),
    )


def pack_search(
    pack: dict[str, Any],
    query: str,
    top: int,
    filters: phase1a.SearchFilters,
) -> list[dict[str, Any]]:
    chunks = pack["chunks"]
    postings = pack["postings"]
    candidate_indexes: set[int] = set()
    for token in phase1a.tokenize(query):
        for posting in postings.get(token, []):
            candidate_indexes.add(int(posting["chunk"]))

    if not candidate_indexes:
        candidate_indexes = set(range(len(chunks)))

    query_terms = phase1a.tokenize(query)
    results: list[dict[str, Any]] = []
    for chunk_index in candidate_indexes:
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        if not phase1a.chunk_matches_filters(chunk, filters):
            continue
        score = phase1a.score_chunk(chunk, query, query_terms)
        if score > 0:
            results.append({"score": score, "chunk": chunk})

    results.sort(
        key=lambda item: (
            -item["score"],
            item["chunk"].get("path", ""),
            item["chunk"].get("line_start", 0),
        )
    )
    return results[:top]


def search_report(
    pack: dict[str, Any],
    query: str,
    top: int,
    filters: phase1a.SearchFilters,
    include_full_text: bool,
) -> dict[str, Any]:
    results = pack_search(pack, query, top, filters)
    items: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        record = {
            "rank": rank,
            "score": item["score"],
            "citation": phase1a.citation_for(chunk),
            "preview": phase1a.preview_text(chunk.get("text", ""), query, 280),
        }
        if include_full_text:
            record["text"] = chunk.get("text", "")
        items.append(record)

    return {
        "schema": "cyxwiz.assistant.knowledge_pack.search_report.v1",
        "query": query,
        "query_terms": phase1a.tokenize(query),
        "pack": {
            "engine_version": pack["manifest"].get("engine_version", ""),
            "build_id": pack["manifest"].get("build_id", ""),
            "content_revision": pack["manifest"].get("content_revision", ""),
            "chunk_count": pack["manifest"].get("chunk_count", 0),
            "source_file_count": pack["manifest"].get("source_file_count", 0),
        },
        "filters": {
            "source_types": filters.source_types,
            "path_contains": filters.path_contains,
            "title_contains": filters.title_contains,
            "tags": filters.tags,
        },
        "result_count": len(results),
        "results": items,
    }


def make_packet(pack: dict[str, Any], query: str, top: int, filters: phase1a.SearchFilters) -> dict[str, Any]:
    results = pack_search(pack, query, top, filters)
    evidence = []
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        evidence.append(
            {
                "rank": rank,
                "score": item["score"],
                "citation": phase1a.citation_for(chunk),
                "text": chunk.get("text", ""),
            }
        )

    notes = []
    if not evidence:
        notes.append("No matching local evidence was found in the knowledge pack.")

    return {
        "schema": "cyxwiz.assistant.knowledge_pack.answer_packet.v1",
        "question": query,
        "pack": {
            "engine_version": pack["manifest"].get("engine_version", ""),
            "build_id": pack["manifest"].get("build_id", ""),
            "content_revision": pack["manifest"].get("content_revision", ""),
        },
        "filters": {
            "source_types": filters.source_types,
            "path_contains": filters.path_contains,
            "title_contains": filters.title_contains,
            "tags": filters.tags,
        },
        "answer_contract": {
            "mode": "retrieval_only",
            "model_runtime": "not_used",
            "rules": [
                "answer only from cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not claim unsupported CyxWiz behavior",
            ],
        },
        "evidence": evidence,
        "missing_evidence_notes": notes,
    }


def print_search_text(report: dict[str, Any]) -> None:
    print(f"Query: {report['query']}")
    pack = report.get("pack", {})
    print(
        "Pack: "
        f"engine={pack.get('engine_version', '')} "
        f"build={pack.get('build_id', '')} "
        f"chunks={pack.get('chunk_count', 0)}"
    )
    if not report.get("results"):
        print("No matches.")
        return
    for item in report["results"]:
        citation = item["citation"]
        print(
            f"{item['rank']}. score={item['score']} "
            f"{citation['path']}:{citation['line_start']}-{citation['line_end']}"
        )
        print(f"   title: {citation['title']}")
        print(f"   type: {citation['source_type']}")
        print(f"   preview: {item['preview']}")


def cmd_build(args: argparse.Namespace) -> int:
    revision = args.content_revision or git_revision(phase1a.REPO_ROOT)
    manifest = build_pack(args.pack, args.config, args.engine_version, args.build_id, revision)
    print(
        f"Wrote knowledge pack {args.pack} with "
        f"{manifest['source_file_count']} files and {manifest['chunk_count']} chunks."
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ok, failures, pack = validate_pack(args.pack, args.expected_engine_version)
    if args.json:
        payload = {
            "schema": "cyxwiz.assistant.knowledge_pack.validation.v1",
            "ok": ok,
            "pack": str(args.pack),
            "failures": failures,
            "manifest": pack.get("manifest", {}) if pack else {},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if ok else 1

    print(f"OK: {ok}")
    if pack:
        manifest = pack["manifest"]
        print(f"Schema: {manifest.get('schema')}")
        print(f"Engine version: {manifest.get('engine_version')}")
        print(f"Build id: {manifest.get('build_id')}")
        print(f"Content revision: {manifest.get('content_revision')}")
        print(f"Files: {manifest.get('source_file_count')}")
        print(f"Chunks: {manifest.get('chunk_count')}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    try:
        pack = load_pack(args.pack)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load knowledge pack: {exc}")
        return 2
    report = search_report(pack, args.query, args.top, filters_from_args(args), args.include_full_text)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print_search_text(report)
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    try:
        pack = load_pack(args.pack)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load knowledge pack: {exc}")
        return 2
    packet = make_packet(pack, args.query, args.top, filters_from_args(args))
    if args.json:
        print(json.dumps(packet, indent=2, ensure_ascii=True))
    else:
        phase1a.print_answer_packet_markdown(
            {
                "question": packet["question"],
                "preset": "",
                "filters": packet["filters"],
                "evidence": packet["evidence"],
                "missing_evidence_notes": packet["missing_evidence_notes"],
            }
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        pack = load_pack(args.pack)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load knowledge pack: {exc}")
        return 2

    config = phase1a.load_config(args.config)
    failed = 0
    for check in config["success_checks"]:
        query = check["query"]
        results = pack_search(pack, query, 1, phase1a.SearchFilters([], [], [], []))
        if not results:
            print(f"FAIL: {query}")
            print("  no results")
            failed += 1
            continue

        chunk = results[0]["chunk"]
        path_ok = chunk["path"] == check["expected_path"]
        title_ok = chunk["title"] == check["expected_title"]
        status = "PASS" if path_ok and title_ok else "FAIL"
        print(f"{status}: {query}")
        print(f"  got:      {chunk['path']} :: {chunk['title']}")
        print(f"  expected: {check['expected_path']} :: {check['expected_title']}")
        if not (path_ok and title_ok):
            failed += 1

    if failed:
        print(f"{failed} knowledge-pack retrieval check(s) failed.")
        return 1

    print("All knowledge-pack retrieval checks passed.")
    return 0


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-type", action="append", choices=["source", "markdown", "cyxgraph", "cyxgraph_node", "cyxgraph_links", "text"])
    parser.add_argument("--path-contains", action="append")
    parser.add_argument("--title-contains", action="append")
    parser.add_argument("--tag", action="append")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and query CyxWiz assistant knowledge packs")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="build a knowledge pack from the configured corpus")
    build_parser.add_argument("--config", type=Path, default=phase1a.DEFAULT_CONFIG)
    build_parser.add_argument("--engine-version", default="dev")
    build_parser.add_argument("--build-id", default="local")
    build_parser.add_argument("--content-revision", default="")
    build_parser.set_defaults(func=cmd_build)

    validate_parser = sub.add_parser("validate", help="validate a knowledge pack")
    validate_parser.add_argument("--expected-engine-version", default="")
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(func=cmd_validate)

    check_parser = sub.add_parser("check", help="run configured retrieval checks against the pack")
    check_parser.add_argument("--config", type=Path, default=phase1a.DEFAULT_CONFIG)
    check_parser.set_defaults(func=cmd_check)

    search_parser = sub.add_parser("search", help="search a knowledge pack without rescanning the repo")
    search_parser.add_argument("query")
    search_parser.add_argument("--top", type=int, default=5)
    search_parser.add_argument("--include-full-text", action="store_true")
    search_parser.add_argument("--json", action="store_true")
    add_filter_args(search_parser)
    search_parser.set_defaults(func=cmd_search)

    packet_parser = sub.add_parser("packet", help="build a retrieval-only answer packet from a knowledge pack")
    packet_parser.add_argument("query")
    packet_parser.add_argument("--top", type=int, default=5)
    packet_parser.add_argument("--json", action="store_true")
    add_filter_args(packet_parser)
    packet_parser.set_defaults(func=cmd_packet)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
