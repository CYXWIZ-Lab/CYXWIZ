#!/usr/bin/env python3
"""Phase 1A local retrieval prototype for tofix42.

This is intentionally small:
- stdlib only
- manual rebuild
- JSON index
- lexical scoring only
- no model runtime, embeddings, watcher, database, or Studio UI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = Path(__file__).with_name("phase1a_index.json")
DEFAULT_CONFIG = Path(__file__).with_name("phase1a_config.json")
MAX_SOURCE_CHUNK_LINES = 120
SOURCE_CHUNK_OVERLAP_LINES = 20

FALLBACK_PATTERNS = [
    "docs/Data Studio/*.md",
    "docs/usage/*.md",
    "examples/cyxgraph/**/*.md",
    "examples/cyxgraph/**/*.cyxgraph",
    "cyxwiz-engine/src/core/**/*.h",
    "cyxwiz-engine/src/core/**/*.hpp",
    "cyxwiz-engine/src/core/**/*.cpp",
]

FALLBACK_SUCCESS_CHECKS = [
    {
        "query": "What source file defines DebugTraceRecord",
        "expected_path": "cyxwiz-engine/src/core/debug_trace_record.h",
        "expected_title": "DebugTraceRecord",
    },
    {
        "query": "terminal_reason training trace",
        "expected_path": "cyxwiz-engine/src/core/training_trace_collector.h",
        "expected_title": "TrainingTraceCollector",
    },
    {
        "query": "TFIDFVectorizer sentiment graph",
        "expected_path": "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph",
        "expected_title": "graph:sentiment_analysis_tfidf_mlp_classifier",
    },
]

FALLBACK_SAVED_QUERIES = [
    {
        "name": "training_terminal",
        "description": "Training terminal status and terminal_reason flows in source files.",
        "query": "terminal_reason",
        "source_types": ["source"],
        "path_contains": ["training"],
    },
    {
        "name": "pin_memory_truth",
        "description": "Pinned host-memory truth and compiler/runtime compatibility warnings.",
        "query": "DataLoader pin_memory unsupported current batchers compatibility",
        "source_types": ["source", "markdown"],
    },
    {
        "name": "sentiment_graphs",
        "description": "Sentiment graph examples from cyxgraph assets.",
        "query": "sentiment",
        "source_types": ["cyxgraph"],
        "path_contains": ["examples/cyxgraph"],
    },
    {
        "name": "tfidf_validation",
        "description": "TF-IDF validation and graph references.",
        "query": "TFIDFVectorizer min_df",
        "source_types": ["source", "cyxgraph"],
        "tag": ["tfidf"],
    },
]

WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "does",
    "file",
    "for",
    "from",
    "in",
    "is",
    "me",
    "of",
    "show",
    "source",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "who",
}
SOURCE_TYPE_BOOST = {
    "source": 40,
    "cyxgraph": 30,
    "cyxgraph_node": 25,
    "cyxgraph_links": 10,
    "markdown": 0,
    "text": 0,
}
BROAD_HELP_TERMS = {
    "assist",
    "assistant",
    "capabilities",
    "capability",
    "help",
    "overview",
    "use",
}
CPP_DECL_RE = re.compile(
    r"^\s*(?:struct|class|enum\s+class|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
CPP_METHOD_RE = re.compile(
    r"^\s*(?:[A-Za-z_:<>~*&]+\s+)+([A-Za-z_][A-Za-z0-9_:~]*::[A-Za-z_][A-Za-z0-9_~]*)\s*\("
)
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    id: str
    source_type: str
    path: str
    line_start: int
    line_end: int
    title: str
    text: str
    content_hash: str
    tags: list[str]


@dataclass
class IndexedFile:
    path: str
    source_type: str
    content_hash: str
    size_bytes: int


@dataclass
class SearchFilters:
    source_types: list[str]
    path_contains: list[str]
    title_contains: list[str]
    tags: list[str]


@dataclass
class SavedQueryPreset:
    name: str
    description: str
    query: str
    filters: SearchFilters


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def stable_id(path: str, title: str, line_start: int, text: str) -> str:
    digest = hashlib.sha1(f"{path}:{title}:{line_start}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tags_for(path: Path, source_type: str, extra: Iterable[str] = ()) -> list[str]:
    parts = re.split(r"[/\\ ._-]+", repo_rel(path).lower())
    tags = {source_type, path.suffix.lower().lstrip(".")}
    tags.update(part for part in parts if part)
    tags.update(item.lower() for item in extra if item)
    return sorted(tags)


def make_chunk(
    path: Path,
    source_type: str,
    line_start: int,
    line_end: int,
    title: str,
    text: str,
    extra_tags: Iterable[str] = (),
) -> Chunk:
    rel = repo_rel(path)
    return Chunk(
        id=stable_id(rel, title, line_start, text),
        source_type=source_type,
        path=rel,
        line_start=line_start,
        line_end=line_end,
        title=title,
        text=text.strip(),
        content_hash=content_hash(text),
        tags=tags_for(path, source_type, extra_tags),
    )


def chunk_markdown(path: Path, text: str) -> list[Chunk]:
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            starts.append((idx, match.group(2).strip()))

    if not starts:
        return [make_chunk(path, "markdown", 1, len(lines), path.name, text)]

    chunks: list[Chunk] = []
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start - 1 : end])
        chunks.append(make_chunk(path, "markdown", start, end, title, body))
    return chunks


def chunk_cpp(path: Path, text: str) -> list[Chunk]:
    lines = text.splitlines()
    declarations: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        match = CPP_DECL_RE.match(line)
        if match:
            declarations.append((idx, match.group(1)))
            continue
        match = CPP_METHOD_RE.match(line)
        if match:
            stripped = line.lstrip()
            if stripped.startswith(("return ", ":")):
                continue
            prefix = line[: match.start(1)]
            if "=" in prefix:
                continue
            declarations.append((idx, match.group(1)))

    chunks: list[Chunk] = []
    for i, (start, title) in enumerate(declarations):
        end = declarations[i + 1][0] - 1 if i + 1 < len(declarations) else min(len(lines), start + 120)
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + MAX_SOURCE_CHUNK_LINES - 1)
            body = "\n".join(lines[chunk_start - 1 : chunk_end])
            if not body.strip():
                if chunk_end >= end:
                    break
                chunk_start = max(chunk_start + 1, chunk_end - SOURCE_CHUNK_OVERLAP_LINES + 1)
                continue
            chunk_title = title
            if chunk_start != start or chunk_end != end:
                chunk_title = f"{title}:{chunk_start}-{chunk_end}"
            chunks.append(
                make_chunk(path, "source", chunk_start, chunk_end, chunk_title, body, [title])
            )
            if chunk_end >= end:
                break
            chunk_start = max(chunk_start + 1, chunk_end - SOURCE_CHUNK_OVERLAP_LINES + 1)

    if chunks:
        return chunks

    window = 80
    for start in range(1, len(lines) + 1, window):
        end = min(len(lines), start + window - 1)
        body = "\n".join(lines[start - 1 : end])
        chunks.append(make_chunk(path, "source", start, end, f"{path.name}:{start}-{end}", body))
    return chunks


def chunk_cyxgraph(path: Path, text: str) -> list[Chunk]:
    try:
        graph = json.loads(text)
    except json.JSONDecodeError:
        return [make_chunk(path, "cyxgraph", 1, len(text.splitlines()), path.name, text)]

    chunks: list[Chunk] = []
    name = str(graph.get("name") or path.stem)
    description = str(graph.get("description") or "")
    params = graph.get("parameters") or []
    summary = {
        "name": name,
        "description": description,
        "parameters": params,
    }
    chunks.append(
        make_chunk(
            path,
            "cyxgraph",
            1,
            len(text.splitlines()),
            f"graph:{name}",
            json.dumps(summary, indent=2, ensure_ascii=True),
            ["graph", name],
        )
    )

    nodes = graph.get("nodes") or []
    for node in nodes:
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "")
        node_name = str(node.get("name") or node_id or node_type)
        title = f"node:{node_name}:{node_type}"
        chunks.append(
            make_chunk(
                path,
                "cyxgraph_node",
                1,
                len(text.splitlines()),
                title,
                json.dumps(node, indent=2, ensure_ascii=True),
                ["node", node_id, node_type, node_name],
            )
        )

    links = graph.get("links") or graph.get("connections") or []
    if links:
        chunks.append(
            make_chunk(
                path,
                "cyxgraph_links",
                1,
                len(text.splitlines()),
                f"links:{name}",
                json.dumps(links, indent=2, ensure_ascii=True),
                ["links", name],
            )
        )

    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    text = read_text(path)
    suffix = path.suffix.lower()
    if suffix == ".md":
        return chunk_markdown(path, text)
    if suffix in {".h", ".hpp", ".cpp", ".cc", ".cxx"}:
        return chunk_cpp(path, text)
    if suffix == ".cyxgraph":
        return chunk_cyxgraph(path, text)
    return [make_chunk(path, "text", 1, len(text.splitlines()), path.name, text)]


def source_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix in {".h", ".hpp", ".cpp", ".cc", ".cxx"}:
        return "source"
    if suffix == ".cyxgraph":
        return "cyxgraph"
    return "text"


def file_metadata(path: Path) -> IndexedFile:
    text = read_text(path)
    encoded = text.encode("utf-8")
    return IndexedFile(
        path=repo_rel(path),
        source_type=source_type_for(path),
        content_hash=content_hash(text),
        size_bytes=len(encoded),
    )


def iter_initial_files(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                found.append(path)
                seen.add(path)
    return sorted(found)


def build_index(config: dict) -> dict:
    chunks: list[Chunk] = []
    patterns = config["source_patterns"]
    files = iter_initial_files(patterns)
    for path in files:
        chunks.extend(chunk_file(path))
    indexed_files = [file_metadata(path) for path in files]

    return {
        "schema": "cyxwiz.tofix42.phase1a.lexical_index.v1",
        "config_schema": config["schema"],
        "config_path": config["config_path"],
        "config_hash": config["config_hash"],
        "repo_root": str(REPO_ROOT),
        "source_patterns": patterns,
        "file_count": len(files),
        "chunk_count": len(chunks),
        "files": [asdict(item) for item in indexed_files],
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def save_index(index: dict, path: Path) -> None:
    path.write_text(json.dumps(index, indent=2, ensure_ascii=True), encoding="utf-8")


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    if not path.exists():
        return {
            "schema": "cyxwiz.tofix42.phase1a.config.fallback",
            "config_path": str(path),
            "config_hash": "",
            "source_patterns": FALLBACK_PATTERNS,
            "saved_queries": FALLBACK_SAVED_QUERIES,
            "success_checks": FALLBACK_SUCCESS_CHECKS,
        }
    raw = path.read_text(encoding="utf-8")
    config = json.loads(raw)
    return {
        "schema": config.get("schema", "cyxwiz.tofix42.phase1a.config.unknown"),
        "config_path": str(path),
        "config_hash": content_hash(raw),
        "source_patterns": config.get("source_patterns") or FALLBACK_PATTERNS,
        "saved_queries": config.get("saved_queries") or FALLBACK_SAVED_QUERIES,
        "success_checks": config.get("success_checks") or FALLBACK_SUCCESS_CHECKS,
    }


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in WORD_RE.findall(text):
        token = raw.lower()
        variants = [token, normalize_token(token)]
        if token.endswith("vectorizer"):
            variants.append(token.removesuffix("vectorizer"))
            variants.append(normalize_token(token.removesuffix("vectorizer")))
        for variant in variants:
            if variant and variant not in STOPWORDS and len(variant) > 1 and variant not in seen:
                out.append(variant)
                seen.add(variant)
    return out


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", text.lower())


def normalize_filter_values(values: list[str] | None) -> list[str]:
    return [value.strip().lower() for value in (values or []) if value and value.strip()]


def filters_from_args(args: argparse.Namespace) -> SearchFilters:
    return SearchFilters(
        source_types=normalize_filter_values(getattr(args, "source_type", None)),
        path_contains=normalize_filter_values(getattr(args, "path_contains", None)),
        title_contains=normalize_filter_values(getattr(args, "title_contains", None)),
        tags=normalize_filter_values(getattr(args, "tag", None)),
    )


def merge_filter_values(first: list[str], second: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in [*first, *second]:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def merge_filters(base: SearchFilters, extra: SearchFilters) -> SearchFilters:
    return SearchFilters(
        source_types=merge_filter_values(base.source_types, extra.source_types),
        path_contains=merge_filter_values(base.path_contains, extra.path_contains),
        title_contains=merge_filter_values(base.title_contains, extra.title_contains),
        tags=merge_filter_values(base.tags, extra.tags),
    )


def normalize_preset(raw: dict) -> SavedQueryPreset:
    return SavedQueryPreset(
        name=str(raw.get("name") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        query=str(raw.get("query") or "").strip(),
        filters=SearchFilters(
            source_types=normalize_filter_values(raw.get("source_types")),
            path_contains=normalize_filter_values(raw.get("path_contains")),
            title_contains=normalize_filter_values(raw.get("title_contains")),
            tags=normalize_filter_values(raw.get("tag") or raw.get("tags")),
        ),
    )


def preset_map(config: dict) -> dict[str, SavedQueryPreset]:
    presets: dict[str, SavedQueryPreset] = {}
    for raw in config.get("saved_queries", []):
        if not isinstance(raw, dict):
            continue
        preset = normalize_preset(raw)
        if preset.name:
            presets[preset.name] = preset
    return presets


def resolve_query_and_filters(
    config: dict,
    args: argparse.Namespace,
) -> tuple[str, SearchFilters, SavedQueryPreset | None]:
    presets = preset_map(config)
    preset_name = str(getattr(args, "preset", "") or "").strip()
    preset = presets.get(preset_name) if preset_name else None
    if preset_name and preset is None:
        available = ", ".join(sorted(presets))
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")
    query = str(getattr(args, "query", "") or "").strip()
    if not query and preset is not None:
        query = preset.query
    if not query:
        raise ValueError("A query is required unless --preset provides one.")
    explicit_filters = filters_from_args(args)
    filters = merge_filters(preset.filters, explicit_filters) if preset else explicit_filters
    return query, filters, preset


def has_active_filters(filters: SearchFilters) -> bool:
    return bool(
        filters.source_types
        or filters.path_contains
        or filters.title_contains
        or filters.tags
    )


def chunk_matches_filters(chunk: dict, filters: SearchFilters) -> bool:
    source_type = str(chunk.get("source_type", "")).lower()
    path = str(chunk.get("path", "")).lower()
    title = str(chunk.get("title", "")).lower()
    tags = {str(tag).lower() for tag in chunk.get("tags", [])}

    if filters.source_types and source_type not in filters.source_types:
        return False
    if filters.path_contains and not all(term in path for term in filters.path_contains):
        return False
    if filters.title_contains and not all(term in title for term in filters.title_contains):
        return False
    if filters.tags and not all(tag in tags for tag in filters.tags):
        return False
    return True


def score_chunk(chunk: dict, query: str, query_terms: list[str]) -> int:
    text = chunk.get("text", "").lower()
    title = chunk.get("title", "").lower()
    path = chunk.get("path", "").lower()
    tags = " ".join(chunk.get("tags", [])).lower()
    query_lower = query.lower()
    text_norm = normalize_token(text)
    title_norm = normalize_token(title)
    path_norm = normalize_token(path)
    tags_norm = normalize_token(tags)

    score = 0
    score += SOURCE_TYPE_BOOST.get(chunk.get("source_type", ""), 0)
    broad_help_query = (
        "cyxwiz" in query_terms
        and (
            len(query_terms) <= 4
            or any(term in BROAD_HELP_TERMS for term in query_terms)
        )
    )
    if broad_help_query:
        source_type = chunk.get("source_type", "")
        if source_type in {"markdown", "text"}:
            score += 120
        if "knowledge_seed" in path or "overview" in path or "readme" in path:
            score += 100
        if "usage" in path or "assistant" in path or "capabilities" in path:
            score += 40
    if query_lower and query_lower in text:
        score += 30
    if query_lower and query_lower in title:
        score += 40
    matched_terms = 0
    for term in query_terms:
        if not term:
            continue
        exact_weight = min(len(term), 16)
        term_norm = normalize_token(term)
        text_hits = min(max(text.count(term), text_norm.count(term_norm)), 3)
        title_hits = min(max(title.count(term), title_norm.count(term_norm)), 2)
        path_hits = min(max(path.count(term), path_norm.count(term_norm)), 2)
        tag_hits = min(max(tags.count(term), tags_norm.count(term_norm)), 2)
        if text_hits or title_hits or path_hits or tag_hits:
            matched_terms += 1
        score += text_hits * max(1, exact_weight // 4)
        score += title_hits * (10 + exact_weight)
        score += path_hits * (6 + exact_weight)
        score += tag_hits * (4 + exact_weight)
    if matched_terms:
        score += matched_terms * matched_terms * 3
        if matched_terms == len(query_terms):
            score += 30
        elif matched_terms >= max(3, len(query_terms) - 1):
            score += 15
    return score


def search(index: dict, query: str, top: int, filters: SearchFilters | None = None) -> list[dict]:
    query_terms = tokenize(query)
    filters = filters or SearchFilters([], [], [], [])
    results: list[dict] = []
    for chunk in index.get("chunks", []):
        if not chunk_matches_filters(chunk, filters):
            continue
        score = score_chunk(chunk, query, query_terms)
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


def preview_text(text: str, query: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact

    lower = compact.lower()
    best = -1
    for term in sorted(tokenize(query), key=len, reverse=True):
        idx = lower.find(term.lower())
        if idx >= 0:
            best = idx
            break

    if best < 0:
        return compact[:limit]

    start = max(0, best - limit // 3)
    end = min(len(compact), start + limit)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def print_results(results: list[dict], query: str) -> None:
    if not results:
        print("No matches.")
        return

    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        citation = f"{chunk['path']}:{chunk['line_start']}-{chunk['line_end']}"
        preview = preview_text(chunk["text"], query, 280)
        print(f"{rank}. score={item['score']} {citation}")
        print(f"   title: {chunk['title']}")
        print(f"   type: {chunk['source_type']}")
        print(f"   preview: {preview}")


def summarize_results_by_file(results: list[dict], top_files: int) -> list[dict]:
    summary: dict[str, dict] = {}
    for item in results:
        chunk = item["chunk"]
        path = chunk["path"]
        existing = summary.get(path)
        if existing is None:
            summary[path] = {
                "path": path,
                "source_type": chunk.get("source_type", ""),
                "best_score": item["score"],
                "hit_count": 1,
                "best_title": chunk.get("title", ""),
                "line_start": chunk.get("line_start", 0),
                "line_end": chunk.get("line_end", 0),
            }
            continue
        existing["hit_count"] += 1
        if item["score"] > existing["best_score"]:
            existing["best_score"] = item["score"]
            existing["best_title"] = chunk.get("title", "")
            existing["line_start"] = chunk.get("line_start", 0)
            existing["line_end"] = chunk.get("line_end", 0)
            existing["source_type"] = chunk.get("source_type", "")
    ordered = sorted(
        summary.values(),
        key=lambda item: (-item["best_score"], item["path"]),
    )
    return ordered[:top_files]


def make_search_report(
    index: dict,
    query: str,
    top: int,
    top_files: int,
    include_full_text: bool,
    filters: SearchFilters | None = None,
    preset_name: str = "",
) -> dict:
    filters = filters or SearchFilters([], [], [], [])
    results = search(index, query, top, filters)
    query_terms = tokenize(query)
    items = []
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        record = {
            "rank": rank,
            "score": item["score"],
            "citation": citation_for(chunk),
            "preview": preview_text(chunk["text"], query, 280),
        }
        if include_full_text:
            record["text"] = chunk["text"]
        items.append(record)
    return {
        "schema": "cyxwiz.tofix42.phase1a.search_report.v1",
        "query": query,
        "query_terms": query_terms,
        "preset": preset_name,
        "filters": {
            "source_types": filters.source_types,
            "path_contains": filters.path_contains,
            "title_contains": filters.title_contains,
            "tags": filters.tags,
        },
        "index": {
            "path": index.get("config_path", ""),
            "file_count": index.get("file_count", 0),
            "chunk_count": index.get("chunk_count", 0),
        },
        "result_count": len(results),
        "file_summary": summarize_results_by_file(results, top_files),
        "results": items,
    }


def print_search_report_text(report: dict) -> None:
    print(f"Query: {report['query']}")
    if report.get("preset"):
        print(f"Preset: {report['preset']}")
    terms = ", ".join(report.get("query_terms", [])) or "(none)"
    print(f"Query terms: {terms}")
    filters = report.get("filters", {})
    if any(filters.get(key) for key in ("source_types", "path_contains", "title_contains", "tags")):
        print("Filters:")
        if filters.get("source_types"):
            print(f"  source_types: {', '.join(filters['source_types'])}")
        if filters.get("path_contains"):
            print(f"  path_contains: {', '.join(filters['path_contains'])}")
        if filters.get("title_contains"):
            print(f"  title_contains: {', '.join(filters['title_contains'])}")
        if filters.get("tags"):
            print(f"  tags: {', '.join(filters['tags'])}")
    index_meta = report.get("index", {})
    print(
        "Index: "
        f"{index_meta.get('file_count', 0)} files, "
        f"{index_meta.get('chunk_count', 0)} chunks"
    )

    file_summary = report.get("file_summary", [])
    if file_summary:
        print("\nTop files:")
        for item in file_summary:
            citation = f"{item['path']}:{item['line_start']}-{item['line_end']}"
            print(
                f"- score={item['best_score']} hits={item['hit_count']} "
                f"{citation}"
            )
            print(f"  title: {item['best_title']}")
            print(f"  type: {item['source_type']}")

    results = report.get("results", [])
    if not results:
        print("\nNo matches.")
        return

    print("\nTop chunks:")
    for item in results:
        citation = item["citation"]
        line_ref = f"{citation['line_start']}-{citation['line_end']}"
        print(f"{item['rank']}. score={item['score']} {citation['path']}:{line_ref}")
        print(f"   title: {citation['title']}")
        print(f"   type: {citation['source_type']}")
        print(f"   preview: {item['preview']}")
        if "text" in item:
            print("   text:")
            for line in str(item["text"]).splitlines():
                print(f"     {line}")


def citation_for(chunk: dict) -> dict:
    return {
        "path": chunk["path"],
        "line_start": chunk["line_start"],
        "line_end": chunk["line_end"],
        "title": chunk["title"],
        "source_type": chunk["source_type"],
    }


def make_answer_packet(index: dict, query: str, top: int, filters: SearchFilters | None = None) -> dict:
    filters = filters or SearchFilters([], [], [], [])
    results = search(index, query, top, filters)
    evidence = []
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        evidence.append(
            {
                "rank": rank,
                "score": item["score"],
                "citation": citation_for(chunk),
                "text": chunk["text"],
            }
        )

    missing_notes = []
    if not evidence:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if evidence and len(evidence) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")

    return {
        "schema": "cyxwiz.tofix42.phase1a.answer_packet.v1",
        "question": query,
        "preset": "",
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
        "missing_evidence_notes": missing_notes,
    }


def print_answer_packet_markdown(packet: dict) -> None:
    print(f"# Answer Packet\n")
    print(f"Question: {packet['question']}\n")
    if packet.get("preset"):
        print(f"Preset: {packet['preset']}\n")
    print("Mode: retrieval_only")
    print("Model runtime: not_used\n")
    filters = packet.get("filters", {})
    if any(filters.get(key) for key in ("source_types", "path_contains", "title_contains", "tags")):
        print("Filters:")
        if filters.get("source_types"):
            print(f"- source_types: {', '.join(filters['source_types'])}")
        if filters.get("path_contains"):
            print(f"- path_contains: {', '.join(filters['path_contains'])}")
        if filters.get("title_contains"):
            print(f"- title_contains: {', '.join(filters['title_contains'])}")
        if filters.get("tags"):
            print(f"- tags: {', '.join(filters['tags'])}")
        print("")

    evidence = packet.get("evidence", [])
    if not evidence:
        print("No evidence found.")
    else:
        print("## Evidence")
        for item in evidence:
            citation = item["citation"]
            line_ref = f"{citation['line_start']}-{citation['line_end']}"
            preview = preview_text(item["text"], packet["question"], 500)
            print(f"{item['rank']}. score={item['score']} {citation['path']}:{line_ref}")
            print(f"   title: {citation['title']}")
            print(f"   type: {citation['source_type']}")
            print(f"   preview: {preview}")

    notes = packet.get("missing_evidence_notes", [])
    if notes:
        print("\n## Missing Evidence Notes")
        for note in notes:
            print(f"- {note}")


def cmd_check(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}")
        print("Run build first.")
        return 2

    index = load_index(args.index)
    config = load_config(args.config)
    failed = 0
    for check in config["success_checks"]:
        results = search(index, check["query"], top=1)
        if not results:
            print(f"FAIL: {check['query']}")
            print("  no results")
            failed += 1
            continue

        chunk = results[0]["chunk"]
        path_ok = chunk["path"] == check["expected_path"]
        title_ok = chunk["title"] == check["expected_title"]
        status = "PASS" if path_ok and title_ok else "FAIL"
        print(f"{status}: {check['query']}")
        print(f"  got:      {chunk['path']} :: {chunk['title']}")
        print(f"  expected: {check['expected_path']} :: {check['expected_title']}")
        if not (path_ok and title_ok):
            failed += 1

    if failed:
        print(f"{failed} Phase 1A retrieval check(s) failed.")
        return 1

    print("All Phase 1A retrieval checks passed.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}")
        print("Run build first.")
        return 2

    index = load_index(args.index)
    config = load_config(args.config)
    indexed_by_path = {
        item["path"]: item
        for item in index.get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    current_files = [file_metadata(path) for path in iter_initial_files(config["source_patterns"])]
    current_by_path = {item.path: item for item in current_files}

    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    config_changed = index.get("config_hash") != config.get("config_hash")

    for path, current in current_by_path.items():
        indexed = indexed_by_path.get(path)
        if indexed is None:
            added.append(path)
            continue
        if indexed.get("content_hash") != current.content_hash:
            changed.append(path)

    for path in indexed_by_path:
        if path not in current_by_path:
            removed.append(path)

    print(f"Index: {args.index}")
    print(f"Config: {args.config}")
    print(f"Indexed files: {len(indexed_by_path)}")
    print(f"Current files: {len(current_by_path)}")
    print(f"Indexed chunks: {index.get('chunk_count', 'unknown')}")

    if not changed and not added and not removed and not config_changed:
        print("Status: fresh")
        return 0

    print("Status: stale")
    if config_changed:
        print("config changed:")
        print(f"  indexed: {index.get('config_path', '')}")
        print(f"  current: {config.get('config_path', '')}")
    for label, paths in (("changed", changed), ("added", added), ("removed", removed)):
        if paths:
            print(f"{label}:")
            for path in sorted(paths):
                print(f"  {path}")
    return 1


def cmd_build(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    index = build_index(config)
    save_index(index, args.index)
    print(
        f"Wrote {args.index} with {index['file_count']} files and "
        f"{index['chunk_count']} chunks."
    )
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    presets = preset_map(config)
    ordered = [presets[name] for name in sorted(presets)]
    if args.json:
        payload = {
            "schema": "cyxwiz.tofix42.phase1a.saved_queries.v1",
            "preset_count": len(ordered),
            "presets": [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "query": preset.query,
                    "filters": {
                        "source_types": preset.filters.source_types,
                        "path_contains": preset.filters.path_contains,
                        "title_contains": preset.filters.title_contains,
                        "tags": preset.filters.tags,
                    },
                }
                for preset in ordered
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    if not ordered:
        print("No saved query presets.")
        return 0
    print("Saved query presets:")
    for preset in ordered:
        print(f"- {preset.name}")
        print(f"  description: {preset.description or '(none)'}")
        print(f"  query: {preset.query or '(none)'}")
        if has_active_filters(preset.filters):
            print("  filters:")
            if preset.filters.source_types:
                print(f"    source_types: {', '.join(preset.filters.source_types)}")
            if preset.filters.path_contains:
                print(f"    path_contains: {', '.join(preset.filters.path_contains)}")
            if preset.filters.title_contains:
                print(f"    title_contains: {', '.join(preset.filters.title_contains)}")
            if preset.filters.tags:
                print(f"    tags: {', '.join(preset.filters.tags)}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}")
        print("Run build first.")
        return 2
    index = load_index(args.index)
    config = load_config(args.config)
    try:
        query, filters, preset = resolve_query_and_filters(config, args)
    except ValueError as exc:
        print(exc)
        return 2
    report = make_search_report(
        index,
        query,
        args.top,
        args.top_files,
        args.include_full_text,
        filters,
        preset.name if preset else "",
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print_search_report_text(report)
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}")
        print("Run build first.")
        return 2
    index = load_index(args.index)
    config = load_config(args.config)
    try:
        query, filters, preset = resolve_query_and_filters(config, args)
    except ValueError as exc:
        print(exc)
        return 2
    packet = make_answer_packet(index, query, args.top, filters)
    packet["preset"] = preset.name if preset else ""
    if args.json:
        print(json.dumps(packet, indent=2, ensure_ascii=True))
    else:
        print_answer_packet_markdown(packet)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="tofix42 Phase 1A retrieval prototype")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="rebuild the local JSON index").set_defaults(func=cmd_build)

    sub.add_parser("check", help="run Phase 1A success checks").set_defaults(func=cmd_check)

    sub.add_parser("status", help="check whether indexed files changed").set_defaults(func=cmd_status)

    presets_parser = sub.add_parser("presets", help="list saved query presets")
    presets_parser.add_argument("--json", action="store_true")
    presets_parser.set_defaults(func=cmd_presets)

    search_parser = sub.add_parser("search", help="search the local JSON index")
    search_parser.add_argument("query", nargs="?")
    search_parser.add_argument("--preset")
    search_parser.add_argument("--top", type=int, default=5)
    search_parser.add_argument("--top-files", type=int, default=3)
    search_parser.add_argument("--include-full-text", action="store_true")
    search_parser.add_argument("--json", action="store_true", help="emit a structured search report")
    search_parser.add_argument("--source-type", action="append", choices=["source", "markdown", "cyxgraph", "cyxgraph_node", "cyxgraph_links", "text"])
    search_parser.add_argument("--path-contains", action="append")
    search_parser.add_argument("--title-contains", action="append")
    search_parser.add_argument("--tag", action="append")
    search_parser.set_defaults(func=cmd_search)

    packet_parser = sub.add_parser("packet", help="build a retrieval-only answer packet")
    packet_parser.add_argument("query", nargs="?")
    packet_parser.add_argument("--preset")
    packet_parser.add_argument("--top", type=int, default=5)
    packet_parser.add_argument("--json", action="store_true", help="emit packet as JSON")
    packet_parser.add_argument("--source-type", action="append", choices=["source", "markdown", "cyxgraph", "cyxgraph_node", "cyxgraph_links", "text"])
    packet_parser.add_argument("--path-contains", action="append")
    packet_parser.add_argument("--title-contains", action="append")
    packet_parser.add_argument("--tag", action="append")
    packet_parser.set_defaults(func=cmd_packet)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
