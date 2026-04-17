# Node Registration Unification — Plan

**Filed:** 2026-04-17
**Source:** Plan agent
**Scope:** Make `NodeMetadataRegistry` the single source of truth; search palette + context menu read from it
**Size:** 3 commits, ~1-2 hours total

## Problem

Adding a new node type requires editing 3 hand-maintained lists that can drift:

| Consumer | Source | Format |
|---|---|---|
| Node Browser panel | `core/node_metadata_registry.cpp::Initialize*Nodes()` + JSON templates | Rich `NodeMetadata` struct |
| Search palette (Ctrl+Space) | `gui/node_editor_add_search.cpp::BuildSearchableNodes()` | `addNode(type, name, category, keywords)` calls |
| Right-click context menu | `gui/node_editor_context_menu.cpp` ~line 447 | `{NodeType::X, "Display Name"}` entries |

Surfaced when BarChart shipped today: browser was empty because the first list
was missed. Plus a related 4th list: `ShouldShowOpenDialogButton` whitelist in
`node_config_dialog.cpp:970`. Plus dual `StringToNodeType` maps in
`node_editor_io.cpp:20` + `pattern_library.cpp:331`.

## NodeMetadata Field Coverage

`NodeMetadata` struct at `core/node_metadata.h:50-83` already carries
everything both consumers need. No new fields required.

## Category Mapping (add_search → NodeCategory)

Most existing add_search categories map 1:1 to `NodeCategory` enum.
Exceptions:

- "Image Transforms" → no enum, keyword-only (route under Preprocessing/DataPipeline)
- "Layers > Dense/Linear", "Layers > Convolution", etc. → flatten to existing enum (cosmetic regression)
- "Attention", "Transformer" → merge into `Attention`
- "Loss Functions", "Optimizers", "Schedulers" → rolled into `Training`
- "Composite" (Subgraph only) → map to `Utility`
- Template-only buckets (AutoML, Privacy, Optimization) → already routed via JSON loader's fallback to Utility/Explainability

## Migration — 3 Commits

### Commit 1 — Search palette reads registry

File: `cyxwiz-engine/src/gui/node_editor_add_search.cpp`

Replace the body of `InitializeSearchableNodes` (lines 79-421) with:

```cpp
auto& reg = cyxwiz::NodeMetadataRegistry::Instance();
if (!reg.IsInitialized()) reg.Initialize();
for (const auto& cat : reg.GetCategories()) {
    for (const auto* meta : reg.GetByCategory(cat, /*include_templates=*/true)) {
        SearchableNode n;
        n.type = meta->type;
        n.name = meta->name;
        n.category = cyxwiz::GetCategoryDisplayName(meta->category);
        n.keywords = join(meta->keywords, " ");
        n.status = meta->status;
        n.description = meta->brief_description;
        n.tooltip = meta->help_text;
        all_searchable_nodes_.push_back(std::move(n));
    }
}
```

Keep the existing plugin-node branch (lines 310-326) as-is.
Delete the ~400 hand-coded `addNode`/`addTemplateNode` calls.

### Commit 2 — Context menu reads registry

File: `cyxwiz-engine/src/gui/node_editor_context_menu.cpp`

Replace hand-built `nodes_by_category_` init (lines 374-475) with:

```cpp
auto& reg = cyxwiz::NodeMetadataRegistry::Instance();
if (!reg.IsInitialized()) reg.Initialize();
for (const auto& cat : reg.GetCategories()) {
    for (const auto* meta : reg.GetByCategory(cat, /*include_templates=*/false)) {
        nodes_by_category_[cat].push_back({meta->type, meta->name});
    }
}
nodes_by_category_initialized_ = true;
```

Context menu auto-expands to every enum category with entries. Several new
submenus surface (Pooling, Normalization, Training, Attention, etc.) — this
is a feature gain.

### Commit 3 — Cleanup + document invariant

- Remove the transitional `addNode` helper.
- Unify `GetCategoryName` (context menu) and `GetCategoryDisplayName` (registry).
- Add note to CLAUDE.md: "Adding a node type → edit `NodeMetadataRegistry::Initialize<Cat>Nodes()` only. Search palette and context menu read from it automatically."
- Remove the tofix entry.
- Add a smoke assert: `assert(reg.GetMetadata(NodeType::BarChart) != nullptr)` on a representative recent node.

## Risks

- **Sort order change** in search palette: `GetByCategory` sorts by `usage_count` desc then `name` asc. Today it's insertion order. Argue as UX improvement.
- **Display-name drift**: today's 3 sources (registry `name`, addNode display, context-menu display) can disagree. Migration forces single canonical spelling from registry — audit a few to verify.
- **Category ordering**: `reg.GetCategories()` returns `category_order_` order (sensible); context menu's hand-coded order was arbitrary.
- **Hierarchical sublabels** ("Layers > Convolution") lost as rendered subtext. Not load-bearing for search.
- **Registry init timing**: `reg.Initialize()` is idempotent + mutex-guarded; safe to call lazily from `ShowNodeAddSearch`.
- **BarChart regression test**: verify a fresh node with only a registry entry (no `addNode`/context-menu edit) appears in all three UIs.

## Bundle Decision — Defer

Separate follow-ups for:
- `ShouldShowOpenDialogButton` whitelist → derive from
  `NodeConfigDialogFactory::HasDialog(type)`. Requires auditing current
  whitelist entries to confirm they all have registered factories.
- `StringToNodeType` dual maps → requires reverse `name → type` on registry
  with alias handling (e.g. `"BatchNorm2D"` vs `BatchNorm`).

Note both in the CLAUDE.md note: "Known remaining dual-maintained lists —
same unification pattern applies."

## CMake

No changes. All touched files already in engine target.

## Ambiguities

- Whether to add `NodeCategory::Composite` enum value (for Subgraph) vs map
  to Utility. Recommend map to Utility for minimum-impact.
- Plugin nodes stay out of `NodeMetadataRegistry` (separate `PluginNodeRegistry`);
  the search-palette loop appends plugin nodes after the metadata-registry loop.
