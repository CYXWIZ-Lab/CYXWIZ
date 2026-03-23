# CyxWiz Data Studio — Documentation Index

**Last Updated:** 2026-03-19
**Status:** Design Phase Complete, Ready for Implementation

---

## Document Overview

This folder contains comprehensive architectural documentation for **CyxWiz Engine 2.0 with Data Studio integration**. Data Studio is a KNIME-inspired visual data preparation system integrated directly into the CyxWiz Engine.

## Core Documents

### 1. [CyxWiz_DataStudio_UseCases.html](./CyxWiz_DataStudio_UseCases.html)
**Purpose:** Visual reference and use case demonstrations

**Content:**
- Interactive HTML mockup of Data Studio UI
- **Use Case 1:** Data Cleaning Pipeline (real estate dataset)
- **Use Case 2:** Anomaly Detection Pipeline (time-series sensor data)
- Visual examples of all 6 data workflow phases
- Professional UI styling matching CyxWiz design language

**Audience:** Product team, designers, stakeholders

**Key Sections:**
- Phase-by-phase workflow visualization
- Node palette with 50+ data transformation nodes
- Example SQL queries (DuckDB integration)
- Analysis and visualization tabs
- Deploy-to-Node-Editor handoff workflow

---

### 2. [../engine_2.0_architecture.md](../engine_2.0_architecture.md)
**Purpose:** Complete technical architecture specification

**Content:**
- System architecture diagrams
- Component breakdown (DataStudioPanel, PipelineCanvas, QueryEditor, etc.)
- Data flow architecture (raw data → pipeline → training)
- Technology stack and dependencies
- File structure and organization
- Integration points with existing engine
- Backward compatibility strategy
- Testing strategy
- Security considerations

**Audience:** Engineering team, system architects

**Key Sections:**
1. Current Engine Architecture Analysis
2. Data Studio Architecture Design
3. Node Type Catalog (50+ nodes)
4. Technology Stack (DuckDB, ImNodes, ImPlot)
5. File Structure
6. Implementation Phases (11 weeks)
7. Backward Compatibility
8. UI/UX Design
9. Error Handling & Validation
10. Testing Strategy
11. Security & Sandboxing
12. Future Enhancements
13. Open Questions & Decisions

**Document Size:** 600+ lines, ~40 pages

---

### 3. [implementation_roadmap.md](./implementation_roadmap.md)
**Purpose:** Detailed week-by-week implementation guide

**Content:**
- Phase-by-phase breakdown (Phases 1-8)
- Daily task checklists
- Code examples for each feature
- Test case specifications
- Success metrics for each phase
- CMakeLists.txt modifications
- Integration code snippets

**Audience:** Development team, project managers

**Implementation Timeline:**
- **Phase 1 (Weeks 1-2):** Core Infrastructure
- **Phase 2 (Weeks 3-4):** Tabular Transformations
- **Phase 3 (Week 5):** Analysis & Visualization Tabs
- **Phase 4 (Week 6):** DuckDB Query Editor
- **Phase 5 (Week 7):** Node Editor Handoff
- **Phase 6 (Weeks 8-9):** Advanced Nodes
- **Phase 7 (Week 10):** Save/Load & Polish
- **Phase 8 (Week 11):** Performance Optimization

**Total Duration:** 11 weeks (with 2 developers)

---

### 4. [design_patterns_and_edge_cases.md](./design_patterns_and_edge_cases.md)
**Purpose:** Critical implementation patterns and edge case handling

**Content:**
- Design patterns (ImNodes context isolation, ID collision avoidance, etc.)
- Edge case handling (empty datasets, memory limits, cycles, etc.)
- Performance optimization patterns
- Testing strategies
- Error message templates
- Backward compatibility checklist

**Audience:** Implementation engineers, QA team

**Key Patterns:**
1. **Separate ImNodes Context** — Prevents rendering conflicts with ML Node Editor
2. **ID Offset** — Avoids collision between Data Studio and ML nodes
3. **Intermediate Dataset Hiding** — Keeps UI clean
4. **Async Pipeline Execution** — Non-blocking for large datasets
5. **Dataset Lineage Tracking** — Debug failed pipelines

**Edge Cases Covered:**
- Empty datasets
- All-null columns
- Cycle detection in pipeline graphs
- Memory limit exceeded
- Column not found errors
- DuckDB query timeouts
- Dataset deletion during execution
- Multiple output nodes

---

## Quick Start for Developers

### Reading Order (First Time)

1. **Start with UseCases.html** (5 min)
   - Open in browser
   - See visual examples of what we're building
   - Understand the user workflow

2. **Read engine_2.0_architecture.md** (60 min)
   - Section 1: Current Engine Analysis
   - Section 2: Data Studio Architecture
   - Section 3: Node Type Catalog
   - Section 6: Implementation Phases
   - Skim sections 7-14

3. **Review implementation_roadmap.md** (30 min)
   - Phase 1 (first 2 weeks of work)
   - Daily task breakdown
   - Code examples

4. **Bookmark design_patterns_and_edge_cases.md** (reference)
   - Read Section 1 (Critical Design Patterns)
   - Refer to other sections as needed during implementation

**Total Time:** ~2 hours to get up to speed

---

### Implementation Kickoff Checklist

**Day 0 (Pre-Development):**
- [ ] All team members read documentation
- [ ] Architecture review meeting scheduled
- [ ] Development environment setup (DuckDB installed)
- [ ] Git feature branch created (`feature/data-studio`)
- [ ] Project tracking board created (GitHub Projects / Jira)
- [ ] Test data prepared (`test_data/` folder with sample CSVs)

**Day 1 (First Day of Development):**
- [ ] Create directory structure (`src/gui/data_studio/`, etc.)
- [ ] Add Data Studio to CMakeLists.txt
- [ ] Create skeleton classes (DataStudioPanel, PipelineCanvas, etc.)
- [ ] Compile successfully (empty implementations)
- [ ] First commit: "Phase 1: Project structure setup"

**Day 2-10 (Phase 1):**
- Follow `implementation_roadmap.md` → Phase 1 → Day-by-day tasks

---

## Key Technologies

| Technology | Version | Purpose | Documentation |
|------------|---------|---------|---------------|
| **DuckDB** | 1.0.0+ | SQL query engine | https://duckdb.org/docs/ |
| **ImNodes** | 0.5+ | Visual node editor | https://github.com/Nelarius/imnodes |
| **ImPlot** | 0.16+ | Real-time plotting | https://github.com/epezent/implot |
| **ImGui** | 1.89+ | Base UI framework | https://github.com/ocornut/imgui |
| **ArrayFire** | 3.8+ | GPU tensor operations | https://arrayfire.org/docs/ |
| **C++20** | - | Language standard | - |

---

## Success Criteria

### Minimum Viable Product (MVP) — End of Phase 5

**User can:**
1. Import CSV file into Data Studio
2. Build visual data cleaning pipeline (5+ nodes)
3. Execute pipeline and see results
4. Run SQL queries on pipeline output
5. Deploy cleaned dataset to ML Node Editor
6. Train model on cleaned data

**Technical:**
- All Phase 1-5 tests passing (80% coverage)
- No memory leaks (Valgrind clean)
- Pipeline execution < 10s for 100K rows
- UI responsive (no frame drops)

---

### Full Release — End of Phase 8

**Additional Features:**
- 50+ node types implemented
- Save/load pipeline to JSON
- Advanced nodes (Text, Time-Series, Feature Engineering)
- Performance optimization (lazy evaluation, parallel execution)
- Comprehensive error handling
- User documentation

**Performance Targets:**
| Dataset Size | Pipeline (8 nodes) | SQL Query (GROUP BY) |
|--------------|-------------------|---------------------|
| 100K rows | < 5s | < 500ms |
| 1M rows | < 30s | < 3s |
| 10M rows | < 5 min | < 30s |

---

## FAQ

### Q1: Why a separate ImNodes context for Data Studio?

**A:** ImNodes uses global state tied to `ImNodesEditorContext`. If Data Studio and ML Node Editor share the same context, node positions and IDs collide, causing visual corruption. Each editor must have its own context, set active only during its render phase.

**See:** `design_patterns_and_edge_cases.md` → Section 1.1

---

### Q2: How does Data Studio integrate with existing Node Editor?

**A:** Data Studio outputs a cleaned dataset to `DataRegistry`. The `DeployToNodeEditor` node triggers `NodeEditor::SetDatasetFromDataStudio()`, which updates the `DataInput` node to reference the cleaned dataset. User can then build ML model and train.

**See:** `engine_2.0_architecture.md` → Section 2.4.3

---

### Q3: Can users run SQL queries on datasets?

**A:** Yes, via the Query tab. Datasets are registered as in-memory DuckDB tables. Users can write SQL (SELECT, GROUP BY, JOIN, etc.) and results display in a table. Query results can be saved as new datasets.

**See:** `engine_2.0_architecture.md` → Section 2.2.3

---

### Q4: What happens to existing CyxWiz projects?

**A:** Full backward compatibility. Version 1.0 projects load normally (no Data Studio pipeline). Version 2.0 projects include both ML graph and Data Studio pipeline. SaveProject() auto-detects which version to write.

**See:** `engine_2.0_architecture.md` → Section 7

---

### Q5: How long will implementation take?

**A:** 11 weeks with 2 full-time developers. MVP (Phases 1-5) takes 7 weeks. Phases 6-8 add advanced features and polish.

**See:** `implementation_roadmap.md` → All Phases

---

### Q6: What if a node fails during pipeline execution?

**A:** Pipeline stops immediately. Failed node turns red, error message displays in popup. User can view error details, skip the node, or fix parameters and re-run. All upstream nodes remain executed (results cached).

**See:** `design_patterns_and_edge_cases.md` → Section 2

---

### Q7: Can pipelines handle very large datasets (100M+ rows)?

**A:** Yes, via streaming mode. If dataset exceeds memory limit, Data Studio uses chunked loading (load 10K rows at a time, process, discard). Slower but memory-efficient.

**See:** `design_patterns_and_edge_cases.md` → Section 3.3

---

## Contact & Support

**Architecture Questions:**
- Review `engine_2.0_architecture.md` first
- Check "Open Questions & Decisions Needed" (Section 13)
- Discuss in architecture review meetings

**Implementation Questions:**
- Refer to `implementation_roadmap.md` for task breakdown
- Check `design_patterns_and_edge_cases.md` for patterns
- Post in `#cyxwiz-data-studio` Slack channel

**Bug Reports / Edge Cases:**
- Check if covered in `design_patterns_and_edge_cases.md` → Section 2
- If not covered, document in new GitHub issue
- Tag with `data-studio` label

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-19 | Initial architecture documentation complete |
| - | TBD | Phase 1 implementation complete |
| - | TBD | MVP (Phase 5) release |
| - | TBD | Full release (Phase 8) |

---

## Next Steps

1. **Architecture Review Meeting** (Week 0)
   - Present `engine_2.0_architecture.md` to stakeholders
   - Approve technology choices (DuckDB, ImNodes)
   - Finalize timeline and resource allocation

2. **Development Kickoff** (Week 1, Day 1)
   - Assign tasks from Phase 1
   - Create Git feature branch
   - Begin file structure setup

3. **Weekly Progress Reviews** (Every Friday)
   - Review completed work against roadmap
   - Adjust timeline if needed
   - Demo progress to stakeholders

4. **MVP Release** (End of Week 7)
   - Internal beta testing
   - Collect user feedback
   - Plan Phase 6-8 features

5. **Public Release** (End of Week 11)
   - Full documentation
   - User training materials
   - Marketing announcement

---

**Document Status:** Complete and Ready for Implementation
**Maintainer:** CyxWiz Architecture Team
**Last Review:** 2026-03-19
