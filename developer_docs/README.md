# Developer Documentation Index

This directory contains all development documentation for the CyxWiz project, organized by category.

---

## 📁 Folder Structure

### 🏗️ [build/](build/)
Build system documentation, reports, and troubleshooting guides.

**Files:**
- `BUILDING.md` - How to build the project
- `BUILD_REPORT_2025-11-06.md` - Build verification report
- `BUILD_SUCCESS_REPORT.md` - Successful build milestone
- `BUILD_TEST_REPORT.md` - Build testing results
- `FIXES_SUMMARY.md` - Summary of build fixes
- `HEADER_REFACTORING_SUMMARY.md` - Header cleanup documentation

---

### 🎮 [cuda/](cuda/)
CUDA integration guides and setup documentation.

**Files:**
- `CUDA_QUICKSTART.md` - Quick start guide for CUDA
- `CUDA_SETUP_GUIDE.md` - Detailed CUDA setup instructions
- `CUDA_INTEGRATION_STATUS.md` - Current CUDA integration status
- `CUDA_INTEGRATION_SUMMARY.md` - Summary of CUDA features
- `CUDA_INTEGRATION_FINAL_REPORT.md` - Final CUDA integration report

---

### 🔒 [phase3/](phase3/)
Phase 3 documentation: Python Sandbox Security

**Files:**
- `PHASE3_COMPLETION.md` - Phase 3 completion summary
- `PHASE3_SANDBOX_README.md` - Sandbox feature documentation
- `PHASE3_SERVER_NODE_GUIDE.md` - Server Node implementation guide
- `PHASE3_TEST_GUIDE.md` - Testing guide for Phase 3

---

### 📊 [phase4/](phase4/)
Phase 4 documentation: Plotting and Visualization

**Files:**
- `PHASE4_NOTES.md` - Phase 4 development notes

---

### ⚡ [phase5/](phase5/)
Phase 5 documentation: Advanced Features (Auto-completion, File Handlers, Templates)

**Files:**
- `PHASE5_PLAN.md` - Phase 5 planning document
- `PHASE5_AUTOCOMPLETE_GUIDE.md` - Auto-completion implementation guide
- `PHASE5_FILE_HANDLERS_GUIDE.md` - File format handler documentation
- `PHASE5_2_COMPLETE.md` - Phase 5 Task 2 completion
- `PHASE5_3_COMPLETE.md` - Phase 5 Task 3 completion
- `PHASE5_3_PLAN.md` - Phase 5 Task 3 planning
- `PHASE5_COMPLETION_SUMMARY.md` - Phase 5 overall summary
- `PHASE5_SESSION_SUMMARY.md` - Phase 5 session notes
- `PHASE5_TEST_REPORT.md` - Phase 5 testing report
- `test_phase5.md` - Phase 5 testing procedures

---

### 🛡️ [sandbox/](sandbox/)
Python Sandbox security documentation, fixes, and use cases.

**Files:**
- `SANDBOX_PURPOSE_AND_USE_CASES.md` - Why sandbox exists, when to use it
- `SANDBOX_IMPORT_FIX.md` - Import hook fix documentation
- `CRITICAL_BUG_FIX.md` - Critical sandbox bug fix report
- `TESTING_GUIDE.md` - Sandbox testing guide

---

### 📜 [scripting/](scripting/)
Python scripting system documentation.

**Files:**
- `SCRIPT_README.md` - Scripting system overview
- `SECTION_EXECUTION_GUIDE.md` - Section execution feature guide
- `STARTUP_SCRIPTS_DESIGN.md` - Startup scripts architecture
- `TXT_FILE_SUPPORT.md` - TXT file format support

---

### 🎨 [ui-ux/](ui-ux/)
User interface and experience documentation.

**Files:**
- `DOCKING_LAYOUT_CODE_REFERENCE.md` - Docking system code reference
- `DOCKING_LAYOUT_IMPLEMENTATION.md` - Docking implementation guide
- `PLOTTING_SYSTEM.md` - Plotting system documentation
- `TUI_ENHANCEMENT_PLAN.md` - Terminal UI enhancement plan

---

### 🏛️ [architecture/](architecture/)
Project architecture, design decisions, and technical discussions.

**Files:**
- `PROJECT_STRUCTURE.md` - Overall project structure
- `project_overview.md` - High-level project overview
- `ORGANIZATION_PROFILE.md` - Organization and team structure
- `ARRAYFIRE_VS_NUMPY_DISCUSSION.md` - ArrayFire vs NumPy architecture discussion

---

### 🖥️ [server-node/](server-node/)
Server Node component documentation and troubleshooting.

**Files:**
- `SERVER_NODE_BUILD_STATUS.md` - Server Node build status
- `SERVER_NODE_LINKER_ISSUE.md` - Linker issue troubleshooting

---

### 📋 [planning/](planning/)
Project planning, roadmaps, and next steps.

**Files:**
- `NEXT_STEPS.md` - Next development steps
- `RUNNING_SERVICES.md` - Running services documentation

---

## 🔍 Quick Reference

### I want to...

**Build the project**
→ Read [build/BUILDING.md](build/BUILDING.md)

**Set up CUDA**
→ Read [cuda/CUDA_QUICKSTART.md](cuda/CUDA_QUICKSTART.md)

**Understand the sandbox**
→ Read [sandbox/SANDBOX_PURPOSE_AND_USE_CASES.md](sandbox/SANDBOX_PURPOSE_AND_USE_CASES.md)

**Learn about Phase 5 features**
→ Read [phase5/PHASE5_COMPLETION_SUMMARY.md](phase5/PHASE5_COMPLETION_SUMMARY.md)

**Understand project architecture**
→ Read [architecture/PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md)

**Fix build issues**
→ Read [build/FIXES_SUMMARY.md](build/FIXES_SUMMARY.md)

**Test the sandbox**
→ Read [sandbox/TESTING_GUIDE.md](sandbox/TESTING_GUIDE.md)

**Work on scripting features**
→ Read [scripting/SCRIPT_README.md](scripting/SCRIPT_README.md)

**Implement UI features**
→ Read [ui-ux/DOCKING_LAYOUT_IMPLEMENTATION.md](ui-ux/DOCKING_LAYOUT_IMPLEMENTATION.md)

---

## 📊 Documentation Statistics

Total documentation files: **46**
- Build docs: 6
- CUDA docs: 5
- Phase 3 docs: 4
- Phase 4 docs: 1
- Phase 5 docs: 10
- Sandbox docs: 4
- Scripting docs: 4
- UI/UX docs: 4
- Architecture docs: 4
- Server Node docs: 2
- Planning docs: 2

---

## 📝 Contributing to Documentation

When adding new documentation:

1. **Choose the right folder:**
   - Build-related → `build/`
   - Feature-specific → `phase3/`, `phase4/`, `phase5/`
   - Security-related → `sandbox/`
   - Architecture/design → `architecture/`
   - UI/UX → `ui-ux/`
   - Planning → `planning/`

2. **Use clear naming:**
   - Use UPPER_CASE for important docs
   - Use descriptive names: `FEATURE_GUIDE.md` not `doc1.md`
   - Include dates for reports: `REPORT_2025-11-17.md`

3. **Update this index:**
   - Add your file to the appropriate section
   - Update the statistics
   - Add to "Quick Reference" if helpful

---

## 🗂️ Root Directory Files

These files remain in the root directory:

- **README.md** - Main project README (for users and contributors)
- **CLAUDE.md** - AI assistant instructions (stays in root for easy access)

---

## 🔗 Related Documentation

For component-specific documentation, see:
- `cyxwiz-engine/README.md` - Engine client documentation
- `cyxwiz-backend/README.md` - Backend library documentation
- `cyxwiz-server-node/README.md` - Server Node documentation
- `cyxwiz-central-server/README.md` - Central Server documentation
- `cyxwiz-protocol/README.md` - Protocol definitions

---

**Last Updated:** 2025-11-17
