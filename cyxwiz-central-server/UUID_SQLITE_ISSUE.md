# UUID/SQLite Type Mismatch Issue

**Date**: 2025-11-17
**Status**: 🔍 DIAGNOSED
**Component**: CyxWiz Central Server - Database Integration Tests

## Problem

After fixing the PostgreSQL enum compatibility issue (commit 9bcf0a1), tests still fail with the same "Job not found" errors. The root cause is a **type mismatch** between the SQLite schema and the Rust model definitions.

### Symptoms

- 7/10 tests FAILING (all tests that query the database)
- 3/10 tests PASSING (UUID validation only, no database queries)
- Error: `Status { code: NotFound, message: "Job not found: <uuid>", source: None }`
- Jobs are being inserted successfully but cannot be queried back

## Root Cause

### SQLite Schema (TEXT)
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    assigned_node_id TEXT,
    ...
);
```

### Rust Model (Uuid)
```rust
pub struct Job {
    pub id: Uuid,                         // ← Expects native Uuid type
    pub assigned_node_id: Option<Uuid>,   // ← Expects native Uuid type
    ...
}
```

### PostgreSQL Schema (UUID)
```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    assigned_node_id UUID,
    ...
);
```

## Why This Fails

SQLx attempts to deserialize query results into the `Job` struct:

```rust
let job = sqlx::query_as::<_, Job>("SELECT * FROM jobs WHERE id = ?")
    .bind(job_id)  // Uuid type
    .fetch_optional(pool)
    .await?;
```

**What happens**:
1. Test inserts job with UUID converted to TEXT (✅ succeeds)
2. Test calls `get_job_by_id()` which does `SELECT * FROM jobs`
3. SQLx tries to deserialize TEXT column into `Uuid` field (❌ fails)
4. Returns `None` from `fetch_optional`
5. Code returns "Job not found" error

## Solutions

### Option 1: Use TEXT in Rust Model for SQLite Tests (Recommended)

**Approach**: Conditionally compile with TEXT IDs for tests, UUID for production.

```rust
#[cfg(test)]
pub struct Job {
    pub id: String,  // TEXT for SQLite tests
    pub assigned_node_id: Option<String>,
    ...
}

#[cfg(not(test))]
pub struct Job {
    pub id: Uuid,  // UUID for PostgreSQL production
    pub assigned_node_id: Option<Uuid>,
    ...
}
```

**Pros**:
- No schema changes needed
- Clear separation between test and production
- Minimal code impact

**Cons**:
- Duplicate struct definitions
- Need to handle UUID/String conversion in tests

### Option 2: Use UUID Type in SQLite Migration

**Approach**: SQLite doesn't have native UUID type, but we can store as BLOB.

```sql
CREATE TABLE jobs (
    id BLOB PRIMARY KEY,  -- Store UUID as 16-byte BLOB
    assigned_node_id BLOB,
    ...
);
```

**Pros**:
- Schema matches Rust model exactly
- No conditional compilation

**Cons**:
- Binary data less readable in SQL queries
- More complex test setup (need to handle BLOB encoding)

### Option 3: Use PostgreSQL for Integration Tests

**Approach**: Use PostgreSQL (via Docker) for tests instead of SQLite.

**Pros**:
- Test environment matches production exactly
- No type mismatch issues
- Most realistic testing

**Cons**:
- Requires Docker/PostgreSQL setup
- Slower test execution
- More complex CI/CD pipeline

### Option 4: Custom Serialization with TEXT (Current Hybrid)

**Approach**: Keep TEXT in SQLite, add custom serialization for Uuid <-> TEXT.

```rust
impl sqlx::Type<sqlx::Sqlite> for Job {
    // Custom UUID TEXT handling
}
```

**Pros**:
- Works with existing schema
- Transparent to application code

**Cons**:
- Complex implementation
- Potential for encoding bugs

## Recommended Solution

**Use Option 1**: Conditional compilation with TEXT for tests.

### Implementation Steps

1. **Create test-specific model** in `src/database/models.rs`:
   ```rust
   #[cfg(test)]
   pub type JobId = String;

   #[cfg(not(test))]
   pub type JobId = Uuid;

   pub struct Job {
       pub id: JobId,
       pub assigned_node_id: Option<JobId>,
       ...
   }
   ```

2. **Update queries** to handle both types transparently

3. **Update tests** to use string UUID representation

4. **Verify** all 10 tests pass

## Current Work Session Summary

**Completed**:
- ✅ Fixed PostgreSQL enum compatibility (removed `type_name` attributes)
- ✅ Tests compile successfully with PROTOC environment variable
- ✅ Diagnosed UUID/TEXT type mismatch issue
- ✅ Documented enum fix in `ENUM_FIX_SUMMARY.md`
- ✅ Committed enum fix (commits 9bcf0a1, dcf9cce)

**In Progress**:
- 🔍 Investigating UUID/SQLite type mismatch solution

**Next Steps**:
1. Implement chosen solution (likely Option 1)
2. Run tests to verify all 10 pass
3. Document the UUID fix
4. Commit and push changes

## Related Files

- **Model Definitions**: `cyxwiz-central-server/src/database/models.rs` (lines 79-112)
- **Query Functions**: `cyxwiz-central-server/src/database/queries.rs` (line 155-163)
- **SQLite Migration**: `cyxwiz-central-server/migrations/20250105000001_initial_schema.sql` (lines 41-74)
- **Integration Tests**: `cyxwiz-central-server/tests/job_status_service_tests.rs`

## Test Execution Command

```bash
cd cyxwiz-central-server
export PROTOC="D:/Dev/CyxWiz_Claude/vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"
cargo test --test job_status_service_tests
```

---

**Investigation**: Complete
**Solution Design**: Outlined (Option 1 recommended)
**Implementation**: Pending
**Testing**: Awaiting fix implementation
**Documentation**: In progress
