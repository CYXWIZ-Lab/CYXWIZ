## 11) Error model and observability

Error progression layers:
1. Compile-time issues (`CompileResult`)
2. Preflight issues (readiness + graph semantics)
3. Runtime errors (executor exceptions)

User-facing behavior:
- compile popup and issue list render severity context,
- runtime logs/callbacks provide streaming progress and terminal failure details.
- compile/preflight are intended to prevent expensive runtime failures.

---
