## 7) Preflight validation and safety gates

`PreflightValidator::Validate` adds a second-tier gate after compile:
- confirms required nodes are not only present but connected properly for training mode,
- validates dataset & label resolution for runtime.
- validates shape/class constraints tied to selected loss/output combination.
- tracks summary fields and issue metadata consumed by UI launch decision.

If compile passes but preflight fails, launch is blocked before heavy runtime setup.

---
