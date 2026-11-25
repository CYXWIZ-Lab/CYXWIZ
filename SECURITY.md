# Security Policy

## Reporting a vulnerability

Do not open a public issue, discussion or pull request for a suspected vulnerability.

Report it privately by email to **contact@cyxwiz.ai**, with a subject that starts with `Security:` (for example `Security: plugin loader runs unsigned code`). Include:

- the affected component (Engine, Server Node, installer, plugin SDK, protocol) and version or commit;
- the operating system and, where it matters, the compute device;
- the impact and a realistic attack scenario;
- reproduction steps or a minimal proof of concept;
- a suggested mitigation, if you have one.

Do not include real credentials, personal data, proprietary datasets, or destructive payloads. We acknowledge a complete report by email and coordinate validation, the fix and disclosure with you in the same private thread. Please give us a reasonable time to ship a fix before you disclose publicly.

More detail on what to report and how reports are handled: [Report a vulnerability](docs/security/report-a-vulnerability.md).

## Supported versions

CyxWiz is in alpha. Only the latest release gets security fixes; a fix ships as a new release, and older releases are not patched. Update to the latest release before you report, and say if the issue is also in the current source.

## Security boundaries

Treat graphs, scripts, Python nodes, plugins, models, datasets, checkpoints, and remote job payloads as untrusted input: opening or running them can run code with your user's permissions. Run only material you trust and isolate evaluation environments appropriately. Never commit API keys or production credentials to this repository.

What each of these can do, and how to harden a Server Node: [Security](docs/security/index.md).
