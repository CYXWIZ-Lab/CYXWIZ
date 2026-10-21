# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting or security-advisory feature for the [CYXWIZ repository](https://github.com/CYXWIZ-Lab/CYXWIZ/security/advisories/new). Include:

- affected revision and platform;
- impact and realistic attack scenario;
- reproduction steps or proof of concept;
- suggested mitigation, if known.

Do not include real credentials, personal data, proprietary datasets, or destructive payloads. We will acknowledge a complete report through the private channel and coordinate validation and disclosure there.

## Supported versions

CyxWiz is currently pre-release. Security fixes are applied to the active development branch; no stable-version support window is promised yet.

## Security boundaries

Treat graphs, scripts, plugins, models, datasets, checkpoints, and remote job payloads as untrusted input. Run only material you trust and isolate evaluation environments appropriately. Never commit API keys or production credentials to this repository.
