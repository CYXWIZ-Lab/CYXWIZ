# CPU-only alpha assembly

Use the existing `assemble_installer_alpha_release.py` command with
`--cpu-only` to assemble an alpha with one or more signed CPU bases for every
supported target: Windows x64, Linux x64, macOS Intel, and macOS Apple Silicon.
Supply the normal manifest, native installer stage, native setup package,
trust-root, signing-key, version, and immutable hosting arguments.

CPU-only mode rejects all optional backend packs. The release inventory signs
the explicit kind `cyxwiz-alpha-cpu-release-assets`; the publication verifier
uses that authenticated mode to require all four bases without requiring
optional packs. All archive hashes, signatures, trust roles, setup/installer
targets, catalog URLs, and exact uploaded asset checks remain enforced.

Without `--cpu-only`, the existing `cyxwiz-alpha-release-assets` contract still
requires a base plus matching optional pack on every platform. Legacy signed
inventories continue to validate under that full-matrix policy. Older verifier
versions reject the new CPU-only kind; deploy the updated publication verifier
before publishing a CPU-only alpha.

This mode does not generate signing authority, qualify GPU routes, upload
assets, or turn ordinary development setup binaries into release-configured
installers. Use release-configured native setup/installer artifacts and the
approved alpha signing keys. Keep private keys outside source control.
