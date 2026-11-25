---
title: Plugin and Python problems
description: Fix C++ plugins that do not load or start, plugins that crash, and Python nodes and scripts that do not run.
type: troubleshooting
area: plugins
audience: user
status: stable
since: 1.0.7
owner: plugins
last_reviewed: 2026-11-25
related:
  - ../sdk/guides/troubleshooting.md
  - ../usage/extension_nodes.md
  - install-and-first-launch.md
  - ../security/untrusted-files-and-plugins.md
---

A plugin is a folder with a `plugin.json` manifest and a native library.
The Engine reads the manifest, checks the plugin's API version and build
fingerprint, loads the library and starts the plugin. When a step fails,
the **Plugin Manager** (**Apps > Plugin Manager**) shows the plugin as failed
with the reason, and the Console has a line "PluginManager: Failed to load
plugin from ...: ..." or "Plugin ... did not start: ...".

Messages about plugin and SDK versions, extension nodes in a graph and
Python node bundles are in
[Troubleshooting extension nodes](../sdk/guides/troubleshooting.md). This
page covers the loading steps before them.

## "plugin.json not found in ..." or "Cannot open ..."

**Cause:** the plugin folder has no readable `plugin.json`.

**Fix:** reinstall the plugin, or check that the folder you added holds
`plugin.json` at its top level, not in a subfolder.

## "JSON parse error: ..." or "Manifest missing 'id' field"

**Cause:** `plugin.json` is not valid JSON, or it has no `id`.

**Fix:** if you wrote the plugin, correct the manifest; otherwise ask its
author for a fixed version.

## "API version mismatch: plugin requires ..., engine provides ..."

**Cause:** the plugin was built for another plugin API version than this
Engine provides.

**Fix:** install the plugin's version made for your Engine, or update the
Engine. Plugin authors rebuild against the SDK that ships with the Engine.

## "Plugin library not found: ..."

**Cause:** the manifest names a library file that is not in the folder,
often a plugin built for another operating system.

**Fix:** install the plugin's build for your system.

## "Cannot copy the plugin library for loading: ..."

**Cause:** the Engine loads a copy of each plugin library, so the original
can be rebuilt while the Engine runs, and the copy could not be written.

**Fix:** check free disk space and that your user can write the Engine's
temporary folder, then restart the Engine.

## "Invalid signature: ..."

**Cause:** the plugin is signed, but the library does not match its
signature ("Signed hash does not match DLL hash", "Ed25519 signature
verification failed"). The file was changed after it was signed.

**Fix:** do not use this copy. Download the plugin again from its author.

## "Failed to load DLL: ..."

**Cause:** the operating system could not load the library, usually
because a library it needs is missing, or it was built for another
architecture.

**Fix:** install what the plugin's documentation says it needs (for example
a runtime library), or the plugin's build for your system.

## "Missing required exports: CreatePlugin and/or DestroyPlugin"

**Cause:** the library is not a CyxWiz plugin, or was built without the
SDK's entry points.

**Fix:** plugin authors: export the entry points with the SDK macros. See
[Plugin quickstart](../sdk/get-started/first-cpp-plugin.md).

## "CreatePlugin() threw: ...", "CreatePlugin() returned nullptr", "OnLoad() returned false", "OnInitialize() returned false"

**Cause:** the plugin itself refused to start. Its reason is usually in the
Console right before, under the plugin's own category.

**Fix:** read the plugin's lines in the Console. Report it to the plugin's
author with the log.

## The plugin waits for permission approval

The Console says "PluginManager: Plugin ... awaiting permission approval".

**Cause:** the plugin asks for permissions (for example "Read and write
files on your computer", "Make network connections to remote servers" or
"Run arbitrary Python code in the embedded interpreter"), and you have not
decided yet.

**Fix:** answer the permission dialog. Allow only what you trust the plugin
with; what each permission means is in
[Untrusted files and plugins](../security/untrusted-files-and-plugins.md).

## "Plugin '...' crashed during ...: ..."

**Cause:** the plugin crashed in one of its calls (for example an
"Access Violation", with the exception code on Windows). The Engine catches
the crash and stops the plugin; the rest of the Engine keeps running.

**Fix:** save your work and restart the Engine before you use the plugin
again. Report the crash to the plugin's author with the log. A crash
outside a guarded call closes the Engine; see
[Crash dumps and logs](crash-dumps-and-logs.md).

## "a plugin with the id ... is already loaded"

**Cause:** two plugin folders have the same id, for example an old and a
new version.

**Fix:** remove the older folder from the plugin search paths.

## A Python node or script does not run

Python nodes need a working Python for scripting. Check the start page's
Python chip first; if it does not read **Python ... ready**, see
[Install and first launch problems](install-and-first-launch.md#python-not-found).
Bundle approval, "Python is starting for the Python nodes" and other Python
node messages are in
[Troubleshooting extension nodes](../sdk/guides/troubleshooting.md#python-node-bundles).

## Related

- [Troubleshooting extension nodes](../sdk/guides/troubleshooting.md)
- [Your own nodes](../usage/extension_nodes.md)
- [Untrusted files and plugins](../security/untrusted-files-and-plugins.md)
- [Crash dumps and logs](crash-dumps-and-logs.md)
