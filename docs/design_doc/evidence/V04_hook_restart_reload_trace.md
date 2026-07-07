# V-04 — Plugin Hook Restart/Reload Trace

## Fixture

- `fixture_id`:
- `plugin_ids`:
- `date`:
- `environment`:

## Evidence fields

- plugin registration callback list (before reload)
- plugin callback list (after reload/restart)
- unload/reload ordering log
- duplicate callback checks

## Notes

- Populate with matrix of lifecycle transitions and callback order for each plugin.
