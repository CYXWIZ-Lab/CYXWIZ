# To Fix 80 - Registered Non-Tabular Data Preview Adapters

## Status

Open - Track70 follow-up. This ticket owns the registered text, image, and
audio preview work omitted from the original 75-79 follow-up map.

## Decision statement

Extend the existing bounded, cancellable preview service by modality without
creating a second asset registry, raw-file loader, or Data Input parser path.
Data Input and Asset Browser must preview the same registered dataset identity
through the same service and renderer boundary.

## Current truth

Registered Arrow and Parquet datasets use `DataPreviewRequest`, typed preview
status, bounded paging, cooperative cancellation, page-local metadata, and one
shared tabular renderer. Asset Browser does not directly parse registered
tabular files. Registered text, image, and audio assets do not yet have
equivalent adapters, and image/audio preview requests currently report
unsupported.

## Smallest preview contract

Keep request identity, status, cancellation, and stale-result rejection in the
shared preview service. Add a narrow modality-tagged result so non-tabular
content is not forced into a fake table:

- text: bounded records or lines plus encoding/token metadata;
- image: bounded registered samples or thumbnails plus dimensions, channels,
  and available class metadata; and
- audio: bounded registered samples plus duration, sample rate, channels, and
  a downsampled waveform or envelope.

Adapters read only registered dataset handles. They must not rediscover files,
duplicate format detection, or bypass the authoritative Data Input settings.
Expensive decode or thumbnail work runs through the existing task system and
honors cancellation.

## GUI behavior

- Data Input and Asset Browser dispatch the same preview request for the same
  registered asset.
- The renderer is selected by result modality; tabular rendering remains
  unchanged.
- Paging or sample navigation is bounded and lazy. Closing, replacing, or
  reapplying the source invalidates in-flight results.
- Unsupported registrations and decode failures show typed user-facing reasons
  without freezing the UI.
- Preview is inspection only; it cannot change dataset roles, parsing settings,
  labels, partitions, or cached data.

## Implementation phases

1. Introduce the minimal modality-tagged result and shared dispatch boundary.
2. Implement the registered text adapter and renderer.
3. Implement bounded image thumbnail/sample preview.
4. Implement bounded audio metadata/waveform preview.
5. Connect both GUI entry points and remove any now-redundant placeholder path.

Each adapter is a separate production slice. Do not implement all modalities
in one unreviewable change.

## Acceptance criteria

- The same registered asset produces the same preview identity and metadata in
  Data Input and Asset Browser.
- Text, image, and audio previews read bounded data and remain responsive on
  large sources.
- Cancellation and stale-request tests prevent obsolete results from being
  installed.
- No adapter opens an unregistered raw path or duplicates Data Input parsing.
- Unsupported/corrupt samples fail visibly without crashing or blocking the
  engine.
- Existing tabular preview tests and rendering behavior remain unchanged.

## Non-goals

- a media editor, annotation suite, or full waveform/image processing tool;
- training batch construction or role resolution, which belongs to To Fix 78;
- embedding full media payloads in the asset registry; and
- format support beyond formats already accepted by the corresponding Data
  Input adapter.

## Dependencies

This ticket depends only on the existing registered-asset and preview-task
contracts. It is independent of the To Fix 75-79 training dependency chain.
