# ImPlot3D (vendored)

- Upstream: https://github.com/brenocq/implot3d, tag v0.2 (MIT, see LICENSE).
- Archive SHA-512: 163aeb62d7d4bd4cac0ea0bad26b4d2dd399ac078cfa6fb414b969006ef3683c3865f5db322fd8d46d7b74e32d7492cd0574fbf30fcd6ac5696f1f1d04e0f7cb
  (the same as the vcpkg port in the pinned baseline 594ad887).
- Files: the library only (no demo, no examples).
- One change (marked "CyxWiz:" in implot3d.cpp, HandleInput): the mouse buttons for turning and
  panning are swapped, so a left drag turns the box and a right drag pans, as in matplotlib and
  other 3D plot tools (owner check, TOFIX134 P4). Double-click left still fits, right resets the turn.
- Why vendored: build trees share one vcpkg install with manifest install off (TOFIX134 P4,
  decision D1 "ImPlot3D via vcpkg or vendored").
