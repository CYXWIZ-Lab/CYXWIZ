"""Figures and plots from Python into the Engine (TOFIX134 P5).

The Engine calls install(...) once when Python starts. From then on
matplotlib uses the bundled backend cyxwiz_mpl_backend, so plt.show() in a
script, a notebook cell or the Console sends each open figure to the Engine
(Plot Output, or the cell's outputs) as a PNG; matplotlib itself is imported
only when the user's code imports it. `import cyxwiz` finds the bundled
plot API (cyxwiz.py), whose calls reach the Engine through plot sinks here.
"""

import importlib.abc
import importlib.util
import os
import sys

BACKEND = 'cyxwiz_mpl_backend'
# Bundled modules found by name without putting the tools folder on sys.path.
_MODULES = (BACKEND, 'cyxwiz')

_sink = None
_plot_sink = None   # (request_json, [(name, 'num' | 'text', values)]) -> problem text or ''
_close_sink = None  # (title) -> None


def emit(png, width, height, title):
    """Hand one rendered figure to the Engine (no-op before install)."""
    if _sink is not None:
        _sink(png, width, height, title)


def close_plot(title):
    """Close the Engine Plot window with this title (cyxwiz.close)."""
    if _close_sink is not None:
        _close_sink(title)


class _BackendFinder(importlib.abc.MetaPathFinder):
    """Finds the bundled modules next to this file (the matplotlib backend,
    the cyxwiz plot API), without putting the tools folder on sys.path. It
    comes last, so a user's own module of the same name wins."""

    def __init__(self, folder):
        self._folder = folder

    def find_spec(self, name, path=None, target=None):
        if name not in _MODULES:
            return None
        return importlib.util.spec_from_file_location(name, os.path.join(self._folder, name + '.py'))


def install(sink, plot_sink=None, close_sink=None):
    """Route figures to `sink(png_bytes, width, height, title)` and cyxwiz
    plots to `plot_sink` / `close_sink`."""
    global _sink, _plot_sink, _close_sink
    _sink = sink
    _plot_sink = plot_sink
    _close_sink = close_sink
    here = os.path.dirname(os.path.abspath(__file__))
    if not any(isinstance(f, _BackendFinder) for f in sys.meta_path):
        sys.meta_path.append(_BackendFinder(here))
    os.environ['MPLBACKEND'] = 'module://' + BACKEND
    # matplotlib already imported (a restarted session keeps modules): switch now.
    if 'matplotlib' in sys.modules:
        try:
            sys.modules['matplotlib'].use('module://' + BACKEND, force=True)
        except Exception:
            pass
