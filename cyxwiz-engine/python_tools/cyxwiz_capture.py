"""Figures from Python into the Engine (TOFIX134 P5.2).

The Engine calls install(sink) once when Python starts. From then on
matplotlib uses the bundled backend cyxwiz_mpl_backend, so plt.show() in a
script, a notebook cell or the Console sends each open figure to the Engine
(Plot Output, or the cell's outputs) as a PNG. matplotlib itself is imported
only when the user's code imports it.
"""

import importlib.abc
import importlib.util
import os
import sys

BACKEND = 'cyxwiz_mpl_backend'

_sink = None


def emit(png, width, height, title):
    """Hand one rendered figure to the Engine (no-op before install)."""
    if _sink is not None:
        _sink(png, width, height, title)


class _BackendFinder(importlib.abc.MetaPathFinder):
    """Finds the backend module next to this file for matplotlib, without
    putting the tools folder on sys.path (user modules keep their names)."""

    def __init__(self, path):
        self._path = path

    def find_spec(self, name, path=None, target=None):
        if name != BACKEND:
            return None
        return importlib.util.spec_from_file_location(name, self._path)


def install(sink):
    """Route figures to `sink(png_bytes, width, height, title)`."""
    global _sink
    _sink = sink
    here = os.path.dirname(os.path.abspath(__file__))
    if not any(isinstance(f, _BackendFinder) for f in sys.meta_path):
        sys.meta_path.append(_BackendFinder(os.path.join(here, BACKEND + '.py')))
    os.environ['MPLBACKEND'] = 'module://' + BACKEND
    # matplotlib already imported (a restarted session keeps modules): switch now.
    if 'matplotlib' in sys.modules:
        try:
            sys.modules['matplotlib'].use('module://' + BACKEND, force=True)
        except Exception:
            pass
