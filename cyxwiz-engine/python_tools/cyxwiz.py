"""Plots in CyxWiz Engine windows from Python, like pyplot (TOFIX134 P5, board 18).

    import cyxwiz as cx
    cx.scatter("artist_popularity", "track_popularity", data=df, color="album_type", title="Popularity")
    cx.hist(df.track_duration_min, bins=40)

Each call opens an Engine Plot window (the same window as a Plot node's: the
type, columns and colours can be changed there) and lists it in Plot Output.
Calling again with the same title updates that window. Columns are names in
`data=` (a pandas / Polars DataFrame or a dict) or arrays, lists and Series.

Common options: title, xlabel, ylabel, color (a column), bins, logx, logy,
legend, density, smooth, rows (plot the first N rows); any other Plot window
setting by its saved name (bar_layout="stacked", scale="viridis", ...).
"""

import json
import math
import os
import sys
from array import array

__all__ = [
    'plot', 'line', 'scatter', 'bar', 'hist', 'area', 'step', 'stem', 'pie', 'polar', 'errorbar',
    'box', 'boxplot', 'violin', 'violinplot', 'kde', 'parallel', 'pairplot',
    'heatmap', 'matshow', 'hist2d', 'hexbin', 'contour', 'contourf',
    'quiver', 'streamplot', 'imshow',
    'confusion_matrix', 'roc_curve', 'pr_curve', 'calibration', 'residuals', 'learning_curve', 'importance',
    'sankey', 'treemap', 'map_points', 'map_regions',
    'scatter3d', 'plot3d', 'plot_surface', 'plot_trisurf',
    'network', 'tree', 'close', 'Plot',
]

_OPTION_NAMES = {'xlabel': 'x_label', 'ylabel': 'y_label', 'logx': 'log_x', 'logy': 'log_y'}


class Plot:
    """The window a call opened: update() plots again into it, close() closes it."""

    def __init__(self, title, function, args, kwargs):
        self.title = title
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def update(self, *args, **kwargs):
        merged = dict(self._kwargs)
        merged.update(kwargs)
        merged['title'] = self.title
        return self._function(*(args or self._args), **merged)

    def close(self):
        close(self.title)

    def __repr__(self):
        return 'cyxwiz.Plot(%r)' % self.title


def close(title):
    """Close the Engine Plot window with this title."""
    _capture().close_plot(str(title))


# ---- columns -----------------------------------------------------------------

def _capture():
    capture = sys.modules.get('cyxwiz_capture')
    if capture is None or getattr(capture, '_plot_sink', None) is None:
        raise RuntimeError('cyxwiz plots open in the CyxWiz Engine (run this in its Script Editor, a notebook or the Console)')
    return capture


def _values(obj):
    """('num', array of doubles) or ('text', list of str) for one column."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        try:
            a = np.asarray(obj)
        except Exception:
            a = None
        if a is not None and a.ndim == 1 and a.dtype.kind != 'O':  # object arrays: checked item by item below
            if a.dtype.kind in 'biuf':
                out = array('d')
                out.frombytes(np.ascontiguousarray(a, dtype=np.float64).tobytes())
                return 'num', out
            if a.dtype.kind == 'M':  # dates: as text, like the table views show them
                return 'text', [str(v) for v in a.astype(str)]
            return 'text', [str(v) for v in a.tolist()]
    items = list(obj)
    if all(isinstance(v, (int, float, bool)) or v is None or type(v).__name__.startswith(('float', 'int')) for v in items):
        return 'num', array('d', (float('nan') if v is None else float(v) for v in items))
    return 'text', ['' if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in items]


def _table(data):
    """Column access for data=: a DataFrame (pandas / Polars) or a mapping."""
    if data is None:
        return None
    if hasattr(data, 'columns') and hasattr(data, '__getitem__'):
        return data
    if isinstance(data, dict):
        return data
    raise TypeError('data= takes a DataFrame or a dict of columns')


class _Columns:
    def __init__(self, data):
        self.data = _table(data)
        self.names = []
        self.columns = []
        self.length = None

    def add(self, obj, default):
        """A column for one argument: a name in data=, or values. Returns its name."""
        if obj is None:
            return ''
        if isinstance(obj, str):
            if self.data is None:
                raise ValueError("'%s' is a column name: pass the table as data=" % obj)
            try:
                values = self.data[obj]
            except Exception:
                cols = list(getattr(self.data, 'columns', self.data.keys() if isinstance(self.data, dict) else []))
                raise ValueError("column '%s' is not in data= (columns: %s)" % (obj, ', '.join(map(str, cols[:12]))))
            name = obj
        else:
            values = obj
            name = getattr(obj, 'name', None)
            name = str(name) if name not in (None, '') else default
        if name in self.names:
            if obj is self._source_of(name):
                return name
            base, n = name, 2
            while '%s_%d' % (base, n) in self.names:
                n += 1
            name = '%s_%d' % (base, n)
        kind, vals = _values(values)
        if self.length is None:
            self.length = len(vals)
        elif len(vals) != self.length:
            raise ValueError("'%s' has %d values, the other columns %d" % (name, len(vals), self.length))
        self.names.append(name)
        self.columns.append((name, kind, vals, obj))
        return name

    def _source_of(self, name):
        for n, _, _, obj in self.columns:
            if n == name:
                return obj
        return None


def _caller():
    """The user's file and line: the first frame outside this module."""
    here = os.path.abspath(__file__)
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if os.path.abspath(name) != here:
            # Code run from the editor or the Console has no file ("<string>").
            return ('' if name.startswith('<') else name), frame.f_lineno
        frame = frame.f_back
    return '', 0


def _send(kind, label, cols, spec, options, source):
    title = options.pop('title', None)
    spec['version'] = 1
    spec['kind'] = kind
    for key, value in list(options.items()):
        if value is None:
            continue
        if key == 'color':
            spec['color'] = cols.add(value, 'color')
        elif key == 'rows':
            spec['rows'] = {'mode': 'first', 'first': int(value)}
        else:
            spec[_OPTION_NAMES.get(key, key)] = value
    if not title:
        used = [spec.get('x', '')] + list(spec.get('y', [])) + [spec.get('z', ''), spec.get('value', '')]
        title = label + ': ' + ' / '.join(u for u in used if u)
    spec['title'] = str(title)
    filename, line = _caller()
    request = {'spec': spec, 'columns': cols.names, 'source': source, 'file': filename, 'line': line}
    capture = _capture()
    columns = [(name, kind_, vals) for name, kind_, vals, _ in cols.columns]
    problem = capture._plot_sink(json.dumps(request), columns)
    if problem:
        raise ValueError(problem)
    return spec['title']


def _source(data):
    """How the data is named in the caller: the variable that holds data=."""
    if data is None:
        return 'arrays'
    here = os.path.abspath(__file__)
    frame = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename) == here:
        frame = frame.f_back
    if frame is not None:
        for name, value in frame.f_locals.items():
            if value is data and not name.startswith('_'):
                return name
    return 'data'


def _kind(kind, label, layout):
    """One public function: `layout` says what the positional arguments are."""

    def function(*args, data=None, **options):
        original = dict(options)
        cols = _Columns(data)
        spec = {}
        if layout == 'xy':  # plot(y) or plot(x, y, y2, ...)
            if len(args) == 1:
                spec['y'] = [cols.add(args[0], 'y')]
            elif len(args) >= 2:
                spec['x'] = cols.add(args[0], 'x')
                spec['y'] = [cols.add(a, 'y%d' % i if i else 'y') for i, a in enumerate(args[1:])]
        elif layout == 'x':  # bar(categories, values=None), hist(values)
            if args:
                spec['x'] = cols.add(args[0], 'x')
            if len(args) > 1:
                spec['y'] = [cols.add(args[1], 'y')]
        elif layout == 'ys':  # box(a, b, ...)
            spec['y'] = [cols.add(a, 'y%d' % (i + 1)) for i, a in enumerate(args)]
        elif layout == 'xyz':
            for key, a in zip(('x', 'y', 'z'), args):
                if key == 'y':
                    spec['y'] = [cols.add(a, 'y')]
                else:
                    spec[key] = cols.add(a, key)
        elif layout == 'xyv':  # heatmap(x, y, value=None)
            if args:
                spec['x'] = cols.add(args[0], 'x')
            if len(args) > 1:
                spec['y'] = [cols.add(args[1], 'y')]
            if len(args) > 2:
                spec['value'] = cols.add(args[2], 'value')
        elif layout == 'xyuv':
            for key, a in zip(('x', 'y', 'u', 'v'), args):
                if key == 'y':
                    spec['y'] = [cols.add(a, 'y')]
                else:
                    spec[key] = cols.add(a, key)
        elif layout == 'steps':  # sankey(step1, step2, ..., value=None)
            spec['y'] = [cols.add(a, 'level%d' % (i + 1)) for i, a in enumerate(args)]
        elif layout == 'xys':  # learning_curve(x, y1, y2, ...)
            if args:
                spec['x'] = cols.add(args[0], 'x')
            spec['y'] = [cols.add(a, 'y%d' % (i + 1)) for i, a in enumerate(args[1:])]
        for key in ('value', 'weight', 'C', 'values', 'size'):
            if key in options:
                spec['value'] = cols.add(options.pop(key), 'value')
        title = _send(kind, label, cols, spec, options, _source(data))
        return Plot(title, function, args, dict(original, data=data) if data is not None else original)

    function.__name__ = label
    return function


def _surface(*args, data=None, **options):
    """plot_surface(Z) with a 2D array of heights (rows = Y, columns = X), or
    plot_surface(x, y, z) with one point per row."""
    original = dict(options)
    if len(args) == 1 and data is None:
        try:
            import numpy as np
            grid = np.asarray(args[0], dtype=float)
        except Exception:
            grid = None
        if grid is not None and grid.ndim == 2:
            columns = {'c%d' % (i + 1): grid[:, i] for i in range(grid.shape[1])}
            cols = _Columns(columns)
            spec = {'surface_from': 'grid', 'y': [cols.add(n, n) for n in columns]}
            title = _send('surface', 'plot_surface', cols, spec, options, 'array %dx%d' % grid.shape)
            return Plot(title, _surface, args, original)
    cols = _Columns(data)
    spec = {'surface_from': 'xyz'}
    for key, a in zip(('x', 'y', 'z'), args):
        if key == 'y':
            spec['y'] = [cols.add(a, 'y')]
        else:
            spec[key] = cols.add(a, key)
    title = _send('surface', 'plot_surface', cols, spec, options, _source(data))
    return Plot(title, _surface, args, dict(original, data=data) if data is not None else original)


def _imshow(*args, data=None, **options):
    """imshow(image) with a 2D array (one image) or a table of pixel columns."""
    original = dict(options)
    if len(args) == 1 and data is None:
        try:
            import numpy as np
            img = np.asarray(args[0], dtype=float)
        except Exception:
            img = None
        if img is not None and img.ndim in (2, 3):
            h, w = img.shape[:2]
            channels = 1 if img.ndim == 2 else img.shape[2]
            flat = img.reshape(1, -1)
            columns = {'p%d' % i: flat[:, i] for i in range(flat.shape[1])}
            cols = _Columns(columns)
            spec = {'y': [cols.add(n, n) for n in columns], 'image_mode': 'one_row', 'image_width': w,
                    'image_channels': channels}
            title = _send('image', 'imshow', cols, spec, options, 'array %dx%d' % (h, w))
            return Plot(title, _imshow, args, original)
    return _kind('image', 'imshow', 'ys')(*args, data=data, **options)


# Basic
plot = line = _kind('line', 'plot', 'xy')
scatter = _kind('scatter', 'scatter', 'xy')
bar = _kind('bar', 'bar', 'x')
hist = _kind('histogram', 'hist', 'x')
area = _kind('area', 'area', 'xy')
step = _kind('step', 'step', 'xy')
stem = _kind('stem', 'stem', 'xy')
pie = _kind('pie', 'pie', 'x')
polar = _kind('polar', 'polar', 'xy')
errorbar = _kind('error_bars', 'errorbar', 'xy')
# Distribution
box = boxplot = _kind('box', 'box', 'ys')
violin = violinplot = _kind('violin', 'violin', 'ys')
kde = _kind('kde', 'kde', 'ys')
parallel = _kind('parallel', 'parallel', 'ys')
pairplot = _kind('pair_plot', 'pairplot', 'ys')
# Grid and density
heatmap = _kind('heatmap', 'heatmap', 'xyv')
matshow = _kind('matrix', 'matshow', 'ys')
hist2d = _kind('histogram_2d', 'hist2d', 'xyv')
hexbin = _kind('hexbin', 'hexbin', 'xyv')
contour = _kind('contour', 'contour', 'xyv')
contourf = _kind('filled_contour', 'contourf', 'xyv')
# Vector fields, images
quiver = _kind('quiver', 'quiver', 'xyuv')
streamplot = _kind('stream', 'streamplot', 'xyuv')
imshow = _imshow
# Model results
confusion_matrix = _kind('confusion_matrix', 'confusion_matrix', 'xyv')
roc_curve = _kind('roc_curve', 'roc_curve', 'xyv')
pr_curve = _kind('pr_curve', 'pr_curve', 'xyv')
calibration = _kind('calibration', 'calibration', 'xyv')
residuals = _kind('residuals', 'residuals', 'xyv')
learning_curve = _kind('learning_curve', 'learning_curve', 'xys')
importance = _kind('feature_importance', 'importance', 'xyv')
# Flows and maps
sankey = _kind('sankey', 'sankey', 'steps')
treemap = _kind('treemap', 'treemap', 'steps')
map_points = _kind('map_points', 'map_points', 'xyv')
map_regions = _kind('map_regions', 'map_regions', 'xyv')
# 3D
scatter3d = _kind('scatter3d', 'scatter3d', 'xyz')
plot3d = _kind('line3d', 'plot3d', 'xyz')
plot_surface = _surface
plot_trisurf = _kind('mesh', 'plot_trisurf', 'xyz')
# Graphs
network = _kind('network', 'network', 'xyv')
tree = _kind('tree', 'tree', 'xyv')
