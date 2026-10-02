"""Variable Explorer and Data Viewer reader (TOFIX133 P5, boards 9-10).

The Engine calls these on its variables worker with a namespace dict: the
Python session's __main__.__dict__ or one notebook's own namespace. Every
function is bounded (reprlib limits, row/item caps) so a large value cannot
stall the Engine, never imports a library the user has not imported, and
leaves nothing behind in the namespace.

A path names a value inside the namespace: a list of steps, each
["name", str] (first step), ["item", int] (n-th entry of a dict, list,
tuple or set, in iteration order), ["attr", str] or ["col", str]
(DataFrame column).
"""

import reprlib
import sys
import types

MAX_CHILDREN = 200
MAX_VALUE = 200
TABLE_ROWS = 200_000
COPY_LIMIT = 1_000_000
HASH_BYTES = 8 * 1024 * 1024
STATS_ELEMENTS = 10_000_000

_short = reprlib.Repr()
_short.maxstring = MAX_VALUE
_short.maxother = MAX_VALUE
_short.maxlist = _short.maxtuple = _short.maxset = _short.maxfrozenset = _short.maxdeque = 20
_short.maxdict = 20
_short.maxlevel = 2

_HIDDEN = {'In', 'Out', 'exit', 'quit', 'get_ipython'}


# ---------------------------------------------------------------------------
# What a value is

def _module_of(value):
    return getattr(type(value), '__module__', '') or ''


def _is_frame(value):
    return type(value).__name__ == 'DataFrame' and _module_of(value).startswith('pandas')


def _is_series(value):
    return type(value).__name__ == 'Series' and _module_of(value).startswith('pandas')


def _is_ndarray(value):
    np = sys.modules.get('numpy')
    return np is not None and isinstance(value, np.ndarray)


def _is_tensor(value):
    mod = _module_of(value)
    return type(value).__name__ == 'Tensor' and (mod.startswith('torch') or mod.startswith('pycyxwiz'))


def _shape(value):
    """Shape as a tuple of ints, or None. pycyxwiz Tensor.shape is a method."""
    try:
        shape = getattr(value, 'shape', None)
        if shape is None:
            return None
        if callable(shape) and _module_of(value).startswith('pycyxwiz'):
            shape = shape()
        if callable(shape):
            return None
        return tuple(int(n) for n in shape)
    except Exception:
        return None


def _dtype(value):
    try:
        dtype = getattr(value, 'dtype', None)
        if callable(dtype) and _module_of(value).startswith('pycyxwiz'):
            dtype = dtype()
        return '' if dtype is None or callable(dtype) else str(dtype)
    except Exception:
        return ''


def _kind(value):
    """The chip a value belongs to (board 9)."""
    if isinstance(value, types.ModuleType):
        return 'module'
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)):
        return 'function'
    if isinstance(value, type):
        return 'class'
    if isinstance(value, bool) or isinstance(value, (int, float, complex)):
        return 'number'
    if value is None:
        return 'other'
    if isinstance(value, (str, bytes, bytearray)):
        return 'text'
    if _is_frame(value) or _is_series(value):
        return 'table'
    if _is_ndarray(value) or _is_tensor(value):
        return 'array'
    np = sys.modules.get('numpy')
    if np is not None and isinstance(value, np.generic):
        return 'number'
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return 'collection'
    return 'other'


def _type_name(value):
    name = type(value).__name__
    if _is_ndarray(value) or _is_tensor(value) or _is_series(value):
        dtype = _dtype(value)
        if dtype:
            return '%s[%s]' % (name, dtype)
    return name


def _size(value, kind):
    shape = _shape(value) if kind in ('table', 'array') else None
    if shape is not None:
        return '(' + ', '.join(str(n) for n in shape) + (',)' if len(shape) == 1 else ')')
    try:
        if isinstance(value, (list, tuple)):
            return '(%d,)' % len(value)
        if isinstance(value, (dict, set, frozenset)):
            return '%d items' % len(value)
        if isinstance(value, (str, bytes, bytearray)):
            return 'len=%d' % len(value)
    except Exception:
        pass
    return ''


def _memory(value, kind):
    """Bytes held by arrays and tables; -1 for everything else (a shallow
    getsizeof would understate a list of objects)."""
    try:
        if _is_ndarray(value):
            return int(value.nbytes)
        if _is_frame(value):
            deep = len(value) <= 100_000
            return int(value.memory_usage(index=True, deep=deep).sum())
        if _is_series(value):
            return int(value.memory_usage(index=True, deep=len(value) <= 100_000))
        if _is_tensor(value):
            nbytes = getattr(value, 'nbytes', None)
            if nbytes is not None and not callable(nbytes):
                return int(nbytes)
            if hasattr(value, 'element_size') and hasattr(value, 'nelement'):
                return int(value.element_size() * value.nelement())
    except Exception:
        pass
    return -1


def _number_text(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return '%.3g' % x if abs(x) >= 1e-3 and abs(x) < 1e6 else '%.3e' % x


def _array_summary(value):
    dtype = _dtype(value)
    try:
        np = sys.modules['numpy']
        arr = value if _is_ndarray(value) else None
        if arr is None and _is_tensor(value) and hasattr(value, 'numpy'):
            arr = value.detach().cpu().numpy() if hasattr(value, 'detach') else value.numpy()
        if arr is not None and arr.size and arr.size <= STATS_ELEMENTS and np.issubdtype(arr.dtype, np.number) \
                and not np.issubdtype(arr.dtype, np.complexfloating):
            return '%s · min %s · max %s · mean %s' % (
                dtype, _number_text(arr.min()), _number_text(arr.max()), _number_text(arr.mean()))
        if arr is not None:
            return '%s · %s' % (dtype, _short.repr(arr.ravel()[:20].tolist()))
    except Exception:
        pass
    return dtype


def _value_text(value, kind):
    try:
        if _is_frame(value):
            cols = ['%s (%s)' % (c, value[c].dtype) if value.columns.is_unique else str(c) for c in list(value.columns)[:12]]
            return 'columns: ' + ', '.join(cols) + (', ...' if len(value.columns) > 12 else '')
        if _is_series(value):
            head = ', '.join('%s → %s' % (k, v) for k, v in list(value.head(3).items()))
            name = value.name if value.name is not None else 'unnamed'
            return '%s, dtype %s: %s%s' % (name, value.dtype, head, ', ...' if len(value) > 3 else '')
        if kind == 'array':
            return _array_summary(value)
        text = _short.repr(value)
    except Exception:
        return '<no preview>'
    text = text.replace('\n', ' ')
    return text if len(text) <= MAX_VALUE else text[:MAX_VALUE] + '...'


def _digest(value, kind, size, text):
    """Changes when the value changes (for the changed-in-last-run mark)."""
    try:
        if _is_ndarray(value) and value.nbytes <= HASH_BYTES:
            return '%s|%s|%x' % (value.dtype, value.shape, hash(value.tobytes()))
        if _is_frame(value) and len(value) <= 100_000:
            pd = sys.modules['pandas']
            return '%s|%x' % (value.shape, int(pd.util.hash_pandas_object(value, index=True).sum()) & 0xFFFFFFFFFFFF)
    except Exception:
        pass
    return '%s|%s|%s' % (type(value).__name__, size, text)


def _attributes(value):
    try:
        d = vars(value)
    except TypeError:
        return []
    return [(k, v) for k, v in d.items() if not k.startswith('_')]


def _expandable(value, kind):
    if kind in ('module', 'function', 'class', 'number', 'text'):
        return False
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        try:
            return len(value) > 0
        except Exception:
            return False
    if _is_frame(value):
        return len(value.columns) > 0
    if kind in ('array', 'table') or value is None:
        return False
    return bool(_attributes(value))


def _viewable(value, kind):
    if kind in ('table', 'array'):
        return True
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        return isinstance(first, (int, float, str, bool, list, tuple)) or first is None
    return False


def _row(label, value, step):
    kind = _kind(value)
    size = _size(value, kind)
    text = _value_text(value, kind)
    return {
        'name': label,
        'step': step,
        'type': _type_name(value),
        'kind': kind,
        'size': size,
        'memory': _memory(value, kind),
        'value': text,
        'expandable': _expandable(value, kind),
        'viewable': _viewable(value, kind),
        'digest': _digest(value, kind, size, text),
    }


# ---------------------------------------------------------------------------
# Public functions

def list_variables(namespace, show_all=False):
    """Top-level variables, sorted by name. Modules, functions and classes
    only with show_all."""
    out = []
    for name, value in list(namespace.items()):
        if not isinstance(name, str) or name.startswith('_') or name in _HIDDEN:
            continue
        try:
            row = _row(name, value, ['name', name])
        except Exception:
            continue
        if not show_all and row['kind'] in ('module', 'function', 'class'):
            continue
        out.append(row)
    out.sort(key=lambda r: r['name'].lower())
    return out


def resolve(namespace, path):
    if not path or path[0][0] != 'name':
        raise KeyError('path must start with a name')
    value = namespace[path[0][1]]
    for kind, key in path[1:]:
        if kind == 'item':
            if isinstance(value, dict):
                value = list(value.values())[int(key)]
            elif isinstance(value, (set, frozenset)):
                value = list(value)[int(key)]
            else:
                value = value[int(key)]
        elif kind == 'attr':
            value = vars(value)[key]
        elif kind == 'col':
            value = value[key]
        else:
            raise KeyError(kind)
    return value


def children(namespace, path):
    """The rows under an expandable value; a last row {'more': n} when cut."""
    value = resolve(namespace, path)
    rows = []
    total = 0
    if isinstance(value, dict):
        items = list(value.items())
        total = len(items)
        for i, (k, v) in enumerate(items[:MAX_CHILDREN]):
            rows.append(_row(_short.repr(k), v, ['item', i]))
    elif isinstance(value, (list, tuple)):
        total = len(value)
        for i, v in enumerate(value[:MAX_CHILDREN]):
            rows.append(_row('[%d]' % i, v, ['item', i]))
    elif isinstance(value, (set, frozenset)):
        items = list(value)
        total = len(items)
        for i, v in enumerate(items[:MAX_CHILDREN]):
            rows.append(_row('{%d}' % i, v, ['item', i]))
    elif _is_frame(value):
        cols = list(value.columns)
        total = len(cols)
        for c in cols[:MAX_CHILDREN]:
            col = value[c]
            row = _row(str(c), col, ['col', c])
            if _is_series(col):
                row['value'] = _column_summary(col)
            rows.append(row)
    else:
        attrs = _attributes(value)
        total = len(attrs)
        for k, v in attrs[:MAX_CHILDREN]:
            rows.append(_row(k, v, ['attr', k]))
    if total > MAX_CHILDREN:
        rows.append({'more': total - MAX_CHILDREN})
    return rows


def _column_summary(col):
    try:
        types_api = sys.modules['pandas'].api.types
        if len(col) and types_api.is_numeric_dtype(col.dtype) and not types_api.is_bool_dtype(col.dtype) \
                and not types_api.is_complex_dtype(col.dtype):
            return 'min %s · max %s · mean %s' % (_number_text(col.min()), _number_text(col.max()), _number_text(col.mean()))
        values = [str(v) for v in col.head(5).tolist()]
        return ', '.join(values) + (', ...' if len(col) > 5 else '')
    except Exception:
        return ''


def value_text(namespace, path, limit=COPY_LIMIT):
    """The value as text for Copy value: repr, cut at `limit` characters."""
    value = resolve(namespace, path)
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + '\n... (%d more characters)' % (len(text) - limit)


def delete(namespace, path):
    """Deletes a top-level variable; returns '' or why not."""
    if len(path) != 1:
        return 'Only variables can be deleted, not the values inside them'
    name = path[0][1]
    if name not in namespace:
        return 'This variable is gone'
    del namespace[name]
    return ''


def save_csv(namespace, path, file_path):
    """Writes a table, array or list to CSV; returns '' or why not."""
    value = resolve(namespace, path)
    if hasattr(value, 'to_csv') and (_is_frame(value) or _is_series(value)):
        value.to_csv(file_path)
        return ''
    data = table(namespace, path, max_rows=None)
    if 'error' in data:
        return data['error']
    import csv
    with open(file_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([c['name'] for c in data['columns']])
        for i in range(data['shown']):
            writer.writerow([c['values'][i] for c in data['columns']])
    return ''


def _plain(values):
    """Python list of plain values (str, int, float, bool, None)."""
    # .tolist() already gives Python numbers; dates, decimals and objects
    # are shown as their text.
    return [v if v is None or isinstance(v, (str, int, float, bool)) else str(v) for v in values]


def table(namespace, path, max_rows=TABLE_ROWS, index=None):
    """Columns for the Data Viewer (board 10).

    Returns {'kind', 'shape', 'rows' (total), 'shown', 'index_name',
    'index' (values or None), 'columns': [{'name', 'dtype', 'values'}],
    'slice' (leading-axis indices used for arrays with more than two axes)}
    or {'error': why}."""
    try:
        value = resolve(namespace, path)
    except Exception as e:
        return {'error': 'This value is gone (%s)' % e}
    limit = max_rows if max_rows is not None else float('inf')

    if _is_series(value):
        value = value.to_frame()
    if _is_frame(value):
        total = len(value)
        shown = int(min(total, limit))
        part = value.iloc[:shown]
        cols = []
        for i, c in enumerate(part.columns):
            col = part.iloc[:, i]
            cols.append({'name': str(c), 'dtype': str(col.dtype), 'values': _plain(col.tolist())})
        return {'kind': 'frame', 'shape': list(value.shape), 'rows': total, 'shown': shown,
                'index_name': str(part.index.name) if part.index.name is not None else 'index',
                'index': _plain(part.index.tolist()), 'columns': cols, 'slice': []}

    if _is_tensor(value):
        try:
            value = value.detach().cpu().numpy() if hasattr(value, 'detach') else value.numpy()
        except Exception as e:
            return {'error': 'This tensor could not be read (%s)' % e}
    if _is_ndarray(value):
        shape = list(value.shape)
        lead = list(index or [])[:max(0, value.ndim - 2)]
        lead += [0] * (max(0, value.ndim - 2) - len(lead))
        lead = [max(0, min(int(i), n - 1)) for i, n in zip(lead, shape)]
        arr = value[tuple(lead)] if lead else value
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        total = arr.shape[0]
        shown = int(min(total, limit))
        dtype = str(value.dtype)
        cols = [{'name': str(j) if value.ndim > 1 else 'value', 'dtype': dtype, 'values': _plain(arr[:shown, j].tolist())}
                for j in range(arr.shape[1])]
        return {'kind': 'array', 'shape': shape, 'rows': total, 'shown': shown, 'index_name': '',
                'index': None, 'columns': cols, 'slice': lead}

    if isinstance(value, (list, tuple)):
        total = len(value)
        shown = int(min(total, limit))
        part = list(value[:shown])
        if part and all(isinstance(r, (list, tuple)) for r in part):
            width = max(len(r) for r in part)
            cols = [{'name': str(j), 'dtype': 'object',
                     'values': _plain([r[j] if j < len(r) else None for r in part])} for j in range(min(width, 1000))]
        else:
            cols = [{'name': 'value', 'dtype': 'object', 'values': _plain(part)}]
        return {'kind': 'list', 'shape': [total], 'rows': total, 'shown': shown, 'index_name': '',
                'index': None, 'columns': cols, 'slice': []}

    return {'error': 'Only tables, arrays and lists open in the Data Viewer'}
