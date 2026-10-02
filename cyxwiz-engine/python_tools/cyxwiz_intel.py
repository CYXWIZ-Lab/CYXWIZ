"""Language intelligence for the CyxWiz Script Editor (TOFIX133 P3, decision D2).

Runs in the Engine's own Python on a worker thread: Jedi for completion,
hover, signature help and go to definition; pyflakes for problems. Jedi,
parso and pyflakes ship with the Engine in this folder (owner decision
2026-10-02) and are imported from here without leaving the folder on
sys.path.

Every function takes plain values and returns plain lists/dicts (JSON-able).
Lines are 1-based and columns are 0-based character offsets, as in Jedi.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_jedi = None
_pyflakes_api = None
_load_error = ''


def _load():
    """Imports jedi and pyflakes from this folder once."""
    global _jedi, _pyflakes_api, _load_error
    if _jedi is not None or _load_error:
        return _jedi is not None
    added = _HERE not in sys.path
    if added:
        sys.path.insert(0, _HERE)
    try:
        import jedi
        from pyflakes import api as pyflakes_api
        _jedi = jedi
        _pyflakes_api = pyflakes_api
    except Exception as exc:  # a broken install: completion stays basic
        _load_error = '%s: %s' % (type(exc).__name__, exc)
    finally:
        if added:
            try:
                sys.path.remove(_HERE)
            except ValueError:
                pass
    return _jedi is not None


def status():
    ok = _load()
    return {
        'ok': ok,
        'error': _load_error,
        'jedi': getattr(_jedi, '__version__', '') if ok else '',
    }


def _script(source, path, project_root):
    project = _jedi.Project(project_root) if project_root else None
    # The running interpreter (the project's environment): no subprocess.
    env = _jedi.InterpreterEnvironment()
    return _jedi.Script(source, path=path or None, project=project, environment=env)


def _interpreter(source, namespace):
    return _jedi.Interpreter(source, [namespace])


_KINDS = {
    'module': 'module', 'class': 'class', 'instance': 'variable', 'function': 'function',
    'param': 'variable', 'path': 'path', 'keyword': 'keyword', 'property': 'property',
    'statement': 'variable',
}


def _short_doc(name, limit=600):
    try:
        doc = name.docstring(raw=True) or ''
    except Exception:
        doc = ''
    doc = doc.strip()
    if len(doc) > limit:
        doc = doc[:limit].rstrip() + '...'
    return doc


def _signature_text(name):
    try:
        sigs = name.get_signatures()
        if sigs:
            return sigs[0].to_string()
    except Exception:
        pass
    return ''


def complete(source, line, column, path='', project_root='', namespace=None, limit=200):
    """Completions at (line, column). With `namespace` (a notebook's dict)
    the live objects are used, so DataFrame columns and attributes complete."""
    if not _load():
        return []
    try:
        script = _interpreter(source, namespace) if namespace is not None else _script(source, path, project_root)
        items = script.complete(line, column)
    except Exception:
        return []
    out = []
    for c in items[:limit]:
        kind = _KINDS.get(c.type, c.type or 'text')
        detail = ''
        if c.type in ('function', 'class'):
            detail = _signature_text(c)
        elif c.type == 'module':
            detail = c.full_name or c.name
        elif c.type in ('instance', 'statement', 'param'):
            try:
                inferred = c.infer()
                if inferred:
                    detail = inferred[0].name
            except Exception:
                pass
        out.append({
            'name': c.name,
            'complete': c.complete,  # the text still to type
            'kind': kind,
            'detail': detail,
            'module': c.module_name or '',
        })
    return out


def describe(source, line, column, path='', project_root='', namespace=None, name=''):
    """Docstring and signature of one completion (`name`) for the details
    side of the list; computed only for the selected item."""
    if not _load():
        return {}
    try:
        script = _interpreter(source, namespace) if namespace is not None else _script(source, path, project_root)
        for c in script.complete(line, column):
            if c.name == name:
                return {'name': c.name, 'kind': _KINDS.get(c.type, c.type), 'signature': _signature_text(c),
                        'doc': _short_doc(c), 'module': c.module_name or ''}
    except Exception:
        pass
    return {}


def hover(source, line, column, path='', project_root='', namespace=None):
    """What the name under the cursor is: kind, signature or type, docstring,
    and where it is defined."""
    if not _load():
        return {}
    try:
        script = _interpreter(source, namespace) if namespace is not None else _script(source, path, project_root)
        names = script.help(line, column) or script.infer(line, column)
    except Exception:
        return {}
    if not names:
        return {}
    n = names[0]
    info = {
        'name': n.name,
        'kind': _KINDS.get(n.type, n.type),
        'signature': _signature_text(n),
        'type': '',
        'doc': _short_doc(n, 1500),
        'module': n.module_name or '',
        'path': str(n.module_path) if n.module_path else '',
        'line': n.line or 0,
    }
    if n.type in ('instance', 'statement', 'param'):
        try:
            inferred = script.infer(line, column)
            if inferred:
                info['type'] = inferred[0].name
        except Exception:
            pass
    return info


def signatures(source, line, column, path='', project_root='', namespace=None):
    """The call around the cursor: parameters and which one is being typed."""
    if not _load():
        return []
    try:
        script = _interpreter(source, namespace) if namespace is not None else _script(source, path, project_root)
        sigs = script.get_signatures(line, column)
    except Exception:
        return []
    out = []
    for s in sigs[:3]:
        params = []
        for p in s.params:
            try:
                params.append(p.to_string())
            except Exception:
                params.append(p.name)
        out.append({
            'name': s.name,
            'params': params,
            'index': s.index if s.index is not None else -1,
            'doc': _short_doc(s, 400),
        })
    return out


def definition(source, line, column, path='', project_root='', namespace=None):
    """Where the name under the cursor is defined (goto, following imports)."""
    if not _load():
        return []
    try:
        script = _interpreter(source, namespace) if namespace is not None else _script(source, path, project_root)
        names = script.goto(line, column, follow_imports=True, follow_builtin_imports=False)
    except Exception:
        return []
    out = []
    for n in names[:10]:
        out.append({
            'name': n.name,
            'path': str(n.module_path) if n.module_path else '',
            'line': n.line or 0,
            'column': n.column or 0,
            'module': n.module_name or '',
            'in_source': n.module_path is None or (path and os.path.normcase(str(n.module_path)) == os.path.normcase(path)),
        })
    return out


class _Collector:
    """pyflakes reporter that keeps messages as dicts."""

    def __init__(self):
        self.items = []

    def unexpectedError(self, filename, msg):
        self.items.append({'line': 1, 'column': 0, 'severity': 'error', 'message': str(msg), 'code': 'internal'})

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.items.append({'line': lineno or 1, 'column': max(0, (offset or 1) - 1), 'severity': 'error',
                           'message': 'Syntax error: %s' % msg, 'code': 'syntax'})

    def flake(self, message):
        text = message.message % message.message_args
        # Unused imports/variables are warnings; undefined names and the
        # like would fail at run time.
        kind = type(message).__name__
        severity = 'warning' if kind in ('UnusedImport', 'UnusedVariable', 'RedefinedWhileUnused',
                                          'ImportShadowedByLoopVar', 'ImportStarUsed', 'ImportStarUsage',
                                          'UnusedAnnotation', 'UnusedIndirectAssignment') else 'error'
        self.items.append({'line': message.lineno, 'column': getattr(message, 'col', 0) or 0,
                           'severity': severity, 'message': text, 'code': kind})


def diagnostics(source, path='', builtins_extra=None):
    """Problems pyflakes finds without running the code. `builtins_extra`
    names (e.g. a notebook's earlier variables) are not reported as undefined."""
    if not _load():
        return []
    import ast
    from pyflakes import checker
    collector = _Collector()
    filename = path or '<editor>'
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        collector.syntaxError(filename, e.msg, e.lineno, e.offset, e.text)
        return collector.items
    except Exception as exc:
        collector.unexpectedError(filename, exc)
        return collector.items
    try:
        w = checker.Checker(tree, filename=filename, builtins=set(builtins_extra or ()))
        for message in w.messages:
            collector.flake(message)
    except Exception as exc:
        collector.unexpectedError(filename, exc)
    collector.items.sort(key=lambda d: (d['line'], d['column']))
    return collector.items
