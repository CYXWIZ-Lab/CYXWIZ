"""Script Editor debugger runtime (TOFIX133 P6, boards 11-12).

The Engine runs a script or notebook cell under `session` on its script
worker thread. Pausing waits on that thread (the GIL is released while
waiting), so the Engine stays responsive and other threads can read the
paused frames. Only frames of the files being debugged are traced; library
code runs at full speed.

The UI side (under the GIL, any thread) calls set_breakpoints, command,
stack, frame_namespace, evaluate and console. The worker reports every
state change through `session.on_state(state, reason, info_json)`, which
the Engine stores without needing the GIL to read it.
"""

import ast
import contextlib
import io
import json
import linecache
import reprlib
import sys
import threading

_short = reprlib.Repr()
_short.maxstring = 1000
_short.maxother = 1000
_short.maxlist = _short.maxtuple = _short.maxset = _short.maxfrozenset = _short.maxdeque = 50
_short.maxdict = 50
_short.maxlevel = 3


class _Breakpoint:
    __slots__ = ('line', 'condition', 'hit_target', 'enabled', 'hits')

    def __init__(self, line, condition='', hit_target=0, enabled=True):
        self.line = line
        self.condition = condition
        self.hit_target = hit_target
        self.enabled = enabled
        self.hits = 0


class Session:
    def __init__(self):
        self.breakpoints = {}        # filename -> {line: _Breakpoint}
        self.stop_on_error = True
        self.on_state = None         # callable(state, reason, info_json), set by the Engine
        self.is_cancelled = None     # callable() -> bool, set by the Engine
        self.files = set()
        self._event = threading.Event()
        self._reset()

    def _reset(self):
        self.state = 'idle'          # idle, running, paused
        self.reason = ''             # breakpoint, step, error, pause
        self.error = ''
        self._frames = []            # [(frame, line)] innermost first, debugged files only
        self._cmd = None
        self._mode = None            # None, 'into', 'over', 'out'
        self._mode_frame = None
        self._pause_requested = False

    # ------------------------------------------------------------ settings
    def set_breakpoints(self, filename, items):
        """items: [{'line', 'condition', 'hit', 'enabled'}]; hit counts of
        lines that stay are kept."""
        old = self.breakpoints.get(filename, {})
        new = {}
        for it in items:
            line = int(it['line'])
            bp = _Breakpoint(line, it.get('condition', '') or '', int(it.get('hit', 0) or 0), bool(it.get('enabled', True)))
            if line in old:
                bp.hits = old[line].hits
            new[line] = bp
        if new:
            self.breakpoints[filename] = new
        else:
            self.breakpoints.pop(filename, None)
        return True

    def hits(self):
        return {f: {str(l): b.hits for l, b in bps.items()} for f, bps in self.breakpoints.items()}

    # ------------------------------------------------------------ running (worker thread)
    def begin(self, files):
        self._reset()
        for bps in self.breakpoints.values():
            for bp in bps.values():
                bp.hits = 0
        self.files = set(files)
        self.state = 'running'
        self._notify()
        sys.settrace(self._trace)

    def end(self):
        sys.settrace(None)
        self._reset()
        self.files = set()
        self._notify()

    def run_script(self, source, filename, namespace):
        """Runs a script under the debugger (the Engine's Debug)."""
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        code = compile(source, filename, 'exec')
        self.begin([filename])
        try:
            exec(code, namespace)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: B902 - the error is shown, then raised again
            self.stop_on_exception(exc)
            raise
        finally:
            self.end()

    def run_cell(self, run, source, key, filename, count):
        """Runs a notebook cell under the debugger: `run` is the Engine's
        cell runner, which reports an uncaught error through on_error."""
        self.begin([filename])
        try:
            return run(source, key, filename, count, self.stop_on_exception)
        finally:
            self.end()

    def stop_on_exception(self, exc):
        """An uncaught error: pause at its deepest line in a debugged file."""
        if not self.stop_on_error or self.state != 'running':
            return
        tb = exc.__traceback__
        last = None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename in self.files:
                last = tb
            tb = tb.tb_next
        if last is None:
            return
        sys.settrace(None)  # nothing more runs here before the error goes on
        message = '%s: %s' % (type(exc).__name__, exc) if str(exc) else type(exc).__name__
        self._pause(last.tb_frame, 'error', line=last.tb_lineno, error=message)

    def request_pause(self):
        self._pause_requested = True

    def _check_cancel(self):
        if self.is_cancelled is not None and self.is_cancelled():
            raise KeyboardInterrupt('Script cancelled by user')

    def _trace(self, frame, event, arg):
        if frame.f_code.co_filename in self.files:
            return self._local
        return None

    def _local(self, frame, event, arg):
        self._check_cancel()
        if event == 'line':
            self._on_line(frame)
        elif event == 'return' and self._mode in ('over', 'out') and frame is self._mode_frame:
            caller = frame.f_back
            # Leaving the frame being stepped: stop at the caller's next line.
            if caller is not None and caller.f_code.co_filename in self.files:
                self._mode = 'over'
                self._mode_frame = caller
                caller.f_trace = self._local
            else:
                self._mode = None
                self._mode_frame = None
        return self._local

    def _on_line(self, frame):
        reason = None
        if self._pause_requested:
            reason = 'pause'
        elif self._mode == 'into':
            reason = 'step'
        elif self._mode == 'over' and frame is self._mode_frame:
            reason = 'step'
        bp = self.breakpoints.get(frame.f_code.co_filename, {}).get(frame.f_lineno)
        if bp is not None and bp.enabled and reason is None:
            hit = True
            if bp.condition:
                try:
                    hit = bool(eval(bp.condition, frame.f_globals, frame.f_locals))
                except Exception as exc:  # a broken condition stops, and says why
                    self.error = 'Condition %r failed: %s: %s' % (bp.condition, type(exc).__name__, exc)
                    hit = True
            if hit:
                bp.hits += 1
                if not bp.hit_target or bp.hits >= bp.hit_target:
                    reason = 'breakpoint'
        if reason is not None:
            self._pause(frame, reason, error=self.error)

    def _pause(self, frame, reason, line=None, error=''):
        frames = []
        f = frame
        first = True
        while f is not None:
            if f.f_code.co_filename in self.files:
                frames.append((f, line if (first and line is not None) else f.f_lineno))
            first = False
            f = f.f_back
        self._frames = frames
        self.state = 'paused'
        self.reason = reason
        self.error = error or ''
        self._cmd = None
        self._pause_requested = False
        self._event.clear()
        self._notify()
        while not self._event.wait(0.05):  # the GIL is free while waiting
            self._check_cancel()
        cmd = self._cmd
        self._frames = []
        self.error = ''
        self.state = 'running'
        self._mode, self._mode_frame = None, None
        if cmd == 'into':
            self._mode = 'into'
        elif cmd == 'over':
            self._mode, self._mode_frame = 'over', frame
        elif cmd == 'out':
            self._mode, self._mode_frame = 'out', frame
        self._notify()

    def _notify(self):
        if self.on_state is not None:
            try:
                self.on_state(self.state, self.reason, json.dumps({'stack': self.stack(), 'error': self.error, 'hits': self.hits()}))
            except Exception:
                pass

    # ------------------------------------------------------------ UI side
    def command(self, cmd):
        """continue, over, into, out. False when not paused."""
        if self.state != 'paused' or cmd not in ('continue', 'over', 'into', 'out'):
            return False
        self._cmd = cmd
        self._event.set()
        return True

    def stack(self):
        return [{'name': f.f_code.co_name, 'file': f.f_code.co_filename, 'line': line} for f, line in self._frames]

    def frame_namespace(self, index, which='locals'):
        if not (0 <= index < len(self._frames)):
            return {}
        f = self._frames[index][0]
        return f.f_globals if which == 'globals' else f.f_locals

    def evaluate(self, expr, index=0):
        """A Watch expression in a paused frame: {'value', 'type'} or {'error'}."""
        if self.state != 'paused' or not (0 <= index < len(self._frames)):
            return {'error': 'Not paused'}
        f = self._frames[index][0]
        try:
            value = eval(expr, f.f_globals, f.f_locals)
        except NameError as exc:
            return {'error': str(exc), 'undefined': True}
        except Exception as exc:
            return {'error': '%s: %s' % (type(exc).__name__, exc)}
        try:
            text = _short.repr(value)
        except Exception:
            text = '<no preview>'
        return {'value': text, 'type': type(value).__name__}

    def console(self, source, index=0):
        """The Console while paused: statements run in the frame, a last
        expression is echoed. {'output', 'value', 'error'}."""
        if self.state != 'paused' or not (0 <= index < len(self._frames)):
            return {'error': 'Not paused'}
        f = self._frames[index][0]
        out = io.StringIO()
        result = {'output': '', 'value': None, 'error': ''}
        try:
            tree = ast.parse(source, '<debug console>', 'exec')
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            scope = f.f_locals
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(tree, '<debug console>', 'exec'), f.f_globals, scope)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), '<debug console>', 'eval'), f.f_globals, scope)
                    if value is not None:
                        result['value'] = repr(value)
        except Exception as exc:
            result['error'] = '%s: %s' % (type(exc).__name__, exc)
        result['output'] = out.getvalue()
        return result


session = Session()
