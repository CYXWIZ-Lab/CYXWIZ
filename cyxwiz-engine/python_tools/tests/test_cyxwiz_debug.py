"""Tests for cyxwiz_debug (TOFIX133 P6): run with the folder that holds
cyxwiz_debug.py as the first argument (the build's <exe dir>/python_tools).
Each test runs a script on a worker thread, as the Engine does, and drives
it from the test thread, as the UI does."""

import json
import os
import sys
import threading
import time
import unittest

TOOLS = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, TOOLS)
import cyxwiz_debug  # noqa: E402
sys.path.remove(TOOLS)

SCRIPT = '''def compute(values, scale):
    total = 0
    for v in values:
        total += v * scale
    return total


def train(epochs):
    history = []
    for epoch in range(epochs):
        loss = compute([1, 2, 3], epoch + 1)
        history.append(loss)
    return history


history = train(3)
done = True
'''
FILE = 'debugged.py'


class Runner:
    """The script on a worker thread; states recorded as the Engine would."""

    def __init__(self, session, source=SCRIPT, filename=FILE):
        self.session = session
        self.states = []
        self.cancel = False
        self.error = None
        self.ns = {'__name__': '__main__'}
        self.seen = 0
        session.on_state = lambda state, reason, info: self.states.append((state, reason, json.loads(info)))
        session.is_cancelled = lambda: self.cancel
        self.thread = threading.Thread(target=self._run, args=(source, filename), daemon=True)
        self.thread.start()

    def _run(self, source, filename):
        try:
            self.session.run_script(source, filename, self.ns)
        except BaseException as exc:  # noqa: B902
            self.error = exc

    def wait_paused(self, timeout=5.0):
        """The next pause after the last one seen (a command takes effect on
        the worker a moment later)."""
        end = time.time() + timeout
        while time.time() < end:
            pauses = [st for st in self.states if st[0] == 'paused']
            if len(pauses) > self.seen:
                self.seen = len(pauses)
                return pauses[-1]
            if not self.thread.is_alive():
                return None
            time.sleep(0.01)
        raise AssertionError('never paused')

    def finish(self, timeout=5.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), 'still running'


class DebugTest(unittest.TestCase):
    def setUp(self):
        self.s = cyxwiz_debug.Session()

    def test_runs_to_the_end_without_breakpoints(self):
        r = Runner(self.s)
        r.finish()
        self.assertIsNone(r.error)
        self.assertEqual(r.ns['history'], [6, 12, 18])
        self.assertEqual(self.s.state, 'idle')
        self.assertEqual(sys.gettrace(), None)

    def test_breakpoint_stack_and_frame_values(self):
        self.s.set_breakpoints(FILE, [{'line': 4}])
        r = Runner(self.s)
        state, reason, info = r.wait_paused()
        self.assertEqual((state, reason), ('paused', 'breakpoint'))
        self.assertEqual([(f['name'], f['line']) for f in info['stack']], [('compute', 4), ('train', 11), ('<module>', 16)])
        self.assertEqual(self.s.frame_namespace(0)['v'], 1)
        self.assertEqual(self.s.frame_namespace(1)['epoch'], 0)
        self.assertEqual(self.s.evaluate('v * scale + 1'), {'value': '2', 'type': 'int'})
        self.assertTrue(self.s.evaluate('missing')['undefined'])
        self.assertIn('ZeroDivisionError', self.s.evaluate('1 / 0')['error'])
        self.assertEqual(info['hits'][FILE]['4'], 1)
        self.s.set_breakpoints(FILE, [])
        self.assertTrue(self.s.command('continue'))
        r.finish()
        self.assertEqual(r.ns['history'], [6, 12, 18])

    def test_condition_and_hit_count(self):
        self.s.set_breakpoints(FILE, [{'line': 11, 'condition': 'epoch == 2'}])
        r = Runner(self.s)
        r.wait_paused()
        self.assertEqual(self.s.frame_namespace(0)['epoch'], 2)
        self.s.command('continue')
        r.finish()
        s2 = cyxwiz_debug.Session()
        s2.set_breakpoints(FILE, [{'line': 4, 'hit': 5}])
        r2 = Runner(s2)
        r2.wait_paused()
        self.assertEqual(s2.frame_namespace(0)['v'], 2)  # 5th hit: epoch 1, v = 2
        s2.set_breakpoints(FILE, [])
        s2.command('continue')
        r2.finish()

    def test_broken_condition_stops_and_says_why(self):
        self.s.set_breakpoints(FILE, [{'line': 11, 'condition': 'nope > 1'}])
        r = Runner(self.s)
        _, _, info = r.wait_paused()
        self.assertIn("Condition 'nope > 1' failed: NameError", info['error'])
        self.s.set_breakpoints(FILE, [])
        self.s.command('continue')
        r.finish()

    def test_stepping(self):
        self.s.set_breakpoints(FILE, [{'line': 11}])
        r = Runner(self.s)
        r.wait_paused()
        self.s.set_breakpoints(FILE, [])
        self.s.command('over')  # compute runs whole
        _, reason, info = r.wait_paused()
        self.assertEqual((reason, info['stack'][0]['name'], info['stack'][0]['line']), ('step', 'train', 12))
        self.assertEqual(self.s.frame_namespace(0)['loss'], 6)
        self.s.command('over')  # back to the for line
        _, _, info = r.wait_paused()
        self.assertEqual((info['stack'][0]['name'], info['stack'][0]['line']), ('train', 10))
        self.s.command('over')
        _, _, info = r.wait_paused()
        self.assertEqual((info['stack'][0]['name'], info['stack'][0]['line']), ('train', 11))
        self.s.command('into')
        _, _, info = r.wait_paused()
        self.assertEqual((info['stack'][0]['name'], info['stack'][0]['line']), ('compute', 2))
        self.s.command('out')  # back in train after compute returns
        _, _, info = r.wait_paused()
        self.assertEqual((info['stack'][0]['name'], info['stack'][0]['line']), ('train', 12))
        self.assertEqual(self.s.frame_namespace(0)['loss'], 12)
        self.s.command('continue')
        r.finish()
        self.assertEqual(r.ns['history'], [6, 12, 18])

    def test_console_in_the_paused_frame(self):
        self.s.set_breakpoints(FILE, [{'line': 4}])
        r = Runner(self.s)
        r.wait_paused()
        out = self.s.console('print("v is", v)\nv + 10')
        self.assertEqual((out['output'], out['value'], out['error']), ('v is 1\n', '11', ''))
        self.assertIn('NameError', self.s.console('nope')['error'])
        self.s.set_breakpoints(FILE, [])
        self.s.command('continue')
        r.finish()

    def test_stop_on_uncaught_error_then_raise(self):
        source = 'def f(x):\n    return 10 / x\n\nvalues = [2, 1, 0]\nout = [f(v) for v in values]\n'
        r = Runner(self.s, source, 'err.py')
        _, reason, info = r.wait_paused()
        self.assertEqual(reason, 'error')
        self.assertEqual(info['error'], 'ZeroDivisionError: division by zero')
        self.assertEqual((info['stack'][0]['name'], info['stack'][0]['line']), ('f', 2))
        self.assertEqual(self.s.frame_namespace(0)['x'], 0)
        self.s.command('continue')
        r.finish()
        self.assertIsInstance(r.error, ZeroDivisionError)
        s2 = cyxwiz_debug.Session()
        s2.stop_on_error = False
        r2 = Runner(s2, source, 'err.py')
        r2.finish()
        self.assertIsInstance(r2.error, ZeroDivisionError)

    def test_stop_while_paused(self):
        self.s.set_breakpoints(FILE, [{'line': 4}])
        r = Runner(self.s)
        r.wait_paused()
        r.cancel = True  # the Engine's Stop
        r.finish()
        self.assertIsInstance(r.error, KeyboardInterrupt)
        self.assertEqual(self.s.state, 'idle')

    def test_pause_while_running(self):
        source = 'import time\nn = 0\nwhile n < 10_000_000:\n    n += 1\n'
        r = Runner(self.s, source, 'loop.py')
        time.sleep(0.2)
        self.s.request_pause()
        _, reason, info = r.wait_paused()
        self.assertEqual(reason, 'pause')
        self.assertIn(info['stack'][0]['line'], (3, 4))
        r.cancel = True
        r.finish()

    def test_library_code_is_not_traced(self):
        self.s.set_breakpoints(FILE, [{'line': 4}])
        r = Runner(self.s, 'import json\nx = json.dumps({"a": 1})\n', FILE)
        r.finish()
        self.assertEqual(r.ns['x'], '{"a": 1}')
        self.assertFalse(any(s[0] == 'paused' for s in r.states))


if __name__ == '__main__':
    unittest.main(verbosity=1)
