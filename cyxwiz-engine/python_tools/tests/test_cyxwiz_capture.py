"""Tests for cyxwiz_capture + cyxwiz_mpl_backend (TOFIX134 P5.2): run with the
folder that holds them as the first argument (the build's <exe dir>/python_tools).
The matplotlib part runs when the interpreter has matplotlib."""

import os
import sys
import unittest

TOOLS = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, TOOLS)
import cyxwiz_capture  # noqa: E402
sys.path.remove(TOOLS)

received = []
cyxwiz_capture.install(lambda png, w, h, title: received.append((png, w, h, title)))

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class CaptureTest(unittest.TestCase):
    def test_install_selects_the_backend(self):
        self.assertEqual(os.environ['MPLBACKEND'], 'module://cyxwiz_mpl_backend')
        # The tools folder is not left on sys.path.
        self.assertNotIn(TOOLS, [os.path.abspath(p) for p in sys.path])

    def test_emit_before_install_is_safe(self):
        saved = cyxwiz_capture._sink
        cyxwiz_capture._sink = None
        try:
            cyxwiz_capture.emit(b'x', 1, 1, '')  # no error, nothing sent
        finally:
            cyxwiz_capture._sink = saved

    @unittest.skipIf(plt is None, 'matplotlib not installed')
    def test_show_sends_every_figure_and_closes_them(self):
        received.clear()
        self.assertEqual(matplotlib.get_backend(), 'module://cyxwiz_mpl_backend')
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title('squares')
        fig = plt.figure(figsize=(2, 2))
        fig.suptitle('second')
        plt.show()
        self.assertEqual(len(received), 2)
        png, width, height, title = received[0]
        self.assertTrue(png.startswith(b'\x89PNG'))
        self.assertEqual((width, height), (400, 300))
        self.assertEqual(title, 'squares')
        self.assertEqual(received[1][3], 'second')
        self.assertEqual(plt.get_fignums(), [])

    @unittest.skipIf(plt is None, 'matplotlib not installed')
    def test_show_with_no_figures_sends_nothing(self):
        received.clear()
        plt.show()
        self.assertEqual(received, [])


if __name__ == '__main__':
    unittest.main()
