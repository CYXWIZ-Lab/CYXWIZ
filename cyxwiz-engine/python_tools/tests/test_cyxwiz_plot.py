"""Tests for cyxwiz.py, the matplotlib-like plot API (TOFIX134 P5): run with
the folder that holds it as the first argument (the build's
<exe dir>/python_tools). The Engine's sinks are replaced by recorders; the
NumPy and pandas parts run when the interpreter has them."""

import json
import math
import os
import sys
import unittest

TOOLS = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, TOOLS)
import cyxwiz_capture  # noqa: E402
sys.path.remove(TOOLS)

sent, closed = [], []
answer = {'problem': ''}


def plot_sink(request, columns):
    sent.append((json.loads(request), columns))
    return answer['problem']


cyxwiz_capture.install(lambda *a: None, plot_sink, closed.append)
import cyxwiz as cx  # noqa: E402  (found by the bundled finder, not sys.path)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None


def last():
    return sent[-1]


class PlotApiTest(unittest.TestCase):
    def setUp(self):
        sent.clear()
        closed.clear()
        answer['problem'] = ''

    def test_import_comes_from_the_tools_folder(self):
        self.assertEqual(os.path.dirname(os.path.abspath(cx.__file__)), TOOLS)

    def test_scatter_on_named_columns(self):
        df = {'pop': [10, 20, 30], 'track': [1.5, 2.5, None], 'kind': ['album', 'single', 'album']}
        p = cx.scatter('pop', 'track', data=df, color='kind', title='Popularity')
        request, columns = last()
        spec = request['spec']
        self.assertEqual((spec['kind'], spec['x'], spec['y'], spec['color'], spec['title']),
                         ('scatter', 'pop', ['track'], 'kind', 'Popularity'))
        self.assertEqual(request['columns'], ['pop', 'track', 'kind'])
        self.assertEqual(request['source'], 'df')
        self.assertTrue(request['file'].endswith('test_cyxwiz_plot.py') and request['line'] > 0)
        self.assertEqual([c[1] for c in columns], ['num', 'num', 'text'])
        self.assertTrue(math.isnan(columns[1][2][2]))
        self.assertEqual(columns[2][2], ['album', 'single', 'album'])
        self.assertEqual(p.title, 'Popularity')

    def test_arrays_and_default_title(self):
        cx.plot([3, 1, 2])
        spec = last()[0]['spec']
        self.assertEqual((spec['kind'], spec['y'], spec.get('x', '')), ('line', ['y'], ''))
        self.assertEqual(spec['title'], 'plot: y')
        self.assertEqual(last()[0]['source'], 'arrays')
        cx.bar(['a', 'b', 'a'])
        self.assertEqual(last()[0]['spec']['title'], 'bar: x')

    def test_options_and_value_columns(self):
        cx.hist([1, 2, 2, 3], bins=40, logy=True, xlabel='minutes', rows=2, density=True)
        spec = last()[0]['spec']
        self.assertEqual((spec['kind'], spec['bins'], spec['log_y'], spec['x_label'], spec['density']),
                         ('histogram', 40, True, 'minutes', True))
        self.assertEqual(spec['rows'], {'mode': 'first', 'first': 2})
        cx.network(['a', 'b'], ['b', 'c'], weight=[1, 2])
        spec = last()[0]['spec']
        self.assertEqual((spec['kind'], spec['x'], spec['y'], spec['value']), ('network', 'x', ['y'], 'value'))
        cx.heatmap(['r1', 'r2'], ['c1', 'c2'], [5, 6], bar_layout='grouped')
        self.assertEqual(last()[0]['spec']['value'], 'value')

    def test_same_column_twice_is_sent_once(self):
        df = {'a': [1, 2], 'b': [3, 4]}
        cx.scatter('a', 'b', data=df, color='a')
        self.assertEqual(last()[0]['columns'], ['a', 'b'])

    def test_mistakes_raise_in_python(self):
        with self.assertRaisesRegex(ValueError, 'pass the table as data='):
            cx.scatter('a', 'b')
        with self.assertRaisesRegex(ValueError, r"column 'zz' is not in data= \(columns: a\)"):
            cx.hist('zz', data={'a': [1]})
        with self.assertRaisesRegex(ValueError, 'has 2 values, the other columns 3'):
            cx.scatter([1, 2, 3], [1, 2])
        answer['problem'] = 'scatter needs y values'
        with self.assertRaisesRegex(ValueError, 'scatter needs y values'):
            cx.scatter([1, 2])

    def test_update_and_close(self):
        df = {'a': [1, 2], 'b': [3, 4]}
        p = cx.scatter('a', 'b', data=df, title='T', color='a')
        p.update(data={'a': [5, 6, 7], 'b': [8, 9, 10]})
        request, columns = last()
        self.assertEqual(request['spec']['title'], 'T')
        self.assertEqual(request['spec']['color'], 'a')
        self.assertEqual(list(columns[0][2]), [5.0, 6.0, 7.0])
        p.close()
        cx.close('other')
        self.assertEqual(closed, ['T', 'other'])

    def test_outside_the_engine(self):
        saved = cyxwiz_capture._plot_sink
        cyxwiz_capture._plot_sink = None
        try:
            with self.assertRaisesRegex(RuntimeError, 'CyxWiz Engine'):
                cx.scatter([1], [2])
        finally:
            cyxwiz_capture._plot_sink = saved

    @unittest.skipIf(np is None, 'numpy not installed')
    def test_numpy_surface_and_image(self):
        z = np.arange(12, dtype=float).reshape(3, 4)
        cx.plot_surface(z, title='grid')
        request, columns = last()
        self.assertEqual(request['spec']['surface_from'], 'grid')
        self.assertEqual(request['spec']['y'], ['c1', 'c2', 'c3', 'c4'])
        self.assertEqual(list(columns[0][2]), [0.0, 4.0, 8.0])
        self.assertEqual(request['source'], 'array 3x4')
        cx.imshow(np.zeros((2, 3)))
        spec = last()[0]['spec']
        self.assertEqual((spec['kind'], spec['image_mode'], spec['image_width'], len(spec['y'])), ('image', 'one_row', 3, 6))
        cx.scatter(np.array([1, 2]), np.array([True, False]))
        self.assertEqual(last()[1][1][1], 'num')

    @unittest.skipIf(pd is None, 'pandas not installed')
    def test_pandas_series_names_and_dates(self):
        df = pd.DataFrame({'when': pd.to_datetime(['2026-01-01', '2026-01-02']), 'v': [1.0, None]})
        cx.scatter(df.when, df.v)
        request, columns = last()
        self.assertEqual(request['columns'], ['when', 'v'])
        self.assertEqual(columns[0][1], 'text')
        self.assertTrue(columns[0][2][0].startswith('2026-01-01'))
        cx.box('v', data=df)
        self.assertEqual(last()[0]['source'], 'df')


if __name__ == '__main__':
    unittest.main()
