"""Tests for cyxwiz_vars (TOFIX133 P5): run with the folder that holds
cyxwiz_vars.py as the first argument (the build's <exe dir>/python_tools).
NumPy and pandas parts run when the interpreter has them."""

import os
import sys
import tempfile
import types
import unittest

TOOLS = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, TOOLS)
import cyxwiz_vars as cv  # noqa: E402
sys.path.remove(TOOLS)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None


class Point:
    def __init__(self):
        self.x = 1
        self.y = 'two'
        self._hidden = 3


def by_name(rows):
    return {r['name']: r for r in rows}


class PlainValuesTest(unittest.TestCase):
    def setUp(self):
        self.ns = {
            '__name__': '__main__', '_private': 1, 'In': [], 'Out': {},
            'epochs': 20, 'learning_rate': 0.001, 'done': True, 'name': 'sentiment run',
            'labels': ['negative', 'neutral', 'positive'],
            'config': {'epochs': 20, 'lr': 0.001, 'labels': ['a', 'b']},
            'nothing': None, 'point': Point(), 'big': list(range(1000)),
            'os': os, 'helper': by_name, 'Point': Point,
        }

    def test_hidden_names_and_definitions(self):
        names = by_name(cv.list_variables(self.ns))
        for hidden in ('__name__', '_private', 'In', 'Out', 'os', 'helper', 'Point'):
            self.assertNotIn(hidden, names)
        everything = by_name(cv.list_variables(self.ns, show_all=True))
        self.assertEqual(everything['os']['kind'], 'module')
        self.assertEqual(everything['helper']['kind'], 'function')
        self.assertEqual(everything['Point']['kind'], 'class')

    def test_rows(self):
        rows = cv.list_variables(self.ns)
        self.assertEqual([r['name'] for r in rows], sorted([r['name'] for r in rows], key=str.lower))
        n = by_name(rows)
        self.assertEqual((n['epochs']['type'], n['epochs']['kind'], n['epochs']['value']), ('int', 'number', '20'))
        self.assertEqual(n['done']['kind'], 'number')
        self.assertEqual((n['name']['size'], n['name']['value']), ('len=13', "'sentiment run'"))
        self.assertEqual((n['labels']['size'], n['labels']['kind']), ('(3,)', 'collection'))
        self.assertEqual(n['config']['size'], '3 items')
        self.assertTrue(n['config']['expandable'] and n['point']['expandable'])
        self.assertFalse(n['epochs']['expandable'] or n['nothing']['expandable'])
        self.assertEqual(n['epochs']['memory'], -1)
        self.assertTrue(n['labels']['viewable'])
        self.assertFalse(n['config']['viewable'])

    def test_values_are_bounded(self):
        self.ns['huge'] = 'x' * 100000
        row = by_name(cv.list_variables(self.ns))['huge']
        self.assertLessEqual(len(row['value']), cv.MAX_VALUE + 3)
        self.assertLessEqual(len(by_name(cv.list_variables(self.ns))['big']['value']), cv.MAX_VALUE + 3)

    def test_children_and_paths(self):
        kids = by_name(cv.children(self.ns, [['name', 'config']]))
        self.assertEqual(set(kids), {"'epochs'", "'lr'", "'labels'"})
        labels_step = kids["'labels'"]['step']
        inner = cv.children(self.ns, [['name', 'config'], labels_step])
        self.assertEqual([r['name'] for r in inner], ['[0]', '[1]'])
        attrs = by_name(cv.children(self.ns, [['name', 'point']]))
        self.assertEqual(set(attrs), {'x', 'y'})
        cut = cv.children(self.ns, [['name', 'big']])
        self.assertEqual(len(cut), cv.MAX_CHILDREN + 1)
        self.assertEqual(cut[-1], {'more': 1000 - cv.MAX_CHILDREN})

    def test_digest_follows_the_value(self):
        before = by_name(cv.list_variables(self.ns))['labels']['digest']
        self.ns['labels'] = ['negative', 'positive']
        self.assertNotEqual(before, by_name(cv.list_variables(self.ns))['labels']['digest'])

    def test_copy_delete_and_list_table(self):
        self.assertEqual(cv.value_text(self.ns, [['name', 'labels']]), "['negative', 'neutral', 'positive']")
        self.assertTrue(cv.value_text(self.ns, [['name', 'big']], limit=10).startswith('[0, 1, 2, '))
        self.assertIn('more characters', cv.value_text(self.ns, [['name', 'big']], limit=10))
        self.assertEqual(cv.delete(self.ns, [['name', 'done']]), '')
        self.assertNotIn('done', self.ns)
        self.assertTrue(cv.delete(self.ns, [['name', 'config'], ['item', 0]]))  # refused: not a variable
        t = cv.table(self.ns, [['name', 'labels']])
        self.assertEqual((t['kind'], t['rows'], t['columns'][0]['values']), ('list', 3, ['negative', 'neutral', 'positive']))
        self.ns['grid'] = [[1, 2], [3, 4], [5]]
        t = cv.table(self.ns, [['name', 'grid']])
        self.assertEqual([c['values'] for c in t['columns']], [[1, 3, 5], [2, 4, None]])
        self.assertIn('error', cv.table(self.ns, [['name', 'config']]))
        self.assertIn('error', cv.table(self.ns, [['name', 'gone']]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'grid.csv')
            self.assertEqual(cv.save_csv(self.ns, [['name', 'grid']], path), '')
            with open(path, encoding='utf-8') as handle:
                self.assertEqual(handle.read().splitlines(), ['0,1', '1,2', '3,4', '5,'])

    def test_no_library_is_imported(self):
        before = set(sys.modules)
        cv.list_variables(self.ns)
        self.assertEqual(set(sys.modules) - before, set())


@unittest.skipIf(np is None, 'numpy not installed')
class ArrayTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ns = {'weights': rng.normal(size=(128, 64)).astype(np.float32), 'scores': np.linspace(0, 1, 5000),
                   'batch': np.arange(32 * 28 * 28, dtype=np.uint8).reshape(32, 28, 28), 'one': np.float64(2.5)}

    def test_rows(self):
        n = by_name(cv.list_variables(self.ns))
        w = n['weights']
        self.assertEqual((w['type'], w['kind'], w['size'], w['memory']), ('ndarray[float32]', 'array', '(128, 64)', 32768))
        self.assertTrue(w['value'].startswith('float32 · min -3.9'))
        self.assertEqual(n['scores']['size'], '(5000,)')
        self.assertEqual(n['scores']['value'], 'float64 · min 0 · max 1 · mean 0.5')
        self.assertTrue(w['viewable'] and not w['expandable'])
        self.assertEqual(n['one']['kind'], 'number')

    def test_tables(self):
        t = cv.table(self.ns, [['name', 'weights']], max_rows=10)
        self.assertEqual((t['kind'], t['rows'], t['shown'], len(t['columns'])), ('array', 128, 10, 64))
        self.assertAlmostEqual(t['columns'][0]['values'][0], float(self.ns['weights'][0, 0]), places=5)
        t = cv.table(self.ns, [['name', 'batch']], index=[3])
        self.assertEqual((t['slice'], t['rows'], len(t['columns'])), ([3], 28, 28))
        self.assertEqual(t['columns'][0]['values'][0], int(self.ns['batch'][3, 0, 0]))
        self.assertEqual(cv.table(self.ns, [['name', 'batch']], index=[99])['slice'], [31])
        t = cv.table(self.ns, [['name', 'scores']])
        self.assertEqual((t['columns'][0]['name'], t['rows']), ('value', 5000))

    def test_digest(self):
        before = by_name(cv.list_variables(self.ns))['weights']['digest']
        self.ns['weights'][0, 0] += 1
        self.assertNotEqual(before, by_name(cv.list_variables(self.ns))['weights']['digest'])


@unittest.skipIf(pd is None, 'pandas not installed')
class FrameTest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'text': ['good movie', 'bad plot', 'fine', 'loved it', 'boring'] * 2000,
                           'label': [2, 0, 1, 2, 0] * 2000, 'length': [10, 8, 4, 8, 6] * 2000})
        self.ns = {'df': df, 'counts': df['label'].value_counts()}

    def test_rows(self):
        n = by_name(cv.list_variables(self.ns))
        self.assertEqual((n['df']['kind'], n['df']['size']), ('table', '(10000, 3)'))
        self.assertTrue(n['df']['value'].startswith('columns: text ('))
        self.assertGreater(n['df']['memory'], 0)
        self.assertTrue(n['df']['expandable'] and n['df']['viewable'])
        self.assertTrue(n['counts']['value'].startswith('count, dtype int64: 2 → 4000'))
        cols = by_name(cv.children(self.ns, [['name', 'df']]))
        self.assertEqual(cols['label']['value'], 'min 0 · max 2 · mean 1')
        self.assertEqual(cols['text']['value'], 'good movie, bad plot, fine, loved it, boring, ...')

    def test_table(self):
        t = cv.table(self.ns, [['name', 'df']], max_rows=200)
        self.assertEqual((t['kind'], t['rows'], t['shown'], t['index'][:2]), ('frame', 10000, 200, [0, 1]))
        self.assertEqual([c['name'] for c in t['columns']], ['text', 'label', 'length'])
        self.assertEqual(t['columns'][1]['dtype'], 'int64')
        self.assertEqual(t['columns'][0]['values'][:2], ['good movie', 'bad plot'])
        t = cv.table(self.ns, [['name', 'counts']])
        self.assertEqual((t['rows'], t['columns'][0]['name']), (3, 'count'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
