"""Tests for cyxwiz_intel (TOFIX133 P3): run with the folder that holds the
unpacked jedi/parso/pyflakes and cyxwiz_intel.py as the first argument
(the build's <exe dir>/python_tools)."""

import os
import sys
import tempfile
import unittest

TOOLS = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, TOOLS)
import cyxwiz_intel as intel  # noqa: E402
sys.path.remove(TOOLS)

SOURCE = '''import os
import json


def load_rows(path, limit=10):
    """Read rows from a JSON file."""
    with open(path) as handle:
        return json.load(handle)[:limit]


rows = load_rows("data.json")
os.pa
'''


class IntelTest(unittest.TestCase):
    def test_loads_bundled_tools_without_leaving_them_on_path(self):
        st = intel.status()
        self.assertTrue(st['ok'], st['error'])
        self.assertTrue(st['jedi'])
        self.assertNotIn(TOOLS, sys.path)

    def test_module_attribute_completion(self):
        names = {c['name']: c for c in intel.complete(SOURCE, 12, 5)}
        self.assertIn('path', names)       # os.path
        self.assertIn('pardir', names)
        self.assertNotIn('pass', names)    # the old completer offered the keyword
        self.assertEqual(names['path']['complete'], 'th')

    def test_function_completion_has_signature(self):
        items = intel.complete('def load_rows(path, limit=10):\n    pass\nload_', 3, 5)
        self.assertEqual(items[0]['name'], 'load_rows')
        self.assertEqual(items[0]['kind'], 'function')
        self.assertIn('limit=10', items[0]['detail'])

    def test_hover_and_docstring(self):
        info = intel.hover(SOURCE, 11, 9)  # load_rows in "rows = load_rows(...)"
        self.assertEqual(info['name'], 'load_rows')
        self.assertIn('Read rows from a JSON file', info['doc'])
        self.assertIn('path', info['signature'])

    def test_signature_help_active_parameter(self):
        src = 'def load_rows(path, limit=10):\n    pass\nload_rows("a.json", '
        sigs = intel.signatures(src, 3, len('load_rows("a.json", '))
        self.assertEqual(sigs[0]['name'], 'load_rows')
        self.assertEqual(sigs[0]['index'], 1)
        self.assertEqual(sigs[0]['params'][1], 'limit=10')

    def test_definition_in_file_and_in_library(self):
        here = intel.definition(SOURCE, 11, 9)
        self.assertEqual(here[0]['line'], 5)
        lib = intel.definition('import json\njson.load', 2, 6)
        self.assertTrue(lib and lib[0]['path'].endswith(os.path.join('json', '__init__.py')))

    def test_diagnostics(self):
        problems = intel.diagnostics('import os\nprint(undefined_name)\n')
        codes = {p['code']: p for p in problems}
        self.assertEqual(codes['UnusedImport']['severity'], 'warning')
        self.assertEqual(codes['UndefinedName']['severity'], 'error')
        self.assertEqual(codes['UndefinedName']['line'], 2)
        self.assertIn("undefined_name", codes['UndefinedName']['message'])
        # A notebook cell may use names an earlier cell made.
        self.assertEqual(intel.diagnostics('print(df)\n', builtins_extra=['df']), [])
        syntax = intel.diagnostics('def broken(:\n')
        self.assertEqual(syntax[0]['code'], 'syntax')
        self.assertEqual(syntax[0]['line'], 1)

    def test_live_namespace_for_notebooks(self):
        class Frame:
            columns = ['statement', 'status']

            def head(self, n=5):
                """First n rows."""
                return self

        ns = {'df': Frame()}
        names = [c['name'] for c in intel.complete('df.he', 1, 5, namespace=ns)]
        self.assertIn('head', names)

    def test_project_modules_complete(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, 'helpers_p3.py'), 'w') as f:
                f.write('def tokenize(text):\n    return text.split()\n')
            src = 'import helpers_p3\nhelpers_p3.tok'
            names = [c['name'] for c in intel.complete(src, 2, 14, path=os.path.join(root, 'main.py'), project_root=root)]
            self.assertIn('tokenize', names)


if __name__ == '__main__':
    unittest.main(verbosity=1)
