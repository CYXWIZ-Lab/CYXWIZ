"""Checks the hand-written extension docs against the code (TOFIX125).

    python check_extension_docs.py <folder holding pycyxwiz> [<folder of its DLLs>]

For every guide in docs/extending (the generated reference pages are checked
by generate_extension_reference.py --check) and docs/usage/extension_nodes.md:

- Python samples run. The ```python blocks of a page run in order, in one
  namespace, in a fresh scratch folder, with the pycyxwiz under test. A block
  that must not run is preceded by <!-- not run: <reason> -->.
- C++ and CMake samples are real code. A ```cpp or ```cmake block is
  preceded by <!-- excerpt: <path from the repository root> -->, and every
  line of it appears, in order, in that file (a file the build compiles or
  the SDK installs). "// ..." and "# ..." lines stand for omitted code. A
  block that is not an excerpt says <!-- not compiled: <reason> -->.
- The page says which versions it was checked against:
  "Checked against plugin API <x.y.z> and pycyxwiz <version>." with the
  versions of this tree. When either changes, every page fails until
  someone re-reads it against the code and updates the line.
- Relative links resolve, in these pages, docs/usage/README.md and the root
  README.md.

Exit code 0 when everything holds; otherwise each problem is printed with its
page and line.
"""
import io
import os
import re
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import generate_extension_reference as reference  # noqa: E402

GENERATED = {"python-api-reference.md", "tensor-operations.md", "sdk-contents.md"}


def guides():
    folder = os.path.join(REPO, "docs", "extending")
    pages = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if n.endswith(".md") and n not in GENERATED]
    return pages + [os.path.join(REPO, "docs", "usage", "extension_nodes.md")]


def rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def blocks(text):
    """(language, first line number, code, marker) for each fenced block."""
    lines = text.splitlines()
    found, i = [], 0
    while i < len(lines):
        m = re.match(r"^```(\w*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        language, start = m.group(1), i + 1
        marker = ""
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0:
            mm = re.match(r"^<!--\s*(excerpt|not run|not compiled):\s*(.+?)\s*-->$", lines[j].strip())
            if mm:
                marker = (mm.group(1), mm.group(2))
        end = start
        while end < len(lines) and not lines[end].startswith("```"):
            end += 1
        found.append((language, start + 1, "\n".join(lines[start:end]), marker))
        i = end + 1
    return found


def check_excerpt(code, path):
    full = os.path.join(REPO, *path.split("/"))
    if not os.path.isfile(full):
        return "excerpt file {} does not exist".format(path)
    with open(full, encoding="utf-8", errors="replace") as f:
        source = [l.strip() for l in f.read().splitlines()]
    position = 0
    for line in code.splitlines():
        s = line.strip()
        if not s or s in ("// ...", "# ...", "..."):
            continue
        try:
            position = source.index(s, position) + 1
        except ValueError:
            return "line not found (in order) in {}: {}".format(path, s)
    return None


def run_python_blocks(page, python_blocks, problems):
    scratch = tempfile.mkdtemp(prefix="cyxwiz_docs_")
    namespace = {"__name__": "__docs__"}
    old_cwd = os.getcwd()
    os.chdir(scratch)
    try:
        for line, code in python_blocks:
            out = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = out
            try:
                exec(compile(code, "{}:{}".format(rel(page), line), "exec"), namespace)
            except BaseException:  # noqa: BLE001 - every failure of a sample is a doc problem
                problems.append("{}:{}: the Python sample fails:\n{}".format(
                    rel(page), line, traceback.format_exc(limit=3)))
                return
            finally:
                sys.stdout = old_stdout
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def check_links(page, text, problems):
    for n, line in enumerate(text.splitlines(), 1):
        for target in re.findall(r"\]\(([^)\s]+)\)", line):
            if re.match(r"^[a-z]+://", target) or target.startswith("#") or target.startswith("mailto:"):
                continue
            path = target.split("#", 1)[0]
            full = os.path.normpath(os.path.join(os.path.dirname(page), path))
            if not os.path.exists(full):
                problems.append("{}:{}: broken link {}".format(rel(page), n, target))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    folder = os.path.abspath(args[0])
    pc = reference.import_pycyxwiz(folder, os.path.abspath(args[1]) if len(args) > 1 else folder)
    stamp = "Checked against plugin API {} and pycyxwiz {}.".format(reference.plugin_api_version(),
                                                                     pc.get_version())
    problems = []
    for page in guides():
        with open(page, encoding="utf-8") as f:
            text = f.read()
        if stamp not in text:
            problems.append("{}: missing or old version line; after re-reading the page against the code, "
                            "it must say: {}".format(rel(page), stamp))
        python_blocks = []
        for language, line, code, marker in blocks(text):
            kind = marker[0] if marker else ""
            if language == "python":
                if kind == "not run":
                    continue
                if kind:
                    problems.append("{}:{}: a Python block takes <!-- not run: reason --> or nothing".format(
                        rel(page), line))
                python_blocks.append((line, code))
            elif language in ("cpp", "c++", "cmake"):
                if kind == "excerpt":
                    error = check_excerpt(code, marker[1])
                    if error:
                        problems.append("{}:{}: {}".format(rel(page), line, error))
                elif kind != "not compiled":
                    problems.append("{}:{}: a {} block needs <!-- excerpt: path --> or "
                                    "<!-- not compiled: reason --> before it".format(rel(page), line, language))
        if python_blocks:
            run_python_blocks(page, python_blocks, problems)
        check_links(page, text, problems)
    for extra in (os.path.join(REPO, "README.md"), os.path.join(REPO, "docs", "usage", "README.md")):
        with open(extra, encoding="utf-8") as f:
            check_links(extra, f.read(), problems)
    for name in sorted(GENERATED):
        with open(os.path.join(REPO, "docs", "extending", name), encoding="utf-8") as f:
            check_links(os.path.join(REPO, "docs", "extending", name), f.read(), problems)

    for problem in problems:
        print("FAIL " + problem)
    if problems:
        print("{} problem(s) in the extension docs".format(len(problems)))
        return 1
    print("extension docs: samples run, excerpts match, versions current, links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
