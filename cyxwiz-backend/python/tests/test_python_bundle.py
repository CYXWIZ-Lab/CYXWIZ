"""nodes.bundle (TOFIX125 P3 step 3.5): Softsign and Snake from
example_nodes.py are written as one bundle; the manifest matches the
decorators, records the gradient check, and hashes the code. Run like
test_python_modules.py.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()

from pycyxwiz import nodes  # noqa: E402
from example_nodes import Snake, Softsign, SoftsignWrong  # noqa: E402

work = tempfile.mkdtemp(prefix="cyxnode_test_")
try:
    path = nodes.bundle(Softsign, Snake, SoftsignWrong, folder=work, name="example-acts")
    check(os.path.normpath(path) == os.path.join(work, "example-acts.cyxnode"),
          "the bundle is <folder>/<name>.cyxnode")
    check(sorted(os.listdir(path)) == ["node.json", "node.py"], "it holds node.json and node.py")

    source = os.path.join(HERE, "example_nodes.py")
    with open(source, "rb") as a, open(os.path.join(path, "node.py"), "rb") as b:
        code = a.read()
        check(code == b.read(), "node.py is the author's file, unchanged")
    with open(os.path.join(path, "node.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)

    check(manifest["bundle_format"] == 1 and manifest["id"] == "com.example.acts", "bundle format and id")
    check(manifest["content_hash"] == "sha256:" + hashlib.sha256(code).hexdigest(),
          "content_hash is the SHA-256 of node.py")
    check(manifest["requires"]["pycyxwiz"] == ">=" + cyx.get_version(), "requires this pycyxwiz or newer")
    by_name = {entry["name"]: entry for entry in manifest["nodes"]}
    check(sorted(by_name) == ["Snake", "Softsign", "SoftsignWrong"], "one entry per node")

    snake = by_name["Snake"]
    check(snake["entry"] == "node.py:Snake" and snake["tier"] == "module", "Snake: entry and tier")
    check(snake["parameters"] == nodes.describe(Snake)["parameters"], "Snake: settings as describe() gives them")
    check(snake["shape_rule"] == "same_as_input" and snake["role"] == "activation", "Snake: shape rule, role")
    check(snake["verification"]["status"] == "passed"
          and snake["verification"]["max_relative_error"] < snake["verification"]["tolerance"],
          "Snake: gradient check passed and recorded (%.1e)" % snake["verification"]["max_relative_error"])
    check(by_name["Softsign"]["tier"] == "activation"
          and by_name["Softsign"]["verification"]["status"] == "passed", "Softsign: tier 1, passed")
    wrong = by_name["SoftsignWrong"]["verification"]
    check(wrong["status"] == "failed" and wrong["where"] == "input",
          "a wrong gradient is recorded as failed, the bundle is still written")

    again = nodes.bundle(Snake, folder=work, name="example-acts", check_gradients=False)
    with open(os.path.join(again, "node.json"), encoding="utf-8") as handle:
        rewritten = json.load(handle)
    check([e["name"] for e in rewritten["nodes"]] == ["Snake"]
          and rewritten["nodes"][0]["verification"]["status"] == "not_checked",
          "bundling again replaces the manifest; a skipped check is recorded as not_checked")

    for args, kwargs, text in (((), {}, "needs at least one node class"),
                               ((Snake,), {"name": "Bad Name"}, "must be lower case"),
                               ((cyx.Module,), {}, "is not a node")):
        try:
            nodes.bundle(*args, folder=work, **kwargs)
            check(False, "refused: " + text)
        except (ValueError, TypeError) as error:
            check(text in str(error), "refused: %s" % error)

    @nodes.node(id="com.example.other", name="Other", version="1.0.0")
    class Other(cyx.Module):
        def forward(self, x):
            return x

        def backward(self, grad_output):
            return grad_output

    try:
        nodes.bundle(Snake, Other, folder=work)
        check(False, "nodes with different ids are refused")
    except ValueError as error:
        check("share one id" in str(error), "nodes with different ids are refused")
finally:
    shutil.rmtree(work, ignore_errors=True)

check.finish("test_python_bundle")
