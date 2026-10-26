"""pycyxwiz.nodes: describe a Python module as a CyxWiz node (TOFIX125 P3).

Two ways to write a node (Part C4):

- Tier 1, @nodes.activation: write f(x) and its derivative df(x); the module,
  its forward and backward are generated. Output shape = input shape.
- Tier 2, @nodes.node on a pycyxwiz.Module subclass: write forward and
  backward yourself, optionally build(input_shape) and output_shape.
- A loss, @nodes.loss on a pycyxwiz.Loss subclass: forward(predictions,
  targets) gives one value, backward gives dLoss/dpredictions (P7).

Settings are declared with nodes.Param as class attributes. A graph stores
settings as text; create() converts and checks them.

This file is compiled into pycyxwiz and installed as the submodule
pycyxwiz.nodes when pycyxwiz is imported.
"""

import re as _re

try:
    _cyx  # set by pycyxwiz when it installs this file as pycyxwiz.nodes
except NameError:
    import pycyxwiz as _cyx

__all__ = ["Param", "activation", "node", "loss", "create", "output_shape", "describe", "gradcheck",
           "bundle", "is_node", "is_loss"]

BUNDLE_FORMAT = 1

_ID_PATTERN = _re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = _re.compile(r"^\d+\.\d+\.\d+$")
_ROLES = ("activation", "layer")
_TARGETS = ("categorical", "continuous")
_TYPES = {float: "float", int: "int", bool: "bool", str: "string"}


class Param:
    """A node setting: Param(float, default=1.0, min=0.01, description="...")."""

    def __init__(self, type=float, default=None, min=None, max=None, choices=None, description=""):
        if type not in _TYPES:
            raise TypeError("Param type must be float, int, bool or str")
        self.type = type
        self.min = min
        self.max = max
        self.choices = list(choices) if choices is not None else None
        self.description = description
        self.name = None
        if default is None:
            default = {float: 0.0, int: 0, bool: False, str: ""}[type] if choices is None else self.choices[0]
        self.default = self.convert(default, check=False)
        self.check(self.default, "the default of this Param")

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get("_cyxwiz_settings", {}).get(self.name, self.default)

    def __set__(self, instance, value):
        raise AttributeError("setting '%s' is read-only inside the node; set it in the graph" % self.name)

    def convert(self, value, check=True):
        """A graph value (text, or already typed) as this Param's type."""
        label = self.name or "value"
        if self.type is bool:
            if isinstance(value, str):
                if value.strip().lower() in ("true", "1", "yes"):
                    result = True
                elif value.strip().lower() in ("false", "0", "no", ""):
                    result = False
                else:
                    raise ValueError("%s must be true or false, not '%s'" % (label, value))
            else:
                result = bool(value)
        elif self.type is str:
            result = str(value)
        else:
            try:
                result = self.type(value)
            except (TypeError, ValueError):
                raise ValueError("%s must be a%s number, not '%s'"
                                 % (label, "" if self.type is float else "n integer", value)) from None
        if check:
            self.check(result, label)
        return result

    def check(self, value, label):
        if self.min is not None and value < self.min:
            raise ValueError("%s must be at least %s" % (label, self.min))
        if self.max is not None and value > self.max:
            raise ValueError("%s must be at most %s" % (label, self.max))
        if self.choices is not None and value not in self.choices:
            raise ValueError("%s must be one of %s" % (label, ", ".join(map(str, self.choices))))

    def validation(self):
        """The rule as ParameterDefinition writes it: "0.01-", "0-1", "a|b"."""
        if self.choices is not None:
            return "|".join(map(str, self.choices))
        if self.min is None and self.max is None:
            return ""
        return "%s-%s" % ("" if self.min is None else self.min, "" if self.max is None else self.max)

    def describe(self):
        return {"name": self.name, "type": _TYPES[self.type], "default": _text(self.default),
                "validation": self.validation(), "description": self.description,
                "choices": [_text(c) for c in self.choices] if self.choices is not None else []}


def _text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _params(cls):
    found = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, Param):
                found[name] = value
    return found


def _check_identity(id, name, version, role, roles=_ROLES):
    if not isinstance(id, str) or not _ID_PATTERN.match(id):
        raise ValueError("id '%s' must be lower case: a-z, 0-9, '.', '-', '_' (for example com.you.my-nodes)" % id)
    if not isinstance(name, str) or not _re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
        raise ValueError("name '%s' must start with a letter and use letters, digits and '_'" % name)
    if not isinstance(version, str) or not _VERSION_PATTERN.match(version):
        raise ValueError("version '%s' must look like 1.0.0" % version)
    if role not in roles:
        raise ValueError("role must be one of %s" % ", ".join(roles))


def _attach(cls, tier, id, name, version, category, role, description, display_name, roles=_ROLES):
    _check_identity(id, name, version, role, roles)
    cls.__cyxwiz_node__ = {
        "id": id,
        "name": name,
        "version": version,
        "tier": tier,
        "display_name": display_name or name,
        "category": category,
        "role": role,
        "description": description or (cls.__doc__ or "").strip().split("\n")[0],
    }
    return cls


def activation(id, name, version, category="Activation", description="", display_name=None):
    """Tier 1: a class with f(self, x) and df(self, x) becomes a module.

    forward returns f(x); backward returns grad_output * df(x).
    """
    def wrap(spec):
        for method in ("f", "df"):
            if not callable(getattr(spec, method, None)):
                raise TypeError("@nodes.activation needs %s(self, x) on %s" % (method, spec.__name__))

        def forward(self, x):
            self._cyxwiz_x = x
            return self.f(x)

        def backward(self, grad_output):
            return grad_output * self.df(self._cyxwiz_x)

        namespace = {k: v for k, v in vars(spec).items() if k not in ("__dict__", "__weakref__")}
        namespace.update({"forward": forward, "backward": backward})
        cls = type(spec.__name__, (_cyx.Module,), namespace)
        cls.__qualname__ = spec.__qualname__
        return _attach(cls, "activation", id, name, version, category, "activation", description, display_name)
    return wrap


def node(id, name, version, category="Custom", role="layer", description="", display_name=None):
    """Tier 2: a pycyxwiz.Module subclass with forward and backward."""
    def wrap(cls):
        if not (isinstance(cls, type) and issubclass(cls, _cyx.Module)):
            raise TypeError("@nodes.node goes on a subclass of pycyxwiz.Module")
        for method in ("forward", "backward"):
            if method not in {k for klass in cls.__mro__ if klass is not _cyx.Module for k in vars(klass)}:
                raise TypeError("@nodes.node needs %s on %s" % (method, cls.__name__))
        return _attach(cls, "module", id, name, version, category, role, description, display_name)
    return wrap


def loss(id, name, version, target="categorical", category="Loss", description="", display_name=None):
    """A loss node: a pycyxwiz.Loss subclass with forward and backward.

    forward(self, predictions, targets) returns one value; backward(self,
    predictions, targets) returns dLoss/dpredictions with the predictions'
    shape, the mean scaling included (as the built-in losses).
    target: "categorical" (targets are one-hot rows as wide as the model's
    output; accuracy by argmax) or "continuous" (target values; regression
    metrics).
    """
    def wrap(cls):
        if not (isinstance(cls, type) and issubclass(cls, _cyx.Loss)):
            raise TypeError("@nodes.loss goes on a subclass of pycyxwiz.Loss")
        for method in ("forward", "backward"):
            if method not in {k for klass in cls.__mro__ if klass is not _cyx.Loss for k in vars(klass)}:
                raise TypeError("@nodes.loss needs %s(self, predictions, targets) on %s" % (method, cls.__name__))
        if target not in _TARGETS:
            raise ValueError("target must be one of %s" % ", ".join(_TARGETS))
        _attach(cls, "loss", id, name, version, category, "loss", description, display_name, roles=("loss",))
        cls.__cyxwiz_node__["target"] = target
        return cls
    return wrap


def is_node(cls):
    return isinstance(getattr(cls, "__cyxwiz_node__", None), dict)


def is_loss(cls):
    return is_node(cls) and cls.__cyxwiz_node__["role"] == "loss"


def _require_node(cls):
    if not is_node(cls):
        raise TypeError("%r is not a node; decorate it with @nodes.node, @nodes.activation or @nodes.loss"
                        % (cls,))


def _same_as_input(cls):
    for klass in cls.__mro__:
        if klass is _cyx.Module:
            return True
        if "output_shape" in vars(klass):
            return False
    return True


def create(cls, settings=None, input_shape=None):
    """A ready module (or loss): settings converted and checked, build() called.

    settings: {name: value}; text values as a graph stores them are converted.
    input_shape: one sample's shape (no batch), as the Engine passes it; a
    loss takes none.
    """
    _require_node(cls)
    params = _params(cls)
    settings = dict(settings or {})
    unknown = sorted(set(settings) - set(params))
    if unknown:
        raise ValueError("%s has no setting %s" % (cls.__cyxwiz_node__["name"], ", ".join(unknown)))
    values = {name: param.convert(settings[name]) if name in settings else param.default
              for name, param in params.items()}
    module = cls.__new__(cls)
    module.__dict__["_cyxwiz_settings"] = values
    module.__init__()
    if input_shape is not None and callable(getattr(module, "build", None)):
        module.build(list(input_shape))
    return module


def output_shape(cls, settings=None, input_shape=()):
    """The output shape for one sample's input shape (no batch)."""
    _require_node(cls)
    if is_loss(cls):
        raise TypeError("%s is a loss: it gives one value, not an output per sample" % cls.__cyxwiz_node__["name"])
    if _same_as_input(cls):
        return list(input_shape)
    return list(create(cls, settings).output_shape(list(input_shape)))


def describe(cls):
    """The node's manifest entry (the fields node.json uses)."""
    _require_node(cls)
    info = dict(cls.__cyxwiz_node__)
    if is_loss(cls):
        # The built-in losses' pins.
        info["inputs"] = [{"name": "Predictions", "type": "Tensor"}, {"name": "Targets", "type": "Labels"}]
        info["outputs"] = [{"name": "Loss", "type": "Loss"}]
    else:
        info["shape_rule"] = "same_as_input" if _same_as_input(cls) else "python"
        info["inputs"] = [{"name": "Input", "type": "Tensor"}]
        info["outputs"] = [{"name": "Output", "type": "Tensor"}]
    info["parameters"] = [p.describe() for p in _params(cls).values()]
    return info


def gradcheck(cls, settings=None, **options):
    """pycyxwiz.gradcheck (gradcheck_loss for a loss) on fresh modules made with these settings."""
    _require_node(cls)
    if is_loss(cls):
        targets = "one_hot" if cls.__cyxwiz_node__["target"] == "categorical" else "values"
        return _cyx.gradcheck_loss(lambda: create(cls, settings), targets=targets, **options)
    return _cyx.gradcheck(lambda shape: create(cls, settings, list(shape)[1:]), **options)


# --- used by the Engine (TOFIX125 P4): a bundle's code, imported privately --

_engine_modules = {}


def _engine_load(folder, private_name):
    """Imports <folder>/node.py as the module `private_name` (once).

    The name includes the bundle's hash, so an edited bundle loads fresh and
    two bundles never collide. The folder is not added to sys.path: a
    bundle must not shadow real packages. No __pycache__ is written into
    the bundle folder (it belongs to the user, or to the repository).
    """
    import importlib.util
    import os
    import sys

    module = _engine_modules.get(private_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(private_name, os.path.join(folder, "node.py"))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load node.py from %s" % folder)
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[private_name]
        raise
    finally:
        sys.dont_write_bytecode = previous
    _engine_modules[private_name] = module
    return module


def _engine_class(folder, private_name, entry_class):
    cls = getattr(_engine_load(folder, private_name), entry_class, None)
    if cls is None:
        raise LookupError("node.py has no class %s" % entry_class)
    _require_node(cls)
    return cls


def _engine_create(folder, private_name, entry_class, settings, input_shape):
    """(module, its C++ address) for the Engine's model builder."""
    module = create(_engine_class(folder, private_name, entry_class), settings, input_shape)
    return module, _cyx._module_address(module)


def _engine_create_loss(folder, private_name, entry_class, settings):
    """(loss, its C++ address) for the Engine's model builder."""
    cls = _engine_class(folder, private_name, entry_class)
    if not is_loss(cls):
        raise TypeError("%s is not a loss node" % entry_class)
    loss = create(cls, settings)
    return loss, _cyx._loss_address(loss)


def _engine_output_shape(folder, private_name, entry_class, settings, input_shape):
    return output_shape(_engine_class(folder, private_name, entry_class), settings, input_shape)


# --- used by the Engine's "Create node from script" (TOFIX125 P4 step 4.5) --

_script_imports = [0]


def _engine_script_nodes(path):
    """The node classes defined in the script at `path`, imported afresh.

    Each call imports the file under a new private name, so an edited script
    is read again. Classes imported into the script from elsewhere are left
    out: a bundle holds the classes its own file defines.
    """
    import importlib.util
    import sys

    _script_imports[0] += 1
    name = "cyxwiz_script_%d" % _script_imports[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot read %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.dont_write_bytecode = previous
    return [value for value in vars(module).values()
            if isinstance(value, type) and is_node(value) and value.__module__ == name]


def _engine_inspect(path):
    """JSON: describe() of every node class the script defines."""
    import json
    return json.dumps([describe(cls) for cls in _engine_script_nodes(path)])


def _engine_check(path):
    """JSON: the gradient check of every node, with its default settings."""
    import json
    results = []
    for cls in _engine_script_nodes(path):
        entry = {"name": cls.__cyxwiz_node__["name"], "status": "not_checked", "max_relative_error": 0.0,
                 "where": "", "error": ""}
        try:
            result = gradcheck(cls)
            if result.error:
                entry["error"] = result.error.splitlines()[0]
            else:
                entry["status"] = "passed" if result.passed else "failed"
                entry["max_relative_error"] = result.max_relative_error
                entry["where"] = result.worst
        except Exception as error:
            entry["error"] = str(error).splitlines()[0]
        results.append(entry)
    return json.dumps(results)


def _engine_bundle(path, folder, name):
    """Bundles every node the script defines into <folder>/<name>.cyxnode."""
    classes = _engine_script_nodes(path)
    if not classes:
        raise ValueError("no node in %s" % path)
    return bundle(*classes, folder=folder, name=name)


def _source_file(cls):
    import inspect
    module = __import__("sys").modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    if not path:
        try:
            path = inspect.getsourcefile(cls)
        except TypeError:
            path = None
    if not path or not __import__("os").path.isfile(path):
        raise ValueError("%s is not defined in a .py file; save it in a file and import it to bundle it"
                         % cls.__name__)
    return __import__("os").path.abspath(path)


def bundle(*classes, folder="nodes", name=None, check_gradients=True):
    """Write the nodes into <folder>/<name>.cyxnode/: node.json and node.py.

    Every class must be a node, share one id and live in one .py file; that
    file is copied as node.py. The gradient check runs with default settings
    and its result is recorded; a failure is recorded, not refused (Part C5).
    Returns the bundle folder.
    """
    import datetime
    import hashlib
    import json
    import os

    if not classes:
        raise ValueError("bundle() needs at least one node class")
    for cls in classes:
        _require_node(cls)
    ids = sorted({cls.__cyxwiz_node__["id"] for cls in classes})
    if len(ids) != 1:
        raise ValueError("the nodes of one bundle share one id; found %s" % ", ".join(ids))
    names = [cls.__cyxwiz_node__["name"] for cls in classes]
    if len(set(names)) != len(names):
        raise ValueError("two nodes are named %s" % sorted(n for n in names if names.count(n) > 1)[0])
    sources = sorted({_source_file(cls) for cls in classes})
    if len(sources) != 1:
        raise ValueError("the nodes of one bundle live in one .py file; found %s" % ", ".join(sources))
    with open(sources[0], "rb") as handle:
        code = handle.read()

    bundle_name = name or names[0].lower()
    if not _re.match(r"^[a-z0-9][a-z0-9_-]*$", bundle_name):
        raise ValueError("bundle name '%s' must be lower case: a-z, 0-9, '-', '_'" % bundle_name)
    path = os.path.join(folder, bundle_name + ".cyxnode")
    today = datetime.date.today().isoformat()

    entries = []
    for cls in classes:
        info = describe(cls)
        entry = {key: info[key] for key in ("name", "version")}
        entry["entry"] = "node.py:" + cls.__name__
        for key in ("tier", "display_name", "category", "role", "description", "shape_rule", "target",
                    "inputs", "outputs", "parameters"):
            if key in info:
                entry[key] = info[key]
        if check_gradients:
            try:
                result = gradcheck(cls)
                if result.error:
                    entry["verification"] = {"status": "not_checked", "checked_on": today,
                                             "error": result.error.splitlines()[0]}
                else:
                    entry["verification"] = {"status": "passed" if result.passed else "failed",
                                             "checked_on": today, "tolerance": result.tolerance,
                                             "max_relative_error": result.max_relative_error,
                                             "where": result.worst}
            except Exception as error:  # the defaults may be refused; record why
                entry["verification"] = {"status": "not_checked", "checked_on": today,
                                         "error": str(error).splitlines()[0]}
        else:
            entry["verification"] = {"status": "not_checked", "checked_on": today,
                                     "error": "the gradient check was skipped"}
        entries.append(entry)

    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "id": ids[0],
        "nodes": entries,
        "requires": {"pycyxwiz": ">=" + _cyx.get_version(), "python_packages": []},
        "content_hash": "sha256:" + hashlib.sha256(code).hexdigest(),
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "node.py"), "wb") as handle:
        handle.write(code)
    with open(os.path.join(path, "node.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return path
