"""pycyxwiz.nodes: describe a Python module as a CyxWiz node (TOFIX125 P3).

Two ways to write a node (Part C4):

- Tier 1, @nodes.activation: write f(x) and its derivative df(x); the module,
  its forward and backward are generated. Output shape = input shape.
- Tier 2, @nodes.node on a pycyxwiz.Module subclass: write forward and
  backward yourself, optionally build(input_shape) and output_shape.

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

__all__ = ["Param", "activation", "node", "create", "describe", "gradcheck", "is_node"]

_ID_PATTERN = _re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = _re.compile(r"^\d+\.\d+\.\d+$")
_ROLES = ("activation", "layer")
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


def _check_identity(id, name, version, role):
    if not isinstance(id, str) or not _ID_PATTERN.match(id):
        raise ValueError("id '%s' must be lower case: a-z, 0-9, '.', '-', '_' (for example com.you.my-nodes)" % id)
    if not isinstance(name, str) or not _re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
        raise ValueError("name '%s' must start with a letter and use letters, digits and '_'" % name)
    if not isinstance(version, str) or not _VERSION_PATTERN.match(version):
        raise ValueError("version '%s' must look like 1.0.0" % version)
    if role not in _ROLES:
        raise ValueError("role must be one of %s" % ", ".join(_ROLES))


def _attach(cls, tier, id, name, version, category, role, description, display_name):
    _check_identity(id, name, version, role)
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


def is_node(cls):
    return isinstance(getattr(cls, "__cyxwiz_node__", None), dict)


def _require_node(cls):
    if not is_node(cls):
        raise TypeError("%r is not a node; decorate it with @nodes.node or @nodes.activation" % (cls,))


def _same_as_input(cls):
    for klass in cls.__mro__:
        if klass is _cyx.Module:
            return True
        if "output_shape" in vars(klass):
            return False
    return True


def create(cls, settings=None, input_shape=None):
    """A ready module: settings converted and checked, build() called.

    settings: {name: value}; text values as a graph stores them are converted.
    input_shape: one sample's shape (no batch), as the Engine passes it.
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
    if _same_as_input(cls):
        return list(input_shape)
    return list(create(cls, settings).output_shape(list(input_shape)))


def describe(cls):
    """The node's manifest entry (the fields node.json uses)."""
    _require_node(cls)
    info = dict(cls.__cyxwiz_node__)
    info["shape_rule"] = "same_as_input" if _same_as_input(cls) else "python"
    info["inputs"] = [{"name": "Input", "type": "Tensor"}]
    info["outputs"] = [{"name": "Output", "type": "Tensor"}]
    info["parameters"] = [p.describe() for p in _params(cls).values()]
    return info


def gradcheck(cls, settings=None, **options):
    """pycyxwiz.gradcheck on fresh modules made with these settings."""
    _require_node(cls)
    return _cyx.gradcheck(lambda shape: create(cls, settings, list(shape)[1:]), **options)
