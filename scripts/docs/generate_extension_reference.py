"""Generates the reference pages of pycyxwiz and the SDK from the code.

    python generate_extension_reference.py <folder holding pycyxwiz> [<folder of its DLLs>] [--check]

Writes, from what the build actually provides:

- docs/python/reference/: every public name of pycyxwiz with its signature
  and docstring, read from the built module. One page per submodule
  (pycyxwiz.linalg, pycyxwiz.signal, ...), one page per group of the
  top-level names (PYTHON_GROUPS: tensors and devices, modules, layers, ...)
  and an index. A top-level name no group claims goes to other.md, so a new
  name is never left out; give it a group.
- docs/sdk/tensor-operations.md: the tensor operations a node can use,
  side by side for C++ (cyxwiz::Tensor in the SDK headers) and Python
  (pycyxwiz.Tensor and the module functions).
- docs/sdk/sdk-contents.md: every header the plugin SDK installs, with
  its summary and the types it declares.

Each page starts with the docs front matter (generated: true). Its
last_reviewed date is kept from the committed page, so regenerating does not
change it; set it by hand after someone has read the page. Internal ticket
numbers and the plan references next to them, copied from header comments
and docstrings, are left out of the pages.

With --check nothing is written: the script exits 1 and names each page whose
committed text differs from what the code gives, so a change to pycyxwiz or
to the SDK headers cannot land without its reference page.
"""
import datetime
import inspect
import os
import re
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS = os.path.join(REPO, "docs")
# The SDK pages: path under docs/ and front matter. redirect_from entries are
# relative to the page, as on every page of the docs.
SDK_META = {"area": "sdk", "audience": "plugin-developer", "owner": "sdk"}
PAGES = {
    "tensor": ("sdk/tensor-operations.md", dict(SDK_META, **{
        "title": "Tensor operations for nodes",
        "description": "The tensor operations a node's forward and backward can use, C++ and Python side by side.",
        "since": "1.0.8",
        "redirect_from": ["../extending/tensor-operations.md"],
    })),
    "sdk": ("sdk/sdk-contents.md", dict(SDK_META, **{
        "title": "What the plugin SDK contains",
        "description": "Every header the plugin SDK installs, with its summary and the types it declares.",
        "since": "1.0.8",
        "redirect_from": ["../extending/sdk-contents.md"],
    })),
}

# The pycyxwiz reference: docs/python/reference/<page>.md.
PYTHON_REFERENCE = "python/reference"
PYTHON_META = {"area": "python", "audience": "user", "owner": "python", "since": "1.0.8"}
# Top-level names of pycyxwiz, grouped into pages: (page, title, description,
# names, base classes whose subclasses belong here, submodule shown on the page).
# A name is claimed by the first group that names it, then by the first base.
PYTHON_GROUPS = [
    ("tensors-and-devices", "Tensors and devices",
     "The Tensor class, the compute devices and starting the backend.",
     ["Tensor", "Device", "DeviceInfo", "DeviceActivationResult", "initialize", "shutdown", "get_version",
      "cuda_available", "opencl_available", "oneapi_available", "metal_available", "get_available_devices",
      "get_device", "set_device", "DataType", "DeviceType", "DeviceKind", "DeviceActivationStage",
      "DeviceIdentityConfidence", "DeviceMetadataStatus"], [], None),
    ("modules-and-models", "Modules and Sequential models",
     "Module, the built-in modules, the Sequential model and the gradient checks.",
     ["Module", "Sequential", "create_module", "ModuleType", "gradcheck", "gradcheck_loss",
      "GradientCheckResult"], ["Module"], None),
    ("layers", "Layers",
     "The layer classes: dense, convolution, pooling, normalization, recurrent and attention layers.",
     ["Layer", "UpsampleMode"], ["Layer"], None),
    ("activations", "Activations and tensor functions",
     "The activation classes and the activation, dropout and flatten functions on tensors.",
     ["Activation", "create_activation", "ActivationType", "relu", "relu6", "sigmoid", "tanh", "softmax", "gelu",
      "elu", "leaky_relu", "mish", "silu", "swish", "hardsigmoid", "hardswish", "dropout", "flatten"],
     ["Activation"], None),
    ("losses", "Losses",
     "The loss classes and their options.",
     ["Loss", "Reduction", "TripletDistanceType"], ["Loss"], None),
    ("optimizers", "Optimizers",
     "The optimizers, learning rate warmup and the optimizer factory.",
     ["Optimizer", "LRWarmup", "create_optimizer", "create_lr_warmup", "OptimizerType", "WarmupType"],
     ["Optimizer"], None),
    ("data-loading", "Data loading",
     "The DuckDB data loader: CSV, Parquet and JSON files, SQL queries and batches.",
     ["DataLoader", "DataLoaderConfig", "ColumnInfo", "BatchIterator", "duckdb_available"], [], None),
    ("audio", "Audio",
     "Loading audio, spectrograms, mel spectrograms, MFCCs and audio augmentation.",
     ["AudioData", "AudioFeatures", "AudioProcessing", "SpectrogramConfig", "MelConfig", "MFCCConfig"], [], None),
    ("tokenizers", "Tokenizers and vocabularies",
     "Text tokenizers and vocabularies.",
     ["Tokenizer", "Vocabulary", "TokenizerType", "BPEInitialUnit", "ByteBPEPiecePolicy",
      "is_sentencepiece_tokenizer_available", "sentencepiece_tokenizer_unavailable_message"], [], None),
    ("rl", "Reinforcement learning",
     "Replay buffers, exploration schedules, RL metrics and plugin environments as Gymnasium environments.",
     ["RLBatch", "RLTransition", "ReplayBuffer", "EpsilonSchedule", "StepResult", "EnvInfo",
      "rl_set_metric_callback", "rl_update_metric"], [], "rl"),
]
PYTHON_OTHER = ("other", "Other names", "Top-level names of pycyxwiz that no reference page groups yet.")
PYTHON_INDEX = {
    "title": "pycyxwiz API reference",
    "description": "Every public name of the pycyxwiz Python module, with its signature and description, one page per part.",
    "redirect_from": ["../../sdk/python-api/index.md", "../../extending/python-api-reference.md"],
}
BACKEND_HEADERS = os.path.join(REPO, "cyxwiz-backend", "include", "cyxwiz")
GENERATOR = "scripts/docs/generate_extension_reference.py"

NOTICE = ("<!-- Generated by {0}; do not edit. Run it after changing pycyxwiz or the SDK "
          "headers (ctest extension_docs_reference fails until you do). -->\n").format(GENERATOR)


# ------------------------------------------------------------ page framing --

def front_matter(meta, last_reviewed):
    """The docs front matter of a generated page (see docs/README.md)."""
    lines = ["---",
             "title: " + meta["title"],
             "description: " + meta["description"],
             "type: reference",
             "area: " + meta["area"],
             "audience: " + meta["audience"],
             "status: experimental",
             "since: " + meta["since"],
             "owner: " + meta["owner"],
             "last_reviewed: " + last_reviewed,
             "generated: true"]
    redirects = meta.get("redirect_from") or []
    if redirects:
        lines.append("redirect_from:")
        lines.extend("  - " + r for r in redirects)
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def committed_last_reviewed(path):
    """last_reviewed of the committed page, or today for a new page."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            m = re.search(r"^last_reviewed:\s*(\d{4}-\d{2}-\d{2})\s*$", f.read(), flags=re.M)
        if m:
            return m.group(1)
    return datetime.date.today().isoformat()


# Plan references that come with a ticket number in the code's comments:
# "P2", "P2 step 2.4", "step 5.1", "Part D6", "D11", "Group C".
_PLAN_PART = re.compile(r"^(?:(?i:TOFIX)\d+\b.*|P\d+(?: step [\d.]+)?|step [\d.]+|Part [A-Z]\d+|D\d+|Group [A-Z])$")


def drop_tickets(text):
    """The text without internal ticket numbers and the plan references next to them.

    "(TOFIX125 P2, plugin API 1.1)" -> "(plugin API 1.1)"; " (TOFIX140)" -> "";
    "(decision D1: X)" keeps X; "TOFIX129 D11: a model ..." -> "a model ...".
    """
    def parenthetical(m):
        inner = m.group(2)
        if "TOFIX" not in inner.upper() and not re.fullmatch(r"Part [A-Z]\d+", inner):
            return m.group(0)
        kept = []
        for part in inner.split(", "):
            part = re.sub(r"^decision D\d+:\s*", "", part.strip())
            if part and not _PLAN_PART.match(part) and not re.fullmatch(r"decision D\d+", part):
                kept.append(part)
        return "{}({})".format(m.group(1), ", ".join(kept)) if kept else ""
    text = re.sub(r"( ?)\(([^()\n]*)\)", parenthetical, text)
    return re.sub(r"\b(?i:TOFIX)\d+(?: (?:P\d+|D\d+|Group [A-Z]|step [\d.]+))*:?\s*", "", text)


# ---------------------------------------------------------------- pycyxwiz --

def import_pycyxwiz(folder, dll_folder):
    dirs = [folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"]
    if os.environ.get("CUDA_PATH"):
        dirs.append(os.path.join(os.environ["CUDA_PATH"], "bin"))
    for d in dirs:
        if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, folder)
    import pycyxwiz
    return pycyxwiz


_TYPE_WORDS = [
    (r"collections\.abc\.Sequence\[typing\.SupportsInt\]", "Sequence[int]"),
    (r"collections\.abc\.Sequence\[typing\.SupportsFloat\]", "Sequence[float]"),
    (r"collections\.abc\.Sequence\[collections\.abc\.Sequence\[typing\.SupportsInt\]\]", "Sequence[Sequence[int]]"),
    (r"collections\.abc\.", ""),
    (r"typing\.SupportsInt", "int"),
    (r"typing\.SupportsFloat", "float"),
    (r"typing\.", ""),
    (r"numpy\.typing\.NDArray", "NDArray"),
    (r"pycyxwiz\.", ""),
]


def tidy(text):
    for pattern, word in _TYPE_WORDS:
        text = re.sub(pattern, word, text)
    # Enum defaults print as <DataType.Float32: 0>, Python types as <class 'float'>.
    text = re.sub(r"<([A-Za-z_][\w.]*): -?\d+>", r"\1", text)
    text = re.sub(r"<class '([\w.]+)'>", r"\1", text)
    # Object defaults print with their address: <MelConfig object at 0x...>.
    text = re.sub(r"<([\w.]+) object at 0x[0-9A-Fa-f]+>", r"\1()", text)
    # object.__init__'s docstring says nothing about the class.
    return text.replace("Initialize self. See help(type(self)) for accurate signature.", "")


def pybind_entries(obj, name):
    """[(signature, description)] from a pybind11 docstring (one per overload)."""
    doc = inspect.getdoc(obj) or ""
    lines = doc.splitlines()
    entries = []
    if lines and lines[0].startswith(name + "(*args, **kwargs)"):
        # Overloaded: "1. sig\n\ndesc\n\n2. sig ..."
        body = "\n".join(lines[1:])
        for part in re.split(r"^\d+\. ", body, flags=re.M)[1:]:
            part_lines = part.strip().splitlines()
            signature = part_lines[0].strip()
            description = " ".join(l.strip() for l in part_lines[1:] if l.strip())
            entries.append((tidy(signature), description))
        return entries
    if lines and lines[0].startswith(name + "("):
        description = " ".join(l.strip() for l in lines[1:] if l.strip())
        return [(tidy(lines[0].strip()), description)]
    return []


def python_entries(obj, name):
    try:
        signature = name + str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = name + "(...)"
    return [(tidy(signature), tidy(" ".join((inspect.getdoc(obj) or "").split())).strip())]


def entries_of(obj, name):
    return pybind_entries(obj, name) or python_entries(obj, name)


def is_pybind_enum(obj):
    return inspect.isclass(obj) and hasattr(obj, "__members__") and hasattr(obj, "value")


SHOWN_DUNDERS = {
    "__add__": "a + b", "__radd__": "b + a", "__sub__": "a - b", "__rsub__": "b - a",
    "__mul__": "a * b", "__rmul__": "b * a", "__truediv__": "a / b", "__rtruediv__": "b / a",
    "__neg__": "-a", "__matmul__": "a @ b", "__getitem__": "a[i]", "__len__": "len(a)",
    "__gt__": "a > b", "__ge__": "a >= b", "__lt__": "a < b", "__le__": "a <= b",
    "__eq__": "a == b", "__ne__": "a != b", "__pow__": "a ** b", "__abs__": "abs(a)",
    "__repr__": "repr(a)",
}


def class_section(cls, name, out, level="###"):
    out.append("{} `{}`\n".format(level, name))
    # The class's own docstring only: inspect.getdoc falls back to a base
    # class's, which would describe Module under every layer.
    own = cls.__dict__.get("__doc__")
    doc = " ".join((inspect.cleandoc(own) if isinstance(own, str) else "").split())
    if doc and not doc.startswith(name + "("):
        out.append(doc + "\n")
    bases = [b.__name__ for b in cls.__mro__[1:] if b.__module__ == cls.__module__ and b.__name__ != name]
    if bases:
        out.append("Derives from " + ", ".join("`{}`".format(b) for b in bases[:1]) + ".\n")
    rows = []
    init = cls.__dict__.get("__init__")
    if init is not None:
        for signature, description in entries_of(init, "__init__"):
            rows.append(("`" + signature.replace("__init__(self, ", name + "(").replace("__init__(self)", name + "()")
                         .replace("__init__(", name + "(") + "`", description))
    for attr in sorted(cls.__dict__):
        if attr.startswith("_") and attr not in SHOWN_DUNDERS:
            continue
        member = cls.__dict__[attr]
        if isinstance(member, property) or type(member).__name__ == "property":
            doc = " ".join((inspect.getdoc(member) or "").split())
            rows.append(("`{}` (property)".format(attr), "" if doc.startswith("(self") else tidy(doc)))
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if not callable(member):
            continue
        for signature, description in entries_of(member, attr):
            if attr in SHOWN_DUNDERS:
                description = "`{}`. {}".format(SHOWN_DUNDERS[attr], description).strip()
            rows.append(("`" + signature + "`", description))
    if is_pybind_enum(cls):
        out.append("Values: " + ", ".join("`{}.{}`".format(name, v) for v in cls.__members__) + "\n")
        return
    if rows:
        out.append("| Member | What it does |\n|---|---|")
        for left, right in rows:
            out.append("| {} | {} |".format(left.replace("|", "\\|"), right.replace("|", "\\|")))
        out.append("")


def function_rows(module, names):
    rows = []
    for name in names:
        for signature, description in entries_of(getattr(module, name), name):
            rows.append("| `{}` | {} |".format(signature.replace("|", "\\|"), description.replace("|", "\\|")))
    return rows


def public_names(module):
    return sorted(n for n in dir(module) if not n.startswith("_"))


def module_parts(module):
    """(functions, enums, classes): the public names of a module that the reference shows."""
    names = public_names(module)
    classes = [n for n in names if inspect.isclass(getattr(module, n)) and not is_pybind_enum(getattr(module, n))]
    enums = [n for n in names if is_pybind_enum(getattr(module, n))]
    functions = [n for n in names if callable(getattr(module, n)) and n not in classes and n not in enums]
    return functions, enums, classes


def module_section(module, title, out, level, class_level="###"):
    functions, enums, classes = module_parts(module)
    if functions:
        out.append("{} Functions\n".format(level))
        out.append("| Function | What it does |\n|---|---|")
        out.extend(function_rows(module, functions))
        out.append("")
    if enums:
        out.append("{} Enumerations\n".format(level))
        for n in enums:
            out.append("- `{}`: {}".format(n, ", ".join("`{}`".format(v) for v in getattr(module, n).__members__)))
        out.append("")
    if classes:
        out.append("{} Classes\n".format(level))
        for n in classes:
            class_section(getattr(module, n), n, out, class_level)


def anchor(heading):
    """The anchor GitHub and the docs site give a heading: lower case, no punctuation, spaces as '-'."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def python_submodules(pc):
    return [n for n in public_names(pc) if isinstance(getattr(pc, n), types.ModuleType)]


def python_top_level(pc):
    """{name: object} of the top-level names the reference shows (no submodules, no enum values)."""
    subs = python_submodules(pc)
    ns = types.SimpleNamespace(**{n: getattr(pc, n) for n in public_names(pc) if n not in subs})
    functions, enums, classes = module_parts(ns)
    return {n: getattr(ns, n) for n in functions + enums + classes}


def python_groups(pc):
    """[(page, title, description, {name: object}, submodule or None)]: the top-level names by page."""
    top = python_top_level(pc)
    claimed = {}
    for page, _, _, names, _, _ in PYTHON_GROUPS:
        for n in names:
            if n in top:
                claimed.setdefault(n, page)
    for n, obj in top.items():
        if n in claimed or not inspect.isclass(obj):
            continue
        for page, _, _, _, bases, _ in PYTHON_GROUPS:
            if any(inspect.isclass(getattr(pc, b, None)) and issubclass(obj, getattr(pc, b)) for b in bases):
                claimed[n] = page
                break
    groups = []
    for page, title, description, _, _, submodule in PYTHON_GROUPS:
        groups.append((page, title, description, {n: o for n, o in top.items() if claimed.get(n) == page}, submodule))
    other = {n: o for n, o in top.items() if n not in claimed}
    if other:
        groups.append(PYTHON_OTHER + (other, None))
    return groups


def submodule_doc(pc, name):
    return " ".join((inspect.getdoc(getattr(pc, name)) or "").split())


def python_page_intro(version, extra=""):
    return ("Part of the [pycyxwiz API reference](index.md), pycyxwiz {}: each name with the signature and "
            "description the module itself reports. Types are shown without the `pycyxwiz.` prefix; "
            "`Sequence[int]` is any list or tuple of integers.{}\n").format(version, extra)


def python_reference(pc):
    """[(path under docs/, meta, body)]: the index and one page per group and submodule."""
    version = pc.get_version()
    subs = python_submodules(pc)
    groups = python_groups(pc)
    in_groups = {g[4] for g in groups if g[4]}
    pages = []
    rows = []  # (page file, title, what it holds, names)

    for page, title, description, members, submodule in groups:
        out = [NOTICE, python_page_intro(version)]
        module_section(types.SimpleNamespace(**members), title, out, "##")
        names = sorted(members, key=str.lower)
        if submodule:
            out.append("## pycyxwiz.{}\n".format(submodule))
            doc = submodule_doc(pc, submodule)
            if doc:
                out.append(doc + "\n")
            module_section(getattr(pc, submodule), "pycyxwiz." + submodule, out, "###", "####")
            sub_parts = module_parts(getattr(pc, submodule))
            names += ["{}.{}".format(submodule, n) for n in sorted(sum(sub_parts, []), key=str.lower)]
        meta = dict(PYTHON_META, title=title, description=description)
        pages.append(("{}/{}.md".format(PYTHON_REFERENCE, page), meta, "\n".join(out).rstrip() + "\n"))
        rows.append((page + ".md", title, description, names))

    for sub in subs:
        if sub in in_groups:
            continue
        doc = submodule_doc(pc, sub)
        out = [NOTICE, python_page_intro(version, " Use it as `pycyxwiz.{}.<name>` after `import pycyxwiz`.".format(sub))]
        if doc:
            out.append(doc + "\n")
        module_section(getattr(pc, sub), "pycyxwiz." + sub, out, "##")
        description = drop_tickets(doc.split(". ")[0]).strip().rstrip(".") if doc else ""
        description = re.sub(r"^pycyxwiz\.{}:\s*".format(sub), "", description)
        description = (description[:1].upper() + description[1:] + ".") if description else             "The pycyxwiz.{} submodule.".format(sub)
        if len(description) > 158:
            description = description[:155].rsplit(" ", 1)[0] + "..."
        meta = dict(PYTHON_META, title="pycyxwiz." + sub, description=description)
        pages.append(("{}/{}.md".format(PYTHON_REFERENCE, sub), meta, "\n".join(out).rstrip() + "\n"))
        rows.append((sub + ".md", "pycyxwiz." + sub, description, sorted(sum(module_parts(getattr(pc, sub)), []),
                                                                        key=str.lower)))

    out = [NOTICE,
           "Every public name of the `pycyxwiz` module in this build, pycyxwiz {}, with the signature and "
           "description the module itself reports. The top-level names (`pycyxwiz.Tensor`, "
           "`pycyxwiz.Sequential`, ...) are grouped by what they do; each submodule (`pycyxwiz.linalg`, "
           "`pycyxwiz.signal`, ...) has its own page.\n".format(version),
           "To learn the library, start with the [Python library guide](../index.md) and the "
           "[quickstart](../quickstart.md). Python nodes use mainly `Tensor`, `Module`, `gradcheck` and "
           "[`pycyxwiz.nodes`](nodes.md); [Tensor operations for nodes](../../sdk/tensor-operations.md) lists the "
           "tensor operations next to their C++ names.\n",
           "## Pages\n",
           "| Page | What it holds |\n|---|---|"]
    for page_file, title, description, _ in rows:
        out.append("| [{}]({}) | {} |".format(title, page_file, description.replace("|", "\\|")))
    out.append("")
    out.append("## Every name\n")
    out.append("Each name links to its entry; a name with a dot is in a submodule.\n")
    for page_file, title, _, names in rows:
        links = []
        for n in names:
            short = n.split(".")[-1]
            obj_is_class = "." not in n and inspect.isclass(getattr(pc, n, None)) and not is_pybind_enum(getattr(pc, n))
            sub = n.split(".")[0] if "." in n else None
            is_class = obj_is_class or (sub and inspect.isclass(getattr(getattr(pc, sub), short, None))
                                        and not is_pybind_enum(getattr(getattr(pc, sub), short)))
            # Classes have their own heading; functions and enumerations are rows of a table.
            target = page_file + ("#" + anchor("`{}`".format(short)) if is_class else "")
            links.append("[`{}`]({})".format(n, target))
        out.append("- **{}**: {}".format(title, ", ".join(links)))
    out.append("")
    meta = dict(PYTHON_META, **PYTHON_INDEX)
    pages.insert(0, ("{}/index.md".format(PYTHON_REFERENCE), meta, "\n".join(out).rstrip() + "\n"))
    return pages


# ------------------------------------------------------------- C++ headers --

def sdk_headers():
    """(installed path, source path) of every header the plugin SDK installs.

    The plugin headers are the CYXWIZ_PLUGIN_SDK_HEADERS list in
    cyxwiz-engine/CMakeLists.txt (installed under <sdk>/include as listed);
    the backend headers are cyxwiz-backend/include/cyxwiz without
    distributed/, as cmake/CyxWizPluginSDK.cmake installs them.
    """
    with open(os.path.join(REPO, "cyxwiz-engine", "CMakeLists.txt"), encoding="utf-8") as f:
        cmake = f.read()
    listed = re.search(r"set\(CYXWIZ_PLUGIN_SDK_HEADERS(.*?)\)", cmake, flags=re.S).group(1).split()
    found = [(rel, os.path.join(REPO, "cyxwiz-engine", "src", *rel.split("/"))) for rel in listed]
    for folder, dirs, files in os.walk(BACKEND_HEADERS):
        dirs[:] = sorted(d for d in dirs if d != "distributed")
        for name in sorted(files):
            if name.endswith(".h"):
                path = os.path.join(folder, name)
                found.append(("cyxwiz/" + os.path.relpath(path, BACKEND_HEADERS).replace(os.sep, "/"), path))
    return found


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def header_summary(text):
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#pragma") or s.startswith("// SPDX") or "Part of the CyxWiz plugin SDK" in s:
            continue
        if s.startswith("//"):
            lines.append(s.lstrip("/").strip())
            continue
        if s.startswith("/*") or s.startswith("*"):
            lines.append(s.strip("/* ").strip())
            continue
        if s == "" and not lines:
            continue
        break
    summary = " ".join(l for l in lines if l)
    return summary


def declared_types(text):
    code = strip_comments(text)
    names = re.findall(r"^\s*(?:class|struct)\s+(?:CYXWIZ_\w+\s+)?(\w+)\s*(?:final\s*)?[:{]", code, flags=re.M)
    names += re.findall(r"^\s*enum\s+class\s+(\w+)", code, flags=re.M)
    seen = []
    for n in names:
        if n not in seen and not n.startswith("_"):
            seen.append(n)
    return seen


def sdk_contents(api_version):
    out = [NOTICE,
           "Every header the plugin SDK installs under `<sdk>/include`, with the summary at the top of the "
           "header and the types it declares. Plugin API {}. The SDK also installs `cyxwiz-backend`'s import "
           "library, the CMake package `CyxWizSDK`, the template plugin, the data plugin example (source and built) and the Python example nodes; see "
           "[Your first C++ plugin](get-started/first-cpp-plugin.md). The SDK ships no ArrayFire, ImGui or spdlog "
           "headers: anything below that needs them (an `af::array` in a signature, a panel's ImGui calls) "
           "builds only inside the CyxWiz source tree.\n".format(api_version)]
    groups = {}
    for display, path in sdk_headers():
        group = "Backend (`cyxwiz`)" if display.startswith("cyxwiz/") else "Plugin interface"
        groups.setdefault(group, []).append((display, path))
    for group in ("Plugin interface", "Backend (`cyxwiz`)"):
        out.append("## {}\n".format(group))
        out.append("| Header | What it is | Types |\n|---|---|---|")
        for display, path in groups.get(group, []):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            summary = header_summary(text).replace("|", "\\|")
            if len(summary) > 300:
                summary = summary[:297].rsplit(" ", 1)[0] + "..."
            types_ = ", ".join("`{}`".format(t) for t in declared_types(text))
            needs_af = " Needs ArrayFire headers." if re.search(r"#include\s*<arrayfire|af::", strip_comments(text)) else ""
            out.append("| `{}` | {}{} | {} |".format(display, summary, needs_af, types_))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def tensor_methods_cpp():
    """{snake_name: [(declaration, sdk_usable)]} of cyxwiz::Tensor's public members and free functions."""
    path = os.path.join(BACKEND_HEADERS, "tensor.h")
    with open(path, encoding="utf-8") as f:
        code = strip_comments(f.read())
    start = code.index("class CYXWIZ_API Tensor") if "class CYXWIZ_API Tensor" in code else code.index("class Tensor")
    body_start = code.index("{", start) + 1
    depth, i = 1, body_start
    while depth:
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        i += 1
    body, rest = code[body_start:i - 1], code[i:]
    public_parts = []
    access = "private"
    for part in re.split(r"^\s*(public|private|protected)\s*:", body, flags=re.M):
        if part in ("public", "private", "protected"):
            access = part
        elif access == "public":
            public_parts.append(part)
    members = {}

    def add(decl):
        decl = re.sub(r"\s+", " ", decl).strip()
        decl = re.sub(r"^template\s*<[^>]*>\s*", "", decl)
        m = re.match(r"^(?:static |explicit |inline |virtual |constexpr |friend )*[\w:<>,*& ]+?\b(operator\s*\S+?|\w+)\s*\(", decl)
        if not m:
            return
        name = m.group(1).replace(" ", "")
        if name in ("Tensor",) or name.startswith("operator="):
            return
        members.setdefault(name, []).append((decl, "af::" not in decl))

    for part in public_parts:
        # Inline bodies become ";" so each declaration is one statement.
        flat, depth = [], 0
        for ch in part:
            if ch == "{":
                if depth == 0:
                    flat.append(";")
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0:
                flat.append(ch)
        for statement in "".join(flat).split(";"):
            if "(" in statement and not statement.strip().startswith("#"):
                add(statement)
    for decl in re.findall(r"^\s*(?:CYXWIZ_API\s+)?Tensor\s+\w+\([^;{]*\)\s*;", rest, flags=re.M):
        add(decl.replace("CYXWIZ_API", ""))
    return members


PYTHON_NAME_FOR = {
    "operator+": "__add__", "operator-": "__sub__", "operator*": "__mul__", "operator/": "__truediv__",
    "operator>": "__gt__", "operator<": "__lt__", "operator>=": "__ge__", "operator<=": "__le__",
    "operator==": "__eq__", "operator!=": "__ne__",
}


ACTIVATION_FUNCTIONS = ("relu", "sigmoid", "tanh", "softmax", "gelu", "elu", "leaky_relu", "mish", "silu",
                        "swish", "dropout")


def op_key(name):
    """One key for a C++ and a Python name of the same operation: BatchMatMul / batch_matmul."""
    if name.startswith("operator"):
        return PYTHON_NAME_FOR.get(name, name)
    return name.replace("_", "").lower()


def tensor_operations(pc, api_version):
    cpp = {}
    for name, decls in tensor_methods_cpp().items():
        cpp.setdefault(op_key(name), []).extend(decls)
    py_methods = {n: getattr(pc.Tensor, n) for n in dir(pc.Tensor) if not n.startswith("_") or n in SHOWN_DUNDERS}
    py_by_key = {}
    for name in py_methods:
        if name == "__repr__":
            continue
        py_by_key.setdefault(op_key(name) if not name.startswith("_") else name, []).append(name)
    functions = {op_key(n): n for n in ACTIVATION_FUNCTIONS if callable(getattr(pc, n, None))}

    def py_cell(key):
        cells = []
        for name in py_by_key.get(key, []):
            label = SHOWN_DUNDERS.get(name)
            if label:
                cells.append("`{}`".format(label))
                continue
            for signature, _ in entries_of(py_methods[name], name):
                cells.append("`" + signature.replace("(self: Tensor, ", "(").replace("(self: Tensor)", "()") + "`")
        if key in functions:
            for signature, _ in entries_of(getattr(pc, functions[key]), functions[key]):
                cells.append("`pycyxwiz." + signature + "`")
        return "<br>".join(cells)

    def label(key):
        names = py_by_key.get(key, [])
        if key in functions:
            return functions[key]
        if names:
            return SHOWN_DUNDERS.get(names[0], names[0])
        for decl, _ in cpp.get(key, []):
            m = re.search(r"\b(operator\s*\S+?|\w+)\s*\(", decl)
            return m.group(1) if m else key
        return key

    keys = sorted(set(cpp) | set(py_by_key) | set(functions), key=lambda k: label(k).lstrip("_").lower())
    out = [NOTICE,
           "What a node's `forward` and `backward` can do with a tensor, from the C++ header "
           "`cyxwiz/tensor.h` (plugin API {0}) and the built pycyxwiz module ({1}). A blank cell means "
           "the operation does not exist on that side. Tensors are `float32` unless created otherwise; "
           "dimension 0 is the batch.\n".format(api_version, pc.get_version()),
           "\"Source tree only\" marks C++ members whose signature uses ArrayFire types: the plugin SDK "
           "ships no ArrayFire headers, so a plugin built against the SDK cannot call them. Activation "
           "functions such as `tanh`, `relu` and `sigmoid` are module functions on the Python side "
           "(`pycyxwiz.tanh(x)`); in C++ they are layers and activations in `cyxwiz/activation.h` and "
           "`cyxwiz/activations/` (see [What the plugin SDK contains](sdk-contents.md)).\n",
           "| Operation | C++ (`cyxwiz::Tensor`) | Python |\n|---|---|---|"]
    for key in keys:
        cpp_cells = []
        for decl, usable in cpp.get(key, []):
            cpp_cells.append("`{}`{}".format(decl.replace("|", "\\|"), "" if usable else " (source tree only)"))
        py = py_cell(key).replace("|", "\\|")
        if not cpp_cells and not py:
            continue
        out.append("| {} | {} | {} |".format(label(key).replace("|", "\\|"), "<br>".join(cpp_cells), py))
    return "\n".join(out).rstrip() + "\n"


def plugin_api_version():
    with open(os.path.join(REPO, "cyxwiz-engine", "src", "plugin", "plugin_types.h"), encoding="utf-8") as f:
        text = f.read()
    parts = [re.search(r"PLUGIN_API_VERSION_{}\s*=\s*(\d+)".format(p), text).group(1)
             for p in ("MAJOR", "MINOR", "PATCH")]
    return ".".join(parts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    if not args:
        print(__doc__)
        return 2
    folder = os.path.abspath(args[0])
    pc = import_pycyxwiz(folder, os.path.abspath(args[1]) if len(args) > 1 else folder)
    api = plugin_api_version()
    pages = python_reference(pc)
    pages.append(PAGES["tensor"] + (tensor_operations(pc, api),))
    pages.append(PAGES["sdk"] + (sdk_contents(api),))
    stale = []
    for name, meta, body in pages:
        path = os.path.join(DOCS, *name.split("/"))
        text = front_matter(meta, committed_last_reviewed(path)) + drop_tickets(body)
        current = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                current = f.read()
        if current == text:
            continue
        if check:
            stale.append("docs/" + name)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print("wrote docs/" + name)
    # A generated reference page the module no longer gives (a group emptied,
    # a submodule removed) goes, so no page describes names that are gone.
    written = {os.path.normcase(os.path.join(DOCS, *name.split("/"))) for name, _, _ in pages}
    for path in generated_python_pages():
        if os.path.normcase(path) in written:
            continue
        rel = os.path.relpath(path, REPO).replace(os.sep, "/")
        if check:
            stale.append(rel + " (no longer generated)")
        else:
            os.remove(path)
            print("removed " + rel)
    if stale:
        print("out of date (run python {} <pycyxwiz folder>): {}".format(GENERATOR, ", ".join(stale)))
        return 1
    if check:
        print("docs/python/reference and docs/sdk reference pages match the code")
    return 0


def generated_python_pages():
    """The pages in docs/python/reference that say generated: true."""
    folder = os.path.join(DOCS, *PYTHON_REFERENCE.split("/"))
    found = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if name.endswith(".md"):
                with open(path, encoding="utf-8") as f:
                    if re.search(r"^generated:\s*true\s*$", f.read(2000), flags=re.M):
                        found.append(path)
    return found


def generated_pages():
    """Every generated page under docs/ (paths relative to docs/), for the checks."""
    names = [name for name, _ in PAGES.values()]
    names += [os.path.relpath(p, DOCS).replace(os.sep, "/") for p in generated_python_pages()]
    return sorted(names)


if __name__ == "__main__":
    sys.exit(main())
