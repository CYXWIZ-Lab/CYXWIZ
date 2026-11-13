#!/usr/bin/env python3
# TOFIX178 1.9g
"""Turns a PyTorch network into Engine canvas nodes, named as PyTorch names its modules.

The network is traced with torch.fx, and each module becomes the node the Engine has for it:
Conv2d (Conv2D, or Depthwise Conv2D when each channel has its own kernels), BatchNorm2d, ReLU, ReLU6,
Hardswish, Hardsigmoid, SiLU (Swish), MaxPool2d, AdaptiveAvgPool2d(1) (Global Avg Pool, which also
flattens), Linear (Dense), Dropout and torchvision.ops.SqueezeExcitation. An addition of two branches
becomes an Add node. Every node that holds weights is named after its module (conv1, layer1.0.bn1,
features.3.block.1), so the network's state-dict fills the graph by name. Anything else is refused with
the module's name, so a graph never differs silently from the network.

As a script it writes the examples:
  examples/cyxgraph/cats_dogs_resnet18_finetune.cyxgraph   torchvision's ResNet-18, frozen, one-logit head
The test graphs are written by tests/computation_truth/reference/generate_pretrained_network_fixtures.py,
which uses convert() and network_graph() from here.

Run from the repository root: python scripts/examples/torch_to_cyxgraph.py
"""
from __future__ import annotations

import copy
import json
import operator
import re
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
from torch import nn
from torchvision.ops.misc import SqueezeExcitation

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "examples" / "cyxgraph" / "cats_dogs_resnet_scratch.cyxgraph"
STEP_X, ROW_Y = 220.0, 420.0


def node_types() -> dict[str, int]:
    """gui::NodeType's values, read from graph_model.h (the number a saved graph stores)."""
    text = (ROOT / "cyxwiz-engine" / "src" / "core" / "graph_model.h").read_text(encoding="utf-8")
    start = text.index("enum class NodeType")
    body = re.sub(r"//.*", "", text[start:text.index("};", start)])
    body = body[body.index("{") + 1:]
    values, value = {}, 0
    for name in (part.strip() for part in body.split(",")):
        if not name:
            continue
        if "=" in name:
            name, number = (piece.strip() for piece in name.split("="))
            value = int(number, 0)
        values[name] = value
        value += 1
    return values


TYPES = node_types()


class Refused(Exception):
    pass


class Tracer(fx.Tracer):
    """Keeps Squeeze-Excitation whole: the Engine has one node for it."""

    def is_leaf_module(self, module: nn.Module, qualified_name: str) -> bool:
        return isinstance(module, SqueezeExcitation) or super().is_leaf_module(module, qualified_name)


def number(value: float) -> str:
    return repr(float(value))


def padding_of(kernel: int, pad: int) -> str:
    return "same" if kernel % 2 == 1 and pad == (kernel - 1) // 2 else str(pad)


def square(value: Any, what: str, name: str) -> int:
    if isinstance(value, tuple):
        if len(set(value)) != 1:
            raise Refused(f"{name}: a {what} of {value} is not square; the Engine's nodes take one size")
        value = value[0]
    return int(value)


ACTIVATIONS = {nn.ReLU: "ReLU", nn.ReLU6: "ReLU6", nn.Hardswish: "Hardswish", nn.Hardsigmoid: "Hardsigmoid",
               nn.SiLU: "Swish", nn.Sigmoid: "Sigmoid", nn.Tanh: "Tanh"}
SE_INNER = {nn.ReLU: "relu", nn.Hardswish: "hardswish", nn.SiLU: "swish"}
SE_GATE = {nn.Sigmoid: "sigmoid", nn.Hardsigmoid: "hardsigmoid"}


def module_node(name: str, module: nn.Module) -> tuple[str, dict[str, str]] | None:
    """The node type and settings for one module; None = the module passes its input on unchanged."""
    if isinstance(module, nn.Conv2d):
        if square(module.dilation, "dilation", name) != 1:
            raise Refused(f"{name}: dilated convolutions have no Engine node")
        kernel = square(module.kernel_size, "kernel", name)
        settings = {"kernel_size": str(kernel), "stride": str(square(module.stride, "stride", name)),
                    "padding": padding_of(kernel, square(module.padding, "padding", name)),
                    "use_bias": "true" if module.bias is not None else "false"}
        if module.groups == 1:
            return "Conv2D", {"filters": str(module.out_channels), **settings}
        if module.groups == module.in_channels and module.out_channels % module.in_channels == 0:
            return "DepthwiseConv2D", {**settings, "depth_multiplier": str(module.out_channels // module.in_channels)}
        raise Refused(f"{name}: a grouped convolution ({module.groups} groups) has no Engine node")
    if isinstance(module, nn.BatchNorm2d):
        return "BatchNorm", {"eps": number(module.eps), "momentum": number(module.momentum)}
    for kind, node in ACTIVATIONS.items():
        if type(module) is kind:
            return node, {}
    if isinstance(module, nn.MaxPool2d):
        if square(module.dilation, "dilation", name) != 1 or module.ceil_mode:
            raise Refused(f"{name}: the Engine's MaxPool2D has no dilation or ceil mode")
        return "MaxPool2D", {"pool_size": str(square(module.kernel_size, "pool size", name)),
                             "stride": str(square(module.stride, "stride", name)),
                             "padding": str(square(module.padding, "padding", name))}
    if isinstance(module, nn.AdaptiveAvgPool2d):
        if square(module.output_size, "output size", name) != 1:
            raise Refused(f"{name}: only a pool to 1 x 1 has an Engine node (Global Avg Pool)")
        return "GlobalAvgPool", {}
    if isinstance(module, nn.Linear):
        return "Dense", {"units": str(module.out_features)}
    if isinstance(module, nn.Dropout):
        return "Dropout", {"rate": number(module.p)}
    if isinstance(module, SqueezeExcitation):
        inner = SE_INNER.get(type(module.activation))
        gate = SE_GATE.get(type(module.scale_activation))
        if inner is None or gate is None:
            raise Refused(f"{name}: Squeeze-Excitation with {type(module.activation).__name__} / "
                          f"{type(module.scale_activation).__name__} has no Engine choice")
        return "SqueezeExcitation", {"width_mode": "exact", "squeeze_channels": str(module.fc1.out_channels),
                                     "inner_activation": inner, "gate": gate,
                                     "use_bias": "true" if module.fc1.bias is not None else "false"}
    if isinstance(module, (nn.Identity, nn.Flatten)):
        return None
    raise Refused(f"{name}: {type(module).__name__} has no Engine node")


HOLDS_WEIGHTS = {"Conv2D", "DepthwiseConv2D", "BatchNorm", "Dense", "SqueezeExcitation"}


def convert(model: nn.Module) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    """The network as nodes (index, type, name, settings, group) and links (from, to, to_pin).
    Node 0 stands for the network's input."""
    graph = Tracer().trace(model)
    modules = dict(model.named_modules())
    nodes: list[dict[str, Any]] = []
    links: list[tuple[int, int, int]] = []
    produced: dict[fx.Node, int] = {}
    used_names: dict[str, int] = {}

    def add(kind: str, name: str, settings: dict[str, str], sources: list[int]) -> int:
        count = used_names.get(name, 0)
        used_names[name] = count + 1
        unique = name if count == 0 else f"{name}_{count}"  # a module called twice (BasicBlock's relu)
        group = ".".join(name.split(".")[:2]) if "." in name else name
        nodes.append({"index": len(nodes) + 1, "type": kind, "name": unique, "settings": settings, "group": group})
        for pin, source in enumerate(sources):
            links.append((source, len(nodes), pin))
        return len(nodes)

    for node in graph.nodes:
        if node.op == "placeholder":
            produced[node] = 0
        elif node.op == "call_module":
            source = produced[node.args[0]]
            made = module_node(node.target, modules[node.target])
            produced[node] = source if made is None else add(made[0], node.target, made[1], [source])
        elif node.op == "call_function" and node.target in (operator.add, operator.iadd, torch.add):
            a, b = (produced[arg] for arg in node.args[:2])
            owner = node.meta.get("nn_module_stack")
            scope = list(owner.values())[-1][0] if owner else ""
            produced[node] = add("Add", f"{scope}.add" if scope else "add", {}, [a, b])
        elif node.op == "call_function" and node.target is torch.flatten:
            produced[node] = produced[node.args[0]]  # Global Avg Pool already gives one value per channel
        elif node.op == "call_function" and node.target is nn.functional.adaptive_avg_pool2d:
            if square(node.args[1], "output size", "adaptive_avg_pool2d") != 1:
                raise Refused("adaptive_avg_pool2d: only a pool to 1 x 1 has an Engine node")
            produced[node] = add("GlobalAvgPool", "avgpool", {}, [produced[node.args[0]]])
        elif node.op == "output":
            produced[node] = produced[node.args[0]]
            last = produced[node]
        else:
            raise Refused(f"{node.op} {node.target} has no Engine node")
    if not nodes or last != nodes[-1]["index"]:
        raise Refused("the network's output is not its last layer")
    return nodes, links


class Graph:
    """A saved Engine graph being written."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []

    def node(self, kind: str | int, name: str, x: float, y: float, settings: dict[str, str] | None = None,
             template: dict[str, Any] | None = None) -> int:
        node = copy.deepcopy(template) if template else {"description": "", "parameters": {}}
        node.update({"id": len(self.nodes) + 1, "type": TYPES[kind] if isinstance(kind, str) else kind,
                     "name": name, "pos_x": x, "pos_y": y})
        node.setdefault("parameters", {}).update(settings or {})
        self.nodes.append(node)
        return node["id"]

    def link(self, source: int, target: int, to_pin: int = 0, from_pin: int = 0) -> None:
        self.links.append({"id": len(self.links) + 1, "from_node": source, "from_pin": 0, "from_pin_index": from_pin,
                           "to_node": target, "to_pin": 0, "to_pin_index": to_pin, "link_type": 0})

    def group(self, name: str, description: str, node_ids: list[int]) -> None:
        self.groups.append({"id": len(self.groups) + 1, "name": name, "description": description,
                            "node_ids": node_ids, "collapsed": False, "padding": 20.0})

    def document(self, description: str) -> dict[str, Any]:
        scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))
        out = {k: v for k, v in scratch.items() if k not in ("nodes", "links", "groups", "annotations")}
        out["workflow_description"] = description
        out.update({"nodes": self.nodes, "links": self.links, "annotations": [], "groups": self.groups})
        return out

    def write(self, path: Path, description: str) -> None:
        path.write_text(json.dumps(self.document(description), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}: {len(self.nodes)} nodes, {len(self.links)} links")


def place_network(graph: Graph, model: nn.Module, source: int, y0: float, frozen: bool,
                  head_units: int | None = None) -> tuple[int, int]:
    """Adds the network after `source`; returns (the last node, the head's node). The head is the last Dense,
    which is never frozen; `head_units` replaces its output count."""
    nodes, links = convert(model)
    head = max(n["index"] for n in nodes if n["type"] == "Dense")
    ids = {0: source}
    row, x, current, members = -1, 0, None, []
    for n in nodes:
        if n["group"] != current:
            if members:
                graph.group(current, "", members)
            current, members, row, x = n["group"], [], row + 1, 0
        settings = dict(n["settings"])
        if n["index"] == head and head_units is not None:
            settings["units"] = str(head_units)
        if frozen and n["type"] in HOLDS_WEIGHTS and n["index"] != head:
            settings["frozen"] = "true"
        ids[n["index"]] = graph.node(n["type"], n["name"], 160.0 + x * STEP_X, y0 + row * ROW_Y, settings)
        members.append(ids[n["index"]])
        x += 1
    if members:
        graph.group(current, "", members)
    for source_index, target_index, pin in links:
        graph.link(ids[source_index], ids[target_index], to_pin=pin)
    return ids[nodes[-1]["index"]], ids[head]


def scratch_nodes() -> dict[str, dict[str, Any]]:
    return {n["name"]: n for n in json.loads(SCRATCH.read_text(encoding="utf-8"))["nodes"]}


def network_graph(model: nn.Module, classes: int, description: str) -> Graph:
    """A test graph: Images -> Split -> Loader -> Resize 64 -> the network -> Cross entropy -> Adam -> Output."""
    template = scratch_nodes()
    graph = Graph()
    data = graph.node(template["Cats_Dogs Data"]["type"], "Images", 0.0, 300.0, template=template["Cats_Dogs Data"])
    graph.nodes[-1]["parameters"].update({"dataset_name": "network_fixture", "folder_path": "D:/tmp/network_fixture",
                                          "image_layout": "0"})
    split = graph.node(template["Split 80/10/10"]["type"], "Split 80/10/10", STEP_X, 300.0, template=template["Split 80/10/10"])
    loader = graph.node(template["Data Loader"]["type"], "Data Loader", 2 * STEP_X, 300.0, template=template["Data Loader"])
    resize = graph.node(template["Resize 72"]["type"], "Resize 64", 3 * STEP_X, 300.0, {"width": "64", "height": "64"},
                        template=template["Resize 72"])
    graph.link(data, split)
    graph.link(split, loader)
    graph.link(loader, resize)
    last, _ = place_network(graph, model, resize, 720.0, frozen=False)
    y = max(n["pos_y"] for n in graph.nodes) + ROW_Y
    loss = graph.node("CrossEntropyLoss", "Cross entropy", 160.0, y)
    optimizer = graph.node("Adam", "Adam", 160.0 + STEP_X, y)
    output = graph.node("Output", "Output", 160.0 + 2 * STEP_X, y, {"num_classes": str(classes)}, template=template["Output"])
    graph.link(last, loss, to_pin=0)
    graph.link(loader, loss, to_pin=1, from_pin=1)
    graph.link(loss, optimizer)
    graph.link(optimizer, output)
    return graph


def finetune_example(model: nn.Module, weights_file: str, description: str) -> Graph:
    """cats vs dogs: the data and training nodes of the from-scratch example, the network frozen, a one-logit head."""
    template = scratch_nodes()
    graph = Graph()
    data_names = ["Cats_Dogs Data", "Quality Analyzer", "Split 80/10/10", "Data Loader", "Resize 72", "Center Crop 64",
                  "Random Resized Crop", "Horizontal Flip", "Color Jitter", "Normalize (ImageNet)"]
    ids = [graph.node(template[name]["type"], name, i * STEP_X, 300.0, template=template[name])
           for i, name in enumerate(data_names)]
    loader = ids[data_names.index("Data Loader")]
    graph.nodes[loader - 1]["parameters"].update({"batch_size": "64", "epochs": "3",
                                                  "initialize_from_weights_file": weights_file})
    for a, b in zip(ids, ids[1:]):
        graph.link(a, b)
    graph.group("Data and augmentation",
                f"Data Input - Quality Analyzer - Split 80/10/10 - Loader (starts from {weights_file}); Resize 72 and "
                "Center Crop 64 once (as the from-scratch example); per batch: Random Resized Crop, Flip, Color "
                "Jitter; ImageNet Normalize, which the published weights expect", list(ids))
    last, head = place_network(graph, model, ids[-1], 720.0, frozen=True, head_units=1)
    y = max(n["pos_y"] for n in graph.nodes) + ROW_Y
    loss = graph.node(template["BCE with Logits"]["type"], "BCE with Logits", 160.0, y, {"label_smoothing": "0.0"},
                      template=template["BCE with Logits"])
    optimizer = graph.node(template["AdamW"]["type"], "AdamW", 160.0 + STEP_X, y,
                           {"learning_rate": "0.001", "weight_decay": "0.0"}, template=template["AdamW"])
    output = graph.node(template["Output"]["type"], "Output", 160.0 + 2 * STEP_X, y, template=template["Output"])
    graph.link(last, loss, to_pin=0)
    graph.link(loader, loss, to_pin=1, from_pin=1)
    graph.link(loss, optimizer)
    graph.link(optimizer, output)
    graph.group("Training", "One logit from the new head; BCE with logits; AdamW. Only the head trains while the "
                "network is frozen", [loss, optimizer, output])
    return graph


def main() -> None:
    import torchvision
    finetune_example(torchvision.models.resnet18(), "weights/resnet18-f37072fd.pth", "").write(
        ROOT / "examples" / "cyxgraph" / "cats_dogs_resnet18_finetune.cyxgraph",
        "Cats vs dogs by fine-tuning ResNet-18 from torchvision's ImageNet weights: the backbone frozen, a new head")


if __name__ == "__main__":
    main()
