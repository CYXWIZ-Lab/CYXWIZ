#!/usr/bin/env python3
# TOFIX178 1.9f
"""Writes the ResNet-18 graphs: the fine-tuning example and two test graphs.

Every layer node is named as torchvision names its layer (conv1, bn1, layer1.0.conv1, ...,
layer2.0.downsample.0, fc), so a torchvision weight file fills the graph by name. The convolutions have
their Bias off and the first pool is size 3, stride 2, padding 1, as torchvision's ResNet.

  examples/cyxgraph/cats_dogs_resnet18_finetune.cyxgraph
      cats vs dogs from ImageNet weights: the backbone frozen, a one-logit head, weights/resnet18-f37072fd.pth
  cyxwiz-engine/tests/computation_truth/fixtures/resnet18_thin.cyxgraph
      the same network an eighth as wide with 10 classes, for the test that compares it with PyTorch
  cyxwiz-engine/tests/computation_truth/fixtures/resnet18_imagenet.cyxgraph
      the full network with 1000 classes, for the optional check against the published weights

The data and training nodes of the example come from cats_dogs_resnet_scratch.cyxgraph, so the two examples
read the same folder the same way.

Run from the repository root: python scripts/examples/make_resnet18_graphs.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "examples" / "cyxgraph" / "cats_dogs_resnet_scratch.cyxgraph"
FIXTURES = ROOT / "cyxwiz-engine" / "tests" / "computation_truth" / "fixtures"

# Node type numbers (gui::NodeType) and categories, as the editor saves them.
DENSE, CONV, MAXPOOL, GAP, BATCHNORM, RELU, ADD = 0, 2, 5, 8, 10, 29, 47
CROSS_ENTROPY, OUTPUT, ADAM = 73, 71, 81
CATEGORY = {DENSE: 6, CONV: 6, BATCHNORM: 9, RELU: 7}
STEP_X, ROW_Y = 220.0, 420.0


class Graph:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.next_id = 1

    def node(self, node_type: int, name: str, x: float, y: float, parameters: dict[str, str] | None = None,
             template: dict[str, Any] | None = None) -> int:
        node = copy.deepcopy(template) if template else {"description": "", "parameters": {}}
        node.update({"id": self.next_id, "type": node_type, "name": name, "pos_x": x, "pos_y": y})
        node.setdefault("parameters", {}).update(parameters or {})
        if node_type in CATEGORY and "category" not in node:
            node["category"] = CATEGORY[node_type]
        self.nodes.append(node)
        self.next_id += 1
        return node["id"]

    def link(self, source: int, target: int, to_pin: int = 0, from_pin: int = 0) -> None:
        self.links.append({"id": len(self.links) + 1, "from_node": source, "from_pin": 0, "from_pin_index": from_pin,
                           "to_node": target, "to_pin": 0, "to_pin_index": to_pin, "link_type": 0})

    def group(self, name: str, description: str, node_ids: list[int]) -> None:
        self.groups.append({"id": len(self.groups) + 1, "name": name, "description": description,
                            "node_ids": node_ids, "collapsed": False, "padding": 20.0})


def backbone(graph: Graph, source: int, base: int, y0: float, frozen: bool) -> int:
    """conv1 ... layer4.1 with torchvision's names; returns the last node."""
    fine_tuning = {"frozen": "true"} if frozen else {}

    def conv(name: str, filters: int, kernel: int, stride: int, x: float, y: float) -> int:
        return graph.node(CONV, name, x, y, {"filters": str(filters), "kernel_size": str(kernel), "stride": str(stride),
                                             "padding": "same", "use_bias": "false", **fine_tuning})

    def norm(name: str, x: float, y: float) -> int:
        return graph.node(BATCHNORM, name, x, y, {"eps": "0.00001", "momentum": "0.1", **fine_tuning})

    ids = [conv("conv1", base, 7, 2, 160.0, y0)]
    ids.append(norm("bn1", 160.0 + STEP_X, y0))
    ids.append(graph.node(RELU, "relu", 160.0 + 2 * STEP_X, y0))
    ids.append(graph.node(MAXPOOL, "maxpool", 160.0 + 3 * STEP_X, y0, {"pool_size": "3", "stride": "2", "padding": "1"}))
    graph.link(source, ids[0])
    for a, b in zip(ids, ids[1:]):
        graph.link(a, b)
    graph.group(f"Stem ({base} channels)", "conv1 7x7 stride 2 without bias - bn1 - ReLU - max pool 3, stride 2, padding 1",
                list(ids))
    last = ids[-1]
    row = 1
    for stage in range(1, 5):
        planes = base * 2 ** (stage - 1)
        for block in range(2):
            y = y0 + row * ROW_Y
            row += 1
            stride = 2 if stage > 1 and block == 0 else 1
            prefix = f"layer{stage}.{block}"
            c1 = conv(f"{prefix}.conv1", planes, 3, stride, 160.0, y)
            b1 = norm(f"{prefix}.bn1", 160.0 + STEP_X, y)
            r1 = graph.node(RELU, f"{prefix}.relu", 160.0 + 2 * STEP_X, y)
            c2 = conv(f"{prefix}.conv2", planes, 3, 1, 160.0 + 3 * STEP_X, y)
            b2 = norm(f"{prefix}.bn2", 160.0 + 4 * STEP_X, y)
            members = [c1, b1, r1, c2, b2]
            for a, b in zip([last, c1, b1, r1, c2], members):
                graph.link(a, b)
            skip = last
            if stride != 1:
                d0 = conv(f"{prefix}.downsample.0", planes, 1, stride, 160.0 + 3 * STEP_X, y + 170.0)
                d1 = norm(f"{prefix}.downsample.1", 160.0 + 4 * STEP_X, y + 170.0)
                graph.link(last, d0)
                graph.link(d0, d1)
                members += [d0, d1]
                skip = d1
            add = graph.node(ADD, f"{prefix}.add", 160.0 + 5 * STEP_X, y)
            out = graph.node(RELU, f"{prefix}.out", 160.0 + 6 * STEP_X, y)
            graph.link(b2, add, to_pin=0)
            graph.link(skip, add, to_pin=1)
            graph.link(add, out)
            members += [add, out]
            graph.group(f"{prefix} ({planes} channels)",
                        "Two 3x3 Conv - BatchNorm; the block's input is added back, then ReLU"
                        + ("; the skip is a 1x1 stride-2 Conv - BatchNorm" if stride != 1 else ""), members)
            last = out
    return last


def document(graph: Graph, description: str) -> dict[str, Any]:
    scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))
    out = {k: v for k, v in scratch.items() if k not in ("nodes", "links", "groups", "annotations")}
    out["workflow_description"] = description
    out.update({"nodes": graph.nodes, "links": graph.links, "annotations": [], "groups": graph.groups})
    return out


def write(path: Path, graph: Graph, description: str) -> None:
    path.write_text(json.dumps(document(graph, description), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(graph.nodes)} nodes, {len(graph.links)} links")


def example() -> None:
    scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))
    by_name = {n["name"]: n for n in scratch["nodes"]}
    graph = Graph()
    data_names = ["Cats_Dogs Data", "Quality Analyzer", "Split 80/10/10", "Data Loader", "Resize 72", "Center Crop 64",
                  "Random Resized Crop", "Horizontal Flip", "Color Jitter", "Normalize (ImageNet)"]
    ids = []
    for i, name in enumerate(data_names):
        template = by_name[name]
        node_id = graph.node(template["type"], name, i * STEP_X, 300.0, template=template)
        ids.append(node_id)
    node = {n["name"]: n for n in graph.nodes}
    node["Data Loader"]["parameters"].update({"batch_size": "64", "epochs": "3",
                                             "initialize_from_weights_file": "weights/resnet18-f37072fd.pth"})
    for a, b in zip(ids, ids[1:]):
        graph.link(a, b)
    graph.group("Data and augmentation",
                "Data Input - Quality Analyzer - Split 80/10/10 - Loader (starts from weights/resnet18-f37072fd.pth); "
                "Resize 72 and Center Crop 64 once (as the from-scratch example); per batch: Random Resized Crop, Flip, Color Jitter; ImageNet "
                "Normalize, which the published weights expect", list(ids))
    last = backbone(graph, ids[-1], 64, 720.0, frozen=True)
    y = 720.0 + 9 * ROW_Y
    pool = graph.node(GAP, "avgpool", 160.0, y, template=by_name["Global Avg Pool"])
    head = graph.node(DENSE, "fc", 160.0 + STEP_X, y, {"units": "1"}, template=by_name["Dense 1"])
    loss = graph.node(by_name["BCE with Logits"]["type"], "BCE with Logits", 160.0 + 2 * STEP_X, y,
                      {"label_smoothing": "0.0"}, template=by_name["BCE with Logits"])
    optimizer = graph.node(by_name["AdamW"]["type"], "AdamW", 160.0 + 3 * STEP_X, y,
                           {"learning_rate": "0.001", "weight_decay": "0.0"}, template=by_name["AdamW"])
    output = graph.node(OUTPUT, "Output", 160.0 + 4 * STEP_X, y, template=by_name["Output"])
    graph.link(last, pool)
    graph.link(pool, head)
    graph.link(head, loss, to_pin=0)
    graph.link(node["Data Loader"]["id"], loss, to_pin=1, from_pin=1)
    graph.link(loss, optimizer)
    graph.link(optimizer, output)
    graph.group("Head and training",
                "Global average pool - fc with one logit (starts fresh: the file's fc has 1000 classes); BCE with "
                "logits; AdamW. Only fc trains while the backbone is frozen", [pool, head, loss, optimizer, output])
    write(ROOT / "examples" / "cyxgraph" / "cats_dogs_resnet18_finetune.cyxgraph", graph,
          "Cats vs dogs by fine-tuning ResNet-18 from torchvision's ImageNet weights: the backbone frozen, a new head")


def fixture(file_name: str, base: int, classes: int, description: str) -> None:
    scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))
    by_name = {n["name"]: n for n in scratch["nodes"]}
    graph = Graph()
    data = graph.node(by_name["Cats_Dogs Data"]["type"], "Images", 0.0, 300.0, template=by_name["Cats_Dogs Data"])
    graph.nodes[-1]["parameters"].update({"dataset_name": "resnet18_fixture", "folder_path": "D:/tmp/resnet18_fixture",
                                          "image_layout": "0"})
    split = graph.node(by_name["Split 80/10/10"]["type"], "Split 80/10/10", STEP_X, 300.0, template=by_name["Split 80/10/10"])
    loader = graph.node(by_name["Data Loader"]["type"], "Data Loader", 2 * STEP_X, 300.0, template=by_name["Data Loader"])
    resize = graph.node(by_name["Resize 72"]["type"], "Resize 64", 3 * STEP_X, 300.0, {"width": "64", "height": "64"},
                        template=by_name["Resize 72"])
    graph.link(data, split)
    graph.link(split, loader)
    graph.link(loader, resize)
    last = backbone(graph, resize, base, 720.0, frozen=False)
    y = 720.0 + 9 * ROW_Y
    pool = graph.node(GAP, "avgpool", 160.0, y, template=by_name["Global Avg Pool"])
    head = graph.node(DENSE, "fc", 160.0 + STEP_X, y, {"units": str(classes)}, template=by_name["Dense 1"])
    loss = graph.node(CROSS_ENTROPY, "Cross entropy", 160.0 + 2 * STEP_X, y)
    optimizer = graph.node(ADAM, "Adam", 160.0 + 3 * STEP_X, y)
    output = graph.node(OUTPUT, "Output", 160.0 + 4 * STEP_X, y, {"num_classes": str(classes)}, template=by_name["Output"])
    graph.link(last, pool)
    graph.link(pool, head)
    graph.link(head, loss, to_pin=0)
    graph.link(loader, loss, to_pin=1, from_pin=1)
    graph.link(loss, optimizer)
    graph.link(optimizer, output)
    write(FIXTURES / file_name, graph, description)


def main() -> None:
    example()
    fixture("resnet18_thin.cyxgraph", 8, 10, "ResNet-18 an eighth as wide, 10 classes: compared with PyTorch in tests")
    fixture("resnet18_imagenet.cyxgraph", 64, 1000, "ResNet-18 as torchvision's, 1000 classes: checked against the "
            "published ImageNet weights when the file is at hand")


if __name__ == "__main__":
    main()
