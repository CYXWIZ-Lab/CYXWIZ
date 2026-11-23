"""A group job for one machine: shuffled tabular parts, a graph, a job file and one ticket per worker.

    python make_job.py <job folder> [workers]

Then run it with run_local.ps1. Needs numpy and pyarrow.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

root = Path(sys.argv[1])
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(166)

# Three classes over four features, shuffled before the split into parts.
ROWS, PARTS = 200_000, 4
labels = rng.integers(0, 3, ROWS)
features = rng.normal(0.0, 0.6, (ROWS, 4)).astype(np.float32)
features[np.arange(ROWS), labels] += 2.0
chunks = []
for part in range(PARTS):
    rows = slice(part * ROWS // PARTS, (part + 1) * ROWS // PARTS)
    table = pa.table({**{f"x{i}": features[rows, i] for i in range(4)}, "class": labels[rows].astype(np.int64)})
    path = root / f"part_{part:03d}.parquet"
    pq.write_table(table, path, row_group_size=10_000)
    chunks.append({"path": str(path).replace("\\", "/"), "rows": table.num_rows})

# The graph format numbers node types as the Engine's NodeType enum does.
def node_types(*names: str) -> dict:
    header = Path(__file__).resolve().parents[3] / "cyxwiz-engine" / "src" / "core" / "graph_model.h"
    text = header.read_text(encoding="utf-8")
    start = re.search(r"enum class NodeType\s*\{", text).end()
    body = re.sub(r"//.*", "", text[start:text.index("};", start)])
    value, numbers = -1, {}
    for item in (part.strip() for part in body.split(",")):
        if not item:
            continue
        if "=" in item:
            item, number = (piece.strip() for piece in item.split("="))
            value = int(number, 0)
        else:
            value += 1
        numbers[item] = value
    return {name: numbers[name] for name in names}


types = node_types("DataInput", "DataSplit", "DataLoader", "Dense", "ReLU", "CrossEntropyLoss", "Adam", "Output")
nodes = [
    {"id": 1, "type": types["DataInput"], "name": "Blobs",
     "parameters": {"dataset_name": "group_blobs", "file_category": "tabular", "file_type": "parquet", "label_column": "class"}},
    {"id": 2, "type": types["DataSplit"], "name": "Split",
     "parameters": {"seed": "42", "stratified": "false", "train_ratio": "0.8", "val_ratio": "0.1", "test_ratio": "0.1"}},
    {"id": 3, "type": types["DataLoader"], "name": "Loader",
     "parameters": {"batch_size": "32", "epochs": "2", "shuffle": "true", "seed": "11", "model_seed": "7"}},
    {"id": 4, "type": types["Dense"], "name": "Dense 64", "parameters": {"units": "64"}},
    {"id": 5, "type": types["ReLU"], "name": "ReLU", "parameters": {}},
    {"id": 6, "type": types["Dense"], "name": "Classes", "parameters": {"units": "3"}},
    {"id": 7, "type": types["CrossEntropyLoss"], "name": "Cross entropy", "parameters": {}},
    {"id": 8, "type": types["Adam"], "name": "Adam", "parameters": {"learning_rate": "0.001"}},
    {"id": 9, "type": types["Output"], "name": "Output", "parameters": {"num_classes": "3"}},
]
links = []
for index, (a, ai, b, bi) in enumerate([(1, 0, 2, 0), (2, 0, 3, 0), (3, 0, 4, 0), (4, 0, 5, 0), (5, 0, 6, 0), (6, 0, 7, 0),
                                        (3, 1, 7, 1), (7, 0, 8, 0), (8, 0, 9, 0)], start=1):
    links.append({"id": index, "from_node": a, "from_pin": 100 + index, "from_pin_index": ai,
                  "to_node": b, "to_pin": 200 + index, "to_pin_index": bi, "link_type": 0})
(root / "model.cyxgraph").write_text(json.dumps({"version": "1.0", "data_boundary_version": 2, "nodes": nodes, "links": links}))

tickets = []
for index in range(workers):
    ticket = rng.bytes(24).hex()
    (root / f"node{index}.ticket").write_text(ticket)
    tickets.append({"name": f"node{index}", "ticket": ticket})
job = {
    "group_id": "one-pc-run",
    "graph_file": "model.cyxgraph",
    "batch_size": 32,
    "chunks": chunks,
    "sample_chunk": 0,
    "shuffled_when_split": True,
    "workers": tickets,
    "stop": {"target_epochs": 1, "max_rounds": 0, "max_seconds": 0},
    "round": {"train_seconds": 2.0, "report_grace_seconds": 20, "open_seconds": 120, "warm_up_steps": 20},
    "update_encoding": "int8",
    "contract": {"min_data_rounds": 5},
    "state_dir": str(root / "state").replace("\\", "/"),
    "listen": "127.0.0.1:50061",
    "seed": 1,
}
(root / "job.json").write_text(json.dumps(job, indent=1))
print("ok", root)
