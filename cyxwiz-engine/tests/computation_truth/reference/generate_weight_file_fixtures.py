#!/usr/bin/env python3
# TOFIX178 1.9c
"""Weight files written by PyTorch and safetensors, for the Engine's own reader.

The reader (cyxwiz-engine/src/core/weight_file.cpp) opens these without PyTorch. expected.json holds every
tensor's name, shape, type and values as torch reports them, so the test compares value by value.

Files:
  tiny_state_dict.pth   a module's state_dict plus tensors that stress the reader: a transposed
                        (non-contiguous) matrix, a slice that starts inside shared storage, float64,
                        float16, an int64 counter and a scalar
  protocol4.pth         the same kind of dict saved with pickle protocol 4
  checkpoint.pth        a training checkpoint: {"epoch", "lr", "state_dict": {...}}
  tiny.safetensors      safetensors with metadata, float16 and an int64 tensor
  whole_model.pth       torch.save(module): refused, it is not a state-dict
  hostile.pth           a pickle that asks to run os.system: refused, nothing runs
  legacy.pth            the dict of tiny_state_dict.pth in the format before PyTorch 1.6
  truncated.pth         the first half of tiny_state_dict.pth
  deep.pth              a pickle nested 200,000 levels deep: refused, no stack overflow
  expanded.pth          one stored value shown as a million (stride 0): refused, not expanded
  overlap.safetensors   two entries on the same bytes: refused

Run: python generate_weight_file_fixtures.py
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "weight_files"

DTYPES = {torch.float32: "F32", torch.float64: "F64", torch.float16: "F16", torch.int64: "I64"}


def describe(state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    entries = []
    for name, tensor in state.items():
        entry: dict[str, Any] = {"name": name, "shape": list(tensor.shape), "dtype": DTYPES[tensor.dtype]}
        if tensor.dtype.is_floating_point:
            entry["values"] = tensor.detach().to(torch.float32).contiguous().reshape(-1).tolist()
        entries.append(entry)
    return entries


def tiny_state() -> OrderedDict[str, torch.Tensor]:
    torch.manual_seed(1909)
    model = torch.nn.Sequential(OrderedDict([
        ("conv", torch.nn.Conv2d(3, 2, 3, bias=False)),
        ("bn", torch.nn.BatchNorm2d(2)),
    ]))
    model(torch.randn(4, 3, 5, 5))  # moves the running statistics and the counter
    state = model.state_dict()
    base = torch.randn(2, 4)
    state["fc.weight"] = base.t()                                   # [4, 2], strides (1, 4): not contiguous
    state["fc.bias"] = torch.randn(4, dtype=torch.float64)
    shared = torch.arange(10, dtype=torch.float32) * 0.5
    state["slice"] = shared[3:7]                                    # starts 3 elements into its storage
    state["half"] = torch.tensor([0.5, -1.25, 65504.0, 6e-8, 0.0], dtype=torch.float16)  # largest and a subnormal
    state["scalar"] = torch.tensor(2.5)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    out: Path = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    expected: dict[str, Any] = {"generator": "generate_weight_file_fixtures.py", "torch_version": torch.__version__,
                                "files": {}}

    state = tiny_state()
    torch.save(state, out / "tiny_state_dict.pth")
    expected["files"]["tiny_state_dict.pth"] = describe(state)

    torch.save(state, out / "protocol4.pth", pickle_protocol=4)
    expected["files"]["protocol4.pth"] = describe(state)

    torch.manual_seed(19091)
    inner = OrderedDict([("head.weight", torch.randn(2, 3)), ("head.bias", torch.randn(2))])
    torch.save({"epoch": 3, "lr": 0.1, "note": "after warmup", "history": [1.5, 1.2], "state_dict": inner},
               out / "checkpoint.pth")
    expected["files"]["checkpoint.pth"] = describe(inner)

    torch.manual_seed(19092)
    safe = {"b.weight": torch.randn(3, 2), "a.half": torch.tensor([1.5, -2.0], dtype=torch.float16),
            "a.count": torch.tensor([7, 8, 9], dtype=torch.int64), "c.double": torch.randn(2, dtype=torch.float64)}
    save_file(safe, str(out / "tiny.safetensors"), metadata={"model_name": "tiny"})
    # The reader lists tensors in the order of their data in the file.
    header_size = int.from_bytes((out / "tiny.safetensors").read_bytes()[:8], "little")
    header = json.loads((out / "tiny.safetensors").read_bytes()[8:8 + header_size])
    order = sorted((k for k in header if k != "__metadata__"), key=lambda k: header[k]["data_offsets"][0])
    expected["files"]["tiny.safetensors"] = describe({k: safe[k] for k in order})

    torch.save(torch.nn.Linear(2, 2), out / "whole_model.pth")
    torch.save(state, out / "legacy.pth", _use_new_zipfile_serialization=False)
    expected["files"]["legacy.pth"] = describe(state)
    data = (out / "tiny_state_dict.pth").read_bytes()
    (out / "truncated.pth").write_bytes(data[: len(data) // 2])

    # GLOBAL os.system, a text argument, TUPLE1, REDUCE, STOP: what a hostile "weight file" looks like.
    command = b"echo this must never run"
    hostile = b"\x80\x02cos\nsystem\nX" + len(command).to_bytes(4, "little") + command + b"\x85R."
    with zipfile.ZipFile(out / "hostile.pth", "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", hostile)
        archive.writestr("archive/version", "3\n")

    # A pickle nested 200,000 deep: must be refused without exhausting the stack.
    deep = b"\x80\x02)" + b"\x85" * 200000 + b"."
    with zipfile.ZipFile(out / "deep.pth", "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", deep)
    # One stored value presented as a million (stride 0): must not be expanded.
    torch.save({"w": torch.zeros(1).expand(1000000)}, out / "expanded.pth")
    # Two safetensors entries that claim the same bytes.
    header = json.dumps({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                         "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (out / "overlap.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))

    (out / "expected.json").write_text(json.dumps(expected, indent=1) + "\n", encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in sorted(out.iterdir())}
    print(f"wrote {out}: {sizes}")


if __name__ == "__main__":
    main()
