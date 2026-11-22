#!/usr/bin/env python3
"""Level 2 oracle fixtures (TOFIX228 1.5b): each zoo model trained a few steps
by PyTorch running the Engine's own PyTorch export of the graph.

Input: fixtures/train_trajectory/<model>.py and <model>.json, written by
`test_oracle_train --write-exports <that folder>` (the unstamped export and
the model's spec: input shape, rows, batch size, task).

For each model this script runs the export as a user would:
  1. its setup (everything before the commented example loop): the model,
     the criterion, the optimizer and any scheduler, with the export's
     `total_updates = 1000` placeholder set to this run's update count;
  2. the lazy layers are materialised by one forward pass in eval mode (no
     BatchNorm statistics move) and the starting weights are saved as a
     PyTorch state-dict, <model>_init.pth, which the Engine starts from;
  3. the example training loop, uncommented, runs over the fixture's rows in
     order (one epoch, `num_epochs = 1`); its loader records, before handing
     out each next batch, the step that just finished: the epoch's running
     loss and every parameter and buffer.

Output: fixtures/train_trajectory_pytorch.json (torch layouts; the Engine
test converts them with its own weight matcher).

    py -3.12 generate_train_trajectory_fixtures.py [--steps N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
SEED = 2281
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODELS_DIR = FIXTURES / "train_trajectory"
DEFAULT_OUTPUT = FIXTURES / "train_trajectory_pytorch.json"
DEFAULT_STEPS = 5
TOLERANCE = {"atol": 2e-5, "rtol": 2e-4, "noise_atol": 1e-3}
NOISE_GRADIENT = 1e-6
LOOP_MARKER = "    # Example training loop:\n"
SETUP_END = "    # TODO: Add your training data here\n"


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(torch.float32).contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def state(model: torch.nn.Module) -> dict[str, Any]:
    """Every parameter and float buffer (BatchNorm running statistics), by
    state-dict name; num_batches_tracked is a counter, not compared."""
    return {k: plain(v) for k, v in model.state_dict().items() if v.is_floating_point()}


def split_export(code: str) -> tuple[str, str]:
    """(setup, loop): the script up to the example loop, and the loop with its
    comment marks removed."""
    if SETUP_END not in code or LOOP_MARKER not in code:
        raise ValueError("the export has no example training loop")
    setup = code[: code.index(SETUP_END)]
    loop_lines = []
    for line in code[code.index(LOOP_MARKER) + len(LOOP_MARKER):].splitlines():
        if not line.strip():
            continue
        match = re.match(r"^    # ?(.*)$", line)
        if not match:
            raise ValueError("unexpected line in the example loop: " + line)
        loop_lines.append(match.group(1))
    return setup, "\n".join(loop_lines) + "\n"


def make_rows(spec: dict[str, Any], generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    rows = spec["rows"]
    x = torch.randn(rows, *spec["input_shape"], generator=generator)
    if spec["task"] == "classification":
        y = torch.randint(0, spec["classes"], (rows,), generator=generator)
    else:
        y = torch.randn(rows, generator=generator)
    return x, y


class RecordingLoader:
    """The example loop's train_loader: the rows in order, batch by batch.
    Before each next batch (and at the end) it records the step just done."""

    def __init__(self, x, y, batch_size, namespace, steps):
        self.x, self.y, self.batch_size = x, y, batch_size
        self.namespace, self.steps = namespace, steps
        self.records: list[dict[str, Any]] = []
        self.losses: list[float] = []
        self.grad_max: dict[str, torch.Tensor] = {}  # each element's largest |gradient| so far

    def record(self):
        loss = self.namespace["loss"]
        self.losses.append(float(loss.detach()))
        for name, p in self.namespace["model"].named_parameters():
            g = p.grad.detach().abs() if p.grad is not None else torch.zeros_like(p)
            self.grad_max[name] = torch.maximum(self.grad_max[name], g) if name in self.grad_max else g
        self.records.append({
            "step": len(self.records) + 1,
            "batch_loss": self.losses[-1],
            "running_loss": sum(self.losses) / len(self.losses),
            "parameters": state(self.namespace["model"]),
        })

    def __iter__(self):
        batches = self.x.shape[0] // self.batch_size
        for b in range(min(batches, self.steps)):
            if b > 0:
                self.record()
            lo = b * self.batch_size
            yield self.x[lo:lo + self.batch_size], self.y[lo:lo + self.batch_size]
        self.record()


def run_model(name: str, steps: int) -> dict[str, Any]:
    code = (MODELS_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    spec = json.loads((MODELS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if spec["rows"] < steps * spec["batch_size"]:
        raise ValueError(f"{name}: {spec['rows']} rows are fewer than {steps} batches")
    setup, loop = split_export(code)
    setup = setup.replace("total_updates = 1000  #", f"total_updates = {steps}  #")

    torch.manual_seed(SEED)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(setup, f"{name}.py (setup)", "exec"), namespace)
    model = namespace["model"]
    x, y = make_rows(spec, torch.Generator().manual_seed(SEED + 1))

    # Image rows are [H, W, C] for the Engine; torch takes images channels
    # first, as image data reaches a torch model.
    torch_x = x.permute(0, 3, 1, 2).contiguous() if spec.get("image_rows") else x
    model.eval()
    with torch.no_grad():
        model(torch_x[: spec["batch_size"]])  # materialises the lazy layers
    model.train()
    init_name = f"{name}_init.pth"
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, MODELS_DIR / init_name)

    loader = RecordingLoader(torch_x, y, spec["batch_size"], namespace, steps)
    namespace.update({"train_loader": loader, "num_epochs": 1})
    exec(compile(loop, f"{name}.py (example loop)", "exec"), namespace)
    # Elements no gradient reaches (|g| < NOISE_GRADIENT at every step: a key
    # bias softmax ignores, a bias a BatchNorm takes the mean of): only
    # rounding noise moves them, which Adam / RMSprop scale up to steps; the
    # Engine test holds them to an absolute bound.
    noise_mask = {}
    for pname, g in loader.grad_max.items():
        mask = (g < NOISE_GRADIENT).to(torch.float32)
        if mask.any():
            noise_mask[pname] = plain(mask)
    print(f"{name}: {len(loader.records)} steps, loss {loader.losses[0]:.6f} -> {loader.losses[-1]:.6f}, "
          f"noise-driven {sorted(noise_mask)}")
    return {
        "name": name,
        "summary": spec["summary"],
        "export_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "task": spec["task"],
        "rows": steps * spec["batch_size"],
        "batch_size": spec["batch_size"],
        "init_weights": init_name,
        "x": plain(x[: steps * spec["batch_size"]].reshape(steps * spec["batch_size"], -1)),
        "y": y[: steps * spec["batch_size"]].tolist(),
        "steps": loader.records,
        "noise_mask": noise_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    names = sorted(p.stem for p in MODELS_DIR.glob("*.json"))
    if not names:
        raise SystemExit(f"no exports in {MODELS_DIR}: run test_oracle_train --write-exports there first")
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "seed": SEED,
        "tolerance": TOLERANCE,
        "models": [run_model(name, args.steps) for name in names],
    }
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
