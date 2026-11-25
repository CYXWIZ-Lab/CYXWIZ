"""PyTorch export harness (TOFIX112).

The Node Editor's PyTorch export used to be compile-checked only. For each
graph this harness:
  1. exports the code headless (cyxwiz-engine --export-code pytorch ...),
  2. runs the generated script (model, optimizer and LR-schedule setup),
  3. imports GeneratedModel and runs forward + backward on synthetic input
     shaped from the graph's Data Input,
  4. compares its parameter count with the model CyxWiz itself builds for the
     graph from the same sample shape (cyxwiz-graph-compile-check --parameters
     --input-shape): offline no data sets the Data Input's shape.

Usage:
  python check_pytorch_export.py --bin <build>/bin/Release [--work DIR] graph.cyxgraph [...]
Exit code 0 when every graph passes.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


# gui::NodeType values (cyxwiz-engine/src/core/graph_model.h); a pattern
# template names the type instead.
EMBEDDING = (21, "Embedding")
CONCATENATE = (46, "Concatenate")


def load_graph(path: Path) -> dict:
    """The graph's nodes and links in one form, from either file the Engine
    opens: a saved graph (`nodes` / `links`, numeric types, `parameters`) or a
    pattern template (`template` with `nodes` / `links`, type names, `params`
    whose "$name" values are the template's parameter defaults)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("nodes"):
        return {"nodes": doc["nodes"], "links": doc.get("links", [])}
    template = doc.get("template") or {}
    if not template.get("nodes"):
        raise ValueError("neither a saved graph nor a pattern template (no nodes)")
    defaults = {p["name"]: str(p.get("default_value", "")) for p in doc.get("parameters", [])}

    def value(v):
        v = str(v)
        return defaults.get(v[1:], v) if v.startswith("$") else v

    # A block's one-sample shape for its first layer ("[8, $d_model]").
    sample_shape = re.sub(r"\$(\w+)", lambda m: defaults.get(m.group(1), m.group(0)), doc.get("sample_shape", ""))

    nodes = [{"id": n["id"], "type": n.get("type"), "name": n.get("name", ""),
              "parameters": {k: value(v) for k, v in (n.get("params") or n.get("parameters") or {}).items()}}
             for n in template["nodes"]]
    links = [{"from_node": l.get("from"), "to_node": l.get("to"),
              "to_pin_index": l.get("to_pin", 0)} for l in template.get("links", [])]
    return {"nodes": nodes, "links": links, "sample_shape": sample_shape, "needs": doc.get("needs", "")}


# Why the Engine's headless tool cannot count a model's parameters, when the
# reason is this PC and not the graph: nothing to compare the export with.
OFFLINE_REASONS = (
    ("could not open", "a file the model needs is not on this PC"),
    ("a block: it has no Data Input", "a block has no Data Input: the Engine builds its model only inside a graph"),
)


def cyxwiz_parameter_count(compile_check: Path, graph: Path, sample_shape):
    """(count, None) for the model CyxWiz builds from one sample of
    `sample_shape` (the Engine's layout: an image is [H, W, C]); (None, why)
    when it cannot build it: `why` names an OFFLINE_REASONS case, or starts
    with "FAIL" for any other reason (the Engine refuses or cannot build the
    graph)."""
    out = subprocess.run([str(compile_check), "--parameters", "--input-shape", json.dumps(list(sample_shape)),
                          str(graph)], capture_output=True, text=True)
    match = re.search(r"parameters=(\d+)", out.stdout)
    if match:
        return int(match.group(1)), None
    text = (out.stdout + out.stderr).strip()
    for marker, why in OFFLINE_REASONS:
        if marker in text:
            return None, why
    return None, "FAIL " + text[-300:]


def word_pos_embeddings(graph_json: dict):
    """(word, POS) Embedding nodes when a Concatenate joins two Embeddings on
    Input 1 / Input 2 (the compiler's word + POS fusion), else None. Links
    address inputs by to_pin_index."""
    nodes = {n["id"]: n for n in graph_json.get("nodes", [])}
    for concat in nodes.values():
        if concat.get("type") not in CONCATENATE:
            continue
        sources = {l.get("to_pin_index", 0): nodes.get(l["from_node"])
                   for l in graph_json.get("links", []) if l["to_node"] == concat["id"]}
        pair = (sources.get(0), sources.get(1))
        if all(n is not None and n.get("type") in EMBEDDING for n in pair):
            return pair
    return None


class NoOfflineInput(Exception):
    """Nothing can be fed to the model here (its input width comes from
    preprocessing). Its own class: an IndexError or KeyError from the model
    is a LookupError too, and was reported as skipped instead of failed."""


def synthetic_input(graph_json: dict, batch: int = 2):
    """(x, vocab, sample_shape): token ids for graphs that start with an
    Embedding (shorter than every length limit in the graph), floats shaped
    like the Data Input otherwise. sample_shape is one sample as the Engine
    holds it; an image sample [H, W, C] goes to PyTorch as [C, H, W]."""

    def floats(shape):
        x = torch.randn(batch, *shape)
        return (x.permute(0, 3, 1, 2).contiguous() if len(shape) == 3 else x), None, shape

    nodes = graph_json["nodes"]
    limits = [16]
    for node in nodes:
        for key in ("max_len", "max_length", "sequence_length", "max_sequence_length"):
            value = node.get("parameters", {}).get(key, "")
            if str(value).isdigit() and int(value) > 1:
                limits.append(int(value))
    fused = word_pos_embeddings(graph_json)
    if fused:  # word + POS fusion: packed [batch, seq, 2] ids, as CyxWiz feeds it
        word, pos = (int(n["parameters"]["num_embeddings"]) for n in fused)
        length = min(limits)
        ids = torch.stack([torch.randint(1, word, (batch, length)),
                           torch.randint(1, pos, (batch, length))], dim=-1)
        return ids.float(), None, [length, 2]
    for node in nodes:
        params = node.get("parameters", {})
        if "embedding_dim" in params and ("num_embeddings" in params or "vocab_size" in params):
            vocab = int(params.get("num_embeddings") or params["vocab_size"])
            return torch.randint(4, vocab, (batch, min(limits))), vocab, [min(limits)]
    if graph_json.get("sample_shape"):  # a pattern: the shape its first layer reads
        return floats([int(d) for d in json.loads(graph_json["sample_shape"])])
    for node in nodes:
        params = node.get("parameters", {})
        if params.get("source_type") == "file" or "file_type" in params:
            try:
                shape = json.loads(params.get("shape", "null"))
            except (TypeError, ValueError):
                shape = None
            if shape:
                return floats([int(d) for d in shape])
    raise NoOfflineInput("the input width is set by preprocessing and unknown offline")


def check(graph: Path, bin_dir: Path, work: Path):
    result = {"graph": graph.name}
    # A pattern the Engine marks not ready is refused on open, saying why.
    needs = load_graph(graph).get("needs")
    if needs:
        return {**result, "ok": None, "stage": "blocked", "detail": f"the pattern is not ready yet: {needs}"[:200]}
    code = work / (graph.stem + ".py")
    export = subprocess.run([str(bin_dir / "cyxwiz-engine.exe"), "--export-code", "pytorch", str(graph), str(code)],
                            capture_output=True, text=True)
    if export.returncode != 0 or not code.exists():
        text = export.stderr + export.stdout
        # The Engine refuses to open a graph holding a node it has not built
        # (a template node), saying which: blocked, as the export's refusals.
        refusal = re.search(r"Cannot open .*? as a graph: (.*a template that cannot be inserted)", text)
        if refusal:
            return {**result, "ok": None, "stage": "blocked", "detail": refusal.group(1)[:200]}
        return {**result, "ok": False, "stage": "export", "detail": text[-400:]}

    # The generator refuses graphs CyxWiz cannot train (it emits a leading
    # `raise ValueError(...)` naming why): reported as blocked, not failed.
    head = code.read_text(encoding="utf-8").splitlines()[:3]
    blocked = next((line for line in head if line.startswith("raise ValueError(")), None)
    if blocked:
        return {**result, "ok": None, "stage": "blocked", "detail": blocked[len("raise ValueError("):-1][:200]}

    # An RL graph exports a Stable-Baselines3 training run on the Engine's own
    # environments, not a GeneratedModel: it is compiled here, not trained.
    text = code.read_text(encoding="utf-8")
    if "from stable_baselines3 import" in text:
        try:
            compile(text, str(code), "exec")
        except SyntaxError as error:
            return {**result, "ok": False, "stage": "rl", "detail": f"SyntaxError: {error}"}
        return {**result, "ok": None, "stage": "rl", "detail": "a Stable-Baselines3 training script: compiled, not run"}

    script = subprocess.run([sys.executable, str(code)], capture_output=True, text=True, cwd=work)
    if script.returncode != 0:
        return {**result, "ok": False, "stage": "script", "detail": script.stderr[-600:]}

    try:
        spec = importlib.util.spec_from_file_location(graph.stem.replace("-", "_"), code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(0)
        model = module.GeneratedModel()
        x, vocab, sample_shape = synthetic_input(load_graph(graph))
        y = model(x)  # materializes lazy layers
        if vocab is not None and y.dim() == 3 and y.shape[-1] == vocab:  # next-token logits (a tagger's are tags)
            loss = torch.nn.functional.cross_entropy(y.reshape(-1, y.shape[-1]), x.reshape(-1))
        else:
            loss = y.float().pow(2).mean()
        loss.backward()
        params = list(model.parameters())
        missing = sum(1 for p in params if p.requires_grad and p.grad is None)
        # The Engine lists a BatchNorm's running mean and variance with its
        # parameters (they are saved and loaded with them); torch keeps them
        # as buffers. Count them on both sides.
        running = sum(b.numel() for name, b in model.named_buffers()
                      if name.endswith(("running_mean", "running_var")))
        result.update(output=list(y.shape), loss=float(loss.detach()),
                      pytorch_parameters=sum(p.numel() for p in params) + running, parameters_without_grad=missing)
    except NoOfflineInput as reason:  # nothing to feed it offline: not a verdict on the export
        return {**result, "ok": None, "stage": "skipped", "detail": str(reason)}
    except Exception as error:  # the export ran but the model does not train
        return {**result, "ok": False, "stage": "forward/backward", "detail": f"{type(error).__name__}: {error}"}

    expected, unavailable = cyxwiz_parameter_count(bin_dir / "cyxwiz-graph-compile-check.exe", graph, sample_shape)
    result["cyxwiz_parameters"] = expected
    if unavailable and unavailable.startswith("FAIL "):
        return {**result, "ok": False, "stage": "parameters",
                "detail": f"the Engine cannot build the model: {unavailable[5:]}"[:400]}
    if unavailable:  # this PC, not the graph: nothing to compare with
        return {**result, "ok": None, "stage": "skipped",
                "detail": f"the Engine's parameter count is not available here: {unavailable}"}
    same = expected is not None and expected == result["pytorch_parameters"]
    finite = torch.isfinite(torch.tensor(result["loss"])).item()
    result["ok"] = bool(same and finite and result["parameters_without_grad"] == 0)
    if not same:
        result["stage"] = "parameters"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--json", type=Path, help="write the results here")
    parser.add_argument("graphs", nargs="+", type=Path)
    args = parser.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="cyxwiz_pytorch_export_"))
    work.mkdir(parents=True, exist_ok=True)
    # The Engine changes directory at startup: pass absolute paths.
    results = [check(graph.resolve(), args.bin.resolve(), work.resolve()) for graph in args.graphs]
    for r in results:
        status = "ok  " if r["ok"] else (r["stage"].upper() if r["ok"] is None else "FAIL")
        extra = (f"params pytorch={r.get('pytorch_parameters')} cyxwiz={r.get('cyxwiz_parameters')} "
                 f"out={r.get('output')}") if "pytorch_parameters" in r else f"{r.get('stage')}: {r.get('detail')}"
        print(f"{status} {r['graph']}: {extra}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(1 for r in results if r["ok"])
    blocked = sum(1 for r in results if r["ok"] is None)
    failed = len(results) - passed - blocked
    print(f"{passed} pass, {blocked} blocked or skipped (the generator refuses them / no offline input), {failed} fail")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
