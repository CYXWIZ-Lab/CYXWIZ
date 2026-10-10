"""PyTorch fixtures for LSTMModule / GRUModule / RNNModule (TOFIX140, recurrent layers on the device).

torch.nn.LSTM / nn.GRU / nn.RNN (tanh and relu) with batch_first=True, 1-2 layers, uni- and bidirectional, full
sequence or last step. Each case stores the input, the weights under CyxWiz's parameter keys,
the output, a fixed upstream gradient, dL/dx and every weight / bias gradient.

Key names (cyxwiz-backend/src/algorithms/sequential/recurrent_modules.cpp):
  unidirectional:  layer{L}_W_ih, layer{L}_W_hh, layer{L}_b_ih, layer{L}_b_hh
  bidirectional:   layer{L}.forward.W_ih ... layer{L}.reverse.b_hh

    py -3.12 generate_recurrent_fixtures.py
"""

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "recurrent_pytorch.json"


def tensor_json(t):
    t = t.detach().to(torch.float32).contiguous()
    return {"shape": list(t.shape), "values": [round(v, 7) for v in t.reshape(-1).tolist()]}


def key(layer, name, bidirectional, reverse):
    if not bidirectional:
        return f"layer{layer}_{name}"
    return f"layer{layer}.{'reverse' if reverse else 'forward'}.{name}"


def make_case(name, kind, input_size, hidden, layers, bidirectional, return_sequences, batch, seq, seed,
              nonlinearity="tanh"):
    torch.manual_seed(seed)
    if kind == "RNN":
        net = torch.nn.RNN(input_size, hidden, num_layers=layers, nonlinearity=nonlinearity, batch_first=True,
                           bidirectional=bidirectional)
    else:
        cls = torch.nn.LSTM if kind == "LSTM" else torch.nn.GRU
        net = cls(input_size, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
    with torch.no_grad():
        for p in net.parameters():
            p.uniform_(-0.4, 0.4)
    x = torch.randn(batch, seq, input_size, requires_grad=True)
    out, _ = net(x)
    y = out if return_sequences else out[:, -1, :]
    grad = torch.randn_like(y)
    y.backward(grad)

    params, grads = {}, {}
    for layer in range(layers):
        for reverse in ([False, True] if bidirectional else [False]):
            suffix = f"_l{layer}" + ("_reverse" if reverse else "")
            for torch_name, ours in [("weight_ih", "W_ih"), ("weight_hh", "W_hh"), ("bias_ih", "b_ih"), ("bias_hh", "b_hh")]:
                p = getattr(net, torch_name + suffix)
                params[key(layer, ours, bidirectional, reverse)] = tensor_json(p)
                grads[key(layer, ours, bidirectional, reverse)] = tensor_json(p.grad)
    return {
        "name": name, "kind": kind, "nonlinearity": nonlinearity, "input_size": input_size, "hidden_size": hidden, "num_layers": layers,
        "bidirectional": bidirectional, "return_sequences": return_sequences,
        "input": tensor_json(x), "parameters": params, "output": tensor_json(y),
        "grad_output": tensor_json(grad), "grad_input": tensor_json(x.grad), "parameter_gradients": grads,
    }


def main():
    cases = []
    seed = 0
    for kind in ["LSTM", "GRU"]:
        k = kind.lower()
        for name, in_size, hidden, layers, bi, rs, batch, seq in [
            (f"{k}_h8_l1_seq", 5, 8, 1, False, True, 3, 6),
            (f"{k}_h16_l2_last", 6, 16, 2, False, False, 2, 7),
            (f"{k}_bi_h8_l1_seq", 5, 8, 1, True, True, 3, 6),
            (f"{k}_bi_h16_l2_last", 6, 16, 2, True, False, 2, 5),
            (f"{k}_h64_l1_seq", 16, 64, 1, False, True, 4, 12),   # a real size: CUDA JIT trees
        ]:
            seed += 1
            cases.append(make_case(name, kind, in_size, hidden, layers, bi, rs, batch, seq, seed))
    for name, in_size, hidden, layers, bi, rs, batch, seq, act in [
        ("rnn_tanh_h8_l1_seq", 5, 8, 1, False, True, 3, 6, "tanh"),
        ("rnn_relu_h16_l2_last", 6, 16, 2, False, False, 2, 7, "relu"),
        ("rnn_tanh_bi_h8_l2_seq", 5, 8, 2, True, True, 3, 6, "tanh"),
        ("rnn_relu_bi_h16_l1_last", 6, 16, 1, True, False, 2, 5, "relu"),
        ("rnn_tanh_h64_l1_seq", 16, 64, 1, False, True, 4, 12, "tanh"),
    ]:
        seed += 1
        cases.append(make_case(name, "RNN", in_size, hidden, layers, bi, rs, batch, seq, seed, act))
    OUT.write_text(json.dumps({"schema_version": 1, "torch_version": torch.__version__, "cases": cases}))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
