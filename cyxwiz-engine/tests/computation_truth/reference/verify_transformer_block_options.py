"""Live PyTorch parity for the configurable decoder block (tofix112).

Includes rotary position embedding (position_encoding=rope).

Checks every TransformerDecoder block option against an explicit PyTorch
reference: RMSNorm vs LayerNorm, gated (GLU-family) vs plain MLP feed-forward,
each supported feed-forward activation, and feed-forward bias on/off, in both
post-norm and pre-norm layouts, as two-block causal stacks. The reference class
is the same one the Engine's PyTorch export emits (ConfigurableCausalDecoderBlock),
so this also proves exported code matches the backend.

Compares outputs, input gradients and every parameter gradient
(atol 3e-5, rtol 3e-4, as verify_transformer_stacks.py).

  py -3.12 verify_transformer_block_options.py --runtime <build>/bin/Release \
      --output <dir> --backend cpu --dll-dir "C:/Program Files/ArrayFire/v3/lib"
"""
import argparse, ctypes, hashlib, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--backend', choices=['cpu', 'cuda', 'opencl'], default='cpu')
parser.add_argument('--dll-dir', action='append', type=Path, default=[])
args = parser.parse_args()
if os.name != 'nt':
    parser.error('This runtime loader currently supports Windows only')
BIN = args.runtime.resolve()
HANDLES = [os.add_dll_directory(str(p.resolve())) for p in [BIN] + args.dll_dir]
sys.path.insert(0, str(BIN))
import numpy as np
import torch
from torch import nn
backend_handle = ctypes.WinDLL(str(BIN / 'cyxwiz-backend.dll'))
import pycyxwiz as cx
torch.set_num_threads(1)
activation = cx.Device(getattr(cx.DeviceType, args.backend.upper()), 0).activate_exact(True)
assert activation.success and activation.execution_validated, activation.message

D_MODEL, HEADS, FF = 8, 2, 12


class ConfigurableCausalDecoderBlock(nn.Module):
    # Identical to the class emitted by node_editor_codegen.cpp.
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, norm_first=False, ffn_dropout=0.0,
                 norm_type='layer_norm', norm_eps=1e-5, ffn_type='mlp', ffn_activation='relu', ffn_bias=True,
                 position_encoding='external', rope_base=10000.0):
        super().__init__()
        self.norm_first = norm_first
        self.nhead, self.rope, self.rope_base, self.attn_dropout = nhead, position_encoding == 'rope', rope_base, dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        make_norm = (lambda: nn.RMSNorm(d_model, eps=norm_eps)) if norm_type == 'rms_norm' else (lambda: nn.LayerNorm(d_model, eps=norm_eps))
        self.norm1, self.norm2 = make_norm(), make_norm()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_bias)  # up
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_bias)  # down
        self.gate = nn.Linear(d_model, dim_feedforward, bias=ffn_bias) if ffn_type == 'gated' else None
        self.act = {'relu': nn.ReLU(), 'gelu': nn.GELU(approximate='tanh'), 'silu': nn.SiLU(), 'mish': nn.Mish(),
                    'elu': nn.ELU(), 'selu': nn.SELU(), 'leaky_relu': nn.LeakyReLU(0.01), 'sigmoid': nn.Sigmoid(),
                    'tanh': nn.Tanh(), 'hardswish': nn.Hardswish()}[ffn_activation]
        self.dropout1, self.dropout3, self.ffn_dropout = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(ffn_dropout)

    def _rotary(self, x):
        # half-split RoPE: x*cos + rotate_half(x)*sin, frequencies base^(-2i/head_dim)
        seq, dim = x.size(-2), x.size(-1)
        inv_freq = self.rope_base ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
        angle = torch.arange(seq, device=x.device, dtype=x.dtype)[:, None] * inv_freq[None, :]
        cos, sin = torch.cat([angle.cos(), angle.cos()], -1), torch.cat([angle.sin(), angle.sin()], -1)
        x1, x2 = x[..., :dim // 2], x[..., dim // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def _self_attention(self, x):
        if not self.rope:
            causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1)
            return self.dropout1(self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0])
        batch, seq, width = x.shape
        q, k, v = torch.nn.functional.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias).chunk(3, -1)
        heads = lambda t: t.view(batch, seq, self.nhead, width // self.nhead).transpose(1, 2)
        q, k, v = self._rotary(heads(q)), self._rotary(heads(k)), heads(v)
        context = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0)
        return self.dropout1(self.self_attn.out_proj(context.transpose(1, 2).reshape(batch, seq, width)))

    def _feed_forward(self, x):
        hidden = self.act(self.gate(x)) * self.linear1(x) if self.gate is not None else self.act(self.linear1(x))
        return self.dropout3(self.linear2(self.ffn_dropout(hidden)))

    def forward(self, x):
        if self.norm_first:
            x = x + self._self_attention(self.norm1(x))
            return x + self._feed_forward(self.norm2(x))
        x = self.norm1(x + self._self_attention(x))
        return self.norm2(x + self._feed_forward(x))


def tensor(a):
    return cx.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def mapping(ref, grads=False):
    """Backend parameter name -> PyTorch tensor (or its .grad)."""
    pick = (lambda t: t.grad) if grads else (lambda t: t)
    out = {}
    attn = ref.self_attn
    for i, letter in enumerate('qkv'):
        out[f'self_attn.W_{letter}'] = pick(attn.in_proj_weight)[i * D_MODEL:(i + 1) * D_MODEL]
        out[f'self_attn.b_{letter}'] = pick(attn.in_proj_bias)[i * D_MODEL:(i + 1) * D_MODEL]
    out['self_attn.W_o'] = pick(attn.out_proj.weight)
    out['self_attn.b_o'] = pick(attn.out_proj.bias)
    for name in ('norm1', 'norm2'):
        norm = getattr(ref, name)
        out[f'{name}.gamma'] = pick(norm.weight)
        if isinstance(norm, nn.LayerNorm):
            out[f'{name}.beta'] = pick(norm.bias)
    for backend_name, module in (('linear1', ref.linear1), ('linear2', ref.linear2), ('ffn_gate', ref.gate)):
        if module is None:
            continue
        out[f'{backend_name}.weights'] = pick(module.weight)
        if module.bias is not None:
            out[f'{backend_name}.bias'] = pick(module.bias)
    return out


def compare(actual, expected):
    actual = np.asarray(actual)
    expected = expected.detach().numpy()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    return float(np.max(np.abs(actual - expected)))


def case(options, pre):
    torch.manual_seed(52)
    refs, layers = [], []
    for _ in range(2):
        ref = ConfigurableCausalDecoderBlock(D_MODEL, HEADS, FF, 0.0, pre, 0.0, **options)
        # Non-trivial norm scales/shifts so gamma/beta gradients are exercised.
        with torch.no_grad():
            for norm in (ref.norm1, ref.norm2):
                norm.weight.uniform_(0.5, 1.5)
                if getattr(norm, 'bias', None) is not None:
                    norm.bias.uniform_(-0.2, 0.2)
        layer = cx.TransformerDecoderLayer(D_MODEL, HEADS, FF, 0.0, pre, 0.0, **options)
        layer.set_training(True)
        params = {k: tensor(v.detach().numpy()) for k, v in mapping(ref).items()}
        layer.set_parameters(params)
        backend_names = {k for k in layer.get_parameters() if '.grad_' not in k}
        # cross_attn and norm3 exist in the backend for seq2seq and are unused here.
        used = {k for k in backend_names if not k.startswith(('cross_attn.', 'norm3.'))}
        assert used == set(params), ('parameter names differ', sorted(used ^ set(params)))
        refs.append(ref)
        layers.append(layer)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    upstream = torch.randn_like(x)
    actual, expected = tensor(x.detach().numpy()), x
    for ref, layer in zip(refs, layers):
        actual = layer.forward(actual)
        expected = ref(expected)
    errors = {'output': compare(actual.to_numpy(), expected)}
    expected.backward(upstream)
    derivative = tensor(upstream.numpy())
    for layer in reversed(layers):
        derivative = layer.backward(derivative)
    errors['input_gradient'] = compare(derivative.to_numpy(), x.grad)
    for i, (ref, layer) in enumerate(zip(refs, layers)):
        grads = layer.get_parameters()
        for name, value in mapping(ref, grads=True).items():
            prefix, suffix = name.rsplit('.', 1)
            errors[f'layer{i}.{name}'] = compare(grads[f'{prefix}.grad_{suffix}'].to_numpy(), value)
    return {'options': options, 'norm_first': pre, 'depth': 2,
            'max_abs_error': max(errors.values()), 'comparisons': len(errors), 'errors': errors}


CASES = [
    {},  # classic block (control)
    {'norm_type': 'rms_norm', 'ffn_type': 'gated', 'ffn_activation': 'silu', 'ffn_bias': False},  # LLaMA-style SwiGLU
    {'ffn_activation': 'gelu'},                                   # GPT-2-style FFN
    {'norm_type': 'rms_norm'},
    {'norm_type': 'rms_norm', 'norm_eps': 1e-6},
    {'ffn_type': 'gated', 'ffn_activation': 'gelu'},              # GEGLU
    {'ffn_type': 'gated', 'ffn_activation': 'relu'},              # ReGLU
    {'ffn_type': 'gated', 'ffn_activation': 'sigmoid'},           # GLU
    {'ffn_type': 'gated', 'ffn_activation': 'silu'},              # SwiGLU with bias
    {'ffn_bias': False},
    {'position_encoding': 'rope'},                                  # RoPE only
    {'position_encoding': 'rope', 'rope_base': 500.0},
    {'norm_type': 'rms_norm', 'ffn_type': 'gated', 'ffn_activation': 'silu', 'ffn_bias': False,
     'position_encoding': 'rope'},                                  # full LLaMA-style block
] + [{'ffn_activation': a} for a in ('silu', 'mish', 'elu', 'selu', 'leaky_relu', 'tanh', 'hardswish')]

results = {'torch_version': torch.__version__, 'device': 'ArrayFire ' + args.backend,
           'shape': [2, 5, D_MODEL], 'heads': HEADS, 'dim_feedforward': FF,
           'atol': 3e-5, 'rtol': 3e-4, 'cases': []}
failures = []
for options in CASES:
    for pre in (False, True):
        try:
            result = case(options, pre)
            results['cases'].append(result)
            print('PASS', options or 'classic', 'norm_first=', pre,
                  'max_abs_error=%.2e' % result['max_abs_error'], flush=True)
        except AssertionError as error:
            failures.append({'options': options, 'norm_first': pre, 'error': str(error)[:2000]})
            print('FAIL', options or 'classic', 'norm_first=', pre, str(error)[:600], flush=True)
results['failures'] = failures
results['hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [BIN / 'cyxwiz-backend.dll', Path(cx.__file__), Path(__file__)]}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / f'block_options_parity_{args.backend}.json').write_text(json.dumps(results, indent=2) + '\n',
                                                                        encoding='utf-8')
print('SUMMARY', len(results['cases']), 'passed,', len(failures), 'failed', flush=True)
sys.exit(1 if failures else 0)
