"""Live PyTorch parity for the configurable decoder block (tofix112).

Includes rotary position embedding (position_encoding=rope).

Checks every TransformerDecoder block option against an explicit PyTorch
reference: RMSNorm vs LayerNorm, gated (GLU-family) vs plain MLP feed-forward,
each supported feed-forward activation, and feed-forward bias on/off, in both
post-norm and pre-norm layouts, as two-block causal stacks. The reference class
is executed from the text the Engine's PyTorch export emits
(ConfigurableCausalDecoderBlock in node_editor_codegen.cpp), so this also proves
exported code matches the backend.

Compares outputs, input gradients and every parameter gradient
(atol 3e-5, rtol 3e-4, as verify_transformer_stacks.py).

  py -3.12 verify_transformer_block_options.py --runtime <build>/bin/Release \
      --output <dir> --backend cpu --dll-dir "C:/Program Files/ArrayFire/v3/lib"
"""
import argparse, ctypes, hashlib, json, math, os, sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--backend', choices=['cpu', 'cuda', 'opencl', 'oneapi'], default='cpu')
parser.add_argument('--device', type=int, default=0, help='device index for the backend (OpenCL: 1 = Intel iGPU here)')
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
activation = cx.Device(getattr(cx.DeviceType, args.backend.upper()), args.device).activate_exact(True)
assert activation.success and activation.execution_validated, activation.message

D_MODEL, HEADS, FF = 16, 4, 12


CODEGEN = Path(__file__).resolve().parents[3] / 'src' / 'gui' / 'node_editor_codegen.cpp'


def exported_block_source():
    """The PyTorch text the Engine's export emits for the configurable block."""
    lines = CODEGEN.read_text(encoding='utf-8').splitlines()
    first = next(i for i, l in enumerate(lines) if l.strip() == 'code += "class SquaredReLU(nn.Module):\\n";')
    out = []
    for line in lines[first:]:
        text = line.strip()
        if not text.startswith('code += "'):
            break
        out.append(text[len('code += "'):-2].encode('utf-8').decode('unicode_escape'))
    return ''.join(out)


# The reference IS the exported code: parity here also proves the export.
EXPORTED_BLOCK = exported_block_source()
exec(compile(EXPORTED_BLOCK, str(CODEGEN) + ' (exported ConfigurableCausalDecoderBlock)', 'exec'), globals())

def tensor(a):
    return cx.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def attention_mapping(attn, prefix, grads=False):
    """Backend attention parameter names -> ConfigurableAttention tensors."""
    pick = (lambda t: t.grad) if grads else (lambda t: t)
    out = {}
    for letter, proj in (('q', attn.q_proj), ('k', attn.k_proj), ('v', attn.v_proj), ('o', attn.out_proj)):
        out[f'{prefix}W_{letter}'] = pick(proj.weight)
        if proj.bias is not None:
            out[f'{prefix}b_{letter}'] = pick(proj.bias)
    if attn.q_norm is not None:
        out[f'{prefix}q_norm_gamma'] = pick(attn.q_norm.weight)
        out[f'{prefix}k_norm_gamma'] = pick(attn.k_norm.weight)
    return out


def mapping(ref, grads=False):
    """Backend parameter name -> PyTorch tensor (or its .grad)."""
    pick = (lambda t: t.grad) if grads else (lambda t: t)
    out = attention_mapping(ref.self_attn, 'self_attn.', grads)
    for name in ('norm1', 'norm2', 'post_attn_norm', 'post_ffn_norm'):
        norm = getattr(ref, name)
        if norm is None:
            continue
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
            for norm in (ref.norm1, ref.norm2, ref.post_attn_norm, ref.post_ffn_norm,
                         ref.self_attn.q_norm, ref.self_attn.k_norm):
                if norm is None:
                    continue
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
     'position_encoding': 'rope', 'attention_bias': False},        # full LLaMA-style block (preset llama_style)
    {'attention_bias': False},
    {'qk_norm': True},
    {'qk_norm': True, 'norm_eps': 1e-6, 'position_encoding': 'rope'},
    {'norm_type': 'rms_norm', 'ffn_type': 'gated', 'ffn_activation': 'silu', 'ffn_bias': False,
     'position_encoding': 'rope', 'attention_bias': False, 'qk_norm': True},  # LLaMA + QK-norm (Qwen3-like, no GQA)
    {'ffn_type': 'gated', 'ffn_activation': 'squared_relu'},
    # groups 2-4 (2026-09-25)
    {'position_encoding': 'alibi'},
    {'position_encoding': 'rope', 'rope_fraction': 0.5},
    {'position_encoding': 'rope', 'rope_fraction': 0.75, 'qk_norm': True},  # rounds down to 2 of 4
    {'num_kv_heads': 2},
    {'num_kv_heads': 1, 'position_encoding': 'rope'},
    {'num_kv_heads': 2, 'qk_norm': True, 'attention_bias': False},
    {'sliding_window': 2},
    {'sliding_window': 3, 'position_encoding': 'alibi'},
    {'attn_logit_softcap': 1.0},
    {'attn_logit_softcap': 0.5, 'position_encoding': 'rope', 'qk_norm': True},
    {'block_layout': 'parallel', 'pre_only': True},
    {'block_layout': 'parallel', 'norm_type': 'rms_norm', 'ffn_type': 'gated', 'ffn_activation': 'gelu',
     'position_encoding': 'rope', 'pre_only': True},                       # GPT-J / PaLM-like
    {'sandwich_norm': True, 'pre_only': True},
    {'sandwich_norm': True, 'norm_type': 'rms_norm', 'ffn_type': 'gated', 'ffn_activation': 'gelu',
     'position_encoding': 'rope', 'qk_norm': True, 'attn_logit_softcap': 2.0, 'num_kv_heads': 2,
     'sliding_window': 4, 'attention_bias': False, 'ffn_bias': False, 'pre_only': True},  # Gemma-2/3-like
] + [{'ffn_activation': a} for a in ('silu', 'mish', 'elu', 'selu', 'leaky_relu', 'tanh', 'hardswish',
                                     'gelu_exact', 'squared_relu')]

results = {'torch_version': torch.__version__, 'device': 'ArrayFire ' + args.backend,
           'shape': [2, 5, D_MODEL], 'heads': HEADS, 'dim_feedforward': FF,
           'atol': 3e-5, 'rtol': 3e-4, 'cases': []}
failures = []
for options in CASES:
    options = dict(options)
    pre_only = options.pop('pre_only', False)
    for pre in ((True,) if pre_only else (False, True)):
        try:
            result = case(options, pre)
            results['cases'].append(result)
            print('PASS', options or 'classic', 'norm_first=', pre,
                  'max_abs_error=%.2e' % result['max_abs_error'], flush=True)
        except AssertionError as error:
            failures.append({'options': options, 'norm_first': pre, 'error': str(error)[:2000]})
            print('FAIL', options or 'classic', 'norm_first=', pre, str(error)[:600], flush=True)


def run_extra(label, fn):
    try:
        result = fn()
        result['label'] = label
        results['cases'].append(result)
        print('PASS', label, 'max_abs_error=%.2e' % result['max_abs_error'], flush=True)
    except AssertionError as error:
        failures.append({'label': label, 'error': str(error)[:2000]})
        print('FAIL', label, str(error)[:600], flush=True)


def learned_positions_case():
    torch.manual_seed(52)
    ref = LearnedPositionalEncoding(D_MODEL, 7)
    module = cx.LearnedPositionalEmbedding(D_MODEL, 7)
    module.set_parameters({'weight': tensor(ref.weight.detach().numpy())})
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    upstream = torch.randn_like(x)
    expected = ref(x)
    errors = {'output': compare(module.forward(tensor(x.detach().numpy())).to_numpy(), expected)}
    expected.backward(upstream)
    errors['input_gradient'] = compare(module.backward(tensor(upstream.numpy())).to_numpy(), x.grad)
    errors['weight_gradient'] = compare(module.get_gradients()['weight'].to_numpy(), ref.weight.grad)
    return {'max_abs_error': max(errors.values()), 'errors': errors}


def attention_module_case(options):
    torch.manual_seed(52)
    bias = options.get('bias', True)
    ref = ConfigurableAttention(D_MODEL, HEADS, 0.0, bias, options.get('num_kv_heads', 0), options.get('causal', False),
                                options.get('qk_norm', False), 1e-5, options.get('position_encoding', 'none'),
                                options.get('rope_base', 10000.0), options.get('rope_fraction', 1.0),
                                options.get('sliding_window', 0), options.get('logit_softcap', 0.0))
    with torch.no_grad():
        for norm in (ref.q_norm, ref.k_norm):
            if norm is not None:
                norm.weight.uniform_(0.5, 1.5)
    module = cx.AttentionModule(D_MODEL, HEADS, 0.0, bias, **{k: v for k, v in options.items() if k != 'bias'})
    module.set_training(True)
    params = {k: tensor(v.detach().numpy()) for k, v in attention_mapping(ref, '').items()}
    module.set_parameters(params)
    assert set(module.get_parameters()) == set(params), sorted(set(module.get_parameters()) ^ set(params))
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    upstream = torch.randn_like(x)
    expected = ref(x)[0]
    errors = {'output': compare(module.forward(tensor(x.detach().numpy())).to_numpy(), expected)}
    expected.backward(upstream)
    errors['input_gradient'] = compare(module.backward(tensor(upstream.numpy())).to_numpy(), x.grad)
    grads = module.get_gradients()
    for name, value in attention_mapping(ref, '', grads=True).items():
        errors[name] = compare(grads[name].to_numpy(), value)
    return {'options': options, 'max_abs_error': max(errors.values()), 'errors': errors}


run_extra('learned positions', learned_positions_case)
for attention_options in [
    {},
    {'causal': True},
    {'causal': True, 'position_encoding': 'rope', 'qk_norm': True},
    {'causal': True, 'position_encoding': 'alibi', 'sliding_window': 3},
    {'position_encoding': 'rope', 'rope_fraction': 0.5, 'num_kv_heads': 2},  # bidirectional
    {'causal': True, 'logit_softcap': 1.0, 'num_kv_heads': 1, 'bias': False},
]:
    run_extra('attention module ' + json.dumps(attention_options),
              lambda o=attention_options: attention_module_case(o))
results['failures'] = failures
results['hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [BIN / 'cyxwiz-backend.dll', Path(cx.__file__), Path(__file__), CODEGEN]}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / f'block_options_parity_{args.backend}{args.device}.json').write_text(json.dumps(results, indent=2) + '\n',
                                                                        encoding='utf-8')
print('SUMMARY', len(results['cases']), 'passed,', len(failures), 'failed', flush=True)
sys.exit(1 if failures else 0)
