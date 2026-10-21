"""Live two-block PyTorch parity for a built CyxWiz Windows runtime.

Pass --runtime, --output and dependency --dll-dir paths explicitly. No experiment
files or corpus are required. The default backend is ArrayFire CPU.
"""
import argparse,ctypes,hashlib,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--backend',choices=['cpu','cuda','opencl'],default='cpu')
parser.add_argument('--dll-dir',action='append',type=Path,default=[])
args=parser.parse_args()
if os.name!='nt':parser.error('This runtime loader currently supports Windows only')
BIN=args.runtime.resolve()
HANDLES=[os.add_dll_directory(str(p.resolve())) for p in [BIN]+args.dll_dir]
sys.path.insert(0,str(BIN))
import numpy as np
import torch
backend_handle=ctypes.WinDLL(str(BIN/'cyxwiz-backend.dll'))
import pycyxwiz as cx
torch.set_num_threads(1)
backend=args.backend
activation=cx.Device(getattr(cx.DeviceType,backend.upper()),0).activate_exact(True)
assert activation.success and activation.execution_validated,activation.message


def tensor(a):
    return cx.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def mapping(ref, kind):
    """Map independently initialized PyTorch parameters into existing names."""
    result = {}
    for prefix, attention in [('self_attn', ref.self_attn)] + (
            [('cross_attn', ref.multihead_attn)] if kind == 'memory' else []):
        for i, letter in enumerate('qkv'):
            result[f'{prefix}.W_{letter}'] = attention.in_proj_weight[i*4:(i+1)*4]
            result[f'{prefix}.b_{letter}'] = attention.in_proj_bias[i*4:(i+1)*4]
        result[f'{prefix}.W_o'] = attention.out_proj.weight
        result[f'{prefix}.b_o'] = attention.out_proj.bias
    for name in ['norm1', 'norm2'] + (['norm3'] if kind == 'memory' else []):
        result[f'{name}.gamma'] = getattr(ref, name).weight
        result[f'{name}.beta'] = getattr(ref, name).bias
    for name in ['linear1', 'linear2']:
        result[f'{name}.weights'] = getattr(ref, name).weight
        result[f'{name}.bias'] = getattr(ref, name).bias
    return result


def grad_mapping(ref, kind):
    # Same mapping, using .grad on leaf Parameters before slicing Q/K/V.
    class GradView:
        pass
    view = GradView()
    for name in ['self_attn'] + (['multihead_attn'] if kind == 'memory' else []):
        original = getattr(ref, name)
        attention = GradView()
        attention.in_proj_weight = original.in_proj_weight.grad
        attention.in_proj_bias = original.in_proj_bias.grad
        attention.out_proj = GradView()
        attention.out_proj.weight = original.out_proj.weight.grad
        attention.out_proj.bias = original.out_proj.bias.grad
        setattr(view, name, attention)
    for name in ['norm1', 'norm2', 'linear1', 'linear2'] + (
            ['norm3'] if kind == 'memory' else []):
        module = GradView()
        module.weight = getattr(ref, name).weight.grad
        module.bias = getattr(ref, name).bias.grad
        setattr(view, name, module)
    return mapping(view, kind)


def compare(actual, expected):
    actual = np.asarray(actual)
    expected = expected.detach().numpy()
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    return float(np.max(np.abs(actual - expected)))


def stack_case(kind, pre):
    torch.manual_seed(52)
    refs, layers = [], []
    for _ in range(2):
        ref_cls = torch.nn.TransformerDecoderLayer if kind == 'memory' else torch.nn.TransformerEncoderLayer
        ref = ref_cls(4, 2, 7, dropout=0., batch_first=True, norm_first=pre)
        cls = cx.TransformerEncoderLayer if kind == 'encoder' else cx.TransformerDecoderLayer
        layer = cls(4, 2, 7, 0., pre)
        layer.set_training(True)
        layer.set_parameters({k: tensor(v.detach().numpy()) for k, v in mapping(ref, kind).items()})
        refs.append(ref)
        layers.append(layer)
    x = torch.randn(2, 3, 4, requires_grad=True)
    memory = torch.randn(2, 5, 4, requires_grad=True)
    upstream = torch.randn_like(x)
    mask = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
    actual, expected = tensor(x.detach().numpy()), x
    cmemory, cmask = tensor(memory.detach().numpy()), tensor(mask.numpy())
    for ref, layer in zip(refs, layers):
        if kind == 'memory':
            actual = layer.forward_with_memory(actual, cmemory, cmask)
            expected = ref(expected, memory, tgt_mask=mask)
        else:
            actual = layer.forward(actual)
            expected = ref(expected, src_mask=mask if kind == 'causal' else None)
    errors = {'output': compare(actual.to_numpy(), expected)}
    expected.backward(upstream)
    derivative = tensor(upstream.numpy())
    memory_derivatives = []
    for layer in reversed(layers):
        derivative = layer.backward(derivative)
        if kind == 'memory':
            memory_derivatives.append(layer.get_last_memory_gradient().to_numpy())
    errors['input_gradient'] = compare(derivative.to_numpy(), x.grad)
    if memory_derivatives:
        errors['memory_gradient'] = compare(sum(memory_derivatives), memory.grad)
    for i, (ref, layer) in enumerate(zip(refs, layers)):
        grads = layer.get_parameters()
        for name, value in grad_mapping(ref, kind).items():
            prefix, suffix = name.rsplit('.', 1)
            errors[f'layer{i}.{name}'] = compare(grads[f'{prefix}.grad_{suffix}'].to_numpy(), value)
    return {'kind': kind, 'norm_first': pre, 'depth': 2,
            'max_abs_error': max(errors.values()), 'comparisons': len(errors), 'errors': errors}


results = {'torch_version': torch.__version__, 'device': 'ArrayFire ' + backend,
           'dropout': 0, 'atol': 3e-5, 'rtol': 3e-4, 'cases': []}
for kind in ['encoder', 'causal', 'memory']:
    for pre in [False, True]:
        case = stack_case(kind, pre)
        results['cases'].append(case)
        print('PASS', kind, 'norm_first=', pre, 'max_abs_error=', case['max_abs_error'], flush=True)

# An accepted additive mask with a completely blocked row must not silently
# poison a model. Record the current behavior as a review finding.
attention = cx.MultiHeadAttention(4, 2, 0., True)
x = tensor(np.arange(24).reshape(2, 3, 4) / 10)
mask = np.zeros((3, 3), dtype=np.float32)
mask[0, :] = -np.inf
y = attention.forward_qkv(x, x, x, tensor(mask)).to_numpy()
results['fully_masked_row'] = {'finite': bool(np.isfinite(y).all()),
                              'nonfinite_elements': int((~np.isfinite(y)).sum())}
print('OBSERVED fully masked row:', results['fully_masked_row'], flush=True)
assert results['fully_masked_row']['finite'], 'Fully blocked row must remain finite'
results['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [BIN/'cyxwiz-backend.dll',Path(cx.__file__),Path(__file__)]}
args.output.mkdir(parents=True,exist_ok=True)
(args.output/f'parity_{backend}.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
