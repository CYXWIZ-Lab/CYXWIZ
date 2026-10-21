"""Compare emitted ArrayFire dropout fixtures against live PyTorch autograd.

Uses each backend's exported Bernoulli mask, because PyTorch and ArrayFire RNG
streams are different. The C++ producer separately verifies seed replay and
finite differences with the same seed; no PyTorch RNG equality is assumed.
Usage: python verify_transformer_dropout.py emitted.json
"""
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)


def tensor(item, grad=False):
    return torch.tensor(item['data'], dtype=torch.float32).reshape(item['shape']).requires_grad_(grad)


def compare(item, expected):
    actual = tensor(item)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    return (actual-expected).abs().max().item()


def attention(case):
    p = {k: tensor(v, True) for k,v in case['parameters'].items()}
    q = tensor(case['input'], True)
    self_attention = case['kind']=='attention_self'
    k,v = (q,q) if self_attention else (tensor(case['key'], True), tensor(case['value'], True))
    def projection(x, letter):
        z = torch.nn.functional.linear(x, p['W_'+letter], p['b_'+letter])
        return z.reshape(z.shape[0],z.shape[1],2,2).transpose(1,2)
    qh,kh,vh = projection(q,'q'),projection(k,'k'),projection(v,'v')
    scores = qh @ kh.transpose(-1,-2) / (2**0.5)
    # The producer explicitly blocks the first row. Compute softmax only on
    # other rows; concatenate exact zero probabilities with zero derivatives.
    weights = torch.cat([torch.zeros_like(scores[...,:1,:]), scores[...,1:,:].softmax(-1)],dim=-2)
    weights = weights*tensor(case['mask']).permute(2,3,0,1)/(1-case['p'])
    context = (weights@vh).transpose(1,2).reshape(q.shape)
    y = torch.nn.functional.linear(context,p['W_o'],p['b_o'])
    y.backward(tensor(case['upstream']))
    errors = {'output':compare(case['output'],y), 'dx':compare(case['dx'],q.grad)}
    if not self_attention:
        errors.update(dk=compare(case['dk'],k.grad),dv=compare(case['dv'],v.grad))
    for name,value in p.items():
        errors[name] = compare(case['gradients']['grad_'+name],value.grad)
    return errors


class FixedMask(torch.nn.Module):
    def __init__(self, mask, p):
        super().__init__()
        self.mask,self.p = mask,p
    def forward(self,x):
        return x*self.mask/(1-self.p)


def transformer(case):
    # Decoder-only block is the encoder block computation plus a causal mask.
    is_memory = case['kind']=='memory'
    cls = torch.nn.TransformerDecoderLayer if is_memory else torch.nn.TransformerEncoderLayer
    ref = cls(4,2,5,dropout=0.,batch_first=True,norm_first=case['norm_first'])
    ref.dropout = FixedMask(tensor(case['mask']),case['p'])
    mapping = {}
    for i,c in enumerate('qkv'):
        mapping[f'self_attn.W_{c}'] = ref.self_attn.in_proj_weight[i*4:(i+1)*4]
        mapping[f'self_attn.b_{c}'] = ref.self_attn.in_proj_bias[i*4:(i+1)*4]
    mapping['self_attn.W_o'] = ref.self_attn.out_proj.weight
    mapping['self_attn.b_o'] = ref.self_attn.out_proj.bias
    if is_memory:
        for i,c in enumerate('qkv'):
            mapping[f'cross_attn.W_{c}'] = ref.multihead_attn.in_proj_weight[i*4:(i+1)*4]
            mapping[f'cross_attn.b_{c}'] = ref.multihead_attn.in_proj_bias[i*4:(i+1)*4]
        mapping['cross_attn.W_o'] = ref.multihead_attn.out_proj.weight
        mapping['cross_attn.b_o'] = ref.multihead_attn.out_proj.bias
    for name in ['norm1','norm2','linear1','linear2']+(['norm3'] if is_memory else []):
        module = getattr(ref,name)
        mapping[name+('.gamma' if name.startswith('norm') else '.weights')] = module.weight
        mapping[name+('.beta' if name.startswith('norm') else '.bias')] = module.bias
    with torch.no_grad():
        for name,value in mapping.items():
            value.copy_(tensor(case['parameters'][name]))
    x = tensor(case['input'],True)
    mask = torch.triu(torch.full((2,2),float('-inf')),diagonal=1) if case['kind']!='encoder' else None
    if is_memory:
        memory = tensor(case['memory'],True)
        y = ref(x,memory,tgt_mask=mask)
    else:
        y = ref(x,src_mask=mask)
    y.backward(tensor(case['upstream']))
    errors = {'output':compare(case['output'],y),'dx':compare(case['dx'],x.grad)}
    if is_memory:
        errors['dm'] = compare(case['dm'],memory.grad)
    for name,value in mapping.items():
        prefix,suffix = name.rsplit('.',1)
        attention = ref.multihead_attn if prefix=='cross_attn' else ref.self_attn
        if suffix.startswith('W_') and suffix[-1] in 'qkv':
            i = 'qkv'.index(suffix[-1]);grad = attention.in_proj_weight.grad[i*4:(i+1)*4]
        elif suffix.startswith('b_') and suffix[-1] in 'qkv':
            i = 'qkv'.index(suffix[-1]);grad = attention.in_proj_bias.grad[i*4:(i+1)*4]
        else:
            grad = value.grad
        errors[name] = compare(case['gradients'][prefix+'.grad_'+suffix],grad)
    return errors


source = Path(sys.argv[1])
results = {'torch_version':torch.__version__,'fixture':str(source),'atol':3e-5,'rtol':3e-4,'cases':[]}
for case in json.loads(source.read_text()):
    errors = attention(case) if case['kind'].startswith('attention') else transformer(case)
    record = {'kind':case['kind'],'norm_first':case.get('norm_first'),'errors':errors,'max_abs_error':max(errors.values())}
    results['cases'].append(record)
    print('PASS',record['kind'],record['norm_first'],record['max_abs_error'],flush=True)
source.with_suffix('.pytorch.json').write_text(json.dumps(results,indent=2)+'\n')
