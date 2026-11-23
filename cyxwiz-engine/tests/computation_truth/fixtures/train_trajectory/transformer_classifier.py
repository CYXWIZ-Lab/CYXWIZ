# Auto-generated PyTorch model from CyxWiz Node Editor
# Generated at: (not stamped)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term[:encoding[0, :, 1::2].shape[-1]])
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError('sequence length exceeds max_sequence_length')
        return x + self.encoding[:, :x.size(1), :]

def configure_ffn_dropout(block, probability):
    block.dropout.p = probability
    return block

class CausalTransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_first=False, ffn_dropout=0.0):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=norm_first)
        self.block.dropout.p = ffn_dropout

    def forward(self, x):
        causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1)
        return self.block(x, src_mask=causal_mask)

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()

class ConfigurableAttention(nn.Module):
    # Self-attention with the CyxWiz options; returns (output, None) like nn.MultiheadAttention (key/value args unused).
    def __init__(self, d_model, nhead, dropout=0.0, bias=True, num_kv_heads=0, causal=True, qk_norm=False,
                 norm_eps=1e-5, position_encoding='none', rope_base=10000.0, rope_fraction=1.0,
                 sliding_window=0, logit_softcap=0.0):
        super().__init__()
        self.nhead, self.kv_heads, self.head_dim = nhead, num_kv_heads or nhead, d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # QK-norm: per-head RMSNorm of queries and keys before the scores (OLMo 2, Gemma 3, Qwen3)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.causal, self.position_encoding, self.rope_base = causal, position_encoding, rope_base
        self.rope_dims = int(rope_fraction * self.head_dim / 2 + 1e-6) * 2
        self.sliding_window, self.logit_softcap, self.dropout = sliding_window, logit_softcap, nn.Dropout(dropout)

    def _rotary(self, x):
        # half-split RoPE on the first rope_dims features: x*cos + rotate_half(x)*sin, base^(-2i/rope_dims)
        seq, dim = x.size(-2), self.rope_dims
        inv_freq = self.rope_base ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
        angle = torch.arange(seq, device=x.device, dtype=x.dtype)[:, None] * inv_freq[None, :]
        cos, sin = torch.cat([angle.cos(), angle.cos()], -1), torch.cat([angle.sin(), angle.sin()], -1)
        rot, rest = x[..., :dim], x[..., dim:]
        x1, x2 = rot[..., :dim // 2], rot[..., dim // 2:]
        return torch.cat([rot * cos + torch.cat([-x2, x1], -1) * sin, rest], -1)

    def _alibi_slopes(self):
        # Press et al. 2022, including head counts that are not a power of two
        power_of_two = lambda n: [(2.0 ** (-8.0 / n)) ** i for i in range(1, n + 1)]
        closest = 2 ** math.floor(math.log2(self.nhead))
        return power_of_two(closest) + power_of_two(2 * closest)[0::2][:self.nhead - closest]

    def forward(self, x, key=None, value=None, need_weights=False):
        batch, seq, width = x.shape
        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.position_encoding == 'rope':
            q, k = self._rotary(q), self._rotary(k)
        groups = self.nhead // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.logit_softcap > 0:
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
        pos = torch.arange(seq, device=x.device)
        if self.position_encoding == 'alibi':
            slopes = torch.tensor(self._alibi_slopes(), device=x.device, dtype=x.dtype)
            scores = scores + slopes[:, None, None] * (pos[None, :] - pos[:, None]).to(x.dtype)
        if self.causal:
            hidden = pos[None, :] > pos[:, None]
            if self.sliding_window > 0:
                hidden = hidden | (pos[:, None] - pos[None, :] >= self.sliding_window)
            scores = scores.masked_fill(hidden, float('-inf'))
        context = self.dropout(torch.softmax(scores, -1)) @ v
        return self.out_proj(context.transpose(1, 2).reshape(batch, seq, width)), None

class ConfigurableCausalDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, norm_first=False, ffn_dropout=0.0,
                 norm_type='layer_norm', norm_eps=1e-5, ffn_type='mlp', ffn_activation='relu', ffn_bias=True,
                 position_encoding='external', rope_base=10000.0, attention_bias=True, qk_norm=False,
                 rope_fraction=1.0, block_layout='sequential', sandwich_norm=False, residual_init_scale=1.0,
                 attn_logit_softcap=0.0, sliding_window=0, num_kv_heads=0):
        super().__init__()
        self.norm_first, self.parallel = norm_first, block_layout == 'parallel'
        self.self_attn = ConfigurableAttention(d_model, nhead, dropout, attention_bias, num_kv_heads, True, qk_norm,
                                               norm_eps, 'none' if position_encoding == 'external' else position_encoding,
                                               rope_base, rope_fraction, sliding_window, attn_logit_softcap)
        make_norm = (lambda: nn.RMSNorm(d_model, eps=norm_eps)) if norm_type == 'rms_norm' else (lambda: nn.LayerNorm(d_model, eps=norm_eps))
        self.norm1 = make_norm()
        self.norm2 = None if self.parallel else make_norm()  # parallel blocks share norm1
        # sandwich norm (Gemma 2/3): also normalize each sub-layer output before the residual add
        self.post_attn_norm = make_norm() if sandwich_norm else None
        self.post_ffn_norm = make_norm() if sandwich_norm else None
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_bias)  # up
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_bias)  # down
        self.gate = nn.Linear(d_model, dim_feedforward, bias=ffn_bias) if ffn_type == 'gated' else None
        self.act = {'relu': nn.ReLU(), 'gelu': nn.GELU(approximate='tanh'), 'gelu_exact': nn.GELU(), 'silu': nn.SiLU(),
                    'mish': nn.Mish(), 'elu': nn.ELU(), 'selu': nn.SELU(), 'leaky_relu': nn.LeakyReLU(0.01),
                    'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'hardswish': nn.Hardswish(),
                    'squared_relu': SquaredReLU()}[ffn_activation]
        self.dropout1, self.dropout3, self.ffn_dropout = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(ffn_dropout)
        if residual_init_scale != 1.0:
            with torch.no_grad():  # GPT-2 style scaled init of the residual output projections
                self.self_attn.out_proj.weight.mul_(residual_init_scale)
                self.linear2.weight.mul_(residual_init_scale)

    def _ffn(self, x):
        hidden = self.act(self.gate(x)) * self.linear1(x) if self.gate is not None else self.act(self.linear1(x))
        return self.linear2(self.ffn_dropout(hidden))

    def forward(self, x):
        attend = lambda h: self.self_attn(h)[0]
        if self.parallel:
            h = self.norm1(x)
            return x + self.dropout1(attend(h)) + self.dropout3(self._ffn(h))
        if self.post_attn_norm is not None:
            x = x + self.dropout1(self.post_attn_norm(attend(self.norm1(x))))
            return x + self.dropout3(self.post_ffn_norm(self._ffn(self.norm2(x))))
        if self.norm_first:
            x = x + self.dropout1(attend(self.norm1(x)))
            return x + self.dropout3(self._ffn(self.norm2(x)))
        x = self.norm1(x + self.dropout1(attend(x)))
        return self.norm2(x + self.dropout3(self._ffn(x)))

class LearnedPositionalEncoding(nn.Module):
    # GPT-2 learned absolute positions: x + weight[:seq_len]
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.weight[:x.size(1)]

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        self.layer0 = configure_ffn_dropout(nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True, norm_first=False), 0.0)
        self.layer1 = nn.LazyLinear(out_features=3)

    def forward(self, x):
        x = x.reshape(x.size(0), *[6, 8])
        x = self.layer0(x)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        return x

# Training setup
if __name__ == '__main__':
    # Create model
    model = GeneratedModel()
    print(model)

    # Loss (from the graph's loss node) and optimizer (from its optimizer node)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    # lr_schedule=warmup_cosine per optimizer update (CyxWiz WarmupDecayLR)
    total_updates = 1000  # TODO: epochs * ceil(batches_per_epoch / grad_accum_steps)
    warmup = round(0.4 * total_updates)
    min_ratio = 0.1
    def lr_factor(done):  # done = completed updates; returns the factor for the next one
        u = min(max(done + 1, 1), total_updates)
        if u <= warmup:
            return u / warmup
        p = (u - warmup) / max(1, total_updates - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    # TODO: Add your training data here
    # Example training loop:
    # for epoch in range(num_epochs):
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #         optimizer.step()
    #         scheduler.step()
