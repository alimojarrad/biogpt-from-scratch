import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Args:
    vocab_size : int = 42384
    dim : int = 1024
    layers : int = 24
    n_heads : int = 16
    hidden_dim : int = 4096
    dropout_rate : float = 0.1
    context_length : int = 1024
    eps : float = 1e-12
    bias : bool = True


class FeedForward(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.fc1 = nn.Linear(args.dim, args.hidden_dim, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_dim, args.dim, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads, bias=True):
        super().__init__()
        assert (d_out % num_heads == 0), "the d_out should be divisible to number of heads"
        self.d_in = d_in
        self.num_head = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.dropout_rate = dropout_rate
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T_new, C = x.shape

        queries = self.W_query(x).view(B, T_new, self.num_head, self.head_dim).transpose(1, 2)
        keys    = self.W_key(x)  .view(B, T_new, self.num_head, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(B, T_new, self.num_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_keys, cached_values = kv_cache
            keys   = torch.cat([cached_keys,   keys],   dim=2)
            values = torch.cat([cached_values, values], dim=2)

        new_kv_cache = (keys, values)

        T_total = keys.size(2)
        dropout_p = self.dropout_rate if self.training else 0.0

        if self._use_flash:
            if T_new == T_total:
                context_vector = F.scaled_dot_product_attention(
                    queries, keys, values,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                T_q_offset = T_total - T_new
                attn_mask = torch.zeros(
                    T_new, T_total, dtype=queries.dtype, device=queries.device
                )
                mask_bool = self.mask.bool()[T_q_offset: T_q_offset + T_new, :T_total]
                attn_mask.masked_fill_(mask_bool, float("-inf"))
                context_vector = F.scaled_dot_product_attention(
                    queries, keys, values,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            T_q_offset = T_total - T_new
            attn_score = queries @ keys.transpose(2, 3)
            mask_bool  = self.mask.bool()
            attn_mask  = mask_bool[T_q_offset: T_q_offset + T_new, :T_total]
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
            attn_weight = torch.softmax(attn_score / (self.head_dim ** 0.5), dim=-1)
            attn_weight = self.dropout(attn_weight)
            context_vector = attn_weight @ values

        context_vector = context_vector.transpose(1, 2).contiguous().view(B, T_new, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector, new_kv_cache


class LayerNorm(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.eps = args.eps
        self.scale = nn.Parameter(torch.zeros(args.dim))
        self.shift = nn.Parameter(torch.zeros(args.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class Transformer(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.attn = CasualAttention(
            d_in=args.dim,
            d_out=args.dim,
            context_length=args.context_length,
            dropout_rate=args.dropout_rate,
            num_heads=args.n_heads,
            bias=args.bias
        )
        self.ffn = FeedForward(args)
        self.ln1 = LayerNorm(args)
        self.ln2 = LayerNorm(args)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, new_kv_cache


class Model(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.token_embed = nn.Embedding(args.vocab_size, args.dim, padding_idx=1)
        self.pos_embed = nn.Embedding(1026, args.dim)
        self.drop = nn.Dropout(args.dropout_rate)
        self.final_norm = LayerNorm(args)
        self.trf_layers = nn.Sequential(*[Transformer(args) for _ in range(args.layers)])
        self.out_proj = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.embed_scale = math.sqrt(args.dim)

    def forward(
        self,
        x_in: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        B, T = x_in.shape
        offset = 0
        if kv_caches is not None and kv_caches[0] is not None:
            offset = kv_caches[0][0].size(2)

        tok_embeds = self.token_embed(x_in) * self.embed_scale
        positions = torch.arange(offset + 2, offset + T + 2, device=x_in.device)
        pos_embeds = self.pos_embed(positions)
        x = self.drop(tok_embeds + pos_embeds)

        new_kv_caches = []
        for i, layer in enumerate(self.trf_layers):
            layer_cache = kv_caches[i] if (kv_caches is not None) else None
            x, new_cache = layer(x, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits, new_kv_caches

    def calculate_params(self):
        return f"{sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.out_proj.parameters()):,}"