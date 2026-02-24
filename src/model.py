import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
"""
Configs collected from official website https://huggingface.co/docs/transformers/en/model_doc/biogpt
"""
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
"""
    Feed-Forwad Block
    This is the implementation of ffn from the official GPT-2 architecture
"""

class FeedForward(nn.Module):
    def __init__(self, args : Args):
        super().__init__()
        self.fc1 = nn.Linear(args.dim, args.hidden_dim, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_dim, args.dim, bias=args.bias)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

"""
    The Attention Mechanism
"""

class CasualAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads, bias = True):
        super().__init__()
        assert (d_out % num_heads == 0), "the d_out should be divisible to number of heads"
        self.d_in = d_in
        self.num_head = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
  def forward(self, x : torch.Tensor) -> torch.Tensor:
        B,T,C = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(B,T, self.num_head, self.head_dim)
        keys = keys.view(B,T, self.num_head, self.head_dim)
        values = values.view(B,T, self.num_head, self.head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_score = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:T, :T]

        attn_score = attn_score.masked_fill(mask_bool, float('-inf'))

        attn_weight = torch.softmax(attn_score / (self.head_dim ** 0.5), dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vector = (attn_weight @ values).transpose(1,2)
        context_vector = context_vector.contiguous().view(B,T,self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector

"""
    The Layer Normalization Block
    This is the implementation of the layernorm block used and explained in GPT-2
"""

class LayerNorm(nn.Module):
    def __init__(self, args : Args):
        super().__init__()
        self.eps = args.eps
        self.scale = nn.Parameter(torch.zeros(args.dim))
        self.shift = nn.Parameter(torch.zeros(args.dim))
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
"""
    Transformer block
    The most important part of the implementation and where the magic basically happens.
    As explained this is in the implementation of the GPT-2.
"""

class Transformer(nn.Module):
    def __init__(self, args : Args):
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
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

"""
    The model
"""

class Model(nn.Module):
    def __init__(self, args : Args) -> None:
        super().__init__()
        self.args = args
        self.token_embed = nn.Embedding(args.vocab_size, args.dim, padding_idx=1)
        self.pos_embed = nn.Embedding(1026, args.dim)
        self.drop = nn.Dropout(args.dropout_rate)
        self.final_norm = LayerNorm(args)
        self.trf_layers = nn.Sequential(*[Transformer(args) for _ in range(args.layers)])
        self.out_proj = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.embed_scale = math.sqrt(args.dim)
    def forward(self, x_in : torch.Tensor) -> torch.Tensor:
        B, T = x_in.shape
        tok_embeds = self.token_embed(x_in) * self.embed_scale
        positions = torch.arange(2, T + 2, device=x_in.device) 
        pos_embeds = self.pos_embed(positions)  
        x = tok_embeds + pos_embeds
        x = self.drop(x)
        x = self.trf_layers(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits
    def calculate_params(self):
        return f"{sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.out_proj.parameters()):,}"
