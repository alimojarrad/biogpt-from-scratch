import torch
import torch.nn as nn
from .model import Model, Args
from typing import Dict

params = torch.load("BioGPT_weights/pytorch_model.bin")

def assign(left : torch.Tensor, right : torch.Tensor) -> nn.Parameter:
    assert left.shape == right.shape, "Should be the same size"
    return nn.Parameter(right.detach())

def apply_weights(model : Model, args : Args) -> None:
    model.token_embed.weight = assign(model.token_embed.weight, params["biogpt.embed_tokens.weight"])
    model.pos_embed.weight = assign(model.pos_embed.weight, params["biogpt.embed_positions.weight"])
    for i in range(args.layers):
        trf_block = model.trf_layers[i]
        trf_block.attn.W_key.weight = assign(
            trf_block.attn.W_key.weight,
            params[f"biogpt.layers.{i}.self_attn.k_proj.weight"]
        )
        trf_block.attn.W_key.bias = assign(
            trf_block.attn.W_key.bias,
            params[f"biogpt.layers.{i}.self_attn.k_proj.bias"]
        )
        trf_block.attn.W_query.weight = assign(
            trf_block.attn.W_query.weight,
            params[f"biogpt.layers.{i}.self_attn.q_proj.weight"]
        )
        trf_block.attn.W_query.bias = assign(
            trf_block.attn.W_query.bias,
            params[f"biogpt.layers.{i}.self_attn.q_proj.bias"]
        )
        trf_block.attn.W_value.weight = assign(
            trf_block.attn.W_value.weight,
            params[f"biogpt.layers.{i}.self_attn.v_proj.weight"]
        )
        trf_block.attn.W_value.bias = assign(
            trf_block.attn.W_value.bias,
            params[f"biogpt.layers.{i}.self_attn.v_proj.bias"]
        )
        trf_block.attn.out_proj.weight = assign(
            trf_block.attn.out_proj.weight,
            params[f"biogpt.layers.{i}.self_attn.out_proj.weight"]
        )
        trf_block.attn.out_proj.bias = assign(
            trf_block.attn.out_proj.bias,
            params[f"biogpt.layers.{i}.self_attn.out_proj.bias"]
        )
        trf_block.ffn.fc1.weight = assign(
            trf_block.ffn.fc1.weight,
            params[f"biogpt.layers.{i}.fc1.weight"]
        )
        trf_block.ffn.fc1.bias = assign(
            trf_block.ffn.fc1.bias,
            params[f"biogpt.layers.{i}.fc1.bias"]
        )
        trf_block.ffn.fc2.weight = assign(
            trf_block.ffn.fc2.weight,
            params[f"biogpt.layers.{i}.fc2.weight"]
        )
        trf_block.ffn.fc2.bias = assign(
            trf_block.ffn.fc2.bias,
            params[f"biogpt.layers.{i}.fc2.bias"]
        )
        trf_block.ln1.scale = assign(
            trf_block.ln1.scale,
            params[f"biogpt.layers.{i}.self_attn_layer_norm.weight"]
        )
        trf_block.ln1.shift = assign(
            trf_block.ln1.shift,
            params[f"biogpt.layers.{i}.self_attn_layer_norm.bias"]
        )
        trf_block.ln2.scale = assign(
            trf_block.ln2.scale,
            params[f"biogpt.layers.{i}.final_layer_norm.weight"]
        )
        trf_block.ln2.shift = assign(
            trf_block.ln2.shift,
            params[f"biogpt.layers.{i}.final_layer_norm.bias"]
        )
    model.final_norm.scale = assign(
        model.final_norm.scale,
        params["biogpt.layer_norm.weight"]
    )
    model.final_norm.shift = assign(
        model.final_norm.shift,
        params["biogpt.layer_norm.bias"]
    )
    model.out_proj.weight = model.token_embed.weight
    print("weights loaded successfully")

    
