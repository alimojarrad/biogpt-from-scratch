# BioGPT From Scratch

A clean PyTorch reimplementation of Microsoft's BioGPT model, including
manual HuggingFace weight loading, weight tying verification, and
custom autoregressive text generation.

This project reproduces the architecture and behavior of the official
BioGPT causal language model for biomedical text generation.

---

## Overview

This repository implements:

- Token embeddings
- Positional embeddings
- Multi-head causal self-attention
- Transformer decoder blocks
- Layer normalization
- Language modeling head
- Manual HuggingFace weight mapping
- Custom autoregressive generation (temperature, top-k, repetition penalty)

The goal of this project is to deeply understand GPT-style decoder architectures
by rebuilding BioGPT from scratch and validating correctness against the
official pretrained model.

---

## Architecture

The model follows the standard decoder-only Transformer structure:

Token Embedding  
→ Positional Embedding  
→ N × Transformer Blocks  
→ Final LayerNorm  
→ LM Head (weight tied with token embedding)

Causal masking is applied to ensure autoregressive generation.

---

## Pretrained Weights

Official pretrained weights from:

`microsoft/biogpt`

are used only for verification and are NOT included in this repository.

See:

```
biogpt_weights/README.md
```

for instructions on downloading the weights using `huggingface_hub`.

---

## Installation

Tested with:

- Python 3.10
- PyTorch ≥ 2.0
- Transformers ≥ 4.35

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Text Generation

```bash
python generate.py
```

Or with arguments:

```bash
python generate.py --max_new_tokens 200 --temperature 0.7 --top_k 50 --num_samples 3
```

Example prompt:

```
COVID-19 is
```

Example output:

```
COVID-19 is a disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). A new global health problem.
```

---

## Verification

Correctness was validated through:

- Parameter-by-parameter weight mapping
- Weight tying confirmation
- Matching vocabulary alignment
- Coherent biomedical text generation
- Output style comparison with HuggingFace implementation

---

## Project Structure

```
biogpt-from-scratch/
├── generate.py
├── README.md
├── requirements.txt
│
├── BioGPT_weights/
│   └── pytorch_model.bin
│   └── README.md
│
├── src/
│   ├── model.py
│   └── loading_weight.py
│
└── Tokenizer/
    ├── tokenizer.py
    ├── vocab.json
    ├── merges.txt
    └── README.md
```

---

## Educational Purpose

This project is designed to:

- Understand GPT decoder internals
- Practice implementing attention mechanisms
- Learn weight mapping between custom models and HuggingFace
- Explore biomedical language modeling

It is not intended to replace the official implementation.

---

## Future Improvements

- KV-cache for faster decoding
- Top-p (nucleus) sampling
- LoRA fine-tuning support
- PubMedQA adaptation
- Perplexity benchmarking against official model

---

## License

Please refer to the original BioGPT model license on HuggingFace.

This repository only contains architectural code and does not distribute pretrained weights.