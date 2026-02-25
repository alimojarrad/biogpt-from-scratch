# BioGPT — From-Scratch Inference

A clean PyTorch reimplementation of [Microsoft's BioGPT](https://github.com/microsoft/BioGPT) for biomedical text generation, built without the HuggingFace `transformers` library.

## Project Structure

```
bioGPT/
├── generate.py               # CLI text generation script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── BioGPT_weights/
│   └── pytorch_model.bin     # Pretrained BioGPT weights
│
├── src/
│   ├── model.py              # BioGPT model architecture + Args config
│   └── loading_weight.py     # Pretrained weight loader
│
└── Tokenizer/
    ├── tokenizer.py          # BioGPT tokenizer (Moses BPE, no HuggingFace)
    ├── vocab.json            # BioGPT vocabulary (42,384 tokens)
    ├── merges.txt            # BPE merge rules (40,000 merges)
    └── README.md             # Tokenizer-specific documentation
```

## Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch
pip install sacremoses   # optional but recommended for full tokenizer fidelity
```

> **Note:** If `sacremoses` is not installed, the tokenizer falls back to a built-in Moses approximation that covers all standard biomedical text cases. Install it for training or when exact parity with the official tokenizer is required.

## Quickstart

```bash
python generate.py
```

You will be prompted to enter a biomedical text prompt:

```
Enter your prompt > COVID-19 is
```

**Example output:**
```
COVID-19 is a pandemic caused by the novel coronavirus SARS-CoV-2, which was first identified in December 2019 in Wuhan, China and subsequently spread worldwide.
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--max_new_tokens` | `150` | Maximum number of tokens to generate |
| `--temperature` | `0.8` | Sampling temperature. Lower = more focused |
| `--top_k` | `50` | Restrict sampling to top-k tokens. `0` to disable |
| `--num_samples` | `1` | Number of independent outputs to generate |
| `--min_new_tokens` | `20` | Minimum tokens before EOS is allowed |
| `--repetition_penalty` | `1.0` | Penalise repeated tokens. `> 1.0` reduces repetition |

### Example Commands

```bash
# Default generation
python generate.py

# More creative output
python generate.py --temperature 1.0 --top_k 100

# Deterministic / greedy output (same result every run)
python generate.py --temperature 0.0 --top_k 0

# Generate multiple samples
python generate.py --num_samples 3 --temperature 0.9

# Longer output with repetition penalty
python generate.py --max_new_tokens 300 --repetition_penalty 1.3
```

## Generation Settings Guide

| Goal | `temperature` | `top_k` |
|---|---|---|
| Deterministic (same output every run) | `0.0` | `0` |
| Focused / factual | `0.3 – 0.6` | `30 – 50` |
| Balanced (default) | `0.8` | `50` |
| Creative / diverse | `1.0 – 1.2` | `100+` |

## Tokenizer

The tokenizer (`Tokenizer/tokenizer.py`) is a faithful reimplementation of the [official BioGPT tokenizer](https://huggingface.co/microsoft/biogpt) with no HuggingFace dependency.

**Pipeline:**
```
Raw text
  └─► Moses tokenization   (aggressive dash splits: COVID-19 → COVID @-@ 19)
  └─► BPE encoding         (</w> marks end-of-word boundaries)
  └─► Vocab lookup         (42,384 token vocabulary)
  └─► Prepend </s> as BOS  (fairseq convention, id=2)
```

**Key design notes:**
- End-of-word marker is `</w>` (fairseq-style), not `Ġ` (GPT-2-style)
- BOS token is `</s>` (id=2), matching fairseq's `sep_token` convention
- Hyphens between words are replaced by `@-@` (id=9) during tokenization and restored on decode
- HTML entities (`&amp;`, `&lt;`, etc.) are escaped on encode and unescaped on decode
- No `</s>` is appended to prompts — the model generates its own EOS

**Usage:**

```python
from Tokenizer.tokenizer import BioGPTTokenizer

tok = BioGPTTokenizer("Tokenizer/vocab.json", "Tokenizer/merges.txt")

# Tokenize
tokens = tok.tokenize("SARS-CoV-2 causes COVID-19.")
# ['SARS</w>', '@-@</w>', 'CoV</w>', '@-@</w>', '2</w>', 'causes</w>', 'COVID</w>', '@-@</w>', '19</w>', '.</w>']

# Encode (returns ids with </s> prepended as BOS)
ids = tok.encode("Breast cancer is")
# [2, 5888, 101, 21]

# Decode
text = tok.decode([2, 5888, 101, 21, 14, 998, 76, 4])
# 'Breast cancer is a viral disease.'

# Batch encode with padding
batch = tok.batch_encode(
    ["COVID-19 is", "The patient was diagnosed"],
    pad=True
)
```

## Credits

- Original model: [Microsoft BioGPT](https://github.com/microsoft/BioGPT) — Luo et al., 2022
- Tokenizer reference: [HuggingFace BioGptTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/biogpt/tokenization_biogpt.py)