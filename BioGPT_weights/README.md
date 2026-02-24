# BioGPT Pretrained Weights

This directory stores the official pretrained weights of **BioGPT** used to verify
the correctness of this custom PyTorch implementation.

These weights are downloaded directly from HuggingFace Hub.

---

## Source

Model repository:
microsoft/biogpt

The weights are NOT trained in this project.
They are only used for:

- Manual parameter mapping
- Weight tying verification
- Output generation comparison
- Architectural validation

---

## Downloading the Weights

Weights can be downloaded using `huggingface_hub`.

### Option 1 — Download Entire Repository

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/biogpt",
    local_dir="./biogpt_weights",
    local_dir_use_symlinks=False
)
```

---

### Option 2 — Download Specific Files

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="microsoft/biogpt",
    filename="pytorch_model.bin",
    local_dir="./biogpt_weights"
)
```

---

## Important

⚠ Pretrained weight files are large (>100MB).

They are NOT included in this GitHub repository.

To avoid exceeding GitHub file size limits:

- Weight files are ignored via `.gitignore`
- Only this README file is tracked

---

## Reproducibility

This project verifies correctness by:

- Parameter-by-parameter matching
- Confirming weight tying
- Comparing generated outputs with HuggingFace's official implementation

---

## License

Please refer to the original model repository on HuggingFace for license details.