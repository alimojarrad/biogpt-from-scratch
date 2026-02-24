import torch
import argparse
from transformers import BioGptTokenizer

from src.model import Model, Args
from src.loading_weight import apply_weights


def load_model(device: str):
    args = Args()
    model = Model(args)
    model.to(device)
    apply_weights(model, args)
    model.eval()
    return model, args


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True
    )
    return encoded["input_ids"].to(device)


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
    eos_id=None,
    min_new_tokens=0,
    repetition_penalty=1.0,
):
    tokens_generated = 0

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for b in range(idx.size(0)):
                for token_id in set(idx[b].tolist()):
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered

        if eos_id is not None and tokens_generated < min_new_tokens:
            logits[:, eos_id] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)
        tokens_generated += 1

    return idx


def main():
    parser = argparse.ArgumentParser(description="BioGPT Text Generation")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)

    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    model, model_args = load_model(device)

    prompt = input("Enter your prompt > ")

    encoded = text_to_token_ids(prompt, tokenizer, device)

    for _ in range(args_cli.num_samples):
        output_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=args_cli.max_new_tokens,
            context_size=model_args.context_length,
            temperature=args_cli.temperature,
            top_k=args_cli.top_k,
            eos_id = tokenizer.eos_token_id,
            min_new_tokens=20,
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print(text)


if __name__ == "__main__":
    main()