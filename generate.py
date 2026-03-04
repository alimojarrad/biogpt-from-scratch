import torch
import argparse
from Tokenizer.tokenizer import BioGPTTokenizer
from src.model import Model, Args
from src.loading_weight import apply_weights


def load_model(device: str):
    args = Args()
    model = Model(args)
    model.to(device)
    apply_weights(model, args)
    model.eval()
    return model, args


def text_to_token_ids(text: str, tokenizer: BioGPTTokenizer, device: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor([ids], dtype=torch.long, device=device)


def token_ids_to_text(ids: torch.Tensor, tokenizer: BioGPTTokenizer) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


@torch.no_grad()
def generate_stream(
    model,
    idx: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_id: int | None = 2,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
):
    tokens_generated = 0
    decoded_so_far = tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
    idx_cond = idx[:, -context_size:]
    logits, kv_caches = model(idx_cond, kv_caches=None)
    logits = logits[:, -1, :]
    idx_next = _sample(
        logits, idx, tokens_generated,
        temperature, top_k, eos_id, min_new_tokens, repetition_penalty,
    )

    if eos_id is not None and idx_next.item() == eos_id:
        print()
        return idx

    idx = torch.cat((idx, idx_next), dim=1)
    full_text = tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
    print(full_text[len(decoded_so_far):], end="", flush=True)
    decoded_so_far = full_text
    tokens_generated += 1
    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model(idx_next, kv_caches=kv_caches)
        logits = logits[:, -1, :]

        idx_next = _sample(
            logits, idx, tokens_generated,
            temperature, top_k, eos_id, min_new_tokens, repetition_penalty,
        )

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        full_text = tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
        print(full_text[len(decoded_so_far):], end="", flush=True)
        decoded_so_far = full_text
        tokens_generated += 1

    print()
    return idx


def _sample(
    logits: torch.Tensor,
    idx: torch.Tensor,
    tokens_generated: int,
    temperature: float,
    top_k: int | None,
    eos_id: int | None,
    min_new_tokens: int,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty != 1.0:
        for token_id in set(idx[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    if temperature != 1.0:
        logits = logits / temperature

    if eos_id is not None and tokens_generated < min_new_tokens:
        logits[:, eos_id] = -float("inf")

    if top_k is not None:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(1, indices, values)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="BioGPT Text Generation")
    parser.add_argument("--max_new_tokens",     type=int,   default=100)
    parser.add_argument("--temperature",        type=float, default=0.6)
    parser.add_argument("--top_k",              type=int,   default=40)
    parser.add_argument("--num_samples",        type=int,   default=1)
    parser.add_argument("--min_new_tokens",     type=int,   default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BioGPTTokenizer("Tokenizer/vocab.json", "Tokenizer/merges.txt")
    model, model_args = load_model(device)

    print("\nType 'quit' to exit.\n")

    while True:
        prompt = input("\nEnter your prompt > ").strip()

        if prompt.lower() == "quit":
            print("Exiting...")
            break

        if not prompt:
            print("Empty prompt — try again.")
            continue

        encoded = text_to_token_ids(prompt, tokenizer, device)

        for i in range(args_cli.num_samples):
            print(f"\n{prompt} ", end="", flush=True)

            generate_stream(
                model=model,
                idx=encoded.clone(),
                tokenizer=tokenizer,
                max_new_tokens=args_cli.max_new_tokens,
                context_size=model_args.context_length,
                temperature=args_cli.temperature,
                top_k=args_cli.top_k,
                eos_id=tokenizer.eos_token_id,
                min_new_tokens=args_cli.min_new_tokens,
                repetition_penalty=args_cli.repetition_penalty,
            )

        print("=" * 80)


if __name__ == "__main__":
    main()