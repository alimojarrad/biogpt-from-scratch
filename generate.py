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
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_id: int | None = 2,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Auto-regressive token generation.

    Args:
        model          : the BioGPT Model instance
        idx            : (batch, seq_len) LongTensor of prompt token ids
        max_new_tokens : hard cap on tokens to generate
        context_size   : maximum sequence length the model accepts
        temperature    : softmax temperature  (1.0 = unscaled)
        top_k          : if set, restrict sampling to top-k logits
        eos_id         : stop when all beams emit this token id
        min_new_tokens : suppress eos_id for the first N generated tokens
        repetition_penalty : > 1.0 penalises tokens already in the sequence
    Returns:
        (batch, prompt_len + tokens_generated) LongTensor
    """
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
        
        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            if temperature != 1.0:
                logits = logits / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and tokens_generated < min_new_tokens:
            logits[:, eos_id] = -float("inf")

        probs    = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)
        tokens_generated += 1

    return idx

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
    print(tokenizer)

    model, model_args = load_model(device)

    prompt = input("\nEnter your prompt > ").strip()
    if not prompt:
        print("Empty prompt — exiting.")
        return

    encoded = text_to_token_ids(prompt, tokenizer, device)
    # print(f"Prompt tokens : {tokenizer.tokenize(prompt)}")
    # print(f"Prompt ids    : {encoded[0].tolist()}")

    for i in range(args_cli.num_samples):
        output_ids = generate(
            model             = model,
            idx               = encoded,
            max_new_tokens    = args_cli.max_new_tokens,
            context_size      = model_args.context_length,
            temperature       = args_cli.temperature,
            top_k             = args_cli.top_k,
            eos_id            = tokenizer.eos_token_id,
            min_new_tokens    = args_cli.min_new_tokens,
            repetition_penalty= args_cli.repetition_penalty,
        )

        generated_ids = output_ids[0, encoded.size(1):]
        text = token_ids_to_text(generated_ids, tokenizer)

        print(f"\n{'=' * 80}")
        if args_cli.num_samples > 1:
            print(f"[Sample {i + 1}/{args_cli.num_samples}]")
        print(f"{prompt} {text}")

    print("=" * 80)


if __name__ == "__main__":
    main()