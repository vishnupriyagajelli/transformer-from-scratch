import torch
from config import Config
from tokenizer import BPETokenizer
from model import TransformerLM


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    batch_size, prompt_len = input_ids.shape

    n_heads = model.layers[0].attn.n_heads
    head_dim = model.layers[0].attn.head_dim
    kv_caches = [
        {
            "k": torch.zeros(batch_size, n_heads, 0, head_dim, device=device),
            "v": torch.zeros(batch_size, n_heads, 0, head_dim, device=device),
        }
        for _ in model.layers
    ]

    logits = model(input_ids, kv_caches=kv_caches, start_pos=0)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    for i in range(max_new_tokens - 1):
        pos = prompt_len + i
        logits = model(next_token, kv_caches=kv_caches, start_pos=pos)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

    return tokenizer.decode(generated)


def main():
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = BPETokenizer()
    tokenizer.load(cfg.tokenizer_path)

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
    ).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    prompt = "the quick brown"
    output = generate(model, tokenizer, prompt, max_new_tokens=30, device=device)
    print("prompt:", prompt)
    print("generated:", output)


if __name__ == "__main__":
    main()