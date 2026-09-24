import math
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import Config
from tokenizer import BPETokenizer
from dataset import build_dataset
from model import TransformerLM


def get_lr(step, warmup_steps, max_lr, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = BPETokenizer()
    if os.path.exists(cfg.tokenizer_path):
        tokenizer.load(cfg.tokenizer_path)
    else:
        with open(cfg.data_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        tokenizer.train(text, vocab_size=cfg.vocab_size)
        tokenizer.save(cfg.tokenizer_path)

    dataset = build_dataset(cfg.data_path, tokenizer, cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=0.1)
    scaler = GradScaler(enabled=(device == "cuda"))
    total_steps = cfg.epochs * len(dataloader)
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            lr = get_lr(step, cfg.warmup_steps, cfg.max_lr, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} lr {lr:.6f}")

            if step % 200 == 0 and step > 0:
                torch.save(model.state_dict(), cfg.checkpoint_path)

            step += 1

        torch.save(model.state_dict(), cfg.checkpoint_path)
        print(f"saved checkpoint after epoch {epoch}")


if __name__ == "__main__":
    main()