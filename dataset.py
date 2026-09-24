import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "targets": targets}


def build_dataset(text_path: str, tokenizer, seq_len: int):
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    return TextDataset(token_ids, seq_len)