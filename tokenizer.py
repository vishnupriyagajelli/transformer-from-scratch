import json
import re
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (token_a, token_b) -> merged_token
        self.vocab = {}           # token_str -> id
        self.inverse_vocab = {}   # id -> token_str

    def _get_stats(self, word_freqs):
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, word_freqs):
        merged = "".join(pair)
        pattern = re.escape(" ".join(pair))
        replacement = merged
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = re.sub(pattern, replacement, word)
            new_word_freqs[new_word] = freq
        return new_word_freqs, merged

    def train(self, text: str, vocab_size: int):
        words = text.strip().split()
        word_freqs = Counter(words)

        word_freqs = {" ".join(list(w)) + " </w>": f for w, f in word_freqs.items()}

        base_chars = set()
        for word in word_freqs:
            base_chars.update(word.split())

        self.vocab = {ch: i for i, ch in enumerate(sorted(base_chars))}

        num_merges = vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs, merged_token = self._merge_vocab(best_pair, word_freqs)
            self.merges[best_pair] = merged_token
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

        self.inverse_vocab = {i: t for t, i in self.vocab.items()}

    def _word_to_symbols(self, word: str):
        symbols = list(word) + ["</w>"]
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            pair_to_merge = mergeable[0]
            merged = self.merges[pair_to_merge]
            idx = pairs.index(pair_to_merge)
            symbols = symbols[:idx] + [merged] + symbols[idx + 2:]
        return symbols

    def encode(self, text: str):
        ids = []
        for word in text.strip().split():
            symbols = self._word_to_symbols(word)
            for sym in symbols:
                if sym in self.vocab:
                    ids.append(self.vocab[sym])
                else:
                    continue
        return ids

    def decode(self, ids: list):
        tokens = [self.inverse_vocab.get(i, "") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "merges": {f"{a} {b}": merged for (a, b), merged in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(i): t for t, i in self.vocab.items()}
        self.merges = {}
        for k, merged in data["merges"].items():
            a, b = k.split(" ")
            self.merges[(a, b)] = merged


if __name__ == "__main__":
    sample_text = "the quick brown fox jumps over the lazy dog the dog barks"
    tok = BPETokenizer()
    tok.train(sample_text, vocab_size=100)
    ids = tok.encode("the quick fox")
    print("encoded:", ids)
    print("decoded:", tok.decode(ids))