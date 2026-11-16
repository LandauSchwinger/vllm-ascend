"""
The task: "needles in Shakespeare".

A stream of Shakespeare prose with small records spliced in:

    ...the king hath [q=47] sent for you, my lord ... and so [q?47] farewell...
                      ^ SET: key q gets value 47          ^ QUERY: what was q?

Predicting prose needs only *local* context. Predicting the two digits of a
QUERY needs exactly one far-away place in the context: the matching SET.
That is the access pattern sparse attention is built for, and it gives us a
sharp metric (query digit accuracy, chance = 10%) to watch while we make
attention sparse and squeeze the model into int8.
"""

import os
import random
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "input.txt")

EXTRA = "0123456789[]=?"
KEY_CHARS = "abcdefghijklmnopqrstuvwxyz"


def load_text():
    if not os.path.exists(PATH):
        print(f"downloading tiny shakespeare -> {PATH}")
        urllib.request.urlretrieve(URL, PATH)
    with open(PATH, encoding="utf-8") as f:
        return f.read()


class Needles:
    def __init__(self, seed=1337, gap=(15, 45)):
        self.gap = gap  # how many prose characters between two records (min, max)
        text = load_text()
        self.chars = sorted(set(text) | set(EXTRA))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))
        self.vocab_size = len(self.chars)
        n = int(0.9 * len(text))
        self.text = {"train": text[:n], "val": text[n:]}
        self.rng = random.Random(seed)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def sample(self, split, length):
        """One stream of `length` chars.

        Returns (string, queries) where queries is a list of
        (position of a query digit, position of the same digit in its SET).
        """
        rng, text = self.rng, self.text[split]
        p = rng.randrange(len(text) - 2 * length)
        out, book, queries = [], {}, []  # book: key -> position of its first value digit
        while len(out) < length:
            gap = rng.randint(*self.gap)
            out.extend(text[p : p + gap])
            p += gap
            if book and rng.random() < 0.5:  # QUERY an earlier key
                key = rng.choice(list(book))
                src = book[key]
                value = out[src] + out[src + 1]
                at = len(out) + 3  # after "[", key, "?"
                queries += [(at, src), (at + 1, src + 1)]
                out.extend(f"[{key}?{value}]")
            else:  # SET a fresh key
                key = rng.choice(KEY_CHARS)
                if key in book:  # keys are never reassigned
                    continue
                book[key] = len(out) + 3  # after "[", key, "="
                out.extend(f"[{key}={rng.randrange(100):02d}]")
        return "".join(out[:length]), [(q, s) for q, s in queries if q < length]

    def batch(self, split, batch_size, block_size, device="cpu"):
        """x, y as in every GPT lecture, plus two things for measuring retrieval:

        qmask[b, t] : True where y[b, t] is a digit of a QUERY (the hard tokens)
        src[b, t]   : where in x that digit was SET (the needle), else -1
        """
        x = torch.empty(batch_size, block_size, dtype=torch.long)
        y = torch.empty(batch_size, block_size, dtype=torch.long)
        qmask = torch.zeros(batch_size, block_size, dtype=torch.bool)
        src = torch.full((batch_size, block_size), -1, dtype=torch.long)
        for b in range(batch_size):
            s, queries = self.sample(split, block_size + 1)
            ids = torch.tensor(self.encode(s))
            x[b], y[b] = ids[:-1], ids[1:]
            for q, needle in queries:
                qmask[b, q - 1] = True  # y index q-1 is string index q
                src[b, q - 1] = needle
        return x.to(device), y.to(device), qmask.to(device), src.to(device)


if __name__ == "__main__":
    data = Needles()
    s, queries = data.sample("train", 400)
    print(s)
    print("\nvocab", data.vocab_size, "| (query digit pos, needle pos):", queries)
