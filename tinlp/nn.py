import gzip
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from .utils.data import Corpus
from .utils.text import get_ngrams


def get_ngram_tensors(
    corpus: Corpus,
    tokenizer,
    n: int,
    min_occ: int = 0,
    vocab: dict | None = None,
    device: str = "cpu",
):
    """given a Corpus, returns n-gram tensors for language model training"""
    # NOTE: I don't like having this here, but I have it here to avoid having
    # to have torch as a hard requirement for the package

    data = [tokenizer.tokenize(seq) for seq in corpus.data]

    if vocab:
        pad = vocab["<pad>"]
        unk = vocab["<unk>"]
        eos = vocab["<eos>"]
    else:
        # we first define our special tokens
        pad = 0
        unk = 1
        eos = 2
        # count occurences for each word in tokenized sequences
        word_count = Counter()
        for seq in data:
            for word in seq:
                word_count[word] += 1
        # initialize vocab along with special tokens
        vocab = {"<pad>": pad, "<unk>": unk, "<eos>": eos}
        # create vocab mapping if occurence > min_occ
        for word, c in word_count.items():
            if c < min_occ:
                continue
            vocab[word] = len(vocab)
    # create mapping vectors for each sequence
    # since <unk> == 1
    data = [[vocab.get(word, unk) for word in seq] for seq in data]
    # convert mapping vectors to ngram tuples (e.g. if n=3 then ([1, 2, 3], 4))
    X_list = []
    y_list = []
    for seq in data:
        for ngram in get_ngrams(seq, pad=pad, eos=eos, n=n + 1):
            if ngram[-1] == unk:
                continue
            X_list.append(ngram[:-1])
            y_list.append(ngram[-1])
    return (
        torch.tensor(X_list, dtype=torch.long, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
        vocab,
    )


class NGram(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        ngram_size=2,
        emb_size=64,
        hid_size=64,
        batch_size=16,
        lr=0.001,
        grad_clip=None,
        steps=500,
        seed=None,
        pbar=True,
        device="cpu",
    ):
        super().__init__()
        torch.manual_seed(seed) if seed else None
        self.bs, self.lr, self.clip, self.steps = batch_size, lr, grad_clip, steps
        self.device, self.pbar = torch.device(device), pbar
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.fc = nn.Linear(emb_size * ngram_size, hid_size)
        self.out = nn.Linear(hid_size, vocab_size)
        self.optim = AdamW(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.to(self.device)

    def __call__(self, x: torch.Tensor):
        return self.out(self.fc(self.emb(x).flatten(1)).tanh())

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        for _ in (pb := tqdm(range(self.steps))):
            self.optim.zero_grad()
            idx = torch.randint(0, X.size(0), (self.bs,))
            loss = self.criterion(self(X[idx]), y[idx])
            loss.backward()
            if self.clip:
                clip_grad_norm_(self.parameters(), self.clip)
            self.optim.step()
            pb.set_postfix(loss=f"{loss:.2f}")

    def export_vectors(self, vocab: dict, out: Path):
        inverse_vocab = {v: k for k, v in vocab.items()}
        embeddings = self.emb.weight.data
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        with gzip.open(out, "wt") as f:
            for i in tqdm(range(len(vocab))):
                glove = embeddings[i].tolist()
                f.write(f"{inverse_vocab[i]} {' '.join([str(x) for x in glove])}\n")

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            logits = self(X)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).sum().item()
            total = y.size(0)
        return correct / total
