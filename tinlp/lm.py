import gzip
from collections import Counter
from math import log
from pathlib import Path
from typing import Iterable
import numpy as np

from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn import state
from tinygrad.nn.optim import Adam
from tqdm import tqdm


class NGramNN:
    def __init__(self, params):
        self.params = params
        self.vocab_size = params["vocab_size"]
        self.emb_size = params.get("embedding_size", 64)
        self.hid_size = params.get("hidden_size", 128)
        self.batch_size = params.get("batch_size", 16)
        self.n = params.get("ngram_size", 2)
        self.grad_clip = params.get("gradient_clipping_value")

        self.seed = params.get("seed")
        if self.seed:
            Tensor.manual_seed(self.seed)

        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.fc = nn.Linear(self.emb_size * self.n, self.hid_size)
        self.out = nn.Linear(self.hid_size, self.vocab_size)
        self.optimizer = Adam(
            state.get_parameters(self), lr=params.get("learning_rate", 0.1)
        )

    def __call__(self, x: Tensor) -> Tensor:
        fc_out = self.fc(self.emb(x).flatten(1)).tanh()
        assert isinstance(fc_out, Tensor)
        return self.out(fc_out)

    def fit(self, X: Tensor, y: Tensor):
        if self.seed:
            Tensor.manual_seed(self.seed)

        def step():
            Tensor.training = True
            samples = Tensor.randint(self.batch_size, high=int(X.shape[0]))
            xb, yb = X[samples], y[samples]
            self.optimizer.zero_grad()
            loss = self(xb).cross_entropy(yb).backward()
            # in gradient clipping we iterate over each parameter and clamp it
            if self.grad_clip:
                for p in state.get_parameters(self):
                    if p.grad is not None:
                        p.grad = p.grad.clamp(-self.grad_clip, self.grad_clip)
            self.optimizer.step()
            return loss

        # we wrap the step function for speed
        step = TinyJit(step)

        for _ in (
            pb := tqdm(
                range(self.params.get("steps", 500)),
                disable=not self.params.get("show_progress"),
            )
        ):
            loss = step()
            pb.set_postfix(loss=f"{loss.item():.2f}")

    def export_vectors(self, vocab: dict, out: Path) -> None:
        inverse_vocab = {v: k for k, v in vocab.items()}
        print("creating tensor")
        vocab_tensor = Tensor([i for i in range(len(vocab))])
        print("embedding tensor")
        embedding_tensor = self.emb(vocab_tensor)
        print("writing to file")
        with gzip.open(out, "wt") as f:
            for i in tqdm(range(len(vocab))):
                word = inverse_vocab[i]
                glove = embedding_tensor[i].tolist()
                assert isinstance(glove, list)
                f.write(f"{word} {' '.join([str(x) for x in glove])}\n")

    def eval(self, X: Tensor, y: Tensor):
        Tensor.training = False
        return (self(X).argmax(axis=-1) == y).mean().item()


class NGramModel:
    def __init__(self, n: int, k: float, sentences: Iterable[tuple]):
        self.n: int = n
        self.k: float = k
        self.ngrams: Counter = Counter()
        self.vocab: Counter = Counter()
        self.pad: tuple[str] = ("<s>",)
        self.end: tuple[str] = ("<e>",)

        for sentence in sentences:
            self.vocab[self.end] += 1  # we want self.end to be sentence length
            for ngram in self.get_ngrams(sentence, n=self.n):
                self.ngrams[ngram] += 1
            for word in sentence:
                self.vocab[word] += 1
            if self.n == 1:  # we don't need context for unigram
                continue
            for ngram in self.get_ngrams(sentence, n=self.n - 1):
                self.ngrams[ngram] += 1
        self.token_sum = sum(self.vocab.values())

    def get_ngrams(self, sentence: tuple, n: int) -> Iterable[tuple]:
        """given a tuple of strings returns sliding window ngram iterable"""
        s = self.pad * n + sentence + self.end
        for i in range(len(s) - n + 1):
            yield (s[i : i + n])

    def p(self, word: str, context: tuple[str]) -> float:
        """returns the probability of a word given its previous context"""
        numerator: float = self.k + self.ngrams[context + (word,)]
        if self.n == 1:
            denominator: float = self.k * len(self.vocab) + self.token_sum
        else:
            denominator: float = self.k * len(self.vocab) + self.ngrams[context]
        return numerator / denominator

    def score(self, sentence: tuple) -> float:
        """given a sentence as a tuple, return the log prob of that sentence
        given model"""
        context = list(self.get_ngrams(sentence, n=self.n - 1))
        prob = [
            log(self.p(word, context[i])) for i, word in enumerate(sentence + self.end)
        ]
        return sum(prob)
