import json
import random
import re
from abc import abstractmethod
from collections import Counter
from pathlib import Path

from .data import CorpusPlain


class Tokenizer:
    def __init__(self, seed: int | None = None):
        self.seed = seed

    @abstractmethod
    def tokenize(self, inp: str) -> list[str]:
        raise NotImplementedError


class RegExTokenizer(Tokenizer):
    """very rudimentary regex tokenizer"""

    def tokenize(self, inp: str) -> list[str]:
        return [x.lower() for x in re.findall(r"\w+|[^\w\s]+", inp)]


class CharTokenizer(Tokenizer):
    """almost dummy like tokenizer: just says each character is a token"""

    def tokenize(self, inp: str) -> list[str]:
        return [x for x in inp if x != " "]


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        data: None | Path = None,
        model: str | Path | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        if isinstance(model, str):
            model = Path(model)
        self.model = model
        if (not self.model) and (not data):
            raise ValueError("FATAL: must provide model if not providing data.")
        self.data = data
        if self.data:
            with open(self.data) as f:
                self.n_lines = sum([1 for _ in f])
        else:
            self.n_lines = 0

        if self.model:
            with open(self.model) as f:
                self.vocab: dict = {x: None for x in json.load(f)}
        else:
            self.vocab: dict = {}

    def _merge(self, merge_tok: str, text: list[str]):
        if len(merge_tok) == 1:
            return text
        merged = []
        i = 0
        while i < (len(text) - len(merge_tok)):
            if "".join(text[i : i + len(merge_tok)]) == merge_tok:
                merged.append(merge_tok)
                i += len(merge_tok)
            else:
                merged.append(text[i])
                i += 1
        if i < len(text):
            for i in range(i, len(text)):
                merged.append(text[i])
        return merged

    def tokenize(self, inp: str) -> list[str]:
        text = [
            c if c in self.vocab else "<unk>"
            for w in [x + "_" for x in inp.split()]
            for c in w
        ]
        for subword in sorted(self.vocab.keys(), key=len):
            text = self._merge(subword, text)
        return text

    def train(
        self, out: Path = Path("./bpe.json"), vocab_size: int = 500, k: int = 10000
    ):
        """trains bpe and exports to out. vocab_size controls the size of the
        vocabulary, and k sets the max amount of lines used for training
        (otherwise sampling is used)"""
        if not self.data:
            raise ValueError("FATAL: cannot train BPE without provided data")

        # we do random sampling bc we want to keep data in memory for
        # performance, but not entire file if file is large
        if k < self.n_lines:
            if self.seed:
                random.seed(self.seed)
            sample_idx = sorted(list(set(random.sample(range(self.n_lines), k=k))))
            data = [
                [c for c in w] + ["_"]
                for i, x in enumerate(CorpusPlain(self.data))
                if i in sample_idx
                for w in x.split()
            ]
        else:
            data = [
                [c for c in w] + ["_"]
                for x in CorpusPlain(self.data)
                for w in x.split()
            ]
        for line in data:
            for c in line:
                self.vocab[c] = None

        # TODO: rewrite to switch to self._merge instead here
        def merge_text(merge_tok):
            for idx, line in enumerate(data):
                merged = []
                i = 0
                while i < (len(line) - 1):
                    if (line[i] == merge_tok[0]) and (line[i + 1] == merge_tok[1]):
                        merged.append(line[i] + line[i + 1])
                        i += 2
                    else:
                        merged.append(line[i])
                        i += 1
                if i < len(line):
                    merged.append(line[i])
                data[idx] = merged

        while len(self.vocab) < vocab_size:
            counts = Counter()
            for text in data:
                for i in range(0, len(text) - 1):
                    counts[(text[i], text[i + 1])] += 1

            # to catch what happens if all whitespace separated words are in vocab
            try:
                merge_tok = counts.most_common(1)[0]
            except IndexError:
                break

            self.vocab[merge_tok[0][0] + merge_tok[0][1]] = None
            merge_text(merge_tok[0])
        with open(out, "w") as f:
            json.dump(list(self.vocab), f)
        self.model = out


def load_tokenizer(name: str | tuple) -> Tokenizer:
    if isinstance(name, tuple):
        # NOTE: not sure what I did here
        # NOTE: possibly better just to use declarations instead, i.e.
        #       instances to pass around
        if name[0].lower() == "bpe":
            return BPETokenizer(name[0], name[1])
    return RegExTokenizer()
