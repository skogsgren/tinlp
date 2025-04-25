import gzip
import random
from abc import abstractmethod
from collections import Counter
from csv import DictReader
from pathlib import Path
from typing import Iterable

from tinygrad import Tensor

from .text import get_ngrams


class Corpus:
    """abstract class for Corpus subclasses"""

    def __init__(self, data_path: str | Path, **params):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data = self._process(data_path, **params)

    @abstractmethod
    def _process(self, data: Path, **params):
        raise NotImplementedError

    def get_arrays(self, unsqueeze: bool = False) -> tuple[list, list]:
        """returns two lists, X with independent, and y with dependent variables"""
        data = [x for x in self.data]
        X = [x[0] for x in data]
        y = [x[1] for x in data]
        if unsqueeze:
            X = [(x,) for x in X]
            y = [(y,) for y in y]
        return X, y

    def get_ngram_tensors(
        self,
        tokenizer,
        n: int,
        min_occ: int = 0,
    ) -> tuple[Tensor, Tensor, dict]:
        # we first define our special tokens
        pad = 0
        unk = 1
        eos = 2
        # tokenize each sequence in inp according to self.tokenization
        print("tokenizing sequences")
        data = [tokenizer.tokenize(seq) for seq in self.data]
        # count occurences for each word in tokenized sequences
        print("counting occurences")
        word_count = Counter()
        for seq in data:
            for word in seq:
                word_count[word] += 1
        # initialize vocab along with special tokens
        self.vocab = {"<pad>": pad, "<unk>": unk, "<eos>": eos}
        # create vocab mapping if occurence > min_occ
        print("creating vocab mappings")
        for word, c in word_count.items():
            if c < min_occ:
                continue
            self.vocab[word] = len(self.vocab)
        # create mapping vectors for each sequence
        print("creating mapping vectors")
        # since <unk> == 1
        data = [[self.vocab.get(word, unk) for word in seq] for seq in data]
        # convert mapping vectors to ngram tuples (e.g. if n=3 then ([1, 2, 3], 4))
        X_list = []
        y_list = []
        for seq in data:
            for ngram in get_ngrams(seq, pad=pad, eos=eos, n=n + 1):
                if ngram[-1] == unk:
                    continue
                X_list.append(ngram[:-1])
                y_list.append(ngram[-1])
        # convert mapped ngram vectors to Tensors
        print("converting mapping vectors to tensors")
        return Tensor(X_list), Tensor(y_list), self.vocab

    def train_test_split(
        self, test_size: float = 0.2, seed=None, unsqueeze: bool = False
    ) -> tuple[list, list, list, list]:
        """returns test/train split: (X_train, X_test, y_train, y_test)"""
        if seed:
            random.seed(seed)
        data = list(zip(*self.get_arrays(unsqueeze=unsqueeze)))
        n_test = int(len(data) * test_size)
        random.shuffle(data)
        X_shuffled, y_shuffled = zip(*data)
        return (
            list(X_shuffled[:-n_test]),
            list(X_shuffled[-n_test:]),
            list(y_shuffled[:-n_test]),
            list(y_shuffled[-n_test:]),
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)


class CorpusCSV(Corpus):
    def _process(self, data: Path, **params):
        """returns iterator for tsv file with labels"""
        delim = params.get("delimiter", ",")
        header = params.get("header", False)
        with open(data, newline="", encoding="utf-8") as f:
            fieldnames = params.get("fieldnames", ["text", "label"])
            reader = DictReader(f, delimiter=delim, fieldnames=fieldnames)
            if header:
                next(reader)
            for line in reader:
                yield self._process_line(line, params)

    def _process_line(self, line: dict, params: dict) -> tuple:
        if params.get("label_to_int", False):
            return (str(line["text"]), int(line["label"]))
        else:
            return (str(line["text"]), line["label"])


class CorpusUNIMORPH(CorpusCSV):
    def _process_line(self, line: dict, params: dict) -> tuple:
        split_y = line["label"].split(";")
        pos, tag = split_y[0], ";".join(split_y[1:])
        return ((line["text"], pos), tag)


class CorpusSubDir(Corpus):
    def _process(self, data: Path, **params):
        SUBDIR_LABELS = ["neg", "pos"]
        folders = sorted(
            [x for x in data.iterdir() if x.is_dir() and x.name in SUBDIR_LABELS],
            key=lambda x: x.name,  # to ensure the order is neg, pos
        )
        for label in SUBDIR_LABELS:
            if label in [x.name for x in folders]:
                continue
            raise ValueError(f"{label} not in subdirectories for {data}")
        for i, folder in enumerate(folders):
            for file in folder.iterdir():
                with open(file) as f:
                    yield (f.read().strip(), i)


class CorpusCONLL2003(Corpus):
    def _process(self, data: Path, **params):
        with open(data) as f:
            seq = ([], [])
            for i, line in enumerate(x.strip().split() for x in f):
                if i == 0:
                    continue
                if not line:
                    if i == 1:
                        continue
                    yield tuple(seq[0]), tuple(seq[1])
                    seq = ([], [])
                    continue
                seq[0].append((line[0], line[1], line[2]))
                seq[1].append(line[3])


class CorpusPlain(Corpus):
    def _process(self, data: Path, **params) -> Iterable[str]:
        """returns iterator for file without labels"""
        if data.suffixes[-1] == ".gz":
            with gzip.open(data, "rt") as f:
                for line in f:
                    yield line.strip()
        else:
            with open(data) as f:
                for line in f:
                    yield line.strip()
