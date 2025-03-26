from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
import math

from .utils.data import CategorizedCorpus
from .utils.tokenizer import load_tokenizer


class Classifier:
    def __init__(self, data_fn: Path, params: dict = {}):
        self.data = CategorizedCorpus(data_fn)
        self.params = params
        self.tokenizer = load_tokenizer(self.params.get("tokenizer", "regex"))

    @abstractmethod
    def predict(self, x: str):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self, data_fn: Path):
        raise NotImplementedError


class NaiveBayesClassifier(Classifier):
    def fit(self):
        self.ndoc = Counter()
        self.ntoken = defaultdict(Counter)
        for text, label in self.data:
            self.ndoc[label] += 1
            for word in self.tokenizer.tokenize(text):
                self.ntoken[label][word] += 1
        self.token_sum = {
            label: sum(self.ntoken[label].values()) for label in self.ndoc.keys()
        }

    def predict(self, x: str):
        log_sums = {label: -math.inf for label in self.ndoc.keys()}
        for token in self.tokenizer.tokenize(x):
            for label in self.ndoc.keys():
                calc = math.log(
                    self.ntoken[label].get(token, 1)
                    / (self.token_sum[label] + len(list(self.ntoken[label].keys())))
                )
                if log_sums[label] == -math.inf:
                    log_sums[label] = calc
                else:
                    log_sums[label] += calc
        return max(log_sums, key=lambda x: log_sums[x])

    def eval(self, data_fn: Path) -> float:
        correct: int = 0
        total: int = 0
        for text, label in CategorizedCorpus(data_fn):
            total += 1
            if self.predict(text) == label:
                correct += 1
        return correct / total
