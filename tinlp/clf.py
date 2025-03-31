import math
import random
import re
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .eval import Metric
from .lm import NGramModel
from .utils.tokenizer import load_tokenizer


class Classifier:
    def __init__(self, params: dict = {}):
        self.params = params
        self.tokenizer = load_tokenizer(self.params.get("tokenizer", "regex"))
        self.metric = self.params.get("metric", "accuracy")

    @abstractmethod
    def predict(self, x: str):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: list, y: list):
        raise NotImplementedError

    def eval(self, X: list, y: list) -> float:
        return Metric(self.metric)(y=y, y_hat=[self.predict(x) for x in X])


class NaiveBayesClassifier(Classifier):
    def fit(self, X: list, y: list):
        self.ndoc = Counter()
        self.ntoken = defaultdict(Counter)
        for text, label in zip(X, y):
            self.ndoc[label] += 1
            for word in self.tokenizer.tokenize(text):
                self.ntoken[label][word] += 1
        self.token_sum = {
            label: sum(self.ntoken[label].values()) for label in self.ndoc.keys()
        }

    def predict(self, x: str):
        vocab_size = len(set.union(*[set(v.keys()) for v in self.ntoken.values()]))
        total_docs = sum(self.ndoc.values())
        log_sums = {}
        for label in self.ndoc:
            log_sums[label] = math.log(self.ndoc[label] / total_docs)
        for token in self.tokenizer.tokenize(x):
            for label in self.ndoc:
                prob = (self.ntoken[label][token] + 1) / (
                    self.token_sum[label] + vocab_size
                )
                log_sums[label] += math.log(prob)

        return max(log_sums, key=lambda x: log_sums[x])


class NGramClassifier(Classifier):
    def fit(self, X: list, y: list):
        self.models = {}
        for label in set(y):
            filtered_X = [x for i, x in enumerate(X) if y[i] == label]
            self.models[label] = NGramModel(
                n=self.params.get("ngram_size", 1),
                k=self.params.get("ngram_smoothing", 0.01),
                sentences=[tuple(self.tokenizer.tokenize(x)) for x in filtered_X],
            )

    def predict(self, x: str):
        tokenized = tuple(self.tokenizer.tokenize(x))
        scores = {}
        for label, model in self.models.items():
            scores[label] = model.score(tokenized)
        return max(scores.keys(), key=lambda x: scores[x])


class PerceptronMorphClassifier(Classifier):
    def fit(self, X: list, y: list):
        self.k = self.params.get("affix_len", 5)
        # dirty trick since we basically need a set with original order
        self.tags = tuple(dict.fromkeys(y).keys())
        self.weights = defaultdict(int)
        for _ in range(self.params.get("epochs", 5)):
            for word, label in zip(X, y):
                prediction = self.predict(word)
                if prediction == label:
                    continue
                for f in self._get_features(word, label):
                    self.weights[f] += 1
                for f in self._get_features(word, prediction):
                    self.weights[f] -= 1
            print(f"TRAIN_ACC={self.eval(X, y)}, N_FEATURES={len(self.weights)}")

    def _get_features(self, x: str, y: str) -> Iterable[str]:
        for i in range(1, min(self.k + 1, len(x) + 1)):
            pos, tag = y.split(";")[0], ";".join(y.split(";")[1:])
            yield f"{pos} prefix={x[:i]} tag={tag}"
            yield f"{pos} suffix={x[-i:]} tag={tag}"

            yield f"{pos} x={x} tag={tag}"
            yield f"{pos} tag={tag}"
            for part in y.split(";")[1:]:
                yield f"{pos} prefix={x[:i]} part={part}"
                yield f"{pos} suffix={x[-i:]} part={part}"

    def predict(self, x: str):
        sc = []
        for y in self.tags:
            sc.append((sum(self.weights[f] for f in self._get_features(x, y)), y))
        return max(sc, key=lambda x: x[0])[1]


class PerceptronSeqClassifier(Classifier):
    def fit(self, X: list, y: list):
        self.k = self.params.get("affix_len", 5)
        self.weights = defaultdict(int)

    def predict(self, x: str):
        pass
