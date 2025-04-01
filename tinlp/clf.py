import math
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Callable, Iterable

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
    # TODO: add shuffle after each epoch, perhaps split fit into fit and epoch
    # function, where fit is rigid and expects certain things, while epoch is
    # flexible
    def fit(self, X: list, y: list):
        raise NotImplementedError

    # TODO: add token evaluation for sequence models
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
        scores = {}
        for label, model in self.models.items():
            scores[label] = model.score(tuple(self.tokenizer.tokenize(x)))
        return max(scores.keys(), key=lambda x: scores[x])


class PerceptronClassifier(Classifier):
    def __init__(self, params, feature_fn: Callable | None = None):
        super().__init__(params)
        self._get_features = feature_fn

    def _reset(self, y: list):
        # just sorting by alphabetical order here
        self.labels = set()
        for seq_y in y:
            for label in seq_y:
                self.labels.add(label)
        self.labels = sorted(self.labels)
        self.weights = Counter()

    def fit(self, X: list, y: list):
        if not self._get_features:
            raise NameError("Cannot fit model without defined feature function.")
        self._reset(y)
        for _ in range(self.params.get("epochs", 5)):
            for seq_x, seq_y in zip(X, y):
                prediction = self._predict_seq(seq_x)
                if seq_y == prediction:
                    continue
                for i, _ in enumerate(seq_y):
                    for f in self._get_features(i, seq_x, seq_y):
                        self.weights[f] += 1
                    for f in self._get_features(i, seq_x, prediction):
                        self.weights[f] -= 1
            if self.params.get("show_progress", False):
                print(f"{self.metric}={self.eval(X, y)}")

    def _predict_seq(self, X: Iterable[str]):
        if not self._get_features:
            raise NameError("Cannot predict without defined feature function.")

        # greedy: assumes previous prediction is correct
        seq_y = []
        for i in range(len(list(X))):
            sc = []
            for y in self.labels:
                score = (
                    sum(self.weights[f] for f in self._get_features(i, X, seq_y + [y])),
                    y,
                )
                sc.append(score)
            seq_y.append(max(sc, key=lambda x: x[0])[1])
        return tuple(seq_y)

    def predict(self, x: str | tuple):
        # FIX: make general instead
        if isinstance(x, tuple):
            return self._predict_seq(x)
        if self.params.get("feature_fn") == "morph_tag":
            return self._predict_seq((x,))
        return self._predict_seq(self.tokenizer.tokenize(x))
