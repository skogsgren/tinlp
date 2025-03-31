import math
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Callable, Iterable

from tqdm import tqdm

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


class PerceptronClassifier(Classifier):
    def __init__(self, params):
        super().__init__(params)
        if isinstance(self.params.get("feature_fn", None), Callable):
            self.get_features = self.params["feature_fn"]
        else:
            self._get_features = self._params_to_feature_function()

    def fit(self, X: list, y: list):
        if not self._get_features:
            raise NameError("Cannot fit model without defined feature function.")
        self.weights = Counter()

        # just sorting by alphabetical order here
        self.labels = set()
        for seq_y in y:
            for label in seq_y:
                self.labels.add(label)
        self.labels = sorted(self.labels)

        for _ in range(self.params.get("epochs", 5)):
            pbar = tqdm(
                zip(X, y),
                total=len(X),
                disable=not self.params.get("show_progress", False),
            )
            for seq_x, seq_y in pbar:
                prediction = self._predict_seq(seq_x)
                for i, label in enumerate(seq_y):
                    if seq_y == prediction:
                        continue
                    for f in self._get_features(i, seq_x, seq_y):
                        self.weights[f] += 1
                    for f in self._get_features(i, seq_x, prediction):
                        self.weights[f] -= 1
            if self.params.get("show_progress", False):
                # print(f"{self.metric}={self.eval([x[0] for x in X], y)}")
                print(f"{self.metric}={self.eval(X, y)}")

    def _feat_seq_conll(self, i: int, seq_x: str, seq_y: str) -> Iterable[str]:
        # context features
        if i != 0:
            yield f"{seq_y[i]} prev={seq_x[i - 1]}"
        yield f"{seq_y[i]} curr={seq_x[i]}"
        if (i + 1) != len(seq_y):
            yield f"{seq_y[i]} next={seq_x[i + 1]}"

        def has_capital(s):
            return any(c.isupper() for c in s)

        def has_number(s):
            return any(c.isdigit() for c in s)

        # word features
        yield f"{seq_y[i]} has_capital={has_capital(seq_x[i][0][0])}"
        # yield f"{seq_y[i]} has_number={has_number(seq_x[i][0][0])}"

    def _feat_morph_tag(self, i: int, seq_x: tuple, seq_y: tuple) -> Iterable[str]:
        """generates morphological features on affixes. assumes 2D data in 3D
        format (e.g. ['word'] not just 'word')"""
        if len(seq_x) != 1 or len(seq_y) != 1:
            raise ValueError("Data length != 1 (Make sure data is 3D)")
        for i in range(1, min(self.k + 1, len(seq_x[0]) + 1)):
            pos, tag = seq_y[0].split(";")[0], ";".join(seq_y[0].split(";")[1:])
            yield f"{pos} prefix={seq_x[0][:i]} tag={tag}"
            yield f"{pos} suffix={seq_x[0][-i:]} tag={tag}"

    def _params_to_feature_function(self) -> Callable | None:
        choice = self.params.get("feature_fn", None)
        if choice == "seq_conll":
            self.k = self.params.get("affix_len", 5)
            return self._feat_seq_conll
        if choice == "morph_tag":
            self.k = self.params.get("affix_len", 5)
            return self._feat_morph_tag

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
        if isinstance(x, tuple):
            return self._predict_seq(x)
        if self.params.get("feature_fn") == "morph_tag":
            return self._predict_seq((x))
        return self._predict_seq(self.tokenizer.tokenize(x))
