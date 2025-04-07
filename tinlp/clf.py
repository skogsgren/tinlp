import math
from abc import abstractmethod
from collections import Counter, defaultdict
from itertools import chain
from typing import Callable, Iterable
from functools import partial

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
    # TODO: add shuffle after each epoch, perhaps split fit into fit and epoch
    # function, where fit is rigid and expects certain things, while epoch is
    # flexible
    def fit(self, X: list, y: list):
        raise NotImplementedError

    def eval(self, X: list, y: list) -> float:
        # for token level we flatten our y/yhat arrays
        if self.params.get("eval_level", "sequence") == "token":
            y_hat = list(chain.from_iterable(self.predict(x) for x in X))
            y = list(chain.from_iterable(y))
        else:
            y_hat = [self.predict(x) for x in X]
        return Metric(self.metric)(y=y, y_hat=y_hat)


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
        for label in sorted(set(y)):
            if self.params.get("show_progress", False):
                print(f"creating ngram model for {label}")
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
    def __init__(self, params, feature_fn: Callable):
        super().__init__(params)
        self._get_features = partial(feature_fn, self)

    def _reset(self, y: list):
        # just sorting by alphabetical order here
        self.labels = set()
        for seq_y in y:
            for label in seq_y:
                self.labels.add(label)
        self.labels = sorted(self.labels)
        self.weights = {y: Counter() for y in self.labels}

    def fit(self, X: list, y: list):
        self._reset(y)
        prev_epoch_acc = 0.0
        for epoch_n in range(1, self.params.get("epochs", 5) + 1):
            pbar = tqdm(list(zip(X, y)), disable=not self.params.get("show_progress"))
            if epoch_n > 1:
                pbar.set_description(
                    f"{epoch_n} (PREV_{self.metric.upper()}={prev_epoch_acc:.2f})"
                )
            else:
                pbar.set_description(f"{epoch_n}")
            for seq_x, seq_y in pbar:
                prediction = self._predict_seq(seq_x)
                for i in range(len(seq_x)):
                    features = self._get_features(seq_x, i)
                    for f in features:
                        self.weights[seq_y[i]][f] += 1
                        self.weights[prediction[i]][f] -= 1
            if self.params.get("show_progress", False):
                prev_epoch_acc = self.eval(X, y)

    def _predict_seq(self, X: tuple[str]):
        seq_y = []
        for i in range(len(list(X))):
            sc = []
            features = list(self._get_features(X, i))
            for y in self.labels:
                score = (sum(self.weights[y][f] for f in features), y)
                sc.append(score)
            seq_y.append(max(sc)[1])
        return tuple(seq_y)

    def _get_features(self, X: tuple, i: int) -> Iterable:
        raise NotImplementedError

    def get_top_k_features(self, k: int):
        flattened_features = []
        for label, features in self.weights.items():
            for feature, weight in features.items():
                flattened_features.append((weight, label, feature))
        return sorted(flattened_features, reverse=True)[:k]

    def predict(self, x: str | tuple):
        if isinstance(x, str):
            x = (x,)
        return self._predict_seq(x)


class MultiNomialLRClassifier(PerceptronClassifier):
    def _reset(self, y: list):
        # just sorting by alphabetical order here
        self.labels = set()
        for seq_y in y:
            for label in seq_y:
                self.labels.add(label)
        self.labels = sorted(self.labels)
        self.weights = {k: Counter() for k in self.labels}

    def _avg_grad(self, grads: dict, n: int) -> dict:
        for k, v in grads.items():
            for f, w in v.items():
                grads[k][f] = w / n
        return grads

    def _concat_grad(self, grad_one: dict, grad_two: dict) -> dict:
        concat_grad = grad_one.copy()
        for k, v in grad_two.items():
            for f, w in v.items():
                if k not in concat_grad:
                    concat_grad[k] = defaultdict(float)
                concat_grad[k][f] += w
        return concat_grad

    def _softmax(self, scores):
        max_score = max(scores.values(), default=0)
        exp_scores = {label: math.exp(s - max_score) for label, s in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {label: exp_scores[label] / total for label in scores}

    def _compute_gradients(self, i: int, seq_x: tuple, seq_y: tuple):
        """computes gradient for single element in sequence"""
        y = seq_y[i]
        features = [x for x in self._get_features(seq_x, i)]
        scores = {
            label: self.bias[label] + sum(self.weights[label][f] for f in features)
            for label in self.labels
        }
        probs = self._softmax(scores)
        grads = {label: defaultdict(float) for label in self.labels}
        grads_bias = {label: 0.0 for label in self.labels}

        for label in self.labels:
            error = (1 if label == y else 0) - probs[label]
            for f in features:
                grads[label][f] += error
            grads_bias[label] = error
        loss = -math.log(probs[y] + 1e-100)
        return grads, grads_bias, loss

    def fit(self, X: list, y: list):
        self.lr = self.params.get("lr", 1)
        self._reset(y)
        self.bias = defaultdict(float)  # a mapping from labels to floats

        prev_epoch_acc = 0.0
        for epoch_n in range(1, self.params.get("epochs", 5) + 1):
            total_loss = 0.0
            grads = {}
            grads_bias = {k: 0.0 for k in self.labels}

            pbar = tqdm(
                list(enumerate(list(zip(X, y)))),
                disable=not self.params.get("show_progress"),
            )
            if epoch_n > 1:
                pbar.set_description(
                    f"{epoch_n} (PREV_{self.metric.upper()}={prev_epoch_acc:.2f})"
                )
            else:
                pbar.set_description(f"{epoch_n}")
            for current_seq, (seq_x, seq_y) in pbar:
                # we go through entire sequence before updating weights
                for i in range(len(seq_x)):
                    grad, grad_bias, loss = self._compute_gradients(i, seq_x, seq_y)

                    # TODO: currently dirty and slow
                    if len(seq_x) != 1:
                        grads = self._concat_grad(grads, grad)
                        grads_bias = {
                            k: v + grad_bias[k] for k, v in grads_bias.items()
                        }
                    else:
                        # NOTE: this works fine
                        grads = grad
                        grads_bias = grad_bias

                    total_loss += loss
                # we average the grads in sequences
                if len(seq_x) != 1:
                    grads = self._avg_grad(grads, len(seq_x))
                    grads_bias = {k: v / len(seq_x) for k, v in grads_bias.items()}

                for label in self.labels:
                    for f, v in grads[label].items():
                        self.weights[label][f] += self.lr * v
                    self.bias[label] += self.lr * grads_bias[label]
                pbar.set_postfix(loss=f"{total_loss / (current_seq + 1):.2f}")
            if self.params.get("show_progress", False):
                prev_epoch_acc = self.eval(X, y)

    def _predict_seq(self, X: tuple):
        def _score(X, i, seq_y):
            fw = (self.weights[seq_y[i]][f] for f in self._get_features(X, i))
            return self.bias[seq_y[i]] + sum(fw)

        seq_y = []
        for i, _ in enumerate(X):
            scores = [(_score(X, i, seq_y + [y]), y) for y in self.labels]
            seq_y.append(max(scores, key=lambda x: x[0])[1])
        return tuple(seq_y)

    def predict(self, x: str | tuple):
        if isinstance(x, str):
            x = (x,)
        return self._predict_seq(x)
