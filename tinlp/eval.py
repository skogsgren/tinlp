import random
from collections import Counter
from pprint import pprint

import numpy as np
from tqdm import tqdm


class Metric:
    def __init__(self, metric: str = "accuracy"):
        if metric == "f1":
            self.process = self._f1_score
        elif metric == "verbose":
            self.process = self._verbose
        else:
            self.process = self._accuracy

    def _accuracy(self, y, y_hat):
        label_to_int = {label: i for i, label in enumerate(set(y) | set(y_hat))}
        y_int = np.array([label_to_int[label] for label in y])
        y_hat_int = np.array([label_to_int[label] for label in y_hat])
        return float(np.mean(y_int == y_hat_int))

    def _verbose(self, y, y_hat):
        """returns accuracy, prints class specific accuracy"""
        cl_total = Counter()
        cl_correct = Counter()
        for i, correct_label in enumerate(y):
            cl_total[correct_label] += 1
            if y_hat[i] == correct_label:
                cl_correct[correct_label] += 1
        cl_acc = {}
        for label, total in cl_total.items():
            cl_acc[label] = (round(cl_correct[label] / total, 3), total)
        pprint(sorted(cl_acc.items(), key=lambda x: x[1][0]))
        return self._accuracy(y, y_hat)

    def _f1_score(self, y, y_hat):
        labels = list(set(y) | set(y_hat))
        label_to_int = {label: i for i, label in enumerate(labels)}

        y_int = np.array([label_to_int[label] for label in y])
        y_hat_int = np.array([label_to_int[label] for label in y_hat])

        f1s = []
        for i in range(len(labels)):
            tp = np.sum((y_int == i) & (y_hat_int == i))
            fp = np.sum((y_int != i) & (y_hat_int == i))
            fn = np.sum((y_int == i) & (y_hat_int != i))

            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0
            )
            f1s.append(f1)

        return float(np.mean(f1s) if f1s else 0)

    def __call__(self, y, y_hat) -> float:
        return self.process(y, y_hat)


def cross_val_score(clf, X: list, y: list, k: int = 10):
    metrics = []
    fold_size = len(X) // k
    indices = list(range(len(X)))
    folds_idx = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds_idx.append((train_indices, test_indices))
    for c, (train_idx, test_idx) in enumerate(folds_idx):
        clf.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        metrics.append(clf.eval([X[i] for i in test_idx], [y[i] for i in test_idx]))
        print("fold", c + 1, metrics[-1])
    return metrics


def paired_bootstrap(
    clf_one_pred: list,
    clf_two_pred: list,
    statistic: str,
    test_y: list,
    n: int = 10000,
    seed: int | None = None,
):
    """assumes deterministic classification, since it takes cached predictions as input. one-tailed"""
    if seed:
        random.seed(seed)
    metric = Metric(statistic)
    scores = []
    data = list(zip(test_y, clf_one_pred, clf_two_pred, test_y))

    clf_one_score = metric(test_y, clf_one_pred)
    clf_two_score = metric(test_y, clf_two_pred)
    delta = float(clf_one_score - clf_two_score)

    print(f"delta={delta} (one={clf_one_score}, two={clf_two_score})")
    for _ in tqdm(range(n), total=n):
        indices = random.choices(range(len(data)), k=len(data))
        sample_y = [data[i][0] for i in indices]
        sample_y_one_pred = [data[i][1] for i in indices]
        sample_y_two_pred = [data[i][2] for i in indices]
        clf_one_score = metric(sample_y, sample_y_one_pred)
        clf_two_score = metric(sample_y, sample_y_two_pred)
        effect_size = clf_one_score - clf_two_score
        scores.append(effect_size)

    count = sum(1 for score in scores if score >= 2 * delta)
    p_value = float((count + 1) / (n + 1))  # some smoothing to avoid p_value=0
    scores = sorted(scores)
    ci = (float(scores[int(0.025 * n)]), float(scores[int(0.975 * n)]))
    print(f"p_value={p_value:.2f}")
    print(f"ci={ci}")
    return p_value, ci
