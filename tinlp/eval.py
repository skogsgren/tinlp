class Metric:
    def __init__(self, metric: str = "accuracy"):
        if metric == "f1" or metric == "f1_score":
            self.process = self._f1_score
        else:
            self.process = self._accuracy

    def _accuracy(self, y, y_hat):
        return sum(y == y_hat for y, y_hat in zip(y, y_hat)) / len(y)

    def _f1_score(self, y, y_hat):
        def f1_class(y, y_hat, class_label):
            tp = sum(1 for t, p in zip(y, y_hat) if t == p == class_label)
            fp = sum(
                1 for t, p in zip(y, y_hat) if t != class_label and p == class_label
            )
            fn = sum(
                1 for t, p in zip(y, y_hat) if t == class_label and p != class_label
            )

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0

            return (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) != 0
                else 0
            )

        return sum(f1_class(y, y_hat, label) for label in set(y)) / len(set(y))

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
