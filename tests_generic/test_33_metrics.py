from tinlp.eval import Metric

Y = [0, 1, 2, 2, 0, 1, 2, 0, 1, 2]
Y_HAT = [0, 0, 2, 2, 0, 2, 1, 0, 1, 2]

ACC = 0.70
F1 = 0.67


def test_accuracy_metric():
    assert round(Metric("accuracy")(Y, Y_HAT), 2) == ACC


def test_f1_score_metric():
    assert round(Metric("f1")(Y, Y_HAT), 2) == F1
