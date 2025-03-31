from tinlp.eval import Metric


Y = [1, 1, 1, 0, 0]
Y_HAT = [1, 0, 1, 0, 1]

ACC = 0.6000
F1 = 0.5833


def test_accuracy_metric():
    assert round(Metric("accuracy")(Y, Y_HAT), 4) == ACC


def test_f1_score_metric():
    assert round(Metric("f1")(Y, Y_HAT), 4) == F1
